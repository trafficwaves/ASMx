import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def fft_four_convs(Dp, Mp, k_cong, k_free, use_ortho=True):
    """
    Compute via FFT:
        sum_cong = conv2d(Dp, k_cong)
        sum_free = conv2d(Dp, k_free)
        N_cong   = conv2d(Mp, k_cong)
        N_free   = conv2d(Mp, k_free)
    Inputs:
      Dp, Mp:   (B, C, H, W)
      k_cong,:  (F, C, Kh, Kw)
      k_free:   (F, C, Kh, Kw)
    Returns:
      sum_cong, N_cong, sum_free, N_free each of shape (B, F, H-Kh+1, W-Kw+1)
    """
    # ——— sanitize inputs ———
    Dp = torch.nan_to_num(Dp, nan=0.0, posinf=0.0, neginf=0.0)
    Mp = torch.nan_to_num(Mp, nan=0.0, posinf=0.0, neginf=0.0)

    B, C, H, W        = Dp.shape
    F, _, Kh, Kw      = k_cong.shape
    Fh, Fw            = H + Kh - 1, W + Kw - 1
    device, dtype     = Dp.device, Dp.dtype

    # ——— pad inputs ———
    Dp_pad = torch.zeros(B, C, Fh, Fw, device=device, dtype=dtype)
    Mp_pad = torch.zeros(B, C, Fh, Fw, device=device, dtype=dtype)
    Dp_pad[..., :H, :W] = Dp
    Mp_pad[..., :H, :W] = Mp

    # ——— pad kernels ———
    k1_pad = torch.zeros(F, C, Fh, Fw, device=device, dtype=dtype)
    k2_pad = torch.zeros(F, C, Fh, Fw, device=device, dtype=dtype)
    k1_pad[..., :Kh, :Kw] = k_cong
    k2_pad[..., :Kh, :Kw] = k_free

    # choose normalization
    norm = "ortho" if use_ortho else None

    # ——— FFT both inputs and kernels ———
    Df  = torch.fft.rfftn(Dp_pad, dim=(-2, -1), s=(Fh, Fw), norm=norm)
    Mf  = torch.fft.rfftn(Mp_pad, dim=(-2, -1), s=(Fh, Fw), norm=norm)
    Kf1 = torch.fft.rfftn(k1_pad, dim=(-2, -1), s=(Fh, Fw), norm=norm)
    Kf2 = torch.fft.rfftn(k2_pad, dim=(-2, -1), s=(Fh, Fw), norm=norm)

    # ——— pointwise multiply in freq domain ———
    Y1 = Df * Kf1    # for sum_cong
    Y2 = Df * Kf2    # for sum_free
    Z1 = Mf * Kf1    # for N_cong
    Z2 = Mf * Kf2    # for N_free

    # ——— inverse FFT back to real ———
    y1 = torch.fft.irfftn(Y1, dim=(-2, -1), s=(Fh, Fw), norm=norm)
    y2 = torch.fft.irfftn(Y2, dim=(-2, -1), s=(Fh, Fw), norm=norm)
    z1 = torch.fft.irfftn(Z1, dim=(-2, -1), s=(Fh, Fw), norm=norm)
    z2 = torch.fft.irfftn(Z2, dim=(-2, -1), s=(Fh, Fw), norm=norm)

    # ——— crop “valid” region ———
    oh, ow = H - Kh + 1, W - Kw + 1
    sum_cong = y1[..., Kh-1:Kh-1+oh, Kw-1:Kw-1+ow]
    sum_free = y2[..., Kh-1:Kh-1+oh, Kw-1:Kw-1+ow]
    N_cong   = z1[..., Kh-1:Kh-1+oh, Kw-1:Kw-1+ow]
    N_free   = z2[..., Kh-1:Kh-1+oh, Kw-1:Kw-1+ow]

    return sum_cong, N_cong, sum_free, N_free

class AdaptiveSmoothing(nn.Module):
    def __init__(self,
                 kernel_time_window: float,
                 kernel_space_window: float,
                 dx: float,
                 dt: float,
                 init_delta: float = 0.10, # mile
                 init_tau: float = 9.0, # seconds
                 init_c_cong: float = 12.0,
                 init_c_free: float = -45.0,
                 init_v_thr: float = 40.0,
                 init_v_delta: float = 10.0):
        super().__init__()
        self.size_t = int(kernel_time_window / dt)
        self.size_x = int(kernel_space_window / dx)
        self.dt = dt
        self.dx = dx

        t_offs = torch.arange(-self.size_t, self.size_t + 1) * dt
        # print(t_offs)
        x_offs = torch.arange(-self.size_x, self.size_x + 1) * dx
        # print(x_offs)
        X, T = torch.meshgrid(x_offs, t_offs, indexing='ij')
        self.register_buffer('T_offsets', T.float())
        self.register_buffer('X_offsets', X.float())

        self.delta   = nn.Parameter(torch.tensor(init_delta))
        self.tau     = nn.Parameter(torch.tensor(init_tau))
        self.c_cong  = nn.Parameter(torch.tensor(init_c_cong))
        self.c_free  = nn.Parameter(torch.tensor(init_c_free))
        self.v_thr   = nn.Parameter(torch.tensor(init_v_thr))
        self.v_delta = nn.Parameter(torch.tensor(init_v_delta))

    def forward(self, raw_data: torch.Tensor):
        # Ensure input is 4D: (B, C, H, W) where H=space, W=time
        if raw_data.ndim == 2:
            raw_data = raw_data.unsqueeze(0).unsqueeze(0)
        elif raw_data.ndim == 3:
            raw_data = raw_data.unsqueeze(1)

        mask = (~raw_data.isnan()).float()
        data = torch.nan_to_num(raw_data, nan=0.0)

        c_cong_s = self.c_cong / 3600.0 # convert from mph to miles per second
        c_free_s = self.c_free / 3600.0
        # print('T_offsize:', self.T_offsets.size())
        t_cong = self.T_offsets - self.X_offsets / c_cong_s
        t_free = self.T_offsets - self.X_offsets / c_free_s

        k_cong = torch.exp(-(t_cong.abs() / self.tau + self.X_offsets.abs() / self.delta))
        # size of k_cong
        # print('k_cong size:', k_cong.size())
        k_free = torch.exp(-(t_free.abs() / self.tau + self.X_offsets.abs() / self.delta))

        k_cong = k_cong.unsqueeze(0).unsqueeze(0)  # (1,1,Kt,Kx)
        k_free = k_free.unsqueeze(0).unsqueeze(0)

        pad = (self.size_t, self.size_t, self.size_x, self.size_x) # to deal with the edge effects
        Dp = F.pad(data, pad, value=0.0)
        # print data size
        # print('Data size:', data.size())
        # print('Dp size:', Dp.size())
        Mp = F.pad(mask, pad, value=0.0)

        # sum_cong = F.conv2d(Dp, k_cong)
        # N_cong   = F.conv2d(Mp, k_cong)
        # sum_free = F.conv2d(Dp, k_free)
        # N_free   = F.conv2d(Mp, k_free)
        # use FFT to compute the convolutions
        sum_cong, N_cong, sum_free, N_free = fft_four_convs(Dp, Mp, k_cong, k_free)

        # Define minimum weight threshold as fraction of the center kernel weight
        # The center of the kernel (where the point being estimated is located) has weight 1.0
        # We require at least some fraction of nearby data to have meaningful contribution
        # Using the center weight as reference makes this adaptive to different tau/delta values
        center_weight = 1.0  # exp(0) = 1 at the center of the kernel
        min_weight_fraction = 0.001  # require at least 0.1% of center weight coverage
        min_weight_threshold = min_weight_fraction * center_weight
        
        # Create validity masks based on whether enough data contributes to the estimate
        valid_cong = (N_cong > min_weight_threshold).float()
        valid_free = (N_free > min_weight_threshold).float()
        
        # Add small epsilon only where we have valid data to avoid division by zero
        eps = 1e-12
        N_cong_safe = N_cong + eps
        N_free_safe = N_free + eps

        v_cong = sum_cong / N_cong_safe
        v_free = sum_free / N_free_safe
        
        v_min = torch.min(v_cong, v_free)
        w = 0.5 * (1 + torch.tanh((self.v_thr - v_min) / self.v_delta))
        v = w * v_cong + (1 - w) * v_free

        # Handle cases with insufficient data:
        # - If both have valid data: use weighted combination
        # - If only cong has valid data: use v_cong
        # - If only free has valid data: use v_free  
        # - If neither has valid data: output NaN
        both_valid = valid_cong * valid_free
        only_cong = valid_cong * (1 - valid_free)
        only_free = valid_free * (1 - valid_cong)
        neither_valid = (1 - valid_cong) * (1 - valid_free)
        
        v = both_valid * v + only_cong * v_cong + only_free * v_free
        
        # Set output to NaN where neither kernel has sufficient valid data
        v = torch.where(neither_valid > 0.5, torch.tensor(float('nan'), device=v.device, dtype=v.dtype), v)
        
        # print size of v   
        # print('v size:', v.size())
        return v.squeeze(1)