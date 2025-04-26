import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

def fft_four_convs(Dp, Mp, k_cong, k_free):
    """
    Compute
        sum_cong = conv2d(Dp, k_cong)
        sum_free = conv2d(Dp, k_free)
        N_cong   = conv2d(Mp, k_cong)
        N_free   = conv2d(Mp, k_free)
    all via FFT.
    
    Dp, Mp:   (B, C, H, W)
    k_cong,:  (F, C, Kh, Kw)
    k_free:   (F, C, Kh, Kw)  (here F=1, but generalizable)
    returns:  sum_cong, N_cong, sum_free, N_free each of shape (B, F, H-Kh+1, W-Kw+1)
    """
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

    # ——— FFT both inputs and kernels ———
    # real→complex FFT over the last two dims
    Df  = torch.fft.rfftn(Dp_pad, dim=(-2, -1), s=(Fh, Fw))
    Mf  = torch.fft.rfftn(Mp_pad, dim=(-2, -1), s=(Fh, Fw))
    Kf1 = torch.fft.rfftn(k1_pad, dim=(-2, -1), s=(Fh, Fw))
    Kf2 = torch.fft.rfftn(k2_pad, dim=(-2, -1), s=(Fh, Fw))

    # ——— pointwise multiply in freq domain ———
    Y1 = Df * Kf1    # for sum_cong
    Y2 = Df * Kf2    # for sum_free
    Z1 = Mf * Kf1    # for N_cong
    Z2 = Mf * Kf2    # for N_free

    # ——— inverse FFT back to real ———
    y1 = torch.fft.irfftn(Y1, dim=(-2, -1), s=(Fh, Fw))
    y2 = torch.fft.irfftn(Y2, dim=(-2, -1), s=(Fh, Fw))
    z1 = torch.fft.irfftn(Z1, dim=(-2, -1), s=(Fh, Fw))
    z2 = torch.fft.irfftn(Z2, dim=(-2, -1), s=(Fh, Fw))

    # ——— crop “valid” region ———
    oh, ow = H - Kh + 1, W - Kw + 1
    sum_cong = y1[..., Kh-1:Kh-1+oh, Kw-1:Kw-1+ow]
    sum_free = y2[..., Kh-1:Kh-1+oh, Kw-1:Kw-1+ow]
    N_cong   = z1[..., Kh-1:Kh-1+oh, Kw-1:Kw-1+ow]
    N_free   = z2[..., Kh-1:Kh-1+oh, Kw-1:Kw-1+ow]

    return sum_cong, N_cong, sum_free, N_free

def fill_space_time_matrix(raw_data, dx = 0.1, dt = 10, lane = 1, data_type = 'speed'):
    data = raw_data[['milemarker', 'time_unix_fix', f'lane{lane}_{data_type}']].copy()
    min_milemarker = round(data['milemarker'].min(),0)
    max_milemarker = round(data['milemarker'].max(),0)
    min_time_unix = int(data['time_unix_fix'].min())
    max_time_unix = int(data['time_unix_fix'].max()) + 30
    milemarkers = np.round(np.arange(min_milemarker, max_milemarker, dx), 2)
    time_range_unix = np.round(np.arange(min_time_unix, max_time_unix, dt), 2)
    matrix = pd.DataFrame(index=time_range_unix, columns=milemarkers)
    for index, row in data.iterrows():
        time_index = row['time_unix_fix']
        milemarker_index = row['milemarker']
        nearest_time = matrix.index.get_indexer([time_index], method='nearest')[0]
        nearest_milemarker = matrix.columns.get_indexer([milemarker_index], method='nearest')[0]
        matrix.iloc[nearest_time, nearest_milemarker] = row[f'lane{lane}_{data_type}']
    return matrix

class AdaptiveSmoothing(nn.Module):
    def __init__(self,
                 smoothing_time_window: float,
                 smoothing_space_window: float,
                 dx: float,
                 dt: float,
                 init_delta: float = 0.10, # mile
                 init_tau: float = 9.0, # seconds
                 init_c_cong: float = -12.0,
                 init_c_free: float = 45.0,
                 init_v_thr: float = 30.0,
                 init_v_delta: float = 10.0):
        super().__init__()
        self.half_t = int(smoothing_time_window / dt)
        self.half_x = int(smoothing_space_window / dx)
        self.dt = dt
        self.dx = dx

        t_offs = torch.arange(-self.half_t, self.half_t + 1) * dt
        x_offs = torch.arange(-self.half_x, self.half_x + 1) * dx
        print(f"t_offs: {t_offs}, x_offs: {x_offs}")
        T, X = torch.meshgrid(t_offs, x_offs, indexing='ij')
        self.register_buffer('T_offsets', T.float())
        self.register_buffer('X_offsets', X.float())

        self.delta   = nn.Parameter(torch.tensor(init_delta))
        self.tau     = nn.Parameter(torch.tensor(init_tau))
        self.c_cong  = nn.Parameter(torch.tensor(init_c_cong))
        self.c_free  = nn.Parameter(torch.tensor(init_c_free))
        self.v_thr   = nn.Parameter(torch.tensor(init_v_thr))
        self.v_delta = nn.Parameter(torch.tensor(init_v_delta))

    def forward(self, raw_data: torch.Tensor):
        # Ensure input is 4D: (B, C, T, X)
        if raw_data.ndim == 2:
            raw_data = raw_data.unsqueeze(0).unsqueeze(0)
        elif raw_data.ndim == 3:
            raw_data = raw_data.unsqueeze(1)

        mask = (~raw_data.isnan()).float()
        plt.imshow(mask[0, 0], cmap='gray', interpolation='nearest', origin='lower')
        plt.savefig('figures/mask.png', dpi=300, bbox_inches='tight')
        plt.close()
        data = torch.nan_to_num(raw_data, nan=0.0)

        c_cong_s = self.c_cong / 3600.0
        c_free_s = self.c_free / 3600.0

        t_cong = self.T_offsets - self.X_offsets / c_cong_s
        t_free = self.T_offsets - self.X_offsets / c_free_s

        k_cong = torch.exp(-(t_cong.abs() / self.tau + self.X_offsets.abs() / self.delta))
        k_free = torch.exp(-(t_free.abs() / self.tau + self.X_offsets.abs() / self.delta))

        k_cong = k_cong.unsqueeze(0).unsqueeze(0)  # (1,1,Kt,Kx)
        k_free = k_free.unsqueeze(0).unsqueeze(0)

        pad = (self.half_x, self.half_x, self.half_t, self.half_t)
        print(f"pad: {pad}")
        Dp = F.pad(data, pad, value=0.0)
        Mp = F.pad(mask, pad, value=0.0)

        # sum_cong = F.conv2d(Dp, k_cong)
        # N_cong   = F.conv2d(Mp, k_cong)
        # sum_free = F.conv2d(Dp, k_free)
        # N_free   = F.conv2d(Mp, k_free)
        sum_cong, N_cong, sum_free, N_free = fft_four_convs(Dp, Mp, k_cong, k_free)

        v_cong = sum_cong / (N_cong)
        v_free = sum_free / (N_free)

        v_min = torch.min(v_cong, v_free)
        w = 0.5 * (1 + torch.tanh((self.v_thr - v_min) / self.v_delta))
        v = w * v_cong + (1 - w) * v_free

        valid_cong = (N_cong >=0).float()
        valid_free = (N_free >=0).float()
        # if no cong data → use free; if no free data → use cong
        v = valid_cong*valid_free*v + (1-valid_cong)*v_free + (1-valid_free)*v_cong
        return v.squeeze(1)


def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    # # raw_data = pd.read_csv('data/2023-10-26.csv')
    # raw_data = pd.read_csv('data/2025-04-09.csv', low_memory=False)
    # speed = fill_space_time_matrix(raw_data, dx = 0.1, dt = 30, lane = 2, data_type = 'speed')
    # output = speed.T.values.copy()
    # # make output to be float32
    # output = output.astype(np.float32)
    # np.save('data/speed.npy', output)
    speed = np.load('data/speed.npy')
    # make the data less than 0 to be nan
    speed[speed < 0] = np.nan
    # get the size of the speed
    time_size, space_size = speed.shape
    # print(f"Time size: {time_size}, Space size: {space_size}")
    # Hyperparameters
    dx = 0.1                   # distance per cell
    dt = 30.0                    # time per cell
    smoothing_time_window = time_size * dt  # seconds
    smoothing_space_window = space_size * dx  # same units as dx
    print(f"Smoothing time window: {smoothing_time_window}, Smoothing space window: {smoothing_space_window}")
    # Instantiate the model
    model = AdaptiveSmoothing(smoothing_time_window,
                              smoothing_space_window,
                              dx, dt).to(device)
    model.eval()
    plt.figure(figsize=(12, 6))
    plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})
    plt.imshow(speed, cmap='RdYlGn', interpolation='nearest', origin='lower',vmin=0, vmax=80, aspect='auto')
    plt.colorbar(label='Speed')
    plt.title('Raw Data')
    plt.tight_layout()
    # plt.gca().invert_yaxis()
    plt.savefig('figures/pre_speed.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    raw = torch.from_numpy(speed).to(device)
    start_time = time.time()
    with torch.no_grad():
        smoothed = model(raw)
    sm = smoothed[0].cpu().numpy()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    # see if the smoothed data is nan
    if np.isnan(sm).any():
        print("Smoothed data contains NaN values.")
    else:
        print("Smoothed data does not contain NaN values.")
    plt.figure(figsize=(12, 6))
    # make the font to be elegant
    plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})
    plt.imshow(sm, cmap='RdYlGn', interpolation='nearest', origin='lower',vmin=0, vmax=80, aspect='auto')
    plt.colorbar(label='Speed')
    plt.title('ASM')
    # reverse the y-axis
    # plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('figures/smoothed_speed.pdf', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
    
