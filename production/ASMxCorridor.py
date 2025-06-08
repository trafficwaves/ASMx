import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import os

def fft_four_convs(Dp, Mp, k_cong, k_free, eps=0, use_ortho=True):
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

    # ——— optional epsilon to counts to avoid zero division downstream ———
    N_cong = N_cong + eps
    N_free = N_free + eps

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
                 kernel_time_window: float,
                 kernel_space_window: float,
                 dx: float,
                 dt: float,
                 init_delta: float = 0.09, # mile
                 init_tau: float = 9.64, # seconds
                 init_c_cong: float = 13.05,
                 init_c_free: float = -47.99,
                 init_v_thr: float = 55.26,
                 init_v_delta: float = 10.55):
                #  init_delta: float = 0.15, # mile
                #  init_tau: float = 15.0, # seconds
                #  init_c_cong: float = 9.3,
                #  init_c_free: float = -43.5,
                #  init_v_thr: float = 37.3,
                #  init_v_delta: float = 12.4):
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
        # Ensure input is 4D: (B, C, T, X)
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

        v_cong = sum_cong / N_cong
        v_free = sum_free / N_free
        v_min = torch.min(v_cong, v_free)
        w = 0.5 * (1 + torch.tanh((self.v_thr - v_min) / self.v_delta))
        v = w * v_cong + (1 - w) * v_free

        valid_cong = (N_cong > 0).float()
        valid_free = (N_free > 0).float()
        # if no cong data → use free; if no free data → use cong
        v = valid_cong*valid_free*v + (1-valid_cong)*v_free + (1-valid_free)*v_cong
        # check if there's nan if so print
        if torch.isnan(v).any():
            print("Warning! NaN detected in output")
            print(N_cong)
        # print size of v   
        # print('v size:', v.size())
        return v.squeeze(1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # Load the data
    speed = np.load('data/corridor_data/rds/lane1/2024-07-09.npy')
    speed[speed < 0] = np.nan
    # get the size of the speed
    print('speed shape:', speed.shape)
    time_size, space_size = speed.shape
    # Hyperparameters
    dx = 0.02                  # distance per cell
    dt = 4.0                    # time per cell
    kernel_time_window = time_size * dt  # seconds
    kernel_space_window = space_size * dx  # same units as dx
    # Instantiate the model
    model = AdaptiveSmoothing(kernel_time_window,
                              kernel_space_window,
                              dx, dt).to(device)
    # model.load_state_dict(torch.load(best_model_path))
    model.eval()
    plt.figure(figsize=(12, 6))
    plt.rcParams.update({'font.size': 20, 'font.family': 'serif'})
    plt.imshow(speed, cmap='RdYlGn', interpolation='nearest', origin='lower',vmin=0, vmax=80, aspect='auto')
    plt.colorbar(label='Speed')
    plt.title('RDS Raw Data')
    plt.tight_layout()
    # reverse the y-axis
    plt.gca().invert_yaxis()
    plt.savefig('figures/pre_speed_corridor.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    raw = torch.from_numpy(speed).to(device)
    start_time = time.time()
    with torch.no_grad():
        smoothed = model(raw)
    sm = smoothed[0].cpu().numpy()
    # save the smoothed data
    sm = sm.astype(np.float32)
    # save the results to the folder 
    # np.save('2024-07-09_sm_4.npy', sm)
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    # # see if the smoothed data is nan
    # if np.isnan(sm).any():
    #     print("Smoothed data contains NaN values.")
    # else:
    #     print("Smoothed data does not contain NaN values.")
    plt.figure(figsize=(12, 6))
    # make the font to be elegant
    plt.rcParams.update({'font.size': 20, 'font.family': 'serif'})
    plt.imshow(sm, cmap='RdYlGn', interpolation='nearest', origin='lower',vmin=0, vmax=80, aspect='auto')
    plt.colorbar(label='Speed')
    plt.title('ASM')
    # reverse the y-axis
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('figures/corridor.pdf', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
    