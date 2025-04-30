import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from ASM_utils import AdaptiveSmoothing
import warnings 
import os
warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('medium')

def rmse(pred, target, mask=None):
    """Compute RMSE; if mask is provided, only over mask==1.
    """
    # Compute RMSE
    diff = (pred - target) ** 2
    if mask is not None:
        diff = diff * mask
    rmse_value = torch.sqrt(diff.mean())
    return rmse_value

def weighted_rmse(pred, target, mask=None, threshold=15.0, high_weight=10.0):
    """
    Compute a masked, weighted RMSE.

    Args:
        pred (Tensor): predicted values.
        target (Tensor): ground-truth values.
        mask (Tensor, optional): same shape as pred/target, binary (0/1) where 1=keep, 0=ignore.
        threshold (float): targets below this get up-weighted.
        high_weight (float): multiplier applied to any sample with target < threshold.

    Returns:
        Tensor: scalar RMSE.
    """
    # squared error
    se = (pred - target) ** 2

    # base weight = 1 everywhere (or 0 where mask==0)
    if mask is not None:
        weight = mask.to(dtype=se.dtype)
    else:
        weight = torch.ones_like(se)

    # up-weight low-target samples
    low_target = (target < threshold).to(dtype=se.dtype)
    weight = weight * (1 + (high_weight - 1) * low_target)

    # compute weighted mean squared error
    # avoid division by zero if all weights are zero
    total_weight = weight.sum()
    if total_weight == 0:
        return torch.tensor(0., dtype=se.dtype, device=se.device)

    weighted_mse = (se * weight).sum() / total_weight

    return torch.sqrt(weighted_mse)


def wasserstein_distance(pred, target, mask=None):
    """Compute empirical 1D Wasserstein distance between pred and target."""
    # Flatten tensors
    p = pred.view(-1)
    t = target.view(-1)
    if mask is not None:
        m = mask.view(-1) > 0
        p = p[m]
        t = t[m]
    if p.numel() == 0 or t.numel() == 0:
        return torch.tensor(0.0, device=pred.device)
    # Sort and compute mean absolute difference
    p_sorted, _ = torch.sort(p)
    t_sorted, _ = torch.sort(t)
    n = min(p_sorted.numel(), t_sorted.numel())
    return torch.mean(torch.abs(p_sorted[:n] - t_sorted[:n]))


def combined_loss(pred, target, mask=None, alpha=1.0):
    """
    Combined loss = alpha * RMSE + (1-alpha) * Wasserstein Distance.
    Use alpha in [0,1] to balance.
    """
    l1 = weighted_rmse(pred, target, mask)
    l2 = wasserstein_distance(pred, target, mask)
    return alpha * l1 + (1 - alpha) * l2

def quantize(tensor: torch.Tensor, decimals: int = 2):
    """
    In‐place quantization of `tensor` to the given number of decimals.
    E.g. decimals=2 → nearest 0.01.
    """
    scale = 10.0 ** decimals
    tensor.data.mul_(scale).round_().div_(scale)

def calib(lane):
    # use the runid based on current time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load dates
    dates = pd.read_csv('dates.csv').date.tolist()
    train_dates = dates[1:2]
    val_date = dates[1]

    # Load all training data
    train_raws, train_gts = [], []
    for date in train_dates:
        gt_np = np.load(f'data/processed_data/motion/lane{lane}/{date}.npy')
        sp_np = np.load(f'data/processed_data/rds/lane{lane}/{date}.npy')
        # sp_np < 0.0 → NaN
        sp_np[sp_np < 0.0] = np.nan
        raw = torch.from_numpy(sp_np).float().unsqueeze(0).unsqueeze(0)  # (1,1,T,X)
        gt = torch.from_numpy(gt_np).float().unsqueeze(0).unsqueeze(0)    # (1,1,T,X)

        train_raws.append(raw)
        train_gts.append(gt)

    # Stack training data along batch dim
    train_raw = torch.cat(train_raws, dim=0).to(device)  # (B,1,T,X)
    train_gt = torch.cat(train_gts, dim=0).to(device)    # (B,1,T,X)

    # Validation data
    val_gt_np = np.load(f'data/processed_data/motion/lane{lane}/{val_date}.npy')
    val_sp_np = np.load(f'data/processed_data/rds/lane{lane}/{val_date}.npy')
    val_sp_np[val_sp_np < 0.0] = np.nan
    val_raw = torch.from_numpy(val_sp_np).float().unsqueeze(0).unsqueeze(0).to(device)
    val_gt = torch.from_numpy(val_gt_np).float().unsqueeze(0).unsqueeze(0).to(device)

    # Masks
    train_mask = (~torch.isnan(train_gt)).float()
    train_gt = torch.nan_to_num(train_gt, nan=0.0)

    val_mask = (~torch.isnan(val_gt)).float()
    val_gt = torch.nan_to_num(val_gt, nan=0.0)

    # Hyperparameters
    dx = 0.02
    dt = 4.0
    kernel_time_window = val_sp_np.shape[1] * dt
    kernel_space_window = val_sp_np.shape[0] * dx
    print('kernel_time_window:', kernel_time_window)
    print('kernel_space_window:', kernel_space_window)
    # Model & optimizer
    model = AdaptiveSmoothing(kernel_time_window, kernel_space_window, dx, dt,
                              init_tau= 15.0, init_delta= 0.15, 
                              init_c_cong= 9.3, init_c_free= -43.5,
                              init_v_thr= 37.3, init_v_delta= 12.4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=1e-1)
    num_epochs = 1000
    best_val_rmse = float('inf')
    best_model_path = f'best_model_lane{lane}.pth'

    # Training loop
    for epoch in range(1, num_epochs+1):
        model.train()
        optimizer.zero_grad()

        smoothed = model(train_raw)
        loss = combined_loss(smoothed, train_gt, train_mask)

        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            for param in (
                    model.tau, model.delta,
                    model.c_cong, 
                    model.c_free,
                    model.v_thr, model.v_delta
                ):
                    quantize(param, decimals=2)

            val_pred = model(val_raw)
            val_rmse = combined_loss(val_pred, val_gt, val_mask)

        if val_rmse.item() < best_val_rmse:
            best_val_rmse = val_rmse.item()
            torch.save(model.state_dict(), best_model_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} — Train RMSE: {loss.item():.4f} — Val RMSE: {val_rmse.item():.4f}")
            # save all the parameters to a jason file as well
            # Save all model parameters to JSON, appending each epoch's params to a list
            params = {
                'epoch': epoch,
                'tau': model.tau.item(),
                'delta': model.delta.item(),
                'c_cong': model.c_cong.item(),
                'c_free': model.c_free.item(),
                'v_thr': model.v_thr.item(),
                'v_delta': model.v_delta.item(),
                'train_rmse': loss.item(),
                'val_rmse': val_rmse.item()
            }
            # Append to params_history.json
            params_file = f'params_history_lane{lane}.json'
            if os.path.exists(params_file):
                with open(params_file, 'r') as f:
                    params_list = json.load(f)
            else:
                params_list = []
            params_list.append(params)
            with open(params_file, 'w') as f:
                json.dump(params_list, f, indent=4)
        if epoch % 100 == 0 or epoch == 1:
            print(f"tau: {model.tau.item():.2f}, delta: {model.delta.item():.2f}, "
                  f"c_cong: {model.c_cong.item():.2f}, c_free: {model.c_free.item():.2f}, "
                  f"v_thr: {model.v_thr.item():.2f}, v_delta: {model.v_delta.item():.2f}")
    print(f"\nBest Validation RMSE: {best_val_rmse:.4f}")
    print(f"Best model saved at {best_model_path}")

    # Optionally, load and print final best model
    model.load_state_dict(torch.load(best_model_path))
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.data:.2f}")
     # run the results for the validation set
     # load the best model
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    with torch.no_grad():
        smoothed = model(train_raw)
    sm = smoothed[0].cpu().numpy()
    print('sm shape:', sm.shape)
    # get only the first 200 rows and 200 columns
    # sm = sm[:200, :200]
    # visualize the results
    plt.figure(figsize=(12, 6))
    plt.rcParams.update({'font.size': 20, 'font.family': 'serif'})
    plt.imshow(sm, cmap='RdYlGn', interpolation='nearest', origin='lower',vmin=0, vmax=80, aspect='auto')
    plt.colorbar(label='Speed')
    plt.title('ASM Smoothed Speed')
    plt.tight_layout()
    # reverse the y-axis
    plt.gca().invert_yaxis()
    plt.savefig(f'figures/smoothed_speed_val_lane{lane}.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    import time
    for lane in range(1, 5):
        start_time = time.time()
        print(f'Calibrating lane {lane}...')
        calib(lane)
        end_time = time.time()
        print(f'Calibration time for lane {lane}: {end_time - start_time:.2f} seconds')
        print(f'Finished calibrating lane {lane}.')
if __name__ == "__main__":
    main()