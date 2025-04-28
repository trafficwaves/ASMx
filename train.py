import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ASM_utils import AdaptiveSmoothing
import warnings 
warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('medium')

def rmse(pred, target, mask=None):
    """Compute weighted RMSE; if mask is provided, only over mask==1.
    Congested (target<15) weighted 4x, free (>=15) weighted 1x."""
    congest = (target < 30).float()
    free = (target >= 30).float()
    weights = 4.0 * congest + 1.0 * free
    valid = weights if mask is None else weights * mask
    diff2 = (pred - target) ** 2 * valid
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return torch.sqrt(diff2.sum())


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
    l1 = rmse(pred, target, mask)
    l2 = wasserstein_distance(pred, target, mask)
    return alpha * l1 + (1 - alpha) * l2

def quantize(tensor: torch.Tensor, decimals: int = 2):
    """
    In‐place quantization of `tensor` to the given number of decimals.
    E.g. decimals=2 → nearest 0.01.
    """
    scale = 10.0 ** decimals
    tensor.data.mul_(scale).round_().div_(scale)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dates
    dates = pd.read_csv('dates.csv').date.tolist()
    train_dates = dates[0:1]
    val_date = dates[0]

    # Load all training data
    train_raws, train_gts = [], []
    for date in train_dates:
        gt_np = np.load(f'data/processed_data/motion/lane1/{date}.npy')
        sp_np = np.load(f'data/processed_data/rds/lane1/{date}.npy')
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
    val_gt_np = np.load(f'data/processed_data/motion/lane1/{val_date}.npy')
    val_sp_np = np.load(f'data/processed_data/rds/lane1/{val_date}.npy')
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
                              init_tau= 30.0, init_delta= 1.0, 
                              init_c_cong= 20.0, init_c_free= -45.0,
                              init_v_thr= 50.0, init_v_delta= 10.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=1e-2, 
                                 weight_decay=1e-5)
    num_epochs = 200

    best_val_rmse = float('inf')
    best_model_path = 'best_model.pth'

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
            # model.tau.clamp_(min=1.0, max=30.0)
            # model.delta.clamp_(min=0.05, max=0.50)
            # model.c_cong.clamp_(min=11.0, max=12.0)
            # model.c_free.clamp_(max= -60.0, min= -40.0)
            # model.v_thr.clamp_(min=20.0, max=60.0)
            # model.v_delta.clamp_(min=5.0, max=10.0)
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
        if epoch % 100 == 0 or epoch == 1:
            print(f"tau: {model.tau.item():.2f}, delta: {model.delta.item():.2f}, "
                  f"c_cong: {model.c_cong.item():.2f}, c_free: {model.c_free.item():.2f}, "
                  f"v_thr: {model.v_thr.item():.2f}, v_delta: {model.v_delta.item():.2f}")
    # draw the training and validation RMSE
    plt.plot(range(1, num_epochs+1), [loss.item() for _ in range(num_epochs)], label='Train RMSE')
    plt.plot(range(1, num_epochs+1), [val_rmse.item() for _ in range(num_epochs)], label='Validation RMSE')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.title('Training and Validation RMSE')
    plt.legend()
    plt.savefig('train_val_rmse.pdf', dpi=300, bbox_inches='tight')
    plt.close()
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
    plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})
    plt.imshow(sm, cmap='RdYlGn', interpolation='nearest', origin='lower',vmin=0, vmax=80, aspect='auto')
    plt.colorbar(label='Speed')
    plt.title('ASM Smoothed Speed')
    plt.tight_layout()
    plt.savefig('figures/smoothed_speed_val.pdf', dpi=300, bbox_inches='tight')
    plt.close()
if __name__ == "__main__":
    main()