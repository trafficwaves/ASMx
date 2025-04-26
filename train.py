import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ASM_utils import AdaptiveSmoothing, fft_four_convs 

def rmse(pred, target, mask=None):
    """Compute RMSE; if mask is provided, only over mask==1. 
    Additionally, only consider target values < 15."""
    valid = (target < 80).float()
    if mask is not None:
        valid = valid * mask
    diff2 = (pred - target) ** 2 * valid
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    mse = diff2.sum() / valid.sum()
    return mse

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dates
    dates = pd.read_csv('dates.csv').date.tolist()
    train_dates = dates[0:1]
    val_date = dates[1]

    # Load all training data
    train_raws, train_gts = [], []
    for date in train_dates:
        gt_np = np.load(f'data/processed_data/motion/lane1/{date}.npy')
        sp_np = np.load(f'data/processed_data/rds/lane1/{date}.npy')
        
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
    kernel_time_window = val_sp_np.shape[0] * dt
    kernel_space_window = val_sp_np.shape[1] * dx

    # Model & optimizer
    model = AdaptiveSmoothing(kernel_time_window, kernel_space_window, dx, dt,
                              init_tau= 30.0, init_delta= 1.0, 
                              init_c_cong= 12.0, init_c_free= -60.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
    num_epochs = 1000

    best_val_rmse = float('inf')
    best_model_path = 'best_model.pth'

    # Training loop
    for epoch in range(1, num_epochs+1):
        model.train()
        optimizer.zero_grad()

        smoothed = model(train_raw)
        loss = rmse(smoothed, train_gt, train_mask)

        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(val_raw)
            val_rmse = rmse(val_pred, val_gt, val_mask)

        if val_rmse.item() < best_val_rmse:
            best_val_rmse = val_rmse.item()
            torch.save(model.state_dict(), best_model_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} — Train RMSE: {loss.item():.4f} — Val RMSE: {val_rmse.item():.4f}")
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
            print(f"{name}: {param.data}")

     # run the results for the validation set
     # load the best model
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    with torch.no_grad():
        smoothed = model(train_raw)
    sm = smoothed[0].cpu().numpy()
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