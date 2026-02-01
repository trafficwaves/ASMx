import torch
import numpy as np
import pandas as pd
import warnings 
import os
import random
import time
from ASM_utils import AdaptiveSmoothing

# Suppress warnings
warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('medium')

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def weighted_rmse(pred, target, mask=None, threshold=15.0, high_weight=10.0):
    """Compute weighted RMSE (same as train.py)."""
    se = (pred - target) ** 2
    if mask is not None:
        weight = mask.to(dtype=se.dtype)
    else:
        weight = torch.ones_like(se)
    
    # Up-weight low-speed samples
    low_target = (target < threshold).to(dtype=se.dtype)
    weight = weight * (1 + (high_weight - 1) * low_target)
    
    total_weight = weight.sum()
    if total_weight == 0:
        return torch.tensor(0., dtype=se.dtype, device=se.device)
    
    weighted_mse = (se * weight).sum() / total_weight
    return torch.sqrt(weighted_mse)

def quantize(tensor: torch.Tensor, decimals: int = 2):
    """In-place quantization to fixed decimals (same as train.py)."""
    scale = 10.0 ** decimals
    tensor.data.mul_(scale).round_().div_(scale)

def run_calibration(seed, lane=1, num_epochs=1000):
    """Runs a full calibration for a specific seed and returns best params."""
    # 1. Set Seed
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load Data (assuming relative path structure matches train.py)
    if not os.path.exists('../dates.csv'):
        print("Error: ../dates.csv not found. Run from calibration/ folder.")
        return None
        
    dates_df = pd.read_csv('../dates.csv')
    dates = dates_df.date.tolist()
    # Using index 1 (2024-07-09) for train/val as in original train.py
    target_date = dates[1]
    
    gt_path = f'../data/processed_data/motion/lane{lane}/{target_date}.npy'
    sp_path = f'../data/processed_data/rds/lane{lane}/{target_date}.npy'
    
    if not os.path.exists(gt_path) or not os.path.exists(sp_path):
        print(f"Data missing for lane {lane}, {target_date}")
        return None

    # Load numpy files
    gt_np = np.load(gt_path)
    sp_np = np.load(sp_path)
    sp_np[sp_np < 0.0] = np.nan # Treat negative speeds as missing

    # Prepare Tensors
    raw_tensor = torch.from_numpy(sp_np).float().unsqueeze(0).unsqueeze(0).to(device)
    gt_tensor = torch.from_numpy(gt_np).float().unsqueeze(0).unsqueeze(0).to(device)
    
    mask = (~torch.isnan(gt_tensor)).float()
    target = torch.nan_to_num(gt_tensor, nan=0.0)

    # Physics Constants
    dx = 0.02
    dt = 4.0
    kernel_time_window = sp_np.shape[1] * dt
    kernel_space_window = sp_np.shape[0] * dx
    
    # 3. Initialize Model with default hyperparameters
    model = AdaptiveSmoothing(kernel_time_window, kernel_space_window, dx, dt,
                              init_tau=15.0, init_delta=0.15, 
                              init_c_cong=9.3, init_c_free=-43.5,
                              init_v_thr=37.3, init_v_delta=12.4).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    
    # Track best performance
    best_val_rmse = float('inf')
    best_params = {}

    # 4. Training Loop
    for _ in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        smoothed = model(raw_tensor)
        loss = weighted_rmse(smoothed, target, mask)

        loss.backward()
        optimizer.step()
        
        # Constraints & In-place Quantization (Critical step for reproducibility check)
        with torch.no_grad():
            model.c_free.clamp_(min=-60.0)
            
            model.eval()
            for param in (model.tau, model.delta, model.c_cong, model.c_free, model.v_thr, model.v_delta):
                quantize(param, decimals=2)
            
            val_pred = model(raw_tensor)
            val_rmse = weighted_rmse(val_pred, target, mask)
        
        if val_rmse.item() < best_val_rmse:
            best_val_rmse = val_rmse.item()
            best_params = {
                'tau': model.tau.item(),
                'delta': model.delta.item(),
                'c_cong': model.c_cong.item(),
                'c_free': model.c_free.item(),
                'v_thr': model.v_thr.item(),
                'v_delta': model.v_delta.item(),
                'RMSE': best_val_rmse
            }
            
    return best_params

def main():
    results = []
    seeds = range(1, 10) # Seeds 1 to 10
    lane = 1   # Using Lane 1 as representative
    
    print(f"--- Starting Sensitivity Analysis ---\nLane: {lane}\nSeeds: 1-10\nEpochs: 1000 per seed\n")
    
    start_time = time.time()
    
    for i, seed in enumerate(seeds):
        # Print progress
        print(f"Processing Seed {seed:>3} / 10...", end='\r')
        params = run_calibration(seed, lane=lane)
        if params:
            params['seed'] = seed
            results.append(params)
    
    total_time = time.time() - start_time
    print(f"\n\nAnalysis completed in {total_time:.1f} seconds.")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    if df.empty:
        print("No results collected. Check data paths.")
        return

    # Save details
    csv_filename = 'sensitivity_results.csv'
    df.to_csv(csv_filename, index=False)
    print(f"Full results saved to {csv_filename}")
    
    # Calculate Summary Statistics
    summary = df[['tau', 'delta', 'c_cong', 'c_free', 'v_thr', 'v_delta', 'RMSE']].describe().T
    summary = summary[['mean', 'std', 'min', 'max']]
    
    print("\n=== Parameter Stability Summary (100 Runs) ===")
    print(summary)

if __name__ == "__main__":
    main()