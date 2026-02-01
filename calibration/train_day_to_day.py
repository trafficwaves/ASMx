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
import random
import thop
import time
import logging

# Set random seeds for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
# If using CUDA:
torch.cuda.manual_seed_all(seed)
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

def count_adaptive_smoothing(m, x, y):
    # x is input tuple
    input_tensor = x[0]
    
    # Check input dimensions to determine B, C, H, W
    if input_tensor.dim() == 4:
        b, c, h, w = input_tensor.shape
    elif input_tensor.dim() == 3:
        # unsqueeze(1) -> (B, 1, H, W)
        b, h, w = input_tensor.shape
        c = 1
    elif input_tensor.dim() == 2:
         # unsqueeze(0).unsqueeze(0) -> (1, 1, H, W)
         b, c = 1, 1
         h, w = input_tensor.shape
    else:
        # Fallback
        return

    # Kernel sizes (from ASM_utils Logic)
    # size_t, size_x are used for padding and kernel creation
    size_t = m.size_t
    size_x = m.size_x
    
    # 1. Padding input (F.pad)
    # Dp shape: H + 2*size_t, W + 2*size_x
    hp = h + 2 * size_t
    wp = w + 2 * size_x
    
    # 2. Kernel shape
    # Kh = 2*size_t + 1, Kw = 2*size_x + 1
    kh = 2 * size_t + 1
    kw = 2 * size_x + 1
    
    # 3. FFT Padding in fft_four_convs
    # Fh = Hp + Kh - 1 => (H + 2st) + (2st + 1) - 1 = H + 4st
    # Fw = Wp + Kw - 1 => W + 4sx
    fh = hp + kh - 1
    fw = wp + kw - 1
    
    # Number of elements in standard domain
    n_elements = fh * fw
    
    # Operations in fft_four_convs:
    # A. 4 RFFTs (Dp, Mp, k_cong, k_free)
    #    Approx MACs per FFT: 2.5 * N * log2(N)
    #    (Note: k_cong/k_free are computed once per batch but here treated as dynamic inputs to fft function)
    fft_macs = 2.5 * n_elements * np.log2(n_elements)
    total_fft_macs = 4 * fft_macs * b * c
    
    # B. 4 Complex Multiplications (Y1, Y2, Z1, Z2)
    #    Complex Input Size: Fh * (Fw // 2 + 1)
    n_complex = fh * (fw // 2 + 1)
    #    1 Complex Mult ~ 4 Real Mults (MACs).
    complex_mult_macs = 4 * n_complex
    total_mult_macs = 4 * complex_mult_macs * b * c
    
    # C. 4 IRFFTs
    #    Same as RFFT
    total_ifft_macs = 4 * fft_macs * b * c
    
    # Total MACs for convolution part
    m.total_ops += torch.DoubleTensor([int(total_fft_macs + total_mult_macs + total_ifft_macs)])


def benchmark_performance(model, input_tensor, device, num_iter=50):
    logger = logging.getLogger(__name__)
    logger.info("\n--- Computational Performance Benchmark ---")
    model.eval()
    
    # 1. Total Parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total Parameters: {total_params}")

    # 2. FLOPs using thop
    try:
        # thop profile expects a tuple of inputs
        # Define custom_ops to handle AdaptiveSmoothing
        custom_ops = {AdaptiveSmoothing: count_adaptive_smoothing}
        macs, _ = thop.profile(model, inputs=(input_tensor, ), custom_ops=custom_ops, verbose=False)
        flops = 2 * macs 
        logger.info(f"FLOPs (est): {flops:.2e}")
        logger.info(f"MACs (est): {macs:.2e}")
    except Exception as e:
        logger.error(f"FLOPs calculation failed: {e}")


    # 3. Throughput
    # Warmup
    try:
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iter):
                 _ = model(input_tensor)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_iter
        throughput = 1.0 / avg_time
        logger.info(f"Avg Inference Time: {avg_time*1000:.2f} ms")
        logger.info(f"Throughput: {throughput:.2f} inferences/sec (batch size {input_tensor.shape[0]})")
    except Exception as e:
        logger.error(f"Throughput measurement failed: {e}")

    # 4. Memory Footprint
    if device.type == 'cuda':
        try:
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = model(input_tensor)
            max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) # MB
            logger.info(f"Peak CUDA Memory: {max_mem:.2f} MB")
        except Exception as e:
            logger.error(f"Memory tracking failed: {e}")
    else:
        logger.info("Memory footprint tracking requires CUDA.")
    
    # Switch back to train mode if needed, though caller should handle
    logger.info("-------------------------------------------\n")

# generate a runid with characters and numbers based on the current time
import datetime
import string

def generate_runid():
    now = datetime.datetime.now()
    time_str = now.strftime("%Y%m%d_%H%M%S")
    return f"{time_str}"


def calib(lane, runid, date_id=1, base_log_dir=None, base_model_dir=None):
    logger = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup directories
    if base_log_dir:
        log_dir = os.path.join(base_log_dir, str(date_id))
    else:
        log_dir = f'../logs/calibration/{runid}/{date_id}'
        
    if base_model_dir:
        model_dir = os.path.join(base_model_dir, str(date_id))
    else:
        model_dir = f'../model/{runid}/{date_id}'
        
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Load dates
    dates = pd.read_csv('../dates.csv').date.tolist()
    train_dates = dates[date_id:date_id+1]
    val_date = dates[date_id]
    logger.info(f"Training on dates: {train_dates}, Validating on date: {val_date}")
    # Load all training data
    train_raws, train_gts = [], []
    for date in train_dates:
        gt_np = np.load(f'../data/processed_data/motion/lane{lane}/{date}.npy')
        sp_np = np.load(f'../data/processed_data/rds/lane{lane}/{date}.npy')
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
    val_gt_np = np.load(f'../data/processed_data/motion/lane{lane}/{val_date}.npy')
    val_sp_np = np.load(f'../data/processed_data/rds/lane{lane}/{val_date}.npy')
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
    logger.info(f'kernel_time_window: {kernel_time_window}')
    logger.info(f'kernel_space_window: {kernel_space_window}')
    # Model & optimizer
    model = AdaptiveSmoothing(kernel_time_window, kernel_space_window, dx, dt,
                              init_tau= 15.0, init_delta= 0.15, 
                              init_c_cong= 9.3, init_c_free= -43.5,
                              init_v_thr= 37.3, init_v_delta= 12.4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=1e-1)
    
    # Run Benchmark
    benchmark_performance(model, train_raw, device)

    num_epochs = 1000
    best_val_rmse = float('inf')
    params_list = []

    # Training loop
    for epoch in range(1, num_epochs+1):
        model.train()
        optimizer.zero_grad()

        smoothed = model(train_raw)
        loss = weighted_rmse(smoothed, train_gt, train_mask)

        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.c_free.clamp_(min=-60.0)  # c_free should be <= 60
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
            val_rmse = weighted_rmse(val_pred, val_gt, val_mask)

        if val_rmse.item() < best_val_rmse:
            best_val_rmse = val_rmse.item()
            # Save the best model parameters
            best_model_path = os.path.join(model_dir, f'best_model_lane{lane}.pt')
            torch.save({
                'tau': model.tau,
                'delta': model.delta,
                'c_cong': model.c_cong,
                'c_free': model.c_free,
                'v_thr': model.v_thr,
                'v_delta': model.v_delta
            }, best_model_path)
            
        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:3d} — Train RMSE: {loss.item():.4f} — Val RMSE: {val_rmse.item():.4f}")
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
            params_list.append(params)
            
            # Save parameters to a JSON file (overwrite with updated list)
            params_file = os.path.join(log_dir, f'params_history_lane{lane}.json')
            with open(params_file, 'w') as f:
                json.dump(params_list, f, indent=4)

        if epoch % 100 == 0 or epoch == 1:
            logger.info(f"tau: {model.tau.item():.2f}, delta: {model.delta.item():.2f}, "
                  f"c_cong: {model.c_cong.item():.2f}, c_free: {model.c_free.item():.2f}, "
                  f"v_thr: {model.v_thr.item():.2f}, v_delta: {model.v_delta.item():.2f}")
    
    logger.info(f"\nBest Validation RMSE: {best_val_rmse:.4f}")
    if 'best_model_path' in locals():
        logger.info(f"Best model saved at {best_model_path}")

def main():
    import time
    import os
    # use the runid based on current time
    runid = generate_runid()
    
    # Setup directories
    base_log_dir = f'../logs/calibration/{runid}'
    base_model_dir = f'../model/{runid}'
    os.makedirs(base_log_dir, exist_ok=True)
    os.makedirs(base_model_dir, exist_ok=True)
    
    # Setup Logging
    log_file = os.path.join(base_log_dir, 'calibration_log.txt')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Run ID: {runid}")
    
    for lane in range(1, 5):
        start_time = time.time()
        logger.info(f"Calibrating lane {lane}...")
        
        for date_id in range(0, 5):  # Calibrate on dates 1 to 5
            logger.info(f"  Using date index {date_id} for calibration.")
            
            calib(lane, runid, date_id, base_log_dir, base_model_dir)
            
            end_time = time.time()
            elapsed = end_time - start_time
            logger.info(f"Calibration time for lane {lane} and date {date_id}: {elapsed:.2f} seconds")
            logger.info(f"Finished calibrating lane {lane} and date {date_id}.\n")

    logger.info(f"All lanes calibrated. Logs saved to {log_file}")

if __name__ == "__main__":
    main()