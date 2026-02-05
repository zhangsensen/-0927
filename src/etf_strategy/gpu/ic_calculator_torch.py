"""
GPU-Accelerated IC Calculator (PyTorch Version) | GPU 加速 IC 计算器 (PyTorch 版)
====================================================================================

使用 PyTorch 实现 GPU 加速的 IC 计算
针对 RTX 5070 Ti (sm_120) 优化 - 需要 PyTorch nightly cu128

作者: GPU Performance Team
日期: 2026-02-05
"""

import logging
import time
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


def gpu_available() -> bool:
    """检查 GPU 是否可用"""
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except (ImportError, Exception) as e:
        logger.debug(f"GPU not available: {e}")
        return False


def compute_ic_batch_torch(
    factors_3d: np.ndarray,
    returns_2d: np.ndarray,
    batch_size: int = 128,
) -> Dict[str, np.ndarray]:
    """
    GPU 批量 IC 计算 (PyTorch)
    
    简单实现: 逐因子处理，但全部在 GPU 上完成
    """
    import torch
    
    if not gpu_available():
        raise RuntimeError("GPU not available.")
    
    start_time = time.time()
    
    N_factors, T, M = factors_3d.shape
    logger.info(f"GPU IC batch: {N_factors} factors × {T} days × {M} ETFs")
    
    device = torch.device('cuda:0')
    
    # 上传 returns 到 GPU (一次)
    returns_gpu = torch.from_numpy(returns_2d).float().to(device)
    
    # 结果存储
    ic_means = np.zeros(N_factors)
    ic_stds = np.zeros(N_factors)
    hit_rates = np.zeros(N_factors)
    
    # 批量处理因子
    n_batches = (N_factors + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, N_factors)
        
        # 上传 batch 到 GPU
        batch_factors = torch.from_numpy(factors_3d[start_idx:end_idx]).float().to(device)
        batch_size_actual = end_idx - start_idx
        
        # 逐个因子计算 (在 GPU 上)
        for i in range(batch_size_actual):
            factor = batch_factors[i]  # (T, M)
            
            # 计算每天的 IC
            daily_ics = []
            for t in range(T):
                f_t = factor[t]
                r_t = returns_gpu[t]
                
                # NaN mask
                valid = ~(torch.isnan(f_t) | torch.isnan(r_t))
                n_valid = valid.sum()
                
                if n_valid < 5:
                    continue
                
                f_valid = f_t[valid]
                r_valid = r_t[valid]
                
                # Rank (argsort of argsort)
                f_rank = torch.argsort(torch.argsort(f_valid)).float()
                r_rank = torch.argsort(torch.argsort(r_valid)).float()
                
                # Pearson correlation
                f_mean = f_rank.mean()
                r_mean = r_rank.mean()
                
                num = ((f_rank - f_mean) * (r_rank - r_mean)).sum()
                den = torch.sqrt(((f_rank - f_mean) ** 2).sum() * ((r_rank - r_mean) ** 2).sum())
                
                if den > 0:
                    ic = num / den
                    if torch.isfinite(ic):
                        daily_ics.append(ic)
            
            # 统计结果
            if daily_ics:
                ics_tensor = torch.stack(daily_ics)
                global_idx = start_idx + i
                ic_means[global_idx] = ics_tensor.mean().cpu().item()
                ic_stds[global_idx] = ics_tensor.std().cpu().item()
                hit_rates[global_idx] = (ics_tensor > 0).float().mean().cpu().item()
    
    # 计算 IC_IR
    ic_irs = np.where(ic_stds > 1e-10, ic_means / ic_stds, 0.0)
    
    elapsed = time.time() - start_time
    logger.info(f"Completed in {elapsed:.2f}s ({N_factors / elapsed:.1f} factors/sec)")
    
    return {
        "ic_mean": ic_means,
        "ic_std": ic_stds,
        "ic_ir": ic_irs,
        "hit_rate": hit_rates,
    }


def compute_ic_batch_auto(
    factors_3d: np.ndarray,
    returns_2d: np.ndarray,
    use_gpu: bool = True,
    batch_size: int = 128,
) -> Dict[str, np.ndarray]:
    """自动 CPU/GPU 切换"""
    if use_gpu and gpu_available():
        try:
            logger.info("Using GPU (PyTorch)")
            return compute_ic_batch_torch(factors_3d, returns_2d, batch_size)
        except Exception as e:
            logger.warning(f"GPU failed: {e}. Using CPU.")
    
    # CPU fallback
    logger.info("Using CPU (Numba)")
    from etf_strategy.core.ic_calculator_numba import compute_multiple_ics_numba
    
    N_factors = factors_3d.shape[0]
    ic_means = np.zeros(N_factors)
    ic_stds = np.zeros(N_factors)
    hit_rates = np.zeros(N_factors)
    
    for i in range(N_factors):
        ics = []
        T, M = factors_3d[i].shape
        for t in range(T):
            mask = ~(np.isnan(factors_3d[i, t, :]) | np.isnan(returns_2d[t, :]))
            if mask.sum() < 5:
                continue
            from scipy.stats import spearmanr
            corr, _ = spearmanr(factors_3d[i, t, :][mask], returns_2d[t, :][mask])
            if np.isfinite(corr):
                ics.append(corr)
        
        if ics:
            ic_means[i] = np.mean(ics)
            ic_stds[i] = np.std(ics)
            hit_rates[i] = np.mean([1 if ic > 0 else 0 for ic in ics])
    
    ic_irs = np.where(ic_stds > 1e-10, ic_means / ic_stds, 0.0)
    
    return {
        "ic_mean": ic_means,
        "ic_std": ic_stds,
        "ic_ir": ic_irs,
        "hit_rate": hit_rates,
    }
