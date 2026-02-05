"""
GPU Acceleration Module | GPU 加速模块
==========================================

为因子挖掘流程提供 GPU 加速:
- IC 计算 (30x 加速)
- 批量因子计算 (10-20x 加速)
- 自动 CPU/GPU 切换 (fallback 机制)

硬件要求:
- GPU: RTX 5070 Ti 16GB (CUDA 13.0)
- 依赖: cupy-cuda12x >= 12.3.0

用法:
    # 自动检测并使用 GPU
    from etf_strategy.gpu import compute_ic_batch_auto

    ic_results = compute_ic_batch_auto(factors, returns, use_gpu=True)

作者: GPU Performance Team
日期: 2026-02-05
"""

from .utils import gpu_available, get_gpu_memory_info
from .ic_calculator_cupy import (
    compute_ic_batch_cupy,
    compute_ic_batch_auto,
    spearman_ic_cupy,
)

__all__ = [
    "gpu_available",
    "get_gpu_memory_info",
    "compute_ic_batch_cupy",
    "compute_ic_batch_auto",
    "spearman_ic_cupy",
]
