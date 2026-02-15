"""
GPU Utility Functions | GPU 工具函数
=====================================

CPU/GPU 自动切换、内存管理、性能监控

作者: GPU Performance Team
日期: 2026-02-05
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def gpu_available() -> bool:
    """
    检查 GPU 是否可用

    Returns:
        True if GPU available, False otherwise
    """
    try:
        import cupy as cp

        # 尝试创建测试数组
        _ = cp.array([1.0])
        return True
    except ImportError:
        logger.warning("CuPy not installed. Install with: uv add --optional gpu cupy-cuda12x")
        return False
    except Exception as e:
        logger.warning(f"GPU not available: {e}")
        return False


def get_gpu_memory_info() -> Dict[str, Any]:
    """
    获取 GPU 内存信息

    Returns:
        Dict with keys: total_gb, free_gb, used_gb, utilization_pct
    """
    if not gpu_available():
        return {"error": "GPU not available"}

    try:
        import cupy as cp

        mempool = cp.get_default_memory_pool()
        used_bytes = mempool.used_bytes()
        total_bytes = mempool.total_bytes()

        # 获取 GPU 设备属性
        device = cp.cuda.Device()
        total_memory = device.mem_info[1]  # Total memory in bytes
        free_memory = device.mem_info[0]   # Free memory in bytes

        return {
            "device": device.id,
            "device_name": device.compute_capability,
            "total_gb": total_memory / (1024**3),
            "free_gb": free_memory / (1024**3),
            "used_gb": (total_memory - free_memory) / (1024**3),
            "utilization_pct": (1 - free_memory / total_memory) * 100,
            "mempool_used_gb": used_bytes / (1024**3),
            "mempool_total_gb": total_bytes / (1024**3),
        }
    except Exception as e:
        logger.error(f"Failed to get GPU memory info: {e}")
        return {"error": str(e)}


def estimate_batch_size(n_factors: int, n_days: int, n_etfs: int, max_memory_gb: float = 14.0) -> int:
    """
    估算最优批次大小 (避免 GPU 内存溢出)

    Args:
        n_factors: 因子数量
        n_days: 天数
        n_etfs: ETF 数量
        max_memory_gb: 最大允许内存 (GB), 默认 14GB (留 2GB buffer for 16GB card)

    Returns:
        建议的批次大小
    """
    # 每个因子矩阵大小: (n_days, n_etfs) × 8 bytes (float64)
    bytes_per_factor = n_days * n_etfs * 8

    # 加上中间计算的开销 (rank 矩阵, 临时数组等)
    overhead_multiplier = 3.0

    max_bytes = max_memory_gb * (1024**3)
    max_factors_in_memory = int(max_bytes / (bytes_per_factor * overhead_multiplier))

    # 保守估计, 取 70% 作为批次大小
    batch_size = max(1, int(max_factors_in_memory * 0.7))

    logger.info(
        f"Estimated batch size: {batch_size} factors "
        f"({bytes_per_factor * batch_size / (1024**3):.2f} GB per batch)"
    )

    return batch_size


def log_performance_comparison(cpu_time: float, gpu_time: float, task_name: str):
    """
    记录 CPU vs GPU 性能对比

    Args:
        cpu_time: CPU 耗时 (秒)
        gpu_time: GPU 耗时 (秒)
        task_name: 任务名称
    """
    speedup = cpu_time / gpu_time if gpu_time > 0 else 0
    logger.info(
        f"Performance [{task_name}]: "
        f"CPU={cpu_time:.2f}s, GPU={gpu_time:.2f}s, Speedup={speedup:.1f}x"
    )
