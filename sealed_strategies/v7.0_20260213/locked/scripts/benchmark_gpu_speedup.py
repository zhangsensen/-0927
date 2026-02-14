#!/usr/bin/env python3
"""
GPU 性能基准测试 | GPU Performance Benchmark
===========================================

对比 CPU (Numba) vs GPU (CuPy) 的性能:
- IC 计算
- 批量因子处理

预期结果 (10,000 因子 × 1442 天 × 49 ETF):
- CPU: ~1.4 小时
- GPU: ~2-3 分钟 (30x 加速)

用法:
    uv run python scripts/benchmark_gpu_speedup.py --n-factors 1000
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark_gpu_speedup.py --use-gpu

作者: GPU Performance Team
日期: 2026-02-05
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.ic_calculator_numba import compute_spearman_ic_numba
from etf_strategy.gpu import gpu_available, compute_ic_batch_auto


def generate_synthetic_factors(n_factors: int, n_days: int, n_etfs: int) -> np.ndarray:
    """生成合成因子数据 (用于性能测试)"""
    print(f"Generating {n_factors} synthetic factors ({n_days} days × {n_etfs} ETFs)...")
    np.random.seed(42)

    # 模拟真实因子分布: 正态分布 + 20% NaN
    factors = np.random.randn(n_factors, n_days, n_etfs).astype(np.float64)

    # 添加 NaN (模拟停牌/数据缺失)
    nan_mask = np.random.rand(n_factors, n_days, n_etfs) < 0.2
    factors[nan_mask] = np.nan

    return factors


def benchmark_cpu(factors_3d: np.ndarray, returns_2d: np.ndarray) -> float:
    """CPU (Numba) 性能测试"""
    print("\n" + "=" * 80)
    print("CPU (Numba) Benchmark")
    print("=" * 80)

    n_factors = factors_3d.shape[0]
    start = time.time()

    ic_results = []
    for i in range(n_factors):
        ic = compute_spearman_ic_numba(factors_3d[i], returns_2d)
        ic_results.append(ic)

    elapsed = time.time() - start

    print(f"  Processed: {n_factors} factors")
    print(f"  Time: {elapsed:.2f} s")
    print(f"  Throughput: {n_factors / elapsed:.1f} factors/sec")

    return elapsed


def benchmark_gpu(factors_3d: np.ndarray, returns_2d: np.ndarray, batch_size: int = 128) -> float:
    """GPU (CuPy) 性能测试"""
    print("\n" + "=" * 80)
    print(f"GPU (CuPy) Benchmark (batch_size={batch_size})")
    print("=" * 80)

    if not gpu_available():
        print("  ⚠️  GPU not available. Skipping.")
        return 0.0

    n_factors = factors_3d.shape[0]
    start = time.time()

    results = compute_ic_batch_auto(factors_3d, returns_2d, use_gpu=True, batch_size=batch_size)

    elapsed = time.time() - start

    print(f"  Processed: {n_factors} factors")
    print(f"  Time: {elapsed:.2f} s")
    print(f"  Throughput: {n_factors / elapsed:.1f} factors/sec")

    return elapsed


def main():
    parser = argparse.ArgumentParser(description="GPU 性能基准测试")
    parser.add_argument("--n-factors", type=int, default=1000, help="因子数量 (默认 1000)")
    parser.add_argument("--n-days", type=int, default=1442, help="天数 (默认 1442)")
    parser.add_argument("--n-etfs", type=int, default=49, help="ETF 数量 (默认 49)")
    parser.add_argument("--use-real-data", action="store_true", help="使用真实数据 (默认合成)")
    parser.add_argument("--cpu-only", action="store_true", help="仅测试 CPU")
    parser.add_argument("--gpu-only", action="store_true", help="仅测试 GPU")
    parser.add_argument("--batch-size", type=int, default=128, help="GPU 批次大小")
    args = parser.parse_args()

    print("\n" + "█" * 80)
    print("  GPU Performance Benchmark")
    print("  CPU (Numba) vs GPU (CuPy)")
    print("█" * 80)

    # ── Step 1: 准备数据 ──
    if args.use_real_data:
        print("\nLoading real data...")
        data_dir = str(PROJECT_ROOT / "raw" / "ETF" / "daily")
        loader = DataLoader(data_dir=data_dir)
        ohlcv = loader.load_ohlcv(start_date="2020-01-01", end_date="2025-12-31")
        close = ohlcv["close"]
        returns = close.pct_change().shift(-3)  # FREQ=3

        n_days, n_etfs = returns.shape
        returns_2d = returns.fillna(0).values

        # 需要真实因子数据 (这里简化, 仍用合成)
        factors_3d = generate_synthetic_factors(args.n_factors, n_days, n_etfs)
    else:
        factors_3d = generate_synthetic_factors(args.n_factors, args.n_days, args.n_etfs)
        returns_2d = np.random.randn(args.n_days, args.n_etfs) * 0.02  # 模拟 2% 日波动

    print(f"\nData shape:")
    print(f"  Factors: {factors_3d.shape}")
    print(f"  Returns: {returns_2d.shape}")
    print(f"  Total operations: {factors_3d.shape[0] * factors_3d.shape[1] * factors_3d.shape[2]:,}")

    # ── Step 2: 基准测试 ──
    cpu_time = 0
    gpu_time = 0

    if not args.gpu_only:
        cpu_time = benchmark_cpu(factors_3d, returns_2d)

    if not args.cpu_only:
        gpu_time = benchmark_gpu(factors_3d, returns_2d, batch_size=args.batch_size)

    # ── Step 3: 性能对比 ──
    if cpu_time > 0 and gpu_time > 0:
        speedup = cpu_time / gpu_time
        print("\n" + "=" * 80)
        print("Performance Summary")
        print("=" * 80)
        print(f"  CPU Time:  {cpu_time:.2f} s ({cpu_time / 60:.1f} min)")
        print(f"  GPU Time:  {gpu_time:.2f} s")
        print(f"  Speedup:   {speedup:.1f}x")
        print()
        print(f"  Estimated time for 10,000 factors:")
        print(f"    CPU: {cpu_time * 10:.1f} s ({cpu_time * 10 / 60:.1f} min)")
        print(f"    GPU: {gpu_time * 10:.1f} s ({gpu_time * 10 / 60:.1f} min)")
        print("=" * 80)

        # 验证预期
        if speedup >= 20:
            print("  ✓ GPU 加速达标 (>= 20x)")
        elif speedup >= 10:
            print("  ○ GPU 加速中等 (10-20x)")
        else:
            print("  ⚠️  GPU 加速低于预期 (< 10x)")


if __name__ == "__main__":
    main()
