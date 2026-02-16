#!/usr/bin/env python3
"""
GPU/CPU 结果一致性验证 | GPU-CPU Alignment Verification
=====================================================

验证 GPU (CuPy) 和 CPU (Numba) 的 IC 计算结果是否一致

容差标准:
- 绝对差异 < 1e-6 (浮点精度)
- 相对差异 < 0.1%

用法:
    uv run python scripts/verify_gpu_cpu_alignment.py --n-factors 100

作者: GPU Performance Team
日期: 2026-02-05
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from etf_strategy.core.ic_calculator_numba import compute_spearman_ic_numba
from etf_strategy.gpu import gpu_available, compute_ic_batch_auto


def generate_test_data(n_factors: int, n_days: int, n_etfs: int):
    """生成测试数据"""
    np.random.seed(42)
    factors = np.random.randn(n_factors, n_days, n_etfs).astype(np.float64)
    returns = np.random.randn(n_days, n_etfs).astype(np.float64) * 0.02

    # 添加 NaN
    nan_mask = np.random.rand(n_factors, n_days, n_etfs) < 0.15
    factors[nan_mask] = np.nan

    return factors, returns


def main():
    parser = argparse.ArgumentParser(description="GPU/CPU 结果一致性验证")
    parser.add_argument("--n-factors", type=int, default=100, help="因子数量")
    parser.add_argument("--n-days", type=int, default=252, help="天数")
    parser.add_argument("--n-etfs", type=int, default=49, help="ETF 数量")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="容差")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("  GPU/CPU Alignment Verification")
    print("=" * 80)

    if not gpu_available():
        print("  ⚠️  GPU not available. Cannot verify alignment.")
        return

    # ── Step 1: 生成测试数据 ──
    print(f"\nGenerating test data ({args.n_factors} factors × {args.n_days} days × {args.n_etfs} ETFs)...")
    factors_3d, returns_2d = generate_test_data(args.n_factors, args.n_days, args.n_etfs)

    # ── Step 2: CPU 计算 ──
    print("\nCPU (Numba) calculation...")
    cpu_ics = []
    for i in range(args.n_factors):
        ic = compute_spearman_ic_numba(factors_3d[i], returns_2d)
        cpu_ics.append(ic)
    cpu_ics = np.array(cpu_ics)

    # ── Step 3: GPU 计算 ──
    print("GPU (CuPy) calculation...")
    gpu_results = compute_ic_batch_auto(factors_3d, returns_2d, use_gpu=True)
    gpu_ics = gpu_results["ic_mean"]

    # ── Step 4: 对比结果 ──
    print("\n" + "=" * 80)
    print("Alignment Check")
    print("=" * 80)

    # 绝对差异
    abs_diff = np.abs(cpu_ics - gpu_ics)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)

    # 相对差异
    rel_diff = np.abs((cpu_ics - gpu_ics) / (np.abs(cpu_ics) + 1e-10)) * 100
    max_rel_diff = np.max(rel_diff)
    mean_rel_diff = np.mean(rel_diff)

    print(f"\nAbsolute difference:")
    print(f"  Max:  {max_abs_diff:.2e}")
    print(f"  Mean: {mean_abs_diff:.2e}")
    print(f"  Tolerance: {args.tolerance:.2e}")

    print(f"\nRelative difference (%):")
    print(f"  Max:  {max_rel_diff:.4f}%")
    print(f"  Mean: {mean_rel_diff:.4f}%")

    # ── Step 5: 判断 ──
    aligned = max_abs_diff < args.tolerance and max_rel_diff < 0.1

    print("\n" + "=" * 80)
    if aligned:
        print("  ✓ PASS: GPU/CPU 结果一致 (差异在容差范围内)")
    else:
        print("  ✗ FAIL: GPU/CPU 结果不一致")
        print(f"\n  最大差异位置:")
        worst_idx = np.argmax(abs_diff)
        print(f"    Factor #{worst_idx}:")
        print(f"      CPU IC:  {cpu_ics[worst_idx]:+.6f}")
        print(f"      GPU IC:  {gpu_ics[worst_idx]:+.6f}")
        print(f"      Diff:    {abs_diff[worst_idx]:.2e}")
    print("=" * 80)

    # ── Step 6: 详细报告 ──
    if args.n_factors <= 20:
        print("\nDetailed comparison (first 20 factors):")
        print(f"{'#':>3s} {'CPU_IC':>10s} {'GPU_IC':>10s} {'AbsDiff':>10s} {'RelDiff(%)':>12s}")
        print("-" * 50)
        for i in range(min(args.n_factors, 20)):
            print(
                f"{i:>3d} {cpu_ics[i]:>+10.6f} {gpu_ics[i]:>+10.6f} "
                f"{abs_diff[i]:>10.2e} {rel_diff[i]:>12.4f}"
            )


if __name__ == "__main__":
    main()
