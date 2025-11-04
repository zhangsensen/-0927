#!/usr/bin/env python3
"""
Numba JIT性能对比测试
直接对比JIT vs Python实现的实际性能

测试场景: 真实WFO数据，10K策略
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_jit_vs_python():
    """模拟真实WFO数据，对比JIT和Python性能"""

    # 模拟真实数据维度
    T, N = 1028, 50  # WFO时间跨度和股票数
    top_n = 5
    n_strategies = 10000

    print("=" * 70)
    print("🚀 Numba JIT vs Python性能对比测试")
    print("=" * 70)
    print(f"\n数据维度:")
    print(f"  时间: {T} 天")
    print(f"  股票: {N} 只")
    print(f"  TopN: {top_n}")
    print(f"  策略数: {n_strategies}")
    print(f"  总循环迭代: {T * n_strategies:,} 次")

    # 生成模拟数据
    np.random.seed(42)
    returns = np.random.randn(T, N) * 0.01

    # 预生成策略信号
    print(f"\n准备测试数据...")
    all_signals = []
    for _ in range(n_strategies):
        signals = np.random.randn(T, N)
        sig_shifted = np.roll(signals, 1, axis=0)
        sig_shifted[0] = np.nan
        all_signals.append(sig_shifted)

    valid_masks = [~(np.isnan(sig) | np.isnan(returns)) for sig in all_signals]
    print(f"✅ 准备完成")

    # ========== JIT版本测试 ==========
    print(f"\n" + "-" * 70)
    print("【JIT版本测试】")
    print("-" * 70)

    from core.wfo_multi_strategy_selector import NUMBA_AVAILABLE, _topn_core_jit

    if not NUMBA_AVAILABLE:
        print("❌ Numba不可用，跳过JIT测试")
        return

    # 首次编译
    print("  编译中...")
    t0 = time.time()
    _, _ = _topn_core_jit(all_signals[0], returns, valid_masks[0], top_n)
    compile_time = time.time() - t0
    print(f"  编译耗时: {compile_time:.3f}s")

    # 批量测试
    print(f"  运行 {n_strategies} 策略...")
    t0 = time.time()
    for i in range(n_strategies):
        _, _ = _topn_core_jit(all_signals[i], returns, valid_masks[i], top_n)
    jit_time = time.time() - t0

    jit_throughput = n_strategies / jit_time

    print(f"\n  结果:")
    print(f"    总耗时: {jit_time:.2f}s")
    print(f"    吞吐量: {jit_throughput:.1f} strategies/sec")
    print(f"    预估120K: {120000/jit_throughput/60:.1f} 分钟")

    # ========== Python版本测试（参考） ==========
    print(f"\n" + "-" * 70)
    print("【Python参考版本测试】（仅测试100个样本）")
    print("-" * 70)

    def python_version(sig_shifted, returns, valid_mask, top_n):
        """纯Python实现（不使用JIT）"""
        T, N = returns.shape
        daily_ret = np.zeros(T, dtype=float)
        daily_to = np.zeros(T, dtype=float)
        prev_hold_set = None

        for t in range(1, T):
            mask_t = valid_mask[t]
            if not np.any(mask_t):
                daily_to[t] = 0.0 if prev_hold_set is None else 1.0
                prev_hold_set = None
                continue

            valid_sig = sig_shifted[t][mask_t]
            valid_ret = returns[t][mask_t]
            valid_idx = np.where(mask_t)[0]

            n_valid = len(valid_idx)
            k = min(top_n, n_valid)

            if k == 0:
                daily_to[t] = 0.0 if prev_hold_set is None else 1.0
                prev_hold_set = None
                continue

            topk_local = np.argsort(valid_sig)[::-1][:k]
            topk = valid_idx[topk_local]
            daily_ret[t] = float(np.mean(valid_ret[topk_local]))

            if prev_hold_set is None:
                daily_to[t] = 1.0
            else:
                topk_set = set(topk)
                inter_count = len(prev_hold_set & topk_set)
                daily_to[t] = float(1.0 - inter_count / max(1, top_n))

            prev_hold_set = set(topk)

        return daily_ret, daily_to

    # 小样本测试（Python太慢）
    sample_size = 100
    print(f"  运行 {sample_size} 策略（样本）...")
    t0 = time.time()
    for i in range(sample_size):
        _, _ = python_version(all_signals[i], returns, valid_masks[i], top_n)
    py_time = time.time() - t0

    py_throughput = sample_size / py_time
    estimated_full = n_strategies / py_throughput

    print(f"\n  结果:")
    print(f"    样本耗时: {py_time:.2f}s")
    print(f"    吞吐量: {py_throughput:.1f} strategies/sec")
    print(
        f"    预估{n_strategies}耗时: {estimated_full:.1f}s = {estimated_full/60:.1f}分钟"
    )
    print(f"    预估120K: {120000/py_throughput/60:.1f} 分钟")

    # ========== 性能对比 ==========
    print(f"\n" + "=" * 70)
    print("【性能对比总结】")
    print("=" * 70)

    speedup = (
        py_throughput / jit_throughput
        if py_throughput < jit_throughput
        else jit_throughput / py_throughput
    )
    faster = "Python" if py_throughput > jit_throughput else "JIT"

    print(f"\nJIT版本:")
    print(f"  吞吐量: {jit_throughput:.1f} strategies/sec")
    print(f"  120K预估: {120000/jit_throughput/60:.1f} 分钟")

    print(f"\nPython版本:")
    print(f"  吞吐量: {py_throughput:.1f} strategies/sec")
    print(f"  120K预估: {120000/py_throughput/60:.1f} 分钟")

    print(f"\n加速比:")
    if faster == "JIT":
        print(f"  🚀 JIT比Python快 {jit_throughput/py_throughput:.2f}x")
    else:
        print(f"  ⚠️ Python比JIT快 {py_throughput/jit_throughput:.2f}x（异常）")

    print(f"\nvs 历史基线 (367 strategies/sec):")
    print(f"  JIT: {jit_throughput/367:.2f}x")
    print(f"  Python: {py_throughput/367:.2f}x")

    print(f"\n" + "=" * 70)

    if jit_throughput > py_throughput:
        print("✅ Numba JIT优化成功！")
    else:
        print("⚠️ 性能异常，需要进一步调查")


if __name__ == "__main__":
    test_jit_vs_python()
