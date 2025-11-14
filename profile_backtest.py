#!/usr/bin/env python
"""
真实性能剖析脚本：逐行定位 backtest_no_lookahead 瓶颈
不做猜测，只看数据。
"""
import numpy as np
import sys
import time
import cProfile
import pstats
import io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from etf_rotation_optimized.real_backtest.run_production_backtest import backtest_no_lookahead
from etf_rotation_optimized.core.data_loader import DataLoader
from etf_rotation_optimized.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_rotation_optimized.core.cross_section_processor import CrossSectionProcessor

def profile_full_pipeline():
    """完整管道分析：数据加载→因子计算→回测"""
    print("="*80)
    print("完整管道性能剖析")
    print("="*80)
    
    # 1) 数据加载
    t0 = time.perf_counter()
    loader = DataLoader(data_dir="etf_download_manager/data", cache_dir="cache")
    # 小样本：前10只ETF，500天
    from datetime import datetime, timedelta
    end = datetime(2024, 10, 14)
    start = end - timedelta(days=730)
    ohlcv = loader.load_ohlcv(
        etf_codes=['159915', '159949', '510050', '510300', '510500', '512000', '512690', '512880', '513050', '515000'],
        start_date=start.strftime('%Y-%m-%d'),
        end_date=end.strftime('%Y-%m-%d'),
        use_cache=True
    )
    t1 = time.perf_counter()
    print(f"数据加载耗时: {(t1-t0)*1000:.1f} ms")
    
    # 2) 因子计算
    t0 = time.perf_counter()
    factor_lib = PreciseFactorLibrary()
    factors_df = factor_lib.compute_all_factors(prices=ohlcv)
    factors_dict = {name: factors_df[name] for name in factor_lib.list_factors()}
    t1 = time.perf_counter()
    print(f"因子计算耗时: {(t1-t0)*1000:.1f} ms")
    
    # 3) 横截面标准化
    t0 = time.perf_counter()
    processor = CrossSectionProcessor(lower_percentile=1.0, upper_percentile=99.0, verbose=False)
    standardized_factors = processor.process_all_factors(factors_dict)
    factor_names = sorted(standardized_factors.keys())
    factor_arrays = [standardized_factors[name].values for name in factor_names]
    factors_data = np.stack(factor_arrays, axis=-1)
    returns_df = ohlcv["close"].pct_change(fill_method=None)
    returns = returns_df.values
    etf_names = list(ohlcv["close"].columns)
    t1 = time.perf_counter()
    print(f"标准化与组织耗时: {(t1-t0)*1000:.1f} ms")
    
    T, N, F = factors_data.shape
    print(f"\n数据维度: T={T}, N={N}, F={F}")
    
    # 4) 回测（含cProfile）
    print("\n--- 开始回测（cProfile捕获）---")
    pr = cProfile.Profile()
    pr.enable()
    t0 = time.perf_counter()
    res = backtest_no_lookahead(
        factors_data, returns, etf_names,
        rebalance_freq=8, lookback_window=60, position_size=3, commission_rate=0.00005
    )
    t1 = time.perf_counter()
    pr.disable()
    print(f"回测耗时: {(t1-t0)*1000:.1f} ms (Sharpe={res['sharpe']:.3f})")
    
    # 输出Top函数
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    print("\n--- Top 30 函数 (按累计时间) ---")
    print(s.getvalue())

def profile_backtest_micro():
    """微观剖析：手动计时回测内部关键段"""
    print("\n" + "="*80)
    print("回测内部关键段计时（手动埋点）")
    print("="*80)
    
    # 生成合成数据
    T, N, F = 500, 20, 10
    np.random.seed(42)
    factors_data = np.random.randn(T, N, F)
    returns = np.random.randn(T, N) * 0.001
    etf_names = [f'ETF{i}' for i in range(N)]
    
    # 单次回测并在函数外计时各阶段（需插桩或读profile_data）
    import os
    os.environ['RB_PROFILE_BACKTEST'] = '1'  # 开启内部分段计时
    os.environ['RB_DAILY_IC_PRECOMP'] = '1'
    os.environ['RB_DAILY_IC_MEMMAP'] = '0'  # 避免memmap IO干扰
    
    t0 = time.perf_counter()
    res = backtest_no_lookahead(
        factors_data, returns, etf_names,
        rebalance_freq=10, lookback_window=60, position_size=5, commission_rate=0.0
    )
    t_total = time.perf_counter() - t0
    
    profile_data = res.get('profile', {})
    if profile_data:
        print(f"\n总耗时: {t_total*1000:.1f} ms")
        print(f"  - IC预计算: {profile_data.get('time_precompute_ic', 0)*1000:.1f} ms ({profile_data.get('time_precompute_ic', 0)/t_total*100:.1f}%)")
        print(f"  - 主循环: {profile_data.get('time_main_loop', 0)*1000:.1f} ms ({profile_data.get('time_main_loop', 0)/t_total*100:.1f}%)")
        print(f"  - 调仓次数: {profile_data.get('rebalance_executed', 0)}")
        print(f"  - 循环迭代: {profile_data.get('loop_iterations', 0)}")
        print(f"  - IC路径: {profile_data.get('ic_path', 'unknown')}")
    else:
        print("未获取到profile数据，请确认RB_PROFILE_BACKTEST=1")

def profile_ic_compute_only():
    """单独测试IC权重计算路径"""
    print("\n" + "="*80)
    print("IC权重计算路径性能对比")
    print("="*80)
    
    from etf_rotation_optimized.real_backtest.run_production_backtest import get_ic_weights_matrix_cached, IC_CACHE, PRECOMP_DAILY_IC
    from etf_rotation_optimized.core.ic_calculator_numba import compute_spearman_ic_numba
    
    T, N, F = 500, 30, 10
    np.random.seed(7)
    factors_data = np.random.randn(T, N, F)
    returns = np.random.randn(T, N) * 0.001
    lookback = 60
    rebalance_indices = np.arange(lookback+1, T, 10, dtype=np.int32)
    factor_indices = np.arange(F, dtype=np.int64)
    
    import os
    # 路径1：禁用日级预计算
    os.environ['RB_DAILY_IC_PRECOMP'] = '0'
    os.environ['RB_DAILY_IC_MEMMAP'] = '0'
    t0 = time.perf_counter()
    w1 = get_ic_weights_matrix_cached(factors_data, returns, rebalance_indices, lookback, factor_indices)
    t1 = time.perf_counter()
    print(f"[fallback窗口重算] {(t1-t0)*1000:.1f} ms")
    
    # 路径2：启用日级预计算（内存）
    IC_CACHE.clear()
    PRECOMP_DAILY_IC.clear()
    os.environ['RB_DAILY_IC_PRECOMP'] = '1'
    os.environ['RB_DAILY_IC_MEMMAP'] = '0'
    t0 = time.perf_counter()
    w2 = get_ic_weights_matrix_cached(factors_data, returns, rebalance_indices, lookback, factor_indices)
    t1 = time.perf_counter()
    print(f"[日级IC预计算+内存] {(t1-t0)*1000:.1f} ms (首次)")
    
    # 重复调用（缓存命中）
    t0 = time.perf_counter()
    w3 = get_ic_weights_matrix_cached(factors_data, returns, rebalance_indices, lookback, factor_indices)
    t1 = time.perf_counter()
    print(f"[日级IC预计算+缓存] {(t1-t0)*1000:.1f} ms (命中)")
    
    print(f"权重一致性: {np.allclose(w1, w2, atol=1e-6)}")

if __name__ == '__main__':
    profile_backtest_micro()
    print("\n")
    profile_ic_compute_only()
    # profile_full_pipeline()  # 可选：完整管道（需要真实数据）
