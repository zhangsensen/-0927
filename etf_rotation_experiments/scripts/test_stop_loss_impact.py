#!/usr/bin/env python3
"""
V10 止损机制验证测试
===================
目的：
1. 验证止损逻辑正确性
2. 对比有/无止损的回测指标差异
3. 测试不同止损阈值的参数敏感性（防过拟合）

使用方法：
    cd etf_rotation_experiments
    python scripts/test_stop_loss_impact.py

输出：
    - 不同止损阈值下的关键指标对比表
    - 止损事件统计
"""

import os
import sys
from pathlib import Path

# 设置项目路径
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

os.environ["RB_DAILY_IC_PRECOMP"] = "1"
os.environ["RB_DAILY_IC_MEMMAP"] = "1"
os.environ["RB_STABLE_RANK"] = "1"

from core.data_loader import DataLoader
from core.precise_factor_library_v2 import PreciseFactorLibrary
from strategies.backtest.production_backtest import backtest_no_lookahead


def run_stop_loss_comparison():
    """运行止损对比测试"""
    print("=" * 80)
    print("V10 止损机制验证测试")
    print("=" * 80)

    # 加载数据
    print("\n[1/3] 加载数据...")
    data_dir = PROJECT_ROOT.parent / "raw" / "ETF" / "daily"
    
    loader = DataLoader(data_dir=str(data_dir))
    
    # 使用部分ETF进行测试（避免过长运行时间）
    test_symbols = [
        "159915", "510050", "510300", "510500", "512010",
        "512100", "512800", "512880", "513050", "513100",
    ]
    
    ohlcv_dict = loader.load_ohlcv(
        etf_codes=test_symbols,
        start_date="2020-01-01",
        end_date="2024-10-14",
    )
    
    if not ohlcv_dict or "close" not in ohlcv_dict:
        print("❌ 数据加载失败")
        return
    
    # close 是 DataFrame，列名是 ETF 代码
    price_df = ohlcv_dict["close"].sort_index()
    
    # 计算因子
    print("[2/3] 计算因子...")
    fl = PreciseFactorLibrary()
    
    # 计算所有因子 - 传入完整 ohlcv_dict
    factors_df = fl.compute_all_factors(ohlcv_dict)
    
    # 获取 returns (close-to-close)
    returns_df = price_df.pct_change().iloc[1:]  # 去掉第一行 NaN
    
    # 对齐日期
    common_dates = factors_df.index.intersection(returns_df.index)
    factors_df = factors_df.loc[common_dates]
    returns_df = returns_df.loc[common_dates]
    
    # 转换为 numpy 数组 (T, N, F) 格式
    factor_names = list(factors_df.columns.get_level_values(0).unique())
    etf_names = list(price_df.columns)
    
    T = len(common_dates)
    N = len(etf_names)
    F = len(factor_names)
    
    factors_data = np.zeros((T, N, F), dtype=np.float64)
    for f_idx, f_name in enumerate(factor_names):
        for n_idx, etf in enumerate(etf_names):
            if (f_name, etf) in factors_df.columns:
                factors_data[:, n_idx, f_idx] = factors_df[(f_name, etf)].values
    
    returns = returns_df[etf_names].values  # (T, N)
    
    print(f"   数据形状: factors={factors_data.shape}, returns={returns.shape}")
    print(f"   因子数量: {len(factor_names)}")
    print(f"   ETF数量: {len(etf_names)}")
    print(f"   可用因子: {factor_names}")
    
    # 选择一个测试组合（使用实际可用的因子）
    # 动量 + 波动率 + RSI
    test_combo = ["MOM_20D", "RET_VOL_20D", "RSI_14"]
    factor_indices = [factor_names.index(f) for f in test_combo if f in factor_names]
    
    if len(factor_indices) < 2:
        print(f"❌ 测试因子不足: 期望{test_combo}, 实际找到{len(factor_indices)}个")
        return
    
    print(f"   测试组合: {'+'.join(test_combo)}")
    
    # 测试参数
    rebalance_freq = 8  # 8天调仓
    position_size = 4   # Top 4
    lookback_window = 252
    
    # 不同止损阈值测试
    stop_loss_levels = [0.0, 0.03, 0.05, 0.07, 0.10]
    
    print("\n[3/3] 运行回测对比...")
    results = []
    
    for sl in stop_loss_levels:
        label = "无止损" if sl == 0 else f"止损{sl:.0%}"
        print(f"   测试: {label}...")
        
        result = backtest_no_lookahead(
            factors_data=factors_data[:, :, factor_indices],
            returns=returns,
            etf_names=etf_names,
            rebalance_freq=rebalance_freq,
            lookback_window=lookback_window,
            position_size=position_size,
            initial_capital=1_000_000.0,
            commission_rate=0.00005,
            factors_data_full=factors_data,
            factor_indices_for_cache=np.array(factor_indices, dtype=np.int64),
            etf_stop_loss=sl,
        )
        
        results.append({
            "止损阈值": label,
            "年化收益": result["annual_ret"],
            "最大回撤": result["max_dd"],
            "夏普比率": result["sharpe"],
            "胜率": result["win_rate"],
            "Calmar": result["calmar_ratio"],
            "止损次数": result.get("n_stop_loss", 0),
        })
    
    # 输出结果
    df = pd.DataFrame(results)
    df["年化收益"] = df["年化收益"].apply(lambda x: f"{x:.2%}")
    df["最大回撤"] = df["最大回撤"].apply(lambda x: f"{x:.2%}")
    df["夏普比率"] = df["夏普比率"].apply(lambda x: f"{x:.3f}")
    df["胜率"] = df["胜率"].apply(lambda x: f"{x:.1%}")
    df["Calmar"] = df["Calmar"].apply(lambda x: f"{x:.2f}")
    
    print("\n" + "=" * 80)
    print("止损机制对比结果")
    print("=" * 80)
    print(df.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("分析总结")
    print("=" * 80)
    
    # 分析最优阈值
    results_raw = []
    for sl in stop_loss_levels:
        result = backtest_no_lookahead(
            factors_data=factors_data[:, :, factor_indices],
            returns=returns,
            etf_names=etf_names,
            rebalance_freq=rebalance_freq,
            lookback_window=lookback_window,
            position_size=position_size,
            initial_capital=1_000_000.0,
            commission_rate=0.00005,
            factors_data_full=factors_data,
            factor_indices_for_cache=np.array(factor_indices, dtype=np.int64),
            etf_stop_loss=sl,
        )
        results_raw.append({
            "sl": sl,
            "sharpe": result["sharpe"],
            "win_rate": result["win_rate"],
            "max_dd": result["max_dd"],
        })
    
    baseline_sharpe = results_raw[0]["sharpe"]
    baseline_win_rate = results_raw[0]["win_rate"]
    baseline_dd = results_raw[0]["max_dd"]
    
    for r in results_raw[1:]:
        sl = r["sl"]
        sharpe_diff = r["sharpe"] - baseline_sharpe
        win_rate_diff = r["win_rate"] - baseline_win_rate
        dd_diff = r["max_dd"] - baseline_dd  # 负值变大说明回撤减少
        
        print(f"\n止损 {sl:.0%} vs 无止损:")
        print(f"  夏普变化: {sharpe_diff:+.3f}")
        print(f"  胜率变化: {win_rate_diff:+.1%}")
        print(f"  最大回撤变化: {dd_diff:+.2%} ({'改善' if dd_diff > 0 else '恶化'})")
    
    print("\n" + "=" * 80)
    print("🔍 防过拟合检验建议:")
    print("=" * 80)
    print("1. 若不同止损阈值的结果差异巨大，可能存在过拟合")
    print("2. 理想情况：3%-7% 范围内的止损阈值应表现相近")
    print("3. 止损次数过多(>年调仓次数的20%)说明阈值过紧")
    print("4. 建议使用 WFO 滚动窗口验证参数稳定性")
    

if __name__ == "__main__":
    run_stop_loss_comparison()
