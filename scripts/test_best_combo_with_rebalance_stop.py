#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试最佳策略在不同止损模式下的表现
对比：
1. 无止损 (baseline)
2. 每日 10% 止损
3. 调仓日 10% 止损
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import shift_timing_signal

# 从 batch_vec_backtest 导入核心函数
from batch_vec_backtest import run_vec_backtest, calculate_atr


def main():
    print("=" * 80)
    print("🔬 最佳策略止损模式对比测试")
    print("=" * 80)
    
    # 最佳策略组合
    best_combo = "CORRELATION_TO_MARKET_20D + MAX_DD_60D + PRICE_POSITION_120D + PRICE_POSITION_20D"
    factor_names_in_combo = [
        "CORRELATION_TO_MARKET_20D",
        "MAX_DD_60D",
        "PRICE_POSITION_120D",
        "PRICE_POSITION_20D",
    ]
    
    print(f"\n✅ 测试策略: {best_combo}")
    print()
    
    # 1. 加载配置
    config_path = ROOT / "configs/combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    backtest_config = config.get("backtest", {})
    FREQ = backtest_config.get("freq")
    POS_SIZE = backtest_config.get("pos_size")
    LOOKBACK = backtest_config.get("lookback")
    INITIAL_CAPITAL = float(backtest_config.get("initial_capital"))
    COMMISSION_RATE = float(backtest_config.get("commission_rate"))
    
    print(f"回测参数: FREQ={FREQ}, POS_SIZE={POS_SIZE}, LOOKBACK={LOOKBACK}")
    
    # 2. 加载数据
    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )
    
    # 3. 计算因子
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)
    factor_names_list = raw_factors_df.columns.get_level_values(0).unique().tolist()
    raw_factors = {fname: raw_factors_df[fname] for fname in factor_names_list}
    
    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)
    
    first_factor = std_factors[factor_names_list[0]]
    dates = first_factor.index
    etf_codes = first_factor.columns.tolist()
    T, N = first_factor.shape
    
    # 只保留组合中的因子
    factors_3d = np.stack([std_factors[f].values for f in factor_names_in_combo], axis=-1)
    
    # 价格数据
    close_prices = ohlcv["close"][etf_codes].ffill().bfill().values
    open_prices = ohlcv["open"][etf_codes].ffill().bfill().values
    high_prices = ohlcv["high"][etf_codes].ffill().bfill().values
    low_prices = ohlcv["low"][etf_codes].ffill().bfill().values
    
    # 择时
    timing_config = config.get("backtest", {}).get("timing", {})
    timing_module = LightTimingModule(
        extreme_threshold=timing_config.get("extreme_threshold", -0.3),
        extreme_position=timing_config.get("extreme_position", 0.3),
    )
    timing_series = timing_module.compute_position_ratios(ohlcv["close"])
    timing_arr_raw = timing_series.reindex(dates).fillna(1.0).values
    timing_arr = shift_timing_signal(timing_arr_raw)
    
    # 因子索引（使用所有因子）
    factor_indices = list(range(len(factor_names_in_combo)))
    
    print(f"✅ 数据加载完成：{T} 天 × {N} 只 ETF × {len(factor_names_in_combo)} 个因子")
    print()
    
    # 4. 测试三种模式
    results = []
    
    # 模式 1: 无止损
    print("▶ 测试模式 1: 无止损")
    ret1, wr1, pf1, trades1, _, risk1 = run_vec_backtest(
        factors_3d, close_prices, open_prices, high_prices, low_prices,
        timing_arr, factor_indices,
        freq=FREQ, pos_size=POS_SIZE, initial_capital=INITIAL_CAPITAL,
        commission_rate=COMMISSION_RATE, lookback=LOOKBACK,
        trailing_stop_pct=0.0,  # 无止损
        stop_on_rebalance_only=False,  # 无关紧要
    )
    results.append({
        "模式": "无止损",
        "收益率": ret1 * 100,
        "Calmar": risk1["calmar_ratio"],
        "Sharpe": risk1["sharpe_ratio"],
        "交易次数": trades1,
    })
    print(f"  收益率: {ret1*100:.2f}%, Calmar: {risk1['calmar_ratio']:.3f}, 交易: {trades1}")
    
    # 模式 2: 每日 10% 止损
    print("▶ 测试模式 2: 每日 10% 止损")
    ret2, wr2, pf2, trades2, _, risk2 = run_vec_backtest(
        factors_3d, close_prices, open_prices, high_prices, low_prices,
        timing_arr, factor_indices,
        freq=FREQ, pos_size=POS_SIZE, initial_capital=INITIAL_CAPITAL,
        commission_rate=COMMISSION_RATE, lookback=LOOKBACK,
        trailing_stop_pct=0.10,  # 10% 止损
        stop_on_rebalance_only=False,  # 每日检查
    )
    results.append({
        "模式": "每日 10%",
        "收益率": ret2 * 100,
        "Calmar": risk2["calmar_ratio"],
        "Sharpe": risk2["sharpe_ratio"],
        "交易次数": trades2,
    })
    print(f"  收益率: {ret2*100:.2f}%, Calmar: {risk2['calmar_ratio']:.3f}, 交易: {trades2}")
    
    # 模式 3: 调仓日 10% 止损
    print("▶ 测试模式 3: 调仓日 10% 止损")
    ret3, wr3, pf3, trades3, _, risk3 = run_vec_backtest(
        factors_3d, close_prices, open_prices, high_prices, low_prices,
        timing_arr, factor_indices,
        freq=FREQ, pos_size=POS_SIZE, initial_capital=INITIAL_CAPITAL,
        commission_rate=COMMISSION_RATE, lookback=LOOKBACK,
        trailing_stop_pct=0.10,  # 10% 止损
        stop_on_rebalance_only=True,  # 仅调仓日检查 ✅
    )
    results.append({
        "模式": "调仓日 10%",
        "收益率": ret3 * 100,
        "Calmar": risk3["calmar_ratio"],
        "Sharpe": risk3["sharpe_ratio"],
        "交易次数": trades3,
    })
    print(f"  收益率: {ret3*100:.2f}%, Calmar: {risk3['calmar_ratio']:.3f}, 交易: {trades3}")
    
    # 5. 总结对比
    print()
    print("=" * 80)
    print("📊 对比总结")
    print("=" * 80)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print()
    
    # 计算改进幅度
    baseline_ret = ret1 * 100
    daily_ret = ret2 * 100
    rebal_ret = ret3 * 100
    
    print("💡 关键发现:")
    print(f"  1. 每日止损 vs 无止损:     {daily_ret - baseline_ret:+.2f}pp ({(daily_ret/baseline_ret - 1)*100:+.1f}%)")
    print(f"  2. 调仓日止损 vs 无止损:   {rebal_ret - baseline_ret:+.2f}pp ({(rebal_ret/baseline_ret - 1)*100:+.1f}%)")
    print(f"  3. 调仓日止损 vs 每日止损: {rebal_ret - daily_ret:+.2f}pp ({(rebal_ret/daily_ret - 1)*100:+.1f}%)")
    
    # 判断结论
    print()
    print("🎯 策略建议:")
    if rebal_ret > baseline_ret * 0.98:  # 调仓日止损保留 98% 以上收益
        print(f"  ✅ 推荐「调仓日 10% 止损」")
        print(f"     - 与无止损差距仅 {baseline_ret - rebal_ret:.2f}pp")
        print(f"     - 比每日止损多赚 {rebal_ret - daily_ret:.2f}pp")
        print(f"     - 保持策略节奏一致性")
    elif daily_ret < baseline_ret * 0.80:  # 每日止损严重损害收益
        print(f"  ⚠️  「每日止损」严重损害收益 ({daily_ret - baseline_ret:.2f}pp)")
        print(f"  ✅ 推荐「调仓日止损」或「无止损」")
    else:
        print(f"  📊 需要进一步评估风险收益比")


if __name__ == "__main__":
    main()
