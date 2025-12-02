#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试熔断机制对最大回撤的控制效果

对比配置：
1. 基线：无止损 + 无熔断
2. 单日熔断：5% 单日跌幅清仓
3. 总回撤熔断：15% 累计回撤清仓
4. 双重熔断：单日 5% + 总回撤 15%
5. 完整风控：调仓日止损 10% + 双重熔断
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
from batch_vec_backtest import run_vec_backtest


def main():
    print("=" * 80)
    print("🔥 熔断机制测试：控制最大回撤")
    print("=" * 80)
    
    # 最佳策略
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
    
    # 2. 加载数据（简化代码，使用缓存）
    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )
    
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
    
    factors_3d = np.stack([std_factors[f].values for f in factor_names_in_combo], axis=-1)
    
    close_prices = ohlcv["close"][etf_codes].ffill().bfill().values
    open_prices = ohlcv["open"][etf_codes].ffill().bfill().values
    high_prices = ohlcv["high"][etf_codes].ffill().bfill().values
    low_prices = ohlcv["low"][etf_codes].ffill().bfill().values
    
    timing_config = config.get("backtest", {}).get("timing", {})
    timing_module = LightTimingModule(
        extreme_threshold=timing_config.get("extreme_threshold", -0.3),
        extreme_position=timing_config.get("extreme_position", 0.3),
    )
    timing_series = timing_module.compute_position_ratios(ohlcv["close"])
    timing_arr_raw = timing_series.reindex(dates).fillna(1.0).values
    timing_arr = shift_timing_signal(timing_arr_raw)
    
    factor_indices = list(range(len(factor_names_in_combo)))
    
    print(f"✅ 数据加载完成：{T} 天 × {N} 只 ETF")
    print()
    
    # 3. 测试配置矩阵
    test_configs = [
        # (名称, 止损%, 仅调仓日, 单日熔断%, 总回撤熔断%, 恢复天数)
        ("基线（无风控）", 0.0, False, 0.0, 0.0, 5),
        ("单日熔断 5%", 0.0, False, 0.05, 0.0, 5),
        ("总回撤熔断 15%", 0.0, False, 0.0, 0.15, 5),
        ("总回撤熔断 20%", 0.0, False, 0.0, 0.20, 5),
        ("双重熔断 (5% + 15%)", 0.0, False, 0.05, 0.15, 5),
        ("双重熔断 (5% + 20%)", 0.0, False, 0.05, 0.20, 5),
        ("调仓日止损 10%", 0.10, True, 0.0, 0.0, 5),
        ("止损 + 双熔断 (15%)", 0.10, True, 0.05, 0.15, 5),
        ("止损 + 双熔断 (20%)", 0.10, True, 0.05, 0.20, 5),
    ]
    
    results = []
    
    print("开始测试...")
    for i, (name, stop_pct, rebal_only, cb_day, cb_total, cb_recovery) in enumerate(test_configs, 1):
        print(f"  [{i}/{len(test_configs)}] {name}...", end=" ", flush=True)
        
        ret, wr, pf, trades, _, risk = run_vec_backtest(
            factors_3d, close_prices, open_prices, high_prices, low_prices,
            timing_arr, factor_indices,
            freq=FREQ, pos_size=POS_SIZE, initial_capital=INITIAL_CAPITAL,
            commission_rate=COMMISSION_RATE, lookback=LOOKBACK,
            trailing_stop_pct=stop_pct,
            stop_on_rebalance_only=rebal_only,
            circuit_breaker_day=cb_day,
            circuit_breaker_total=cb_total,
            circuit_recovery_days=cb_recovery,
        )
        
        results.append({
            "配置": name,
            "收益率": ret * 100,
            "最大回撤": risk["max_drawdown"] * 100,
            "Calmar": risk["calmar_ratio"],
            "Sharpe": risk["sharpe_ratio"],
            "交易次数": trades,
        })
        
        print(f"收益 {ret*100:6.2f}%, 回撤 {risk['max_drawdown']*100:6.2f}%")
    
    print()
    print("=" * 80)
    print("📊 测试结果对比")
    print("=" * 80)
    df = pd.DataFrame(results)
    
    # 格式化输出
    for i, row in df.iterrows():
        print(f"{row['配置']:20s} | 收益: {row['收益率']:6.2f}% | 回撤: {row['最大回撤']:6.2f}% | "
              f"Calmar: {row['Calmar']:5.3f} | 交易: {row['交易次数']:3.0f}")
    
    print()
    print("=" * 80)
    print("💡 分析与建议")
    print("=" * 80)
    
    # 找出最佳配置（Calmar 最高）
    best_idx = df['Calmar'].idxmax()
    best_row = df.iloc[best_idx]
    
    # 找出回撤最小的配置
    min_dd_idx = df['最大回撤'].idxmin()
    min_dd_row = df.iloc[min_dd_idx]
    
    # 基线
    baseline = df.iloc[0]
    
    print(f"\n1. 基线表现:")
    print(f"   收益率: {baseline['收益率']:.2f}%")
    print(f"   最大回撤: {baseline['最大回撤']:.2f}% ← 不可接受")
    print(f"   Calmar: {baseline['Calmar']:.3f}")
    
    print(f"\n2. 最佳 Calmar（风险调整收益）:")
    print(f"   配置: {best_row['配置']}")
    print(f"   收益率: {best_row['收益率']:.2f}% (vs 基线 {best_row['收益率'] - baseline['收益率']:+.2f}pp)")
    print(f"   最大回撤: {best_row['最大回撤']:.2f}% (vs 基线 {best_row['最大回撤'] - baseline['最大回撤']:+.2f}pp)")
    print(f"   Calmar: {best_row['Calmar']:.3f} (vs 基线 {best_row['Calmar'] - baseline['Calmar']:+.3f})")
    
    print(f"\n3. 最小回撤:")
    print(f"   配置: {min_dd_row['配置']}")
    print(f"   收益率: {min_dd_row['收益率']:.2f}%")
    print(f"   最大回撤: {min_dd_row['最大回撤']:.2f}% ← 风险控制最严")
    print(f"   Calmar: {min_dd_row['Calmar']:.3f}")
    
    # 推荐
    print(f"\n🎯 策略推荐:")
    if best_row['最大回撤'] < 20.0:
        print(f"  ✅ 推荐配置：{best_row['配置']}")
        print(f"     - 收益率: {best_row['收益率']:.2f}%")
        print(f"     - 最大回撤: {best_row['最大回撤']:.2f}% (可接受)")
        print(f"     - Calmar: {best_row['Calmar']:.3f} (风险调整收益最佳)")
    elif min_dd_row['最大回撤'] < 15.0:
        print(f"  ⚠️  最佳 Calmar 配置回撤仍超 20%")
        print(f"  ✅ 如需严格风控，推荐：{min_dd_row['配置']}")
        print(f"     - 回撤: {min_dd_row['最大回撤']:.2f}% (最小)")
        print(f"     - 收益率: {min_dd_row['收益率']:.2f}% (可能偏低)")
    else:
        print(f"  ⚠️  所有配置的回撤都较大")
        print(f"  💡 建议：")
        print(f"     1. 降低仓位水平（如 POS_SIZE=2）")
        print(f"     2. 使用更激进的择时（extreme_position=0.2）")
        print(f"     3. 考虑动态杠杆（target_vol=0.10）")
    
    # 保存结果
    output_dir = ROOT / "results" / "circuit_breaker_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "results.csv", index=False)
    print(f"\n✅ 结果已保存至: {output_dir}")


if __name__ == "__main__":
    main()
