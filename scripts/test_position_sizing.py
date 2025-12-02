#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试持仓数量 (POS_SIZE) 和择时参数对最大回撤的影响

对比维度：
1. 持仓数量：1/2/3
2. 择时强度：
   - 温和 (threshold=-0.3, position=0.3) 当前配置
   - 激进 (threshold=-0.2, position=0.2) 更早降仓
   - 极端 (threshold=-0.1, position=0.1) 最早降仓
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
    print("📉 降仓机制测试：持仓数量 + 择时强度")
    print("=" * 80)
    
    # 最佳策略
    factor_names_in_combo = [
        "CORRELATION_TO_MARKET_20D",
        "MAX_DD_60D",
        "PRICE_POSITION_120D",
        "PRICE_POSITION_20D",
    ]
    
    # 1. 加载配置
    config_path = ROOT / "configs/combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    backtest_config = config.get("backtest", {})
    FREQ = backtest_config.get("freq")
    LOOKBACK = backtest_config.get("lookback")
    INITIAL_CAPITAL = float(backtest_config.get("initial_capital"))
    COMMISSION_RATE = float(backtest_config.get("commission_rate"))
    
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
    
    factor_indices = list(range(len(factor_names_in_combo)))
    
    print(f"✅ 数据加载完成：{T} 天 × {N} 只 ETF")
    print()
    
    # 3. 测试矩阵：持仓数量 × 择时强度
    test_configs = []
    
    # 持仓数量
    pos_sizes = [1, 2, 3]
    
    # 择时参数
    timing_configs = [
        ("温和", -0.3, 0.3),
        ("激进", -0.2, 0.2),
        ("极端", -0.1, 0.1),
    ]
    
    # 生成测试组合
    for pos_size in pos_sizes:
        for timing_name, threshold, position in timing_configs:
            test_configs.append({
                "pos_size": pos_size,
                "timing_name": timing_name,
                "threshold": threshold,
                "position": position,
            })
    
    results = []
    
    print(f"开始测试 {len(test_configs)} 个配置...")
    for i, cfg in enumerate(test_configs, 1):
        pos_size = cfg["pos_size"]
        timing_name = cfg["timing_name"]
        threshold = cfg["threshold"]
        position = cfg["position"]
        
        config_name = f"POS={pos_size}, 择时={timing_name}"
        print(f"  [{i:2d}/{len(test_configs)}] {config_name:25s}...", end=" ", flush=True)
        
        # 生成对应的择时信号
        timing_module = LightTimingModule(
            extreme_threshold=threshold,
            extreme_position=position,
        )
        timing_series = timing_module.compute_position_ratios(ohlcv["close"])
        timing_arr_raw = timing_series.reindex(dates).fillna(1.0).values
        timing_arr = shift_timing_signal(timing_arr_raw)
        
        # 运行回测
        ret, wr, pf, trades, _, risk = run_vec_backtest(
            factors_3d, close_prices, open_prices, high_prices, low_prices,
            timing_arr, factor_indices,
            freq=FREQ, pos_size=pos_size, initial_capital=INITIAL_CAPITAL,
            commission_rate=COMMISSION_RATE, lookback=LOOKBACK,
            trailing_stop_pct=0.0,  # 不使用止损
            stop_on_rebalance_only=False,
        )
        
        results.append({
            "持仓数": pos_size,
            "择时": timing_name,
            "threshold": threshold,
            "position": position,
            "收益率": ret * 100,
            "最大回撤": risk["max_drawdown"] * 100,
            "Calmar": risk["calmar_ratio"],
            "Sharpe": risk["sharpe_ratio"],
            "交易次数": trades,
        })
        
        print(f"收益 {ret*100:6.2f}%, 回撤 {risk['max_drawdown']*100:6.2f}%, Calmar {risk['calmar_ratio']:.3f}")
    
    print()
    print("=" * 80)
    print("📊 测试结果对比")
    print("=" * 80)
    df = pd.DataFrame(results)
    
    # 按持仓数分组显示
    for pos_size in pos_sizes:
        print(f"\n▶ 持仓数 = {pos_size}")
        sub_df = df[df["持仓数"] == pos_size]
        for _, row in sub_df.iterrows():
            print(f"  {row['择时']:4s} (t={row['threshold']:.1f}, p={row['position']:.1f}) | "
                  f"收益: {row['收益率']:6.2f}% | 回撤: {row['最大回撤']:6.2f}% | "
                  f"Calmar: {row['Calmar']:5.3f} | 交易: {row['交易次数']:3.0f}")
    
    print()
    print("=" * 80)
    print("💡 分析与推荐")
    print("=" * 80)
    
    # 找出最佳配置
    best_calmar_idx = df['Calmar'].idxmax()
    best_calmar = df.iloc[best_calmar_idx]
    
    min_dd_idx = df['最大回撤'].idxmin()
    min_dd = df.iloc[min_dd_idx]
    
    # 基线（POS=3, 温和择时）
    baseline = df[(df["持仓数"] == 3) & (df["择时"] == "温和")].iloc[0]
    
    print(f"\n1. 基线（当前配置）:")
    print(f"   持仓数=3, 择时=温和 (threshold=-0.3)")
    print(f"   收益率: {baseline['收益率']:.2f}%")
    print(f"   最大回撤: {baseline['最大回撤']:.2f}% ← 不可接受")
    print(f"   Calmar: {baseline['Calmar']:.3f}")
    
    print(f"\n2. 最佳 Calmar:")
    print(f"   配置: 持仓数={best_calmar['持仓数']}, 择时={best_calmar['择时']}")
    print(f"   收益率: {best_calmar['收益率']:.2f}% (vs 基线 {best_calmar['收益率'] - baseline['收益率']:+.2f}pp)")
    print(f"   最大回撤: {best_calmar['最大回撤']:.2f}% (vs 基线 {best_calmar['最大回撤'] - baseline['最大回撤']:+.2f}pp)")
    print(f"   Calmar: {best_calmar['Calmar']:.3f} (vs 基线 {best_calmar['Calmar'] - baseline['Calmar']:+.3f})")
    
    print(f"\n3. 最小回撤:")
    print(f"   配置: 持仓数={min_dd['持仓数']}, 择时={min_dd['择时']}")
    print(f"   收益率: {min_dd['收益率']:.2f}%")
    print(f"   最大回撤: {min_dd['最大回撤']:.2f}% ← 风控最严")
    print(f"   Calmar: {min_dd['Calmar']:.3f}")
    
    # 推荐
    print(f"\n🎯 策略推荐:")
    
    # 找出回撤 < 20% 的配置
    acceptable = df[df['最大回撤'] < 20.0]
    if len(acceptable) > 0:
        best_in_acceptable = acceptable.loc[acceptable['Calmar'].idxmax()]
        print(f"  ✅ 推荐配置（回撤 < 20%，Calmar 最佳）:")
        print(f"     持仓数={best_in_acceptable['持仓数']}, 择时={best_in_acceptable['择时']}")
        print(f"     收益率: {best_in_acceptable['收益率']:.2f}%")
        print(f"     最大回撤: {best_in_acceptable['最大回撤']:.2f}% ← 可接受")
        print(f"     Calmar: {best_in_acceptable['Calmar']:.3f}")
    else:
        print(f"  ⚠️  所有配置的回撤都 >= 20%")
        print(f"  💡 回撤最小的配置:")
        print(f"     持仓数={min_dd['持仓数']}, 择时={min_dd['择时']}")
        print(f"     最大回撤: {min_dd['最大回撤']:.2f}%")
    
    # 保存结果
    output_dir = ROOT / "results" / "position_sizing_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "results.csv", index=False)
    print(f"\n✅ 结果已保存至: {output_dir}")


if __name__ == "__main__":
    main()
