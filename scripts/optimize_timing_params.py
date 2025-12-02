#!/usr/bin/env python3
"""
择时参数网格搜索：寻找低回撤、高卡玛的最优参数组合。

目标函数：
- 筛选条件：total_return >= 100%
- 排序依据：calmar_ratio 降序

参数空间：
- extreme_threshold: [-0.6, -0.5, -0.4, -0.3, -0.2]
- extreme_position: [0.0, 0.1, 0.2, 0.3, 0.5]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

import yaml
import pandas as pd
import numpy as np
from datetime import datetime

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import shift_timing_signal

# 添加 scripts 目录到路径以便导入
sys.path.insert(0, str(ROOT / "scripts"))
from batch_vec_backtest import run_vec_backtest, LOOKBACK

# 参数空间
THRESHOLDS = [-0.6, -0.5, -0.4, -0.3, -0.2]
POSITIONS = [0.0, 0.1, 0.2, 0.3, 0.5]

# 最佳策略组合（Top 1）
BEST_COMBO = "CORRELATION_TO_MARKET_20D + MAX_DD_60D + PRICE_POSITION_120D + PRICE_POSITION_20D"


def main():
    print("=" * 80)
    print("择时参数网格搜索")
    print("=" * 80)
    print(f"参数空间: {len(THRESHOLDS)} x {len(POSITIONS)} = {len(THRESHOLDS) * len(POSITIONS)} 组")
    print(f"目标组合: {BEST_COMBO}")
    print()

    # 1. 加载数据
    config_path = ROOT / "configs/combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )

    # 2. 计算因子
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)

    factor_names_list = raw_factors_df.columns.get_level_values(0).unique().tolist()
    raw_factors = {fname: raw_factors_df[fname] for fname in factor_names_list}

    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)

    factor_names = sorted(std_factors.keys())
    first_factor = std_factors[factor_names[0]]
    dates = first_factor.index
    etf_codes = first_factor.columns.tolist()
    T, N = first_factor.shape

    factors_3d = np.stack([std_factors[f].values for f in factor_names], axis=-1)
    close_prices = ohlcv["close"][etf_codes].ffill().bfill().values
    open_prices = ohlcv["open"][etf_codes].ffill().bfill().values

    # 准备因子索引
    factor_index_map = {name: idx for idx, name in enumerate(factor_names)}
    factor_indices = [factor_index_map[f.strip()] for f in BEST_COMBO.split(" + ")]

    print(f"数据加载完成: {T} 天 x {N} 只 ETF x {len(factor_names)} 个因子")
    print()

    # 3. 网格搜索
    results = []
    print("开始网格搜索...")
    print("-" * 80)

    for threshold in THRESHOLDS:
        for position in POSITIONS:
            # 实例化择时模块
            timing_module = LightTimingModule(
                extreme_threshold=threshold,
                extreme_position=position,
            )
            timing_series = timing_module.compute_position_ratios(ohlcv["close"])
            timing_arr_raw = timing_series.reindex(dates).fillna(1.0).values
            timing_arr = shift_timing_signal(timing_arr_raw)

            # 运行回测
            ret, wr, pf, trades, rounding, risk = run_vec_backtest(
                factors_3d, close_prices, open_prices, timing_arr, factor_indices
            )

            results.append({
                "threshold": threshold,
                "position": position,
                "total_return": ret,
                "max_drawdown": risk["max_drawdown"],
                "annual_return": risk["annual_return"],
                "annual_volatility": risk["annual_volatility"],
                "sharpe_ratio": risk["sharpe_ratio"],
                "calmar_ratio": risk["calmar_ratio"],
                "win_rate": wr,
                "trades": trades,
            })

            # 实时输出
            status = "OK" if ret >= 1.0 else "SKIP"
            print(f"  threshold={threshold:+.1f}, position={position:.1f} -> "
                  f"Return={ret*100:6.1f}%, MaxDD={risk['max_drawdown']*100:5.1f}%, "
                  f"Calmar={risk['calmar_ratio']:5.2f} [{status}]")

    print("-" * 80)
    print()

    # 4. 结果分析
    df_results = pd.DataFrame(results)
    
    # 筛选满足收益约束的参数
    df_valid = df_results[df_results["total_return"] >= 1.0].copy()
    
    if len(df_valid) == 0:
        print("警告: 没有参数组合满足 Return >= 100% 的约束")
        print("显示所有结果（按 Calmar 降序）:")
        df_sorted = df_results.sort_values("calmar_ratio", ascending=False)
    else:
        print(f"满足 Return >= 100% 约束的参数组合: {len(df_valid)} / {len(df_results)}")
        df_sorted = df_valid.sort_values("calmar_ratio", ascending=False)

    print()
    print("=" * 80)
    print("结果排名 (按 Calmar Ratio 降序)")
    print("=" * 80)
    print()
    
    # 打印前 10 名
    top_n = min(10, len(df_sorted))
    for i, (_, row) in enumerate(df_sorted.head(top_n).iterrows()):
        marker = " <-- RECOMMENDED" if i == 0 else ""
        print(f"#{i+1}: threshold={row['threshold']:+.1f}, position={row['position']:.1f}")
        print(f"     Return={row['total_return']*100:.2f}%, MaxDD={row['max_drawdown']*100:.2f}%, "
              f"Calmar={row['calmar_ratio']:.3f}, Sharpe={row['sharpe_ratio']:.3f}{marker}")
        print()

    # 5. 最优参数推荐
    best = df_sorted.iloc[0]
    print("=" * 80)
    print("推荐配置 (combo_wfo_config.yaml)")
    print("=" * 80)
    print(f"""
backtest:
  timing:
    enabled: true
    type: "light_timing"
    extreme_threshold: {best['threshold']}
    extreme_position: {best['position']}
""")

    # 与基准对比
    baseline = df_results[(df_results["threshold"] == -0.4) & (df_results["position"] == 0.3)]
    if len(baseline) > 0:
        baseline = baseline.iloc[0]
        print()
        print("=" * 80)
        print("与当前默认参数 (threshold=-0.4, position=0.3) 对比")
        print("=" * 80)
        print(f"{'指标':<20} {'基准':>12} {'推荐':>12} {'变化':>12}")
        print("-" * 60)
        print(f"{'Total Return':<20} {baseline['total_return']*100:>11.2f}% {best['total_return']*100:>11.2f}% {(best['total_return']-baseline['total_return'])*100:>+11.2f}%")
        print(f"{'Max Drawdown':<20} {baseline['max_drawdown']*100:>11.2f}% {best['max_drawdown']*100:>11.2f}% {(best['max_drawdown']-baseline['max_drawdown'])*100:>+11.2f}%")
        print(f"{'Calmar Ratio':<20} {baseline['calmar_ratio']:>12.3f} {best['calmar_ratio']:>12.3f} {best['calmar_ratio']-baseline['calmar_ratio']:>+12.3f}")
        print(f"{'Sharpe Ratio':<20} {baseline['sharpe_ratio']:>12.3f} {best['sharpe_ratio']:>12.3f} {best['sharpe_ratio']-baseline['sharpe_ratio']:>+12.3f}")

    # 6. 保存完整结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = ROOT / "results" / f"timing_grid_search_{timestamp}.csv"
    df_results.to_csv(output_path, index=False)
    print()
    print(f"完整结果已保存: {output_path}")


if __name__ == "__main__":
    main()
