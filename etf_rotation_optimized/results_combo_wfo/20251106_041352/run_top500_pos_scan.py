#!/usr/bin/env python
"""
Auto-generated Top 500 position size grid search script
Generated: 20251106_041352
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

import yaml
from joblib import Parallel, delayed

from core.cross_section_processor import CrossSectionProcessor
from core.data_loader import DataLoader
from core.factor_library import PreciseFactorLibrary
from test_freq_no_lookahead import backtest_no_lookahead


def load_and_prepare_data():
    """加载数据"""
    print("📥 加载数据...")

    with open("configs/combo_wfo_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
        use_cache=True,
    )

    # 计算因子
    factor_lib = PreciseFactorLibrary()
    factors_df = factor_lib.compute_all_factors(prices=ohlcv)
    factors_dict = {name: factors_df[name] for name in factor_lib.list_factors()}

    # 横截面标准化
    processor = CrossSectionProcessor(
        lower_percentile=config["cross_section"]["winsorize_lower"] * 100,
        upper_percentile=config["cross_section"]["winsorize_upper"] * 100,
        verbose=False,
    )
    standardized_factors = processor.process_all_factors(factors_dict)

    # 组织数据
    factor_names = sorted(standardized_factors.keys())
    factor_arrays = [standardized_factors[name].values for name in factor_names]
    factors_data = np.stack(factor_arrays, axis=-1)

    returns_df = ohlcv["close"].pct_change(fill_method=None)
    returns = returns_df.values
    etf_names = list(ohlcv["close"].columns)

    return config, factors_data, returns, etf_names


def run_top500_grid_search():
    """执行Top 500网格搜索"""

    print("\n" + "=" * 80)
    print("🚀 Top 500 持仓数网格搜索")
    print("=" * 80)
    print()

    config, factors_data, returns, etf_names = load_and_prepare_data()

    # 读取网格配置
    grid_file = (
        "results_combo_wfo/20251106_041352/top500_pos_grid_tasks_20251106_041352.csv"
    )
    grid_df = pd.read_csv(grid_file)

    print(f"📋 加载了 {len(grid_df)} 个任务")
    print()

    def run_task(row):
        """运行单个任务"""
        try:
            result = backtest_no_lookahead(
                factors_data=factors_data,
                returns=returns,
                etf_names=etf_names,
                rebalance_freq=int(row["test_freq"]),
                lookback_window=config["backtest"]["lookback_window"],
                position_size=int(row["position_size"]),
                transaction_cost=config["backtest"]["transaction_cost"],
                initial_capital=config["backtest"]["initial_capital"],
            )

            result["combo"] = row["combo"]
            result["wfo_freq"] = row["wfo_freq"]
            result["test_freq"] = row["test_freq"]
            result["position_size"] = int(row["position_size"])
            result["test_position_size"] = int(row["position_size"])

            return result
        except Exception as e:
            print(f"❌ 错误: {str(e)[:50]}")
            return None

    # 并行执行
    print("⚙️  开始执行任务...")
    results = Parallel(
        n_jobs=config["backtest"]["max_workers"],
        backend="loky",
        verbose=1,
    )(delayed(run_task)(row) for idx, row in grid_df.iterrows())

    # 处理结果
    valid_results = [r for r in results if r is not None]
    print(f"\n✅ 成功完成 {len(valid_results)}/{len(grid_df)} 个任务")

    # 保存结果
    df_results = pd.DataFrame(valid_results)

    timestamp = "20251106_041352"
    output_dir = Path("results_combo_wfo") / timestamp
    output_file = output_dir / f"top500_pos_scan_{timestamp}.csv"
    df_results.to_csv(output_file, index=False)

    print(f"💾 结果已保存: {output_file}")
    print()

    # 分析结果
    print("📊 分析结果...")
    stats = (
        df_results.groupby("test_position_size")
        .agg(
            {
                "sharpe": ["mean", "std"],
                "annual_ret": ["mean", "std"],
                "max_dd": "mean",
                "win_rate": "mean",
            }
        )
        .round(4)
    )

    print(stats)
    print()

    # 找到最优持仓数
    optimal_pos = df_results.groupby("test_position_size")["sharpe"].mean().idxmax()
    print(f"🎯 最优持仓数: {optimal_pos}")
    print()

    return df_results


if __name__ == "__main__":
    df_results = run_top500_grid_search()
