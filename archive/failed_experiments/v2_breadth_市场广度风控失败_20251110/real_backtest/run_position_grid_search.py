"""
Top 500 收益参数的持仓数网格搜索

设计理念：
1. 提取 all_freq_scan 中收益最高的 500 个参数配置
2. 只测试持仓数 1-10 的变化
3. 大幅降低计算量（从 30000+ 降到 5000）
4. 找到最优的持仓数配置
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# 可选：tqdm 进度条（建议安装 tqdm 与 tqdm-joblib）
try:
    from tqdm.auto import tqdm  # type: ignore
    try:
        from tqdm_joblib import tqdm_joblib  # type: ignore
    except Exception:  # pragma: no cover
        tqdm_joblib = None  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore
    tqdm_joblib = None  # type: ignore
from joblib import Parallel, delayed
from scipy.stats import spearmanr

from run_production_backtest import backtest_no_lookahead


def extract_top500_params():
    """从all_freq_scan中提取Top 500收益的参数"""

    # 读取全部扫描结果(从父目录)
    freq_scan_file = (
        "../results_combo_wfo/20251106_021606/all_freq_scan_20251106_021606.csv"
    )

    if not Path(freq_scan_file).exists():
        print(f"❌ 文件不存在: {freq_scan_file}")
        return None

    df = pd.read_csv(freq_scan_file)

    # 按annual_ret排序，取Top 500
    df_sorted = df.nlargest(500, "annual_ret")

    print(f"📊 已从扫描结果中提取 Top 500 参数")
    print(
        f"   收益范围: {df_sorted['annual_ret'].min():.4f} ~ {df_sorted['annual_ret'].max():.4f}"
    )
    print(
        f"   Sharpe范围: {df_sorted['sharpe'].min():.4f} ~ {df_sorted['sharpe'].max():.4f}"
    )
    print()

    # 提取唯一的因子组合和频率
    unique_configs = []
    seen = set()

    for idx, row in df_sorted.iterrows():
        # 使用combo + wfo_freq作为唯一标识
        config_key = (row["combo"], row["wfo_freq"])
        if config_key not in seen:
            seen.add(config_key)
            unique_configs.append(
                {
                    "combo": row["combo"],
                    "wfo_freq": row["wfo_freq"],
                    "test_freq": row["test_freq"],
                    "top_annual_ret": row["annual_ret"],
                }
            )

    print(f"✅ 提取了 {len(unique_configs)} 个唯一参数配置")
    print()

    return unique_configs


def create_grid_search_task_list(unique_configs, position_size_range=range(1, 11)):
    """创建网格搜索任务列表

    对每个唯一的因子组合 + 频率配置，测试所有持仓数
    """

    tasks = []
    for config in unique_configs:
        for pos_size in position_size_range:
            tasks.append(
                {
                    "combo": config["combo"],
                    "wfo_freq": config["wfo_freq"],
                    "test_freq": config["test_freq"],
                    "position_size": pos_size,
                }
            )

    print(f"📋 生成了 {len(tasks)} 个任务")
    print(f"   配置数: {len(unique_configs)}")
    print(f"   持仓数范围: {min(position_size_range)}-{max(position_size_range)}")
    print()

    return tasks


def load_data_and_config():
    """加载数据和配置"""

    from core.cross_section_processor import CrossSectionProcessor
    from core.data_loader import DataLoader
    from core.precise_factor_library_v2 import PreciseFactorLibrary

    # 加载配置
    with open("configs/combo_wfo_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 加载数据
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
    dates = returns_df.index.strftime("%Y-%m-%d").tolist()

    return config, factors_data, returns, etf_names, dates


def run_grid_search(output_dir="../results_combo_wfo"):
    """执行 Top 500 收益参数的持仓数网格搜索"""

    print("\n" + "=" * 80)
    print("🚀 Top 500 收益参数 - 持仓数网格搜索")
    print("=" * 80)
    print()

    # 第1步：提取Top 500参数
    unique_configs = extract_top500_params()
    if unique_configs is None:
        return

    # 第2步：创建任务列表
    position_size_range = range(1, 11)
    tasks = create_grid_search_task_list(unique_configs, position_size_range)

    # 第3步：加载数据和配置
    print("📥 加载数据和配置...")
    config, factors_data, returns, etf_names, dates = load_data_and_config()

    print(f"✅ 已加载数据:")
    print(f"   时间范围: {dates[0]} ~ {dates[-1]}")
    print(f"   交易日数: {len(dates)}")
    print(f"   ETF数量: {len(etf_names)}")
    print(f"   因子数量: {factors_data.shape[2]}")
    print()

    # 第4步：执行任务
    print("⚙️  开始执行任务...")
    print("-" * 80)

    def run_task(task):
        """运行单个任务"""
        try:
            result = backtest_no_lookahead(
                factors_data=factors_data,
                returns=returns,
                etf_names=etf_names,
                rebalance_freq=task["test_freq"],
                lookback_window=config["backtest"]["lookback_window"],
                position_size=task["position_size"],
                commission_rate=config["backtest"].get("commission_rate", 0.00005),
                initial_capital=config["backtest"]["initial_capital"],
            )

            # 添加任务信息
            result["combo"] = task["combo"]
            result["wfo_freq"] = task["wfo_freq"]
            result["test_freq"] = task["test_freq"]
            result["position_size"] = task["position_size"]
            result["test_position_size"] = task["position_size"]

            return result
        except Exception as e:
            print(
                f"❌ 错误 - {task['combo'][:30]}... pos={task['position_size']}: {str(e)[:50]}"
            )
            return None

    # 并行执行（集成 tqdm 进度条）
    use_tqdm = (tqdm is not None) and (tqdm_joblib is not None)

    if use_tqdm:
        print("📟 使用 tqdm 进度条监控任务进度（如需关闭请卸载 tqdm-joblib 或改用 verbose）")
        with tqdm_joblib(tqdm(total=len(tasks), desc="回测进度", dynamic_ncols=True)):
            results = Parallel(
                n_jobs=config["backtest"]["max_workers"],
                backend="loky",
                verbose=0,
            )(delayed(run_task)(task) for task in tasks)
    else:
        if tqdm is None or tqdm_joblib is None:
            print("ℹ️ 未检测到 tqdm/tqdm-joblib，回退为 joblib 自带进度日志。\n   安装建议: pip install tqdm tqdm-joblib")
        results = Parallel(
            n_jobs=config["backtest"]["max_workers"],
            backend="loky",
            verbose=10,
        )(delayed(run_task)(task) for task in tasks)

    print()
    print("-" * 80)

    # 第5步：处理结果
    valid_results = [r for r in results if r is not None]
    print(f"✅ 成功完成 {len(valid_results)}/{len(tasks)} 个任务")
    print()

    # 第6步：保存结果
    df_results = pd.DataFrame(valid_results)

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(output_dir) / timestamp
    result_dir.mkdir(parents=True, exist_ok=True)

    # 保存CSV
    output_file = result_dir / f"top500_pos_scan_{timestamp}.csv"
    df_results.to_csv(output_file, index=False)
    print(f"💾 结果已保存: {output_file}")

    # 第7步：分析结果
    print()
    print("📊 分析结果...")
    print("=" * 80)

    stats = (
        df_results.groupby("test_position_size")
        .agg(
            {
                "sharpe": ["mean", "std", "min", "max"],
                "annual_ret": ["mean", "std", "min", "max"],
                "max_dd": ["mean"],
                "win_rate": ["mean"],
            }
        )
        .round(4)
    )

    print(stats)
    print()

    # 找到最优持仓数
    optimal_pos = df_results.groupby("test_position_size")["sharpe"].mean().idxmax()
    optimal_sharpe = df_results.groupby("test_position_size")["sharpe"].mean().max()
    optimal_annual = df_results[df_results["test_position_size"] == optimal_pos][
        "annual_ret"
    ].mean()

    print("🎯 最优持仓数分析:")
    print(f"   最优持仓数: {optimal_pos}")
    print(f"   平均Sharpe: {optimal_sharpe:.4f}")
    print(f"   平均年化: {optimal_annual:.4f}")
    print()

    # 保存分析报告
    report_file = result_dir / f"top500_analysis_{timestamp}.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("Top 500 收益参数 - 持仓数网格搜索分析\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"执行时间: {timestamp}\n")
        f.write(f"任务数: {len(tasks)}\n")
        f.write(f"成功率: {len(valid_results)}/{len(tasks)}\n\n")
        f.write("性能统计:\n")
        f.write(str(stats) + "\n\n")
        f.write(f"最优持仓数: {optimal_pos}\n")
        f.write(f"最优Sharpe: {optimal_sharpe:.4f}\n")
        f.write(f"最优年化: {optimal_annual:.4f}\n")

    print(f"📄 报告已保存: {report_file}")

    return df_results, result_dir


if __name__ == "__main__":
    df, output_dir = run_grid_search()
    print()
    print(f"✅ 所有任务完成！输出目录: {output_dir}")
