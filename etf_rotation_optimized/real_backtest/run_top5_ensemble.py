"""
Top 5 等权组合策略
================================================================================
将表现最好的 Top 5 因子组合等权组合，分散单一组合风险。

原理
----
1. 根据 WFO 排名选取 Top 5 因子组合
2. 每个组合独立运行回测，生成每日净值曲线
3. 将 5 个组合等权组合：最终净值 = 平均(各组合净值)
4. 计算组合后的 Sharpe、年化收益、最大回撤

优势
----
- 分散单一因子组合的过拟合风险
- 平滑收益曲线，降低回撤
- 更稳健的样本外表现
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

# --- ensure package import works ---
_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent
for p in (_HERE, _PKG_ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.append(sp)

from core.data_loader import DataLoader
from core.precise_factor_library_v2 import PreciseFactorLibrary
from core.market_timing import LightTimingModule
from run_production_backtest import backtest_no_lookahead


def load_config():
    """加载配置"""
    config_paths = [
        Path(__file__).resolve().parent.parent.parent / "configs" / "combo_wfo_config.yaml",
        Path(__file__).resolve().parent.parent / "configs" / "wfo_config.yaml",
    ]
    for cfg_path in config_paths:
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(f"Config not found in: {config_paths}")


def get_top5_combos_from_latest_run():
    """从最新回测结果中获取 Top 5 组合"""
    results_dir = Path(__file__).resolve().parent.parent.parent / "results_combo_wfo"
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    # 找到最新的结果目录
    run_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()], reverse=True)
    if not run_dirs:
        raise FileNotFoundError("No run directories found")
    
    latest_dir = run_dirs[0]
    print(f"📂 使用最新结果目录: {latest_dir.name}")
    
    # 查找完整结果文件
    full_result_files = list(latest_dir.glob("*_full.csv"))
    if not full_result_files:
        raise FileNotFoundError(f"No full result CSV found in {latest_dir}")
    
    result_file = full_result_files[0]
    print(f"📄 读取结果文件: {result_file.name}")
    
    df = pd.read_csv(result_file)
    
    # 按 Sharpe 排序取 Top 5
    df_sorted = df.sort_values("sharpe", ascending=False).head(5)
    
    print("\n🏆 Top 5 组合:")
    for idx, (_, row) in enumerate(df_sorted.iterrows(), 1):
        factors = row["combo"].split(" + ")
        print(f"  {idx}. {row['combo']}")
        print(f"     Sharpe: {row['sharpe']:.3f} | 年化: {row['annual_ret']*100:.1f}% | 回撤: {row['max_dd']*100:.1f}%")
    
    # 返回因子列表
    return [row["combo"].split(" + ") for _, row in df_sorted.iterrows()]


def run_single_combo_backtest(
    factor_names: list,
    factors_data_full: np.ndarray,
    factor_name_to_idx: dict,
    returns: np.ndarray,
    etf_names: list,
    timing_signal: np.ndarray,
    config: dict,
):
    """运行单个组合的回测，返回每日净值"""
    # 提取因子数据
    factor_indices = [factor_name_to_idx[f] for f in factor_names]
    factors_data = factors_data_full[:, :, factor_indices]
    
    backtest_cfg = config.get("backtest", {})
    
    result = backtest_no_lookahead(
        factors_data=factors_data,
        returns=returns,
        etf_names=etf_names,
        rebalance_freq=backtest_cfg.get("rebalance_freq", 8),
        lookback_window=backtest_cfg.get("lookback_window", 252),
        position_size=backtest_cfg.get("position_size", 3),
        initial_capital=backtest_cfg.get("initial_capital", 1_000_000),
        commission_rate=backtest_cfg.get("commission_rate", 0.00005),
        factors_data_full=factors_data_full,
        factor_indices_for_cache=np.array(factor_indices, dtype=np.int64),
        timing_signal=timing_signal,
        etf_stop_loss=0.0,
    )
    
    return {
        "factors": factor_names,
        "nav_series": result["nav"],
        "annual_return": result["annual_ret"],
        "sharpe": result["sharpe"],
        "max_drawdown": result["max_dd"],
        "final_nav": result["final"],
    }


def calculate_ensemble_metrics(nav_matrix: np.ndarray, trading_days_per_year: int = 244):
    """
    计算等权组合的指标
    
    参数:
        nav_matrix: (n_combos, T) 各组合的净值序列
        trading_days_per_year: 年交易日数
    
    返回:
        dict: 组合指标
    """
    # 等权组合净值
    ensemble_nav = np.mean(nav_matrix, axis=0)
    
    # 计算收益率
    returns = np.diff(ensemble_nav) / ensemble_nav[:-1]
    returns = returns[~np.isnan(returns)]
    
    # 年化收益
    total_return = ensemble_nav[-1] / ensemble_nav[0] - 1
    n_years = len(ensemble_nav) / trading_days_per_year
    annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    # Sharpe
    if len(returns) > 0 and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(trading_days_per_year)
    else:
        sharpe = 0
    
    # 最大回撤
    peak = np.maximum.accumulate(ensemble_nav)
    drawdown = (ensemble_nav - peak) / peak
    max_drawdown = np.min(drawdown)
    
    return {
        "ensemble_nav": ensemble_nav,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "final_nav": ensemble_nav[-1],
        "total_return": total_return,
    }


def main():
    print("=" * 80)
    print("Top 5 等权组合策略回测")
    print("=" * 80)
    
    start_time = time.time()
    
    # 加载配置
    config = load_config()
    print(f"\n📝 配置加载完成")
    
    # 获取 Top 5 组合
    try:
        top5_combos = get_top5_combos_from_latest_run()
    except FileNotFoundError as e:
        print(f"\n⚠️ 无法自动获取 Top 5 组合: {e}")
        print("使用默认的 Top 5 组合（基于最新回测结果）:")
        top5_combos = [
            ["MAX_DD_60D", "MOM_20D", "RSI_14", "VOL_RATIO_20D", "VOL_RATIO_60D"],
            ["ADX_14D", "MAX_DD_60D", "MOM_20D", "RSI_14", "VOL_RATIO_60D"],
            ["ADX_14D", "OBV_SLOPE_10D", "PRICE_POSITION_20D", "PV_CORR_20D", "SHARPE_RATIO_20D"],
            ["MAX_DD_60D", "MOM_20D", "RSI_14", "VOL_RATIO_60D"],
            ["MAX_DD_60D", "MOM_20D", "RET_VOL_20D", "RSI_14", "VOL_RATIO_60D"],
        ]
        for i, combo in enumerate(top5_combos, 1):
            print(f"  {i}. {' + '.join(combo)}")
    
    # 加载数据
    print("\n📊 加载数据...")
    data_loader = DataLoader()
    ohlcv_data = data_loader.load_ohlcv()
    close_df = ohlcv_data["close"]
    
    # 转换为 numpy 数组
    close_prices = close_df.values
    dates = close_df.index.tolist()
    etf_names = close_df.columns.tolist()
    print(f"  数据维度: {close_prices.shape[0]} 天 × {close_prices.shape[1]} ETF")
    
    # 计算收益率
    returns = np.zeros_like(close_prices)
    returns[1:] = close_prices[1:] / close_prices[:-1] - 1
    
    # 计算因子
    print("\n🔢 计算因子...")
    factor_lib = PreciseFactorLibrary()
    all_factors_df = factor_lib.compute_all_factors(ohlcv_data)
    
    # 构建因子索引映射和 numpy 数组
    factor_names_list = all_factors_df.columns.get_level_values(0).unique().tolist()
    factor_name_to_idx = {name: i for i, name in enumerate(factor_names_list)}
    
    # 将多层索引 DataFrame 转换为 (T, N, F) 数组
    T, N = close_prices.shape
    F = len(factor_names_list)
    factors_data_full = np.zeros((T, N, F), dtype=np.float64)
    
    for f_idx, f_name in enumerate(factor_names_list):
        factor_df = all_factors_df[f_name]
        # 确保列顺序与 etf_names 一致
        factor_df = factor_df.reindex(columns=etf_names)
        factors_data_full[:, :, f_idx] = factor_df.values
    
    print(f"  因子数量: {len(factor_names_list)}")
    
    # 计算择时信号
    print("\n⏰ 计算择时信号...")
    timing_cfg = config.get("backtest", {}).get("timing", {})
    if timing_cfg.get("enabled", True):
        extreme_threshold = timing_cfg.get("extreme_threshold", -0.3)
        extreme_position = timing_cfg.get("extreme_position", 0.3)
        
        timing_module = LightTimingModule(
            extreme_threshold=extreme_threshold,
            extreme_position=extreme_position
        )
        timing_position = timing_module.compute_position_ratios(close_df)
        
        # 转换为 numpy 数组并 shift(1) 避免未来函数
        timing_position = timing_position.shift(1).fillna(1.0).values
        
        print(f"  择时启用: extreme_threshold={extreme_threshold}, extreme_position={extreme_position}")
        print(f"  防守日占比: {np.mean(timing_position < 1.0)*100:.1f}%")
    else:
        timing_position = np.ones(len(dates))
        print("  择时未启用，全仓运行")
    
    # 运行 Top 5 组合回测
    print("\n🚀 运行 Top 5 组合回测...")
    results = []
    
    for i, combo in enumerate(top5_combos, 1):
        print(f"\n  [{i}/5] {' + '.join(combo)}")
        result = run_single_combo_backtest(
            factor_names=combo,
            factors_data_full=factors_data_full,
            factor_name_to_idx=factor_name_to_idx,
            returns=returns,
            etf_names=etf_names,
            timing_signal=timing_position,
            config=config,
        )
        results.append(result)
        print(f"      Sharpe: {result['sharpe']:.3f} | 年化: {result['annual_return']*100:.1f}% | 回撤: {result['max_drawdown']*100:.1f}%")
    
    # 构建净值矩阵
    nav_matrix = np.array([r["nav_series"] for r in results])
    
    # 计算等权组合指标
    print("\n📈 计算等权组合指标...")
    ensemble_metrics = calculate_ensemble_metrics(nav_matrix)
    
    # 打印结果
    print("\n" + "=" * 80)
    print("回测结果汇总")
    print("=" * 80)
    
    print("\n📊 各组合表现:")
    print("-" * 60)
    for i, result in enumerate(results, 1):
        print(f"  {i}. {' + '.join(result['factors'])}")
        print(f"     Sharpe: {result['sharpe']:.3f} | 年化: {result['annual_return']*100:.1f}% | 回撤: {result['max_drawdown']*100:.1f}%")
    
    print("\n🏆 等权组合表现 (Top 5 平均):")
    print("-" * 60)
    print(f"  年化收益: {ensemble_metrics['annual_return']*100:.1f}%")
    print(f"  Sharpe:   {ensemble_metrics['sharpe']:.3f}")
    print(f"  最大回撤: {ensemble_metrics['max_drawdown']*100:.1f}%")
    print(f"  终值:     100万 → {ensemble_metrics['final_nav']/10000:.1f}万")
    
    # 与单一 Top 1 对比
    print("\n📊 与 Top 1 组合对比:")
    print("-" * 60)
    print(f"  Top 1 Sharpe:    {results[0]['sharpe']:.3f}")
    print(f"  等权组合 Sharpe: {ensemble_metrics['sharpe']:.3f}")
    sharpe_diff = ensemble_metrics['sharpe'] - results[0]['sharpe']
    print(f"  差异: {sharpe_diff:+.3f} ({'提升' if sharpe_diff > 0 else '下降'})")
    
    print(f"\n  Top 1 回撤:    {results[0]['max_drawdown']*100:.1f}%")
    print(f"  等权组合回撤: {ensemble_metrics['max_drawdown']*100:.1f}%")
    dd_diff = ensemble_metrics['max_drawdown'] - results[0]['max_drawdown']
    print(f"  差异: {dd_diff*100:+.1f}% ({'改善' if dd_diff > 0 else '恶化'})")
    
    # 保存结果
    output_dir = Path(__file__).resolve().parent.parent.parent / "results_combo_wfo" / "ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存净值曲线
    nav_df = pd.DataFrame({
        "date": dates[252:252+len(ensemble_metrics["ensemble_nav"])],
        "ensemble_nav": ensemble_metrics["ensemble_nav"],
    })
    for i, result in enumerate(results, 1):
        nav_df[f"combo_{i}_nav"] = result["nav_series"]
    
    nav_file = output_dir / f"top5_ensemble_nav_{timestamp}.csv"
    nav_df.to_csv(nav_file, index=False)
    print(f"\n📁 净值曲线已保存: {nav_file}")
    
    # 保存汇总
    summary = {
        "timestamp": timestamp,
        "ensemble_sharpe": ensemble_metrics["sharpe"],
        "ensemble_annual_return": ensemble_metrics["annual_return"],
        "ensemble_max_drawdown": ensemble_metrics["max_drawdown"],
        "ensemble_final_nav": ensemble_metrics["final_nav"],
    }
    for i, result in enumerate(results, 1):
        summary[f"combo_{i}_factors"] = " + ".join(result["factors"])
        summary[f"combo_{i}_sharpe"] = result["sharpe"]
        summary[f"combo_{i}_annual_return"] = result["annual_return"]
        summary[f"combo_{i}_max_drawdown"] = result["max_drawdown"]
    
    summary_df = pd.DataFrame([summary])
    summary_file = output_dir / f"top5_ensemble_summary_{timestamp}.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"📁 汇总已保存: {summary_file}")
    
    elapsed = time.time() - start_time
    print(f"\n⏱️ 总耗时: {elapsed:.1f}秒")
    
    return ensemble_metrics, results


if __name__ == "__main__":
    main()
