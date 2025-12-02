#!/usr/bin/env python3
"""
风控参数网格搜索优化器

目标：在零杠杆前提下，寻找能最大化 Calmar Ratio 的风控参数组合。

扫描维度：
- 移动止损 (trailing_stop_pct): [0.05, 0.08, 0.10, 0.12, 0.15]
- 阶梯止盈 (profit_ladders): 4种方案
- 冷却期 (cooldown): [1, 3, 5]

约束：
- 零杠杆 (leverage_cap=1.0)
- 熔断机制可选开启

输出：
- 打印最佳参数组合及其对应的收益率和回撤
- 自动更新配置文件为最佳参数

用法：
    uv run python scripts/optimize_risk_params.py
"""
import sys
from pathlib import Path
from itertools import product
from datetime import datetime

ROOT = Path(__file__).parent.parent

import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import shift_timing_signal

# 导入 VEC 回测函数（不修改引擎内核）
from batch_vec_backtest import run_vec_backtest


# ═══════════════════════════════════════════════════════════════════════════════
# 参数搜索空间定义
# ═══════════════════════════════════════════════════════════════════════════════

# 移动止损率
TRAILING_STOP_GRID = [0.05, 0.08, 0.10, 0.12, 0.15]

# 阶梯止盈方案
PROFIT_LADDER_PRESETS = {
    "无": [],
    "激进": [{"threshold": 0.10, "new_stop": 0.05}],
    "稳健": [{"threshold": 0.15, "new_stop": 0.05}, {"threshold": 0.30, "new_stop": 0.03}],
    "宽松": [{"threshold": 0.20, "new_stop": 0.08}, {"threshold": 0.40, "new_stop": 0.05}],
}

# 冷却期
COOLDOWN_GRID = [0, 1, 3, 5]


def run_param_scan(
    factors_3d,
    close_prices,
    open_prices,
    high_prices,
    low_prices,
    timing_arr,
    combo_indices_list,
    combo_names,
    base_config,
):
    """
    执行网格搜索，返回所有参数组合的结果。
    
    Args:
        factors_3d: 因子数据 (T, N, F)
        close/open/high/low_prices: 价格数据
        timing_arr: 择时信号
        combo_indices_list: 因子组合索引列表
        combo_names: 因子组合名称列表
        base_config: 基础配置（freq, pos_size等）
    
    Returns:
        DataFrame: 所有参数组合的回测结果
    """
    results = []
    
    # 生成参数网格
    param_grid = list(product(
        TRAILING_STOP_GRID,
        PROFIT_LADDER_PRESETS.keys(),
        COOLDOWN_GRID,
    ))
    
    total_runs = len(param_grid) * len(combo_indices_list)
    print(f"\n🔍 参数搜索空间:")
    print(f"   止损率: {TRAILING_STOP_GRID}")
    print(f"   止盈方案: {list(PROFIT_LADDER_PRESETS.keys())}")
    print(f"   冷却期: {COOLDOWN_GRID}")
    print(f"   总参数组合: {len(param_grid)}")
    print(f"   总回测次数: {total_runs}")
    print()
    
    with tqdm(total=total_runs, desc="参数扫描") as pbar:
        for stop_pct, ladder_name, cooldown in param_grid:
            profit_ladders = PROFIT_LADDER_PRESETS[ladder_name]
            
            # 对该参数组合运行所有因子组合
            combo_results = []
            for combo_name, factor_indices in zip(combo_names, combo_indices_list):
                ret, wr, pf, trades, rounding, risk = run_vec_backtest(
                    factors_3d, close_prices, open_prices, high_prices, low_prices,
                    timing_arr, factor_indices,
                    # 基础参数
                    freq=base_config["freq"],
                    pos_size=base_config["pos_size"],
                    initial_capital=base_config["initial_capital"],
                    commission_rate=base_config["commission_rate"],
                    lookback=base_config["lookback"],
                    # 动态杠杆（禁用）
                    target_vol=0.20,
                    vol_window=20,
                    dynamic_leverage_enabled=False,
                    # 风控参数（搜索目标）
                    trailing_stop_pct=stop_pct,
                    profit_ladders=profit_ladders,
                    circuit_breaker_day=0.0,  # 暂时禁用熔断
                    circuit_breaker_total=0.0,
                    circuit_recovery_days=5,
                    cooldown_days=cooldown,
                    leverage_cap=1.0,  # 零杠杆
                )
                
                combo_results.append({
                    "combo": combo_name,
                    "return": ret,
                    "max_dd": risk["max_drawdown"],
                    "calmar": risk["calmar_ratio"],
                    "sharpe": risk["sharpe_ratio"],
                    "trades": trades,
                    "win_rate": wr,
                })
                pbar.update(1)
            
            # 汇总该参数组合的统计数据
            df_combo = pd.DataFrame(combo_results)
            
            # 计算统计指标（关注稳健性）
            results.append({
                "stop_pct": stop_pct,
                "ladder": ladder_name,
                "cooldown": cooldown,
                # 平均指标
                "avg_return": df_combo["return"].mean(),
                "avg_calmar": df_combo["calmar"].mean(),
                "avg_sharpe": df_combo["sharpe"].mean(),
                "avg_max_dd": df_combo["max_dd"].mean(),
                "avg_trades": df_combo["trades"].mean(),
                # 稳健性指标（中位数）
                "median_return": df_combo["return"].median(),
                "median_calmar": df_combo["calmar"].median(),
                # 最佳组合指标
                "best_return": df_combo["return"].max(),
                "best_calmar": df_combo["calmar"].max(),
                "best_combo": df_combo.loc[df_combo["calmar"].idxmax(), "combo"],
                # 稳健性（收益为正的组合比例）
                "positive_ratio": (df_combo["return"] > 0).mean(),
                # 各组合详细数据（用于后续分析）
                "_combo_details": df_combo.to_dict("records"),
            })
    
    return pd.DataFrame(results)


def select_best_params(df_results, selection_method="avg_calmar"):
    """
    选择最佳参数组合。
    
    Args:
        df_results: 参数搜索结果
        selection_method: 选择方法
            - "avg_calmar": 平均 Calmar 最高
            - "median_calmar": 中位 Calmar 最高（更稳健）
            - "best_calmar": 最佳单组合 Calmar
            - "robust": 综合评分（Calmar * 正收益比例）
    
    Returns:
        最佳参数字典
    """
    if selection_method == "avg_calmar":
        best_idx = df_results["avg_calmar"].idxmax()
    elif selection_method == "median_calmar":
        best_idx = df_results["median_calmar"].idxmax()
    elif selection_method == "best_calmar":
        best_idx = df_results["best_calmar"].idxmax()
    elif selection_method == "robust":
        # 综合评分：平均Calmar * 正收益比例（惩罚不稳定的参数）
        df_results["robust_score"] = df_results["avg_calmar"] * df_results["positive_ratio"]
        best_idx = df_results["robust_score"].idxmax()
    else:
        raise ValueError(f"Unknown selection method: {selection_method}")
    
    best_row = df_results.loc[best_idx]
    return {
        "trailing_stop_pct": best_row["stop_pct"],
        "profit_ladders": PROFIT_LADDER_PRESETS[best_row["ladder"]],
        "cooldown_days": best_row["cooldown"],
        "ladder_name": best_row["ladder"],
        # 性能指标
        "avg_return": best_row["avg_return"],
        "avg_calmar": best_row["avg_calmar"],
        "avg_max_dd": best_row["avg_max_dd"],
        "positive_ratio": best_row["positive_ratio"],
        "best_combo": best_row["best_combo"],
    }


def update_config_file(best_params, config_path):
    """
    更新配置文件为最佳参数。
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    risk_control = config["backtest"]["risk_control"]
    # 确保转换为 Python 原生类型
    risk_control["trailing_stop_pct"] = float(best_params["trailing_stop_pct"])
    risk_control["profit_ladders"] = [
        {"threshold": float(l["threshold"]), "new_stop": float(l["new_stop"])}
        for l in best_params["profit_ladders"]
    ]
    risk_control["cooldown_days"] = int(best_params["cooldown_days"])
    
    # 备份原配置
    backup_path = config_path.with_suffix(".yaml.bak")
    import shutil
    shutil.copy(config_path, backup_path)
    
    # 写入新配置（使用 safe_dump 确保兼容性）
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    return backup_path


def main():
    print("=" * 80)
    print("🔍 风控参数网格搜索优化器")
    print("=" * 80)
    
    # 1. 加载配置
    config_path = ROOT / "configs/combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    backtest_config = config.get("backtest", {})
    base_config = {
        "freq": backtest_config.get("freq", 8),
        "pos_size": backtest_config.get("pos_size", 3),
        "lookback": backtest_config.get("lookback", 252),
        "initial_capital": float(backtest_config.get("initial_capital", 1_000_000)),
        "commission_rate": float(backtest_config.get("commission_rate", 0.0002)),
    }
    
    print(f"✅ 基础配置:")
    print(f"   FREQ: {base_config['freq']}")
    print(f"   POS_SIZE: {base_config['pos_size']}")
    print(f"   LOOKBACK: {base_config['lookback']}")
    
    # 2. 加载 WFO 结果
    wfo_dirs = sorted([d for d in (ROOT / "results").glob("run_*") if d.is_dir() and not d.is_symlink()])
    if not wfo_dirs:
        print("❌ 未找到 WFO 结果目录")
        return
    
    latest_wfo = wfo_dirs[-1]
    combos_path = latest_wfo / "top100_by_ic.parquet"
    if not combos_path.exists():
        combos_path = latest_wfo / "all_combos.parquet"
    
    df_combos = pd.read_parquet(combos_path)
    print(f"✅ 加载 WFO 结果 ({latest_wfo.name})：{len(df_combos)} 个组合")
    
    # 3. 加载数据
    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )
    
    # 4. 计算因子
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
    high_prices = ohlcv["high"][etf_codes].ffill().bfill().values
    low_prices = ohlcv["low"][etf_codes].ffill().bfill().values
    
    # 5. 择时信号
    timing_config = config.get("backtest", {}).get("timing", {})
    timing_module = LightTimingModule(
        extreme_threshold=timing_config.get("extreme_threshold", -0.3),
        extreme_position=timing_config.get("extreme_position", 0.3),
    )
    timing_series = timing_module.compute_position_ratios(ohlcv["close"])
    timing_arr_raw = timing_series.reindex(dates).fillna(1.0).values
    timing_arr = shift_timing_signal(timing_arr_raw)
    
    print(f"✅ 数据加载完成：{T} 天 × {N} 只 ETF × {len(factor_names)} 个因子")
    
    # 6. 准备因子组合
    factor_index_map = {name: idx for idx, name in enumerate(factor_names)}
    combo_strings = df_combos["combo"].tolist()
    combo_indices_list = [
        [factor_index_map[f.strip()] for f in combo.split(" + ")]
        for combo in combo_strings
    ]
    
    # 7. 执行参数搜索
    print("\n" + "=" * 80)
    print("⚡ 开始参数网格搜索")
    print("=" * 80)
    
    df_results = run_param_scan(
        factors_3d, close_prices, open_prices, high_prices, low_prices,
        timing_arr, combo_indices_list, combo_strings, base_config,
    )
    
    # 8. 保存详细结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / "results" / f"param_scan_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 移除详细数据列以便保存
    df_save = df_results.drop(columns=["_combo_details"])
    df_save.to_parquet(output_dir / "param_scan_results.parquet", index=False)
    df_save.to_csv(output_dir / "param_scan_results.csv", index=False)
    
    # 9. 选择最佳参数（多种方法）
    print("\n" + "=" * 80)
    print("📊 参数搜索结果")
    print("=" * 80)
    
    # 按不同方法选择
    methods = ["avg_calmar", "median_calmar", "robust"]
    best_params_all = {}
    
    for method in methods:
        best = select_best_params(df_results, method)
        best_params_all[method] = best
        print(f"\n🏆 [{method}] 最佳参数:")
        print(f"   止损率: {best['trailing_stop_pct']*100:.0f}%")
        print(f"   止盈方案: {best['ladder_name']}")
        print(f"   冷却期: {best['cooldown_days']} 天")
        print(f"   ---")
        print(f"   平均收益率: {best['avg_return']*100:.2f}%")
        print(f"   平均 Calmar: {best['avg_calmar']:.3f}")
        print(f"   平均最大回撤: {best['avg_max_dd']*100:.2f}%")
        print(f"   正收益比例: {best['positive_ratio']*100:.1f}%")
        print(f"   最佳组合: {best['best_combo']}")
    
    # 10. 显示 Top 10 参数组合
    print("\n" + "=" * 80)
    print("📈 Top 10 参数组合 (按平均 Calmar 排序)")
    print("=" * 80)
    
    top10 = df_results.nlargest(10, "avg_calmar")[
        ["stop_pct", "ladder", "cooldown", "avg_return", "avg_calmar", "avg_max_dd", "positive_ratio"]
    ]
    print(top10.to_string(index=False))
    
    # 11. 询问是否更新配置
    print("\n" + "=" * 80)
    recommended = best_params_all["robust"]  # 推荐使用稳健性选择
    print(f"💡 推荐参数 (robust 方法):")
    print(f"   trailing_stop_pct: {recommended['trailing_stop_pct']}")
    print(f"   profit_ladders: {recommended['ladder_name']} = {recommended['profit_ladders']}")
    print(f"   cooldown_days: {recommended['cooldown_days']}")
    print()
    
    # 自动更新配置
    print("🔧 自动更新配置文件...")
    backup = update_config_file(recommended, config_path)
    print(f"   ✅ 配置已更新: {config_path}")
    print(f"   📦 备份已保存: {backup}")
    
    # 12. 保存推荐参数
    with open(output_dir / "recommended_params.yaml", "w") as f:
        yaml.dump({
            "trailing_stop_pct": recommended["trailing_stop_pct"],
            "profit_ladders": recommended["profit_ladders"],
            "cooldown_days": recommended["cooldown_days"],
            "ladder_name": recommended["ladder_name"],
            "performance": {
                "avg_return": float(recommended["avg_return"]),
                "avg_calmar": float(recommended["avg_calmar"]),
                "avg_max_dd": float(recommended["avg_max_dd"]),
                "positive_ratio": float(recommended["positive_ratio"]),
            }
        }, f, default_flow_style=False, allow_unicode=True)
    
    print(f"\n✅ 参数搜索完成")
    print(f"   输出目录: {output_dir}")


if __name__ == "__main__":
    main()
