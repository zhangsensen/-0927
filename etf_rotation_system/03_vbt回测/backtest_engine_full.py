#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ETF轮动回测引擎 - VectorBT暴力版

适配 etf_rotation_system 的因子筛选结果
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import glob
import itertools
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import vectorbt as vbt

    HAS_VBT = True
except ImportError:
    HAS_VBT = False
    print("警告: 未安装 vectorbt，请运行: pip install vectorbt")


def load_factor_panel(panel_path: str) -> pd.DataFrame:
    """加载因子面板"""
    panel = pd.read_parquet(panel_path)
    if not isinstance(panel.index, pd.MultiIndex):
        raise ValueError("面板必须是 (symbol, date) MultiIndex")
    return panel


def load_price_data(price_dir: str) -> pd.DataFrame:
    """加载价格数据"""
    prices = []
    for f in sorted(glob.glob(f"{price_dir}/*.parquet")):
        df = pd.read_parquet(f)
        symbol = f.split("/")[-1].split("_")[0]
        df["symbol"] = symbol
        df["date"] = pd.to_datetime(df["trade_date"])
        prices.append(df[["date", "close", "symbol"]])

    price_df = pd.concat(prices, ignore_index=True)
    pivot = price_df.pivot(index="date", columns="symbol", values="close")
    return pivot.sort_index().ffill()


def load_top_factors(screening_csv: str, top_k: int = 10) -> List[str]:
    """从筛选结果加载Top K因子"""
    df = pd.read_csv(screening_csv)
    # 修复：筛选结果列名是'factor'不是'panel_factor'
    col_name = "factor" if "factor" in df.columns else "panel_factor"
    return df.head(top_k)[col_name].tolist()


def calculate_composite_score(
    panel: pd.DataFrame,
    factors: List[str],
    weights: Dict[str, float],
    method: str = "zscore",
) -> pd.DataFrame:
    """计算复合因子得分 - 完全向量化实现

    Args:
        panel: 因子面板 (symbol, date) MultiIndex
        factors: 因子列表
        weights: 因子权重字典
        method: 标准化方法 ('zscore' or 'rank')

    Returns:
        得分矩阵 (date, symbol)
    """
    # 重塑为 (date, symbol) 结构
    factor_data = panel[factors].unstack(level="symbol")

    # 向量化标准化
    if method == "zscore":
        normalized = (
            factor_data - factor_data.mean(axis=1, skipna=True).values[:, np.newaxis]
        ) / (factor_data.std(axis=1, skipna=True).values[:, np.newaxis] + 1e-8)
    else:  # rank
        normalized = factor_data.rank(axis=1, pct=True) * 2 - 1  # [-1, 1]

    # 获取维度信息
    n_dates, n_total = normalized.shape
    n_factors = len(factors)
    n_symbols = n_total // n_factors

    # 重塑为 (dates, symbols, factors) 用于矩阵乘法
    reshaped = normalized.values.reshape(n_dates, n_symbols, n_factors)

    # 向量化加权求和 - 无循环
    weight_array = np.array([weights.get(f, 0) for f in factors])
    scores_array = np.sum(reshaped * weight_array[np.newaxis, np.newaxis, :], axis=2)

    # 创建结果DataFrame
    symbols = [col[1] for col in normalized.columns[::n_factors]]  # 提取symbol名称
    scores = pd.DataFrame(scores_array, index=normalized.index, columns=symbols)

    return scores


def build_target_weights(scores: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """构建Top-N目标权重"""
    ranks = scores.rank(axis=1, ascending=False, method="first")
    selection = ranks <= top_n
    weights = selection.astype(float)
    weights = weights.div(weights.sum(axis=1), axis=0).fillna(0.0)
    return weights


def backtest_topn_rotation(
    prices: pd.DataFrame,
    scores: pd.DataFrame,
    top_n: int = 5,
    rebalance_freq: int = 20,
    fees: float = 0.001,
    init_cash: float = 1_000_000,
) -> Dict:
    """Top-N轮动回测 - 向量化实现

    Args:
        prices: 价格矩阵 (date, symbol)
        scores: 得分矩阵 (date, symbol)
        top_n: 持仓数量
        rebalance_freq: 调仓频率（交易日）
        fees: 交易费用率
        init_cash: 初始资金

    Returns:
        回测结果字典
    """
    # 对齐日期
    common_dates = prices.index.intersection(scores.index)
    prices = prices.loc[common_dates]
    scores = scores.loc[common_dates]

    # 构建目标权重
    weights = build_target_weights(scores, top_n)

    # 向量化调仓日权重更新 - 无循环
    rebalance_mask = pd.Series(
        np.arange(len(weights)) % rebalance_freq == 0, index=weights.index
    )
    rebalance_mask.iloc[0] = True  # 第一天调仓

    # 使用 ffill 向前填充权重
    weights_ffill = weights.where(rebalance_mask, np.nan).ffill().fillna(0.0)

    # 计算收益 - 确保列对齐
    asset_returns = prices.pct_change().fillna(0.0)
    prev_weights = weights_ffill.shift().fillna(0.0)

    # 对齐列名
    common_symbols = asset_returns.columns.intersection(prev_weights.columns)
    asset_returns_aligned = asset_returns[common_symbols]
    prev_weights_aligned = prev_weights[common_symbols]

    gross_returns = (prev_weights_aligned * asset_returns_aligned).sum(axis=1)

    # 交易成本
    weight_diff = weights_ffill.diff().abs().sum(axis=1).fillna(0.0)
    turnover = 0.5 * weight_diff
    net_returns = gross_returns - fees * turnover

    # 净值曲线
    equity = (1 + net_returns).cumprod() * init_cash

    # 统计指标
    total_return = (equity.iloc[-1] / init_cash - 1) * 100
    periods_per_year = 252
    sharpe = (
        net_returns.mean() / net_returns.std() * np.sqrt(periods_per_year)
        if net_returns.std() > 0
        else 0
    )

    running_max = equity.cummax()
    drawdown = (equity / running_max - 1) * 100
    max_dd = drawdown.min()

    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "final_value": equity.iloc[-1],
        "turnover": turnover.sum(),
    }


def grid_search_weights(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    factors: List[str],
    top_n_list: List[int] = [3, 5, 8],
    weight_grid: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    max_combos: int = 10000,
    rebalance_freq: int = 20,
) -> pd.DataFrame:
    """网格搜索因子权重组合 - 向量化优化

    Args:
        panel: 因子面板
        prices: 价格数据
        factors: 因子列表
        top_n_list: Top-N候选列表
        weight_grid: 权重网格
        max_combos: 最大组合数
        rebalance_freq: 调仓频率

    Returns:
        结果DataFrame
    """
    # 生成权重组合
    weight_combos = list(itertools.product(weight_grid, repeat=len(factors)))

    # 向量化过滤：权重和接近1
    weight_sums = np.array([sum(w) for w in weight_combos])
    valid_mask = (weight_sums >= 0.7) & (weight_sums <= 1.3)
    valid_combos = [
        weight_combos[i] for i in range(len(weight_combos)) if valid_mask[i]
    ]

    # 限制组合数
    if len(valid_combos) > max_combos:
        valid_combos = valid_combos[:max_combos]

    print(f"开始网格搜索: {len(valid_combos)} 组合 × {len(top_n_list)} Top-N")

    # 预计算所有得分矩阵以避免重复计算
    score_cache = {}

    results = []
    for weights in tqdm(valid_combos, desc="权重组合"):
        weight_dict = dict(zip(factors, weights))
        weights_key = tuple(weights)

        # 缓存得分矩阵
        if weights_key not in score_cache:
            score_cache[weights_key] = calculate_composite_score(
                panel, factors, weight_dict
            )

        scores = score_cache[weights_key]

        # 批量测试所有 top_n 值
        for top_n in top_n_list:
            try:
                result = backtest_topn_rotation(
                    prices=prices,
                    scores=scores,
                    top_n=top_n,
                    rebalance_freq=rebalance_freq,
                )

                results.append(
                    {
                        "weights": str(weight_dict),
                        "top_n": top_n,
                        "total_return": result["total_return"],
                        "sharpe_ratio": result["sharpe_ratio"],
                        "max_drawdown": result["max_drawdown"],
                        "final_value": result["final_value"],
                        "turnover": result["turnover"],
                    }
                )
            except Exception as e:
                print(f"组合失败: {weight_dict}, top_n={top_n}, 错误: {e}")

    # 向量化结果处理
    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df.sort_values("sharpe_ratio", ascending=False)
    return df


def run_backtest(
    panel_path: str,
    price_dir: str,
    screening_csv: str,
    output_dir: str,
    top_k: int = 10,
    top_n_list: List[int] = [3, 5, 8],
    rebalance_freq: int = 20,
    max_combos: int = 1000,
):
    """完整回测流程"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("ETF轮动回测引擎")
    print("=" * 80)
    print(f"时间戳: {timestamp}")
    print(f"面板: {panel_path}")
    print(f"筛选: {screening_csv}")

    # 加载数据
    print("\n加载数据...")
    panel = load_factor_panel(panel_path)
    prices = load_price_data(price_dir)
    factors = load_top_factors(screening_csv, top_k)

    print(f"  因子数: {len(factors)}")
    print(f"  因子: {factors[:5]}...")
    print(f"  ETF数: {len(prices.columns)}")
    print(f"  日期: {prices.index.min().date()} ~ {prices.index.max().date()}")

    # 网格搜索
    print("\n开始回测...")
    results = grid_search_weights(
        panel=panel,
        prices=prices,
        factors=factors,
        top_n_list=top_n_list,
        max_combos=max_combos,
        rebalance_freq=rebalance_freq,
    )

    # 保存结果
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_file = output_path / f"backtest_results_{timestamp}.csv"
    results.to_csv(csv_file, index=False)
    print(f"\n结果: {csv_file}")

    # 输出Top 10
    print("\nTop 10 策略:")
    print(results.head(10).to_string(index=False))

    # 保存最优策略配置
    best = results.iloc[0]
    best_config = {
        "timestamp": timestamp,
        "weights": best["weights"],
        "top_n": int(best["top_n"]),
        "rebalance_freq": rebalance_freq,
        "performance": {
            "total_return": float(best["total_return"]),
            "sharpe_ratio": float(best["sharpe_ratio"]),
            "max_drawdown": float(best["max_drawdown"]),
        },
        "factors": factors,
        "data_source": {"panel": panel_path, "screening": screening_csv},
    }

    config_file = output_path / f"best_strategy_{timestamp}.json"
    with open(config_file, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"最优配置: {config_file}")

    return results, best_config


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print(
            "用法: python backtest_engine.py <panel> <price_dir> <screening_csv> [output_dir]"
        )
        print("\n示例:")
        print("  python 03_vbt回测/backtest_engine_full.py \\")
        print("    ../etf_cross_section_results/panel_20251018_024539.parquet \\")
        print("    ../../raw/ETF/daily \\")
        print("    dummy_screening.csv \\")
        print("    ./results")
        sys.exit(1)

    panel_path = sys.argv[1]
    price_dir = sys.argv[2]
    screening_csv = sys.argv[3]
    output_dir = (
        sys.argv[4] if len(sys.argv) > 4 else "etf_rotation_system/strategies/results"
    )

    run_backtest(
        panel_path=panel_path,
        price_dir=price_dir,
        screening_csv=screening_csv,
        output_dir=output_dir,
        top_k=10,
        top_n_list=[3, 5, 8, 10],
        rebalance_freq=20,
        max_combos=10000,
    )
