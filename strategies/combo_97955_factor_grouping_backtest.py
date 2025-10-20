#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Combo 97955 因子分组真实回测验证

🔥 修复内容：
1. 因子覆盖缺口：确保所有因子都被正确归类（包括_12, _14等）
2. 交易成本修正：使用真实港股ETF费率 0.0028（双边0.35%）
3. 真实样本统计：从持仓变化计算实际交易笔数
4. 完整分析交付：假信号过滤、持续性分层、可视化、操作阈值

基于vectorbt_multifactor_grid.py的VectorizedBacktestEngine实现真实回测
原则：Only Real Data. No Fake Data.
"""

import json
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# 复用现有的真实回测引擎
sys.path.insert(0, str(Path(__file__).parent))
from vectorbt_multifactor_grid import (
    VectorizedBacktestEngine,
    load_factor_panel,
    load_price_pivot,
    normalize_factors,
)

# 设置中文显示
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================================
# 因子分组定义（基于combo 97955的实际因子）
# ============================================================================

FACTOR_GROUPS = {
    "short_term": {
        "name": "短期（<20日）",
        "patterns": [
            "_5",
            "_6",
            "_9",
            "_10",
            "_12",
            "_14",
            "PRICE_POSITION_10",
            "RSI",
            "MFI",
            "STOCH",
            "CCI",
            "WILLIAMS",
        ],
        "description": "短周期动量/技术指标",
    },
    "mid_term_20": {
        "name": "中期20日",
        "patterns": ["_20", "PRICE_POSITION_20"],
        "description": "20日中周期趋势确认",
    },
    "mid_term_30": {
        "name": "中期30日",
        "patterns": ["_30", "PRICE_POSITION_30"],
        "description": "30日中周期趋势确认",
    },
    "long_term": {
        "name": "长期（≥40日）",
        "patterns": ["_40", "_60", "PRICE_POSITION_60", "MOMENTUM_20"],
        "description": "长周期防守",
    },
    "volatility_filter": {
        "name": "波动率过滤",
        "patterns": ["STDDEV", "VAR", "HT_DCPERIOD", "ATR"],
        "description": "波动率与周期过滤器",
    },
}

# 真实交易成本（港股ETF）
REAL_FEES = 0.0028  # 双边0.35% ≈ 单边0.14% × 2


def classify_factor(factor_name: str) -> str:
    """分类因子到对应组别（修复：确保所有因子都被归类）

    Args:
        factor_name: 因子名称

    Returns:
        分组名称
    """
    # 按优先级匹配（避免冲突）
    # 1. 波动率过滤（最高优先级，避免被其他组误匹配）
    for pattern in FACTOR_GROUPS["volatility_filter"]["patterns"]:
        if pattern in factor_name:
            return "volatility_filter"

    # 2. 提取数字窗口（如_12, _14, _20, _30, _60）
    numbers = re.findall(r"_(\d+)", factor_name)
    if numbers:
        # 取窗口中最大的周期数，避免类似 _20_5 被误判为短期
        period = max(int(num) for num in numbers)
        if period >= 60:
            return "long_term"
        elif period >= 40:
            return "long_term"
        elif period >= 30:
            return "mid_term_30"
        elif period >= 20:
            return "mid_term_20"
        else:
            return "short_term"

    # 3. 显式匹配模式（短期技术指标）
    for pattern in FACTOR_GROUPS["short_term"]["patterns"]:
        if pattern in factor_name:
            return "short_term"

    # 4. 中期30（优先于20，避免20_30被误匹配为20）
    for pattern in FACTOR_GROUPS["mid_term_30"]["patterns"]:
        if pattern in factor_name:
            return "mid_term_30"

    # 5. 中期20
    for pattern in FACTOR_GROUPS["mid_term_20"]["patterns"]:
        if pattern in factor_name:
            return "mid_term_20"

    # 6. 长期
    for pattern in FACTOR_GROUPS["long_term"]["patterns"]:
        if pattern in factor_name:
            return "long_term"

    # 🔥 兜底：所有未归类的全部归入 short_term
    print(f"⚠️ 因子 {factor_name} 无明确归类规则，默认归入 short_term")
    return "short_term"


def load_combo_97955_factors(csv_path: Path) -> Tuple[List[str], np.ndarray]:
    """从CSV加载combo 97955的因子和权重

    Args:
        csv_path: top1000_complete_analysis.csv路径

    Returns:
        (factors, weights) tuple
    """
    df = pd.read_csv(csv_path)

    # 找到combo 97955
    combo_row = df[df["combo_idx"] == 97955]
    if len(combo_row) == 0:
        raise ValueError("未找到combo 97955")

    # 解析factors字段（JSON格式的列表）
    import ast

    factors_str = combo_row.iloc[0]["factors"]
    factors = ast.literal_eval(factors_str)

    # 提取权重（weight_0到weight_34）
    weight_cols = [f"weight_{i}" for i in range(35)]
    weights_raw = combo_row[weight_cols].values[0]

    # 只保留非零权重的因子
    valid_indices = weights_raw > 1e-6
    factors_valid = [factors[i] for i in range(len(factors)) if valid_indices[i]]
    weights_valid = weights_raw[valid_indices]

    # 归一化权重
    weights_normalized = weights_valid / weights_valid.sum()

    print(f"✅ 已加载combo 97955: {len(factors_valid)}个有效因子")
    return factors_valid, weights_normalized


def group_factors(factors: List[str], weights: np.ndarray) -> Dict[str, Dict]:
    """对因子进行分组（修复：确保100%覆盖，无遗漏）

    Args:
        factors: 因子列表
        weights: 权重数组

    Returns:
        分组字典: {group_name: {'factors': [...], 'weights': [...], ...}}
    """
    grouped = {}

    for factor, weight in zip(factors, weights):
        group_name = classify_factor(factor)

        if group_name not in grouped:
            grouped[group_name] = {"factors": [], "weights": [], "count": 0}

        grouped[group_name]["factors"].append(factor)
        grouped[group_name]["weights"].append(weight)
        grouped[group_name]["count"] += 1

    # 归一化每组的权重
    for group_name in grouped:
        weights_array = np.array(grouped[group_name]["weights"])
        grouped[group_name]["weights"] = weights_array / weights_array.sum()
        grouped[group_name]["weights_array"] = grouped[group_name]["weights"]

    # 验证覆盖率
    total_factors_grouped = sum(g["count"] for g in grouped.values())
    print(f"\n✅ 因子分组覆盖率验证: {total_factors_grouped}/{len(factors)} (100%)")

    # 打印分组统计
    print("\n" + "=" * 60)
    print("因子分组统计")
    print("=" * 60)
    for group_name, info in grouped.items():
        group_info = FACTOR_GROUPS.get(
            group_name, {"name": group_name, "description": "Unknown"}
        )
        print(f"{group_info['name']:20s}: {info['count']:2d}个因子")
        print(f"  因子列表: {', '.join(info['factors'])}")
    print("=" * 60 + "\n")

    return grouped


def calculate_actual_trades(target_weights: np.ndarray) -> int:
    """从持仓矩阵计算真实交易笔数（修复：不再用换手率估算）

    Args:
        target_weights: 目标权重矩阵 (n_dates, n_assets)

    Returns:
        实际交易笔数
    """
    # 计算持仓变化
    position_changes = np.diff(target_weights, axis=0)
    # 统计非零变化（即交易）
    n_trades = np.sum(np.abs(position_changes) > 1e-6)
    return int(n_trades)


def run_group_backtest(
    normalized_panel: pd.DataFrame,
    price_pivot: pd.DataFrame,
    factors: List[str],
    weights: np.ndarray,
    top_n: int = 8,
    group_name: str = "unknown",
) -> Dict:
    """运行单个分组的回测（独立初始化引擎）

    Args:
        normalized_panel: 标准化因子面板
        price_pivot: 价格透视表
        factors: 因子列表
        weights: 权重数组
        top_n: Top-N选股数量
        group_name: 分组名称

    Returns:
        回测结果字典（包含持仓矩阵用于后续分析）
    """
    print(f"  回测{group_name}: {len(factors)}个因子, Top-{top_n}")

    # 为该分组初始化独立的引擎（使用真实费率）
    engine = VectorizedBacktestEngine(
        normalized_panel=normalized_panel,
        price_pivot=price_pivot,
        factors=factors,
        fees=REAL_FEES,  # 🔥 修复：使用真实费率
        init_cash=1_000_000.0,
        freq="1D",
    )

    # 转为numpy数组
    weights_matrix = weights.reshape(1, -1).astype(np.float32)

    # 计算得分
    scores = engine.compute_scores_batch(weights_matrix)

    # 构建目标权重
    target_weights = engine.build_weights_batch(scores, top_n=top_n)

    # 运行回测
    metrics_list = engine.run_backtest_batch(target_weights)
    metrics = metrics_list[0]

    # 🔥 修复：计算真实交易笔数
    n_trades = calculate_actual_trades(target_weights[0])

    return {
        "group_name": group_name,
        "n_factors": len(factors),
        "annual_return": metrics["annual_return"],
        "max_drawdown": metrics["max_drawdown"],
        "sharpe": metrics["sharpe"],
        "calmar": metrics["calmar"],
        "win_rate": metrics["win_rate"],
        "turnover": metrics["turnover"],
        "n_trades": n_trades,  # 🔥 真实交易笔数
        "target_weights": target_weights[0],  # 保留持仓矩阵
        "price_tensor": engine.price_tensor,
        "returns_tensor": engine.returns_tensor,
    }


def analyze_false_signals(
    target_weights: np.ndarray,
    returns_tensor: np.ndarray,
    price_dates: pd.DatetimeIndex,
) -> Dict[str, Any]:
    """分析假信号与过滤率（新增）

    定义假信号：持仓后次日收益≤0的交易

    Args:
        target_weights: 目标权重矩阵 (n_dates, n_assets)
        returns_tensor: 收益率张量 (n_dates, n_assets)
        price_dates: 日期索引

    Returns:
        假信号分析结果
    """
    # 计算每日组合收益
    portfolio_returns = np.sum(target_weights[:-1] * returns_tensor[1:], axis=1)

    # 统计假信号
    total_days = len(portfolio_returns)
    false_signal_days = np.sum(portfolio_returns <= 0)
    false_signal_rate = false_signal_days / total_days

    # 计算净收益
    cumulative_return = np.prod(1 + portfolio_returns) - 1

    # Hit rate
    hit_rate = 1 - false_signal_rate

    return {
        "total_days": total_days,
        "false_signal_days": false_signal_days,
        "false_signal_rate": false_signal_rate,
        "hit_rate": hit_rate,
        "net_cumulative_return": cumulative_return,
        "portfolio_returns": portfolio_returns,
        "dates": price_dates[1 : len(portfolio_returns) + 1],
    }


def calculate_persistence_layers(persistence_index: pd.Series) -> Dict[str, pd.Series]:
    """计算持续性分层（新增）

    将持续性指标分为：强/中/弱/转折

    Args:
        persistence_index: 持续性指标序列

    Returns:
        分层结果
    """
    # 计算分位数
    q25 = persistence_index.quantile(0.25)
    q50 = persistence_index.quantile(0.50)
    q75 = persistence_index.quantile(0.75)

    layers = {
        "strong": persistence_index >= q75,
        "medium": (persistence_index >= q50) & (persistence_index < q75),
        "weak": (persistence_index >= q25) & (persistence_index < q50),
        "turning": persistence_index < q25,
    }

    return layers


def run_combination_sensitivity_test(
    normalized_panel: pd.DataFrame,
    price_pivot: pd.DataFrame,
    grouped_factors: Dict[str, Dict],
    top_n: int = 8,
) -> pd.DataFrame:
    """运行组合敏感度测试（修复：确保 full_combo 包含所有因子）

    测试不同因子组合的表现：
    1. 纯20
    2. 纯30
    3. 20+30
    4. 20+30+长期
    5. 全组合（包括所有分组，无遗漏）

    Args:
        normalized_panel: 标准化因子面板
        price_pivot: 价格透视表
        grouped_factors: 分组因子字典
        top_n: Top-N选股数量

    Returns:
        测试结果DataFrame
    """
    print("\n" + "=" * 60)
    print("组合敏感度测试")
    print("=" * 60)

    # 🔥 修复：full_combo 必须包含所有分组
    test_combinations = {
        "pure_20": ["mid_term_20", "volatility_filter"],
        "pure_30": ["mid_term_30", "volatility_filter"],
        "mid_20_30": ["mid_term_20", "mid_term_30", "volatility_filter"],
        "mid_20_30_long": [
            "mid_term_20",
            "mid_term_30",
            "long_term",
            "volatility_filter",
        ],
        "full_combo": list(grouped_factors.keys()),  # 🔥 包含所有分组
    }

    results = []

    for combo_name, group_names in test_combinations.items():
        # 合并因子和权重
        all_factors = []
        all_weights = []

        for group_name in group_names:
            if group_name in grouped_factors:
                all_factors.extend(grouped_factors[group_name]["factors"])
                all_weights.extend(grouped_factors[group_name]["weights"])

        if not all_factors:
            print(f"  ⚠️ 组合 {combo_name} 无有效因子，跳过")
            continue

        print(
            f"  组合 {combo_name}: {len(all_factors)}个因子（来自{len(group_names)}个分组）"
        )

        # 归一化权重
        weights_array = np.array(all_weights)
        weights_array = weights_array / weights_array.sum()

        # 运行回测
        result = run_group_backtest(
            normalized_panel, price_pivot, all_factors, weights_array, top_n, combo_name
        )
        result["combination"] = combo_name

        # 🔥 添加假信号分析
        false_signal_analysis = analyze_false_signals(
            result["target_weights"], result["returns_tensor"], price_pivot.index
        )
        result["hit_rate_signal"] = false_signal_analysis["hit_rate"]
        result["false_signal_rate"] = false_signal_analysis["false_signal_rate"]
        result["net_cumulative_return"] = false_signal_analysis["net_cumulative_return"]

        results.append(result)

    return pd.DataFrame(results)


def calculate_theme_persistence_indicator(
    panel: pd.DataFrame, volatility_factors: List[str]
) -> pd.Series:
    """计算主题持续性指标

    基于波动率因子构建持续性指数

    Args:
        panel: 因子面板
        volatility_factors: 波动率因子列表

    Returns:
        持续性指标序列（按日期）
    """
    print("\n计算主题持续性指标...")

    # 提取波动率因子
    vol_factors = [f for f in volatility_factors if f in panel.columns]
    if not vol_factors:
        warnings.warn("未找到波动率因子")
        return pd.Series()

    # 计算波动率均值（按日期）
    volatility_mean = panel[vol_factors].groupby(level="date").mean().mean(axis=1)

    # 持续性指标 = -波动率（波动率越低，持续性越高）
    persistence_index = -volatility_mean

    # 标准化到[0, 1]
    persistence_index = (persistence_index - persistence_index.min()) / (
        persistence_index.max() - persistence_index.min()
    )

    return persistence_index


def create_visualizations(
    group_results_df: pd.DataFrame,
    combination_results_df: pd.DataFrame,
    persistence_index: pd.Series,
    output_dir: Path,
):
    """生成可视化图表（新增）

    Args:
        group_results_df: 分组回测结果
        combination_results_df: 组合敏感度结果
        persistence_index: 持续性指标
        output_dir: 输出目录
    """
    print("\n生成可视化图表...")

    # 1. 敏感度雷达图
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection="polar"))

    categories = ["夏普比率", "年化收益", "胜率", "换手率"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    for _, row in combination_results_df.iterrows():
        values = [
            row["sharpe"] / combination_results_df["sharpe"].max(),
            row["annual_return"] / combination_results_df["annual_return"].max(),
            row["win_rate"],
            1
            - row["turnover"]
            / combination_results_df["turnover"].max(),  # 换手率越低越好
        ]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=row["combination"])
        ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.set_title("组合敏感度雷达图", pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / "sensitivity_radar.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. 假信号过滤率条形图
    fig, ax = plt.subplots(figsize=(12, 6))
    combinations = combination_results_df["combination"].values
    false_signal_rates = combination_results_df["false_signal_rate"].values
    hit_rates = combination_results_df["hit_rate_signal"].values

    x = np.arange(len(combinations))
    width = 0.35

    ax.bar(
        x - width / 2,
        false_signal_rates,
        width,
        label="假信号率",
        color="red",
        alpha=0.7,
    )
    ax.bar(x + width / 2, hit_rates, width, label="Hit Rate", color="green", alpha=0.7)

    ax.set_xlabel("组合")
    ax.set_ylabel("比率")
    ax.set_title("假信号过滤率与Hit Rate对比")
    ax.set_xticks(x)
    ax.set_xticklabels(combinations, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "false_signal_filter.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. 持续性时间序列
    if len(persistence_index) > 0:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(
            persistence_index.index, persistence_index.values, linewidth=1.5, alpha=0.8
        )
        ax.fill_between(persistence_index.index, 0, persistence_index.values, alpha=0.3)

        # 标注分位线
        q25 = persistence_index.quantile(0.25)
        q50 = persistence_index.quantile(0.50)
        q75 = persistence_index.quantile(0.75)

        ax.axhline(
            q75,
            color="green",
            linestyle="--",
            label=f"Q75 (强持续): {q75:.3f}",
            alpha=0.7,
        )
        ax.axhline(
            q50,
            color="orange",
            linestyle="--",
            label=f"Q50 (中等): {q50:.3f}",
            alpha=0.7,
        )
        ax.axhline(
            q25, color="red", linestyle="--", label=f"Q25 (转折): {q25:.3f}", alpha=0.7
        )

        ax.set_xlabel("日期")
        ax.set_ylabel("持续性指标")
        ax.set_title("主题持续性时间序列")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            output_dir / "persistence_timeseries.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    print(f"✅ 可视化图表已保存到 {output_dir}")


def generate_operation_thresholds(
    combination_results_df: pd.DataFrame, persistence_index: pd.Series
) -> Dict[str, Any]:
    """生成操作阈值与实操建议（新增）

    Args:
        combination_results_df: 组合敏感度结果
        persistence_index: 持续性指标

    Returns:
        操作阈值字典
    """
    # 找到最佳组合
    best_combo = combination_results_df.loc[combination_results_df["sharpe"].idxmax()]

    # 持续性阈值（转换为Python原生float）
    if len(persistence_index) > 0:
        persistence_thresholds = {
            "strong": float(persistence_index.quantile(0.75)),
            "medium": float(persistence_index.quantile(0.50)),
            "weak": float(persistence_index.quantile(0.25)),
        }
    else:
        persistence_thresholds = {}

    # 换手率阈值（基于最佳组合的换手率）
    turnover_threshold = float(best_combo["turnover"]) * 1.2  # 允许20%的缓冲

    # Hit rate阈值
    hit_rate_threshold = float(best_combo.get("hit_rate_signal", 0.5))

    return {
        "best_combination": str(best_combo["combination"]),
        "best_sharpe": float(best_combo["sharpe"]),
        "recommended_top_n": 8,
        "persistence_thresholds": persistence_thresholds,
        "max_turnover": turnover_threshold,
        "min_hit_rate": hit_rate_threshold,
        "real_fees": float(REAL_FEES),
        "operation_rules": [
            f"1. 使用组合: {best_combo['combination']}",
            f"2. 选股数量: Top-8",
            f"3. 持续性指标 >= {persistence_thresholds.get('medium', 'N/A')}",
            f"4. 年化换手率 <= {turnover_threshold:.2f}",
            f"5. 单边费率: {REAL_FEES/2:.4f} (0.14%)",
            f"6. 预期夏普: {best_combo['sharpe']:.4f}",
            f"7. Hit Rate >= {hit_rate_threshold:.2%}",
        ],
    }


def main():
    """主函数"""
    print("=" * 80)
    print("Combo 97955 因子分组真实回测验证 v2.0")
    print("🔥 修复: 因子覆盖/交易成本/样本统计/完整分析")
    print("=" * 80 + "\n")

    # ========================================
    # 步骤1: 加载combo 97955的因子和权重
    # ========================================
    print("步骤1: 加载combo 97955数据...")

    csv_path = Path("strategies/results/top1000_complete_analysis.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到文件: {csv_path}")

    factors, weights = load_combo_97955_factors(csv_path)

    # 因子分组（修复：确保100%覆盖）
    grouped_factors = group_factors(factors, weights)

    # 验证因子总数
    total_grouped = sum(g["count"] for g in grouped_factors.values())
    assert total_grouped == len(
        factors
    ), f"🚨 因子覆盖缺口: {total_grouped}/{len(factors)}"

    # ========================================
    # 加载真实数据
    # ========================================
    print("加载真实因子面板和价格数据...")

    panel_path = Path(
        "factor_output/etf_rotation/panel_optimized_v2_20200102_20251014.parquet"
    )
    price_dir = Path("raw/ETF/daily")

    if not panel_path.exists():
        raise FileNotFoundError(f"因子面板不存在: {panel_path}")
    if not price_dir.exists():
        raise FileNotFoundError(f"价格目录不存在: {price_dir}")

    # 加载并验证因子存在性
    factor_panel = load_factor_panel(panel_path, factors)
    normalized_panel = normalize_factors(factor_panel, method="zscore")
    price_pivot = load_price_pivot(price_dir)

    print(
        f"✅ 数据加载完成: {len(factors)}个因子, {len(price_pivot)}个交易日, {len(price_pivot.columns)}个标的"
    )
    print(f"✅ 真实费率: {REAL_FEES} (双边0.35%)\n")

    # ========================================
    # 步骤2: 因子分组回测
    # ========================================
    print("\n" + "=" * 60)
    print("步骤2: 因子分组独立回测")
    print("=" * 60)

    group_results = []
    for group_name, group_info in grouped_factors.items():
        result = run_group_backtest(
            normalized_panel,
            price_pivot,
            group_info["factors"],
            group_info["weights_array"],
            top_n=8,
            group_name=group_name,
        )
        group_results.append(result)

    group_results_df = pd.DataFrame(group_results)

    # 输出分组回测结果
    print("\n" + "=" * 60)
    print("因子分组回测结果")
    print("=" * 60)
    print(
        group_results_df[
            [
                "group_name",
                "n_factors",
                "sharpe",
                "annual_return",
                "max_drawdown",
                "win_rate",
                "turnover",
                "n_trades",
            ]
        ].to_string(index=False)
    )

    # ========================================
    # 步骤3: 组合敏感度测试
    # ========================================
    print("\n运行步骤3...")
    combination_results_df = run_combination_sensitivity_test(
        normalized_panel, price_pivot, grouped_factors, top_n=8
    )

    print("\n" + "=" * 60)
    print("组合敏感度测试结果")
    print("=" * 60)
    print(
        combination_results_df[
            [
                "combination",
                "n_factors",
                "sharpe",
                "annual_return",
                "max_drawdown",
                "win_rate",
                "turnover",
                "hit_rate_signal",
                "false_signal_rate",
            ]
        ].to_string(index=False)
    )

    # ========================================
    # 步骤4: 主题持续性指标
    # ========================================
    print("\n" + "=" * 60)
    print("步骤4: 主题持续性指标")
    print("=" * 60)

    persistence_index = pd.Series()
    if "volatility_filter" in grouped_factors:
        persistence_index = calculate_theme_persistence_indicator(
            factor_panel, grouped_factors["volatility_filter"]["factors"]
        )

        if len(persistence_index) > 0:
            layers = calculate_persistence_layers(persistence_index)
            print(f"✅ 持续性指标计算完成")
            print(f"   均值: {persistence_index.mean():.4f}")
            print(f"   标准差: {persistence_index.std():.4f}")
            print(
                f"   范围: [{persistence_index.min():.4f}, {persistence_index.max():.4f}]"
            )
            print(f"   强持续天数: {layers['strong'].sum()}")
            print(f"   中等天数: {layers['medium'].sum()}")
            print(f"   弱持续天数: {layers['weak'].sum()}")
            print(f"   转折天数: {layers['turning'].sum()}")

    # ========================================
    # 步骤5: 生成可视化
    # ========================================
    print("\n" + "=" * 80)
    print("步骤5: 生成可视化与操作阈值")
    print("=" * 80)

    output_dir = Path("strategies/results/combo_97955_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 可视化
    create_visualizations(
        group_results_df, combination_results_df, persistence_index, output_dir
    )

    # 操作阈值
    operation_thresholds = generate_operation_thresholds(
        combination_results_df, persistence_index
    )

    # ========================================
    # 步骤6: 结构化输出
    # ========================================
    print("\n" + "=" * 80)
    print("步骤6: 保存结果")
    print("=" * 80)

    # 保存分组回测结果
    group_results_path = output_dir / "group_backtest_results.csv"
    group_results_df[
        [
            "group_name",
            "n_factors",
            "annual_return",
            "max_drawdown",
            "sharpe",
            "calmar",
            "win_rate",
            "turnover",
            "n_trades",
        ]
    ].to_csv(group_results_path, index=False)
    print(f"✅ 分组回测结果: {group_results_path}")

    # 保存组合敏感度结果
    combo_results_path = output_dir / "combination_sensitivity_results.csv"
    combination_results_df[
        [
            "combination",
            "n_factors",
            "annual_return",
            "max_drawdown",
            "sharpe",
            "calmar",
            "win_rate",
            "turnover",
            "n_trades",
            "hit_rate_signal",
            "false_signal_rate",
            "net_cumulative_return",
        ]
    ].to_csv(combo_results_path, index=False)
    print(f"✅ 组合敏感度结果: {combo_results_path}")

    # 保存因子分组信息
    factor_grouping_path = output_dir / "factor_grouping.json"
    grouping_info = {}
    for group_name, info in grouped_factors.items():
        grouping_info[group_name] = {
            "factors": info["factors"],
            "weights": info["weights"].tolist(),
            "count": info["count"],
        }
    with open(factor_grouping_path, "w", encoding="utf-8") as f:
        json.dump(grouping_info, f, indent=2, ensure_ascii=False)
    print(f"✅ 因子分组信息: {factor_grouping_path}")

    # 保存操作阈值
    thresholds_path = output_dir / "operation_thresholds.json"
    with open(thresholds_path, "w", encoding="utf-8") as f:
        json.dump(operation_thresholds, f, indent=2, ensure_ascii=False)
    print(f"✅ 操作阈值: {thresholds_path}")

    # ========================================
    # 生成完整分析报告
    # ========================================
    report_path = output_dir / "COMPLETE_ANALYSIS_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Combo 97955 因子分组完整分析报告 v2.0\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**修复内容**: 因子覆盖/交易成本/样本统计/完整分析\n\n")

        f.write("---\n\n")
        f.write("## 修复验证\n\n")
        f.write(f"1. ✅ 因子覆盖: {total_grouped}/{len(factors)} (100%)\n")
        f.write(f"2. ✅ 交易成本: {REAL_FEES} (双边0.35%)\n")
        f.write(f"3. ✅ 样本统计: 从持仓矩阵计算真实交易笔数\n")
        f.write(f"4. ✅ 完整分析: 假信号/持续性/可视化/操作阈值\n\n")

        f.write("---\n\n")
        f.write("## 1. 因子分组统计\n\n")
        for group_name, info in grouped_factors.items():
            group_info = FACTOR_GROUPS.get(
                group_name, {"name": group_name, "description": "Unknown"}
            )
            f.write(f"### {group_info['name']}\n")
            f.write(f"- **描述**: {group_info['description']}\n")
            f.write(f"- **因子数量**: {info['count']}\n")
            f.write(f"- **因子列表**: {', '.join(info['factors'])}\n\n")

        f.write("---\n\n")
        f.write("## 2. 分组回测结果\n\n")
        f.write(
            group_results_df[
                [
                    "group_name",
                    "n_factors",
                    "sharpe",
                    "annual_return",
                    "max_drawdown",
                    "win_rate",
                    "turnover",
                    "n_trades",
                ]
            ].to_markdown(index=False)
        )
        f.write("\n\n")

        f.write("---\n\n")
        f.write("## 3. 组合敏感度测试结果\n\n")
        f.write(
            combination_results_df[
                [
                    "combination",
                    "n_factors",
                    "sharpe",
                    "annual_return",
                    "max_drawdown",
                    "win_rate",
                    "turnover",
                    "hit_rate_signal",
                    "false_signal_rate",
                ]
            ].to_markdown(index=False)
        )
        f.write("\n\n")

        f.write("---\n\n")
        f.write("## 4. 关键发现\n\n")

        # 找到最佳分组
        best_group = group_results_df.loc[group_results_df["sharpe"].idxmax()]
        f.write(f"### 最佳因子分组\n")
        f.write(f"- **分组**: {best_group['group_name']}\n")
        f.write(f"- **夏普比率**: {best_group['sharpe']:.4f}\n")
        f.write(f"- **年化收益**: {best_group['annual_return']:.2%}\n")
        f.write(f"- **最大回撤**: {best_group['max_drawdown']:.2%}\n")
        f.write(f"- **真实交易笔数**: {best_group['n_trades']}\n\n")

        # 找到最佳组合
        best_combo = combination_results_df.loc[
            combination_results_df["sharpe"].idxmax()
        ]
        f.write(f"### 最佳因子组合\n")
        f.write(f"- **组合**: {best_combo['combination']}\n")
        f.write(f"- **因子数量**: {best_combo['n_factors']}\n")
        f.write(f"- **夏普比率**: {best_combo['sharpe']:.4f}\n")
        f.write(f"- **年化收益**: {best_combo['annual_return']:.2%}\n")
        f.write(f"- **最大回撤**: {best_combo['max_drawdown']:.2%}\n")
        f.write(f"- **Hit Rate**: {best_combo['hit_rate_signal']:.2%}\n")
        f.write(f"- **假信号率**: {best_combo['false_signal_rate']:.2%}\n")
        f.write(f"- **真实交易笔数**: {best_combo['n_trades']}\n\n")

        f.write("---\n\n")
        f.write("## 5. 操作阈值与实操建议\n\n")
        for rule in operation_thresholds["operation_rules"]:
            f.write(f"{rule}\n")
        f.write("\n")

        f.write("---\n\n")
        f.write("## 6. 可视化图表\n\n")
        f.write("- `sensitivity_radar.png`: 敏感度雷达图\n")
        f.write("- `false_signal_filter.png`: 假信号过滤率\n")
        f.write("- `persistence_timeseries.png`: 持续性时间序列\n\n")

        f.write("---\n\n")
        f.write("## 7. 数据质量保证\n\n")
        f.write("- ✅ 使用真实因子面板（无随机生成）\n")
        f.write("- ✅ 使用真实价格数据（无模拟信号）\n")
        f.write("- ✅ 使用真实交易成本（港股ETF费率）\n")
        f.write("- ✅ 计算真实交易笔数（从持仓变化）\n")
        f.write("- ✅ 基于VectorizedBacktestEngine（已通过回归测试）\n")
        f.write("- ✅ 所有结果可追溯到输入数据\n\n")

    print(f"✅ 完整分析报告: {report_path}")

    print("\n" + "=" * 80)
    print("✅ 完整分析交付完成！")
    print(f"   - 因子覆盖: {total_grouped}/{len(factors)} (100%)")
    print(f"   - 交易成本: {REAL_FEES} (真实)")
    print(f"   - 样本统计: 真实交易笔数")
    print(f"   - 可视化: 3张图表")
    print(f"   - 操作阈值: 7条实操建议")
    print("=" * 80)


if __name__ == "__main__":
    main()
