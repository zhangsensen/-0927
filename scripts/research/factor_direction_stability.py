"""
因子方向稳定性研究 — Phase 0 验证脚本

目的：
1. 计算所有因子在 Train 和 Holdout 期间的 IC
2. 统计方向一致性（sign(train_ic) == sign(ho_ic) 的比例）
3. 模拟翻转负 IC 因子后的 Holdout 表现
4. 输出分析报告，决定是否值得实施方向处理

用法：
    uv run python scripts/research/factor_direction_stability.py

输出：
    results/factor_direction_stability_YYYYMMDD_HHMMSS/
    ├── direction_analysis.csv      # 详细结果
    ├── direction_summary.md        # 汇总报告
    └── flip_simulation.csv         # 翻转模拟结果

作者: Claude Code
日期: 2026-02-14
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.factor_cache import FactorCache
from etf_strategy.core.factor_mining.quality import (
    compute_forward_returns,
    spearman_ic_series,
)
from etf_strategy.core.factor_registry import FACTOR_SPECS


# =============================================================================
# 配置
# =============================================================================

TRAIN_END_DATE = "2025-04-30"  # Train 截止日期
HO_START_DATE = "2025-05-01"  # Holdout 开始日期
FWD_PERIODS = 5  # 前向收益周期（与 FREQ=5 一致）
MIN_IC_THRESHOLD = 0.02  # 最小 IC 阈值（低于此值方向无意义）
MIN_ICIR_THRESHOLD = 0.1  # 最小 ICIR 阈值（方向稳定性要求）


# =============================================================================
# 核心函数
# =============================================================================


def load_data(config: dict) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, List[str]]:
    """
    加载因子数据和前向收益

    返回:
        std_factors: dict[因子名, DataFrame] - 标准化因子
        fwd_ret: DataFrame - 前向收益
        etf_codes: list[str] - ETF 代码列表
    """
    print("加载数据...")

    # 加载 OHLCV
    data_dir = Path(config["data"]["data_dir"])
    loader = DataLoader(data_dir=str(data_dir))

    # 获取所有 ETF 代码
    etf_codes = config["data"]["symbols"]
    ohlcv = loader.load_ohlcv(etf_codes=etf_codes)

    # 获取标准化因子（通过缓存）
    cache_dir = Path(config["data"]["cache_dir"])
    factor_cache = FactorCache(cache_dir=cache_dir)

    cached = factor_cache.get_or_compute(
        ohlcv=ohlcv,
        config=config,
        data_dir=data_dir,
    )

    std_factors = cached["std_factors"]
    dates = cached["dates"]
    etf_codes = cached["etf_codes"]

    # 计算前向收益
    close = ohlcv["close"]
    fwd_ret = compute_forward_returns(close, FWD_PERIODS)

    # 对齐日期
    common_dates = dates.intersection(fwd_ret.index)
    fwd_ret = fwd_ret.loc[common_dates]

    print(f"  因子数量: {len(std_factors)}")
    print(f"  ETF 数量: {len(etf_codes)}")
    print(f"  日期范围: {dates.min()} ~ {dates.max()}")
    print(f"  交易日数: {len(dates)}")

    return std_factors, fwd_ret, etf_codes


def compute_split_ic(
    factor_df: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    train_end: str,
    ho_start: str,
) -> Dict[str, float]:
    """
    计算 Train 和 Holdout 分别的 IC

    返回:
        dict with keys: train_ic, ho_ic, train_icir, ho_icir, train_n, ho_n
    """
    # 计算全量 IC 序列
    ic_series = spearman_ic_series(factor_df, fwd_ret)

    if len(ic_series) == 0:
        return {
            "train_ic": np.nan,
            "ho_ic": np.nan,
            "train_icir": np.nan,
            "ho_icir": np.nan,
            "train_n": 0,
            "ho_n": 0,
        }

    # 确保索引为字符串格式便于比较
    ic_index = ic_series.index
    if isinstance(ic_index, pd.DatetimeIndex):
        ic_index_str = ic_index.strftime("%Y-%m-%d")
    else:
        ic_index_str = pd.to_datetime(ic_index).strftime("%Y-%m-%d")

    ic_series.index = ic_index_str

    # 分割
    train_mask = ic_series.index <= train_end
    ho_mask = ic_series.index >= ho_start

    train_ic_series = ic_series[train_mask]
    ho_ic_series = ic_series[ho_mask]

    # 计算统计量
    train_ic = train_ic_series.mean() if len(train_ic_series) > 0 else np.nan
    ho_ic = ho_ic_series.mean() if len(ho_ic_series) > 0 else np.nan

    train_std = train_ic_series.std() if len(train_ic_series) > 1 else np.nan
    ho_std = ho_ic_series.std() if len(ho_ic_series) > 1 else np.nan

    train_icir = abs(train_ic / train_std) if train_std and train_std > 0 else np.nan
    ho_icir = abs(ho_ic / ho_std) if ho_std and ho_std > 0 else np.nan

    return {
        "train_ic": train_ic,
        "ho_ic": ho_ic,
        "train_icir": train_icir,
        "ho_icir": ho_icir,
        "train_n": len(train_ic_series),
        "ho_n": len(ho_ic_series),
    }


def analyze_direction_stability(
    std_factors: Dict[str, pd.DataFrame],
    fwd_ret: pd.DataFrame,
    train_end: str = TRAIN_END_DATE,
    ho_start: str = HO_START_DATE,
) -> pd.DataFrame:
    """
    分析所有因子的方向稳定性

    返回:
        DataFrame with columns:
        - factor_name: 因子名
        - source: 数据源
        - train_ic, ho_ic: Train/Holdout IC
        - train_icir, ho_icir: Train/Holdout ICIR
        - direction: "positive" / "negative" / "neutral"
        - direction_stable: 方向是否稳定
        - flip_recommended: 是否推荐翻转
    """
    print("\n分析因子方向稳定性...")

    results = []

    for factor_name, factor_df in std_factors.items():
        # 获取因子元数据
        spec = FACTOR_SPECS.get(factor_name)
        source = spec.source if spec else "unknown"

        # 计算分割 IC
        ic_stats = compute_split_ic(factor_df, fwd_ret, train_end, ho_start)

        train_ic = ic_stats["train_ic"]
        ho_ic = ic_stats["ho_ic"]
        train_icir = ic_stats["train_icir"]
        ho_icir = ic_stats["ho_icir"]

        # 判断方向
        if abs(train_ic) < MIN_IC_THRESHOLD:
            direction = "neutral"
        elif train_ic > 0:
            direction = "positive"
        else:
            direction = "negative"

        # 方向稳定性：Train 和 HO IC 同号
        if np.isnan(train_ic) or np.isnan(ho_ic):
            direction_stable = False
        elif abs(train_ic) < MIN_IC_THRESHOLD or abs(ho_ic) < MIN_IC_THRESHOLD:
            direction_stable = True  # IC 太弱，方向无意义，视为稳定
        else:
            direction_stable = (train_ic * ho_ic) > 0

        # 翻转推荐：
        # 1. Train IC 显著（|IC| > 0.02 且 ICIR > 0.1）
        # 2. Train IC 为负
        # 3. 方向稳定
        # 4. HO IC 也显著（避免翻转后无效果）
        train_significant = (
            abs(train_ic) >= MIN_IC_THRESHOLD
            and not np.isnan(train_icir)
            and train_icir >= MIN_ICIR_THRESHOLD
        )
        ho_significant = abs(ho_ic) >= MIN_IC_THRESHOLD

        flip_recommended = (
            train_significant
            and direction == "negative"
            and direction_stable
            and ho_significant
        )

        results.append(
            {
                "factor_name": factor_name,
                "source": source,
                "train_ic": train_ic,
                "ho_ic": ho_ic,
                "train_icir": train_icir,
                "ho_icir": ho_icir,
                "train_n": ic_stats["train_n"],
                "ho_n": ic_stats["ho_n"],
                "direction": direction,
                "direction_stable": direction_stable,
                "flip_recommended": flip_recommended,
            }
        )

    df = pd.DataFrame(results)
    df = df.sort_values("train_ic", key=abs, ascending=False)

    return df


def simulate_flip_impact(
    std_factors: Dict[str, pd.DataFrame],
    fwd_ret: pd.DataFrame,
    analysis_df: pd.DataFrame,
    ho_start: str = HO_START_DATE,
) -> pd.DataFrame:
    """
    模拟翻转负 IC 因子后的 Holdout 表现

    对比：
    1. 原始因子 HO IC
    2. 翻转后 HO IC（如果 train_ic < 0，则翻转因子值）

    返回:
        DataFrame with columns:
        - factor_name
        - original_ho_ic
        - flipped_ho_ic
        - improvement: 翻转后的 IC 改善
    """
    print("\n模拟翻转效果...")

    results = []

    # 只对 Train IC 为负的因子模拟
    negative_factors = analysis_df[analysis_df["direction"] == "negative"][
        "factor_name"
    ].tolist()

    if not negative_factors:
        print("  没有负 IC 因子，跳过模拟")
        return pd.DataFrame()

    for factor_name in negative_factors:
        factor_df = std_factors[factor_name]

        # 原始 HO IC
        ic_series = spearman_ic_series(factor_df, fwd_ret)
        if len(ic_series) == 0:
            continue

        ic_index = ic_series.index
        if isinstance(ic_index, pd.DatetimeIndex):
            ic_index_str = ic_index.strftime("%Y-%m-%d")
        else:
            ic_index_str = pd.to_datetime(ic_index).strftime("%Y-%m-%d")
        ic_series.index = ic_index_str

        ho_mask = ic_series.index >= ho_start
        original_ho_ic = ic_series[ho_mask].mean() if ho_mask.any() else np.nan

        # 翻转因子值后的 HO IC
        flipped_factor_df = -factor_df
        flipped_ic_series = spearman_ic_series(flipped_factor_df, fwd_ret)
        if len(flipped_ic_series) == 0:
            continue

        flipped_ic_series.index = ic_index_str  # 复用索引
        flipped_ho_ic = flipped_ic_series[ho_mask].mean() if ho_mask.any() else np.nan

        improvement = (
            flipped_ho_ic - original_ho_ic
            if not np.isnan(flipped_ho_ic) and not np.isnan(original_ho_ic)
            else np.nan
        )

        results.append(
            {
                "factor_name": factor_name,
                "original_ho_ic": original_ho_ic,
                "flipped_ho_ic": flipped_ho_ic,
                "improvement": improvement,
            }
        )

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("improvement", ascending=False)

    return df


def generate_report(
    analysis_df: pd.DataFrame,
    flip_df: pd.DataFrame,
    output_dir: Path,
) -> str:
    """
    生成汇总报告
    """
    total_factors = len(analysis_df)
    positive_factors = len(analysis_df[analysis_df["direction"] == "positive"])
    negative_factors = len(analysis_df[analysis_df["direction"] == "negative"])
    neutral_factors = len(analysis_df[analysis_df["direction"] == "neutral"])

    stable_count = analysis_df["direction_stable"].sum()
    stable_rate = stable_count / total_factors * 100 if total_factors > 0 else 0

    flip_recommended_count = analysis_df["flip_recommended"].sum()

    # 计算方向一致性（正/负因子中 Train/HO 同号的比例）
    significant_factors = analysis_df[
        (analysis_df["direction"] != "neutral")
        & (analysis_df["train_n"] > 0)
        & (analysis_df["ho_n"] > 0)
    ]
    if len(significant_factors) > 0:
        consistent_count = significant_factors["direction_stable"].sum()
        consistency_rate = consistent_count / len(significant_factors) * 100
    else:
        consistency_rate = 0

    report = f"""# 因子方向稳定性分析报告

**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Train 截止**: {TRAIN_END_DATE}
**Holdout 开始**: {HO_START_DATE}

---

## 1. 总体统计

| 指标 | 数值 |
|------|------|
| 总因子数 | {total_factors} |
| 正向因子 (train_ic > 0) | {positive_factors} ({positive_factors / total_factors * 100:.1f}%) |
| 负向因子 (train_ic < 0) | {negative_factors} ({negative_factors / total_factors * 100:.1f}%) |
| 中性因子 (|train_ic| < {MIN_IC_THRESHOLD}) | {neutral_factors} |

---

## 2. 方向稳定性

| 指标 | 数值 |
|------|------|
| 方向一致因子数 | {stable_count} |
| 方向一致率 | {stable_rate:.1f}% |
| **显著因子方向一致率** | **{consistency_rate:.1f}%** |

**判断标准**: sign(train_ic) == sign(ho_ic)

"""

    # 关键决策点
    if consistency_rate >= 70:
        decision = "✅ **通过** — 方向稳定性良好，建议进入 Phase 1 实施"
    elif consistency_rate >= 60:
        decision = "⚠️ **边界** — 方向稳定性中等，需进一步分析具体因子"
    else:
        decision = "❌ **不通过** — 方向不稳定，不建议实施方向处理"

    report += f"""
---

## 3. 决策

{decision}

"""

    # 翻转模拟结果
    if not flip_df.empty:
        avg_improvement = flip_df["improvement"].mean()
        positive_improvement = (flip_df["improvement"] > 0).sum()
        total_negative = len(flip_df)

        report += f"""---

## 4. 翻转模拟结果（负 IC 因子）

| 指标 | 数值 |
|------|------|
| 负 IC 因子数 | {total_negative} |
| 翻转后 IC 改善数 | {positive_improvement} ({positive_improvement / total_negative * 100:.1f}%) |
| 平均 IC 改善 | {avg_improvement:+.4f} |

### 翻转推荐因子

| 因子 | 原 HO IC | 翻转后 HO IC | 改善 |
|------|---------|-------------|------|
"""
        recommended = analysis_df[analysis_df["flip_recommended"] == True]
        for _, row in recommended.iterrows():
            fn = row["factor_name"]
            flip_row = flip_df[flip_df["factor_name"] == fn]
            if not flip_row.empty:
                orig = flip_row.iloc[0]["original_ho_ic"]
                flipped = flip_row.iloc[0]["flipped_ho_ic"]
                imp = flip_row.iloc[0]["improvement"]
                report += f"| {fn} | {orig:+.4f} | {flipped:+.4f} | {imp:+.4f} |\n"

    # 详细因子表
    report += """
---

## 5. 所有因子详情

| 因子 | 来源 | Train IC | HO IC | 方向 | 稳定? | 翻转? |
|------|------|---------|-------|------|-------|-------|
"""
    for _, row in analysis_df.head(30).iterrows():
        stable_mark = "✓" if row["direction_stable"] else "✗"
        flip_mark = "✓" if row["flip_recommended"] else ""
        report += f"| {row['factor_name']} | {row['source']} | {row['train_ic']:+.4f} | {row['ho_ic']:+.4f} | {row['direction']} | {stable_mark} | {flip_mark} |\n"

    if len(analysis_df) > 30:
        report += f"\n*... 共 {len(analysis_df)} 个因子，仅显示前 30 个*\n"

    return report


# =============================================================================
# 主函数
# =============================================================================


def main():
    print("=" * 60)
    print("因子方向稳定性研究 — Phase 0 验证")
    print("=" * 60)

    # 加载配置
    config_path = PROJECT_ROOT / "configs" / "combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "results" / f"factor_direction_stability_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    std_factors, fwd_ret, etf_codes = load_data(config)

    # 2. 分析方向稳定性
    analysis_df = analyze_direction_stability(std_factors, fwd_ret)

    # 3. 模拟翻转效果
    flip_df = simulate_flip_impact(std_factors, fwd_ret, analysis_df)

    # 4. 生成报告
    report = generate_report(analysis_df, flip_df, output_dir)

    # 5. 保存结果
    analysis_df.to_csv(output_dir / "direction_analysis.csv", index=False)
    if not flip_df.empty:
        flip_df.to_csv(output_dir / "flip_simulation.csv", index=False)

    with open(output_dir / "direction_summary.md", "w") as f:
        f.write(report)

    print(f"\n结果已保存到: {output_dir}")
    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)

    return analysis_df, flip_df


if __name__ == "__main__":
    main()
