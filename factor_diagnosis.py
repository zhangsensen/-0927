#!/usr/bin/env python3
"""
因子诊断脚本：检查各因子的IC和收益贡献在新期间的变化

比较训练集 (2020-01-01至2025-05-31) vs Holdout期 (2025-06-01至2025-12-08)
"""

import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from datetime import datetime

ROOT = Path(__file__).parent

# 导入核心模块
from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor


def calculate_ic(factor_scores: np.ndarray, future_returns: np.ndarray) -> float:
    """计算IC值"""
    valid_mask = ~(np.isnan(factor_scores) | np.isnan(future_returns))
    if np.sum(valid_mask) < 10:
        return np.nan
    return np.corrcoef(factor_scores[valid_mask], future_returns[valid_mask])[0, 1]


def main():
    print("🔍 因子诊断：比较训练集 vs Holdout期")
    print("=" * 80)

    # 加载配置
    config_path = ROOT / "configs/combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 时间段
    train_start = "2020-01-01"
    train_end = "2025-05-31"
    holdout_start = "2025-06-01"
    holdout_end = "2025-12-08"

    print(f"📅 训练集: {train_start} → {train_end}")
    print(f"📅 Holdout期: {holdout_start} → {holdout_end}")

    # 加载数据
    data_loader = DataLoader(
        data_dir=config["data"]["data_dir"],
        cache_dir=config["data"]["cache_dir"],
    )

    print("\n📊 加载训练集数据...")
    train_ohlcv = data_loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=train_start,
        end_date=train_end,
    )

    print("📊 加载Holdout期数据...")
    holdout_ohlcv = data_loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=holdout_start,
        end_date=holdout_end,
    )

    # 计算因子
    print("\n🔧 计算训练集因子...")
    factor_lib = PreciseFactorLibrary()
    train_factors = factor_lib.compute_all_factors(train_ohlcv)

    print("🔧 计算Holdout期因子...")
    holdout_factors = factor_lib.compute_all_factors(holdout_ohlcv)

    # 跳过标准化，直接使用原始因子计算IC (简化版)
    print("\n📊 计算因子IC (使用原始因子)...")

    # 计算未来收益 (5日后收益)
    print("\n📈 计算未来收益...")

    def calc_future_returns(close_df: pd.DataFrame, periods: int = 5) -> pd.DataFrame:
        future_close = close_df.shift(-periods)
        returns = (future_close - close_df) / close_df
        return returns

    train_returns = calc_future_returns(train_ohlcv["close"])
    holdout_returns = calc_future_returns(holdout_ohlcv["close"])

    # 计算各因子IC
    print("\n📊 计算因子IC...")
    factor_names = sorted(
        list(
            set(
                col.split("_")[0] + "_" + col.split("_")[1]
                for col in train_factors.columns
                if "_" in col
            )
        )
    )

    results = []

    for factor in factor_names:
        factor_cols = [
            col for col in train_factors.columns if col.startswith(factor + "_")
        ]

        if not factor_cols:
            continue

        # 训练集IC
        train_ics = []
        for col in factor_cols:
            etf = col.split("_", 2)[2]  # ETF代码
            if etf in train_returns.columns:
                ic = calculate_ic(train_factors[col].values, train_returns[etf].values)
                if not np.isnan(ic):
                    train_ics.append(ic)

        train_ic_mean = np.mean(train_ics) if train_ics else np.nan
        train_ic_std = np.std(train_ics) if train_ics else np.nan

        # Holdout期IC
        holdout_ics = []
        for col in factor_cols:
            etf = col.split("_", 2)[2]
            if etf in holdout_returns.columns:
                ic = calculate_ic(
                    holdout_factors[col].values, holdout_returns[etf].values
                )
                if not np.isnan(ic):
                    holdout_ics.append(ic)

        holdout_ic_mean = np.mean(holdout_ics) if holdout_ics else np.nan
        holdout_ic_std = np.std(holdout_ics) if holdout_ics else np.nan

        results.append(
            {
                "factor": factor,
                "train_ic_mean": train_ic_mean,
                "train_ic_std": train_ic_std,
                "holdout_ic_mean": holdout_ic_mean,
                "holdout_ic_std": holdout_ic_std,
                "ic_decay": (
                    holdout_ic_mean - train_ic_mean
                    if not (np.isnan(holdout_ic_mean) or np.isnan(train_ic_mean))
                    else np.nan
                ),
                "ic_stability": (
                    abs(holdout_ic_mean / train_ic_mean)
                    if train_ic_mean != 0
                    and not np.isnan(holdout_ic_mean)
                    and not np.isnan(train_ic_mean)
                    else np.nan
                ),
            }
        )

    # 输出结果
    print("\n" + "=" * 80)
    print("因子IC对比 (训练集 vs Holdout期)")
    print("=" * 80)
    print("<10")
    print("-" * 80)

    for r in sorted(
        results,
        key=lambda x: x["ic_decay"] if not np.isnan(x["ic_decay"]) else -999,
        reverse=True,
    ):
        print("<10.4f")

    # 分析
    print("\n🔍 分析:")
    stable_factors = [
        r
        for r in results
        if not np.isnan(r["ic_stability"]) and r["ic_stability"] > 0.5
    ]
    decayed_factors = [
        r for r in results if not np.isnan(r["ic_decay"]) and r["ic_decay"] < -0.1
    ]

    print(
        f"✅ 相对稳定的因子 ({len(stable_factors)}个): {[r['factor'] for r in stable_factors]}"
    )
    print(
        f"❌ 显著衰减的因子 ({len(decayed_factors)}个): {[r['factor'] for r in decayed_factors]}"
    )

    if decayed_factors:
        print("\n⚠️ 建议: 考虑降低衰减因子的权重，或寻找替代因子")


if __name__ == "__main__":
    main()
