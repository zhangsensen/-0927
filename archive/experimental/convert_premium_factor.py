#!/usr/bin/env python3
"""
将折溢价因子转换为WFO标准格式
与其他17个因子对齐时间范围和symbol列表
"""

import pandas as pd
import numpy as np
from pathlib import Path


def convert_premium_to_standard_format(
    premium_dir: Path,
    reference_factor_path: Path,
    output_dir: Path,
    factor_name: str = "PREMIUM_DEVIATION_20D",
) -> None:
    """
    将折溢价因子转换为标准格式

    Args:
        premium_dir: 原始折溢价数据目录
        reference_factor_path: 参考因子文件路径（用于对齐时间范围和symbol）
        output_dir: 输出目录
        factor_name: 输出因子名称
    """
    print(f"[{factor_name}] 开始转换...")

    # 1. 加载参考因子，获取完整时间范围和symbol列表
    ref_df = pd.read_parquet(reference_factor_path)
    full_dates = ref_df.index
    symbols = ref_df.columns.tolist()
    print(f"      参考因子: {len(full_dates)} 天 x {len(symbols)} symbols")
    print(f"      时间范围: {full_dates.min()} 至 {full_dates.max()}")

    # 2. 加载所有折溢价数据
    premium_files = list(premium_dir.glob("premium_rate_*.parquet"))
    print(f"      找到 {len(premium_files)} 个折溢价文件")

    all_premium = []
    for f in premium_files:
        symbol = f.stem.replace("premium_rate_", "")
        if symbol in symbols:  # 只保留配置中的symbol
            df = pd.read_parquet(f)
            df["symbol"] = symbol
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            all_premium.append(df[["trade_date", "symbol", "premium_rate"]])

    if not all_premium:
        print("      ⚠️ 没有匹配的symbol数据")
        return

    # 3. 合并并计算偏离度
    combined = pd.concat(all_premium, ignore_index=True)
    print(f"      原始数据: {len(combined)} 行")

    # 4. 计算20日偏离度
    def calculate_deviation(group):
        group = group.sort_values("trade_date")
        group["rolling_mean"] = (
            group["premium_rate"].rolling(window=20, min_periods=10).mean()
        )
        group["rolling_std"] = (
            group["premium_rate"].rolling(window=20, min_periods=10).std()
        )
        group["deviation"] = (group["premium_rate"] - group["rolling_mean"]) / (
            group["rolling_std"] + 1e-8
        )
        return group

    combined = (
        combined.groupby("symbol").apply(calculate_deviation).reset_index(drop=True)
    )
    print(f"      计算偏离度完成")

    # 5. 转换为宽格式
    pivot_df = combined.pivot(index="trade_date", columns="symbol", values="deviation")
    pivot_df.index = pd.to_datetime(pivot_df.index)
    print(f"      宽格式: {pivot_df.shape}")

    # 6. 对齐到完整时间范围（填充NaN）
    aligned_df = pivot_df.reindex(full_dates)

    # 确保所有symbol都存在（缺失的symbol填充NaN）
    for sym in symbols:
        if sym not in aligned_df.columns:
            aligned_df[sym] = np.nan

    aligned_df = aligned_df[symbols]  # 按参考顺序排列

    print(f"      对齐后: {aligned_df.shape}")
    print(
        f"      有效数据: {aligned_df.count().sum()} / {aligned_df.size} ({aligned_df.count().sum() / aligned_df.size * 100:.1f}%)"
    )

    # 7. 保存
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{factor_name}.parquet"
    aligned_df.to_parquet(output_path)
    print(f"      ✓ 已保存: {output_path}")

    return aligned_df


if __name__ == "__main__":
    PREMIUM_DIR = Path("raw/ETF/factors")
    REF_FACTOR = Path("results/run_20260212_005855/factors/MOM_20D.parquet")
    OUTPUT_DIR = Path("results/run_20260212_005855/factors")

    df = convert_premium_to_standard_format(PREMIUM_DIR, REF_FACTOR, OUTPUT_DIR)

    # 验证数据
    print("\n" + "=" * 60)
    print("数据验证")
    print("=" * 60)
    print(f"日期范围: {df.index.min()} 至 {df.index.max()}")
    print(f"数据点统计:")
    print(f"  - 总数据点: {df.size}")
    print(f"  - 有效数据: {df.count().sum()}")
    print(f"  - 缺失比例: {(1 - df.count().sum() / df.size) * 100:.1f}%")
    print(f"  - 每个symbol平均有效天数: {df.count().mean():.0f}")
    print("\n前5行预览:")
    print(df.head())
