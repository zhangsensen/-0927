#!/usr/bin/env python3
"""
简化版因子筛选脚本 - 针对ETF数据
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def calculate_correlation(factor_data, returns_data):
    """计算因子与未来收益的相关系数"""
    correlations = []

    # 确保日期对齐
    common_dates = factor_data.index.intersection(returns_data.index)

    for date in common_dates:
        if date in factor_data.index and date in returns_data.index:
            factor_vals = factor_data.loc[date].dropna()
            return_vals = returns_data.loc[date].dropna()

            # 找到共同的股票
            common_stocks = factor_vals.index.intersection(return_vals.index)
            if len(common_stocks) >= 3:  # 至少3个股票
                f_vals = factor_vals.loc[common_stocks]
                r_vals = return_vals.loc[common_stocks]

                # 计算相关系数
                if len(f_vals.unique()) > 1 and len(r_vals.unique()) > 1:
                    corr = np.corrcoef(f_vals, r_vals)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)

    return correlations


def main():
    # 数据路径
    factor_path = "/Users/zhangshenshen/深度量化0927/factor_output/etf_rotation_production_fixed/panel_FULL_20200102_20251014.parquet"
    price_dir = "/Users/zhangshenshen/深度量化0927/raw/ETF/daily"

    print("📂 加载因子数据...")
    factor_panel = pd.read_parquet(factor_path)
    print(f"   因子面板形状: {factor_panel.shape}")

    print("📈 加载价格数据...")
    # 加载价格数据
    price_files = list(Path(price_dir).glob("*.parquet"))
    if not price_files:
        print("❌ 未找到价格数据文件")
        return

    price_data = []
    for file in price_files[:10]:  # 限制文件数量避免内存问题
        df = pd.read_parquet(file)
        symbol = file.stem.split("_")[0]
        if "trade_date" in df.columns and "close" in df.columns:
            df["date"] = pd.to_datetime(df["trade_date"])
            df["symbol"] = symbol
            price_data.append(df[["date", "symbol", "close"]])

    if not price_data:
        print("❌ 价格数据格式错误")
        return

    price_df = pd.concat(price_data, ignore_index=True)
    price_pivot = price_df.pivot(index="date", columns="symbol", values="close")
    print(f"   价格数据形状: {price_pivot.shape}")

    # 计算未来收益
    print("🔁 计算未来收益...")
    future_returns = price_pivot.pct_change(periods=5).shift(-5)

    # 分析因子
    print("📊 分析因子表现...")
    factor_results = []

    # 过滤因子列（排除一些明显无用的）
    exclude_patterns = ["RETURN_", "FUTURE_", "TARGET_", "VBT_Price", "VBT_Volume"]
    factor_columns = []

    for col in factor_panel.columns:
        if not any(pattern in col for pattern in exclude_patterns):
            factor_columns.append(col)

    print(f"   候选因子数量: {len(factor_columns)}")

    # 逐个分析因子
    for i, factor in enumerate(factor_columns[:50]):  # 限制分析数量
        if i % 10 == 0:
            print(f"   进度: {i}/{min(50, len(factor_columns))}")

        # 提取因子数据
        factor_series = factor_panel[factor]

        # 检查缺失率
        missing_ratio = factor_series.isna().mean()
        if missing_ratio > 0.3:
            continue

        # 转换为矩阵格式
        factor_matrix = factor_series.unstack(level="symbol")

        # 对齐日期
        factor_matrix = factor_matrix.reindex(future_returns.index)

        # 计算相关性
        correlations = calculate_correlation(factor_matrix, future_returns)

        if len(correlations) >= 20:  # 至少20个观测值
            ic_mean = np.mean(correlations)
            ic_std = np.std(correlations)
            ic_ir = ic_mean / ic_std if ic_std > 0 else 0
            positive_ratio = np.mean(np.array(correlations) > 0)

            factor_results.append(
                {
                    "factor_name": factor,
                    "ic_mean": ic_mean,
                    "ic_std": ic_std,
                    "ic_ir": ic_ir,
                    "positive_ratio": positive_ratio,
                    "samples": len(correlations),
                    "missing_ratio": missing_ratio,
                }
            )

    # 排序并筛选
    print(f"📈 有效因子数量: {len(factor_results)}")

    if factor_results:
        # 按IC均值排序
        factor_results.sort(key=lambda x: abs(x["ic_mean"]), reverse=True)

        # 输出Top 30
        print("\n🏆 Top 30 因子:")
        print("-" * 100)
        print(
            f"{'排名':<4} {'因子名称':<40} {'IC均值':<10} {'IR':<8} {'胜率':<8} {'样本数':<8}"
        )
        print("-" * 100)

        top_factors = factor_results[:30]
        for i, factor in enumerate(top_factors, 1):
            print(
                f"{i:<4} {factor['factor_name']:<40} {factor['ic_mean']:<10.4f} {factor['ic_ir']:<8.2f} {factor['positive_ratio']:<8.2%} {factor['samples']:<8}"
            )

        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = (
            f"/Users/zhangshenshen/深度量化0927/top_30_factors_{timestamp}.json"
        )

        result_data = {
            "timestamp": timestamp,
            "total_factors_analyzed": len(factor_columns),
            "valid_factors": len(factor_results),
            "top_30_factors": top_factors,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        print(f"\n💾 结果已保存至: {output_path}")

        # 生成简化的因子列表
        factor_list_path = (
            f"/Users/zhangshenshen/深度量化0927/top_30_factors_list_{timestamp}.txt"
        )
        with open(factor_list_path, "w", encoding="utf-8") as f:
            f.write("Top 30 ETF因子列表\n")
            f.write("=" * 50 + "\n\n")
            for i, factor in enumerate(top_factors, 1):
                f.write(
                    f"{i:2d}. {factor['factor_name']} (IC: {factor['ic_mean']:.4f}, IR: {factor['ic_ir']:.2f})\n"
                )

        print(f"📄 因子列表已保存至: {factor_list_path}")

    else:
        print("❌ 未找到有效因子")


if __name__ == "__main__":
    main()
