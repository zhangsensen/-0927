#!/usr/bin/env python3
"""
生产环境 - 纯资金流因子计算
直接使用资金流数据，不依赖价格数据
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from factor_system.factor_engine.factors.money_flow.core import (
    Flow_Price_Divergence,
    LargeOrder_Ratio,
    MainFlow_Momentum,
    MainNetInflow_Rate,
    MoneyFlow_Consensus,
    MoneyFlow_Hierarchy,
    OrderConcentration,
    SuperLargeOrder_Ratio,
)
from factor_system.factor_engine.factors.money_flow.enhanced import (
    Flow_Reversal_Ratio,
    Flow_Tier_Ratio_Delta,
    Institutional_Absorption,
)
from factor_system.factor_engine.providers.money_flow_provider_engine import (
    MoneyFlowDataProvider,
)


def main():
    print("=" * 70)
    print("🚀 生产环境 - 资金流因子计算")
    print("=" * 70)

    # 配置
    symbols = ["000600.SZ", "600036.SH", "600519.SH", "000601.SZ", "002241.SZ"]
    start_date = "2024-01-01"
    end_date = "2024-12-31"

    # 因子类映射
    factor_classes = {
        "MainNetInflow_Rate": MainNetInflow_Rate,
        "LargeOrder_Ratio": LargeOrder_Ratio,
        "SuperLargeOrder_Ratio": SuperLargeOrder_Ratio,
        "OrderConcentration": OrderConcentration,
        "MoneyFlow_Hierarchy": MoneyFlow_Hierarchy,
        "MoneyFlow_Consensus": MoneyFlow_Consensus,
        "MainFlow_Momentum": MainFlow_Momentum,
        "Institutional_Absorption": Institutional_Absorption,
        "Flow_Tier_Ratio_Delta": Flow_Tier_Ratio_Delta,
        "Flow_Reversal_Ratio": Flow_Reversal_Ratio,
    }

    print(f"\n📊 计算配置:")
    print(f"   标的: {symbols}")
    print(f"   时间: {start_date} ~ {end_date}")
    print(f"   因子: {len(factor_classes)} 个")

    # 1. 初始化数据提供者
    print("\n1️⃣ 加载资金流数据...")
    money_flow_dir = project_root / "raw/SH/money_flow"
    provider = MoneyFlowDataProvider(
        money_flow_dir=money_flow_dir, enforce_t_plus_1=True
    )

    money_flow_data = provider.load_money_flow_data(
        symbols, "1day", start_date, end_date
    )

    if money_flow_data.empty:
        print("   ❌ 未加载到数据")
        return None

    print(f"   ✅ 数据加载完成: {money_flow_data.shape}")
    print(f"   数据列: {money_flow_data.columns.tolist()[:10]}...")

    # 2. 计算因子
    print("\n2️⃣ 计算因子...")
    all_results = {}

    for factor_name, factor_class in factor_classes.items():
        print(f"   🔧 {factor_name}...", end=" ")
        try:
            factor_instance = factor_class()
            factor_values = []

            for symbol in symbols:
                try:
                    symbol_data = money_flow_data.xs(symbol, level="symbol")
                    if symbol_data.empty:
                        continue

                    values = factor_instance.calculate(symbol_data)

                    if values is not None and isinstance(values, pd.Series):
                        multi_idx = pd.MultiIndex.from_product(
                            [[symbol], values.index], names=["symbol", "trade_date"]
                        )
                        factor_df = pd.DataFrame(
                            {factor_name: values.values}, index=multi_idx
                        )
                        factor_values.append(factor_df)

                except Exception as e:
                    print(f"\n      ⚠️ {symbol} 失败: {e}")
                    continue

            if factor_values:
                all_results[factor_name] = pd.concat(factor_values)
                non_null = all_results[factor_name][factor_name].notna().sum()
                print(f"✅ {non_null} 条有效数据")
            else:
                print("⚠️ 无有效数据")

        except Exception as e:
            print(f"❌ 失败: {e}")

    if not all_results:
        print("\n   ❌ 无任何因子计算成功")
        return None

    # 3. 合并结果
    print("\n3️⃣ 合并结果...")
    final_result = pd.concat(all_results.values(), axis=1)
    print(f"   ✅ 最终形状: {final_result.shape}")

    # 4. 数据质量
    print("\n4️⃣ 数据质量:")
    for col in final_result.columns:
        non_null = final_result[col].notna().sum()
        total = len(final_result)
        pct = non_null / total * 100
        print(f"   {col:30s}: {non_null:6d}/{total:6d} ({pct:5.1f}%)")

    # 5. 样本数据
    print("\n5️⃣ 样本数据（前10条）：")
    sample = final_result.head(10)
    for idx, row in sample.iterrows():
        non_null_values = {k: f"{v:.4f}" for k, v in row.items() if pd.notna(v)}
        if non_null_values:
            print(f"   {idx}: {non_null_values}")

    # 6. 保存结果
    print("\n6️⃣ 保存结果...")
    output_dir = project_root / "factor_system/factor_output/production"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"money_flow_factors_{timestamp}.parquet"

    final_result.to_parquet(output_file, compression="snappy", index=True)
    file_size = output_file.stat().st_size / 1024 / 1024

    print(f"   ✅ 文件: {output_file}")
    print(f"   ✅ 大小: {file_size:.2f} MB")

    # 7. 统计摘要
    print("\n7️⃣ 统计摘要:")
    print(final_result.describe().to_string())

    print("\n" + "=" * 70)
    print("✅ 资金流因子计算完成")
    print("=" * 70)

    return final_result


if __name__ == "__main__":
    result = main()
    if result is not None:
        print(f"\n💾 最终结果: {result.shape[0]} 条记录, {result.shape[1]} 个因子")
