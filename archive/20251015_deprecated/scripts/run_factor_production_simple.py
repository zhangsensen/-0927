#!/usr/bin/env python3
"""
生产环境 - 因子计算脚本
直接使用FactorEngine计算技术指标+资金流因子
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from factor_system.factor_engine.core.engine import FactorEngine
from factor_system.factor_engine.core.registry import get_global_registry
from factor_system.factor_engine.providers.money_flow_provider_engine import (
    MoneyFlowDataProvider,
)
from factor_system.factor_engine.providers.parquet_provider import ParquetDataProvider


def main():
    print("=" * 70)
    print("🚀 生产环境 - 因子计算")
    print("=" * 70)

    # 配置
    symbols = ["000600.SZ", "600036.SH", "600519.SH", "000601.SZ", "600608.SH"]
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    timeframe = "daily"  # 使用daily而非1day

    # 因子列表（技术指标 + 资金流因子）
    factor_ids = [
        # 技术指标
        "RSI",
        "MACD",
        "STOCH",
        # 资金流因子
        "MainNetInflow_Rate",
        "LargeOrder_Ratio",
        "SuperLargeOrder_Ratio",
        "OrderConcentration",
        "MoneyFlow_Hierarchy",
        "MainFlow_Momentum",
    ]

    print(f"\n📊 计算配置:")
    print(f"   标的: {symbols}")
    print(f"   时间: {start_date.date()} ~ {end_date.date()}")
    print(f"   周期: {timeframe}")
    print(f"   因子: {len(factor_ids)} 个")
    print(f"   列表: {factor_ids}")

    # 1. 初始化因子注册表...
    print("\n1️⃣ 初始化因子注册表...")
    registry = get_global_registry(include_money_flow=True)
    print(f"   ✅ 已注册 {len(registry.metadata)} 个因子")

    # 2. 初始化数据提供者
    print("\n2️⃣ 初始化数据提供者...")

    # 价格数据提供者
    raw_data_dir = project_root / "raw"
    price_provider = ParquetDataProvider(raw_data_dir=raw_data_dir)
    print(f"   ✅ 价格数据: {raw_data_dir}")

    # 资金流数据提供者
    money_flow_dir = project_root / "raw/SH/money_flow"
    money_flow_provider = MoneyFlowDataProvider(
        money_flow_dir=money_flow_dir, enforce_t_plus_1=True
    )
    print(f"   ✅ 资金流数据: {money_flow_dir}")

    # 3. 创建FactorEngine
    print("\n3️⃣ 创建FactorEngine...")
    engine = FactorEngine(
        data_provider=price_provider,
        registry=registry,
        money_flow_provider=money_flow_provider,
    )
    print("   ✅ FactorEngine已就绪")

    # 4. 计算因子
    print("\n4️⃣ 开始计算因子...")
    print(f"   处理 {len(symbols)} 个标的...")

    try:
        result = engine.calculate_factors(
            factor_ids=factor_ids,
            symbols=symbols,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            use_cache=False,
        )

        print(f"\n   ✅ 计算完成!")
        print(f"   数据形状: {result.shape}")
        print(f"   因子列: {result.columns.tolist()}")

        # 5. 数据质量检查
        print("\n5️⃣ 数据质量:")
        for col in result.columns:
            non_null = result[col].notna().sum()
            total = len(result)
            pct = non_null / total * 100
            print(f"   {col:30s}: {non_null:6d}/{total:6d} ({pct:5.1f}%)")

        # 6. 显示样本数据
        print("\n6️⃣ 样本数据（前5条）:")
        sample = result.head(5)
        print(sample.to_string())

        # 7. 保存结果
        output_dir = project_root / "factor_system/factor_output/production"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"factors_{timestamp}.parquet"

        result.to_parquet(output_file, compression="snappy", index=True)
        file_size = output_file.stat().st_size / 1024 / 1024

        print(f"\n7️⃣ 结果已保存:")
        print(f"   文件: {output_file}")
        print(f"   大小: {file_size:.2f} MB")

        # 8. 统计摘要
        print("\n8️⃣ 统计摘要:")
        print(result.describe().to_string())

        print("\n" + "=" * 70)
        print("✅ 因子计算完成")
        print("=" * 70)

        return result

    except Exception as e:
        print(f"\n❌ 计算失败: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = main()
    if result is not None:
        print(f"\n💾 最终结果: {result.shape[0]} 条记录, {result.shape[1]} 个因子")
