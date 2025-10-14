#!/usr/bin/env python3
"""
测试资金流因子集成到FactorEngine
验证日线计算时自动加载资金流数据
"""

import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from factor_system.factor_engine.core.engine import FactorEngine
from factor_system.factor_engine.core.registry import get_global_registry
from factor_system.factor_engine.providers.parquet_provider import ParquetDataProvider
from factor_system.factor_engine.providers.money_flow_provider_engine import (
    MoneyFlowDataProvider,
)


def test_integrated_money_flow():
    """测试资金流因子集成"""
    print("=" * 60)
    print("测试资金流因子集成到FactorEngine")
    print("=" * 60)

    # 1. 获取全局注册表（自动注册资金流因子）
    print("\n1. 初始化注册表...")
    registry = get_global_registry(include_money_flow=True)
    print(f"   ✅ 注册表加载完成: {len(registry.metadata)} 个因子")

    # 检查资金流因子是否已注册
    money_flow_factors = [
        "MainNetInflow_Rate",
        "LargeOrder_Ratio",
        "SuperLargeOrder_Ratio",
        "OrderConcentration",
    ]

    registered_mf = [f for f in money_flow_factors if f in registry.metadata]
    print(f"   ✅ 资金流因子已注册: {len(registered_mf)}/{len(money_flow_factors)}")
    print(f"   因子列表: {registered_mf}")

    # 2. 创建数据提供者
    print("\n2. 初始化数据提供者...")

    # 价格数据提供者（这里用空的，因为只测试资金流）
    class DummyPriceProvider:
        def load_price_data(self, symbols, timeframe, start_date, end_date):
            import pandas as pd

            # 返回空的价格数据框架（只有索引）
            dates = pd.date_range(start_date, end_date, freq="D")
            index = pd.MultiIndex.from_product(
                [symbols, dates], names=["symbol", "timestamp"]
            )
            return pd.DataFrame(
                {
                    "open": 100.0,
                    "high": 105.0,
                    "low": 95.0,
                    "close": 102.0,
                    "volume": 1000000,
                },
                index=index,
            )

        def load_fundamental_data(self, *args, **kwargs):
            import pandas as pd

            return pd.DataFrame()

        def get_trading_calendar(self, *args, **kwargs):
            return []

    price_provider = DummyPriceProvider()

    # 资金流数据提供者
    money_flow_dir = project_root / "raw/SH/money_flow"
    money_flow_provider = MoneyFlowDataProvider(
        money_flow_dir=money_flow_dir, enforce_t_plus_1=True
    )

    print("   ✅ 数据提供者初始化完成")

    # 3. 创建FactorEngine（传入资金流提供者）
    print("\n3. 初始化FactorEngine...")
    engine = FactorEngine(
        data_provider=price_provider,
        registry=registry,
        money_flow_provider=money_flow_provider,
    )
    print("   ✅ FactorEngine初始化完成")

    # 4. 计算因子（日线，包含资金流因子）
    print("\n4. 计算因子...")
    symbols = ["000600.SZ", "600036.SH"]
    start_date = datetime(2024, 8, 1)
    end_date = datetime(2024, 12, 31)

    # 混合计算：技术指标 + 资金流因子
    factor_ids = [
        "MainNetInflow_Rate",  # 资金流因子
        "LargeOrder_Ratio",  # 资金流因子
        "MoneyFlow_Hierarchy",  # 资金流因子
    ]

    result = engine.calculate_factors(
        factor_ids=factor_ids,
        symbols=symbols,
        timeframe="1day",
        start_date=start_date,
        end_date=end_date,
        use_cache=False,
    )

    print(f"\n   ✅ 因子计算完成: {result.shape}")
    print(f"   因子列: {result.columns.tolist()}")

    # 显示样本数据
    if not result.empty:
        print("\n   样本数据:")
        sample = result.head(10)
        for idx, row in sample.iterrows():
            non_null = {k: v for k, v in row.items() if pd.notna(v)}
            if non_null:
                print(f"     {idx}: {non_null}")

        # 统计
        print("\n   数据质量:")
        for col in result.columns:
            non_null_count = result[col].notna().sum()
            pct = non_null_count / len(result) * 100
            print(f"     {col}: {non_null_count}/{len(result)} ({pct:.1f}%)")

    print("\n" + "=" * 60)
    print("✅ 资金流因子集成测试完成")
    print("=" * 60)


if __name__ == "__main__":
    import pandas as pd

    test_integrated_money_flow()
