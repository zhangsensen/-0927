#!/usr/bin/env python3
"""
资金流因子集成综合测试

测试场景：
1. 多股票因子计算
2. 边缘情况处理（缺失资金流数据）
3. 因子集调用
4. 数据有效性验证
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from factor_system.factor_engine import api


def test_moneyflow_integration():
    """综合测试资金流因子集成"""
    print("=" * 80)
    print("资金流因子集成综合测试")
    print("=" * 80)

    # 测试股票列表
    test_symbols = [
        "000600.SZ",  # 建投能源（有资金流数据）
        "600036.SH",  # 招商银行（有资金流数据）
        "000001.SZ",  # 平安银行（有资金流数据）
        "600519.SH",  # 贵州茅台（可能有资金流数据）
        "000858.SZ",  # 五粮液（可能有资金流数据）
    ]

    # 测试时间范围
    start_date = datetime(2024, 8, 1)
    end_date = datetime(2024, 12, 31)

    results = {}

    # 测试1: 使用因子集计算
    print("\n" + "=" * 80)
    print("测试1: 使用因子集计算资金流因子")
    print("=" * 80)

    try:
        result = api.calculate_factor_set(
            set_id="a_share_moneyflow_core",
            symbols=test_symbols,
            timeframe="daily",
            start_date=start_date,
            end_date=end_date,
            use_cache=False,
        )

        print(f"✅ 因子集计算成功")
        print(f"   数据形状: {result.shape}")
        print(f"   因子列: {result.columns.tolist()}")
        print(
            f"   数据覆盖率: {(1 - result.isnull().sum().sum() / result.size) * 100:.2f}%"
        )

        results["factor_set"] = result

    except Exception as e:
        print(f"❌ 因子集计算失败: {e}")
        import traceback

        traceback.print_exc()

    # 测试2: 单独计算资金流因子
    print("\n" + "=" * 80)
    print("测试2: 单独计算资金流因子")
    print("=" * 80)

    money_flow_factors = [
        "MainNetInflow_Rate",
        "OrderConcentration",
        "MoneyFlow_Hierarchy",
    ]

    try:
        result = api.calculate_factors(
            factor_ids=money_flow_factors,
            symbols=test_symbols,
            timeframe="daily",
            start_date=start_date,
            end_date=end_date,
            use_cache=False,
        )

        print(f"✅ 单独因子计算成功")
        print(f"   数据形状: {result.shape}")

        # 按股票统计
        if hasattr(result.index, "levels"):
            for symbol in test_symbols:
                try:
                    symbol_data = result.xs(symbol, level="symbol")
                    valid_ratio = (
                        1 - symbol_data.isnull().sum().sum() / symbol_data.size
                    ) * 100
                    print(
                        f"   {symbol}: {len(symbol_data)}天, 有效率{valid_ratio:.2f}%"
                    )
                except:
                    print(f"   {symbol}: 无数据")

        results["individual"] = result

    except Exception as e:
        print(f"❌ 单独因子计算失败: {e}")
        import traceback

        traceback.print_exc()

    # 测试3: 混合计算（技术+资金流）
    print("\n" + "=" * 80)
    print("测试3: 混合计算技术因子和资金流因子")
    print("=" * 80)

    mixed_factors = [
        # 技术因子
        "RSI",
        "MACD",
        # 资金流因子
        "MainNetInflow_Rate",
        "Flow_Price_Divergence",
    ]

    try:
        result = api.calculate_factors(
            factor_ids=mixed_factors,
            symbols=["000600.SZ", "600036.SH"],
            timeframe="daily",
            start_date=start_date,
            end_date=end_date,
            use_cache=False,
        )

        print(f"✅ 混合因子计算成功")
        print(f"   数据形状: {result.shape}")
        print(f"   因子列: {result.columns.tolist()}")

        results["mixed"] = result

    except Exception as e:
        print(f"❌ 混合因子计算失败: {e}")
        import traceback

        traceback.print_exc()

    # 测试4: 列出所有因子集
    print("\n" + "=" * 80)
    print("测试4: 列出所有因子集")
    print("=" * 80)

    try:
        engine = api.get_engine()
        factor_sets = engine.registry.list_factor_sets()

        print(f"✅ 可用因子集: {len(factor_sets)}个")
        for set_id in factor_sets:
            factor_set = engine.registry.get_factor_set(set_id)
            if factor_set:
                print(
                    f"   - {set_id}: {factor_set.get('name')} ({len(factor_set.get('factors', []))}个因子)"
                )

    except Exception as e:
        print(f"❌ 列出因子集失败: {e}")

    # 测试5: 数据质量验证
    print("\n" + "=" * 80)
    print("测试5: 数据质量验证")
    print("=" * 80)

    for test_name, result in results.items():
        if result is not None and not result.empty:
            print(f"\n{test_name}:")
            print(f"  总数据点: {result.size}")
            print(f"  缺失值: {result.isnull().sum().sum()}")
            print(
                f"  有效率: {(1 - result.isnull().sum().sum() / result.size) * 100:.2f}%"
            )

            # 检查异常值
            numeric_cols = result.select_dtypes(include=["float64", "int64"]).columns
            for col in numeric_cols:
                if col in result.columns:
                    col_data = result[col].dropna()
                    if len(col_data) > 0:
                        print(
                            f"  {col}: min={col_data.min():.4f}, max={col_data.max():.4f}, mean={col_data.mean():.4f}"
                        )

    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)

    success_count = len([r for r in results.values() if r is not None and not r.empty])
    total_count = 3

    print(f"✅ 成功测试: {success_count}/{total_count}")
    print(f"📊 总因子数: {len(api.list_available_factors())}")
    print(f"📦 因子集数: {len(api.get_engine().registry.list_factor_sets())}")

    return success_count == total_count


if __name__ == "__main__":
    success = test_moneyflow_integration()

    if success:
        print("\n🎉 资金流因子集成测试全部通过！")
        sys.exit(0)
    else:
        print("\n⚠️  部分测试失败，请检查日志")
        sys.exit(1)
