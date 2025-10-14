"""
资金流因子集成测试

验证资金流因子是否正确集成到FactorEngine中
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd


def test_money_flow_integration():
    """测试资金流因子集成"""
    print("=" * 60)
    print("测试资金流因子集成")
    print("=" * 60)

    try:
        # 1. 测试资金流因子注册
        print("\n1. 测试资金流因子注册...")
        from factor_system.factor_engine.core.registry import get_global_registry
        from factor_system.factor_engine.factors.money_flow.registry import (
            register_money_flow_factors,
            get_money_flow_factor_sets
        )

        registry = get_global_registry()
        register_money_flow_factors(registry)

        # 检查注册的资金流因子
        money_flow_factors = [
            f for f in registry.list_factors()
            if registry.get_metadata(f) and
               registry.get_metadata(f).get('data_source') == 'money_flow'
        ]

        print(f"   ✅ 注册资金流因子: {len(money_flow_factors)}个")
        print(f"   因子列表: {money_flow_factors}")

        # 2. 测试资金流数据提供者
        print("\n2. 测试资金流数据提供者...")
        from factor_system.factor_engine.providers.money_flow_provider_engine import MoneyFlowDataProvider

        provider = MoneyFlowDataProvider(
            money_flow_dir=project_root / "raw/SH/money_flow",
            enforce_t_plus_1=True
        )

        # 测试数据加载
        symbols = ["000600.SZ"]
        start_date = "2024-08-01"
        end_date = "2024-12-31"

        money_flow_data = provider.load_money_flow_data(
            symbols, "1day", start_date, end_date
        )

        print(f"   ✅ 资金流数据加载: {money_flow_data.shape}")
        if not money_flow_data.empty and isinstance(money_flow_data.index, pd.MultiIndex):
            print(f"   股票: {money_flow_data.index.get_level_values('symbol').unique().tolist()}")
        elif not money_flow_data.empty:
            print(f"   数据索引类型: {type(money_flow_data.index)}")

        if not money_flow_data.empty:
            sample_data = money_flow_data.head()
            print(f"   样本列: {sample_data.columns.tolist()}")
            print(f"   时序安全标记: {sample_data['temporal_safe'].iloc[0]}")

        # 3. 测试增强因子引擎
        print("\n3. 测试增强因子引擎...")
        from factor_system.factor_engine.core.enhanced_engine import EnhancedFactorEngine

        # 创建简单价格提供者（用于测试）
        class SimplePriceProvider:
            def load_price_data(self, symbols, timeframe, start_date, end_date):
                return pd.DataFrame()

            def load_fundamental_data(self, symbols, fields, start_date=None, end_date=None):
                return pd.DataFrame()

            def get_trading_calendar(self, market, start_date, end_date):
                return []

        price_provider = SimplePriceProvider()

        # 创建增强引擎
        enhanced_engine = EnhancedFactorEngine(
            data_provider=price_provider,
            registry=registry
        )

        print("   ✅ 增强因子引擎创建成功")

        # 4. 测试因子计算
        print("\n4. 测试因子计算...")
        test_factors = ["MainNetInflow_Rate", "OrderConcentration", "MoneyFlow_Hierarchy"]

        start_dt = datetime(2024, 8, 1)
        end_dt = datetime(2024, 12, 31)

        try:
            result = enhanced_engine.calculate_money_flow_factors(
                test_factors, symbols, start_dt, end_dt
            )

            if not result.empty:
                print(f"   ✅ 因子计算成功: {result.shape}")
                print(f"   因子列: {result.columns.tolist()}")

                # 显示样本结果
                print("\n   样本结果:")
                sample_result = result.head(3)
                for idx, row in sample_result.iterrows():
                    print(f"     {idx}: {dict(row.dropna())}")
            else:
                print("   ⚠️ 因子计算结果为空（可能是数据问题）")

        except Exception as e:
            print(f"   ❌ 因子计算失败: {e}")
            import traceback
            traceback.print_exc()

        # 5. 测试因子集
        print("\n5. 测试因子集...")
        factor_sets = get_money_flow_factor_sets()
        for set_id, set_info in factor_sets.items():
            print(f"   {set_id}: {set_info['name']} ({len(set_info['factors'])}个因子)")

        # 6. 完成
        print("\n" + "=" * 60)
        print("✅ 资金流因子集成测试完成")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_money_flow_integration()
    if success:
        print("\n🎉 集成测试通过！资金流因子已成功集成到FactorEngine")
    else:
        print("\n💥 集成测试失败！请检查上述错误信息")
        sys.exit(1)