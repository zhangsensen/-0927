"""
资金流因子集成示例

展示如何使用EnhancedFactorEngine同时计算技术和资金流因子
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO)

# 添加项目路径
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from factor_system.factor_engine.core.enhanced_engine import EnhancedFactorEngine
from factor_system.factor_engine.core.registry import get_global_registry
from factor_system.factor_engine.factors.money_flow.registry import (
    get_money_flow_factor_sets,
    register_money_flow_factors,
)


def main():
    """主函数 - 展示资金流因子集成"""
    print("=" * 60)
    print("资金流因子集成到FactorEngine示例")
    print("=" * 60)

    # 1. 获取全局因子注册表
    registry = get_global_registry()

    # 2. 注册资金流因子
    print("\n1. 注册资金流因子...")
    register_money_flow_factors(registry)
    print(
        f"   注册完成: {len([f for f in registry.list_factors() if 'money_flow' in str(registry.get_metadata(f).get('data_source', ''))])} 个资金流因子"
    )

    # 3. 创建增强因子引擎
    print("\n2. 创建增强因子引擎...")

    # 价格数据提供者（示例，使用简单实现）
    class SimplePriceProvider:
        def load_price_data(self, symbols, timeframe, start_date, end_date):
            # 这里应该实现实际的价格数据加载逻辑
            # 为了示例，直接返回空DataFrame（实际使用时需要实现）
            return pd.DataFrame()

        def load_fundamental_data(
            self, symbols, fields, start_date=None, end_date=None
        ):
            return pd.DataFrame()

        def get_trading_calendar(self, market, start_date, end_date):
            return []

    price_provider = SimplePriceProvider()

    # 创建增强引擎
    enhanced_engine = EnhancedFactorEngine(
        data_provider=price_provider, registry=registry
    )

    print("   ✅ 增强因子引擎创建完成")

    # 4. 定义测试参数
    symbols = ["000600.SZ", "600036.SH", "000001.SZ"]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)  # 过去60天

    # 5. 计算混合因子
    print("\n3. 计算混合因子...")

    # 技术因子示例
    technical_factors = ["RSI", "MACD", "SMA_20", "EMA_12", "ATR_14"]

    # 资金流因子示例
    money_flow_factors = [
        "MainNetInflow_Rate",
        "OrderConcentration",
        "MoneyFlow_Hierarchy",
        "Flow_Price_Divergence",
        "Institutional_Absorption",
    ]

    print(f"   技术因子: {technical_factors}")
    print(f"   资金流因子: {money_flow_factors}")
    print(f"   测试标的: {symbols}")
    print(f"   时间范围: {start_date.date()} ~ {end_date.date()}")

    try:
        # 计算混合因子
        mixed_result = enhanced_engine.calculate_mixed_factors(
            technical_factors=technical_factors,
            money_flow_factors=money_flow_factors,
            symbols=symbols,
            timeframe="1day",
            start_date=start_date,
            end_date=end_date,
            use_cache=False,  # 示例不使用缓存
        )

        if not mixed_result.empty:
            print(f"\n   ✅ 混合因子计算成功: {mixed_result.shape}")
            print(f"   因子列: {mixed_result.columns.tolist()}")

            # 显示样本数据
            print("\n   样本数据 (前3行):")
            sample_data = mixed_result.head(3)
            for idx, row in sample_data.iterrows():
                print(f"     {idx}: {dict(row.dropna())}")
        else:
            print("   ❌ 混合因子计算结果为空")

    except Exception as e:
        print(f"   ❌ 混合因子计算失败: {e}")
        import traceback

        traceback.print_exc()

    # 6. 分别计算因子类型
    print("\n4. 分别计算因子类型...")

    # 纯技术因子
    try:
        tech_result = enhanced_engine.calculate_technical_factors(
            ["RSI", "MACD"], symbols, "1day", start_date, end_date
        )
        print(
            f"   技术因子结果: {tech_result.shape if not tech_result.empty else '空'}"
        )
    except Exception as e:
        print(f"   技术因子计算失败: {e}")

    # 纯资金流因子
    try:
        money_result = enhanced_engine.calculate_money_flow_factors(
            ["MainNetInflow_Rate", "OrderConcentration"], symbols, start_date, end_date
        )
        print(
            f"   资金流因子结果: {money_result.shape if not money_result.empty else '空'}"
        )
    except Exception as e:
        print(f"   资金流因子计算失败: {e}")

    # 7. 显示因子集信息
    print("\n5. 因子集信息...")
    factor_sets = get_money_flow_factor_sets()
    for set_id, set_info in factor_sets.items():
        print(f"   {set_id}: {set_info['name']} - {len(set_info['factors'])}个因子")

    # 8. 完成
    print("\n" + "=" * 60)
    print("✅ 资金流因子集成示例完成")
    print("=" * 60)

    print("\n💡 使用提示:")
    print("1. 确保已注册资金流因子: register_money_flow_factors(registry)")
    print("2. 创建EnhancedFactorEngine时传入资金流提供者")
    print("3. 使用calculate_mixed_factors()计算混合因子")
    print("4. 资金流因子需要T+1时序安全处理")


if __name__ == "__main__":
    main()
