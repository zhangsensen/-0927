#!/usr/bin/env python3
"""
使用统一因子引擎计算资金流因子的示例
演示CombinedMoneyFlowProvider的使用方法
"""

import sys
from datetime import datetime
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from factor_system.factor_engine import api


def main():
    """主函数 - 演示资金流因子计算"""
    print("🚀 统一因子引擎 - 资金流因子计算示例")
    print("=" * 60)

    # 定义参数
    symbols = ["600036.SH", "600519.SH"]
    factors = [
        "MainNetInflow_Rate",
        "LargeOrder_Ratio",
        "SuperLargeOrder_Ratio",
        "OrderConcentration",
        "MoneyFlow_Hierarchy",
        "MoneyFlow_Consensus",
        "MainFlow_Momentum",
        "Flow_Price_Divergence",
        "Institutional_Absorption",
        "Flow_Tier_Ratio_Delta",
        "Flow_Reversal_Ratio",
        "Northbound_NetInflow_Rate",
    ]

    print(f"📊 股票: {', '.join(symbols)}")
    print(f"📈 因子: {len(factors)}个资金流因子")
    print(f"📅 时间: 2024-01-01 到 2024-12-31")
    print("=" * 60)

    try:
        # 使用统一API计算因子
        df = api.calculate_factors(
            factor_ids=factors,
            symbols=symbols,
            timeframe="daily",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
        )

        print(f"✅ 因子计算完成: {df.shape}")
        print(f"📊 数据列: {list(df.columns)}")
        print(
            f"📅 时间范围: {df.index.get_level_values('timestamp').min()} 到 {df.index.get_level_values('timestamp').max()}"
        )

        # 显示因子统计
        print(f"\n📈 因子有效性统计:")
        for factor in factors:
            if factor in df.columns:
                valid_count = df[factor].notna().sum()
                total_count = len(df)
                valid_ratio = valid_count / total_count * 100
                print(
                    f"  ✅ {factor}: {valid_count}/{total_count} ({valid_ratio:.1f}%)"
                )
            else:
                print(f"  ❌ {factor}: 因子未找到")

        # 显示样本数据
        print(f"\n📋 样本因子值 (最后5个交易日):")
        if len(df) > 0:
            sample_cols = ["close", "volume"] + factors[:5]
            available_cols = [col for col in sample_cols if col in df.columns]
            print(df[available_cols].tail(10).round(4))

        print(f"\n🎉 资金流因子计算成功！")
        print(f"💡 提示: 缺失资金流数据的股票将自动跳过，不影响其他股票计算")

    except Exception as e:
        print(f"❌ 计算失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
