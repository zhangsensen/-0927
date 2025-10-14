"""
资金流因子快速开始示例

演示如何使用资金流因子系统：
1. 加载资金流数据
2. 计算核心因子
3. 生成健康报告
"""

from pathlib import Path

import pandas as pd

from factor_system.factor_engine.factors.money_flow import (
    Flow_Price_Divergence,
    LargeOrder_Ratio,
    MainFlow_Momentum,
    MainNetInflow_Rate,
    MoneyFlow_Consensus,
    MoneyFlow_Hierarchy,
    OrderConcentration,
    SuperLargeOrder_Ratio,
)
from factor_system.factor_engine.providers.money_flow_provider import (
    MoneyFlowProvider,
)


def main():
    """主函数"""
    print("=" * 60)
    print("资金流因子快速开始示例")
    print("=" * 60)

    # 1. 创建数据提供者
    print("\n1. 加载资金流数据...")
    data_dir = Path("raw/SH/money_flow")
    provider = MoneyFlowProvider(data_dir=data_dir)

    try:
        # 加载数据
        df = provider.load_money_flow("600036.SH", "2024-01-01", "2024-12-31")
        print(f"   ✓ 数据加载成功: {len(df)}条记录")
        print(f"   ✓ 数据列数: {len(df.columns)}")
        print(f"   ✓ 日期范围: {df.index[0]} ~ {df.index[-1]}")

        # 2. 计算核心因子
        print("\n2. 计算核心因子...")
        factors = {
            "MainNetInflow_Rate": MainNetInflow_Rate(window=5),
            "LargeOrder_Ratio": LargeOrder_Ratio(window=10),
            "SuperLargeOrder_Ratio": SuperLargeOrder_Ratio(window=20),
            "OrderConcentration": OrderConcentration(),
            "MoneyFlow_Hierarchy": MoneyFlow_Hierarchy(),
            "MoneyFlow_Consensus": MoneyFlow_Consensus(window=5),
            "MainFlow_Momentum": MainFlow_Momentum(short_window=5, long_window=10),
        }

        factor_df = pd.DataFrame(index=df.index)
        for name, factor in factors.items():
            factor_values = factor.calculate(df)
            factor_df[name] = factor_values
            valid_count = (~factor_values.isna()).sum()
            print(f"   ✓ {name}: {valid_count}个有效值")

        # 3. 因子统计
        print("\n3. 因子统计摘要...")
        print(factor_df.describe().to_string())

        # 4. 因子相关性
        print("\n4. 因子相关性矩阵...")
        corr_matrix = factor_df.corr()
        print(corr_matrix.to_string())

        # 5. 可交易性统计
        print("\n5. 可交易性统计...")
        tradable_days = (df["tradability_mask"] == 1).sum()
        total_days = len(df)
        print(f"   总交易日: {total_days}")
        print(f"   可交易日: {tradable_days} ({tradable_days/total_days:.2%})")
        print(f"   被屏蔽日: {total_days-tradable_days} ({(total_days-tradable_days)/total_days:.2%})")

        # 6. 保存结果
        print("\n6. 保存结果...")
        output_dir = Path("output/moneyflow_factors")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存因子值
        factor_df.to_parquet(output_dir / "600036_SH_factors.parquet")
        print(f"   ✓ 因子值已保存: {output_dir / '600036_SH_factors.parquet'}")

        # 保存相关性矩阵
        corr_matrix.to_csv(output_dir / "correlation_matrix.csv")
        print(f"   ✓ 相关性矩阵已保存: {output_dir / 'correlation_matrix.csv'}")

        print("\n" + "=" * 60)
        print("✅ 示例运行成功！")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"\n❌ 错误: 数据文件未找到")
        print(f"   请确保资金流数据存在于: {data_dir}")
        print(f"   错误详情: {e}")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
