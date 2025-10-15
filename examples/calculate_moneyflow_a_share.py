#!/usr/bin/env python3
"""
A股资金流因子计算示例
展示如何使用新的CombinedMoneyFlowProvider计算资金流因子
"""

import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from factor_system.factor_engine.core.engine import FactorEngine
from factor_system.factor_engine.core.registry import get_global_registry
from factor_system.factor_engine.providers.combined_provider import (
    CombinedMoneyFlowProvider,
)
from factor_system.factor_engine.providers.parquet_provider import ParquetDataProvider


def main():
    print("=" * 70)
    print("🚀 A股资金流因子计算示例")
    print("=" * 70)

    # 配置（暂时只用SH市场，因为数据都在raw/SH/）
    symbols = ["600036.SH", "000600.SZ"]  # 000600.SZ数据实际也在SH目录
    start_date = datetime(2024, 8, 1)
    end_date = datetime(2024, 12, 31)

    # 因子列表（技术指标 + 资金流因子）
    factor_ids = [
        # 技术指标
        "RSI",
        "MACD",
        # 资金流因子
        "MainNetInflow_Rate",
        "LargeOrder_Ratio",
        "Institutional_Absorption",
    ]

    print(f"\n📊 配置:")
    print(f"  标的: {symbols}")
    print(f"  时间: {start_date.date()} ~ {end_date.date()}")
    print(f"  因子: {factor_ids}")

    # 1. 初始化注册表
    print("\n1️⃣ 初始化因子注册表...")
    registry = get_global_registry(include_money_flow=True)
    print(f"   ✅ 已注册 {len(registry.metadata)} 个因子")

    # 2. 创建组合数据提供者
    print("\n2️⃣ 创建数据提供者...")
    price_provider = ParquetDataProvider(raw_data_dir=project_root / "raw")

    combined_provider = CombinedMoneyFlowProvider(
        price_provider=price_provider,
        money_flow_dir=project_root / "raw/SH/money_flow",
        enforce_t_plus_1=True,
    )
    print("   ✅ CombinedMoneyFlowProvider已就绪")

    # 3. 创建因子引擎
    print("\n3️⃣ 创建FactorEngine...")
    engine = FactorEngine(
        data_provider=combined_provider,
        registry=registry,
    )
    print("   ✅ FactorEngine已就绪")

    # 4. 计算因子
    print("\n4️⃣ 开始计算因子...")
    try:
        result = engine.calculate_factors(
            factor_ids=factor_ids,
            symbols=symbols,
            timeframe="daily",
            start_date=start_date,
            end_date=end_date,
            use_cache=False,
        )

        print(f"\n✅ 计算完成: {result.shape}")
        print(f"   因子列: {result.columns.tolist()}")

        # 5. 数据质量统计
        print("\n5️⃣ 数据质量:")

        # 时间范围
        if isinstance(result.index, pd.MultiIndex):
            # MultiIndex: (symbol, datetime)
            date_level = result.index.get_level_values("datetime")
            dates = pd.to_datetime(date_level).unique()
        else:
            dates = pd.to_datetime(result.index).unique()

        print(f"   时间范围: {dates.min().date()} ~ {dates.max().date()}")
        print(f"   有效天数: {len(dates)} 天")

        # 因子统计
        for col in result.columns:
            valid = result[col].notna().sum()
            total = len(result)
            pct = valid / total * 100
            mean_val = result[col].mean() if valid > 0 else float("nan")
            print(
                f"   {col:30s}: {valid:4d}/{total:4d} ({pct:5.1f}%) | mean={mean_val:8.4f}"
            )

        # 6. 样本数据
        print("\n6️⃣ 样本数据（前5条）:")
        sample = result.head(5)
        for idx, row in sample.iterrows():
            non_null = {k: f"{v:.4f}" for k, v in row.items() if pd.notna(v)}
            if non_null:
                print(f"   {idx}: {non_null}")

        # 7. 保存结果
        output_dir = project_root / "factor_output/examples"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"a_share_money_flow_{timestamp}.parquet"

        result.to_parquet(output_file, compression="snappy", index=True)
        file_size = output_file.stat().st_size / 1024 / 1024

        print(f"\n7️⃣ 结果已保存:")
        print(f"   文件: {output_file}")
        print(f"   大小: {file_size:.2f} MB")

        print("\n" + "=" * 70)
        print("✅ 示例执行完成")
        print("=" * 70)

        return result

    except Exception as e:
        print(f"\n❌ 计算失败: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    import pandas as pd

    result = main()
    sys.exit(0 if result is not None else 1)
