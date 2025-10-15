#!/usr/bin/env python3
"""
A股资金流因子加工脚本
使用完全时序安全的T+1因子体系
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# 添加项目路径
project_root = Path(__file__).parent
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
    Northbound_NetInflow_Rate,
)
from factor_system.factor_engine.providers.money_flow_provider import MoneyFlowProvider


def main():
    """主函数 - 执行资金流因子加工"""

    print("🚀 A股资金流因子加工 - T+1时序安全版本")
    print("=" * 60)

    # 选择测试股票
    test_symbol = "600036.SH"  # 招商银行，有完整数据
    start_date = "2024-08-23"  # 与新下载的资金流数据时间范围匹配
    end_date = "2025-08-22"  # 使用完整一年的数据

    print(f"📊 测试股票: {test_symbol}")
    print(f"📅 时间范围: {start_date} 到 {end_date}")
    print("=" * 60)

    try:
        # 1. 初始化数据提供者
        print("🔧 初始化MoneyFlowProvider...")
        provider = MoneyFlowProvider(
            data_dir=Path("raw/SH/money_flow"), enforce_t_plus_1=True  # 强制T+1时序安全
        )
        print(f"   ✅ T+1滞后: {'启用' if provider.enforce_t_plus_1 else '禁用'}")
        print()

        # 2. 加载资金流数据
        print(f"📥 加载{test_symbol}资金流数据...")
        df = provider.load_money_flow(test_symbol, start_date, end_date)
        print(f"   ✅ 数据形状: {df.shape}")
        print(f"   ✅ 时间范围: {df.index.min()} 到 {df.index.max()}")
        print(
            f"   ✅ 时序安全: {df['temporal_safe'].all() if 'temporal_safe' in df.columns else 'N/A'}"
        )
        print()

        # 3. 显示数据概览
        print("📋 数据字段概览:")
        key_cols = ["main_net", "turnover_amount", "close", "volume"]
        for col in key_cols:
            if col in df.columns:
                print(
                    f"   ✅ {col}: 范围 [{df[col].min():.3f}, {df[col].max():.3f}], 均值 {df[col].mean():.3f}"
                )
        print()

        # 4. 初始化所有因子（12个T+1安全因子）
        print("⚙️ 初始化因子引擎...")
        factors = {
            # 核心因子（8个）
            "MainNetInflow_Rate": MainNetInflow_Rate(window=5),
            "LargeOrder_Ratio": LargeOrder_Ratio(window=10),
            "SuperLargeOrder_Ratio": SuperLargeOrder_Ratio(window=20),
            "OrderConcentration": OrderConcentration(),
            "MoneyFlow_Hierarchy": MoneyFlow_Hierarchy(),
            "MoneyFlow_Consensus": MoneyFlow_Consensus(window=5),
            "MainFlow_Momentum": MainFlow_Momentum(short_window=5, long_window=10),
            "Flow_Price_Divergence": Flow_Price_Divergence(window=5),
            # 增强因子（4个）
            "Institutional_Absorption": Institutional_Absorption(),
            "Flow_Tier_Ratio_Delta": Flow_Tier_Ratio_Delta(window=5),
            "Flow_Reversal_Ratio": Flow_Reversal_Ratio(),
            "Northbound_NetInflow_Rate": Northbound_NetInflow_Rate(window=5),
        }

        print(f"   ✅ 初始化因子数: {len(factors)}")
        print(f"   ✅ 核心因子: 8个")
        print(f"   ✅ 增强因子: 4个")
        print()

        # 5. 计算所有因子
        print("🧮 计算资金流因子...")
        factor_results = {}

        for factor_name, factor in factors.items():
            print(f"   ⚡ 计算 {factor_name}...")
            try:
                factor_values = factor.calculate(df)
                factor_results[factor_name] = factor_values

                # 显示因子统计
                valid_values = factor_values.dropna()
                if len(valid_values) > 0:
                    print(f"      📊 有效值: {len(valid_values)}/{len(factor_values)}")
                    print(
                        f"      📈 范围: [{valid_values.min():.4f}, {valid_values.max():.4f}]"
                    )
                    print(
                        f"      📉 均值: {valid_values.mean():.4f}, 标准差: {valid_values.std():.4f}"
                    )
                else:
                    print(f"      ⚠️ 无有效值")
                print()

            except Exception as e:
                print(f"      ❌ 计算失败: {e}")
                factor_results[factor_name] = pd.Series(
                    np.nan, index=df.index, name=factor_name
                )
                print()

        # 6. 汇总因子结果
        print("📊 汇总因子计算结果...")
        factor_df = pd.DataFrame(factor_results)
        print(f"   ✅ 因子数据形状: {factor_df.shape}")
        print(f"   ✅ 因子数量: {factor_df.shape[1]}")
        print(f"   ✅ 时间序列数: {factor_df.shape[0]}")
        print()

        # 7. 因子质量分析
        print("🔍 因子质量分析...")
        valid_count = 0
        for col in factor_df.columns:
            valid_ratio = factor_df[col].notna().mean()
            if valid_ratio > 0.5:  # 有效率超过50%
                valid_count += 1
                print(f"   ✅ {col}: 有效率 {valid_ratio:.1%}")
            else:
                print(f"   ⚠️ {col}: 有效率 {valid_ratio:.1%} (较低)")

        print(f"\n   📈 高质量因子数: {valid_count}/{len(factor_df.columns)}")
        print()

        # 8. 保存结果
        print("💾 保存因子结果...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = (
            f"factor_output/moneyflow_factors_{test_symbol}_{timestamp}.parquet"
        )

        # 确保输出目录存在
        Path("factor_output").mkdir(exist_ok=True)

        # 合并原始数据和因子数据
        result_df = pd.concat([df, factor_df], axis=1)
        result_df.to_parquet(output_file)

        print(f"   ✅ 保存至: {output_file}")
        print(
            f"   ✅ 文件大小: {Path(output_file).stat().st_size / 1024 / 1024:.2f} MB"
        )
        print()

        # 9. 显示样本结果
        print("📋 样本因子值（最近5个交易日）:")
        if len(result_df) > 0:
            sample_cols = ["main_net", "turnover_amount"] + list(factor_df.columns[:5])
            available_cols = [col for col in sample_cols if col in result_df.columns]
            print(result_df[available_cols].tail().round(4))
        print()

        # 10. 总结报告
        print("🎉 资金流因子加工完成！")
        print("=" * 60)
        print(f"📊 处理股票: {test_symbol}")
        print(f"📅 时间范围: {start_date} 到 {end_date}")
        print(f"🧮 计算因子: {len(factor_df.columns)}个")
        print(f"✅ 有效因子: {valid_count}个")
        print(f"💾 输出文件: {output_file}")
        print(f"🛡️ 时序安全: {'完全安全' if provider.enforce_t_plus_1 else '未启用'}")
        print("=" * 60)

        return True, result_df

    except Exception as e:
        print(f"❌ 处理过程中发生异常: {e}")
        import traceback

        traceback.print_exc()
        return False, None


if __name__ == "__main__":
    success, result_df = main()
    exit(0 if success else 1)
