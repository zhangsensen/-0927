#!/usr/bin/env python3
"""快速测试动态因子计算"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from factor_system.factor_engine.factors.etf_cross_section import (
    create_etf_cross_section_manager,
    ETFCrossSectionConfig
)

print("=" * 80)
print("动态因子快速测试")
print("=" * 80)

# 创建管理器
config = ETFCrossSectionConfig()
config.enable_dynamic_factors = True
config.enable_legacy_factors = False  # 只测试动态因子
config.max_dynamic_factors = 1000

manager = create_etf_cross_section_manager(config)

# 注册动态因子
print("\n注册动态因子...")
count = manager._register_all_dynamic_factors()
print(f"✅ 注册: {count} 个")

# 获取前10个动态因子
dynamic_factors = manager.factor_registry.list_factors(is_dynamic=True)
test_factors = dynamic_factors[:10]
print(f"\n测试因子: {test_factors}")

# 测试ETF
test_symbols = ['510300.SH', '159915.SZ', '515030.SH']
end_date = datetime(2024, 10, 21)
start_date = end_date - timedelta(days=60)

print(f"\n测试参数:")
print(f"  ETF: {test_symbols}")
print(f"  时间: {start_date.date()} ~ {end_date.date()}")
print(f"  因子数: {len(test_factors)}")

# 计算因子
print("\n开始计算...")
result = manager.calculate_factors(
    symbols=test_symbols,
    timeframe='daily',
    start_date=start_date,
    end_date=end_date,
    factor_ids=test_factors
)

print("\n" + "=" * 80)
print("计算结果")
print("=" * 80)
print(f"成功因子: {len(result.successful_factors)}")
print(f"失败因子: {len(result.failed_factors)}")
print(f"计算时间: {result.calculation_time:.2f}s")
print(f"内存使用: {result.memory_usage_mb:.1f}MB")

if result.factors_df is not None and not result.factors_df.empty:
    print(f"\n数据形状: {result.factors_df.shape}")
    print(f"数据列: {list(result.factors_df.columns)}")
    print(f"\n前5行:")
    print(result.factors_df.head())
else:
    print("\n❌ 无数据返回")

print("\n成功因子列表:")
for i, f in enumerate(result.successful_factors, 1):
    print(f"  {i}. {f}")

if result.failed_factors:
    print("\n失败因子列表:")
    for i, f in enumerate(result.failed_factors, 1):
        print(f"  {i}. {f}")

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)
