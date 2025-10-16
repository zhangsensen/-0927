#!/usr/bin/env python3
"""测试因子注册情况"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from factor_system.factor_engine.factors.etf_cross_section import (
    create_etf_cross_section_manager,
    ETFCrossSectionConfig
)

# 创建管理器
config = ETFCrossSectionConfig()
config.enable_dynamic_factors = True
config.max_dynamic_factors = 1000

manager = create_etf_cross_section_manager(config)

# 强制注册动态因子
print("=" * 80)
print("注册动态因子...")
print("=" * 80)
count = manager._register_all_dynamic_factors()
print(f"✅ 注册完成: {count} 个动态因子")

# 获取可用因子
available_factors = manager.get_available_factors()
print(f"\n✅ 可用因子总数: {len(available_factors)}")

# 获取动态因子列表
dynamic_factors = manager.factor_registry.list_factors(is_dynamic=True)
print(f"✅ 动态因子数: {len(dynamic_factors)}")

# 显示前20个动态因子
print("\n前20个动态因子:")
for i, factor_id in enumerate(dynamic_factors[:20], 1):
    factor_info = manager.factor_registry.get_factor(factor_id)
    if factor_info:
        print(f"  {i}. {factor_id} - {factor_info.category.value}")
    else:
        print(f"  {i}. {factor_id} - ❌ 无元数据")

# 测试获取因子
print("\n" + "=" * 80)
print("测试因子获取...")
print("=" * 80)

test_factors = dynamic_factors[:5]
for factor_id in test_factors:
    factor_info = manager.factor_registry.get_factor(factor_id)
    if factor_info:
        print(f"✅ {factor_id}: 已注册")
        print(f"   类别: {factor_info.category.value}")
        print(f"   参数: {factor_info.parameters}")
    else:
        print(f"❌ {factor_id}: 未注册")

print("\n" + "=" * 80)
print("完成")
print("=" * 80)
