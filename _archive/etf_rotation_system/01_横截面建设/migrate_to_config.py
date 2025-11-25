#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""配置迁移脚本 - 从旧版本迁移到配置驱动版本"""
import shutil
from pathlib import Path

import yaml


def create_backup():
    """备份原始文件"""
    original_file = Path("generate_panel.py")
    backup_file = Path("generate_panel_original.py")

    if original_file.exists() and not backup_file.exists():
        shutil.copy2(original_file, backup_file)
        print(f"✅ 原始文件已备份为: {backup_file}")
    else:
        print("ℹ️  备份文件已存在或原始文件不存在")


def validate_config_structure():
    """验证配置文件结构"""
    config_file = Path("config/factor_panel_config.yaml")

    if not config_file.exists():
        print("❌ 配置文件不存在")
        return False

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        required_sections = [
            "trading",
            "factor_windows",
            "thresholds",
            "paths",
            "processing",
            "factor_enable",
            "data_processing",
            "output",
            "logging",
        ]

        missing_sections = [
            section for section in required_sections if section not in config
        ]
        if missing_sections:
            print(f"❌ 配置文件缺少必要部分: {missing_sections}")
            return False

        print("✅ 配置文件结构验证通过")
        return True

    except Exception as e:
        print(f"❌ 配置文件验证失败: {e}")
        return False


def test_config_loading():
    """测试配置加载"""
    try:
        from config.config_classes import FactorPanelConfig

        config = FactorPanelConfig.from_yaml("config/factor_panel_config.yaml")

        print("✅ 配置加载测试通过")
        print(
            f"   - 交易参数: 年化天数={config.trading.days_per_year}, epsilon={config.trading.epsilon_small}"
        )
        print(f"   - 动量窗口: {config.factor_windows.momentum}")
        print(f"   - 波动率窗口: {config.factor_windows.volatility}")
        print(f"   - 大单阈值: {config.thresholds.large_order_volume_ratio}")
        print(f"   - 启用因子数: {sum(vars(config.factor_enable).values())}")

        return True

    except Exception as e:
        print(f"❌ 配置加载测试失败: {e}")
        return False


def compare_factor_counts():
    """比较新旧版本的因子数量"""
    try:
        from config.config_classes import FactorPanelConfig

        config = FactorPanelConfig.from_yaml("config/factor_panel_config.yaml")
        enabled_count = sum(vars(config.factor_enable).values())

        # 原版本固定35个因子
        original_count = 35

        print(f"📊 因子数量对比:")
        print(f"   - 原版本: {original_count} 个因子 (固定)")
        print(f"   - 新版本: {enabled_count} 个因子 (可配置)")

        if enabled_count < original_count:
            print("⚠️  新版本启用的因子数量少于原版本")
            disabled_factors = [
                name
                for name, enabled in vars(config.factor_enable).items()
                if not enabled
            ]
            print(f"   - 禁用的因子: {disabled_factors}")
        elif enabled_count == original_count:
            print("✅ 因子数量匹配")
        else:
            print("ℹ️  新版本启用了更多因子")

        return enabled_count == original_count

    except Exception as e:
        print(f"❌ 因子数量比较失败: {e}")
        return False


def create_usage_examples():
    """创建使用示例"""
    examples = {
        "basic_usage": """# 基本使用示例 - 使用默认配置
python generate_panel_refactored.py

# 或者指定数据目录
python generate_panel_refactored.py --data-dir raw/ETF/daily --output-dir results/panels
""",
        "custom_config": """# 使用自定义配置
python generate_panel_refactored.py --config config/my_config.yaml

# 覆盖特定参数
python generate_panel_refactored.py --workers 8 --output-dir custom_output
""",
        "config_modification": """# 配置修改示例:
# 编辑 config/factor_panel_config.yaml

# 修改动量窗口
factor_windows:
  momentum: [10, 30, 60, 120]  # 改为更短的窗口

# 禁用某些因子
factor_enable:
  hammer_pattern: false      # 禁用锤子线形态
  doji_pattern: false        # 禁用十字星形态

# 调整阈值
thresholds:
  large_order_volume_ratio: 1.5  # 提高大单阈值
""",
        "migration_commands": """# 迁移步骤:
# 1. 备份原始文件
python migrate_to_config.py --backup

# 2. 验证配置
python migrate_to_config.py --validate

# 3. 测试新版本
python generate_panel_refactored.py --config config/factor_panel_config.yaml

# 4. 比较结果
python compare_results.py original_panel.parquet new_panel.parquet
""",
    }

    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)

    for filename, content in examples.items():
        example_file = examples_dir / f"{filename}.md"
        with open(example_file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ 创建示例文件: {example_file}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="配置迁移脚本")
    parser.add_argument("--backup", action="store_true", help="备份原始文件")
    parser.add_argument("--validate", action="store_true", help="验证配置文件")
    parser.add_argument("--test", action="store_true", help="测试配置加载")
    parser.add_argument("--compare", action="store_true", help="比较因子数量")
    parser.add_argument("--examples", action="store_true", help="创建使用示例")
    parser.add_argument("--all", action="store_true", help="执行所有检查")

    args = parser.parse_args()

    print("🚀 ETF因子面板配置迁移脚本")
    print("=" * 50)

    success = True

    if args.backup or args.all:
        print("\\n📦 备份原始文件...")
        create_backup()

    if args.validate or args.all:
        print("\\n🔍 验证配置文件...")
        if not validate_config_structure():
            success = False

    if args.test or args.all:
        print("\\n🧪 测试配置加载...")
        if not test_config_loading():
            success = False

    if args.compare or args.all:
        print("\\n📊 比较因子数量...")
        compare_factor_counts()

    if args.examples or args.all:
        print("\\n📝 创建使用示例...")
        create_usage_examples()

    print("\\n" + "=" * 50)
    if success:
        print("✅ 迁移准备完成!")
        print("\\n下一步:")
        print("1. 根据需要修改 config/factor_panel_config.yaml")
        print("2. 运行新版本: python generate_panel_refactored.py")
        print("3. 验证结果是否符合预期")
    else:
        print("❌ 迁移准备失败，请修复上述问题")

    return success


if __name__ == "__main__":
    main()
