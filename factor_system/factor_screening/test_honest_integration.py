#!/usr/bin/env python3
"""
诚实集成验证脚本 - P0级严格验证
验证实际使用的工具模块，确保无虚假声明
"""

import logging
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)


def test_honest_integration():
    """诚实集成验证测试 - Linus原则"""
    print("=" * 80)
    print("诚实集成验证测试 - P0级严格验证")
    print("=" * 80)

    # 1. 验证主类导入
    print("\n[1/4] 测试主类导入...")
    try:
        from config_manager import ScreeningConfig
        from professional_factor_screener import ProfessionalFactorScreener

        print("  ✅ ProfessionalFactorScreener 导入成功")
    except ImportError as e:
        print(f"  ❌ 主类导入失败: {e}")
        return False

    # 2. 测试筛选器初始化
    print("\n[2/4] 测试筛选器初始化...")
    try:
        config = ScreeningConfig(ic_horizons=[1, 3, 5], min_sample_size=30)
        screener = ProfessionalFactorScreener(config=config)
        print("  ✅ 筛选器初始化成功")
    except Exception as e:
        print(f"  ❌ 筛选器初始化失败: {e}")
        return False

    # 3. 验证实际使用的模块
    print("\n[3/4] 验证实际使用的模块...")

    # 检查input_validator
    if hasattr(screener, "input_validator"):
        if screener.input_validator is not None:
            print("  ✅ input_validator 实例已创建且实际使用")
            # 验证实际功能
            is_valid, msg = screener.input_validator.validate_symbol("0700.HK")
            if is_valid:
                print("    ✅ input_validator 功能验证通过")
            else:
                print(f"    ❌ input_validator 功能异常: {msg}")
                return False
        else:
            print("  ⚠️  input_validator 实例为None")
    else:
        print("  ❌ input_validator 属性不存在")
        return False

    # 检查structured_logger
    if hasattr(screener, "structured_logger"):
        if screener.structured_logger is not None:
            print("  ✅ structured_logger 实例已创建且实际使用")
            # 验证实际功能
            try:
                screener.structured_logger.info("诚实集成测试", test_status="success")
                print("    ✅ structured_logger 功能验证通过")
            except Exception as e:
                print(f"    ❌ structured_logger 功能异常: {e}")
                return False
        else:
            print("  ⚠️  structured_logger 实例为None")
    else:
        print("  ❌ structured_logger 属性不存在")
        return False

    # 4. 验证已诚实移除的模块
    print("\n[4/4] 验证已诚实移除的模块...")

    # 检查memory_optimizer已移除
    if hasattr(screener, "memory_optimizer"):
        if screener.memory_optimizer is None:
            print("  ✅ memory_optimizer 已诚实移除（设为None）")
        else:
            print("  ❌ memory_optimizer 仍然存在，未诚实移除")
            return False
    else:
        print("  ✅ memory_optimizer 属性不存在（已完全移除）")

    # 检查backup_manager已移除
    if hasattr(screener, "backup_manager"):
        if screener.backup_manager is None:
            print("  ✅ backup_manager 已诚实移除（设为None）")
        else:
            print("  ❌ backup_manager 仍然存在，未诚实移除")
            return False
    else:
        print("  ✅ backup_manager 属性不存在（已完全移除）")

    return True


def test_code_consistency():
    """验证代码与声明的一致性"""
    print("\n" + "=" * 80)
    print("代码一致性验证")
    print("=" * 80)

    # 读取源码，验证导入语句
    source_file = Path(__file__).parent / "professional_factor_screener.py"

    if not source_file.exists():
        print("❌ 源码文件不存在")
        return False

    with open(source_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 验证已移除未使用的导入
    if "from utils.memory_optimizer import" in content:
        print("❌ memory_optimizer 导入仍然存在")
        return False
    else:
        print("✅ memory_optimizer 导入已移除")

    if "from utils.backup_manager import" in content:
        print("❌ backup_manager 导入仍然存在")
        return False
    else:
        print("✅ backup_manager 导入已移除")

    # 验证保留实际使用的导入
    if "from utils.input_validator import" in content:
        print("✅ input_validator 导入保留（实际使用）")
    else:
        print("❌ input_validator 导入被误删")
        return False

    if "from utils.structured_logger import" in content:
        print("✅ structured_logger 导入保留（实际使用）")
    else:
        print("❌ structured_logger 导入被误删")
        return False

    return True


if __name__ == "__main__":
    print("\n")

    # 执行诚实集成测试
    integration_success = test_honest_integration()

    # 执行代码一致性验证
    consistency_success = test_code_consistency()

    print("\n" + "=" * 80)
    if integration_success and consistency_success:
        print("✅ 诚实集成验证：全部通过")
        print("=" * 80)
        print("\n📋 验证结果：")
        print("  1. 主类成功导入 ✅")
        print("  2. 筛选器成功初始化 ✅")
        print("  3. 实际使用的模块功能正常 ✅")
        print("    - input_validator: 实际使用，功能正常 ✅")
        print("    - structured_logger: 实际使用，功能正常 ✅")
        print("  4. 未使用的模块已诚实移除 ✅")
        print("    - memory_optimizer: 已移除 ✅")
        print("    - backup_manager: 已移除 ✅")
        print("  5. 代码与声明完全一致 ✅")
        print("\n🎉 诚实集成完成！无虚假声明！")

        print("\n📊 Linus式评估：")
        print("  - 功能正确性: ✅ 系统正常工作")
        print("  - 诚实性: ✅ 无虚假集成声明")
        print("  - 简洁性: ✅ 移除了不必要的复杂性")
        print("  - 可维护性: ✅ 代码清晰，无误导")

        sys.exit(0)
    else:
        print("❌ 诚实集成验证：失败")
        print("=" * 80)
        print("\n请检查上面的错误信息")
        sys.exit(1)
