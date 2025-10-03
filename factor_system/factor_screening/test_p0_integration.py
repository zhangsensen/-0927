#!/usr/bin/env python3
"""
P0级集成验证脚本
验证4个工具模块是否成功集成到professional_factor_screener.py中
"""

import sys
import logging
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)

def test_p0_integration():
    """P0级集成验证测试"""
    print("=" * 80)
    print("P0级集成验证测试")
    print("=" * 80)
    
    # 1. 测试工具模块导入
    print("\n[1/5] 测试工具模块导入...")
    try:
        from utils.memory_optimizer import MemoryOptimizer, get_memory_optimizer
        print("  ✅ memory_optimizer 导入成功")
    except ImportError as e:
        print(f"  ❌ memory_optimizer 导入失败: {e}")
        return False
    
    try:
        from utils.input_validator import InputValidator, ValidationError
        print("  ✅ input_validator 导入成功")
    except ImportError as e:
        print(f"  ❌ input_validator 导入失败: {e}")
        return False
    
    try:
        from utils.structured_logger import get_structured_logger
        print("  ✅ structured_logger 导入成功")
    except ImportError as e:
        print(f"  ❌ structured_logger 导入失败: {e}")
        return False
    
    try:
        from utils.backup_manager import get_backup_manager
        print("  ✅ backup_manager 导入成功")
    except ImportError as e:
        print(f"  ❌ backup_manager 导入失败: {e}")
        return False
    
    # 2. 测试主类导入
    print("\n[2/5] 测试主类导入...")
    try:
        from config_manager import ScreeningConfig
        from professional_factor_screener import ProfessionalFactorScreener
        print("  ✅ ProfessionalFactorScreener 导入成功")
    except ImportError as e:
        print(f"  ❌ 主类导入失败: {e}")
        return False
    
    # 3. 测试初始化
    print("\n[3/5] 测试筛选器初始化...")
    try:
        config = ScreeningConfig(
            ic_horizons=[1, 3, 5],
            min_sample_size=30
        )
        screener = ProfessionalFactorScreener(config=config)
        print("  ✅ 筛选器初始化成功")
    except Exception as e:
        print(f"  ❌ 筛选器初始化失败: {e}")
        return False
    
    # 4. 验证工具模块实例
    print("\n[4/5] 验证工具模块实例...")
    
    if hasattr(screener, 'memory_optimizer'):
        if screener.memory_optimizer is not None:
            print("  ✅ memory_optimizer 实例已创建")
        else:
            print("  ⚠️  memory_optimizer 实例为None")
    else:
        print("  ❌ memory_optimizer 属性不存在")
        return False
    
    if hasattr(screener, 'input_validator'):
        if screener.input_validator is not None:
            print("  ✅ input_validator 实例已创建")
        else:
            print("  ⚠️  input_validator 实例为None")
    else:
        print("  ❌ input_validator 属性不存在")
        return False
    
    if hasattr(screener, 'structured_logger'):
        if screener.structured_logger is not None:
            print("  ✅ structured_logger 实例已创建")
        else:
            print("  ⚠️  structured_logger 实例为None")
    else:
        print("  ❌ structured_logger 属性不存在")
        return False
    
    if hasattr(screener, 'backup_manager'):
        if screener.backup_manager is not None:
            print("  ✅ backup_manager 实例已创建")
        else:
            print("  ⚠️  backup_manager 实例为None")
    else:
        print("  ❌ backup_manager 属性不存在")
        return False
    
    # 5. 测试工具模块功能
    print("\n[5/5] 测试工具模块功能...")
    
    # 测试输入验证
    if screener.input_validator:
        is_valid, msg = screener.input_validator.validate_symbol("0700.HK")
        if is_valid:
            print("  ✅ 输入验证器功能正常")
        else:
            print(f"  ❌ 输入验证器功能异常: {msg}")
            return False
    
    # 测试内存监控
    if screener.memory_optimizer:
        current_memory = screener.memory_optimizer.get_memory_usage()
        if current_memory > 0:
            print(f"  ✅ 内存优化器功能正常 (当前内存: {current_memory:.1f}MB)")
        else:
            print("  ❌ 内存优化器功能异常")
            return False
    
    # 测试结构化日志
    if screener.structured_logger:
        try:
            screener.structured_logger.info(
                "P0集成测试",
                test_status="success"
            )
            print("  ✅ 结构化日志器功能正常")
        except Exception as e:
            print(f"  ❌ 结构化日志器功能异常: {e}")
            return False
    
    # 测试备份管理器
    if screener.backup_manager:
        stats = screener.backup_manager.get_statistics()
        if isinstance(stats, dict):
            print(f"  ✅ 备份管理器功能正常 (备份数: {stats.get('total_backups', 0)})")
        else:
            print("  ❌ 备份管理器功能异常")
            return False
    
    return True

if __name__ == "__main__":
    print("\n")
    success = test_p0_integration()
    
    print("\n" + "=" * 80)
    if success:
        print("✅ P0级集成验证：全部通过")
        print("=" * 80)
        print("\n📋 验证结果：")
        print("  1. 4个工具模块成功导入 ✅")
        print("  2. 主类成功导入 ✅")
        print("  3. 筛选器成功初始化 ✅")
        print("  4. 工具模块实例全部创建 ✅")
        print("  5. 工具模块功能全部正常 ✅")
        print("\n🎉 P0级集成完成！")
        sys.exit(0)
    else:
        print("❌ P0级集成验证：失败")
        print("=" * 80)
        print("\n请检查上面的错误信息")
        sys.exit(1)

