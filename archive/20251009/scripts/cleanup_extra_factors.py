#!/usr/bin/env python3
"""
清理FactorEngine中多余的因子
"""

import os
import sys

sys.path.insert(0, "/Users/zhangshenshen/深度量化0927")


def identify_extra_factors():
    """识别多余的因子文件"""
    print("🔍 识别多余的因子文件...")

    # factor_generation中实际存在的因子
    valid_factors = {"RSI", "MACD", "STOCH"}

    # FactorEngine中的因子文件
    factors_dir = "/Users/zhangshenshen/深度量化0927/factor_system/factor_engine/factors/technical"

    if os.path.exists(factors_dir):
        factor_files = []
        for file in os.listdir(factors_dir):
            if file.endswith(".py") and file != "__init__.py":
                factor_name = file.replace(".py", "").upper()
                factor_files.append((file, factor_name))

        print(f"FactorEngine技术指标文件: {len(factor_files)} 个")

        extra_factors = []
        for file, name in factor_files:
            if name not in valid_factors:
                extra_factors.append((file, name))
                print(f"  ❌ 多余因子: {file} -> {name}")
            else:
                print(f"  ✅ 有效因子: {file} -> {name}")

        return extra_factors
    else:
        print(f"因子目录不存在: {factors_dir}")
        return []


def cleanup_extra_factors(extra_factors):
    """清理多余的因子文件"""
    print(f"\n🧹 清理多余的因子文件...")

    factors_dir = "/Users/zhangshenshen/深度量化0927/factor_system/factor_engine/factors/technical"

    for file, name in extra_factors:
        file_path = os.path.join(factors_dir, file)
        try:
            # 备份到临时目录
            backup_dir = "/Users/zhangshenshen/深度量化0927/backup_extra_factors"
            os.makedirs(backup_dir, exist_ok=True)
            backup_path = os.path.join(backup_dir, file)

            if os.path.exists(file_path):
                os.rename(file_path, backup_path)
                print(f"  ✅ 已备份并移除: {file} -> {backup_path}")

        except Exception as e:
            print(f"  ❌ 移除失败 {file}: {e}")


def update_init_file():
    """更新__init__.py文件"""
    print(f"\n📝 更新__init__.py文件...")

    init_file = "/Users/zhangshenshen/深度量化0927/factor_system/factor_engine/factors/technical/__init__.py"

    # 只保留有效的因子
    valid_imports = [
        "from .rsi import RSI",
        "from .macd import MACD",
        "from .stoch import STOCH",
    ]

    valid_all = ["RSI", "MACD", "STOCH"]

    try:
        with open(init_file, "w") as f:
            f.write('"""\n技术指标因子模块\n"""\n\n')
            for import_line in valid_imports:
                f.write(f"{import_line}\n")
            f.write(f"\n__all__ = {valid_all}\n")

        print(f"  ✅ 已更新__init__.py，只包含有效因子: {valid_all}")

    except Exception as e:
        print(f"  ❌ 更新__init__.py失败: {e}")


def verify_cleanup():
    """验证清理结果"""
    print(f"\n🔍 验证清理结果...")

    # 检查文件系统
    factors_dir = "/Users/zhangshenshen/深度量化0927/factor_system/factor_engine/factors/technical"

    remaining_files = []
    if os.path.exists(factors_dir):
        for file in os.listdir(factors_dir):
            if file.endswith(".py") and file != "__init__.py":
                remaining_files.append(file.replace(".py", "").upper())

    print(f"剩余的因子文件: {remaining_files}")

    # 检查注册表
    try:
        from factor_system.factor_engine.core.registry import get_global_registry
        from factor_system.factor_engine.factors.technical import MACD, RSI, STOCH

        registry = get_global_registry()
        registry.register(RSI)
        registry.register(MACD)
        registry.register(STOCH)

        all_factors = registry.list_factors()
        print(f"注册表中的因子: {sorted(all_factors)}")

        # 验证是否只包含有效因子
        valid_factors = {"RSI", "MACD", "STOCH"}
        unexpected_factors = set(all_factors) - valid_factors

        if unexpected_factors:
            print(f"❌ 仍有意外因子: {unexpected_factors}")
            return False
        else:
            print(f"✅ 清理验证通过，只包含有效因子")
            return True

    except Exception as e:
        print(f"❌ 验证失败: {e}")
        return False


if __name__ == "__main__":
    print("🚀 开始清理FactorEngine中多余的因子...")

    # 1. 识别多余因子
    extra_factors = identify_extra_factors()

    if extra_factors:
        print(f"\n发现 {len(extra_factors)} 个多余因子，需要清理")

        # 2. 清理多余因子
        cleanup_extra_factors(extra_factors)

        # 3. 更新__init__.py
        update_init_file()

        # 4. 验证清理结果
        success = verify_cleanup()

        if success:
            print(f"\n🎉 清理完成！FactorEngine现在只包含factor_generation中存在的因子")
        else:
            print(f"\n⚠️ 清理过程中出现问题，需要手动检查")
    else:
        print(f"\n✅ 没有发现多余因子，无需清理")
