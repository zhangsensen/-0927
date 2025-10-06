#!/usr/bin/env python3
"""P0 优化演示 - 展示新架构的使用方式

Linus 式设计哲学：
1. Never break userspace - 完全向后兼容
2. 消除特殊情况 - 统一配置管理
3. 简洁实用 - 自动路径发现
"""

from hk_midfreq import FactorScoreLoader, PathConfig, PriceDataLoader


def demo_auto_path_discovery():
    """演示自动路径发现功能"""
    print("=" * 60)
    print("1. 自动路径发现（推荐用法）")
    print("=" * 60)

    # 创建配置 - 自动发现项目根目录
    path_config = PathConfig()
    print("\n自动发现的路径配置：")
    print(path_config)

    # 验证路径
    if path_config.validate_paths():
        print("✅ 所有关键路径验证通过")
    else:
        print("⚠️  某些路径不存在")

    # 使用新配置创建加载器
    loader = PriceDataLoader(path_config=path_config)
    print(f"✅ PriceDataLoader 创建成功，使用路径: {loader._get_root_dir()}")


def demo_backward_compatibility():
    """演示向后兼容性"""
    print("\n" + "=" * 60)
    print("2. 向后兼容性验证（原有代码继续工作）")
    print("=" * 60)

    # 原有用法仍然有效（使用默认配置）
    loader = PriceDataLoader()
    print(f"✅ 原有 API 继续工作，路径: {loader._get_root_dir()}")

    # 也可以显式指定路径（向后兼容）
    from pathlib import Path

    custom_loader = PriceDataLoader(root=Path("/custom/path"))
    print(f"✅ 支持自定义路径: {custom_loader._get_root_dir()}")


def demo_path_config_properties():
    """演示 PathConfig 的所有属性"""
    print("\n" + "=" * 60)
    print("3. 统一路径配置访问")
    print("=" * 60)

    path_config = PathConfig()

    print("\n所有可用路径：")
    print(f"  项目根目录:     {path_config.project_root}")
    print(f"  原始数据层:     {path_config.raw_data_dir}")
    print(f"  港股原始数据:   {path_config.hk_raw_dir}")
    print(f"  因子系统根目录: {path_config.factor_system_dir}")
    print(f"  因子输出层:     {path_config.factor_output_dir}")
    print(f"  因子筛选层:     {path_config.factor_screening_dir}")
    print(f"  优秀因子存储:   {path_config.factor_ready_dir}")


def demo_error_handling():
    """演示标准化错误处理"""
    print("\n" + "=" * 60)
    print("4. 标准化错误处理")
    print("=" * 60)

    from hk_midfreq import DataLoadError

    print("\n可用的标准化异常类：")
    print("  - DataLoadError:   价格数据加载错误")
    print("  - FactorLoadError: 因子数据加载错误")

    # 演示错误处理
    try:
        loader = PriceDataLoader()
        # 尝试加载不存在的数据
        loader.load_price("NONEXISTENT.HK", "60min")
    except DataLoadError as e:
        print(f"\n✅ 捕获标准化异常: {type(e).__name__}")
        print(f"   错误信息: {str(e)[:80]}...")


def demo_factor_interface():
    """演示因子接口的路径解耦"""
    print("\n" + "=" * 60)
    print("5. 因子接口路径解耦")
    print("=" * 60)

    # 新用法：使用 PathConfig
    path_config = PathConfig()
    factor_loader = FactorScoreLoader(path_config=path_config)
    print("✅ FactorScoreLoader 使用统一路径配置")
    print(f"   因子输出目录: {factor_loader._path_config.factor_output_dir}")

    # 原有用法仍然有效
    _ = FactorScoreLoader()
    print("✅ 向后兼容：原有 API 继续工作")


def main():
    """运行所有演示"""
    print("\n🎯 HK 中频交易架构 P0 优化演示")
    print("=" * 60)
    print("\nP0 优化内容：")
    print("  1. 统一配置管理（PathConfig）")
    print("  2. 路径解耦（消除硬编码）")
    print("  3. 错误处理标准化")
    print("  4. 完全向后兼容（Never break userspace）")

    try:
        demo_auto_path_discovery()
        demo_backward_compatibility()
        demo_path_config_properties()
        demo_error_handling()
        demo_factor_interface()

        print("\n" + "=" * 60)
        print("✅ P0 优化验证完成！")
        print("=" * 60)
        print("\n关键改进：")
        print("  ✅ 消除了硬编码路径")
        print("  ✅ 自动发现项目根目录")
        print("  ✅ 统一的异常处理")
        print("  ✅ 100% 向后兼容")
        print("\n符合 Linus 工程哲学：简洁、实用、高效")

    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
