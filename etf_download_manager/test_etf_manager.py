#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF下载管理器测试脚本
用于验证系统功能是否正常
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """测试模块导入"""
    print("=== 测试模块导入 ===")

    try:
        from etf_download_manager import (
            ETFConfig,
            ETFDownloadManager,
            ETFDownloadType,
            ETFListManager,
        )

        print("✅ 核心模块导入成功")

        from etf_download_manager.config import load_config, setup_environment

        print("✅ 配置模块导入成功")

        from etf_download_manager.core.models import DownloadResult, ETFInfo

        print("✅ 模型模块导入成功")

        return True

    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return False


def test_config():
    """测试配置功能"""
    print("\n=== 测试配置功能 ===")

    try:
        from etf_download_manager import ETFConfig

        # 测试默认配置
        config = ETFConfig()
        print(f"✅ 默认配置创建成功")
        print(f"   数据源: {config.source.value}")
        print(f"   数据目录: {config.base_dir}")
        print(f"   下载类型: {[dt.value for dt in config.download_types]}")

        return True

    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        return False


def test_etf_list():
    """测试ETF清单功能"""
    print("\n=== 测试ETF清单功能 ===")

    try:
        from etf_download_manager import ETFListManager

        list_manager = ETFListManager()

        # 测试获取ETF
        all_etfs = list_manager.get_all_etfs()
        print(f"✅ 获取所有ETF: {len(all_etfs)}只")

        core_etfs = list_manager.get_must_have_etfs()
        print(f"✅ 获取核心ETF: {len(core_etfs)}只")

        if core_etfs:
            print(f"   示例核心ETF: {core_etfs[0].code} - {core_etfs[0].name}")

        # 测试筛选功能
        high_priority_etfs = list_manager.filter_etfs(priorities=["core", "must_have"])
        print(f"✅ 筛选高优先级ETF: {len(high_priority_etfs)}只")

        return True

    except Exception as e:
        print(f"❌ ETF清单测试失败: {e}")
        return False


def test_data_manager():
    """测试数据管理功能"""
    print("\n=== 测试数据管理功能 ===")

    try:
        from etf_download_manager import ETFConfig, ETFListManager
        from etf_download_manager.core.data_manager import ETFDataManager

        config = ETFConfig(base_dir="test_raw/ETF")
        data_manager = ETFDataManager(config)

        print(f"✅ 数据管理器创建成功")
        print(f"   数据目录: {config.data_dir}")
        print(f"   日线目录: {config.daily_dir}")
        print(f"   摘要目录: {config.summary_dir}")

        # 测试目录创建
        config.create_directories()
        print("✅ 数据目录创建成功")

        # 清理测试目录
        import shutil

        if Path("test_raw").exists():
            shutil.rmtree("test_raw")
            print("✅ 测试目录清理完成")

        return True

    except Exception as e:
        print(f"❌ 数据管理器测试失败: {e}")
        return False


def test_environment():
    """测试环境设置"""
    print("\n=== 测试环境设置 ===")

    try:
        from etf_download_manager.config.etf_config import setup_environment

        result = setup_environment()
        if result:
            print("✅ 环境检查通过")
            return True
        else:
            print("❌ 环境检查发现问题")
            return False

    except Exception as e:
        print(f"❌ 环境测试失败: {e}")
        return False


def test_token():
    """测试Token设置"""
    print("\n=== 测试Token设置 ===")

    token = os.getenv("TUSHARE_TOKEN")
    if token:
        print(f"✅ Token已设置 (长度: {len(token)})")
        print(f"   Token预览: {token[:8]}...{token[-8:]}")
        return True
    else:
        print("❌ Token未设置")
        print("   请设置环境变量: export TUSHARE_TOKEN='your_token'")
        return False


def test_tushare_connection():
    """测试Tushare连接"""
    print("\n=== 测试Tushare连接 ===")

    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        print("❌ Token未设置，跳过Tushare连接测试")
        return False

    try:
        import tushare as ts

        # 测试API连接
        pro = ts.pro_api(token)

        # 测试获取交易日历（轻量级API）
        df = pro.trade_cal(exchange="SSE", start_date="20240101", end_date="20240105")

        if df is not None and not df.empty:
            print(f"✅ Tushare连接成功")
            print(f"   获取交易日历: {len(df)}条记录")
            return True
        else:
            print("❌ Tushare返回空数据")
            return False

    except Exception as e:
        print(f"❌ Tushare连接失败: {e}")
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("ETF下载管理器测试")
    print("=" * 60)

    tests = [
        ("模块导入", test_imports),
        ("配置功能", test_config),
        ("ETF清单", test_etf_list),
        ("数据管理", test_data_manager),
        ("环境设置", test_environment),
        ("Token设置", test_token),
        ("Tushare连接", test_tushare_connection),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
            results.append((test_name, False))

    # 显示测试结果摘要
    print("\n" + "=" * 60)
    print("测试结果摘要")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:<15} {status}")
        if result:
            passed += 1

    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("🎉 所有测试通过！ETF下载管理器可以正常使用。")
        return True
    else:
        print("⚠️  部分测试失败，请检查相关问题。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
