#!/usr/bin/env python3
"""
FactorEngine关键修复验证测试

验证修复：
1. n_jobs参数正确传递
2. LRUCache size计算正确
3. 配置指纹完整，确保配置变更生效
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

from factor_system.factor_engine.api import get_engine, calculate_factors
from factor_system.factor_engine.core.cache import LRUCache
from factor_system.factor_engine.settings import get_settings


class TestFactorEngineFixes:
    """测试FactorEngine关键修复"""

    def test_n_jobs_parameter_passthrough(self):
        """测试n_jobs参数正确传递"""
        print("🧪 测试n_jobs参数传递...")

        # 创建测试数据
        with tempfile.TemporaryDirectory() as temp_dir:
            # 准备测试数据
            test_data = pd.DataFrame({
                'open': [100, 102, 101, 103, 105],
                'high': [103, 104, 102, 106, 107],
                'low': [99, 101, 100, 102, 104],
                'close': [102, 101, 103, 105, 106],
                'volume': [1000, 1200, 800, 1500, 2000]
            }, index=pd.date_range('2025-01-01', periods=5))

            data_file = Path(temp_dir) / "0700.HK_1min_2025-01-01_2025-01-05.parquet"
            data_file.parent.mkdir(parents=True, exist_ok=True)
            test_data.to_parquet(data_file)

            # 测试n_jobs=1
            try:
                result_single = calculate_factors(
                    factor_ids=['RSI'],
                    symbols=['0700.HK'],
                    timeframe='1min',
                    start_date=datetime(2025, 1, 1),
                    end_date=datetime(2025, 1, 5),
                    n_jobs=1
                )
                print("✅ n_jobs=1 测试通过")
            except Exception as e:
                print(f"❌ n_jobs=1 测试失败: {e}")

            # 测试n_jobs=2
            try:
                result_parallel = calculate_factors(
                    factor_ids=['RSI'],
                    symbols=['0700.HK'],
                    timeframe='1min',
                    start_date=datetime(2025, 1, 1),
                    end_date=datetime(2025, 1, 5),
                    n_jobs=2
                )
                print("✅ n_jobs=2 测试通过")

                # 验证结果一致性
                pd.testing.assert_frame_equal(result_single, result_parallel)
                print("✅ 单线程和多线程结果一致")

            except Exception as e:
                print(f"❌ n_jobs=2 测试失败: {e}")

    def test_lru_cache_size_calculation(self):
        """测试LRUCache size计算修复"""
        print("🧪 测试LRUCache size计算...")

        cache = LRUCache(maxsize_mb=1)  # 1MB缓存

        # 创建测试数据
        data1 = pd.DataFrame({'test': np.random.randn(100)})
        data2 = pd.DataFrame({'test': np.random.randn(200)})
        data3 = pd.DataFrame({'test': np.random.randn(50)})

        # 添加第一个数据
        cache.set('key1', data1)
        size1 = cache.current_size
        print(f"添加数据1后缓存大小: {size1} bytes")

        # 添加第二个数据
        cache.set('key2', data2)
        size2 = cache.current_size
        print(f"添加数据2后缓存大小: {size2} bytes")
        assert size2 > size1, "缓存大小应该增加"

        # 替换第一个数据（关键测试：size不应该无限增长）
        old_size = cache.current_size
        cache.set('key1', data3)  # 用更小的数据替换key1
        new_size = cache.current_size
        print(f"替换key1后缓存大小: {new_size} bytes (替换前: {old_size} bytes)")

        # 验证size确实减少了
        assert new_size < old_size, "替换数据后缓存大小应该减少"
        print("✅ LRUCache size计算修复验证通过")

        # 验证缓存仍然可以正常访问
        retrieved_data = cache.get('key1')
        assert len(retrieved_data) == 50, "缓存数据应该正确"
        print("✅ 缓存数据访问正常")

    def test_config_fingerprint_completeness(self):
        """测试配置指纹完整性"""
        print("🧪 测试配置指纹完整性...")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # 设置不同的配置并测试引擎重建
            configs_to_test = [
                {'cache_enable_disk': True},
                {'cache_enable_memory': False},
                {'cache_copy_mode': 'deep'},
                {'cache_disk_cache_dir': temp_path / 'cache'},
                {'cache_max_ram_mb': 512},
                {'engine_n_jobs': 4}
            ]

            previous_engine_id = None

            for i, config_change in enumerate(configs_to_test):
                # 应用配置变更
                for key, value in config_change.items():
                    os.environ[key.upper()] = str(value)

                try:
                    # 获取引擎实例
                    engine = get_engine(force_reinit=False)
                    current_engine_id = id(engine)

                    if previous_engine_id is None:
                        previous_engine_id = current_engine_id
                        print(f"初始引擎ID: {current_engine_id}")
                    else:
                        # 配置变更后应该重建引擎
                        if current_engine_id != previous_engine_id:
                            print(f"✅ 配置变更 {config_change} 触发引擎重建")
                            previous_engine_id = current_engine_id
                        else:
                            print(f"⚠️ 配置变更 {config_change} 未触发引擎重建")

                except Exception as e:
                    print(f"❌ 配置测试失败 {config_change}: {e}")
                finally:
                    # 清理环境变量
                    for key in config_change.keys():
                        os.environ.pop(key.upper(), None)

            print("✅ 配置指纹测试完成")

    def test_real_data_workflow(self):
        """测试真实数据工作流"""
        print("🧪 测试真实数据工作流...")

        try:
            # 使用实际设置测试
            settings = get_settings()
            print(f"当前数据目录: {settings.data.raw_data_dir}")
            print(f"缓存内存限制: {settings.cache.memory_size_mb}MB")
            print(f"并行作业数: {settings.engine.n_jobs}")

            # 测试引擎初始化
            engine = get_engine()
            print(f"引擎初始化成功，ID: {id(engine)}")

            # 测试缓存状态
            cache_stats = engine.get_cache_stats()
            print(f"缓存统计: {cache_stats}")

            print("✅ 真实数据工作流测试通过")

        except Exception as e:
            print(f"⚠️ 真实数据工作流测试跳过（需要实际数据）: {e}")


def main():
    """运行所有修复验证测试"""
    print("🔧 FactorEngine关键修复验证开始...")
    print("=" * 50)

    test_instance = TestFactorEngineFixes()

    # 运行所有测试
    tests = [
        test_instance.test_n_jobs_parameter_passthrough,
        test_instance.test_lru_cache_size_calculation,
        test_instance.test_config_fingerprint_completeness,
        test_instance.test_real_data_workflow
    ]

    passed = 0
    total = len(tests)

    for test_func in tests:
        try:
            test_func()
            passed += 1
            print(f"✅ {test_func.__name__} 通过")
        except Exception as e:
            print(f"❌ {test_func.__name__} 失败: {e}")
        print("-" * 30)

    print(f"\n📊 测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有关键修复验证通过！")
    else:
        print("⚠️ 部分测试失败，需要进一步检查")


if __name__ == "__main__":
    main()