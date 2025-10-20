#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
未来函数防护组件 - 综合测试套件
作者：量化首席工程师
版本：1.0.0
日期：2025-10-17

功能：
- 全面的未来函数防护组件测试
- 包含静态检查、运行时验证、健康监控测试
- 性能测试和异常处理验证
"""

import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# 导入被测试的组件
from factor_system.future_function_guard import (
    FileCache,
    FutureFunctionGuard,
    GuardConfig,
    HealthMonitor,
    RuntimeValidator,
    SimpleCache,
    StaticChecker,
    create_guard,
    development_guard,
    future_safe,
    monitor_health,
    production_guard,
    research_guard,
    validate_factors,
)
from factor_system.future_function_guard.exceptions import (
    CacheError,
    ConfigurationError,
    FutureFunctionGuardError,
    HealthMonitorError,
    RuntimeValidationError,
    StaticCheckError,
)


class TestFutureFunctionGuard:
    """未来函数防护组件测试类"""

    def __init__(self):
        self.test_results = []
        self.temp_files = []

    def cleanup(self):
        """清理临时文件"""
        for temp_file in self.temp_files:
            try:
                if Path(temp_file).exists():
                    Path(temp_file).unlink()
            except Exception:
                pass
        self.temp_files.clear()

    def run_test(self, test_name, test_func):
        """运行单个测试"""
        try:
            print(f"🧪 运行测试: {test_name}")
            result = test_func()
            if result:
                print(f"   ✅ {test_name} 通过")
                self.test_results.append((test_name, True))
            else:
                print(f"   ❌ {test_name} 失败")
                self.test_results.append((test_name, False))
        except Exception as e:
            print(f"   💥 {test_name} 异常: {e}")
            self.test_results.append((test_name, False))

    def test_configuration_management(self):
        """测试配置管理功能"""
        # 测试环境预设
        dev_config = GuardConfig.preset("development")
        research_config = GuardConfig.preset("research")
        prod_config = GuardConfig.preset("production")

        assert dev_config.mode == "development"
        assert research_config.mode == "research"
        assert prod_config.mode == "production"
        assert prod_config.runtime_validation.strict_mode.value == "enforced"
        assert prod_config.health_monitor.real_time_alerts == True

        # 测试配置序列化
        config_dict = research_config.to_dict()
        rebuilt_config = GuardConfig.from_dict(config_dict)
        assert rebuilt_config.mode == research_config.mode

        # 测试文件操作
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            research_config.save_to_file(f.name)
            self.temp_files.append(f.name)
            loaded_config = GuardConfig.load_from_file(f.name)
            assert loaded_config.mode == research_config.mode

        return True

    def test_static_checking(self):
        """测试静态检查功能"""
        checker = StaticChecker(GuardConfig.preset("research").static_check)

        # 创建测试文件
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
import pandas as pd

def normal_function(data):
    return data.rolling(5).mean()

def problematic_function(data):
    # 未来函数使用
    return data.shift(-1)  # This should be detected
"""
            )
            self.temp_files.append(f.name)

        # 测试文件检查
        result = checker.check_file(f.name)
        assert result["file_path"] == f.name
        assert result["issue_count"] >= 1  # 应该检测到shift(-1)

        # 测试缓存功能
        start_time = time.time()
        result1 = checker.check_file(f.name)
        first_time = time.time() - start_time

        start_time = time.time()
        result2 = checker.check_file(f.name)
        second_time = time.time() - start_time

        # 缓存应该提升性能
        assert result1["file_path"] == result2["file_path"]
        assert second_time <= first_time * 1.1  # 允许10%的误差

        return True

    def test_runtime_validation(self):
        """测试运行时验证功能"""
        validator = RuntimeValidator(GuardConfig.preset("research").runtime_validation)

        # 创建测试数据
        dates = pd.date_range("2025-01-01", periods=100, freq="D")
        normal_data = pd.Series(np.random.randn(100).cumsum(), index=dates)

        # 测试正常数据验证
        result = validator.validate_factor_calculation(
            normal_data, "test_factor", "daily", dates[0]
        )
        assert hasattr(result, "is_valid")
        assert hasattr(result, "message")

        # 测试空数据
        empty_data = pd.Series([], dtype=float)
        result = validator.validate_factor_calculation(
            empty_data, "empty_factor", "daily", pd.Timestamp("2025-01-01")
        )
        assert not result.is_valid

        return True

    def test_health_monitoring(self):
        """测试健康监控功能"""
        monitor = HealthMonitor(GuardConfig.preset("research").health_monitor)

        # 创建测试数据
        dates = pd.date_range("2025-01-01", periods=200, freq="D")

        # 健康数据
        healthy_data = pd.Series(np.random.randn(200).cumsum(), index=dates)
        result = monitor.check_factor_health(healthy_data, "healthy_factor")
        assert result.get_quality_score() > 90

        # 有问题的数据
        problematic_data = healthy_data.copy()
        problematic_data.iloc[50:100] = np.nan  # 25%缺失值
        problematic_data.iloc[150] = problematic_data.iloc[150] * 20  # 极值

        result = monitor.check_factor_health(problematic_data, "problematic_factor")
        assert result.get_quality_score() < 90

        return True

    def test_decorators(self):
        """测试装饰器功能"""

        @future_safe(strict_mode="warn_only")
        def safe_rsi_calculation(data, periods=14):
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        # 测试装饰器正常工作
        dates = pd.date_range("2025-01-01", periods=50, freq="D")
        test_data = pd.Series(np.random.randn(50).cumsum(), index=dates)
        result = safe_rsi_calculation(test_data)
        assert len(result) == len(test_data)

        return True

    def test_convenience_functions(self):
        """测试便捷函数"""
        # 测试便捷函数创建
        dev_guard = development_guard()
        research_guard_instance = research_guard()
        prod_guard = production_guard()

        assert dev_guard.config.mode == "development"
        assert research_guard_instance.config.mode == "research"
        assert prod_guard.config.mode == "production"

        # 测试validate_factors便捷函数
        dates = pd.date_range("2025-01-01", periods=100, freq="D")
        factor_data = pd.Series(np.random.randn(100).cumsum(), index=dates)
        result = validate_factors(factor_data, "test_factor")
        assert "is_valid" in result

        # 测试monitor_health便捷函数
        result = monitor_health(factor_data, "test_factor")
        assert "quality_score" in result

        return True

    def test_exception_handling(self):
        """测试异常处理"""
        # 测试配置文件不存在异常
        try:
            GuardConfig.load_from_file("nonexistent_file.json")
            return False
        except ConfigurationError:
            pass

        # 测试静态检查文件不存在异常
        checker = StaticChecker(GuardConfig.preset("research").static_check)
        try:
            checker.check_file("nonexistent_file.py")
            return False
        except StaticCheckError:
            pass

        # 测试所有异常类都有error_code
        exception_classes = [
            ConfigurationError,
            StaticCheckError,
            RuntimeValidationError,
            HealthMonitorError,
            CacheError,
        ]

        for exc_class in exception_classes:
            instance = exc_class("Test message")
            assert hasattr(instance, "error_code")
            assert instance.error_code is not None

        return True

    def test_caching_mechanisms(self):
        """测试缓存机制"""
        # 测试简单缓存基本功能
        cache = SimpleCache(max_size=10, ttl_hours=1)
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        assert cache.get("key1") == "value1"
        assert cache.get("nonexistent") is None

        # 测试文件缓存
        with tempfile.TemporaryDirectory() as temp_dir:
            config = GuardConfig.preset("research").cache
            file_cache = FileCache(temp_dir, config)

            test_data = {"numbers": [1, 2, 3, 4, 5]}
            file_cache.set("test_key", test_data)

            retrieved_data = file_cache.get("test_key")
            assert retrieved_data == test_data

        return True

    def test_performance(self):
        """测试性能"""
        # 运行时验证性能测试
        validator = RuntimeValidator(GuardConfig.preset("research").runtime_validation)

        sizes = [1000, 5000]
        for size in sizes:
            dates = pd.date_range("2020-01-01", periods=size, freq="D")
            data = pd.Series(np.random.randn(size).cumsum(), index=dates)

            start_time = time.time()
            result = validator.validate_factor_calculation(
                data, f"test_factor_{size}", "daily"
            )
            validation_time = time.time() - start_time

            throughput = size / validation_time
            assert throughput > 100000  # 至少10万数据点/秒

        # 健康监控性能测试
        monitor = HealthMonitor(GuardConfig.preset("research").health_monitor)
        dates = pd.date_range("2025-01-01", periods=1000, freq="D")
        data = pd.Series(np.random.randn(1000).cumsum(), index=dates)

        start_time = time.time()
        result = monitor.check_factor_health(data, "performance_test_factor")
        processing_time = time.time() - start_time

        throughput = 1000 / processing_time
        assert throughput > 100000  # 至少10万数据点/秒

        return True

    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始未来函数防护组件综合测试")
        print("=" * 60)

        # 运行各项测试
        self.run_test("配置管理", self.test_configuration_management)
        self.run_test("静态检查", self.test_static_checking)
        self.run_test("运行时验证", self.test_runtime_validation)
        self.run_test("健康监控", self.test_health_monitoring)
        self.run_test("装饰器功能", self.test_decorators)
        self.run_test("便捷函数", self.test_convenience_functions)
        self.run_test("异常处理", self.test_exception_handling)
        self.run_test("缓存机制", self.test_caching_mechanisms)
        self.run_test("性能测试", self.test_performance)

        # 清理临时文件
        self.cleanup()

        # 汇总结果
        print("\n" + "=" * 60)
        print("📊 测试结果汇总:")

        passed = 0
        total = len(self.test_results)

        for test_name, result in self.test_results:
            status = "✅ 通过" if result else "❌ 失败"
            print(f"  {test_name:12s}: {status}")
            if result:
                passed += 1

        success_rate = (passed / total) * 100
        print(f"\n🎯 总体结果: {passed}/{total} 项测试通过 ({success_rate:.1f}%)")

        if passed == total:
            print("🎉 所有测试通过！未来函数防护组件运行正常。")
            return True
        else:
            print("⚠️  部分测试失败，请检查相关功能。")
            return False


def main():
    """主函数"""
    tester = TestFutureFunctionGuard()
    try:
        return tester.run_all_tests()
    finally:
        tester.cleanup()


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
