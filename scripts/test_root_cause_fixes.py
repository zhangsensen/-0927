#!/usr/bin/env python3
"""
端到端测试验证脚本 - 根本问题修复验证

测试目标：
1. 验证统一数据接口工作正常
2. 验证标准因子ID格式
3. 验证简化的缓存系统
4. 验证策略集成无别名解析
5. 确保所有胶水代码已被删除
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# 设置项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RootCauseFixValidator:
    """根本问题修复验证器"""

    def __init__(self):
        self.test_results = {
            'passed': 0,
            'failed': 0,
            'total': 0,
            'errors': []
        }

    def test_parquet_data_provider(self) -> bool:
        """测试1: ParquetDataProvider统一接口"""
        logger.info("测试1: ParquetDataProvider统一接口")

        try:
            from factor_system.factor_engine.providers.parquet_provider import ParquetDataProvider

            # 测试初始化 - 应该成功（假设已运行数据迁移）
            provider = ParquetDataProvider(Path("raw"))
            logger.info("✓ ParquetDataProvider初始化成功")

            # 测试数据加载 - 应该使用统一格式
            try:
                data = provider.load_price_data(
                    symbols=["0700.HK"],
                    timeframe="15min",
                    start_date=datetime(2025, 9, 1),
                    end_date=datetime(2025, 9, 30),
                )

                if not data.empty:
                    # 验证数据schema
                    expected_columns = {"open", "high", "low", "close", "volume"}
                    actual_columns = set(data.columns)
                    if expected_columns.issubset(actual_columns):
                        logger.info("✓ 数据schema验证通过")
                    else:
                        raise ValueError(f"数据schema不匹配: 期望{expected_columns}, 实际{actual_columns}")

                    # 验证MultiIndex格式
                    if isinstance(data.index, pd.MultiIndex):
                        logger.info("✓ MultiIndex格式正确")
                    else:
                        raise ValueError("Index格式错误，应为MultiIndex")

                else:
                    logger.warning("⚠ 数据为空，可能是数据未迁移")

            except Exception as e:
                if "HK数据目录不存在" in str(e) or "PyArrow是必需的依赖" in str(e):
                    logger.info("⚠ 数据未迁移，跳过数据加载测试")
                else:
                    raise e

            # 测试错误处理 - 应该直接失败，无回退机制
            try:
                # 错误的symbol格式
                provider.load_price_data(
                    symbols=["0700"],  # 错误格式
                    timeframe="15min",
                    start_date=datetime(2025, 9, 1),
                    end_date=datetime(2025, 9, 30),
                )
                raise AssertionError("应该抛出ValueError")
            except ValueError:
                logger.info("✓ 错误处理正确：无回退机制")

            try:
                # 错误的timeframe格式
                provider.load_price_data(
                    symbols=["0700.HK"],
                    timeframe="15m",  # 错误格式
                    start_date=datetime(2025, 9, 1),
                    end_date=datetime(2025, 9, 30),
                )
                raise AssertionError("应该抛出ValueError")
            except ValueError:
                logger.info("✓ 错误处理正确：无回退机制")

            return True

        except Exception as e:
            logger.error(f"✗ ParquetDataProvider测试失败: {e}")
            self.test_results['errors'].append(f"ParquetDataProvider: {e}")
            return False

    def test_factor_registry(self) -> bool:
        """测试2: FactorRegistry无别名解析"""
        logger.info("测试2: FactorRegistry无别名解析")

        try:
            from factor_system.factor_engine.core.registry import FactorRegistry

            registry = FactorRegistry()

            # 测试因子ID格式验证
            try:
                # 尝试注册带参数的因子名 - 应该失败
                class BadFactor:
                    factor_id = "RSI_14"  # 错误格式

                registry.register(BadFactor)
                raise AssertionError("应该抛出ValueError")
            except ValueError as e:
                if "因子ID不能包含参数后缀" in str(e):
                    logger.info("✓ 因子ID格式验证正确")
                else:
                    raise e

            try:
                # 尝试注册带TA_前缀的因子名 - 应该失败
                class BadFactor2:
                    factor_id = "TA_RSI"  # 错误格式

                registry.register(BadFactor2)
                raise AssertionError("应该抛出ValueError")
            except ValueError as e:
                if "因子ID不能使用TA_前缀" in str(e):
                    logger.info("✓ TA_前缀验证正确")
                else:
                    raise e

            # 测试标准因子注册
            class GoodFactor:
                factor_id = "RSI"  # 正确格式

            registry.register(GoodFactor)
            logger.info("✓ 标准因子注册成功")

            # 测试因子获取 - 不支持别名
            factor_instance = registry.get_factor("RSI")
            logger.info("✓ 标准因子获取成功")

            try:
                # 尝试获取别名 - 应该失败
                registry.get_factor("RSI_14")  # 应该失败
                raise AssertionError("应该抛出ValueError")
            except ValueError as e:
                if "未注册的因子" in str(e):
                    logger.info("✓ 别名解析已删除")
                else:
                    raise e

            # 测试新的因子请求创建方法
            configs = [{'factor_id': 'RSI', 'parameters': {'timeperiod': 14}}]
            requests = registry.create_factor_requests(configs)
            logger.info("✓ 标准化因子请求创建成功")

            return True

        except Exception as e:
            logger.error(f"✗ FactorRegistry测试失败: {e}")
            self.test_results['errors'].append(f"FactorRegistry: {e}")
            return False

    def test_cache_manager(self) -> bool:
        """测试3: CacheManager简化缓存键"""
        logger.info("测试3: CacheManager简化缓存键")

        try:
            from factor_system.factor_engine.core.cache import CacheManager
            from factor_system.factor_engine.core.registry import FactorRequest

            cache = CacheManager()

            # 创建测试数据
            test_data = pd.DataFrame({
                'timestamp': pd.date_range('2025-09-01', periods=10),
                'symbol': ['0700.HK'] * 10,
                'RSI': [50.0] * 10
            }).set_index(['timestamp', 'symbol'])

            # 创建因子请求
            requests = [
                FactorRequest(factor_id='RSI', parameters={'timeperiod': 14})
            ]

            # 测试缓存设置
            cache.set(
                test_data,
                requests,
                ['0700.HK'],
                '15min',
                datetime(2025, 9, 1),
                datetime(2025, 9, 30)
            )
            logger.info("✓ 缓存设置成功")

            # 测试缓存获取
            cached_data, missing_ids = cache.get(
                requests,
                ['0700.HK'],
                '15min',
                datetime(2025, 9, 1),
                datetime(2025, 9, 30)
            )

            if cached_data is not None and not cached_data.empty:
                logger.info("✓ 缓存获取成功")
            else:
                raise ValueError("缓存获取失败")

            # 验证缓存键格式简化（不再是复杂JSON）
            # 通过内部方法验证键的生成
            cache_key = cache._build_key(
                requests,
                ['0700.HK'],
                '15min',
                datetime(2025, 9, 1),
                datetime(2025, 9, 30)
            )

            # 简化的键应该是MD5哈希（32字符）
            if len(cache_key) == 32:
                logger.info("✓ 缓存键格式已简化")
            else:
                raise ValueError(f"缓存键格式错误: {cache_key}")

            # 测试深拷贝行为
            cached_data.iloc[0, 0] = 999.0
            fresh_data, _ = cache.get(
                requests,
                ['0700.HK'],
                '15min',
                datetime(2025, 9, 1),
                datetime(2025, 9, 30)
            )

            if fresh_data.iloc[0, 0] != 999.0:
                logger.info("✓ 深拷贝行为正确")
            else:
                raise ValueError("深拷贝行为错误")

            return True

        except Exception as e:
            logger.error(f"✗ CacheManager测试失败: {e}")
            self.test_results['errors'].append(f"CacheManager: {e}")
            return False

    def test_strategy_core(self) -> bool:
        """测试4: StrategyCore无列映射逻辑"""
        logger.info("测试4: StrategyCore无列映射逻辑")

        try:
            # 由于StrategyCore依赖完整的数据流，这里主要测试逻辑验证
            from hk_midfreq.strategy_core import generate_factor_signals, FactorDescriptor

            # 测试因子ID格式验证
            try:
                descriptor = FactorDescriptor(name="RSI_14", timeframe="15min")  # 错误格式
                generate_factor_signals(
                    symbol="0700.HK",
                    timeframe="15min",
                    close=pd.Series(),
                    volume=None,
                    descriptor=descriptor,
                    hold_days=5,
                    stop_loss=0.05,
                    take_profit=0.15
                )
                raise AssertionError("应该抛出ValueError")
            except ValueError as e:
                if "因子ID格式无效" in str(e):
                    logger.info("✓ StrategyCore因子ID格式验证正确")
                else:
                    raise e

            # 测试未注册因子处理
            try:
                descriptor = FactorDescriptor(name="UNKNOWN_FACTOR", timeframe="15min")
                # 这里会因为因子未注册而失败，但需要完整的FactorEngine
                logger.info("⚠ 需要完整环境测试未注册因子处理")
            except Exception:
                logger.info("✓ 未注册因子验证逻辑存在")

            return True

        except Exception as e:
            logger.error(f"✗ StrategyCore测试失败: {e}")
            self.test_results['errors'].append(f"StrategyCore: {e}")
            return False

    def test_no_glue_code(self) -> bool:
        """测试5: 确保胶水代码已删除"""
        logger.info("测试5: 检查胶水代码删除")

        try:
            # 检查ParquetDataProvider中是否还有pandas回退
            provider_file = project_root / "factor_system/factor_engine/providers/parquet_provider.py"
            provider_content = provider_file.read_text()

            if "_load_with_pandas" in provider_content:
                raise ValueError("ParquetDataProvider仍包含pandas回退代码")

            if "_normalize_symbol" in provider_content:
                raise ValueError("ParquetDataProvider仍包含符号标准化代码")

            if "_normalize_timeframe" in provider_content:
                raise ValueError("ParquetDataProvider仍包含时间框架标准化代码")

            logger.info("✓ ParquetDataProvider胶水代码已删除")

            # 检查FactorRegistry中是否还有别名解析
            registry_file = project_root / "factor_system/factor_engine/core/registry.py"
            registry_content = registry_file.read_text()

            if "_resolve_alias" in registry_content:
                raise ValueError("FactorRegistry仍包含别名解析代码")

            if "_generate_alias_candidates" in registry_content:
                raise ValueError("FactorRegistry仍包含别名候选生成代码")

            if "_aliases" in registry_content:
                raise ValueError("FactorRegistry仍包含别名存储")

            logger.info("✓ FactorRegistry胶水代码已删除")

            # 检查CacheManager中是否还有复杂的copy_mode
            cache_file = project_root / "factor_system/factor_engine/core/cache.py"
            cache_content = cache_file.read_text()

            if "copy_mode" in cache_file.read_text():
                raise ValueError("CacheManager仍包含copy_mode配置")

            logger.info("✓ CacheManager胶水代码已删除")

            # 检查StrategyCore中是否还有列映射逻辑
            strategy_file = project_root / "hk_midfreq/strategy_core.py"
            strategy_content = strategy_file.read_text()

            if "column_name = factor_id if factor_id in factor_df.columns else target_id" in strategy_content:
                raise ValueError("StrategyCore仍包含列映射逻辑")

            if "resolve_factor_requests" in strategy_content:
                raise ValueError("StrategyCore仍使用已删除的别名解析方法")

            logger.info("✓ StrategyCore胶水代码已删除")

            return True

        except Exception as e:
            logger.error(f"✗ 胶水代码检查失败: {e}")
            self.test_results['errors'].append(f"胶水代码检查: {e}")
            return False

    def run_all_tests(self) -> bool:
        """运行所有测试"""
        logger.info("=" * 60)
        logger.info("开始根本问题修复验证测试")
        logger.info("=" * 60)

        tests = [
            self.test_parquet_data_provider,
            self.test_factor_registry,
            self.test_cache_manager,
            self.test_strategy_core,
            self.test_no_glue_code,
        ]

        all_passed = True

        for test_func in tests:
            self.test_results['total'] += 1
            try:
                if test_func():
                    self.test_results['passed'] += 1
                else:
                    self.test_results['failed'] += 1
                    all_passed = False
                print()  # 空行分隔
            except Exception as e:
                logger.error(f"测试异常: {e}")
                self.test_results['failed'] += 1
                self.test_results['errors'].append(f"{test_func.__name__}: {e}")
                all_passed = False
                print()

        # 输出总结
        logger.info("=" * 60)
        logger.info("测试总结")
        logger.info("=" * 60)
        logger.info(f"总测试数: {self.test_results['total']}")
        logger.info(f"通过: {self.test_results['passed']}")
        logger.info(f"失败: {self.test_results['failed']}")

        if self.test_results['errors']:
            logger.error("错误详情:")
            for error in self.test_results['errors']:
                logger.error(f"  - {error}")

        if all_passed:
            logger.info("🎉 所有测试通过！根本问题修复成功！")
        else:
            logger.error("❌ 部分测试失败，需要进一步修复")

        return all_passed


def main():
    """主函数"""
    validator = RootCauseFixValidator()
    success = validator.run_all_tests()

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()