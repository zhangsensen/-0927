#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF横截面因子系统全面冒烟测试
验证统一管理器的所有功能，包括800-1200个动态因子的完整集成
"""

import sys
import os
import time
import traceback
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import pandas as pd

# 添加项目路径
sys.path.insert(0, '/Users/zhangshenshen/深度量化0927')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/Users/zhangshenshen/深度量化0927/scripts/comprehensive_smoke_test.log')
    ]
)

logger = logging.getLogger(__name__)


class ComprehensiveSmokeTest:
    """全面冒烟测试类"""

    def __init__(self):
        """初始化测试环境"""
        self.start_time = time.time()
        self.test_results = {}
        self.total_tests = 6
        self.passed_tests = 0

        logger.info("=" * 60)
        logger.info("ETF横截面因子系统全面冒烟测试")
        logger.info("=" * 60)

    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        test_methods = [
            self.test_unified_manager_import,
            self.test_dynamic_factor_registration,
            self.test_factor_calculation,
            self.test_cross_section_building,
            self.test_performance_requirements,
            self.test_end_to_end_workflow
        ]

        for i, test_method in enumerate(test_methods, 1):
            logger.info(f"\n测试 {i}/{self.total_tests}: {test_method.__doc__}")
            logger.info("-" * 40)

            try:
                result = test_method()
                self.test_results[test_method.__name__] = {
                    'status': 'PASSED' if result else 'FAILED',
                    'result': result,
                    'error': None
                }
                if result:
                    self.passed_tests += 1
                    logger.info(f"✅ 测试通过")
                else:
                    logger.error(f"❌ 测试失败")

            except Exception as e:
                self.test_results[test_method.__name__] = {
                    'status': 'ERROR',
                    'result': False,
                    'error': str(e)
                }
                logger.error(f"❌ 测试异常: {str(e)}")
                logger.error(f"详细错误: {traceback.format_exc()}")

        # 生成测试报告
        return self.generate_test_report()

    def test_unified_manager_import(self) -> bool:
        """测试统一管理器导入和初始化"""
        try:
            logger.info("  导入统一管理器...")
            from factor_system.factor_engine.factors.etf_cross_section import (
                ETFCrossSectionUnifiedManager,
                create_etf_cross_section_manager,
                ETFCrossSectionConfig
            )

            logger.info("  导入接口定义...")
            from factor_system.factor_engine.factors.etf_cross_section.interfaces import (
                IFactorCalculator,
                ICrossSectionManager,
                FactorCalculationResult,
                CrossSectionResult
            )

            logger.info("  创建管理器实例...")
            # 使用工厂函数创建
            manager1 = create_etf_cross_section_manager(verbose=True)
            logger.info("    ✅ 工厂函数创建成功")

            # 使用配置创建
            config = ETFCrossSectionConfig()
            config.verbose = True
            config.max_dynamic_factors = 1000  # 限制因子数量以提高测试速度

            from factor_system.factor_engine.factors.etf_cross_section.unified_manager import DefaultProgressMonitor
            progress_monitor = DefaultProgressMonitor(verbose=True)

            manager2 = ETFCrossSectionUnifiedManager(config, progress_monitor)
            logger.info("    ✅ 配置化创建成功")

            # 验证接口实现
            assert isinstance(manager1, IFactorCalculator), "管理器应该实现IFactorCalculator接口"
            assert isinstance(manager1, ICrossSectionManager), "管理器应该实现ICrossSectionManager接口"
            logger.info("    ✅ 接口实现验证通过")

            # 验证系统统计
            stats = manager1.get_system_statistics()
            logger.info(f"    ✅ 系统统计: {stats['available_factors']['total_count']}个可用因子")

            return True

        except Exception as e:
            logger.error(f"  统一管理器测试失败: {str(e)}")
            return False

    def test_dynamic_factor_registration(self) -> bool:
        """测试动态因子注册功能"""
        try:
            logger.info("  初始化统一管理器...")
            from factor_system.factor_engine.factors.etf_cross_section import (
                create_etf_cross_section_manager,
                ETFCrossSectionConfig
            )

            config = ETFCrossSectionConfig()
            config.enable_legacy_factors = False  # 只测试动态因子
            config.enable_dynamic_factors = True
            config.max_dynamic_factors = 500  # 限制数量提高测试速度

            manager = create_etf_cross_section_manager(config)

            logger.info("  注册动态因子...")
            registered_count = manager._register_all_dynamic_factors()
            logger.info(f"    注册了 {registered_count} 个动态因子")

            # 验证因子列表
            available_factors = manager.get_available_factors()
            dynamic_factors = manager.factor_registry.list_factors(is_dynamic=True)

            logger.info(f"    可用因子总数: {len(available_factors)}")
            logger.info(f"    动态因子数量: {len(dynamic_factors)}")

            # 验证因子分类
            categories = manager.get_factor_categories()
            logger.info(f"    因子分类数: {len(categories)}")
            for category, factors in categories.items():
                if factors:
                    logger.info(f"      {category}: {len(factors)}个因子")

            # 验证注册表统计
            registry_stats = manager.factor_registry.get_statistics()
            logger.info(f"    注册表统计: 总计{registry_stats['total_factors']}个因子")

            # 基本断言
            assert len(available_factors) > 0, "应该有可用因子"
            assert len(dynamic_factors) > 0, "应该有动态因子"
            assert registered_count > 0, "应该成功注册因子"

            logger.info("    ✅ 动态因子注册测试通过")
            return True

        except Exception as e:
            logger.error(f"  动态因子注册测试失败: {str(e)}")
            return False

    def test_factor_calculation(self) -> bool:
        """测试因子计算功能"""
        try:
            logger.info("  初始化管理器...")
            from factor_system.factor_engine.factors.etf_cross_section import (
                create_etf_cross_section_manager,
                ETFCrossSectionConfig
            )
            from factor_system.factor_engine.factors.etf_cross_section.interfaces import FactorCalculationResult

            config = ETFCrossSectionConfig()
            config.enable_dynamic_factors = True
            config.enable_legacy_factors = False  # 暂时关闭传统因子避免数据问题
            config.max_dynamic_factors = 50  # 大幅限制因子数量

            manager = create_etf_cross_section_manager(config)

            logger.info("  准备测试数据...")
            # 使用ETF代码进行测试
            test_symbols = ['510300.SH', '159919.SZ', '510500.SH']  # 沪深300、沪深500ETF
            timeframe = 'daily'
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # 30天数据

            logger.info(f"  测试股票: {test_symbols}")
            logger.info(f"  时间范围: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")

            # 获取可用因子
            available_factors = manager.get_available_factors()
            logger.info(f"  可用因子数量: {len(available_factors)}")

            # 选择少量因子进行测试
            test_factors = available_factors[:10] if len(available_factors) > 10 else available_factors
            logger.info(f"  测试因子: {test_factors}")

            if not test_factors:
                logger.warning("  没有可用的测试因子")
                return False

            logger.info("  开始因子计算...")
            start_time = time.time()

            result = manager.calculate_factors(
                symbols=test_symbols,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                factor_ids=test_factors
            )

            calculation_time = time.time() - start_time

            # 验证计算结果
            assert isinstance(result, FactorCalculationResult), "应该返回FactorCalculationResult"

            logger.info(f"    计算耗时: {calculation_time:.2f}s")
            logger.info(f"    内存使用: {result.memory_usage_mb:.1f}MB")
            logger.info(f"    成功因子: {len(result.successful_factors)}")
            logger.info(f"    失败因子: {len(result.failed_factors)}")
            logger.info(f"    成功率: {result.success_rate:.1%}")

            if result.factors_df is not None:
                logger.info(f"    数据形状: {result.factors_df.shape}")

            # 基本断言
            assert calculation_time < 60, "计算时间应该少于60秒"
            assert result.memory_usage_mb < 500, "内存使用应该少于500MB"
            assert len(result.successful_factors) > 0, "应该有成功的因子"

            logger.info("    ✅ 因子计算测试通过")
            return True

        except Exception as e:
            logger.error(f"  因子计算测试失败: {str(e)}")
            logger.error(f"  详细错误: {traceback.format_exc()}")
            return False

    def test_cross_section_building(self) -> bool:
        """测试横截面构建功能"""
        try:
            logger.info("  初始化管理器...")
            from factor_system.factor_engine.factors.etf_cross_section import (
                create_etf_cross_section_manager,
                ETFCrossSectionConfig
            )
            from factor_system.factor_engine.factors.etf_cross_section.interfaces import CrossSectionResult

            config = ETFCrossSectionConfig()
            config.enable_dynamic_factors = True
            config.enable_legacy_factors = False
            config.max_dynamic_factors = 20  # 限制因子数量

            manager = create_etf_cross_section_manager(config)

            logger.info("  准备横截面数据...")
            test_symbols = ['510300.SH', '159919.SZ', '510500.SH']
            test_date = datetime.now() - timedelta(days=1)  # 昨天

            # 获取可用因子
            available_factors = manager.get_available_factors()
            test_factors = available_factors[:5] if len(available_factors) > 5 else available_factors

            if not test_factors:
                logger.warning("  没有可用的测试因子")
                return False

            logger.info(f"  测试日期: {test_date.strftime('%Y-%m-%d')}")
            logger.info(f"  测试股票: {test_symbols}")
            logger.info(f"  测试因子: {test_factors}")

            logger.info("  构建横截面...")
            start_time = time.time()

            result = manager.build_cross_section(
                date=test_date,
                symbols=test_symbols,
                factor_ids=test_factors
            )

            build_time = time.time() - start_time

            # 验证横截面结果
            assert isinstance(result, CrossSectionResult), "应该返回CrossSectionResult"

            logger.info(f"    构建耗时: {build_time:.2f}s")
            logger.info(f"    股票数量: {result.num_stocks}")
            logger.info(f"    因子数量: {result.num_factors}")

            if result.cross_section_df is not None:
                logger.info(f"    数据形状: {result.cross_section_df.shape}")
                logger.info(f"    数据列: {list(result.cross_section_df.columns)}")

            # 验证摘要统计
            if result.summary_stats:
                logger.info(f"    摘要统计因子数: {len(result.summary_stats)}")

            # 基本断言
            assert build_time < 30, "构建时间应该少于30秒"
            assert result.num_stocks > 0, "应该有股票数据"
            assert result.num_factors > 0, "应该有因子数据"

            logger.info("    ✅ 横截面构建测试通过")
            return True

        except Exception as e:
            logger.error(f"  横截面构建测试失败: {str(e)}")
            logger.error(f"  详细错误: {traceback.format_exc()}")
            return False

    def test_performance_requirements(self) -> bool:
        """测试性能要求"""
        try:
            logger.info("  性能基准测试...")
            from factor_system.factor_engine.factors.etf_cross_section import (
                create_etf_cross_section_manager,
                ETFCrossSectionConfig
            )

            config = ETFCrossSectionConfig()
            config.enable_dynamic_factors = True
            config.enable_legacy_factors = False
            config.max_dynamic_factors = 100  # 100个因子进行性能测试

            manager = create_etf_cross_section_manager(config)

            # 测试1: 因子注册性能
            logger.info("  测试因子注册性能...")
            start_time = time.time()
            registered_count = manager._register_all_dynamic_factors()
            registration_time = time.time() - start_time

            logger.info(f"    注册{registered_count}个因子耗时: {registration_time:.2f}s")
            assert registration_time < 30, f"因子注册应该少于30秒，实际: {registration_time:.2f}s"

            # 测试2: 内存使用监控
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # 测试3: 系统统计性能
            logger.info("  测试系统统计性能...")
            start_time = time.time()
            stats = manager.get_system_statistics()
            stats_time = time.time() - start_time

            logger.info(f"    系统统计耗时: {stats_time:.3f}s")
            assert stats_time < 1, f"系统统计应该少于1秒，实际: {stats_time:.3f}s"

            # 验证统计信息结构
            assert 'available_factors' in stats, "应该包含可用因子统计"
            assert 'dynamic_registry' in stats, "应该包含动态注册表统计"
            assert 'dynamic_factory' in stats, "应该包含动态工厂统计"

            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            logger.info(f"    内存增量: {memory_increase:.1f}MB")
            assert memory_increase < 200, f"内存增量应该少于200MB，实际: {memory_increase:.1f}MB"

            # 测试4: 缓存清理性能
            logger.info("  测试缓存清理性能...")
            start_time = time.time()
            manager.clear_cache()
            cache_time = time.time() - start_time

            logger.info(f"    缓存清理耗时: {cache_time:.3f}s")
            assert cache_time < 5, f"缓存清理应该少于5秒，实际: {cache_time:.3f}s"

            logger.info("    ✅ 性能要求测试通过")
            return True

        except Exception as e:
            logger.error(f"  性能要求测试失败: {str(e)}")
            return False

    def test_end_to_end_workflow(self) -> bool:
        """测试端到端工作流程"""
        try:
            logger.info("  完整工作流程测试...")
            from factor_system.factor_engine.factors.etf_cross_section import (
                create_etf_cross_section_manager,
                ETFCrossSectionConfig
            )

            # 步骤1: 创建管理器
            logger.info("  步骤1: 创建管理器")
            config = ETFCrossSectionConfig()
            config.enable_dynamic_factors = True
            config.enable_legacy_factors = True  # 启用所有功能
            config.max_dynamic_factors = 30
            config.verbose = True

            manager = create_etf_cross_section_manager(config)

            # 步骤2: 注册所有因子
            logger.info("  步骤2: 注册因子")
            registered_count = manager._register_all_dynamic_factors()
            logger.info(f"    注册了{registered_count}个动态因子")

            # 步骤3: 获取因子信息
            logger.info("  步骤3: 分析因子信息")
            all_factors = manager.get_available_factors()
            categories = manager.get_factor_categories()
            system_stats = manager.get_system_statistics()

            logger.info(f"    总因子数: {len(all_factors)}")
            logger.info(f"    因子分类数: {len(categories)}")

            # 步骤4: 模拟实际使用场景
            logger.info("  步骤4: 模拟实际使用")
            test_symbols = ['510300.SH', '159919.SZ']
            end_date = datetime.now() - timedelta(days=1)
            start_date = end_date - timedelta(days=10)

            # 选择一些因子进行计算
            test_factors = all_factors[:10] if len(all_factors) >= 10 else all_factors

            if test_factors:
                # 计算因子
                factor_result = manager.calculate_factors(
                    symbols=test_symbols,
                    timeframe='daily',
                    start_date=start_date,
                    end_date=end_date,
                    factor_ids=test_factors
                )

                # 构建横截面
                if factor_result.successful_factors:
                    cross_section_result = manager.build_cross_section(
                        date=end_date,
                        symbols=test_symbols,
                        factor_ids=factor_result.successful_factors[:5]
                    )

                    logger.info(f"    ✅ 完整流程执行成功")
                    logger.info(f"      因子计算: {len(factor_result.successful_factors)}成功")
                    logger.info(f"      横截面构建: {cross_section_result.num_stocks}股票 x {cross_section_result.num_factors}因子")
                else:
                    logger.warning("    ⚠️ 因子计算失败，跳过横截面构建")
                    return False
            else:
                logger.warning("    ⚠️ 没有可用因子")
                return False

            # 步骤5: 验证数据质量
            logger.info("  步骤5: 验证数据质量")
            assert len(manager.get_available_factors()) > 0, "应该有可用因子"
            assert len(manager.get_factor_categories()) > 0, "应该有因子分类"
            assert system_stats['available_factors']['total_count'] > 0, "统计应该显示有因子"

            logger.info("    ✅ 端到端工作流程测试通过")
            return True

        except Exception as e:
            logger.error(f"  端到端工作流程测试失败: {str(e)}")
            logger.error(f"  详细错误: {traceback.format_exc()}")
            return False

    def generate_test_report(self) -> Dict[str, Any]:
        """生成测试报告"""
        total_time = time.time() - self.start_time
        success_rate = (self.passed_tests / self.total_tests) * 100

        report = {
            'summary': {
                'total_tests': self.total_tests,
                'passed_tests': self.passed_tests,
                'failed_tests': self.total_tests - self.passed_tests,
                'success_rate': success_rate,
                'total_time': total_time,
                'timestamp': datetime.now().isoformat()
            },
            'test_results': self.test_results,
            'recommendations': []
        }

        # 生成建议
        if success_rate == 100:
            report['recommendations'].append("🎉 所有测试通过！系统运行良好。")
        elif success_rate >= 80:
            report['recommendations'].append("✅ 大部分测试通过，系统基本可用。")
        else:
            report['recommendations'].append("⚠️ 多个测试失败，需要修复问题。")

        # 输出报告
        logger.info("\n" + "=" * 60)
        logger.info("测试报告")
        logger.info("=" * 60)
        logger.info(f"总测试数: {report['summary']['total_tests']}")
        logger.info(f"通过测试: {report['summary']['passed_tests']}")
        logger.info(f"失败测试: {report['summary']['failed_tests']}")
        logger.info(f"成功率: {report['summary']['success_rate']:.1f}%")
        logger.info(f"总耗时: {report['summary']['total_time']:.2f}s")

        for test_name, result in report['test_results'].items():
            status_icon = "✅" if result['status'] == 'PASSED' else "❌"
            logger.info(f"{status_icon} {test_name}: {result['status']}")
            if result['error']:
                logger.info(f"    错误: {result['error']}")

        for recommendation in report['recommendations']:
            logger.info(recommendation)

        # 保存报告到文件
        report_file = '/Users/zhangshenshen/深度量化0927/scripts/comprehensive_smoke_test_report.json'
        import json
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"\n详细报告已保存到: {report_file}")

        return report


def main():
    """主函数"""
    logger.info("开始执行ETF横截面因子系统全面冒烟测试...")

    # 创建测试实例
    smoke_test = ComprehensiveSmokeTest()

    try:
        # 运行所有测试
        report = smoke_test.run_all_tests()

        # 返回适当的退出码
        if report['summary']['success_rate'] == 100:
            logger.info("\n🎉 所有测试通过！")
            return 0
        else:
            logger.error(f"\n❌ 测试失败，成功率: {report['summary']['success_rate']:.1f}%")
            return 1

    except Exception as e:
        logger.error(f"测试执行异常: {str(e)}")
        logger.error(f"详细错误: {traceback.format_exc()}")
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)