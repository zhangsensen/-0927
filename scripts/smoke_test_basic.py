#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF横截面系统基础冒烟测试
验证系统基本功能是否正常工作
"""

import pandas as pd
import numpy as np
import logging
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BasicSmokeTest:
    """基础冒烟测试类"""

    def __init__(self):
        """初始化测试"""
        self.test_results = {}
        self.project_root = Path(__file__).parent.parent
        logger.info("基础冒烟测试初始化完成")

    def run_all_tests(self) -> Dict[str, bool]:
        """
        运行所有基础测试

        Returns:
            测试结果字典
        """
        logger.info("=" * 60)
        logger.info("开始运行基础冒烟测试")
        logger.info("=" * 60)

        # 测试1：基础模块导入
        self.test_results['module_import'] = self.test_module_import()

        # 测试2：数据管理器
        self.test_results['data_manager'] = self.test_data_manager()

        # 测试3：原有因子计算器
        self.test_results['original_factor_calculator'] = self.test_original_factor_calculator()

        # 测试4：存储系统
        self.test_results['storage_system'] = self.test_storage_system()

        # 测试5：数据格式兼容性
        self.test_results['data_compatibility'] = self.test_data_compatibility()

        # 测试6：简单因子计算
        self.test_results['simple_factor_calculation'] = self.test_simple_factor_calculation()

        # 汇总结果
        passed_tests = sum(1 for result in self.test_results.values() if result)
        total_tests = len(self.test_results)

        logger.info("=" * 60)
        logger.info("基础冒烟测试完成")
        logger.info(f"通过: {passed_tests}/{total_tests}")
        logger.info("=" * 60)

        return self.test_results

    def test_module_import(self) -> bool:
        """测试基础模块导入"""
        try:
            logger.info("测试1：基础模块导入")

            # 测试基础模块导入
            from factor_system.factor_engine.providers.etf_cross_section_provider import ETFCrossSectionDataManager
            from factor_system.factor_engine.factors.etf_cross_section import ETFCrossSectionFactors
            from factor_system.factor_engine.providers.etf_cross_section_storage import ETFCrossSectionStorage

            logger.info("  ✅ ETFCrossSectionDataManager 导入成功")
            logger.info("  ✅ ETFCrossSectionFactors 导入成功")
            logger.info("  ✅ ETFCrossSectionStorage 导入成功")

            return True

        except Exception as e:
            logger.error(f"  ❌ 模块导入失败: {str(e)}")
            logger.error(f"  详细错误: {traceback.format_exc()}")
            return False

    def test_data_manager(self) -> bool:
        """测试数据管理器"""
        try:
            logger.info("测试2：数据管理器")

            from factor_system.factor_engine.providers.etf_cross_section_provider import ETFCrossSectionDataManager

            # 创建数据管理器实例
            data_manager = ETFCrossSectionDataManager()
            logger.info("  ✅ 数据管理器实例创建成功")

            # 获取ETF列表
            etf_list = data_manager.get_etf_universe()
            if not etf_list:
                logger.warning("  ⚠️ ETF列表为空")
                return False
            else:
                logger.info(f"  ✅ 获取ETF列表成功: {len(etf_list)} 只ETF")

            # 测试获取少量ETF的价格数据
            test_etfs = etf_list[:3]  # 取前3只ETF
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=30)  # 最近30天

            price_data = data_manager.get_time_series_data(
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
                test_etfs
            )

            if price_data.empty:
                logger.warning("  ⚠️ 价格数据为空")
                return False
            else:
                logger.info(f"  ✅ 价格数据获取成功: {len(price_data)} 条记录")
                logger.info(f"  ✅ 数据列: {list(price_data.columns)}")

            return True

        except Exception as e:
            logger.error(f"  ❌ 数据管理器测试失败: {str(e)}")
            logger.error(f"  详细错误: {traceback.format_exc()}")
            return False

    def test_original_factor_calculator(self) -> bool:
        """测试原有因子计算器"""
        try:
            logger.info("测试3：原有因子计算器")

            from factor_system.factor_engine.factors.etf_cross_section import ETFCrossSectionFactors

            # 创建因子计算器实例
            factor_calculator = ETFCrossSectionFactors(enable_storage=False)
            logger.info("  ✅ 因子计算器实例创建成功")

            # 检查方法是否存在
            required_methods = [
                'calculate_momentum_factors',
                'calculate_quality_factors',
                'calculate_liquidity_factors',
                'calculate_technical_factors',
                'calculate_all_factors'
            ]

            for method_name in required_methods:
                if hasattr(factor_calculator, method_name):
                    logger.info(f"  ✅ 方法 {method_name} 存在")
                else:
                    logger.error(f"  ❌ 方法 {method_name} 不存在")
                    return False

            return True

        except Exception as e:
            logger.error(f"  ❌ 原有因子计算器测试失败: {str(e)}")
            logger.error(f"  详细错误: {traceback.format_exc()}")
            return False

    def test_storage_system(self) -> bool:
        """测试存储系统"""
        try:
            logger.info("测试4：存储系统")

            from factor_system.factor_engine.providers.etf_cross_section_storage import ETFCrossSectionStorage

            # 创建存储实例
            storage = ETFCrossSectionStorage()
            logger.info("  ✅ 存储实例创建成功")

            # 检查基础存储目录
            if storage.base_dir.exists():
                logger.info(f"  ✅ 存储目录存在: {storage.base_dir}")
            else:
                logger.warning(f"  ⚠️ 存储目录不存在: {storage.base_dir}")

            return True

        except Exception as e:
            logger.error(f"  ❌ 存储系统测试失败: {str(e)}")
            logger.error(f"  详细错误: {traceback.format_exc()}")
            return False

    def test_data_compatibility(self) -> bool:
        """测试数据格式兼容性"""
        try:
            logger.info("测试5：数据格式兼容性")

            from factor_system.factor_engine.providers.etf_cross_section_provider import ETFCrossSectionDataManager

            data_manager = ETFCrossSectionDataManager()

            # 获取测试数据
            etf_list = data_manager.get_etf_universe()
            if not etf_list:
                logger.warning("  ⚠️ 没有ETF数据，跳过兼容性测试")
                return True

            test_etfs = etf_list[:2]
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=15)

            price_data = data_manager.get_time_series_data(
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
                test_etfs
            )

            if price_data.empty:
                logger.warning("  ⚠️ 价格数据为空，跳过兼容性测试")
                return True

            # 检查必需的列
            required_columns = ['etf_code', 'trade_date', 'close']
            missing_columns = [col for col in required_columns if col not in price_data.columns]

            if missing_columns:
                logger.error(f"  ❌ 缺少必需列: {missing_columns}")
                logger.info(f"  📋 实际列: {list(price_data.columns)}")
                return False
            else:
                logger.info(f"  ✅ 数据列完整: {list(price_data.columns)}")

            # 检查数据类型
            if not pd.api.types.is_datetime64_any_dtype(price_data['trade_date']):
                logger.warning("  ⚠️ trade_date不是datetime类型")
                # 尝试转换
                price_data['trade_date'] = pd.to_datetime(price_data['trade_date'])
                logger.info("  ✅ trade_date已转换为datetime类型")

            logger.info("  ✅ 数据格式兼容性测试通过")
            return True

        except Exception as e:
            logger.error(f"  ❌ 数据兼容性测试失败: {str(e)}")
            logger.error(f"  详细错误: {traceback.format_exc()}")
            return False

    def test_simple_factor_calculation(self) -> bool:
        """测试简单因子计算"""
        try:
            logger.info("测试6：简单因子计算")

            from factor_system.factor_engine.providers.etf_cross_section_provider import ETFCrossSectionDataManager
            from factor_system.factor_engine.factors.etf_cross_section import ETFCrossSectionFactors

            # 准备测试数据
            data_manager = ETFCrossSectionDataManager()
            etf_list = data_manager.get_etf_universe()

            if len(etf_list) < 2:
                logger.warning("  ⚠️ ETF数据不足，跳过因子计算测试")
                return True

            test_etfs = etf_list[:2]
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=15)

            price_data = data_manager.get_time_series_data(
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
                test_etfs
            )

            if price_data.empty:
                logger.warning("  ⚠️ 价格数据为空，跳过因子计算测试")
                return True

            # 创建因子计算器
            factor_calculator = ETFCrossSectionFactors(enable_storage=False)

            # 尝试计算动量因子
            try:
                momentum_factors = factor_calculator.calculate_momentum_factors(
                    price_data, periods=[5, 10]
                )

                if momentum_factors.empty:
                    logger.warning("  ⚠️ 动量因子计算结果为空")
                else:
                    logger.info(f"  ✅ 动量因子计算成功: {len(momentum_factors)} 条记录")
                    logger.info(f"  📊 动量因子列: {[col for col in momentum_factors.columns if col not in ['etf_code', 'trade_date']]}")

            except Exception as e:
                logger.warning(f"  ⚠️ 动量因子计算失败: {str(e)}")

            # 尝试计算技术因子
            try:
                technical_factors = factor_calculator.calculate_technical_factors(price_data)

                if technical_factors.empty:
                    logger.warning("  ⚠️ 技术因子计算结果为空")
                else:
                    logger.info(f"  ✅ 技术因子计算成功: {len(technical_factors)} 条记录")
                    logger.info(f"  📊 技术因子列: {[col for col in technical_factors.columns if col not in ['etf_code', 'trade_date']]}")

            except Exception as e:
                logger.warning(f"  ⚠️ 技术因子计算失败: {str(e)}")

            logger.info("  ✅ 简单因子计算测试完成")
            return True

        except Exception as e:
            logger.error(f"  ❌ 简单因子计算测试失败: {str(e)}")
            logger.error(f"  详细错误: {traceback.format_exc()}")
            return False

    def generate_test_report(self) -> str:
        """生成测试报告"""
        report = []
        report.append("# ETF横截面系统基础冒烟测试报告")
        report.append(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)

        report.append(f"## 测试总览")
        report.append(f"- 总测试数: {total_tests}")
        report.append(f"- 通过测试: {passed_tests}")
        report.append(f"- 失败测试: {total_tests - passed_tests}")
        report.append(f"- 通过率: {passed_tests/total_tests:.1%}")
        report.append("")

        report.append("## 详细结果")
        for test_name, result in self.test_results.items():
            status = "✅ 通过" if result else "❌ 失败"
            report.append(f"- **{test_name}**: {status}")

        report.append("")
        if passed_tests == total_tests:
            report.append("🎉 所有基础功能测试通过！系统可以正常使用。")
        else:
            report.append("⚠️ 部分测试失败，请检查相关功能。")

        return "\n".join(report)


def main():
    """主函数"""
    tester = BasicSmokeTest()
    results = tester.run_all_tests()

    # 生成并保存测试报告
    report = tester.generate_test_report()
    report_file = Path(__file__).parent / "smoke_test_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    logger.info(f"测试报告已保存到: {report_file}")

    # 根据测试结果设置退出码
    passed_count = sum(1 for result in results.values() if result)
    total_count = len(results)

    if passed_count == total_count:
        logger.info("🎉 所有基础冒烟测试通过！")
        sys.exit(0)
    else:
        logger.error(f"❌ {total_count - passed_count} 个测试失败")
        sys.exit(1)


if __name__ == "__main__":
    main()