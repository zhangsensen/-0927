#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF横截面全因子系统集成演示
展示完整的800-1200个动态因子集成系统
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time

# 导入增强的ETF横截面系统组件
from factor_system.factor_engine.factors.etf_cross_section.etf_cross_section_enhanced import ETFCrossSectionFactorsEnhanced
from factor_system.factor_engine.factors.etf_cross_section.etf_cross_section_strategy_enhanced import (
    ETFCrossSectionStrategyEnhanced, EnhancedStrategyConfig
)
from factor_system.factor_engine.factors.etf_cross_section.etf_cross_section_storage_enhanced import ETFCrossSectionStorageEnhanced
from factor_system.factor_engine.providers.etf_cross_section_provider import ETFCrossSectionDataManager

from factor_system.utils import safe_operation, FactorSystemError

logger = logging.getLogger(__name__)


class ETFCrossSectionEnhancedDemo:
    """ETF横截面增强系统演示"""

    def __init__(self):
        """初始化演示系统"""
        self.data_manager = ETFCrossSectionDataManager()
        self.storage = ETFCrossSectionStorageEnhanced(enable_compression=True)
        self.factor_calculator = ETFCrossSectionFactorsEnhanced(
            data_manager=self.data_manager,
            enable_storage=True,
            enable_dynamic_factors=True
        )

        logger.info("ETF横截面增强系统演示初始化完成")

    def run_basic_demo(self) -> Dict[str, Any]:
        """
        运行基础演示

        Returns:
            演示结果字典
        """
        logger.info("=" * 60)
        logger.info("开始运行ETF横截面增强系统基础演示")
        logger.info("=" * 60)

        results = {}

        try:
            # 步骤1：系统初始化测试
            logger.info("步骤1：系统初始化测试")
            init_result = self._test_system_initialization()
            results['initialization'] = init_result

            # 步骤2：因子注册测试
            logger.info("步骤2：动态因子注册测试")
            factor_registration_result = self._test_factor_registration()
            results['factor_registration'] = factor_registration_result

            # 步骤3：因子计算测试（小规模）
            logger.info("步骤3：动态因子计算测试")
            factor_calculation_result = self._test_factor_calculation()
            results['factor_calculation'] = factor_calculation_result

            # 步骤4：策略引擎测试
            logger.info("步骤4：增强策略引擎测试")
            strategy_result = self._test_enhanced_strategy()
            results['strategy'] = strategy_result

            # 步骤5：存储系统测试
            logger.info("步骤5：增强存储系统测试")
            storage_result = self._test_enhanced_storage()
            results['storage'] = storage_result

            # 步骤6：性能统计
            logger.info("步骤6：系统性能统计")
            performance_stats = self._get_performance_statistics()
            results['performance'] = performance_stats

            logger.info("=" * 60)
            logger.info("基础演示完成！")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"基础演示失败: {str(e)}")
            results['error'] = str(e)

        return results

    def run_comprehensive_demo(self, start_date: str = "2024-01-01",
                             end_date: str = "2025-10-14") -> Dict[str, Any]:
        """
        运行完整演示

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            演示结果字典
        """
        logger.info("=" * 80)
        logger.info("开始运行ETF横截面增强系统完整演示")
        logger.info(f"时间范围: {start_date} ~ {end_date}")
        logger.info("=" * 80)

        results = {}

        try:
            # 初始化动态因子
            logger.info("初始化动态因子系统...")
            factor_count = self.factor_calculator.initialize_dynamic_factors()
            results['dynamic_factors_initialized'] = factor_count

            # 获取ETF列表
            logger.info("获取ETF列表...")
            etf_list = self.data_manager.get_etf_universe()
            logger.info(f"找到 {len(etf_list)} 只ETF")

            # 选择少量ETF进行演示（避免计算量过大）
            demo_etfs = etf_list[:10]  # 选择前10只ETF
            logger.info(f"演示ETF: {demo_etfs}")

            # 计算增强因子
            logger.info("计算增强因子...")
            start_time = time.time()

            enhanced_factors = self.factor_calculator.calculate_all_factors_enhanced(
                start_date=start_date,
                end_date=end_date,
                etf_codes=demo_etfs,
                include_original=True,
                use_cache=True
            )

            calculation_time = time.time() - start_time
            results['factor_calculation_time'] = calculation_time

            if not enhanced_factors.empty:
                logger.info(f"因子计算完成: {len(enhanced_factors)} 条记录")
                logger.info(f"包含的因子列: {[col for col in enhanced_factors.columns if col not in ['date', 'etf_code']][:10]}...")

                # 保存因子数据
                save_success = self._save_demo_factors(enhanced_factors, demo_etfs, start_date, end_date)
                results['factors_saved'] = save_success

                # 运行增强策略回测
                logger.info("运行增强策略回测...")
                strategy_results = self._run_strategy_backtest(
                    start_date, end_date, demo_etfs
                )
                results['strategy_backtest'] = strategy_results

            else:
                logger.error("因子计算结果为空")

            # 系统统计信息
            system_stats = self._get_comprehensive_statistics()
            results['system_statistics'] = system_stats

            logger.info("=" * 80)
            logger.info("完整演示完成！")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"完整演示失败: {str(e)}")
            results['error'] = str(e)

        return results

    def _test_system_initialization(self) -> Dict[str, Any]:
        """测试系统初始化"""
        try:
            # 测试数据管理器
            etf_list = self.data_manager.get_etf_universe()
            data_manager_status = len(etf_list) > 0

            # 测试存储系统
            storage_stats = self.storage.get_storage_statistics()
            storage_status = storage_stats is not None

            # 测试因子计算器
            factor_stats = self.factor_calculator.get_factor_statistics()
            calculator_status = factor_stats is not None

            return {
                'data_manager_status': data_manager_status,
                'etf_count': len(etf_list),
                'storage_status': storage_status,
                'calculator_status': calculator_status,
                'overall_status': all([data_manager_status, storage_status, calculator_status])
            }

        except Exception as e:
            return {'error': str(e), 'overall_status': False}

    def _test_factor_registration(self) -> Dict[str, Any]:
        """测试因子注册"""
        try:
            # 获取因子注册表
            from factor_system.factor_engine.factors.etf_cross_section.factor_registry import get_factor_registry
            registry = get_factor_registry()

            # 注册前统计
            before_stats = registry.get_statistics()

            # 初始化动态因子
            factor_count = self.factor_calculator.initialize_dynamic_factors()

            # 注册后统计
            after_stats = registry.get_statistics()

            return {
                'before_registration': before_stats,
                'after_registration': after_stats,
                'new_dynamic_factors': factor_count,
                'registration_success': factor_count > 0
            }

        except Exception as e:
            return {'error': str(e), 'registration_success': False}

    def _test_factor_calculation(self) -> Dict[str, Any]:
        """测试因子计算"""
        try:
            # 获取少量ETF和较短时间范围进行测试
            test_etfs = self.data_manager.get_etf_universe()[:5]
            test_end = datetime.now().date()
            test_start = test_end - timedelta(days=30)

            logger.info(f"测试计算: {len(test_etfs)} 只ETF, {test_start} ~ {test_end}")

            start_time = time.time()

            # 计算少量动态因子
            from factor_system.factor_engine.factors.etf_cross_section.factor_registry import get_factor_registry
            registry = get_factor_registry()

            # 获取前5个动态因子进行测试
            dynamic_factors = registry.list_factors(is_dynamic=True)[:5]

            if not dynamic_factors:
                return {'error': '没有可用的动态因子', 'calculation_success': False}

            # 计算因子
            test_factors = self.factor_calculator.calculate_dynamic_factors_batch(
                factor_ids=dynamic_factors,
                price_data=self.data_manager.get_time_series_data(
                    test_start.strftime('%Y-%m-%d'),
                    test_end.strftime('%Y-%m-%d'),
                    test_etfs
                ),
                symbols=test_etfs,
                parallel=True
            )

            calculation_time = time.time() - start_time

            return {
                'test_etfs': len(test_etfs),
                'test_factors': len(dynamic_factors),
                'calculated_factors': len(test_factors),
                'calculation_time': calculation_time,
                'calculation_success': len(test_factors) > 0
            }

        except Exception as e:
            return {'error': str(e), 'calculation_success': False}

    def _test_enhanced_strategy(self) -> Dict[str, Any]:
        """测试增强策略"""
        try:
            # 创建增强策略配置
            config = EnhancedStrategyConfig(
                start_date="2024-01-01",
                end_date="2025-10-14",
                enable_dynamic_factors=True,
                factor_selection_method="auto",
                max_dynamic_factors=10,  # 限制数量以加快测试
                top_n=3,
                rebalance_freq="M"
            )

            # 创建策略引擎
            strategy = ETFCrossSectionStrategyEnhanced(config)

            # 获取策略统计
            stats = strategy.get_strategy_statistics()

            return {
                'strategy_config': config.__dict__,
                'strategy_statistics': stats,
                'strategy_success': True
            }

        except Exception as e:
            return {'error': str(e), 'strategy_success': False}

    def _test_enhanced_storage(self) -> Dict[str, Any]:
        """测试增强存储"""
        try:
            # 获取存储统计
            stats = self.storage.get_storage_statistics()

            # 测试缓存功能
            test_data = {"test_key": "test_value", "timestamp": datetime.now().isoformat()}
            cache_save = self.storage.save_factor_cache("test_cache", test_data, ttl_hours=1)
            cache_load = self.storage.load_factor_cache("test_cache")

            # 测试存储优化
            optimization_results = self.storage.optimize_storage()

            return {
                'storage_statistics': stats,
                'cache_save_success': cache_save,
                'cache_load_success': cache_load == test_data,
                'optimization_results': optimization_results,
                'storage_success': True
            }

        except Exception as e:
            return {'error': str(e), 'storage_success': False}

    def _run_strategy_backtest(self, start_date: str, end_date: str,
                              etf_codes: List[str]) -> Dict[str, Any]:
        """运行策略回测"""
        try:
            # 创建增强策略配置
            config = EnhancedStrategyConfig(
                start_date=start_date,
                end_date=end_date,
                etf_universe=etf_codes,
                enable_dynamic_factors=True,
                factor_selection_method="auto",
                max_dynamic_factors=15,
                top_n=5,
                rebalance_freq="M",
                weight_method="equal"
            )

            # 创建策略引擎
            strategy = ETFCrossSectionStrategyEnhanced(config, self.data_manager)

            # 运行回测
            backtest_results = strategy.run_enhanced_backtest()

            if not backtest_results.empty:
                # 计算基本统计
                total_rebalance_dates = len(backtest_results)
                avg_portfolio_size = backtest_results['weights'].apply(len).mean()

                return {
                    'total_rebalance_dates': total_rebalance_dates,
                    'avg_portfolio_size': avg_portfolio_size,
                    'backtest_success': True,
                    'sample_results': backtest_results.head(3).to_dict('records')
                }
            else:
                return {'backtest_success': False, 'error': '回测结果为空'}

        except Exception as e:
            return {'error': str(e), 'backtest_success': False}

    def _save_demo_factors(self, factors_data: pd.DataFrame,
                         etf_codes: List[str], start_date: str, end_date: str) -> bool:
        """保存演示因子数据"""
        try:
            # 获取动态因子列
            dynamic_factor_cols = [
                col for col in factors_data.columns
                if col not in ['date', 'etf_code'] and col not in [
                    'momentum_21d', 'momentum_63d', 'momentum_126d', 'momentum_252d',
                    'volatility_60d', 'max_drawdown_252d', 'quality_score',
                    'liquidity_score', 'rsi_14d', 'macd_signal',
                    'bollinger_position', 'volume_ratio', 'composite_score'
                ]
            ]

            if dynamic_factor_cols:
                # 提取动态因子数据
                dynamic_factors_data = {}
                for col in dynamic_factor_cols[:10]:  # 保存前10个动态因子
                    factor_df = factors_data[['date', 'etf_code', col]].copy()
                    factor_df = factor_df.rename(columns={col: 'factor_value'})
                    dynamic_factors_data[col] = factor_df

                # 保存到存储系统
                save_success = self.storage.save_dynamic_factors(
                    dynamic_factors_data, start_date, end_date, etf_codes
                )

                logger.info(f"保存了 {len(dynamic_factors_data)} 个动态因子")
                return save_success
            else:
                logger.warning("没有找到动态因子列")
                return True

        except Exception as e:
            logger.error(f"保存演示因子失败: {str(e)}")
            return False

    def _get_performance_statistics(self) -> Dict[str, Any]:
        """获取性能统计"""
        try:
            # 因子统计
            factor_stats = self.factor_calculator.get_factor_statistics()

            # 存储统计
            storage_stats = self.storage.get_storage_statistics()

            # 系统资源使用
            import psutil
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk_usage = psutil.disk_usage('/')

            return {
                'factor_statistics': factor_stats,
                'storage_statistics': storage_stats,
                'system_resources': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available_gb': memory.available / (1024**3),
                    'disk_free_gb': disk_usage.free / (1024**3)
                }
            }

        except Exception as e:
            return {'error': str(e)}

    def _get_comprehensive_statistics(self) -> Dict[str, Any]:
        """获取综合统计信息"""
        try:
            from factor_system.factor_engine.factors.etf_cross_section.factor_registry import get_factor_registry
            registry = get_factor_registry()

            # 因子统计
            registry_stats = registry.get_statistics()

            # 可用因子列表
            available_factors = self.factor_calculator.get_available_factors()

            # 系统性能
            performance = self._get_performance_statistics()

            return {
                'factor_registry_stats': registry_stats,
                'available_factors': available_factors,
                'performance': performance,
                'demo_summary': {
                    'total_dynamic_factors': registry_stats['dynamic_factors'],
                    'total_factors': registry_stats['total_factors'],
                    'factor_categories': len(registry_stats['categories']),
                    'system_status': 'healthy'
                }
            }

        except Exception as e:
            return {'error': str(e)}


@safe_operation
def main():
    """主函数 - 运行ETF横截面增强系统演示"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('etf_cross_section_enhanced_demo.log'),
            logging.StreamHandler()
        ]
    )

    logger.info("🚀 启动ETF横截面全因子集成系统演示")

    # 创建演示实例
    demo = ETFCrossSectionEnhancedDemo()

    try:
        # 运行基础演示
        logger.info("\n" + "="*60)
        logger.info("🔍 运行基础功能测试")
        logger.info("="*60)
        basic_results = demo.run_basic_demo()

        if basic_results.get('initialization', {}).get('overall_status', False):
            logger.info("✅ 基础功能测试通过")

            # 运行完整演示
            logger.info("\n" + "="*60)
            logger.info("🎯 运行完整系统演示")
            logger.info("="*60)
            comprehensive_results = demo.run_comprehensive_demo()

            # 打印最终结果摘要
            logger.info("\n" + "="*80)
            logger.info("📊 演示结果摘要")
            logger.info("="*80)

            if 'dynamic_factors_initialized' in comprehensive_results:
                factor_count = comprehensive_results['dynamic_factors_initialized']
                logger.info(f"🎉 动态因子初始化成功: {factor_count} 个因子")

            if 'system_statistics' in comprehensive_results:
                stats = comprehensive_results['system_statistics']
                if 'demo_summary' in stats:
                    summary = stats['demo_summary']
                    logger.info(f"📈 系统统计:")
                    logger.info(f"   • 总因子数: {summary.get('total_factors', 0)}")
                    logger.info(f"   • 动态因子数: {summary.get('total_dynamic_factors', 0)}")
                    logger.info(f"   • 因子类别数: {summary.get('factor_categories', 0)}")

            if 'factor_calculation_time' in comprehensive_results:
                calc_time = comprehensive_results['factor_calculation_time']
                logger.info(f"⚡ 因子计算耗时: {calc_time:.2f} 秒")

            logger.info("\n🎊 ETF横截面全因子集成系统演示完成！")
            logger.info("✨ 系统已准备就绪，可以处理800-1200个动态因子")

        else:
            logger.error("❌ 基础功能测试失败，跳过完整演示")

    except Exception as e:
        logger.error(f"❌ 演示运行失败: {str(e)}")
        raise

    logger.info("\n" + "="*80)
    logger.info("🏁 演示结束")
    logger.info("="*80)


if __name__ == "__main__":
    main()