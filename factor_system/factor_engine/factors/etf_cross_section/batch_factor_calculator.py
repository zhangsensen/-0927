#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF横截面批量因子计算引擎
支持多进程并行计算，高效处理800-1200个候选因子
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import multiprocessing as mp
from datetime import datetime, timedelta
import pickle
import gc
import psutil

from factor_system.factor_engine import api
from factor_system.utils import safe_operation, FactorSystemError
from .candidate_factor_generator import ETFCandidateFactorGenerator, FactorVariant

logger = logging.getLogger(__name__)


@dataclass
class CalculationTask:
    """计算任务定义"""
    task_id: str
    variant: FactorVariant
    symbols: List[str]
    timeframe: str
    start_date: datetime
    end_date: datetime


@dataclass
class CalculationResult:
    """计算结果"""
    task_id: str
    variant_id: str
    factor_data: pd.DataFrame
    success: bool
    error_message: Optional[str] = None
    calculation_time: Optional[float] = None


class BatchFactorCalculator:
    """批量因子计算器"""

    def __init__(self, max_workers: Optional[int] = None, batch_size: int = 50):
        """
        初始化批量计算器

        Args:
            max_workers: 最大工作进程数，默认使用CPU核心数-1
            batch_size: 每批处理的因子数量
        """
        self.max_workers = max_workers or max(1, mp.cpu_count() - 1)
        self.batch_size = batch_size
        self.memory_threshold = 0.8  # 内存使用阈值

        logger.info(f"批量因子计算器初始化完成")
        logger.info(f"最大工作进程数: {self.max_workers}")
        logger.info(f"批量大小: {self.batch_size}")

    def _check_memory_usage(self):
        """检查内存使用情况"""
        memory_percent = psutil.virtual_memory().percent / 100
        if memory_percent > self.memory_threshold:
            logger.warning(f"内存使用率过高: {memory_percent:.1%}, 执行垃圾回收")
            gc.collect()
        return memory_percent

    def calculate_factors(self,
                         symbols: List[str],
                         factor_ids: List[str],
                         start_date: datetime,
                         end_date: datetime,
                         timeframe: str = 'daily',
                         max_workers: Optional[int] = None,
                         factor_registry = None) -> pd.DataFrame:
        """
        计算因子（统一接口）
        
        Args:
            symbols: ETF代码列表
            factor_ids: 因子ID列表
            start_date: 开始日期
            end_date: 结束日期
            timeframe: 时间框架
            max_workers: 最大工作进程数
            factor_registry: 因子注册表（用于动态因子）
            
        Returns:
            统一格式DataFrame: MultiIndex(date, symbol), columns=factor_ids
        """
        logger.info(f"开始计算因子: {len(symbols)}只ETF, {len(factor_ids)}个因子")
        
        try:
            # 🔥 关键修复：优先使用传入的factor_registry
            if factor_registry is not None:
                # 使用ETF专用注册表计算动态因子
                all_data = []
                
                for factor_id in factor_ids:
                    try:
                        # 从注册表获取因子信息
                        factor_info = factor_registry.get_factor(factor_id)
                        if factor_info is None:
                            # 回退到全局API
                            factor_result = api.calculate_factors(
                                factor_ids=[factor_id],
                                symbols=symbols,
                                timeframe=timeframe,
                                start_date=start_date,
                                end_date=end_date
                            )
                        else:
                            # 使用注册表中的因子函数直接计算
                            # 这里需要加载数据并调用因子函数
                            factor_result = self._calculate_factor_from_registry(
                                factor_id, factor_info, symbols, start_date, end_date, timeframe
                            )
                        
                        if factor_result is not None and not factor_result.empty:
                            if factor_id in factor_result.columns:
                                all_data.append(factor_result[[factor_id]])
                            elif len(factor_result.columns) > 0:
                                temp_df = factor_result.iloc[:, [0]].copy()
                                temp_df.columns = [factor_id]
                                all_data.append(temp_df)
                                
                    except Exception as e:
                        logger.warning(f"因子 {factor_id} 计算失败: {str(e)}")
                        continue
                
                if not all_data:
                    logger.warning("所有因子计算失败，返回空DataFrame")
                    return pd.DataFrame()
                
                result_df = pd.concat(all_data, axis=1)
                
            else:
                # 使用全局API计算
                all_data = []
                
                for factor_id in factor_ids:
                    try:
                        factor_result = api.calculate_factors(
                            factor_ids=[factor_id],
                            symbols=symbols,
                            timeframe=timeframe,
                            start_date=start_date,
                            end_date=end_date
                        )
                        
                        if factor_result is not None and not factor_result.empty:
                            if factor_id in factor_result.columns:
                                all_data.append(factor_result[[factor_id]])
                            elif len(factor_result.columns) > 0:
                                temp_df = factor_result.iloc[:, [0]].copy()
                                temp_df.columns = [factor_id]
                                all_data.append(temp_df)
                                
                    except Exception as e:
                        logger.warning(f"因子 {factor_id} 计算失败: {str(e)}")
                        continue
                
                if not all_data:
                    logger.warning("所有因子计算失败，返回空DataFrame")
                    return pd.DataFrame()
                
                result_df = pd.concat(all_data, axis=1)
            
            # 确保MultiIndex格式 (date, symbol)
            if not isinstance(result_df.index, pd.MultiIndex):
                if 'date' in result_df.columns and 'symbol' in result_df.columns:
                    result_df = result_df.set_index(['date', 'symbol'])
            
            logger.info(f"因子计算完成: {result_df.shape}")
            return result_df
            
        except Exception as e:
            logger.error(f"因子计算失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def _calculate_factor_from_registry(self, factor_id: str, factor_info,
                                       symbols: List[str], start_date: datetime,
                                       end_date: datetime, timeframe: str) -> pd.DataFrame:
        """从注册表中的因子函数计算因子值"""
        try:
            # 🔥 关键实现：直接使用注册表中的因子函数
            # 1. 加载数据
            from factor_system.factor_engine.providers.etf_cross_section_provider import ETFCrossSectionDataManager
            data_manager = ETFCrossSectionDataManager()
            
            # 2. 获取时间序列数据
            price_data = data_manager.get_time_series_data(
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
                symbols
            )
            
            if price_data.empty:
                logger.warning(f"因子 {factor_id}: 无数据")
                return pd.DataFrame()
            
            # 3. 按symbol分组计算因子
            results = []
            factor_func = factor_info.function
            
            for etf_code in symbols:
                etf_data = price_data[price_data['etf_code'] == etf_code].copy()
                if etf_data.empty:
                    continue
                
                try:
                    # 调用因子函数
                    factor_values = factor_func(etf_data)
                    
                    # 构建结果DataFrame
                    result_df = pd.DataFrame({
                        'date': etf_data['trade_date'],
                        'symbol': etf_code,
                        factor_id: factor_values
                    })
                    results.append(result_df)
                    
                except Exception as e:
                    logger.debug(f"因子 {factor_id} 计算失败 ({etf_code}): {str(e)}")
                    continue
            
            if not results:
                return pd.DataFrame()
            
            # 4. 合并结果
            combined = pd.concat(results, ignore_index=True)
            combined['date'] = pd.to_datetime(combined['date'], format='%Y%m%d')
            combined = combined.set_index(['date', 'symbol'])
            
            return combined
            
        except Exception as e:
            logger.error(f"从注册表计算因子失败 {factor_id}: {str(e)}")
            # 回退到API
            return api.calculate_factors(
                factor_ids=[factor_id],
                symbols=symbols,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )

    def _calculate_single_factor(self, task: CalculationTask) -> CalculationResult:
        """计算单个因子"""
        start_time = datetime.now()

        try:
            # 解析因子参数
            factor_id = task.variant.base_factor_id
            parameters = task.variant.parameters

            # 构建因子计算参数
            calc_params = {
                "factor_ids": [factor_id],
                "symbols": task.symbols,
                "timeframe": task.timeframe,
                "start_date": task.start_date,
                "end_date": task.end_date
            }

            # 添加因子参数（如果有的话）
            if parameters:
                calc_params.update(parameters)

            # 执行因子计算
            factor_data = api.calculate_factors(**calc_params)

            if factor_data is None or factor_data.empty:
                return CalculationResult(
                    task_id=task.task_id,
                    variant_id=task.variant_id,
                    factor_data=pd.DataFrame(),
                    success=False,
                    error_message="因子计算返回空结果"
                )

            # 重命名列以包含变体ID
            if not factor_data.empty:
                factor_columns = [col for col in factor_data.columns if col not in ['symbol', 'date']]
                rename_dict = {}
                for col in factor_columns:
                    rename_dict[col] = f"{task.variant.variant_id}_{col}"
                factor_data = factor_data.rename(columns=rename_dict)

            calculation_time = (datetime.now() - start_time).total_seconds()

            return CalculationResult(
                task_id=task.task_id,
                variant_id=task.variant_id,
                factor_data=factor_data,
                success=True,
                calculation_time=calculation_time
            )

        except Exception as e:
            calculation_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"因子计算失败: {str(e)}"
            logger.error(f"{task.task_id}: {error_msg}")

            return CalculationResult(
                task_id=task.task_id,
                variant_id=task.variant_id,
                factor_data=pd.DataFrame(),
                success=False,
                error_message=error_msg,
                calculation_time=calculation_time
            )

    def _process_batch(self, batch_tasks: List[CalculationTask]) -> List[CalculationResult]:
        """处理一批计算任务"""
        results = []

        # 使用多进程处理
        with ProcessPoolExecutor(max_workers=min(self.max_workers, len(batch_tasks))) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(self._calculate_single_factor, task): task
                for task in batch_tasks
            }

            # 收集结果
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)

                    # 进度报告
                    if result.success:
                        logger.debug(f"✅ {task.task_id} 完成 ({result.calculation_time:.3f}s)")
                    else:
                        logger.warning(f"❌ {task.task_id} 失败: {result.error_message}")

                except Exception as e:
                    logger.error(f"❌ {task.task_id} 异常: {str(e)}")
                    results.append(CalculationResult(
                        task_id=task.task_id,
                        variant_id=task.variant.variant_id,
                        factor_data=pd.DataFrame(),
                        success=False,
                        error_message=f"执行异常: {str(e)}"
                    ))

        return results

    def calculate_factors_batch(self,
                              variants: List[FactorVariant],
                              symbols: List[str],
                              timeframe: str,
                              start_date: datetime,
                              end_date: datetime,
                              output_dir: str) -> Dict[str, pd.DataFrame]:
        """
        批量计算因子

        Args:
            variants: 因子变体列表
            symbols: ETF代码列表
            timeframe: 时间框架
            start_date: 开始日期
            end_date: 结束日期
            output_dir: 输出目录

        Returns:
            因子数据字典 {variant_id: factor_data}
        """
        logger.info(f"开始批量计算 {len(variants)} 个因子")
        logger.info(f"标的池: {len(symbols)} 个ETF")
        logger.info(f"时间范围: {start_date.date()} ~ {end_date.date()}")

        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 创建计算任务
        tasks = []
        for i, variant in enumerate(variants):
            task = CalculationTask(
                task_id=f"task_{i:04d}",
                variant=variant,
                symbols=symbols,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            tasks.append(task)

        # 分批处理
        all_results = {}
        total_batches = (len(tasks) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(tasks))
            batch_tasks = tasks[start_idx:end_idx]

            logger.info(f"处理第 {batch_idx + 1}/{total_batches} 批次 ({len(batch_tasks)} 个因子)")

            # 检查内存使用
            memory_percent = self._check_memory_usage()

            # 处理当前批次
            batch_results = self._process_batch(batch_tasks)

            # 保存中间结果
            for result in batch_results:
                all_results[result.variant_id] = result

                # 保存到文件
                if result.success and not result.factor_data.empty:
                    factor_file = output_path / f"{result.variant_id}.parquet"
                    result.factor_data.to_parquet(factor_file, index=False)

            # 批次统计
            success_count = sum(1 for r in batch_results if r.success)
            total_time = sum(r.calculation_time for r in batch_results if r.calculation_time)
            avg_time = total_time / len(batch_results) if batch_results else 0

            logger.info(f"批次 {batch_idx + 1} 完成: {success_count}/{len(batch_tasks)} 成功, "
                       f"平均耗时 {avg_time:.3f}s")

            # 强制垃圾回收
            gc.collect()

        # 保存计算统计
        self._save_calculation_stats(all_results, output_path)

        # 返回成功的因子数据
        successful_factors = {}
        for variant_id, result in all_results.items():
            if result.success and not result.factor_data.empty:
                successful_factors[variant_id] = result.factor_data

        logger.info(f"批量计算完成: {len(successful_factors)}/{len(variants)} 个因子成功")

        return successful_factors

    def _save_calculation_stats(self, results: Dict[str, CalculationResult], output_path: Path):
        """保存计算统计信息"""
        stats = {
            "total_tasks": len(results),
            "successful_tasks": sum(1 for r in results.values() if r.success),
            "failed_tasks": sum(1 for r in results.values() if not r.success),
            "total_calculation_time": sum(r.calculation_time or 0 for r in results.values()),
            "average_calculation_time": np.mean([r.calculation_time for r in results.values() if r.calculation_time]),
            "successful_variants": [variant_id for variant_id, r in results.items() if r.success],
            "failed_variants": [(variant_id, r.error_message) for variant_id, r in results.items() if not r.success],
            "generated_at": datetime.now().isoformat()
        }

        stats_file = output_path / "calculation_stats.json"
        import json
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        logger.info(f"计算统计已保存到: {stats_file}")

    def load_calculated_factors(self, input_dir: str) -> Dict[str, pd.DataFrame]:
        """加载已计算的因子数据"""
        input_path = Path(input_dir)
        factors = {}

        for factor_file in input_path.glob("*.parquet"):
            variant_id = factor_file.stem
            try:
                factor_data = pd.read_parquet(factor_file)
                factors[variant_id] = factor_data
                logger.debug(f"加载因子: {variant_id} ({len(factor_data)} 条记录)")
            except Exception as e:
                logger.error(f"加载因子失败 {variant_id}: {str(e)}")

        logger.info(f"成功加载 {len(factors)} 个因子数据")
        return factors


@safe_operation
def calculate_all_etf_factors(symbols: List[str],
                            start_date: str,
                            end_date: str,
                            timeframe: str = "daily",
                            output_dir: str = None) -> Dict[str, pd.DataFrame]:
    """
    便捷函数：计算所有ETF因子

    Args:
        symbols: ETF代码列表
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        timeframe: 时间框架
        output_dir: 输出目录

    Returns:
        因子数据字典
    """
    # 参数处理
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    if output_dir is None:
        output_dir = f"factor_system/factor_output/etf_cross_section/calculated_factors_{start_date}_{end_date}"

    # 生成候选因子
    generator = ETFCandidateFactorGenerator()
    variants = generator.generate_all_variants()

    logger.info(f"准备计算 {len(variants)} 个候选因子")

    # 批量计算
    calculator = BatchFactorCalculator()
    factors = calculator.calculate_factors_batch(
        variants=variants,
        symbols=symbols,
        timeframe=timeframe,
        start_date=start_dt,
        end_date=end_dt,
        output_dir=output_dir
    )

    return factors


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 测试参数
    test_symbols = ['510300.SH', '159915.SZ', '515030.SH']
    start_date = "2025-09-01"
    end_date = "2025-10-14"

    # 计算因子
    factors = calculate_all_etf_factors(
        symbols=test_symbols,
        start_date=start_date,
        end_date=end_date
    )

    print(f"计算完成，获得 {len(factors)} 个因子数据")