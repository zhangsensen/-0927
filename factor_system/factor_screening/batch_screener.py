#!/usr/bin/env python3
"""
批量因子筛选器
作者：量化首席工程师
版本：1.0.0
日期：2025-09-30

功能：
1. 批量处理多股票、多时间框架
2. 并行执行筛选任务
3. 统一报告生成和汇总
4. 错误处理和重试机制
5. 进度监控和性能统计
"""

import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
import logging
import time
import json
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import traceback
from datetime import datetime
import psutil
import threading
import queue

from config_manager import ConfigManager, ScreeningConfig, BatchConfig
from professional_factor_screener import ProfessionalFactorScreener, FactorMetrics

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """任务结果"""
    task_id: str
    symbol: str
    timeframe: str
    status: str  # "success", "failed", "timeout"
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    factor_count: int = 0
    significant_factors: int = 0
    top_factors: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    results: Optional[Dict[str, FactorMetrics]] = None
    memory_usage_mb: float = 0.0


@dataclass
class BatchResult:
    """批量结果"""
    batch_id: str
    task_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    task_results: List[TaskResult] = field(default_factory=list)
    summary_stats: Dict[str, Any] = field(default_factory=dict)
    performance_stats: Dict[str, Any] = field(default_factory=dict)


class ProgressMonitor:
    """进度监控器"""
    
    def __init__(self, total_tasks: int):
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.start_time = time.time()
        self.lock = threading.Lock()
        self.logger = logging.getLogger(f"{__name__}.ProgressMonitor")
    
    def update(self, success: bool = True):
        """更新进度"""
        with self.lock:
            if success:
                self.completed_tasks += 1
            else:
                self.failed_tasks += 1
            
            total_finished = self.completed_tasks + self.failed_tasks
            progress = total_finished / self.total_tasks * 100
            elapsed = time.time() - self.start_time
            
            if total_finished > 0:
                eta = elapsed * (self.total_tasks - total_finished) / total_finished
                self.logger.info(
                    f"进度: {total_finished}/{self.total_tasks} ({progress:.1f}%) "
                    f"成功: {self.completed_tasks}, 失败: {self.failed_tasks} "
                    f"耗时: {elapsed:.1f}s, 预计剩余: {eta:.1f}s"
                )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        elapsed = time.time() - self.start_time
        total_finished = self.completed_tasks + self.failed_tasks
        
        return {
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": self.completed_tasks / max(total_finished, 1),
            "elapsed_seconds": elapsed,
            "tasks_per_second": total_finished / max(elapsed, 1),
            "is_finished": total_finished >= self.total_tasks
        }


class BatchScreener:
    """批量筛选器"""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config_manager = config_manager or ConfigManager()
        self.logger = self._setup_logger()
        self.results_cache = {}
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger(f"{__name__}.BatchScreener")
        return logger
    
    def _create_task_id(self, symbol: str, timeframe: str) -> str:
        """创建任务ID"""
        return f"{symbol}_{timeframe}_{int(time.time())}"
    
    def _execute_single_task(self, config: ScreeningConfig, 
                           task_id: str) -> TaskResult:
        """执行单个筛选任务"""
        symbol = config.symbols[0] if config.symbols else "UNKNOWN"
        timeframe = config.timeframes[0] if config.timeframes else "UNKNOWN"
        
        result = TaskResult(
            task_id=task_id,
            symbol=symbol,
            timeframe=timeframe,
            status="running",
            start_time=datetime.now()
        )
        
        try:
            # 记录内存使用
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # 创建筛选器实例
            screener = ProfessionalFactorScreener(
                data_root=config.data_root,
                config=config
            )
            
            # 执行筛选
            self.logger.info(f"开始执行任务: {task_id} ({symbol} {timeframe})")
            
            screening_results = screener.screen_factors_comprehensive(
                symbol=symbol,
                timeframe=timeframe
            )
            
            # 获取顶级因子
            top_factors = screener.get_top_factors(
                screening_results, 
                top_n=10, 
                min_score=0.0, 
                require_significant=False
            )
            
            # 统计显著因子
            significant_count = sum(
                1 for metrics in screening_results.values()
                if getattr(metrics, 'is_significant', False)
            )
            
            # 记录结果
            result.status = "success"
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            result.factor_count = len(screening_results)
            result.significant_factors = significant_count
            result.top_factors = [f.name for f in top_factors[:5]]
            result.results = screening_results
            
            # 记录内存使用
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            result.memory_usage_mb = memory_after - memory_before
            
            self.logger.info(
                f"任务完成: {task_id} - "
                f"因子数: {result.factor_count}, "
                f"显著因子: {result.significant_factors}, "
                f"耗时: {result.duration_seconds:.1f}s"
            )
            
        except Exception as e:
            result.status = "failed"
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            result.error_message = str(e)
            
            self.logger.error(f"任务失败: {task_id} - {str(e)}")
            self.logger.debug(f"错误详情: {traceback.format_exc()}")
        
        return result
    
    def run_batch(self, batch_config: BatchConfig) -> BatchResult:
        """运行批量筛选"""
        batch_id = f"batch_{int(time.time())}"
        
        batch_result = BatchResult(
            batch_id=batch_id,
            task_name=batch_config.task_name,
            start_time=datetime.now(),
            total_tasks=len(batch_config.screening_configs)
        )
        
        self.logger.info(f"开始批量筛选: {batch_id}")
        self.logger.info(f"任务数量: {batch_result.total_tasks}")
        
        # 验证配置
        for i, config in enumerate(batch_config.screening_configs):
            errors = self.config_manager.validate_config(config)
            if errors:
                self.logger.error(f"配置验证失败 [{i}]: {errors}")
                if not batch_config.continue_on_error:
                    raise ValueError(f"配置验证失败: {errors}")
        
        # 创建进度监控器
        progress_monitor = ProgressMonitor(batch_result.total_tasks)
        
        # 并行执行任务
        if batch_config.enable_task_parallel and batch_result.total_tasks > 1:
            batch_result.task_results = self._run_parallel_tasks(
                batch_config, progress_monitor
            )
        else:
            batch_result.task_results = self._run_sequential_tasks(
                batch_config, progress_monitor
            )
        
        # 完成批量任务
        batch_result.end_time = datetime.now()
        batch_result.completed_tasks = sum(
            1 for r in batch_result.task_results if r.status == "success"
        )
        batch_result.failed_tasks = sum(
            1 for r in batch_result.task_results if r.status == "failed"
        )
        
        # 生成统计信息
        batch_result.summary_stats = self._generate_summary_stats(batch_result)
        batch_result.performance_stats = progress_monitor.get_stats()
        
        self.logger.info(f"批量筛选完成: {batch_id}")
        self.logger.info(f"成功: {batch_result.completed_tasks}, 失败: {batch_result.failed_tasks}")
        
        return batch_result
    
    def _run_parallel_tasks(self, batch_config: BatchConfig, 
                          progress_monitor: ProgressMonitor) -> List[TaskResult]:
        """并行执行任务"""
        results = []
        max_workers = min(batch_config.max_concurrent_tasks, len(batch_config.screening_configs))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_config = {}
            for config in batch_config.screening_configs:
                task_id = self._create_task_id(
                    config.symbols[0] if config.symbols else "UNKNOWN",
                    config.timeframes[0] if config.timeframes else "UNKNOWN"
                )
                future = executor.submit(self._execute_single_task, config, task_id)
                future_to_config[future] = config
            
            # 收集结果
            for future in as_completed(future_to_config):
                try:
                    result = future.result(timeout=batch_config.screening_configs[0].timeout_minutes * 60)
                    results.append(result)
                    progress_monitor.update(success=(result.status == "success"))
                    
                except concurrent.futures.TimeoutError:
                    config = future_to_config[future]
                    symbol = config.symbols[0] if config.symbols else "UNKNOWN"
                    timeframe = config.timeframes[0] if config.timeframes else "UNKNOWN"
                    
                    timeout_result = TaskResult(
                        task_id=f"timeout_{symbol}_{timeframe}",
                        symbol=symbol,
                        timeframe=timeframe,
                        status="timeout",
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        error_message="任务超时"
                    )
                    results.append(timeout_result)
                    progress_monitor.update(success=False)
                    
                except Exception as e:
                    config = future_to_config[future]
                    symbol = config.symbols[0] if config.symbols else "UNKNOWN"
                    timeframe = config.timeframes[0] if config.timeframes else "UNKNOWN"
                    
                    error_result = TaskResult(
                        task_id=f"error_{symbol}_{timeframe}",
                        symbol=symbol,
                        timeframe=timeframe,
                        status="failed",
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        error_message=str(e)
                    )
                    results.append(error_result)
                    progress_monitor.update(success=False)
        
        return results
    
    def _run_sequential_tasks(self, batch_config: BatchConfig,
                            progress_monitor: ProgressMonitor) -> List[TaskResult]:
        """顺序执行任务"""
        results = []
        
        for config in batch_config.screening_configs:
            task_id = self._create_task_id(
                config.symbols[0] if config.symbols else "UNKNOWN",
                config.timeframes[0] if config.timeframes else "UNKNOWN"
            )
            
            result = self._execute_single_task(config, task_id)
            results.append(result)
            progress_monitor.update(success=(result.status == "success"))
            
            # 如果任务失败且不继续执行，则停止
            if result.status == "failed" and not batch_config.continue_on_error:
                break
        
        return results
    
    def _generate_summary_stats(self, batch_result: BatchResult) -> Dict[str, Any]:
        """生成汇总统计"""
        successful_results = [r for r in batch_result.task_results if r.status == "success"]
        
        if not successful_results:
            return {"message": "没有成功的任务"}
        
        # 基础统计
        total_factors = sum(r.factor_count for r in successful_results)
        total_significant = sum(r.significant_factors for r in successful_results)
        avg_duration = sum(r.duration_seconds for r in successful_results) / len(successful_results)
        total_memory = sum(r.memory_usage_mb for r in successful_results)
        
        # 按股票和时间框架分组统计
        by_symbol = {}
        by_timeframe = {}
        
        for result in successful_results:
            # 按股票统计
            if result.symbol not in by_symbol:
                by_symbol[result.symbol] = {
                    "tasks": 0, "factors": 0, "significant": 0, "duration": 0
                }
            by_symbol[result.symbol]["tasks"] += 1
            by_symbol[result.symbol]["factors"] += result.factor_count
            by_symbol[result.symbol]["significant"] += result.significant_factors
            by_symbol[result.symbol]["duration"] += result.duration_seconds
            
            # 按时间框架统计
            if result.timeframe not in by_timeframe:
                by_timeframe[result.timeframe] = {
                    "tasks": 0, "factors": 0, "significant": 0, "duration": 0
                }
            by_timeframe[result.timeframe]["tasks"] += 1
            by_timeframe[result.timeframe]["factors"] += result.factor_count
            by_timeframe[result.timeframe]["significant"] += result.significant_factors
            by_timeframe[result.timeframe]["duration"] += result.duration_seconds
        
        # 顶级因子统计
        all_top_factors = []
        for result in successful_results:
            all_top_factors.extend(result.top_factors)
        
        factor_frequency = {}
        for factor in all_top_factors:
            factor_frequency[factor] = factor_frequency.get(factor, 0) + 1
        
        most_common_factors = sorted(
            factor_frequency.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        return {
            "total_factors": total_factors,
            "total_significant_factors": total_significant,
            "avg_factors_per_task": total_factors / len(successful_results),
            "avg_significant_per_task": total_significant / len(successful_results),
            "avg_duration_seconds": avg_duration,
            "total_memory_usage_mb": total_memory,
            "by_symbol": by_symbol,
            "by_timeframe": by_timeframe,
            "most_common_top_factors": most_common_factors
        }
    
    def save_results(self, batch_result: BatchResult, 
                    output_dir: str = "./output") -> Dict[str, Path]:
        """保存批量结果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = output_dir / f"batch_{batch_result.batch_id}_{timestamp}"
        batch_dir.mkdir(exist_ok=True)
        
        saved_files = {}
        
        # 1. 保存批量结果摘要
        summary_file = batch_dir / "batch_summary.json"
        summary_data = {
            "batch_id": batch_result.batch_id,
            "task_name": batch_result.task_name,
            "start_time": batch_result.start_time.isoformat(),
            "end_time": batch_result.end_time.isoformat() if batch_result.end_time else None,
            "total_tasks": batch_result.total_tasks,
            "completed_tasks": batch_result.completed_tasks,
            "failed_tasks": batch_result.failed_tasks,
            "summary_stats": batch_result.summary_stats,
            "performance_stats": batch_result.performance_stats
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        saved_files["summary"] = summary_file
        
        # 2. 保存任务结果详情
        tasks_file = batch_dir / "task_results.json"
        tasks_data = []
        
        for result in batch_result.task_results:
            task_data = {
                "task_id": result.task_id,
                "symbol": result.symbol,
                "timeframe": result.timeframe,
                "status": result.status,
                "start_time": result.start_time.isoformat(),
                "end_time": result.end_time.isoformat() if result.end_time else None,
                "duration_seconds": result.duration_seconds,
                "factor_count": result.factor_count,
                "significant_factors": result.significant_factors,
                "top_factors": result.top_factors,
                "error_message": result.error_message,
                "memory_usage_mb": result.memory_usage_mb
            }
            tasks_data.append(task_data)
        
        with open(tasks_file, 'w', encoding='utf-8') as f:
            json.dump(tasks_data, f, indent=2, ensure_ascii=False)
        saved_files["tasks"] = tasks_file
        
        # 3. 生成汇总报告CSV
        if batch_result.task_results:
            report_file = batch_dir / "batch_report.csv"
            report_data = []
            
            for result in batch_result.task_results:
                report_data.append({
                    "任务ID": result.task_id,
                    "股票代码": result.symbol,
                    "时间框架": result.timeframe,
                    "状态": result.status,
                    "因子数量": result.factor_count,
                    "显著因子数": result.significant_factors,
                    "耗时(秒)": result.duration_seconds,
                    "内存使用(MB)": result.memory_usage_mb,
                    "顶级因子": ", ".join(result.top_factors[:3]),
                    "错误信息": result.error_message or ""
                })
            
            df = pd.DataFrame(report_data)
            df.to_csv(report_file, index=False, encoding='utf-8-sig')
            saved_files["report"] = report_file
        
        # 4. 保存详细筛选结果（可选）
        details_dir = batch_dir / "detailed_results"
        details_dir.mkdir(exist_ok=True)
        
        for result in batch_result.task_results:
            if result.status == "success" and result.results:
                detail_file = details_dir / f"{result.symbol}_{result.timeframe}_details.json"
                
                # 转换FactorMetrics为可序列化的格式
                serializable_results = {}
                for factor_name, metrics in result.results.items():
                    serializable_results[factor_name] = {
                        "name": metrics.name,
                        "comprehensive_score": metrics.comprehensive_score,
                        "predictive_power_score": getattr(metrics, 'predictive_power_score', 0),
                        "stability_score": getattr(metrics, 'stability_score', 0),
                        "independence_score": getattr(metrics, 'independence_score', 0),
                        "practicality_score": getattr(metrics, 'practicality_score', 0),
                        "short_term_fitness_score": getattr(metrics, 'short_term_fitness_score', 0),
                        "is_significant": getattr(metrics, 'is_significant', False)
                    }
                
                with open(detail_file, 'w', encoding='utf-8') as f:
                    json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"批量结果已保存到: {batch_dir}")
        return saved_files
    
    def generate_comparison_report(self, batch_results: List[BatchResult]) -> pd.DataFrame:
        """生成批量结果对比报告"""
        comparison_data = []
        
        for batch_result in batch_results:
            for task_result in batch_result.task_results:
                if task_result.status == "success":
                    comparison_data.append({
                        "批次ID": batch_result.batch_id,
                        "任务名称": batch_result.task_name,
                        "股票代码": task_result.symbol,
                        "时间框架": task_result.timeframe,
                        "因子数量": task_result.factor_count,
                        "显著因子数": task_result.significant_factors,
                        "显著率": task_result.significant_factors / max(task_result.factor_count, 1),
                        "耗时(秒)": task_result.duration_seconds,
                        "内存使用(MB)": task_result.memory_usage_mb,
                        "顶级因子": ", ".join(task_result.top_factors[:3])
                    })
        
        return pd.DataFrame(comparison_data)


if __name__ == "__main__":
    # 示例用法
    config_manager = ConfigManager()
    batch_screener = BatchScreener(config_manager)
    
    # 创建批量配置
    batch_config = config_manager.create_batch_config(
        task_name="demo_batch",
        symbols=["0700.HK"],
        timeframes=["60min"],
        preset="quick"
    )
    
    print("批量配置摘要:")
    print(config_manager.get_config_summary(batch_config))
    
    # 运行批量筛选（示例）
    # batch_result = batch_screener.run_batch(batch_config)
    # batch_screener.save_results(batch_result)
