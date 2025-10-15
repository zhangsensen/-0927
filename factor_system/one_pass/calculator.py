"""One Pass全量因子计算器"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..factor_engine.core.registry import FactorRegistry, get_global_registry
from ..factor_engine.providers.base import DataProvider
from .health_monitor import HealthMonitor
from .safety_constraints import SafetyConstraints

logger = logging.getLogger(__name__)


class OnePassCalculator:
    """
    One Pass全量因子计算器

    核心原则:
    - 遍历注册表所有因子，不做前置筛选
    - 4条最小安全约束保证
    - 容错机制：失败因子记录但不阻塞
    - VectorBT优先，避免循环
    """

    def __init__(
        self,
        data_provider: DataProvider,
        registry: Optional[FactorRegistry] = None,
        safety_constraints: Optional[SafetyConstraints] = None,
        health_monitor: Optional[HealthMonitor] = None,
    ):
        """
        初始化One Pass计算器

        Args:
            data_provider: 数据提供者
            registry: 因子注册表
            safety_constraints: 安全约束引擎
            health_monitor: 健康监控系统
        """
        self.provider = data_provider
        self.registry = registry or get_global_registry(include_money_flow=True)
        self.safety_constraints = safety_constraints or SafetyConstraints()
        self.health_monitor = health_monitor or HealthMonitor()

        logger.info(f"初始化OnePass计算器: {len(self.registry.factors)}个已注册因子")

    def calculate_all_factors(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        batch_size: int = 50,
        use_parallel: bool = True,
        n_jobs: int = -1,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        One Pass全量因子计算

        Args:
            symbols: 股票代码列表
            timeframe: 时间框架
            start_date: 开始日期
            end_date: 结束日期
            batch_size: 批处理大小
            use_parallel: 是否使用并行计算
            n_jobs: 并行作业数

        Returns:
            (因子面板DataFrame, 计算报告字典)
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("One Pass全量因子计算开始")
        logger.info("=" * 60)
        logger.info(
            f"参数: {len(symbols)}个标的, {timeframe}, "
            f"{start_date.date()}~{end_date.date()}"
        )

        # 1. 获取所有注册因子
        all_factor_ids = self._get_all_registered_factors()
        logger.info(f"可用因子总数: {len(all_factor_ids)}")

        # 2. 批量计算因子
        factor_panels = []
        calculation_reports = []

        # 分批处理以控制内存使用
        factor_batches = self._create_factor_batches(all_factor_ids, batch_size)

        for i, factor_batch in enumerate(factor_batches, 1):
            logger.info(f"处理第{i}/{len(factor_batches)}批: {len(factor_batch)}个因子")

            batch_panel, batch_report = self._calculate_factor_batch(
                factor_batch,
                symbols,
                timeframe,
                start_date,
                end_date,
                use_parallel=use_parallel,
                n_jobs=n_jobs,
            )

            if not batch_panel.empty:
                factor_panels.append(batch_panel)
                calculation_reports.append(batch_report)

        # 3. 合并所有批次的因子面板
        if factor_panels:
            full_panel = self._merge_factor_panels(factor_panels)
        else:
            logger.warning("所有批次计算失败，返回空面板")
            full_panel = pd.DataFrame()

        # 4. 生成计算报告
        elapsed_time = time.time() - start_time
        final_report = self._generate_final_report(
            calculation_reports, elapsed_time, full_panel.shape
        )

        logger.info("=" * 60)
        logger.info("One Pass全量因子计算完成")
        logger.info(f"总耗时: {elapsed_time:.2f}秒")
        logger.info(
            f"成功因子: {final_report['successful_factors']}/{final_report['total_factors']}"
        )
        logger.info(f"面板形状: {full_panel.shape}")
        logger.info("=" * 60)

        return full_panel, final_report

    def _get_all_registered_factors(self) -> List[str]:
        """获取所有已注册的因子ID"""
        # 合并注册表中的因子和动态加载的因子
        registered_factors = list(self.registry.factors.keys())
        metadata_factors = list(self.registry.metadata.keys())

        # 去重并排序
        all_factors = sorted(set(registered_factors + metadata_factors))

        logger.info(
            f"注册表因子: {len(registered_factors)}, 元数据因子: {len(metadata_factors)}"
        )
        logger.info(f"去重后总数: {len(all_factors)}")

        return all_factors

    def _create_factor_batches(
        self, factor_ids: List[str], batch_size: int
    ) -> List[List[str]]:
        """创建因子批次"""
        batches = []
        for i in range(0, len(factor_ids), batch_size):
            batch = factor_ids[i : i + batch_size]
            batches.append(batch)
        return batches

    def _calculate_factor_batch(
        self,
        factor_ids: List[str],
        symbols: List[str],
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        use_parallel: bool = True,
        n_jobs: int = -1,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        计算单个因子批次

        Returns:
            (因子面板DataFrame, 批次报告)
        """
        batch_start_time = time.time()
        batch_report = {
            "factor_ids": factor_ids,
            "total_factors": len(factor_ids),
            "successful_factors": 0,
            "failed_factors": 0,
            "error_details": [],
            "warnings": [],
        }

        try:
            # 1. 加载原始数据
            logger.debug(f"加载原始数据: {len(symbols)}个标的")
            raw_data = self.provider.load_price_data(
                symbols, timeframe, start_date, end_date
            )

            if raw_data.empty:
                logger.warning("原始数据为空")
                return pd.DataFrame(), batch_report

            # 2. 应用安全约束
            validated_data, constraint_warnings = self.safety_constraints.validate_data(
                raw_data, timeframe
            )
            batch_report["warnings"].extend(constraint_warnings)

            # 3. 向量化批量计算
            if use_parallel and len(symbols) > 1:
                factor_panel = self._parallel_factor_calculation(
                    factor_ids, validated_data, symbols, n_jobs
                )
            else:
                factor_panel = self._sequential_factor_calculation(
                    factor_ids, validated_data, symbols
                )

            # 4. 健康监控（只告警不阻塞）
            health_warnings = self.health_monitor.check_factor_health(
                factor_panel, factor_ids
            )
            batch_report["warnings"].extend(health_warnings)

            # 5. 更新成功统计
            successful_factors = [
                fid for fid in factor_ids if fid in factor_panel.columns
            ]
            batch_report["successful_factors"] = len(successful_factors)
            batch_report["failed_factors"] = len(factor_ids) - len(successful_factors)

            # 记录失败的因子
            failed_factors = set(factor_ids) - set(successful_factors)
            for failed_factor in failed_factors:
                batch_report["error_details"].append(
                    {"factor_id": failed_factor, "error": "计算结果为空或未生成"}
                )

            elapsed_time = time.time() - batch_start_time
            logger.info(
                f"批次完成: {batch_report['successful_factors']}/{batch_report['total_factors']} "
                f"成功, 耗时{elapsed_time:.2f}秒"
            )

            return factor_panel, batch_report

        except Exception as e:
            elapsed_time = time.time() - batch_start_time
            error_msg = f"批次计算失败: {str(e)}"
            logger.error(error_msg)

            batch_report["error_details"].append(
                {"error": error_msg, "elapsed_time": elapsed_time}
            )

            return pd.DataFrame(), batch_report

    def _parallel_factor_calculation(
        self,
        factor_ids: List[str],
        data: pd.DataFrame,
        symbols: List[str],
        n_jobs: int = -1,
    ) -> pd.DataFrame:
        """并行因子计算"""
        from joblib import Parallel, delayed

        def _process_symbol(symbol: str) -> pd.DataFrame:
            symbol_data = data.xs(symbol, level="symbol")
            return self._calculate_single_symbol_factors(
                factor_ids, symbol_data, symbol
            )

        # 并行处理所有股票
        effective_n_jobs = min(n_jobs, len(symbols)) if n_jobs > 0 else len(symbols)
        results = Parallel(n_jobs=effective_n_jobs)(
            delayed(_process_symbol)(symbol) for symbol in symbols
        )

        # 合并结果
        valid_results = [r for r in results if r is not None and not r.empty]
        if valid_results:
            return pd.concat(valid_results)
        else:
            return pd.DataFrame()

    def _sequential_factor_calculation(
        self,
        factor_ids: List[str],
        data: pd.DataFrame,
        symbols: List[str],
    ) -> pd.DataFrame:
        """顺序因子计算"""
        if isinstance(data.index, pd.MultiIndex):
            # 多标的顺序处理
            results = []
            for symbol in symbols:
                try:
                    symbol_data = data.xs(symbol, level="symbol")
                    symbol_results = self._calculate_single_symbol_factors(
                        factor_ids, symbol_data, symbol
                    )
                    if not symbol_results.empty:
                        results.append(symbol_results)
                except Exception as e:
                    logger.warning(f"标的{symbol}计算失败: {e}")
                    continue

            if results:
                return pd.concat(results)
            else:
                return pd.DataFrame()
        else:
            # 单标的情况
            return self._calculate_single_symbol_factors(factor_ids, data, None)

    def _calculate_single_symbol_factors(
        self,
        factor_ids: List[str],
        data: pd.DataFrame,
        symbol: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        计算单个标的的所有因子（向量化优先）

        Args:
            factor_ids: 因子ID列表
            data: 单个标的的OHLCV数据
            symbol: 标的代码（可选）

        Returns:
            因子DataFrame
        """
        results = {}

        for factor_id in factor_ids:
            try:
                # 获取因子实例
                factor = self.registry.get_factor(factor_id)

                # 数据验证
                if not factor.validate_data(data):
                    logger.debug(f"因子{factor_id}数据验证失败: {symbol}")
                    results[factor_id] = pd.Series(
                        np.nan, index=data.index, name=factor_id
                    )
                    continue

                # 计算因子
                factor_values = factor.calculate(data)

                # 验证返回类型
                if not isinstance(factor_values, pd.Series):
                    logger.debug(f"因子{factor_id}返回类型错误: {type(factor_values)}")
                    results[factor_id] = pd.Series(
                        np.nan, index=data.index, name=factor_id
                    )
                    continue

                # 确保索引对齐
                if len(factor_values) != len(data):
                    factor_values = factor_values.reindex(data.index)

                results[factor_id] = factor_values

            except Exception as e:
                logger.debug(f"因子{factor_id}计算失败: {symbol}, {str(e)}")
                results[factor_id] = pd.Series(np.nan, index=data.index, name=factor_id)
                continue

        # 构建结果DataFrame
        if results:
            result_df = pd.DataFrame(results)

            # 添加标的标识
            if symbol is not None:
                result_df["symbol"] = symbol
                result_df = result_df.set_index("symbol", append=True)

            return result_df
        else:
            return pd.DataFrame()

    def _merge_factor_panels(self, panels: List[pd.DataFrame]) -> pd.DataFrame:
        """合并多个因子面板"""
        if not panels:
            return pd.DataFrame()

        # 检查列的一致性
        all_columns = set()
        for panel in panels:
            all_columns.update(panel.columns)

        # 统一列顺序
        sorted_columns = sorted(all_columns - {"symbol"})

        # 确保所有面板具有相同的列
        aligned_panels = []
        for panel in panels:
            missing_columns = sorted_columns - set(panel.columns)
            if missing_columns:
                # 添加缺失的列（填充NaN）
                for col in missing_columns:
                    panel[col] = np.nan

            # 重新排序列
            panel_columns = ["symbol"] if "symbol" in panel.index.names else []
            panel_columns += sorted_columns
            aligned_panels.append(panel[sorted_columns])

        # 合并所有面板
        merged_panel = pd.concat(aligned_panels, axis=0, sort=False)

        logger.info(f"合并面板完成: {len(panels)}个面板 -> {merged_panel.shape}")

        return merged_panel

    def _generate_final_report(
        self,
        batch_reports: List[Dict[str, Any]],
        elapsed_time: float,
        panel_shape: Tuple[int, int],
    ) -> Dict[str, Any]:
        """生成最终计算报告"""
        total_factors = sum(r["total_factors"] for r in batch_reports)
        successful_factors = sum(r["successful_factors"] for r in batch_reports)
        failed_factors = total_factors - successful_factors

        # 收集所有错误
        all_errors = []
        all_warnings = []
        for report in batch_reports:
            all_errors.extend(report.get("error_details", []))
            all_warnings.extend(report.get("warnings", []))

        # 按错误类型分组
        error_summary = {}
        for error in all_errors:
            if isinstance(error, dict):
                error_type = error.get("error", "未知错误")
            else:
                error_type = str(error)

            error_summary[error_type] = error_summary.get(error_type, 0) + 1

        return {
            "total_factors": total_factors,
            "successful_factors": successful_factors,
            "failed_factors": failed_factors,
            "success_rate": (
                successful_factors / total_factors if total_factors > 0 else 0
            ),
            "panel_shape": panel_shape,
            "elapsed_time": elapsed_time,
            "throughput": total_factors / elapsed_time if elapsed_time > 0 else 0,
            "error_summary": error_summary,
            "total_errors": len(all_errors),
            "total_warnings": len(all_warnings),
            "batch_count": len(batch_reports),
        }

    def get_factor_categories_summary(self) -> Dict[str, int]:
        """获取因子分类汇总"""
        category_counts = {}

        for factor_id in self._get_all_registered_factors():
            metadata = self.registry.get_metadata(factor_id)
            if metadata:
                category = metadata.get("category", "unknown")
            else:
                # 尝试从因子ID推断分类
                if any(
                    kw in factor_id.lower()
                    for kw in ["rsi", "macd", "stoch", "willr", "cci"]
                ):
                    category = "oscillator"
                elif any(kw in factor_id.lower() for kw in ["ma", "ema", "sma"]):
                    category = "trend"
                elif any(
                    kw in factor_id.lower() for kw in ["momentum", "trend", "position"]
                ):
                    category = "momentum"
                elif any(kw in factor_id.lower() for kw in ["volume", "obv", "vwap"]):
                    category = "volume"
                elif any(
                    kw in factor_id.lower() for kw in ["atr", "std", "volatility"]
                ):
                    category = "volatility"
                else:
                    category = "technical"

            category_counts[category] = category_counts.get(category, 0) + 1

        return category_counts
