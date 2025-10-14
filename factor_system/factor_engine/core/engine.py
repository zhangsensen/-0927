"""因子引擎核心 - 统一因子计算接口"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from factor_system.factor_engine.core.base_factor import BaseFactor
from factor_system.factor_engine.core.cache import CacheConfig, CacheManager
from factor_system.factor_engine.core.registry import FactorRegistry
from factor_system.factor_engine.providers.base import DataProvider

logger = logging.getLogger(__name__)


class FactorEngine:
    """
    统一因子计算引擎

    职责:
    - 按需计算因子
    - 管理缓存
    - 解析依赖
    - 提供统一API
    """

    def __init__(
        self,
        data_provider: DataProvider,
        registry: Optional[FactorRegistry] = None,
        cache_config: Optional[CacheConfig] = None,
        money_flow_provider: Optional[DataProvider] = None,
    ):
        """
        初始化因子引擎

        Args:
            data_provider: 价格数据提供者
            registry: 因子注册表
            cache_config: 缓存配置
            money_flow_provider: 资金流数据提供者（可选，用于日线资金流因子）
        """
        self.provider = data_provider
        self.money_flow_provider = money_flow_provider
        self.registry = registry or FactorRegistry()
        self.cache = CacheManager(cache_config or CacheConfig())

        logger.info(f"初始化FactorEngine: {len(self.registry.metadata)}个已注册因子")
        if money_flow_provider:
            logger.info("✅ 资金流数据提供者已启用")

    def calculate_factors(
        self,
        factor_ids: List[str],
        symbols: List[str],
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True,
        n_jobs: Optional[int] = None,
        max_ram_mb: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        计算指定因子

        Args:
            factor_ids: 因子ID列表
            symbols: 股票代码列表
            timeframe: 时间框架
            start_date: 开始日期
            end_date: 结束日期
            use_cache: 是否使用缓存

        Returns:
            DataFrame with MultiIndex(timestamp, symbol) and factor columns
        """
        logger.info(
            f"开始计算因子: {len(factor_ids)}个因子, "
            f"{len(symbols)}个标的, {timeframe}, "
            f"{start_date.date()} ~ {end_date.date()}"
        )

        # 1. 创建因子请求
        factor_configs = [{"factor_id": fid, "parameters": {}} for fid in factor_ids]
        factor_requests = self.registry.create_factor_requests(factor_configs)

        if use_cache:
            cached_data, missing_ids = self.cache.get(
                factor_requests,
                symbols,
                timeframe,
                start_date,
                end_date,
            )
            if cached_data is not None:
                logger.info("缓存命中，直接返回")
                return cached_data
        else:
            missing_ids = factor_ids

        # 2. 解析依赖（简化版：暂不处理复杂依赖）
        all_factors = self._resolve_dependencies(missing_ids)

        factor_param_map = {req.factor_id: req.parameters for req in factor_requests}

        # 3. 加载原始数据
        logger.info(f"加载原始数据: {len(symbols)}个标的")
        raw_data = self.provider.load_price_data(
            symbols, timeframe, start_date, end_date
        )

        if raw_data.empty:
            logger.warning("原始数据为空")
            return pd.DataFrame()

        # 3.5. 如果是日线且有资金流提供者，加载并合并资金流数据
        if timeframe in ("1day", "daily") and self.money_flow_provider is not None:
            logger.info("加载资金流数据...")
            money_flow_data = self.money_flow_provider.load_money_flow_data(
                symbols,
                timeframe,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
            )
            if not money_flow_data.empty:
                # 合并资金流数据到价格数据
                raw_data = raw_data.join(money_flow_data, how="left")
                logger.info(f"✅ 资金流数据已合并: {money_flow_data.shape}")

        # 4. 计算因子
        logger.info(f"计算因子: {len(all_factors)}个")
        # 构建参数映射（所有因子默认空参数，缺失时补齐）
        factor_param_map = {fid: factor_param_map.get(fid, {}) for fid in all_factors}

        effective_n_jobs = (
            n_jobs if n_jobs is not None else (self.cache.config.n_jobs or 1)
        )
        effective_max_ram = (
            max_ram_mb if max_ram_mb is not None else self.cache.config.max_ram_mb
        )

        result = self._compute_factors(
            all_factors,
            raw_data,
            factor_param_map,
            n_jobs=effective_n_jobs,
            max_ram_mb=effective_max_ram,
        )

        # 5. 过滤结果，只返回用户请求的因子
        if not result.empty:
            # 确保只返回原始请求的因子，不包含依赖因子
            requested_columns = [fid for fid in factor_ids if fid in result.columns]
            if requested_columns:
                filtered_result = result[requested_columns]

                # 5. 更新缓存（只缓存用户请求的因子）
                if use_cache:
                    self.cache.set(
                        filtered_result,
                        factor_requests,
                        symbols,
                        timeframe,
                        start_date,
                        end_date,
                    )

                logger.info(
                    f"因子计算完成: {filtered_result.shape}, 返回请求的因子: {requested_columns}"
                )
                return filtered_result
            else:
                logger.warning("请求的因子在计算结果中未找到")
                return pd.DataFrame()
        else:
            logger.warning("因子计算结果为空")
            return pd.DataFrame()

    def _resolve_dependencies(self, factor_ids: List[str]) -> List[str]:
        """
        解析因子依赖（支持多级依赖和循环检测）

        Args:
            factor_ids: 因子ID列表

        Returns:
            包含依赖的完整因子列表（拓扑排序）

        Raises:
            ValueError: 检测到循环依赖
        """
        resolved = []
        visiting = set()
        visited = set()

        def _visit(factor_id: str, path: List[str]):
            """深度优先遍历，检测循环依赖"""
            if factor_id in visited:
                return

            if factor_id in visiting:
                cycle = " -> ".join(path + [factor_id])
                raise ValueError(f"检测到循环依赖: {cycle}")

            visiting.add(factor_id)

            # 递归处理依赖
            deps = self.registry.get_dependencies(factor_id)
            for dep in deps:
                _visit(dep, path + [factor_id])

            visiting.remove(factor_id)
            visited.add(factor_id)
            resolved.append(factor_id)

        # 处理所有因子
        for fid in factor_ids:
            try:
                _visit(fid, [])
            except ValueError as e:
                logger.error(f"因子依赖解析失败: {e}")
                raise

        return resolved

    def _compute_factors(
        self,
        factor_ids: List[str],
        raw_data: pd.DataFrame,
        factor_params: Optional[Dict[str, Dict]] = None,
        n_jobs: int = 1,
        max_ram_mb: Optional[int] = None,
    ) -> pd.DataFrame:
        """计算因子"""

        if raw_data.empty:
            logger.warning("原始数据为空")
            return pd.DataFrame()

        is_multi_index = isinstance(raw_data.index, pd.MultiIndex)

        if is_multi_index:
            symbols = raw_data.index.get_level_values("symbol").unique()

            def _process_symbol(sym: str) -> pd.DataFrame:
                # 创建数据副本避免多线程竞争
                symbol_data = raw_data.xs(sym, level="symbol").copy()
                total_rows = len(symbol_data)
                per_factor_mb = symbol_data.memory_usage(deep=True).sum() / 1024 / 1024
                if max_ram_mb and per_factor_mb * len(factor_ids) > max_ram_mb:
                    raise MemoryError(
                        f"symbol={sym} 数据量过大，预计内存 {per_factor_mb * len(factor_ids):.2f}MB > {max_ram_mb}MB"
                    )
                return self._compute_single_symbol_factors(
                    factor_ids, symbol_data, sym, factor_params
                )

            if n_jobs > 1 and len(symbols) > 1:
                from joblib import Parallel, delayed

                results = Parallel(n_jobs=n_jobs)(
                    delayed(_process_symbol)(sym) for sym in symbols
                )
            else:
                results = [_process_symbol(sym) for sym in symbols]

            results = [res for res in results if res is not None and not res.empty]
            if not results:
                return pd.DataFrame()

            return pd.concat(results)

        return self._compute_single_symbol_factors(
            factor_ids, raw_data, symbol=None, factor_params=factor_params
        )

    def _compute_single_symbol_factors(
        self,
        factor_ids: List[str],
        raw_data: pd.DataFrame,
        symbol: Optional[str] = None,
        factor_params: Optional[Dict[str, Dict]] = None,
    ) -> pd.DataFrame:
        """
        计算单个symbol的因子（增强容错）

        Args:
            factor_ids: 因子ID列表
            raw_data: 单个symbol的OHLCV数据（普通Index）
            symbol: 股票代码（可选）
            factor_params: 因子参数字典

        Returns:
            因子数据DataFrame
        """
        results = {}
        errors = []

        for factor_id in factor_ids:
            try:
                # 获取因子实例
                params = factor_params.get(factor_id, {}) if factor_params else {}
                factor = self.registry.get_factor(factor_id, **params)

                # 验证数据
                if not factor.validate_data(raw_data):
                    error_msg = f"数据验证失败"
                    errors.append((factor_id, error_msg))
                    logger.warning(f"因子{factor_id}{error_msg}: symbol={symbol}")
                    # 填充NaN保持结构一致
                    results[factor_id] = pd.Series(
                        np.nan, index=raw_data.index, name=factor_id
                    )
                    continue

                # 计算因子
                factor_values = factor.calculate(raw_data)

                # 验证返回类型
                if not isinstance(factor_values, pd.Series):
                    error_msg = f"返回类型错误: {type(factor_values)}"
                    errors.append((factor_id, error_msg))
                    logger.warning(f"因子{factor_id}{error_msg}: symbol={symbol}")
                    results[factor_id] = pd.Series(
                        np.nan, index=raw_data.index, name=factor_id
                    )
                    continue

                # 验证索引长度
                if len(factor_values) != len(raw_data):
                    error_msg = (
                        f"索引长度不匹配: {len(factor_values)} vs {len(raw_data)}"
                    )
                    errors.append((factor_id, error_msg))
                    logger.warning(f"因子{factor_id}{error_msg}: symbol={symbol}")
                    # 尝试对齐索引
                    factor_values = factor_values.reindex(raw_data.index)

                results[factor_id] = factor_values
                logger.debug(f"✓ 因子{factor_id}计算成功: symbol={symbol}")

            except Exception as e:
                error_msg = str(e)
                errors.append((factor_id, error_msg))
                logger.error(
                    f"因子{factor_id}计算异常: symbol={symbol}, error={error_msg}",
                    exc_info=True,
                )
                # 填充NaN保持结构一致
                results[factor_id] = pd.Series(
                    np.nan, index=raw_data.index, name=factor_id
                )

        # 汇总错误信息
        if errors:
            success_count = len(factor_ids) - len(errors)
            logger.warning(
                f"symbol={symbol}: {success_count}/{len(factor_ids)}个因子计算成功, "
                f"{len(errors)}个失败: {[e[0] for e in errors[:5]]}"
            )

        if not results:
            logger.error(f"symbol={symbol}: 所有因子计算失败")
            return pd.DataFrame()

        # 合并结果
        result_df = pd.DataFrame(results)

        # 如果提供了symbol，添加到MultiIndex
        if symbol is not None:
            result_df["symbol"] = symbol
            result_df = result_df.set_index("symbol", append=True)

        return result_df

    def calculate_single_factor(
        self,
        factor_id: str,
        symbols: List[str],
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.Series:
        """
        计算单个因子（便捷方法）

        Args:
            factor_id: 因子ID
            symbols: 股票代码列表
            timeframe: 时间框架
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            因子值Series
        """
        result = self.calculate_factors(
            [factor_id], symbols, timeframe, start_date, end_date
        )

        if result.empty:
            return pd.Series()

        return result[factor_id]

    def prewarm_cache(
        self,
        factor_ids: List[str],
        symbols: List[str],
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ):
        """
        预热缓存

        Args:
            factor_ids: 因子ID列表
            symbols: 股票代码列表
            timeframe: 时间框架
            start_date: 开始日期
            end_date: 结束日期
        """
        logger.info("开始预热缓存...")
        self.calculate_factors(
            factor_ids, symbols, timeframe, start_date, end_date, use_cache=True
        )
        logger.info("缓存预热完成")

    def get_cache_stats(self) -> Dict:
        """获取缓存统计"""
        return self.cache.get_stats()

    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
