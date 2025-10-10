"""
批量因子计算器 - 替代EnhancedFactorCalculator

基于FactorEngine实现，确保计算逻辑统一
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from factor_system.factor_engine.core.cache import CacheConfig
from factor_system.factor_engine.core.engine import FactorEngine
from factor_system.factor_engine.core.registry import FactorRegistry
from factor_system.factor_engine.providers.parquet_provider import ParquetDataProvider

logger = logging.getLogger(__name__)


class BatchFactorCalculator:
    """
    批量因子计算器

    替代原有的EnhancedFactorCalculator，
    使用FactorEngine作为底层计算引擎，
    确保与回测阶段计算逻辑完全一致
    """

    def __init__(
        self,
        raw_data_dir: Path,
        registry_file: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        enable_cache: bool = True,
    ):
        """
        初始化批量计算器

        ⚠️ 推荐直接使用 factor_engine.api 模块

        Args:
            raw_data_dir: 原始数据目录
            registry_file: 因子注册表文件
            cache_dir: 缓存目录
            enable_cache: 是否启用缓存
        """
        # 使用统一API获取引擎（单例模式）
        from factor_system.factor_engine import api

        # 初始化缓存配置
        if cache_dir is None:
            cache_dir = Path("cache/factor_engine")
        cache_config = CacheConfig(
            memory_size_mb=500,
            disk_cache_dir=cache_dir,
            ttl_hours=24,
            enable_disk=enable_cache,
            enable_memory=enable_cache,
        )

        # 获取全局引擎
        self.engine = api.get_engine(
            raw_data_dir=raw_data_dir,
            registry_file=registry_file,
            cache_config=cache_config,
        )

        # 向后兼容属性
        self.registry = self.engine.registry
        self.data_provider = self.engine.provider

        logger.info(
            f"BatchFactorCalculator初始化完成（使用统一API），已注册{len(self.registry.factors)}个因子"
        )

    def calculate_all_factors(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        factor_ids: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        计算所有因子（兼容EnhancedFactorCalculator接口）

        Args:
            symbol: 股票代码
            timeframe: 时间框架
            start_date: 开始日期
            end_date: 结束日期
            factor_ids: 指定要计算的因子ID列表，None表示计算所有

        Returns:
            因子DataFrame，columns为因子名，index为时间戳
        """
        logger.info(f"开始计算因子: {symbol}, {timeframe}, {start_date} - {end_date}")

        start_time = time.time()

        # 如果未指定因子，使用所有已注册因子
        if factor_ids is None:
            factor_ids = list(self.registry.factors.keys())

        logger.info(f"计算{len(factor_ids)}个因子")

        # 使用FactorEngine计算
        try:
            factors_df = self.engine.calculate_factors(
                factor_ids=factor_ids,
                symbols=[symbol],
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                use_cache=True,
            )

            # 如果是MultiIndex，提取单symbol的数据
            if isinstance(factors_df.index, pd.MultiIndex):
                factors_df = factors_df.xs(symbol, level="symbol")

            calc_time = time.time() - start_time

            logger.info(
                f"✅ 因子计算完成: {factors_df.shape}, "
                f"耗时 {calc_time:.2f}s, "
                f"速度 {len(factors_df) / calc_time:.0f} rows/s"
            )

            return factors_df

        except Exception as e:
            logger.error(f"因子计算失败: {e}", exc_info=True)
            return pd.DataFrame()

    def calculate_comprehensive_factors(
        self,
        df: pd.DataFrame,
        timeframe: str,
    ) -> pd.DataFrame:
        """
        兼容EnhancedFactorCalculator的接口

        Args:
            df: OHLCV数据
            timeframe: 时间框架字符串

        Returns:
            因子DataFrame
        """
        # 时间框架映射
        timeframe_mapping = {
            "1min": "1min",
            "2min": "2min",
            "3min": "3min",
            "5min": "5min",
            "15min": "15min",
            "30min": "30min",
            "60min": "60min",
            "2h": "2h",
            "4h": "4h",
            "daily": "daily",
        }

        normalized_tf = timeframe_mapping.get(timeframe, timeframe)

        # 提取时间范围
        start_date = df.index.min()
        end_date = df.index.max()

        # 假设symbol在数据中（如果不在，使用默认值）
        symbol = "UNKNOWN"
        if "symbol" in df.columns:
            symbol = df["symbol"].iloc[0]

        return self.calculate_all_factors(
            symbol=symbol,
            timeframe=normalized_tf,
            start_date=start_date,
            end_date=end_date,
        )

    def calculate_factors_from_df(
        self,
        df: pd.DataFrame,
        timeframe: str,
        factor_ids: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        直接从 DataFrame 计算因子（新增）

        用于消费重采样数据，避免重复加载

        Args:
            df: OHLCV DataFrame，必须包含 open/high/low/close/volume
            timeframe: 时间框架
            factor_ids: 指定要计算的因子ID列表，None表示计算所有

        Returns:
            因子DataFrame
        """
        if factor_ids is None:
            factor_ids = list(self.registry.factors.keys())

        # 验证必要列
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame缺少必要列: {missing_cols}")

        # 直接调用共享计算器
        from factor_system.shared.factor_calculators import SHARED_CALCULATORS

        factors_dict = {}
        for factor_id in factor_ids:
            try:
                # 🔧 修复：get_factor() 返回的是实例，不需要再次实例化
                factor_instance = self.registry.get_factor(factor_id)
                if factor_instance is None:
                    continue

                # 计算因子
                result = factor_instance.calculate(df)

                if isinstance(result, pd.Series):
                    factors_dict[factor_id] = result
                elif isinstance(result, pd.DataFrame):
                    # 多列因子
                    for col in result.columns:
                        factors_dict[f"{factor_id}_{col}"] = result[col]

            except Exception as e:
                logger.warning(f"因子 {factor_id} 计算失败: {e}")
                continue

        if not factors_dict:
            logger.warning("所有因子计算失败")
            return pd.DataFrame(index=df.index)

        factors_df = pd.DataFrame(factors_dict, index=df.index)
        logger.info(f"✅ 从 DataFrame 计算完成: {factors_df.shape}")

        return factors_df

    def get_cache_stats(self) -> Dict:
        """获取缓存统计"""
        return self.engine.get_cache_stats()

    def clear_cache(self):
        """清空缓存"""
        self.engine.clear_cache()


# 向后兼容：提供EnhancedFactorCalculator别名
EnhancedFactorCalculator = BatchFactorCalculator
