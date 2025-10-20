"""回测因子提供器 - 适配共享因子引擎"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from factor_system.factor_engine.core.engine import FactorEngine
from factor_system.factor_engine.core.registry import FactorRegistry
from factor_system.factor_engine.providers.parquet_provider import ParquetDataProvider

logger = logging.getLogger(__name__)


class BacktestFactorProvider:
    """
    回测专用因子提供器

    职责:
    - 为回测提供即时计算的因子数据
    - 集成共享因子引擎
    - 处理因子集合配置
    """

    def __init__(
        self,
        raw_data_dir: Path,
        registry_file: Path | None = None,
    ):
        """
        初始化回测因子提供器

        Args:
            raw_data_dir: 原始数据目录
            registry_file: 因子注册表文件路径
        """
        # 初始化数据提供者
        self.data_provider = ParquetDataProvider(raw_data_dir)

        # 初始化因子注册表
        self.registry = FactorRegistry(registry_file)

        # 初始化因子引擎
        self.engine = FactorEngine(
            data_provider=self.data_provider,
            registry=self.registry,
        )

        # 注册因子
        self._register_factors()

        logger.info("BacktestFactorProvider初始化完成")

    def _register_factors(self):
        """注册因子到引擎"""
        from factor_system.factor_engine.factors import RSI, STOCH, WILLR

        # 注册因子类
        self.registry.register(RSI)
        self.registry.register(STOCH)
        self.registry.register(WILLR)

        logger.info(f"已注册{len(self.registry.factors)}个因子")

    def get_factors_for_backtest(
        self,
        factor_ids: List[str],
        symbols: List[str],
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        为回测提供因子数据

        Args:
            factor_ids: 因子ID列表
            symbols: 股票列表
            timeframe: 时间框架
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            因子数据 DataFrame with MultiIndex(timestamp, symbol) and factor columns
        """
        logger.info(
            f"为回测提供因子: {len(factor_ids)}个因子, "
            f"{len(symbols)}个标的, {timeframe}"
        )

        # 调用共享引擎计算
        factors = self.engine.calculate_factors(
            factor_ids=factor_ids,
            symbols=symbols,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )

        return factors

    def get_factor_set(
        self,
        set_name: str,
        symbols: List[str],
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        获取预定义的因子集合

        Args:
            set_name: 因子集合名称（如 "hk_midfreq_core"）
            symbols: 股票列表
            timeframe: 时间框架
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            因子数据
        """
        # 从注册表读取因子集合
        factor_set = self._load_factor_set(set_name)

        if not factor_set:
            raise ValueError(f"未找到因子集合: {set_name}")

        factor_ids = factor_set.get("factors", [])

        return self.get_factors_for_backtest(
            factor_ids, symbols, timeframe, start_date, end_date
        )

    def _load_factor_set(self, set_name: str) -> Dict | None:
        """加载因子集合配置"""
        # 从registry文件读取
        if self.registry.registry_file.exists():
            import json

            with open(self.registry.registry_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            factor_sets = data.get("factor_sets", {})
            return factor_sets.get(set_name)

        return None

    def prewarm_factors(
        self,
        factor_ids: List[str],
        symbols: List[str],
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ):
        """预热因子缓存"""
        self.engine.prewarm_cache(factor_ids, symbols, timeframe, start_date, end_date)

    def get_cache_stats(self) -> Dict:
        """获取缓存统计"""
        return self.engine.get_cache_stats()
