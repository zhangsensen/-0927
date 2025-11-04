#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据加载器 - 加载标准化因子和OHLCV数据
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class DataLoader:
    """数据加载器"""

    def __init__(self, factor_dir: str, ohlcv_dir: str, logger: logging.Logger = None):
        self.factor_dir = Path(factor_dir)
        self.ohlcv_dir = Path(ohlcv_dir)
        self.logger = logger or logging.getLogger(__name__)

    def find_latest_run(self, base_dir: Path) -> Path:
        """找到最新的运行结果目录"""
        all_runs = []
        for date_dir in base_dir.iterdir():
            if not date_dir.is_dir():
                continue
            for ts_dir in date_dir.iterdir():
                if (ts_dir / "metadata.json").exists():
                    all_runs.append(ts_dir)

        if not all_runs:
            raise FileNotFoundError(f"未找到有效的结果目录: {base_dir}")

        # 按时间戳排序
        all_runs.sort(key=lambda p: p.name, reverse=True)
        latest = all_runs[0]
        self.logger.info(f"使用最新运行: {latest}")
        return latest

    def load_factors(
        self, factor_names: List[str] = None
    ) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        """
        加载标准化因子

        Returns:
            factors_dict: {factor_name: DataFrame(index=date, columns=etf)}
            factor_list: 因子名称列表
        """
        self.logger.info(f"加载标准化因子: {self.factor_dir}")

        # 找到最新的factor_selection结果
        latest_run = self.find_latest_run(self.factor_dir)
        standardized_dir = latest_run / "standardized"

        if not standardized_dir.exists():
            raise FileNotFoundError(f"标准化因子目录不存在: {standardized_dir}")

        # 读取元数据获取因子列表
        with open(latest_run / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

        available_factors = metadata.get("standardized_factor_names", [])
        self.logger.info(f"可用因子数量: {len(available_factors)}")

        # 如果指定了因子列表，则筛选
        if factor_names:
            factors_to_load = [f for f in factor_names if f in available_factors]
            if not factors_to_load:
                raise ValueError(f"指定的因子都不存在: {factor_names}")
            self.logger.info(
                f"加载指定因子: {len(factors_to_load)}/{len(factor_names)}"
            )
        else:
            factors_to_load = available_factors
            self.logger.info(f"加载全部因子: {len(factors_to_load)}")

        # 加载因子数据
        factors_dict = {}
        for factor_name in factors_to_load:
            parquet_path = standardized_dir / f"{factor_name}.parquet"
            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
                # 确保index是日期，columns是ETF代码
                df.index = pd.to_datetime(df.index)
                factors_dict[factor_name] = df
            else:
                self.logger.warning(f"因子文件不存在: {parquet_path}")

        self.logger.info(f"成功加载因子: {len(factors_dict)}")
        return factors_dict, list(factors_dict.keys())

    def load_ohlcv(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        加载OHLCV数据

        Returns:
            close_df: 收盘价DataFrame
            volume_df: 成交量DataFrame
        """
        self.logger.info(f"加载OHLCV数据: {self.ohlcv_dir}")

        # 找到最新的cross_section结果
        latest_run = self.find_latest_run(self.ohlcv_dir)
        ohlcv_dir = latest_run / "ohlcv"

        if not ohlcv_dir.exists():
            raise FileNotFoundError(f"OHLCV目录不存在: {ohlcv_dir}")

        # 加载收盘价
        close_path = ohlcv_dir / "close.parquet"
        if not close_path.exists():
            raise FileNotFoundError(f"收盘价文件不存在: {close_path}")

        close_df = pd.read_parquet(close_path)
        close_df.index = pd.to_datetime(close_df.index)

        # 加载成交量（可选）
        volume_df = None
        volume_path = ohlcv_dir / "volume.parquet"
        if volume_path.exists():
            volume_df = pd.read_parquet(volume_path)
            volume_df.index = pd.to_datetime(volume_df.index)

        self.logger.info(f"收盘价形状: {close_df.shape}")
        if volume_df is not None:
            self.logger.info(f"成交量形状: {volume_df.shape}")

        return close_df, volume_df

    def align_data(
        self, factors_dict: Dict[str, pd.DataFrame], close_df: pd.DataFrame
    ) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """对齐因子和价格数据的日期和ETF"""
        self.logger.info("对齐因子和价格数据...")

        # 获取所有因子的日期并集
        all_dates = set()
        for factor_df in factors_dict.values():
            all_dates.update(factor_df.index)

        # 与价格日期求交集
        common_dates = sorted(all_dates.intersection(close_df.index))
        self.logger.info(f"共同日期数: {len(common_dates)}")

        # 获取所有因子的ETF并集
        all_etfs = set()
        for factor_df in factors_dict.values():
            all_etfs.update(factor_df.columns)

        # 与价格ETF求交集
        common_etfs = sorted(all_etfs.intersection(close_df.columns))
        self.logger.info(f"共同ETF数: {len(common_etfs)}")

        # 对齐所有数据
        aligned_factors = {}
        for factor_name, factor_df in factors_dict.items():
            aligned_factors[factor_name] = factor_df.reindex(
                index=common_dates, columns=common_etfs
            )

        aligned_close = close_df.reindex(index=common_dates, columns=common_etfs)

        self.logger.info(f"对齐后形状: {aligned_close.shape}")
        return aligned_factors, aligned_close

    def load_all(
        self, factor_names: List[str] = None
    ) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """
        一次性加载并对齐所有数据

        Returns:
            factors_dict: 对齐后的因子字典
            close_df: 对齐后的收盘价
        """
        # 加载因子
        factors_dict, factor_list = self.load_factors(factor_names)

        # 加载OHLCV
        close_df, volume_df = self.load_ohlcv()

        # 对齐数据
        aligned_factors, aligned_close = self.align_data(factors_dict, close_df)

        return aligned_factors, aligned_close
