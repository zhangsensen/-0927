#!/usr/bin/env python3
"""
因子文件对齐工具
确保多时间框架分析使用同批次生成的因子数据
"""
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class FactorFileAligner:
    """因子文件对齐器"""

    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)

    def find_aligned_factors(
        self, symbol: str, timeframes: List[str]
    ) -> Dict[str, Path]:
        """
        查找对齐的因子文件 - 确保所有时间框架使用同批次数据

        Args:
            symbol: 股票代码 (如: 0700.HK)
            timeframes: 时间框架列表

        Returns:
            Dict[timeframe, file_path]: 对齐的因子文件路径
        """
        clean_symbol = symbol.replace(".HK", "")

        # 1. 收集所有时间框架的因子文件
        timeframe_files = {}
        for tf in timeframes:
            tf_dir = self.data_root / tf
            if tf_dir.exists():
                files = list(tf_dir.glob(f"{clean_symbol}_{tf}_factors_*.parquet"))
                timeframe_files[tf] = files

        # 2. 按时间戳分组
        timestamp_groups = self._group_by_timestamp(timeframe_files)

        # 3. 选择最佳时间戳组
        best_group = self._select_best_timestamp_group(timestamp_groups, timeframes)

        if not best_group:
            raise ValueError(f"无法找到对齐的因子文件: {symbol} {timeframes}")

        return best_group

    def _group_by_timestamp(
        self, timeframe_files: Dict[str, List[Path]]
    ) -> Dict[str, Dict[str, Path]]:
        """按时间戳对因子文件进行分组"""
        timestamp_groups = {}

        for tf, files in timeframe_files.items():
            for file_path in files:
                # 从文件名提取时间戳
                timestamp = self._extract_timestamp(file_path.name)
                if timestamp:
                    if timestamp not in timestamp_groups:
                        timestamp_groups[timestamp] = {}
                    timestamp_groups[timestamp][tf] = file_path

        return timestamp_groups

    def _extract_timestamp(self, filename: str) -> Optional[str]:
        """从文件名提取时间戳"""
        # 匹配格式: factors_YYYYMMDD_HHMMSS.parquet
        match = re.search(r"factors_(\d{8}_\d{6})", filename)
        if match:
            return match.group(1)
        return None

    def _select_best_timestamp_group(
        self,
        timestamp_groups: Dict[str, Dict[str, Path]],
        required_timeframes: List[str],
    ) -> Optional[Dict[str, Path]]:
        """选择最佳时间戳组 - 包含所有需要时间框架且最新的组"""
        best_group = None
        best_timestamp = None

        for timestamp, group in timestamp_groups.items():
            # 检查是否包含所有需要的时间框架
            if all(tf in group for tf in required_timeframes):
                # 选择最新的时间戳
                if (
                    best_timestamp is None
                    or self._compare_timestamps(timestamp, best_timestamp) > 0
                ):
                    best_group = group
                    best_timestamp = timestamp

        return best_group

    def _compare_timestamps(self, ts1: str, ts2: str) -> int:
        """比较两个时间戳，返回 1 if ts1 > ts2, -1 if ts1 < ts2, 0 if equal"""
        # 格式: YYYYMMDD_HHMMSS
        dt1 = datetime.strptime(ts1, "%Y%m%d_%H%M%S")
        dt2 = datetime.strptime(ts2, "%Y%m%d_%H%M%S")

        if dt1 > dt2:
            return 1
        elif dt1 < dt2:
            return -1
        else:
            return 0

    def validate_time_alignment(
        self, factor_files: Dict[str, Path], tolerance_days: int = 7
    ) -> Tuple[bool, str]:
        """
        验证因子文件的时间对齐性

        Args:
            factor_files: 因子文件路径字典
            tolerance_days: 允许的时间偏差天数

        Returns:
            (is_aligned, message): 对齐状态和描述信息
        """
        time_ranges = {}

        # 读取所有因子文件的时间范围
        for tf, file_path in factor_files.items():
            try:
                factors = pd.read_parquet(file_path)
                if not factors.empty and isinstance(factors.index, pd.DatetimeIndex):
                    time_ranges[tf] = {
                        "start": factors.index.min(),
                        "end": factors.index.max(),
                        "length": len(factors),
                    }
            except Exception as e:
                return False, f"读取因子文件失败 {tf}: {str(e)}"

        if len(time_ranges) < 2:
            return True, "只有一个时间框架，无需验证对齐性"

        # 检查时间范围重叠
        all_starts = [r["start"] for r in time_ranges.to_numpy()()]
        all_ends = [r["end"] for r in time_ranges.to_numpy()()]

        common_start = max(all_starts)
        common_end = min(all_ends)

        if common_start >= common_end:
            return False, "时间框架之间没有重叠期间"

        # 计算重叠天数
        overlap_days = (common_end - common_start).days
        if overlap_days < tolerance_days:
            return False, f"重叠时间不足: {overlap_days}天 < {tolerance_days}天"

        return True, f"时间对齐良好，重叠期间: {overlap_days}天"


def find_aligned_factor_files(
    data_root: str, symbol: str, timeframes: List[str]
) -> Dict[str, Path]:
    """便捷函数：查找对齐的因子文件"""
    aligner = FactorFileAligner(Path(data_root))
    return aligner.find_aligned_factors(symbol, timeframes)


def validate_factor_alignment(
    data_root: str,
    symbol: str,
    timeframes: List[str],
    factor_files: Dict[str, Path],
    tolerance_days: int = 7,
) -> Tuple[bool, str]:
    """便捷函数：验证因子文件对齐性"""
    aligner = FactorFileAligner(Path(data_root))
    return aligner.validate_time_alignment(factor_files, tolerance_days)
