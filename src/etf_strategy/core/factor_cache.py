"""
Factor Cache | 因子计算缓存层

避免 WFO/VEC/BT 三引擎重复计算因子，基于 mtime 自动失效。

作者: Sensen
日期: 2026-02-05
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FactorCache:
    """因子计算缓存 — 避免 WFO/VEC/BT 重复计算"""

    def __init__(self, cache_dir: Optional[Path] = None):
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent / ".cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_source_mtime(self) -> float:
        """获取因子库和标准化代码的最新 mtime"""
        source_files = [
            Path(__file__).parent / "precise_factor_library_v2.py",
            Path(__file__).parent / "cross_section_processor.py",
        ]
        mtimes = []
        for f in source_files:
            try:
                mtimes.append(f.stat().st_mtime)
            except OSError:
                # 文件不存在时返回0，强制缓存miss
                mtimes.append(0)
        return max(mtimes) if mtimes else 0

    def _get_data_mtime(self, data_dir: Path) -> float:
        """获取OHLCV数据目录的最新 mtime"""
        try:
            parquet_files = list(data_dir.glob("*.parquet"))
            if parquet_files:
                return max(f.stat().st_mtime for f in parquet_files)
            return data_dir.stat().st_mtime
        except OSError:
            logger.warning(f"无法获取数据目录 {data_dir} 的修改时间")
            return 0

    def _generate_cache_key(
        self,
        data_dir: Path,
        etf_codes: List[str],
        start_date: str,
        end_date: str,
    ) -> str:
        """生成缓存键 (包含数据mtime + 源码mtime)"""
        codes_str = "-".join(sorted(etf_codes))
        data_mtime = int(self._get_data_mtime(data_dir))
        source_mtime = int(self._get_source_mtime())
        key_str = f"factors_{codes_str}_{start_date}_{end_date}_{data_mtime}_{source_mtime}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get_or_compute(
        self,
        ohlcv: Dict[str, pd.DataFrame],
        config: dict,
        data_dir: Path,
    ) -> Dict:
        """
        获取或计算因子数据 (带缓存)

        Returns:
            {
                "std_factors": dict[str, pd.DataFrame],
                "factor_names": list[str],
                "factors_3d": np.ndarray,  # (T, N, F)
                "dates": pd.DatetimeIndex,
                "etf_codes": list[str],
            }
        """
        # 从实际 OHLCV 数据推导日期范围 (避免 config 中 training_end_date/end_date 歧义)
        close_df = ohlcv["close"]
        etf_codes = list(close_df.columns)
        actual_start = str(close_df.index[0].date())
        actual_end = str(close_df.index[-1].date())
        n_days = len(close_df)

        cache_key = self._generate_cache_key(data_dir, etf_codes, actual_start, actual_end)
        cache_file = self.cache_dir / f"factor_cache_{cache_key}.pkl"

        # 尝试读取缓存
        if cache_file.exists():
            try:
                t0 = time.time()
                with open(cache_file, "rb") as f:
                    cached = pickle.load(f)
                elapsed = time.time() - t0
                logger.info(
                    f"✅ Loading cached factors: {cache_file.name} ({elapsed:.1f}s)"
                )
                return cached
            except Exception as e:
                logger.warning(f"因子缓存读取失败，将重新计算: {e}")

        # 缓存未命中 — 计算因子
        logger.info("🔧 因子缓存未命中，开始计算...")
        t0 = time.time()

        from .cross_section_processor import CrossSectionProcessor
        from .precise_factor_library_v2 import PreciseFactorLibrary

        factor_lib = PreciseFactorLibrary()
        raw_factors_df = factor_lib.compute_all_factors(prices=ohlcv)

        # 提取因子字典
        factor_names_list = raw_factors_df.columns.get_level_values(0).unique().tolist()
        raw_factors = {fname: raw_factors_df[fname] for fname in factor_names_list}

        # 横截面标准化
        cross_section_cfg = config.get("cross_section", {})
        processor = CrossSectionProcessor(
            lower_percentile=cross_section_cfg.get("winsorize_lower", 0.025) * 100,
            upper_percentile=cross_section_cfg.get("winsorize_upper", 0.975) * 100,
            verbose=False,
        )
        std_factors = processor.process_all_factors(raw_factors)

        # 组织输出
        factor_names = sorted(std_factors.keys())
        first_factor = std_factors[factor_names[0]]
        dates = first_factor.index
        etf_cols = first_factor.columns.tolist()
        factors_3d = np.stack([std_factors[f].values for f in factor_names], axis=-1)

        result = {
            "std_factors": std_factors,
            "factor_names": factor_names,
            "factors_3d": factors_3d,
            "dates": dates,
            "etf_codes": etf_cols,
        }

        # 写入缓存
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
            size_mb = cache_file.stat().st_size / (1024 * 1024)
            elapsed = time.time() - t0
            logger.info(
                f"✅ 因子缓存已写入: {cache_file.name} ({size_mb:.1f}MB, {elapsed:.1f}s)"
            )
        except Exception as e:
            logger.warning(f"因子缓存写入失败 (不影响运行): {e}")

        return result
