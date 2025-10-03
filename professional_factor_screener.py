import json
import logging
import re
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil
import yaml
from scipy import stats

SYMBOL_PATTERN = re.compile(r"^[A-Za-z0-9]+\.[A-Z]{2,5}$")
TIMEFRAME_PATTERN = re.compile(r"^(?:\d+)(?:min|h|d|day)$|^daily$")

# 导入因子对齐工具
try:
    from factor_alignment_tool import FactorAlignmentTool
except ImportError:
    FactorAlignmentTool = None

class ScreeningConfig:
    """筛选配置"""
    def __init__(self,
                 symbols: Optional[List[str]] = None,
                 timeframes: Optional[List[str]] = None,
                 alpha_level: float = 0.05,
                 fdr_method: str = "fdr_bh",
                 data_root: Optional[str] = None,
                 log_root: Optional[str] = None,
                 cache_root: Optional[str] = None,
                 output_root: Optional[str] = None,
                 output_dir: Optional[str] = None):
        self.symbols = symbols if symbols is not None else []
        self.timeframes = timeframes if timeframes is not None else []
        self.alpha_level = alpha_level
        self.fdr_method = fdr_method
        self.data_root = data_root
        self.log_root = log_root
        self.cache_root = cache_root
        self.output_root = output_root
        self.output_dir = output_dir

class ProfessionalFactorScreener:
    """专业级因子筛选器 - 5维度筛选框架"""

    def __init__(self, data_root: Optional[str] = None, config: Optional[ScreeningConfig] = None):
        """初始化筛选器"""

        if config is None:
            config = ScreeningConfig()
        elif not isinstance(config, ScreeningConfig):
            raise TypeError("config 必须是 ScreeningConfig 类型")

        self.config = config

        self._validate_symbol_list(self.config.symbols)
        self._validate_timeframe_list(self.config.timeframes)

        if getattr(self.config, "data_root", None):
            base_data_root = Path(self.config.data_root)
        elif data_root:
            base_data_root = Path(data_root)
        else:
            base_data_root = Path("../因子输出")
        self.data_root = self._ensure_directory(base_data_root, "因子数据目录")

        log_root_cfg = Path(getattr(self.config, "log_root", "./logs/screening"))
        cache_root_cfg = Path(getattr(self.config, "cache_root", self.data_root / "cache"))

        output_root_cfg = Path(getattr(self.config, "output_root", "./因子筛选"))
        legacy_output_dir = getattr(self.config, "output_dir", None)
        if legacy_output_dir:
            output_root_cfg = Path(legacy_output_dir)
        self.screening_results_dir = self._ensure_directory(output_root_cfg, "筛选输出目录")

        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = None

        self.log_root = self._ensure_directory(log_root_cfg, "日志目录")
        self.cache_dir = self._ensure_directory(cache_root_cfg, "缓存目录")

        self.logger = self._setup_logger(self.session_timestamp)

        self.logger.info(
            f"显著性水平={self.config.alpha_level}, FDR方法={self.config.fdr_method}"
        )

    @staticmethod
    def _ensure_directory(path: Path, description: str) -> Path:
        resolved = path.expanduser()
        try:
            resolved.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # pragma: no cover - 文件系统异常
            raise OSError(f"无法创建{description}: {resolved}") from exc
        return resolved

    @staticmethod
    def _validate_symbol(symbol: str) -> None:
        if not isinstance(symbol, str):
            raise TypeError("symbol 必须为字符串")
        if not SYMBOL_PATTERN.fullmatch(symbol):
            raise ValueError(f"股票代码格式非法: {symbol}")

    def _validate_symbol_list(self, symbols: List[str]) -> None:
        for symbol in symbols:
            self._validate_symbol(symbol)

    @staticmethod
    def _validate_timeframe(timeframe: str) -> None:
        if not isinstance(timeframe, str) or not timeframe:
            raise ValueError("timeframe 不能为空")
        if not TIMEFRAME_PATTERN.fullmatch(timeframe):
            raise ValueError(f"时间框架格式非法: {timeframe}")

    def _validate_timeframe_list(self, timeframes: List[str]) -> None:
        for timeframe in timeframes:
            self._validate_timeframe(timeframe)

    @staticmethod
    def _ensure_dataframe(df: pd.DataFrame, description: str) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{description}必须为 pandas.DataFrame")
        if df.empty:
            raise ValueError(f"{description}不能为空")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(f"{description}索引必须为 pandas.DatetimeIndex")

    @staticmethod
    def _ensure_series(series: pd.Series, description: str) -> None:
        if not isinstance(series, pd.Series):
            raise TypeError(f"{description}必须为 pandas.Series")
        if series.empty:
            raise ValueError(f"{description}不能为空")
        if not isinstance(series.index, pd.DatetimeIndex):
            raise TypeError(f"{description}索引必须为 pandas.DatetimeIndex")

    def _setup_logger(self, session_timestamp: str) -> logging.Logger:
        """设置日志记录器"""
        log_file = self.log_root / f"screening_{session_timestamp}.log"
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _load_config(self) -> dict:
        """加载配置文件"""
        config_file = self.data_root / "config.yaml"
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_file}")
        with open(config_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_factor_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """加载因子数据"""
        cache_file = self.cache_dir / f"{symbol}_{timeframe}.pkl"
        if cache_file.exists():
            return pd.read_pickle(cache_file)

        data_file = self.data_root / f"{symbol}_{timeframe}.pkl"
        if not data_file.exists():
            raise FileNotFoundError(f"因子数据文件不存在: {data_file}")

        df = pd.read_pickle(data_file)
        df.set_index("datetime", inplace=True)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        df.to_pickle(cache_file)
        return df

    def _align_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """对齐因子"""
        if FactorAlignmentTool is None:
            self.logger.warning("因子对齐工具未安装，跳过因子对齐。")
            return df

        self.logger.info("开始因子对齐...")
        aligned_df = FactorAlignmentTool.align_factors(df)
        self.logger.info("因子对齐完成。")
        return aligned_df

    def _calculate_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算统计量"""
        self.logger.info("开始计算统计量...")
        # 假设因子列名为 "factor"
        factor_col = "factor"
        if factor_col not in df.columns:
            raise ValueError(f"因子数据中未找到列名 '{factor_col}'")

        # 计算均值和标准差
        mean_factor = df[factor_col].mean()
        std_factor = df[factor_col].std()

        # 计算 t 统计量
        t_stat, p_value = stats.ttest_1samp(df[factor_col], mean_factor)

        # 计算 FDR 校正的 p 值
        if self.config.fdr_method == "fdr_bh":
            _, _, _, p_adj = stats.fdr_correction(p_value)
        elif self.config.fdr_method == "fdr_by":
            _, _, _, p_adj = stats.fdr_correction(p_value)
        else:
            raise ValueError(f"不支持的 FDR 方法: {self.config.fdr_method}")

        # 创建结果数据框
        results = pd.DataFrame({
            "mean": mean_factor,
            "std": std_factor,
            "t_stat": t_stat,
            "p_value": p_value,
            "p_adj": p_adj
        }, index=[0])
        self.logger.info("统计量计算完成。")
        return results

    def _filter_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """筛选因子"""
        self.logger.info("开始筛选因子...")
        # 假设 p_adj 列名为 "p_adj"
        p_adj_col = "p_adj"
        if p_adj_col not in df.columns:
            raise ValueError(f"统计量数据中未找到列名 '{p_adj_col}'")

        # 根据显著性水平筛选
        df_filtered = df[df[p_adj_col] < self.config.alpha_level]
        self.logger.info(f"筛选出 {len(df_filtered)} 个显著因子。")
        return df_filtered

    def _save_results(self, results: pd.DataFrame, symbol: str, timeframe: str):
        """保存筛选结果"""
        results_file = self.screening_results_dir / f"{symbol}_{timeframe}_screening_results.csv"
        results.to_csv(results_file)
        self.logger.info(f"筛选结果已保存至: {results_file}")

    def run(self):
        """运行筛选流程"""
        self.logger.info("因子筛选流程开始...")
        for symbol in self.config.symbols:
            self.logger.info(f"处理股票: {symbol}")
            for timeframe in self.config.timeframes:
                self.logger.info(f"处理时间框架: {timeframe}")
                try:
                    df = self._load_factor_data(symbol, timeframe)
                    df = self._align_factors(df)
                    results = self._calculate_statistics(df)
                    results = self._filter_factors(results)
                    self._save_results(results, symbol, timeframe)
                except Exception as e:
                    self.logger.error(f"处理股票 {symbol} 时间框架 {timeframe} 时发生错误: {e}")
        self.logger.info("因子筛选流程结束。")

if __name__ == "__main__":
    # 示例用法
    config = ScreeningConfig(
        symbols=["000001.SZ", "600000.SH"],
        timeframes=["1min", "5min", "15min", "30min", "60min", "daily"]
    )
    screener = ProfessionalFactorScreener(config=config)
    screener.run()
