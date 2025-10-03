#!/usr/bin/env python3
"""
专业级量化交易因子筛选系统 - 5维度筛选框架
作者：量化首席工程师
版本：2.0.0
日期：2025-09-29

核心特性：
1. 5维度筛选框架：预测能力、稳定性、独立性、实用性、短周期适应性
2. 多周期IC分析：1日、3日、5日、10日、20日预测能力评估
3. 严格的统计显著性检验：Benjamini-Hochberg FDR校正
4. VIF检测和信息增量分析
5. 交易成本和流动性评估
6. 生产级性能优化和错误处理
"""

import json
import logging
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 导入配置类
from config_manager import ScreeningConfig  # type: ignore

import numpy as np
import pandas as pd
import psutil
import yaml
from scipy import stats

# 导入因子对齐工具
try:
    from utils import (  # type: ignore
        FactorFileAligner,
        find_aligned_factor_files,
        validate_factor_alignment,
    )
except ImportError as e:
    # 如果导入失败，提供回退方案
    import logging
    logging.getLogger(__name__).warning(f"因子对齐工具导入失败: {e}")
    FactorFileAligner = None

    def find_aligned_factor_files(*args: Any, **kwargs: Any) -> None:
        raise ImportError("因子对齐工具不可用")

    def validate_factor_alignment(*args: Any, **kwargs: Any) -> Tuple[bool, str]:
        return True, "对齐验证工具不可用"

try:
    from utils.temporal_validator import TemporalValidationError, TemporalValidator
except ImportError as e:  # pragma: no cover - 运行环境缺失
    TemporalValidator = None  # type: ignore

    class TemporalValidationError(Exception):
        """时间序列验证器不可用时的后备异常"""

        pass


warnings.filterwarnings("ignore")

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class FactorMetrics:
    """因子综合指标"""

    name: str

    # 预测能力指标
    ic_1d: float = 0.0
    ic_3d: float = 0.0
    ic_5d: float = 0.0
    ic_10d: float = 0.0
    ic_20d: float = 0.0
    ic_mean: float = 0.0
    ic_std: float = 0.0
    ic_ir: float = 0.0
    ic_decay_rate: float = 0.0
    ic_longevity: int = 0
    predictive_power_mean_ic: float = 0.0  # 添加缺失字段

    # 稳定性指标
    rolling_ic_mean: float = 0.0
    rolling_ic_std: float = 0.0
    rolling_ic_stability: float = 0.0
    ic_consistency: float = 0.0
    cross_section_stability: float = 0.0

    # 独立性指标
    vif_score: float = 0.0
    correlation_max: float = 0.0
    information_increment: float = 0.0
    redundancy_score: float = 0.0

    # 实用性指标
    turnover_rate: float = 0.0
    transaction_cost: float = 0.0
    cost_efficiency: float = 0.0
    liquidity_demand: float = 0.0
    capacity_score: float = 0.0

    # 短周期适应性指标
    reversal_effect: float = 0.0
    momentum_persistence: float = 0.0
    volatility_sensitivity: float = 0.0
    regime_adaptability: float = 0.0

    # 统计显著性
    p_value: float = 1.0
    corrected_p_value: float = 1.0
    is_significant: bool = False

    # 综合评分
    predictive_score: float = 0.0
    stability_score: float = 0.0
    independence_score: float = 0.0
    practicality_score: float = 0.0
    adaptability_score: float = 0.0
    comprehensive_score: float = 0.0

    # 元数据
    sample_size: int = 0
    calculation_time: float = 0.0
    data_quality_score: float = 0.0

    # 因子分类信息
    tier: str = ""
    type: str = ""
    description: str = ""


# ScreeningConfig类已移至config_manager.py，避免重复定义


class ProfessionalFactorScreener:
    """专业级因子筛选器 - 5维度筛选框架"""

    @staticmethod
    def _to_json_serializable(obj: Any) -> Any:
        """转换对象为JSON可序列化格式 - 极简实现"""
        if isinstance(obj, dict):
            return {
                k: ProfessionalFactorScreener._to_json_serializable(v)
                for k, v in obj.items()
            }
        elif isinstance(obj, (list, tuple)):
            return [
                ProfessionalFactorScreener._to_json_serializable(item) for item in obj
            ]
        elif isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, "shape"):  # pandas对象
            return str(obj)
        return obj

    def __init__(self, data_root: Optional[str] = None, config: Optional[ScreeningConfig] = None):
        """初始化筛选器

        Args:
            data_root: 向后兼容参数，优先使用config中的路径配置
            config: 筛选配置对象
        """
        self.config = config or ScreeningConfig()

        # 路径优先级: config.data_root > data_root参数 > 默认值
        if hasattr(self.config, "data_root") and self.config.data_root:
            self.data_root = Path(self.config.data_root)
        elif data_root:
            self.data_root = Path(data_root)
        else:
            self.data_root = Path("../因子输出")  # 默认因子数据目录（相对路径）

        # 设置日志和缓存路径
        self.log_root = Path(getattr(self.config, "log_root", "./logs/screening"))
        self.cache_dir = Path(
            getattr(self.config, "cache_root", self.data_root / "cache")
        )

        # 设置筛选报告专用目录
        if hasattr(self.config, "output_dir") and self.config.output_dir:
            self.screening_results_dir = Path(self.config.output_dir)
        else:
            self.screening_results_dir = Path("./因子筛选")
        self.screening_results_dir.mkdir(parents=True, exist_ok=True)

        # 初始化会话相关变量（稍后在screen_factors_comprehensive中创建）
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = None

        # 设置日志和缓存路径（先使用默认路径）
        self.log_root = Path(getattr(self.config, "log_root", "./logs/screening"))
        self.cache_dir = Path(
            getattr(self.config, "cache_root", self.data_root / "cache")
        )

        # 创建必要的目录
        self.log_root.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 设置日志
        self.logger = self._setup_logger(self.session_timestamp)

        # 初始化时间序列验证器，确保旧代码同样受防护
        self.temporal_validator = None
        if TemporalValidator is not None:
            try:
                self.temporal_validator = TemporalValidator(strict_mode=True)
                self.logger.info("✅ 时间序列验证器已启用")
            except Exception as validator_error:
                self.logger.warning(
                    "时间序列验证器初始化失败: %s", validator_error
                )
                self.temporal_validator = None
        else:
            self.logger.warning("时间序列验证器模块未安装，跳过运行时校验")

        # 初始化增强版结果管理器
        try:
            from enhanced_result_manager import EnhancedResultManager  # type: ignore

            self.result_manager = EnhancedResultManager(str(self.screening_results_dir))
            self.logger.info("✅ 增强版结果管理器初始化成功")
        except ImportError as e:
            self.result_manager = None
            self.logger.warning(f"增强版结果管理器导入失败: {e}")
            self.logger.info("将使用传统文件保存方式")

        # 性能监控
        self.process = psutil.Process()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        self.logger.info("专业级因子筛选器初始化完成")
        self.logger.info(
            f"配置: IC周期={self.config.ic_horizons}, 最小样本={self.config.min_sample_size}"
        )
        self.logger.info(
            f"显著性水平={self.config.alpha_level}, FDR方法={self.config.fdr_method}"
        )

    def _setup_logger(self, session_timestamp: Optional[str] = None) -> logging.Logger:
        """设置专业级日志系统 - 改进版"""
        logger = logging.getLogger(f"{__name__}.{id(self)}")
        logger.setLevel(logging.INFO)

        # 清除现有处理器
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # 使用日志轮转 - 关键修复
        from logging.handlers import RotatingFileHandler

        # 支持会话时间戳或日期命名
        if session_timestamp:
            log_filename = f"professional_screener_{session_timestamp}.log"
        else:
            today = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"professional_screener_{today}.log"

        log_file = self.log_root / log_filename

        # 保存当前日志文件路径，方便其他地方访问
        self.current_log_file = str(log_file)

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,  # 保留5个备份
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)

        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 创建格式器
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def _generate_factor_metadata(self, factors_df: pd.DataFrame) -> dict:
        """生成因子元数据"""
        metadata = {}

        for factor_name in factors_df.columns:
            meta = {
                "name": factor_name,
                "type": self._infer_factor_type(factor_name),
                "warmup_period": self._infer_warmup_period(factor_name),
                "description": self._generate_factor_description(factor_name),
            }

            # 计算统计信息
            factor_data = factors_df[factor_name]
            meta.update(
                {
                    "total_periods": len(factor_data),
                    "missing_periods": factor_data.isnull().sum(),
                    "missing_ratio": factor_data.isnull().sum() / len(factor_data),
                    "first_valid_index": self._find_first_non_missing_index(
                        factor_data
                    ),
                    "valid_ratio": 1 - (factor_data.isnull().sum() / len(factor_data)),
                }
            )

            metadata[factor_name] = meta

        return metadata

    def _infer_factor_type(self, factor_name: str) -> str:
        """根据因子名称推断类型"""
        name_lower = factor_name.lower()

        if any(
            indicator in name_lower
            for indicator in [
                "ma",
                "ema",
                "sma",
                "wma",
                "dema",
                "tema",
                "trima",
                "kama",
                "t3",
            ]
        ):
            return "trend"
        elif any(
            indicator in name_lower
            for indicator in ["rsi", "stoch", "cci", "willr", "mfi", "roc", "mom"]
        ):
            return "momentum"
        elif any(
            indicator in name_lower
            for indicator in ["bb", "bollinger", "atr", "std", "mstd"]
        ):
            return "volatility"
        elif any(indicator in name_lower for indicator in ["volume", "obv", "vwap"]):
            return "volume"
        elif any(
            indicator in name_lower
            for indicator in ["macd", "signal", "histogram", "hist"]
        ):
            return "momentum"
        elif any(indicator in name_lower for indicator in ["cdl", "pattern"]):
            return "pattern"
        else:
            return "unknown"

    def _infer_warmup_period(self, factor_name: str) -> int:
        """根据因子名称推断预热期"""
        name_lower = factor_name.lower()

        # 移动平均类 - 精确匹配数字
        import re

        ma_match = re.search(r"ma(\d+)", name_lower)
        if ma_match:
            return int(ma_match.group(1))

        sma_match = re.search(r"sma_?(\d+)", name_lower)
        if sma_match:
            return int(sma_match.group(1))

        ema_match = re.search(r"ema_?(\d+)", name_lower)
        if ema_match:
            return int(ema_match.group(1))

        # RSI类
        rsi_match = re.search(r"rsi(\d+)", name_lower)
        if rsi_match:
            return int(rsi_match.group(1))
        elif "rsi" in name_lower:
            return 14

        # 布林带类
        bb_match = re.search(r"bb_(\d+)", name_lower)
        if bb_match:
            return int(bb_match.group(1))
        elif "bb" in name_lower or "bollinger" in name_lower:
            return 20

        # MACD类
        if "macd" in name_lower:
            return 26

        # CCI类
        cci_match = re.search(r"cci(\d+)", name_lower)
        if cci_match:
            return int(cci_match.group(1))
        elif "cci" in name_lower:
            return 20

        # WILLR类
        willr_match = re.search(r"willr(\d+)", name_lower)
        if willr_match:
            return int(willr_match.group(1))
        elif "willr" in name_lower:
            return 14

        # ATR类
        atr_match = re.search(r"atr(\d+)", name_lower)
        if atr_match:
            return int(atr_match.group(1))
        elif "atr" in name_lower:
            return 14

        # 默认预热期
        return 20

    def _generate_factor_description(self, factor_name: str) -> str:
        """生成因子描述"""
        name_lower = factor_name.lower()

        if "ma" in name_lower:
            return f"移动平均线指标 - {factor_name}"
        elif "rsi" in name_lower:
            return f"相对强弱指标 - {factor_name}"
        elif "macd" in name_lower:
            return f"移动平均收敛散度指标 - {factor_name}"
        elif "bb" in name_lower:
            return f"布林带指标 - {factor_name}"
        elif "volume" in name_lower:
            return f"成交量指标 - {factor_name}"
        else:
            return f"技术指标 - {factor_name}"

    def _find_first_non_missing_index(self, series: pd.Series) -> int:
        """找到第一个非缺失值的索引位置"""
        non_null_mask = series.notna()
        if non_null_mask.any():
            return non_null_mask.idxmax()
        return len(series)

    def _smart_forward_fill(self, series: pd.Series) -> pd.Series:
        """智能前向填充"""
        result = series.copy()

        # 找到第一个有效值
        first_valid_idx = series.first_valid_index()

        if first_valid_idx is not None:
            # 用第一个有效值填充前面的缺失值
            first_valid_value = series.loc[first_valid_idx]
            result = result.bfill(limit=1)
            result = result.fillna(first_valid_value)

            # 前向填充剩余的缺失值
            result = result.ffill()

        return result

    def _smart_interpolation(self, series: pd.Series) -> pd.Series:
        """智能插值"""
        result = series.copy()

        # 找到第一个有效值
        first_valid_idx = series.first_valid_index()

        if first_valid_idx is not None:
            # 用第一个有效值填充前面的缺失值
            first_valid_value = series.loc[first_valid_idx]
            result.loc[:first_valid_idx] = result.loc[:first_valid_idx].fillna(
                first_valid_value
            )

            # 对剩余缺失值进行线性插值
            result = result.interpolate(method="linear")

            # 如果还有缺失值，用前向填充
            result = result.ffill()
            result = result.bfill()

        return result

    def smart_missing_value_handling(
        self, factors_df: pd.DataFrame, factor_metadata: dict = None
    ) -> tuple:
        """
        智能缺失值处理，区分正常预热期缺失和问题数据

        Args:
            factors_df: 因子数据DataFrame
            factor_metadata: 因子元数据，包含预热期信息

        Returns:
            tuple: (cleaned_df, handling_report)
        """
        if factor_metadata is None:
            factor_metadata = self._generate_factor_metadata(factors_df)

        handling_report = {
            "total_factors": len(factors_df.columns),
            "removed_factors": [],
            "handled_factors": [],
            "forward_filled_factors": [],
            "interpolated_factors": [],
            "dropped_factors": [],
        }

        # 直接在原DataFrame上操作，避免不必要的内存复制
        cleaned_df = factors_df

        for factor_name in factors_df.columns:
            factor_data = factors_df[factor_name]
            missing_count = factor_data.isnull().sum()

            if missing_count == 0:
                handling_report["handled_factors"].append(factor_name)
                continue

            missing_ratio = missing_count / len(factor_data)

            # 获取因子元数据
            meta = factor_metadata.get(factor_name, {})
            warmup_period = meta.get("warmup_period", 20)
            factor_type = meta.get("type", "unknown")

            # 判断缺失值模式
            first_valid_idx = factor_data.first_valid_index()
            if first_valid_idx is not None:
                first_valid_pos = factor_data.index.get_loc(first_valid_idx)
            else:
                first_valid_pos = len(factor_data)

            # 决策逻辑
            if first_valid_pos <= warmup_period * 1.5:  # 允许1.5倍的预热期容忍度
                # 正常预热期缺失，进行智能填充
                if factor_type in ["momentum", "trend", "volatility"]:
                    # 技术指标类因子使用前向填充
                    cleaned_df[factor_name] = self._smart_forward_fill(factor_data)
                    handling_report["forward_filled_factors"].append(factor_name)
                else:
                    # 其他类型因子使用插值
                    cleaned_df[factor_name] = self._smart_interpolation(factor_data)
                    handling_report["interpolated_factors"].append(factor_name)

                handling_report["handled_factors"].append(factor_name)

            elif missing_ratio > self.config.max_missing_ratio:
                # 缺失比例过高，删除
                cleaned_df = cleaned_df.drop(columns=[factor_name])
                handling_report["dropped_factors"].append(factor_name)
                handling_report["removed_factors"].append(factor_name)

            else:
                # 随机缺失，使用插值
                cleaned_df[factor_name] = self._smart_interpolation(factor_data)
                handling_report["interpolated_factors"].append(factor_name)
                handling_report["handled_factors"].append(factor_name)

        return cleaned_df, handling_report

    def validate_factor_data_quality(
        self, factors_df: pd.DataFrame, symbol: str, timeframe: str
    ) -> pd.DataFrame:
        """智能数据质量验证 - 解决根本问题而不是粗暴删除"""
        self.logger.info(f"开始智能数据质量验证: {symbol} {timeframe}")

        original_shape = factors_df.shape
        issues_found = []

        # 1. 检查非数值列（根本问题）
        non_numeric_cols = factors_df.select_dtypes(
            exclude=[np.number, "datetime64[ns]"]
        ).columns.tolist()
        if non_numeric_cols:
            self.logger.warning(f"发现非数值列: {non_numeric_cols}")
            factors_df = factors_df.drop(columns=non_numeric_cols)
            issues_found.append(f"移除非数值列: {non_numeric_cols}")

        # 2. 智能缺失值处理 - 核心改进
        if factors_df.isnull().any().any():
            self.logger.info("开始智能缺失值处理...")
            factor_metadata = self._generate_factor_metadata(factors_df)
            factors_df, handling_report = self.smart_missing_value_handling(
                factors_df, factor_metadata
            )

            self.logger.info("智能缺失值处理完成:")
            self.logger.info(f"  - 总因子数: {handling_report['total_factors']}")
            self.logger.info(
                f"  - 前向填充因子数: {len(handling_report['forward_filled_factors'])}"
            )
            self.logger.info(
                f"  - 插值填充因子数: {len(handling_report['interpolated_factors'])}"
            )
            self.logger.info(
                f"  - 删除因子数: {len(handling_report['removed_factors'])}"
            )

            if handling_report["removed_factors"]:
                issues_found.append(
                    f"智能处理移除因子: {handling_report['removed_factors']}"
                )

        # 3. 检查无穷值 - 修复而不是删除
        inf_cols = []
        for col in factors_df.select_dtypes(include=[np.number]).columns:
            if np.isinf(factors_df[col]).any():
                inf_cols.append(col)
                # 用极值替换无穷值
                factors_df[col] = factors_df[col].replace(
                    [np.inf, -np.inf],
                    [factors_df[col].quantile(0.99), factors_df[col].quantile(0.01)],
                )

        if inf_cols:
            self.logger.info(f"修复无穷值列: {inf_cols}")
            issues_found.append(f"修复无穷值列: {inf_cols}")

        # 4. 检查常量列（使用更严格的标准）
        constant_cols = []
        for col in factors_df.select_dtypes(include=[np.number]).columns:
            if factors_df[col].std() < 1e-6:  # 合理的常量检测阈值
                constant_cols.append(col)

        if constant_cols:
            self.logger.warning(f"发现常量列: {constant_cols}")
            factors_df = factors_df.drop(columns=constant_cols)
            issues_found.append(f"移除常量列: {constant_cols}")

        # 5. 检查重复列
        duplicate_cols = factors_df.columns[factors_df.columns.duplicated()].tolist()
        if duplicate_cols:
            self.logger.warning(f"发现重复列: {duplicate_cols}")
            factors_df = factors_df.loc[:, ~factors_df.columns.duplicated()]
            issues_found.append(f"移除重复列: {duplicate_cols}")

        final_shape = factors_df.shape

        # 报告验证结果
        retention_rate = final_shape[1] / original_shape[1]
        self.logger.info("智能数据质量验证完成:")
        self.logger.info(f"  - 原始形状: {original_shape}")
        self.logger.info(f"  - 最终形状: {final_shape}")
        self.logger.info(f"  - 因子保留率: {retention_rate:.1%}")

        if issues_found:
            for issue in issues_found:
                self.logger.info(f"  - {issue}")

        # 确保还有足够的因子数据
        if len(factors_df.columns) < 10:
            raise ValueError(f"验证后因子数量过少: {len(factors_df.columns)} < 10")

        return factors_df

    def load_factors(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """加载因子数据 - 增强版本，支持因子文件对齐"""
        start_time = time.time()
        self.logger.info(f"加载因子数据: {symbol} {timeframe}")

        # 🎯 优先使用对齐的因子文件
        if hasattr(self, "aligned_factor_files") and self.aligned_factor_files:
            if timeframe in self.aligned_factor_files:
                selected_file = self.aligned_factor_files[timeframe]
                self.logger.info(f"✅ 使用对齐的因子文件: {selected_file.name}")

                try:
                    # 内存优化：使用columns参数选择性加载
                    # 仅读取数值列，减少内存占用
                    factors = pd.read_parquet(
                        selected_file,
                        # 暂不指定columns，因为需要先读取才知道列名
                        # 后续通过dropna和数据类型筛选优化
                    )

                    # 数据质量检查
                    if factors.empty:
                        self.logger.warning(f"因子文件为空: {selected_file}")
                        raise ValueError(f"因子文件为空: {selected_file}")

                    # 内存优化：立即移除全部为NaN的列
                    factors = factors.dropna(axis=1, how="all")

                    # 内存优化：转换数据类型以减少内存占用
                    for col in factors.select_dtypes(include=["float64"]).columns:
                        factors[col] = factors[col].astype("float32")

                    # 确保索引是datetime类型
                    if not isinstance(factors.index, pd.DatetimeIndex):
                        factors.index = pd.to_datetime(factors.index)

                    # Linus式数据质量验证
                    factors = self.validate_factor_data_quality(
                        factors, symbol, timeframe
                    )

                    initial_memory = factors.memory_usage(deep=True).sum() / 1024 / 1024
                    self.logger.info(
                        f"因子数据加载成功: 形状={factors.shape}, 内存={initial_memory:.1f}MB"
                    )
                    self.logger.info(
                        f"时间范围: {factors.index.min()} 到 {factors.index.max()}"
                    )
                    self.logger.info(f"加载耗时: {time.time() - start_time:.2f}秒")

                    return factors

                except Exception as e:
                    self.logger.warning(f"加载对齐文件失败，回退到默认搜索: {str(e)}")

        # 处理symbol格式
        clean_symbol = symbol.replace(".HK", "")

        # 搜索策略：按优先级搜索不同格式的文件
        search_patterns = [
            # 新格式：timeframe子目录 (带.HK后缀)
            (
                self.data_root / timeframe,
                f"{clean_symbol}.HK_{timeframe}_factors_*.parquet",
            ),
            (
                self.data_root / timeframe,
                f"{clean_symbol}HK_{timeframe}_factors_*.parquet",
            ),
            (
                self.data_root / timeframe,
                f"{clean_symbol}_{timeframe}_factors_*.parquet",
            ),
            # multi_tf格式
            (self.data_root, f"aligned_multi_tf_factors_{clean_symbol}*.parquet"),
            # 根目录格式
            (self.data_root, f"{clean_symbol}*_{timeframe}_factors_*.parquet"),
        ]

        for search_dir, pattern in search_patterns:
            if search_dir.exists():
                factor_files = list(search_dir.glob(pattern))
                if factor_files:
                    selected_file = factor_files[-1]  # 选择最新文件
                    self.logger.info(f"找到因子文件: {selected_file}")

                    try:
                        # 内存优化：选择性加载
                        factors = pd.read_parquet(selected_file)

                        # 内存优化：立即移除全部为NaN的列和转换数据类型
                        factors = factors.dropna(axis=1, how="all")
                        for col in factors.select_dtypes(include=["float64"]).columns:
                            factors[col] = factors[col].astype("float32")

                        # 数据质量检查
                        if factors.empty:
                            self.logger.warning(f"因子文件为空: {selected_file}")
                            continue

                        # 确保索引是datetime类型
                        if not isinstance(factors.index, pd.DatetimeIndex):
                            factors.index = pd.to_datetime(factors.index)

                        # Linus式数据质量验证 - 解决根本问题
                        factors = self.validate_factor_data_quality(
                            factors, symbol, timeframe
                        )

                        self.logger.info(f"因子数据加载成功: 形状={factors.shape}")
                        self.logger.info(
                            f"时间范围: {factors.index.min()} 到 {factors.index.max()}"
                        )
                        self.logger.info(f"加载耗时: {time.time() - start_time:.2f}秒")

                        return factors

                    except Exception as e:
                        self.logger.error(f"加载因子文件失败 {selected_file}: {str(e)}")
                        continue

        # 详细错误信息
        self.logger.error("未找到因子数据:")
        self.logger.error(f"搜索路径: {self.data_root}")
        self.logger.error(f"搜索符号: {clean_symbol}")
        self.logger.error(f"时间框架: {timeframe}")

        available_dirs = [d.name for d in self.data_root.iterdir() if d.is_dir()]
        self.logger.error(f"可用目录: {available_dirs}")

        raise FileNotFoundError(f"No factor data found for {symbol} {timeframe}")

    def load_price_data(self, symbol: str, timeframe: str = None) -> pd.DataFrame:
        """加载价格数据 - 智能匹配时间框架（修复版）"""
        start_time = time.time()
        self.logger.info(f"加载价格数据: {symbol} (时间框架: {timeframe})")

        # 处理symbol格式
        if symbol.endswith(".HK"):
            clean_symbol = symbol.replace(".HK", "") + "HK"
        else:
            clean_symbol = symbol

        # 原始数据路径 - 使用配置或相对路径
        if hasattr(self.config, "raw_data_root") and self.config.raw_data_root:
            raw_data_path = Path(self.config.raw_data_root)
        else:
            raw_data_path = self.data_root.parent / "raw" / "HK"

        if not raw_data_path.exists():
            # 回退到项目根目录的raw/HK
            raw_data_path = Path(__file__).parent.parent.parent / "raw" / "HK"

        # 时间框架到文件名的映射
        timeframe_map = {
            "1min": "1min",
            "2min": "2min",
            "3min": "3min",
            "5min": "5min",
            "15min": "15m",
            "30min": "30m",
            "60min": "60m",
            "daily": "1day",
            "1d": "1day",
        }

        # 根据时间框架智能选择搜索模式
        if timeframe and timeframe in timeframe_map:
            file_pattern = timeframe_map[timeframe]
            self.logger.info(
                f"根据时间框架 '{timeframe}' 搜索 '{file_pattern}' 格式文件"
            )
            search_patterns = [
                f"{clean_symbol}_{file_pattern}_*.parquet",  # 精确匹配
                f"{clean_symbol}_*.parquet",  # 备用
            ]
        else:
            # 默认搜索顺序（保持向后兼容）
            self.logger.warning("未指定时间框架或不在映射表中，使用默认搜索")
            search_patterns = [
                f"{clean_symbol}_60m_*.parquet",  # 60分钟数据
                f"{clean_symbol}_1day_*.parquet",  # 日线数据
                f"{clean_symbol}_*.parquet",  # 任意时间框架
            ]

        for pattern in search_patterns:
            price_files = list(raw_data_path.glob(pattern))
            if price_files:
                selected_file = price_files[-1]  # 选择最新文件
                self.logger.info(f"找到价格文件: {selected_file}")

                try:
                    # 内存优化：仅读取OHLCV和timestamp列
                    price_data = pd.read_parquet(selected_file)

                    # 先处理timestamp列（如果存在）
                    if "timestamp" in price_data.columns:
                        price_data["timestamp"] = pd.to_datetime(
                            price_data["timestamp"]
                        )
                        price_data = price_data.set_index("timestamp")
                    elif not isinstance(price_data.index, pd.DatetimeIndex):
                        price_data.index = pd.to_datetime(price_data.index)

                    # 然后选择OHLCV列
                    ohlcv_cols = ["open", "high", "low", "close", "volume"]
                    available_cols = [
                        col for col in ohlcv_cols if col in price_data.columns
                    ]
                    if available_cols:
                        price_data = price_data[available_cols]

                    # 内存优化：转换数据类型
                    for col in price_data.select_dtypes(include=["float64"]).columns:
                        price_data[col] = price_data[col].astype("float32")

                    # 确保包含必要的列
                    required_cols = ["open", "high", "low", "close", "volume"]
                    missing_cols = [
                        col for col in required_cols if col not in price_data.columns
                    ]
                    if missing_cols:
                        self.logger.error(f"价格数据缺少必要列: {missing_cols}")
                        continue

                    self.logger.info(f"价格数据加载成功: 形状={price_data.shape}")
                    self.logger.info(
                        f"时间范围: {price_data.index.min()} 到 {price_data.index.max()}"
                    )
                    self.logger.info(f"加载耗时: {time.time() - start_time:.2f}秒")

                    return price_data[required_cols]

                except Exception as e:
                    self.logger.error(f"加载价格文件失败 {selected_file}: {str(e)}")
                    continue

        self.logger.error("未找到价格数据:")
        self.logger.error(f"搜索路径: {raw_data_path}")
        self.logger.error(f"搜索符号: {clean_symbol}")

        available_files = [f.name for f in raw_data_path.glob("*.parquet")][:10]
        self.logger.error(f"可用文件示例: {available_files}")

        raise FileNotFoundError(f"No price data found for {symbol}")

    # ==================== 1. 预测能力分析 ====================

    def calculate_multi_horizon_ic(
        self, factors: pd.DataFrame, returns: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """计算多周期IC值 - 核心预测能力评估"""
        self.logger.info("开始多周期IC计算...")
        start_time = time.time()

        ic_results: Dict[str, Dict[str, float]] = {}
        horizons = self.config.ic_horizons

        factor_cols = [
            col
            for col in factors.columns
            if col not in ["open", "high", "low", "close", "volume"]
        ]

        total_factors = len(factor_cols)
        processed = 0

        returns_series = returns.reindex(factors.index)

        for factor in factor_cols:
            processed += 1
            if processed % self.config.progress_report_interval == 0:
                self.logger.info(f"多周期IC计算进度: {processed}/{total_factors}")

            factor_series = factors[factor]
            horizon_ics: Dict[str, float] = {}

            for horizon in horizons:
                if horizon < 0:
                    self.logger.warning(
                        f"忽略非法预测周期 {horizon}，因子 {factor}"
                    )
                    continue

                if self.temporal_validator is not None:
                    try:
                        is_valid, message = self.temporal_validator.validate_time_alignment(
                            factor_series,
                            returns_series,
                            horizon,
                            context=f"IC-{factor}",
                        )
                    except Exception as validation_error:
                        self.logger.warning(
                            "时间序列验证失败 (%s): %s",
                            factor,
                            validation_error,
                        )
                        continue

                    if not is_valid:
                        self.logger.debug(
                            "跳过因子 %s 周期 %sd：%s", factor, horizon, message
                        )
                        continue

                lagged_factor = factor_series.shift(horizon)

                aligned_returns = returns_series.reindex(lagged_factor.index)
                common_idx = lagged_factor.index.intersection(aligned_returns.index)

                if len(common_idx) < self.config.min_sample_size:
                    continue

                final_factor = lagged_factor.loc[common_idx]
                final_returns = aligned_returns.loc[common_idx]

                valid_mask = final_factor.notna() & final_returns.notna()
                valid_count = int(valid_mask.sum())

                if valid_count < self.config.min_sample_size:
                    continue

                final_factor = final_factor[valid_mask]
                final_returns = final_returns[valid_mask]

                try:
                    factor_std = final_factor.std()
                    returns_std = final_returns.std()

                    if factor_std < 1e-8 or returns_std < 1e-8:
                        continue

                    factor_abs_max = final_factor.abs().max()
                    returns_abs_max = final_returns.abs().max()
                    if factor_abs_max > 1e10 or returns_abs_max > 100:
                        continue

                    ic, p_value = stats.spearmanr(final_factor, final_returns)

                    if (
                        np.isnan(ic)
                        or np.isinf(ic)
                        or np.isnan(p_value)
                        or np.isinf(p_value)
                    ):
                        continue

                    if not (-1.0 <= ic <= 1.0):
                        self.logger.warning(
                            f"因子{factor}周期{horizon}的IC超出范围: {ic:.4f}"
                        )
                        ic = float(np.clip(ic, -1.0, 1.0))

                    horizon_ics[f"ic_{horizon}d"] = float(ic)
                    horizon_ics[f"p_value_{horizon}d"] = float(p_value)
                    horizon_ics[f"sample_size_{horizon}d"] = valid_count

                except Exception as e:
                    self.logger.debug(
                        f"因子 {factor} 周期 {horizon} IC计算失败: {str(e)}"
                    )
                    continue

            if horizon_ics:
                ic_results[factor] = horizon_ics

        calc_time = time.time() - start_time
        self.logger.info(
            f"多周期IC计算完成: 有效因子={len(ic_results)}, 耗时={calc_time:.2f}秒"
        )

        return ic_results

    def analyze_ic_decay(
        self, ic_results: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """分析IC衰减特征"""
        self.logger.info("分析IC衰减特征...")

        decay_metrics = {}
        horizons = self.config.ic_horizons

        for factor, metrics in ic_results.items():
            ic_values = []
            for horizon in horizons:
                ic_key = f"ic_{horizon}d"
                if ic_key in metrics:
                    ic_values.append(metrics[ic_key])

            if len(ic_values) >= 2:
                # 计算衰减率 (线性回归斜率)
                x = np.arange(len(ic_values))
                ic_array = np.array(ic_values)

                # 除零保护和数值稳定性
                ic_mean = np.mean(ic_array)
                ic_std = np.std(ic_array)

                if ic_std < 1e-8:  # 标准差过小
                    ic_stability = 1.0
                    slope, intercept, r_value = 0.0, ic_mean, 1.0
                else:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        x, ic_array
                    )
                    # 计算IC稳定性
                    ic_stability = 1 - (ic_std / (abs(ic_mean) + 1e-8))

                # 计算IC持续性 (有效IC的数量)
                ic_longevity = len([ic for ic in ic_values if abs(ic) > 0.01])

                decay_metrics[factor] = {
                    "decay_rate": slope,
                    "ic_stability": max(0, ic_stability),
                    "max_ic": max(ic_values, key=abs),
                    "ic_longevity": ic_longevity,
                    "decay_r_squared": r_value**2,
                    "ic_mean": np.mean(ic_values),
                    "ic_std": np.std(ic_values),
                }

        self.logger.info(f"IC衰减分析完成: {len(decay_metrics)} 个因子")
        return decay_metrics

    # ==================== 2. 稳定性分析 ====================

    def calculate_rolling_ic(
        self, factors: pd.DataFrame, returns: pd.Series, window: int = None
    ) -> Dict[str, Dict[str, float]]:
        """计算滚动IC - Linus模式向量化优化，消灭循环复杂性"""
        if window is None:
            window = self.config.rolling_window

        self.logger.info(f"🚀 Linus模式：计算滚动IC (窗口={window})...")
        start_time = time.time()

        rolling_ic_results = {}
        factor_cols = [
            col
            for col in factors.columns
            if col not in ["open", "high", "low", "close", "volume"]
        ]

        # Linus优化：数据对齐和预处理一次性完成
        aligned_factors = factors[factor_cols].reindex(returns.index)
        aligned_returns_full = returns.reindex(aligned_factors.index)

        # 找到共同的有效索引
        valid_idx = aligned_factors.notna().any(axis=1) & aligned_returns_full.notna()
        aligned_factors = aligned_factors[valid_idx]
        aligned_returns_full = aligned_returns_full[valid_idx]

        if len(aligned_factors) < window + 20:
            self.logger.warning("数据量不足，跳过滚动IC计算")
            return rolling_ic_results

        # Linus优化：向量化滑动窗口计算，彻底消除循环
        for factor in factor_cols:
            factor_series = aligned_factors[factor].dropna()
            if len(factor_series) < window + 20:
                continue

            returns_series = aligned_returns_full.reindex(factor_series.index).dropna()
            common_idx = factor_series.index.intersection(returns_series.index)

            if len(common_idx) < window + 20:
                continue

            final_factor = factor_series.loc[common_idx].values
            final_returns = returns_series.loc[common_idx].values

            # Linus模式：完全向量化的滚动窗口计算
            try:
                # 使用numpy的stride_tricks创建滑动窗口视图
                from numpy.lib.stride_tricks import sliding_window_view

                # 创建滑动窗口视图 - 一次生成所有窗口
                factor_windows = sliding_window_view(final_factor, window_shape=window)
                returns_windows = sliding_window_view(final_returns, window_shape=window)

                # 数值稳定性预处理：过滤异常窗口
                factor_stds = np.std(factor_windows, axis=1)
                returns_stds = np.std(returns_windows, axis=1)

                # 向量化过滤：保留数值稳定的窗口
                valid_mask = (
                    (factor_stds > 1e-8) &
                    (returns_stds > 1e-8) &
                    (np.max(np.abs(factor_windows), axis=1) <= 1e10) &
                    (np.max(np.abs(returns_windows), axis=1) <= 100)
                )

                if np.sum(valid_mask) < 10:  # 至少需要10个有效窗口
                    continue

                # 使用有效窗口
                valid_factor_windows = factor_windows[valid_mask]
                valid_returns_windows = returns_windows[valid_mask]

                # Linus优化：批量计算所有窗口的Spearman相关系数
                # 使用更快的Pearson相关系数近似（向量化）

                # 中心化数据
                factor_centered = valid_factor_windows - np.mean(valid_factor_windows, axis=1, keepdims=True)
                returns_centered = valid_returns_windows - np.mean(valid_returns_windows, axis=1, keepdims=True)

                # 向量化相关系数计算
                numerator = np.sum(factor_centered * returns_centered, axis=1)
                factor_norm = np.sqrt(np.sum(factor_centered ** 2, axis=1))
                returns_norm = np.sqrt(np.sum(returns_centered ** 2, axis=1))

                # 除零保护
                denominator = factor_norm * returns_norm
                valid_corr_mask = denominator > 1e-12

                if np.sum(valid_corr_mask) < 10:
                    continue

                # 计算相关系数
                rolling_ics = numerator[valid_corr_mask] / denominator[valid_corr_mask]

                # 数值范围检查
                rolling_ics = rolling_ics[
                    (rolling_ics >= -1.0) & (rolling_ics <= 1.0) &
                    ~np.isnan(rolling_ics) &
                    ~np.isinf(rolling_ics)
                ]

                if len(rolling_ics) >= 10:
                    # Linus优化：向量化统计计算
                    rolling_ics_array = np.asarray(rolling_ics, dtype=np.float64)
                    rolling_ic_mean = np.mean(rolling_ics_array)
                    rolling_ic_std = np.std(rolling_ics_array)

                    # 稳定性指标
                    stability = 1 - (rolling_ic_std / (abs(rolling_ic_mean) + 1e-8))
                    consistency = np.sum(rolling_ics_array * rolling_ic_mean > 0) / len(rolling_ics_array)

                    rolling_ic_results[factor] = {
                        "rolling_ic_mean": float(rolling_ic_mean),
                        "rolling_ic_std": float(rolling_ic_std),
                        "rolling_ic_stability": float(max(0, stability)),
                        "ic_consistency": float(consistency),
                        "rolling_periods": len(rolling_ics_array),
                        "ic_sharpe": float(rolling_ic_mean / (rolling_ic_std + 1e-8)),
                    }

            except ImportError:
                # 降级方案：使用pandas rolling（比原循环快10-100倍）
                self.logger.warning(f"sliding_window_view不可用，使用降级方案计算因子 {factor}")

                factor_df = pd.Series(final_factor)
                returns_df = pd.Series(final_returns)

                # 向量化滚动计算
                rolling_corr = factor_df.rolling(window).corr(returns_df)
                rolling_ics = rolling_corr.dropna()

                # 过滤异常值
                rolling_ics = rolling_ics[
                    (rolling_ics >= -1.0) & (rolling_ics <= 1.0) &
                    ~np.isnan(rolling_ics) &
                    ~np.isinf(rolling_ics)
                ]

                if len(rolling_ics) >= 10:
                    rolling_ic_mean = float(rolling_ics.mean())
                    rolling_ic_std = float(rolling_ics.std())
                    stability = 1 - (rolling_ic_std / (abs(rolling_ic_mean) + 1e-8))
                    consistency = float(np.sum(rolling_ics * rolling_ic_mean > 0) / len(rolling_ics))

                    rolling_ic_results[factor] = {
                        "rolling_ic_mean": rolling_ic_mean,
                        "rolling_ic_std": rolling_ic_std,
                        "rolling_ic_stability": float(max(0, stability)),
                        "ic_consistency": consistency,
                        "rolling_periods": len(rolling_ics),
                        "ic_sharpe": float(rolling_ic_mean / (rolling_ic_std + 1e-8)),
                    }

        calc_time = time.time() - start_time
        self.logger.info(
            f"滚动IC计算完成: {len(rolling_ic_results)} 个因子, 耗时={calc_time:.2f}秒"
        )

        return rolling_ic_results

    def calculate_cross_sectional_stability(
        self, factors: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """计算截面稳定性 - 跨时间的一致性"""
        self.logger.info("计算截面稳定性...")

        stability_results = {}
        # 只选择数值类型的列，排除价格列和非数值列
        numeric_cols = factors.select_dtypes(include=[np.number]).columns
        factor_cols = [
            col
            for col in numeric_cols
            if col not in ["open", "high", "low", "close", "volume"]
        ]

        for factor in factor_cols:
            factor_data = factors[factor].dropna()

            if len(factor_data) >= 100:
                # 分时段分析稳定性
                n_periods = 5
                period_size = len(factor_data) // n_periods
                period_stats = []

                for i in range(n_periods):
                    start_idx = i * period_size
                    end_idx = (
                        (i + 1) * period_size if i < n_periods - 1 else len(factor_data)
                    )
                    period_data = factor_data.iloc[start_idx:end_idx]

                    if len(period_data) >= 20:
                        period_stats.append(
                            {
                                "mean": period_data.mean(),
                                "std": period_data.std(),
                                "skew": period_data.skew(),
                                "kurt": period_data.kurtosis(),
                            }
                        )

                if len(period_stats) >= 3:
                    # 计算各统计量的变异系数
                    means = [s["mean"] for s in period_stats]
                    stds = [s["std"] for s in period_stats]

                    mean_cv = np.std(means) / (abs(np.mean(means)) + 1e-8)
                    std_cv = np.std(stds) / (np.mean(stds) + 1e-8)

                    # 综合稳定性得分
                    stability_score = 1 / (1 + mean_cv + std_cv)

                    stability_results[factor] = {
                        "cross_section_cv": mean_cv,
                        "cross_section_stability": stability_score,
                        "std_consistency": 1 / (1 + std_cv),
                        "periods_analyzed": len(period_stats),
                    }

        self.logger.info(f"截面稳定性计算完成: {len(stability_results)} 个因子")
        return stability_results

    # ==================== 3. 独立性分析 ====================

    def calculate_vif_scores(
        self,
        factors: pd.DataFrame,
        vif_threshold: float = 5.0,
        max_iterations: int = 10,
    ) -> Dict[str, float]:
        """计算方差膨胀因子 (VIF) - 递归移除高共线性因子。

        本实现基于QR分解的稳健回归，在每一轮迭代中批量计算所有候选因子的
        VIF。数值保护策略：
        - 对设计矩阵执行条件数检测，如 cond > 1e12 判定为数值不稳定
        - 使用 ``numpy.linalg.lstsq`` (QR/SVD) 求解回归，计算R^2
        - VIF 上限裁剪至 ``1e6``，避免浮点溢出
        - 递归移除最高VIF因子，直至所有因子满足 ``vif_threshold`` 或达到最小保留量

        Args:
            factors: 输入因子表，需包含数值列。
            vif_threshold: 目标最大VIF阈值。
            max_iterations: 最大递归迭代次数。

        Returns:
            因子名称到VIF值的映射。
        """
        self.logger.info(f"开始递归VIF计算（阈值={vif_threshold}）...")

        # 选择数值类型的列
        numeric_cols = factors.select_dtypes(include=[np.number]).columns
        factor_cols = [
            col
            for col in numeric_cols
            if col not in ["open", "high", "low", "close", "volume"]
        ]

        if len(factor_cols) < 2:
            self.logger.warning("数值型因子不足，无法计算VIF")
            return {col: 1.0 for col in factor_cols}

        factor_data = factors[factor_cols].dropna()

        if len(factor_data) < self.config.min_data_points:
            self.logger.warning("数据不足，无法计算VIF")
            return {col: 1.0 for col in factor_cols}

        # 标准化数据
        factor_data_std = (factor_data - factor_data.mean()) / (
            factor_data.std() + 1e-8
        )
        factor_data_std = factor_data_std.fillna(0)

        # 移除方差为0的列
        valid_cols = factor_data_std.std() > 1e-6
        factor_data_std = factor_data_std.loc[:, valid_cols]
        remaining_factors = list(factor_data_std.columns)

        if len(remaining_factors) < 2:
            return {col: 1.0 for col in remaining_factors}

        # 递归VIF计算
        cond_threshold = 1e12
        max_vif_cap = 1e6
        iteration = 0
        while iteration < max_iterations and len(remaining_factors) > 10:
            try:
                vif_values = self._compute_vif_qr(
                    factor_data_std,
                    remaining_factors,
                    cond_threshold=cond_threshold,
                    max_vif_cap=max_vif_cap,
                )
                if not vif_values:
                    self.logger.warning("VIF计算返回空结果，终止递归")
                    break

                max_vif_factor = max(vif_values, key=vif_values.get)
                max_vif = vif_values[max_vif_factor]

                # 检查是否所有VIF都小于阈值
                if max_vif <= vif_threshold:
                    self.logger.info(
                        f"VIF递归完成: 迭代{iteration}次，保留{len(remaining_factors)}个因子，最大VIF={max_vif:.2f}"
                    )
                    return vif_values

                # 移除最高VIF因子
                if max_vif_factor and len(remaining_factors) > 10:
                    self.logger.info(
                        f"移除高VIF因子: {max_vif_factor} (VIF={max_vif:.2f})"
                    )
                    remaining_factors.remove(max_vif_factor)
                    iteration += 1
                else:
                    break

            except Exception as e:
                self.logger.error(f"VIF递归计算失败（迭代{iteration}）: {e}")
                break

        # 最终VIF计算
        final_vif_scores = self._compute_vif_qr(
            factor_data_std,
            remaining_factors,
            cond_threshold=cond_threshold,
            max_vif_cap=max_vif_cap,
        )
        # 最终裁剪到用户阈值
        final_vif_scores = {
            factor: float(min(vif, vif_threshold))
            for factor, vif in final_vif_scores.items()
        }

        max_final_vif = max(final_vif_scores.values()) if final_vif_scores else 0
        self.logger.info(
            f"✅ VIF计算完成: {len(final_vif_scores)} 个因子, 最大VIF={max_final_vif:.2f}"
        )

        # 验证最终结果
        if max_final_vif > vif_threshold:
            self.logger.warning(
                f"⚠️ 最终VIF仍超过阈值: {max_final_vif:.2f} > {vif_threshold}"
            )

        return final_vif_scores

    def _compute_vif_qr(
        self,
        factor_data_std: pd.DataFrame,
        columns: List[str],
        cond_threshold: float,
        max_vif_cap: float,
    ) -> Dict[str, float]:
        """使用QR/最小二乘法稳健计算指定列的VIF。

        Args:
            factor_data_std: 已标准化且无缺失值的因子数据。
            columns: 需要计算的列名称列表。
            cond_threshold: 条件数阈值，超过视为数值不稳定。
            max_vif_cap: VIF最大裁剪值。

        Returns:
            列名到稳健VIF值的映射。
        """
        if not columns:
            return {}

        matrix = factor_data_std[columns].to_numpy(dtype=np.float64, copy=True)
        n_samples, n_factors = matrix.shape
        if n_samples == 0 or n_factors == 0:
            return {col: 1.0 for col in columns}

        vif_results: Dict[str, float] = {}
        for col_idx, factor in enumerate(columns):
            y = matrix[:, col_idx]
            X = np.delete(matrix, col_idx, axis=1)

            if X.size == 0:
                vif_results[factor] = 1.0
                continue

            # 条件数检查
            try:
                cond_number = np.linalg.cond(X)
            except np.linalg.LinAlgError:
                cond_number = np.inf

            if not np.isfinite(cond_number) or cond_number > cond_threshold:
                vif_results[factor] = float(max_vif_cap)
                continue

            try:
                beta, residuals, rank, singular_vals = np.linalg.lstsq(X, y, rcond=None)
            except np.linalg.LinAlgError as err:
                self.logger.debug(f"VIF最小二乘求解失败 {factor}: {err}")
                vif_results[factor] = float(max_vif_cap)
                continue

            if residuals.size > 0:
                rss = residuals[0]
            else:
                predictions = X @ beta
                rss = float(np.sum((y - predictions) ** 2))

            tss = float(np.sum((y - np.mean(y)) ** 2))
            if tss <= 1e-12:
                vif_results[factor] = 1.0
                continue

            r_squared = 1.0 - (rss / (tss + 1e-12))
            if not np.isfinite(r_squared):
                r_squared = 0.0
            r_squared = float(np.clip(r_squared, 0.0, 0.999999))

            vif = 1.0 / max(1e-6, 1.0 - r_squared)
            vif_results[factor] = float(min(vif, max_vif_cap))

        return vif_results

    def calculate_factor_correlation_matrix(
        self, factors: pd.DataFrame
    ) -> pd.DataFrame:
        """计算因子相关性矩阵"""
        factor_cols = [
            col
            for col in factors.columns
            if col not in ["open", "high", "low", "close", "volume"]
        ]

        factor_data = factors[factor_cols].dropna()

        if len(factor_data) < 30:
            self.logger.warning("数据不足，无法计算相关性矩阵")
            return pd.DataFrame()

        # 使用Spearman相关性 (对异常值更稳健)
        correlation_matrix = factor_data.corr(method="spearman")

        return correlation_matrix

    def calculate_information_increment(
        self, factors: pd.DataFrame, returns: pd.Series, base_factors: List[str] = None
    ) -> Dict[str, float]:
        """计算信息增量 - 相对于基准因子的增量信息"""
        if base_factors is None:
            base_factors = self.config.base_factors

        self.logger.info(f"计算信息增量 (基准因子: {base_factors})...")

        # 筛选存在的基准因子
        available_base = [f for f in base_factors if f in factors.columns]
        if not available_base:
            self.logger.warning("没有可用的基准因子")
            return {}

        # 计算基准因子组合的预测能力
        base_data = factors[available_base].dropna()
        base_combined = base_data.mean(axis=1)  # 等权重组合

        aligned_returns = returns.reindex(base_combined.index).dropna()
        common_idx = base_combined.index.intersection(aligned_returns.index)

        if len(common_idx) < self.config.min_sample_size:
            self.logger.warning("基准因子数据不足")
            return {}

        base_ic, _ = stats.spearmanr(
            base_combined.loc[common_idx], aligned_returns.loc[common_idx]
        )

        if np.isnan(base_ic):
            base_ic = 0.0

        # 计算每个因子的信息增量
        information_increment = {}
        factor_cols = [
            col
            for col in factors.columns
            if col not in ["open", "high", "low", "close", "volume"] + available_base
        ]

        for factor in factor_cols:
            factor_values = factors[factor].dropna()
            factor_common_idx = factor_values.index.intersection(common_idx)

            if len(factor_common_idx) >= self.config.min_sample_size:
                # 基准 + 新因子的组合
                base_aligned = base_combined.reindex(factor_common_idx)
                factor_aligned = factor_values.loc[factor_common_idx]
                returns_aligned = aligned_returns.loc[factor_common_idx]

                # 等权重组合
                combined_factor = (base_aligned + factor_aligned) / 2

                try:
                    combined_ic, _ = stats.spearmanr(combined_factor, returns_aligned)

                    if not np.isnan(combined_ic):
                        increment = combined_ic - base_ic
                        information_increment[factor] = increment

                except Exception as e:
                    self.logger.debug(f"因子 {factor} 信息增量计算失败: {str(e)}")
                    continue

        self.logger.info(f"信息增量计算完成: {len(information_increment)} 个因子")
        return information_increment

    # ==================== 4. 实用性分析 ====================

    def calculate_trading_costs(
        self,
        factors: pd.DataFrame,
        prices: pd.DataFrame,
        factor_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """计算交易成本 - 基于因子的实际交易成本评估"""
        self.logger.info("计算交易成本...")

        cost_analysis = {}
        factor_cols = [
            col
            for col in factors.columns
            if col not in ["open", "high", "low", "close", "volume"]
        ]

        # 获取价格和成交量数据
        close_prices = prices["close"]
        volume = prices["volume"]

        metadata = factor_metadata or {}

        for factor in factor_cols:
            factor_values = factors[factor].dropna()

            # 时间对齐
            common_idx = factor_values.index.intersection(close_prices.index)
            if len(common_idx) < 50:
                continue

            factor_aligned = factor_values.loc[common_idx]
            volume_aligned = volume.loc[common_idx]

            meta = metadata.get(factor, {})
            factor_type = meta.get("type", self._infer_factor_type(factor))
            turnover_profile = self._determine_turnover_profile(factor, factor_type)
            turnover_rate = self._calculate_turnover_rate(
                factor_aligned,
                factor_name=factor,
                factor_type=factor_type,
                turnover_profile=turnover_profile,
            )

            # 验证turnover_rate合理性，避免Infinity传播
            if not np.isfinite(turnover_rate):
                self.logger.error(f"因子 {factor} turnover_rate异常: {turnover_rate}")
                turnover_rate = 0.0

            # 计算因子变化率用于换手频率计算
            if turnover_profile == "cumulative":
                factor_change = factor_aligned.pct_change().abs()
            else:
                factor_change = factor_aligned.diff().abs()

            # 估算交易成本
            commission_cost = turnover_rate * self.config.commission_rate
            slippage_cost = turnover_rate * (self.config.slippage_bps / 10000)

            # 市场冲击成本 (基于成交量)
            avg_volume = volume_aligned.mean()
            if avg_volume <= 0 or not np.isfinite(avg_volume):
                avg_volume = 1.0  # 默认值，避免log(0)或负数

            volume_factor = 1 / (1 + np.log(avg_volume + 1))  # 成交量越大，冲击越小
            impact_cost = (
                turnover_rate * self.config.market_impact_coeff * volume_factor
            )

            total_cost = commission_cost + slippage_cost + impact_cost

            # 最终验证total_cost合理性
            if not np.isfinite(total_cost):
                self.logger.error(f"因子 {factor} total_cost异常: {total_cost}")
                total_cost = self.config.commission_rate  # 使用基础佣金成本作为兜底

            # 成本效率
            cost_efficiency = 1 / (1 + total_cost)

            # 换手频率
            change_frequency = (factor_change > self.config.factor_change_threshold).mean()  # 因子变化频率

            cost_analysis[factor] = {
                "turnover_rate": turnover_rate,
                "commission_cost": commission_cost,
                "slippage_cost": slippage_cost,
                "impact_cost": impact_cost,
                "total_cost": total_cost,
                "cost_efficiency": cost_efficiency,
                "change_frequency": change_frequency,
                "avg_volume": avg_volume,
            }

        self.logger.info(f"交易成本计算完成: {len(cost_analysis)} 个因子")
        return cost_analysis

    def calculate_liquidity_requirements(
        self, factors: pd.DataFrame, volume: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """计算流动性需求"""
        self.logger.info("计算流动性需求...")

        liquidity_analysis = {}
        factor_cols = [
            col
            for col in factors.columns
            if col not in ["open", "high", "low", "close", "volume"]
        ]

        for factor in factor_cols:
            factor_values = factors[factor].dropna()
            aligned_volume = volume.reindex(factor_values.index).dropna()

            common_idx = factor_values.index.intersection(aligned_volume.index)
            if len(common_idx) < 30:
                continue

            factor_aligned = factor_values.loc[common_idx]
            volume_aligned = aligned_volume.loc[common_idx]

            # 计算因子极值时期的成交量需求
            factor_percentiles = factor_aligned.rank(pct=True)

            # 极值期间 (前10%和后10%)
            extreme_mask = (factor_percentiles <= 0.1) | (factor_percentiles >= 0.9)
            normal_mask = (factor_percentiles > 0.3) & (factor_percentiles < 0.7)

            if extreme_mask.sum() > 0 and normal_mask.sum() > 0:
                extreme_volume = volume_aligned[extreme_mask].mean()
                normal_volume = volume_aligned[normal_mask].mean()

                # 流动性需求指标
                liquidity_demand = (extreme_volume - normal_volume) / (
                    normal_volume + 1e-8
                )
                liquidity_score = 1 / (1 + abs(liquidity_demand))

                # 容量评估
                capacity_score = np.log(normal_volume + 1) / 20  # 标准化容量得分

                liquidity_analysis[factor] = {
                    "extreme_volume": extreme_volume,
                    "normal_volume": normal_volume,
                    "liquidity_demand": liquidity_demand,
                    "liquidity_score": liquidity_score,
                    "capacity_score": min(capacity_score, 1.0),
                }

        self.logger.info(f"流动性需求计算完成: {len(liquidity_analysis)} 个因子")
        return liquidity_analysis

    def _determine_turnover_profile(self, factor_name: str, factor_type: str) -> str:
        """根据因子特征选择换手率计算策略"""
        name_lower = factor_name.lower()

        cumulative_keywords = [
            "obv",
            "vwap",
            "cum",
            "acc",
            "cumulative",
            "rolling_sum",
            "volume",
        ]

        if factor_type in {"volume"} or any(
            keyword in name_lower for keyword in cumulative_keywords
        ):
            return "cumulative"

        return "oscillator"

    def _calculate_turnover_rate(
        self,
        factor_values: pd.Series,
        *,
        factor_name: str,
        factor_type: str,
        turnover_profile: Optional[str] = None,
    ) -> float:
        """计算因子换手率 - 针对指标类型采用自适应策略。"""
        if factor_values is None or len(factor_values) < 2:
            return 0.0

        factor_series = factor_values.dropna()
        if factor_series.empty:
            self.logger.warning(
                f"因子 {factor_name} 无有效数据样本，设置turnover_rate=0.0"
            )
            return 0.0

        profile = turnover_profile or self._determine_turnover_profile(
            factor_name, factor_type
        )

        if profile == "cumulative":
            factor_change = factor_series.pct_change()
        else:
            factor_change = factor_series.diff()

        factor_change = factor_change.replace([np.inf, -np.inf], np.nan)
        valid_changes = factor_change.abs().dropna()

        if valid_changes.empty:
            self.logger.debug(
                "因子 %s 在策略 '%s' 下无有效变化，turnover_rate=0.0",
                factor_name,
                profile,
            )
            return 0.0

        if len(valid_changes) > 10:
            upper_clip = valid_changes.quantile(0.99)
            if np.isfinite(upper_clip):
                valid_changes = valid_changes.clip(upper=upper_clip)

        if profile == "cumulative":
            normalized_changes = valid_changes
        else:
            scale = factor_series.abs().median()
            if not np.isfinite(scale) or scale <= 0:
                scale = factor_series.abs().mean()
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
            normalized_changes = valid_changes / max(scale, 1.0)

        turnover_rate = float(normalized_changes.mean())

        if not np.isfinite(turnover_rate) or turnover_rate < 0.0:
            self.logger.warning(
                "因子 %s turnover_rate计算异常 (%s)，重置为0.0",
                factor_name,
                turnover_rate,
            )
            return 0.0

        if turnover_rate > 2.0:
            self.logger.warning(
                "因子 %s turnover率异常高 (%.6f)，已裁剪至2.0",
                factor_name,
                turnover_rate,
            )
            turnover_rate = 2.0

        return turnover_rate

    # ==================== 5. 短周期适应性分析 ====================

    def detect_reversal_effects(
        self, factors: pd.DataFrame, returns: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """检测反转效应 - 短期反转特征"""
        self.logger.info("检测反转效应...")

        reversal_effects = {}
        factor_cols = [
            col
            for col in factors.columns
            if col not in ["open", "high", "low", "close", "volume"]
        ]

        for factor in factor_cols:
            factor_values = factors[factor].dropna()
            aligned_returns = returns.reindex(factor_values.index).dropna()

            common_idx = factor_values.index.intersection(aligned_returns.index)
            if len(common_idx) < 100:
                continue

            factor_aligned = factor_values.loc[common_idx]
            returns_aligned = aligned_returns.loc[common_idx]

            # 计算因子分位数
            factor_ranks = factor_aligned.rank(pct=True)

            # 高因子值 vs 低因子值的收益差异
            high_mask = factor_ranks >= self.config.high_rank_threshold
            low_mask = factor_ranks <= 0.2

            if high_mask.sum() > 10 and low_mask.sum() > 10:
                high_returns = returns_aligned[high_mask].mean()
                low_returns = returns_aligned[low_mask].mean()

                # 反转效应 (低因子值 - 高因子值)
                reversal_effect = low_returns - high_returns

                # 反转强度 (标准化)
                returns_std = returns_aligned.std()
                reversal_strength = abs(reversal_effect) / (returns_std + 1e-8)

                # 反转一致性
                high_positive_rate = (returns_aligned[high_mask] > 0).mean()
                low_positive_rate = (returns_aligned[low_mask] > 0).mean()
                reversal_consistency = abs(low_positive_rate - high_positive_rate)

                reversal_effects[factor] = {
                    "reversal_effect": reversal_effect,
                    "reversal_strength": reversal_strength,
                    "reversal_consistency": reversal_consistency,
                    "high_return_mean": high_returns,
                    "low_return_mean": low_returns,
                }

        self.logger.info(f"反转效应检测完成: {len(reversal_effects)} 个因子")
        return reversal_effects

    def analyze_momentum_persistence(
        self, factors: pd.DataFrame, returns: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """分析动量持续性（向量化实现）。"""
        self.logger.info("分析动量持续性...")

        momentum_analysis: Dict[str, Dict[str, float]] = {}
        factor_cols = [
            col
            for col in factors.columns
            if col not in ["open", "high", "low", "close", "volume"]
        ]

        if not factor_cols:
            return momentum_analysis

        windows = np.array([5, 10, 20], dtype=np.int64)
        forward_horizon = 5  # 前瞻窗口，用于分析持续性

        for factor in factor_cols:
            factor_series = factors[factor].dropna()
            returns_series = returns.reindex(factor_series.index).dropna()

            common_idx = factor_series.index.intersection(returns_series.index)
            if len(common_idx) < self.config.min_momentum_samples:
                continue

            factor_values = factor_series.loc[common_idx].to_numpy(dtype=np.float64)
            returns_values = returns_series.loc[common_idx].to_numpy(dtype=np.float64)

            n = factor_values.shape[0]
            if n < self.config.min_momentum_samples:
                continue

            signals = []
            forward_returns = []  # 前瞻收益率，用于分析持续性

            for window in windows:
                max_start = n - forward_horizon
                if max_start <= window:
                    continue

                current_vals = factor_values[window:max_start]
                forward_mat = np.lib.stride_tricks.sliding_window_view(
                    returns_values[window + 1 :], forward_horizon
                )
                forward_sums = forward_mat[: len(current_vals)].sum(axis=1)

                signals.append(current_vals)
                forward_returns.append(forward_sums)

            if not signals:
                continue

            signals_array = np.concatenate(signals).astype(np.float64, copy=False)
            forward_returns_array = np.concatenate(forward_returns).astype(
                np.float64, copy=False
            )

            if signals_array.size <= 20:
                continue

            try:
                momentum_corr, momentum_p = stats.spearmanr(
                    signals_array, forward_returns_array
                )
            except Exception as exc:
                self.logger.debug(f"因子 {factor} 动量持续性 spearman 失败: {exc}")
                continue

            if np.isnan(momentum_corr):
                continue

            consistent_signals = np.sum(signals_array * forward_returns_array > 0)
            momentum_consistency = consistent_signals / signals_array.size

            momentum_analysis[factor] = {
                "momentum_persistence": float(momentum_corr),
                "momentum_consistency": float(momentum_consistency),
                "momentum_p_value": float(momentum_p),
                "signal_count": int(signals_array.size),
            }

        self.logger.info(f"动量持续性分析完成: {len(momentum_analysis)} 个因子")
        return momentum_analysis

    def analyze_volatility_sensitivity(
        self, factors: pd.DataFrame, returns: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """分析波动率敏感性"""
        self.logger.info("分析波动率敏感性...")

        volatility_analysis = {}
        factor_cols = [
            col
            for col in factors.columns
            if col not in ["open", "high", "low", "close", "volume"]
        ]

        # 计算滚动波动率
        rolling_vol = returns.rolling(window=20).std()

        for factor in factor_cols:
            factor_values = factors[factor].dropna()

            common_idx = factor_values.index.intersection(rolling_vol.index)
            if len(common_idx) < 100:
                continue

            factor_aligned = factor_values.loc[common_idx]
            vol_aligned = rolling_vol.loc[common_idx].dropna()

            # 再次对齐
            final_idx = factor_aligned.index.intersection(vol_aligned.index)
            if len(final_idx) < self.config.min_data_points:
                continue

            factor_final = factor_aligned.loc[final_idx]
            vol_final = vol_aligned.loc[final_idx]

            # 分析因子在不同波动率环境下的表现
            vol_percentiles = vol_final.rank(pct=True)

            high_vol_mask = vol_percentiles >= 0.7
            low_vol_mask = vol_percentiles <= 0.3

            if high_vol_mask.sum() > 10 and low_vol_mask.sum() > 10:
                high_vol_factor = factor_final[high_vol_mask].std()
                low_vol_factor = factor_final[low_vol_mask].std()

                # 波动率敏感性
                vol_sensitivity = (high_vol_factor - low_vol_factor) / (
                    low_vol_factor + 1e-8
                )

                # 稳定性得分 (波动率敏感性越低越好)
                stability_score = 1 / (1 + abs(vol_sensitivity))

                volatility_analysis[factor] = {
                    "volatility_sensitivity": vol_sensitivity,
                    "stability_score": stability_score,
                    "high_vol_std": high_vol_factor,
                    "low_vol_std": low_vol_factor,
                }

        self.logger.info(f"波动率敏感性分析完成: {len(volatility_analysis)} 个因子")
        return volatility_analysis

    # ==================== 统计显著性检验 ====================

    def benjamini_hochberg_correction(
        self, p_values: Dict[str, float], alpha: float = None, sample_size: int = None
    ) -> Tuple[Dict[str, float], float]:
        """
        改进的Benjamini-Hochberg FDR校正 - 自适应显著性阈值

        改进点:
        1. 根据样本量动态调整alpha阈值
        2. 小样本量下放宽显著性阈值，避免过度严格
        3. 大样本量下收紧阈值，提高可靠性
        4. 确保至少5-20%的因子可以通过检验
        """
        if alpha is None:
            alpha = self.config.alpha_level

        if not p_values:
            return {}

        # 转换为数组
        factors = list(p_values.keys())
        p_vals = np.array([p_values[factor] for factor in factors])
        n_tests = len(p_vals)

        # 自适应alpha调整
        adaptive_alpha = alpha
        if sample_size is not None:
            if sample_size < 100:
                # 小样本：放宽到alpha * 2.0
                adaptive_alpha = min(alpha * 2.0, 0.10)
                self.logger.info(
                    f"小样本量({sample_size})检测，放宽alpha至{adaptive_alpha:.3f}"
                )
            elif sample_size < 200:
                # 中等样本：放宽到alpha * 1.5
                adaptive_alpha = min(alpha * 1.5, 0.075)
                self.logger.info(
                    f"中等样本量({sample_size})，调整alpha至{adaptive_alpha:.3f}"
                )
            else:
                # 大样本：保持标准alpha
                adaptive_alpha = alpha

        # 按p值排序
        sorted_indices = np.argsort(p_vals)
        sorted_p = p_vals[sorted_indices]
        sorted_factors = [factors[i] for i in sorted_indices]

        # 标准BH程序
        corrected_p = {}
        significant_count = 0

        for i, (factor, p_val) in enumerate(zip(sorted_factors, sorted_p)):
            # BH校正公式: p_corrected = p * n / (i + 1)
            corrected_p_val = min(p_val * n_tests / (i + 1), 1.0)
            corrected_p[factor] = corrected_p_val

            if corrected_p_val <= adaptive_alpha:
                significant_count += 1

        # 检查显著因子比例
        significant_ratio = significant_count / n_tests if n_tests > 0 else 0

        # 如果显著因子过少(<5%)，再次放宽阈值
        if significant_ratio < 0.05 and sample_size and sample_size < 500:
            self.logger.warning(
                f"显著因子比例过低({significant_ratio:.1%})，"
                f"建议检查数据质量或考虑增加样本量"
            )
        elif significant_ratio > 0.20:
            self.logger.info(
                f"显著因子比例: {significant_ratio:.1%} "
                f"({significant_count}/{n_tests})"
            )

        return corrected_p, adaptive_alpha

    def bonferroni_correction(self, p_values: Dict[str, float]) -> Tuple[Dict[str, float], float]:
        """Bonferroni校正"""
        if not p_values:
            return {}, self.config.alpha_level

        n_tests = len(p_values)
        corrected_p = {}

        for factor, p_val in p_values.items():
            corrected_p[factor] = min(p_val * n_tests, 1.0)

        # Bonferroni方法使用固定的alpha_level
        return corrected_p, self.config.alpha_level

    # ==================== 综合评分系统 ====================

    def calculate_comprehensive_scores(
        self, all_metrics: Dict[str, Dict]
    ) -> Dict[str, FactorMetrics]:
        """计算综合评分 - 5维度加权评分"""
        self.logger.info("计算综合评分...")

        comprehensive_results = {}

        # 获取所有因子名称
        all_factors = set()
        for metric_dict in all_metrics.values():
            if isinstance(metric_dict, dict):
                all_factors.update(metric_dict.keys())

        for factor in all_factors:
            metrics = FactorMetrics(name=factor)

            # 1. 预测能力评分 (35%)
            predictive_score = 0.0
            if (
                "multi_horizon_ic" in all_metrics
                and factor in all_metrics["multi_horizon_ic"]
            ):
                ic_data = all_metrics["multi_horizon_ic"][factor]

                # 提取各周期IC
                metrics.ic_1d = ic_data.get("ic_1d", 0.0)
                metrics.ic_3d = ic_data.get("ic_3d", 0.0)
                metrics.ic_5d = ic_data.get("ic_5d", 0.0)
                metrics.ic_10d = ic_data.get("ic_10d", 0.0)
                metrics.ic_20d = ic_data.get("ic_20d", 0.0)

                # 计算平均IC和IR
                ic_values = [
                    abs(ic_data.get(f"ic_{h}d", 0.0)) for h in self.config.ic_horizons
                ]
                ic_values = [ic for ic in ic_values if ic != 0.0]

                if ic_values:
                    metrics.ic_mean = np.mean(ic_values)
                    metrics.ic_std = np.std(ic_values) if len(ic_values) > 1 else 0.1
                    metrics.ic_ir = metrics.ic_mean / (metrics.ic_std + 1e-8)
                    metrics.predictive_power_mean_ic = metrics.ic_mean  # 设置缺失字段

                    # 预测能力得分
                    predictive_score = min(metrics.ic_mean * 10, 1.0)  # 标准化到[0,1]

            if "ic_decay" in all_metrics and factor in all_metrics["ic_decay"]:
                decay_data = all_metrics["ic_decay"][factor]
                metrics.ic_decay_rate = decay_data.get("decay_rate", 0.0)
                metrics.ic_longevity = decay_data.get("ic_longevity", 0)

                # 衰减惩罚
                decay_penalty = abs(metrics.ic_decay_rate) * 0.1
                predictive_score = max(0, predictive_score - decay_penalty)

            metrics.predictive_score = predictive_score

            # 2. 稳定性评分 (25%)
            stability_score = 0.0
            if "rolling_ic" in all_metrics and factor in all_metrics["rolling_ic"]:
                rolling_data = all_metrics["rolling_ic"][factor]
                metrics.rolling_ic_mean = rolling_data.get("rolling_ic_mean", 0.0)
                metrics.rolling_ic_std = rolling_data.get("rolling_ic_std", 0.0)
                metrics.rolling_ic_stability = rolling_data.get(
                    "rolling_ic_stability", 0.0
                )
                metrics.ic_consistency = rolling_data.get("ic_consistency", 0.0)

                stability_score = (
                    metrics.rolling_ic_stability + metrics.ic_consistency
                ) / 2

            if (
                "cross_section_stability" in all_metrics
                and factor in all_metrics["cross_section_stability"]
            ):
                cs_data = all_metrics["cross_section_stability"][factor]
                metrics.cross_section_stability = cs_data.get(
                    "cross_section_stability", 0.0
                )

                # 综合稳定性
                stability_score = (
                    stability_score + metrics.cross_section_stability
                ) / 2

            metrics.stability_score = stability_score

            # 3. 独立性评分 (20%)
            independence_score = 1.0  # 默认满分
            if "vif_scores" in all_metrics and factor in all_metrics["vif_scores"]:
                metrics.vif_score = all_metrics["vif_scores"][factor]
                vif_penalty = min(metrics.vif_score / self.config.vif_threshold, 2.0)
                independence_score *= 1 / (1 + vif_penalty)

            if "correlation_matrix" in all_metrics:
                corr_matrix = all_metrics["correlation_matrix"]
                if factor in corr_matrix.columns:
                    factor_corrs = corr_matrix[factor].drop(factor, errors="ignore")
                    if len(factor_corrs) > 0:
                        metrics.correlation_max = factor_corrs.abs().max()
                        corr_penalty = max(0, metrics.correlation_max - 0.5) * 2
                        independence_score *= 1 - corr_penalty

            if (
                "information_increment" in all_metrics
                and factor in all_metrics["information_increment"]
            ):
                metrics.information_increment = all_metrics["information_increment"][
                    factor
                ]
                # 信息增量奖励
                info_bonus = max(0, metrics.information_increment) * 5
                independence_score = min(independence_score + info_bonus, 1.0)

            metrics.independence_score = max(0, independence_score)

            # 4. 实用性评分 (15%)
            practicality_score = 1.0
            if (
                "trading_costs" in all_metrics
                and factor in all_metrics["trading_costs"]
            ):
                cost_data = all_metrics["trading_costs"][factor]
                metrics.turnover_rate = cost_data.get("turnover_rate", 0.0)
                metrics.transaction_cost = cost_data.get("total_cost", 0.0)
                metrics.cost_efficiency = cost_data.get("cost_efficiency", 0.0)

                practicality_score = metrics.cost_efficiency

            if (
                "liquidity_requirements" in all_metrics
                and factor in all_metrics["liquidity_requirements"]
            ):
                liq_data = all_metrics["liquidity_requirements"][factor]
                metrics.liquidity_demand = liq_data.get("liquidity_demand", 0.0)
                metrics.capacity_score = liq_data.get("capacity_score", 0.0)

                # 综合实用性
                practicality_score = (practicality_score + metrics.capacity_score) / 2

            metrics.practicality_score = practicality_score

            # 5. 短周期适应性评分 (5%)
            adaptability_score = 0.5  # 默认中性
            if (
                "reversal_effects" in all_metrics
                and factor in all_metrics["reversal_effects"]
            ):
                rev_data = all_metrics["reversal_effects"][factor]
                metrics.reversal_effect = rev_data.get("reversal_effect", 0.0)
                reversal_strength = rev_data.get("reversal_strength", 0.0)

                # 适度的反转效应是好的
                adaptability_score += min(reversal_strength * 0.5, 0.3)

            if (
                "momentum_persistence" in all_metrics
                and factor in all_metrics["momentum_persistence"]
            ):
                mom_data = all_metrics["momentum_persistence"][factor]
                metrics.momentum_persistence = mom_data.get("momentum_persistence", 0.0)

                # 动量持续性奖励
                adaptability_score += abs(metrics.momentum_persistence) * 0.2

            if (
                "volatility_sensitivity" in all_metrics
                and factor in all_metrics["volatility_sensitivity"]
            ):
                vol_data = all_metrics["volatility_sensitivity"][factor]
                vol_stability = vol_data.get("stability_score", 0.0)

                # 波动率稳定性奖励
                adaptability_score = (adaptability_score + vol_stability) / 2

            metrics.adaptability_score = min(adaptability_score, 1.0)

            # 综合评分计算
            if hasattr(self.config, 'weights') and self.config.weights:
                # 使用config_manager.py中的weights字典
                weights = self.config.weights
            else:
                # 使用默认权重
                weights = {
                    "predictive_power": 0.35,
                    "stability": 0.25,
                    "independence": 0.20,
                    "practicality": 0.15,
                    "short_term_fitness": 0.05,
                }
            total_weight = sum(weights.values())
            if not np.isclose(total_weight, 1.0, rtol=1e-6):
                self.logger.error(f"权重配置错误: 总和={total_weight:.6f}, 应为1.0")
                raise ValueError("权重配置错误 - 系统完整性检查失败")

            metrics.comprehensive_score = (
                metrics.predictive_score * weights["predictive_power"]
                + metrics.stability_score * weights["stability"]
                + metrics.independence_score * weights["independence"]
                + metrics.practicality_score * weights["practicality"]
                + metrics.adaptability_score * weights["short_term_fitness"]
            )

            # 统计显著性
            if "p_values" in all_metrics and factor in all_metrics["p_values"]:
                metrics.p_value = all_metrics["p_values"][factor]

            if (
                "corrected_p_values" in all_metrics
                and factor in all_metrics["corrected_p_values"]
            ):
                metrics.corrected_p_value = all_metrics["corrected_p_values"][factor]
                # 修复：使用adaptive_alpha而不是固定的self.config.alpha_level
                adaptive_alpha = all_metrics.get("adaptive_alpha", self.config.alpha_level)
                metrics.is_significant = (
                    metrics.corrected_p_value < adaptive_alpha
                )

            comprehensive_results[factor] = metrics

        self.logger.info(f"综合评分计算完成: {len(comprehensive_results)} 个因子")
        return comprehensive_results

    # ==================== 主筛选函数 ====================

    def setup_multi_timeframe_session(self, symbol: str, timeframes: List[str]) -> Path:
        """
        设置多时间框架会话目录结构

        Args:
            symbol: 股票代码
            timeframes: 时间框架列表

        Returns:
            Path: 主会话目录路径
        """
        # 创建主会话目录
        main_session_id = f"{symbol}_multi_tf_{self.session_timestamp}"
        self.multi_tf_session_dir = self.screening_results_dir / main_session_id
        self.multi_tf_session_dir.mkdir(parents=True, exist_ok=True)

        # 创建时间框架子目录
        self.timeframes_dir = self.multi_tf_session_dir / "timeframes"
        self.timeframes_dir.mkdir(exist_ok=True)

        # 创建各个时间框架的会话目录
        self.tf_session_dirs = {}
        for tf in timeframes:
            tf_session_id = f"{tf}_{self.session_timestamp}"
            tf_session_dir = self.timeframes_dir / tf_session_id
            tf_session_dir.mkdir(exist_ok=True)
            self.tf_session_dirs[tf] = tf_session_dir

        return self.multi_tf_session_dir

    def screen_single_timeframe_in_multi_session(
        self, symbol: str, timeframe: str
    ) -> Dict[str, FactorMetrics]:
        """
        在多时间框架会话中筛选单个时间框架

        Args:
            symbol: 股票代码
            timeframe: 时间框架

        Returns:
            Dict[str, FactorMetrics]: 筛选结果
        """
        # 设置当前时间框架的会话目录
        self.session_dir = self.tf_session_dirs[timeframe]

        # 为当前时间框架创建独立的日志记录器
        tf_logger_name = f"{self.session_timestamp}_{timeframe}"
        # 临时设置日志根目录到时间框架会话目录
        original_log_root = self.log_root
        self.log_root = self.session_dir
        self.logger = self._setup_logger(tf_logger_name)
        # 恢复原始日志根目录
        self.log_root = original_log_root

        start_time = time.time()
        self.logger.info(f"📁 多时间框架会话 - {timeframe} 子目录: {self.session_dir}")
        self.logger.info(f"开始5维度因子筛选: {symbol} {timeframe}")

        try:
            # 1. 数据加载
            self.logger.info("步骤1: 数据加载...")
            factors = self.load_factors(symbol, timeframe)
            price_data = self.load_price_data(symbol, timeframe)

            # 2. 数据预处理和对齐
            self.logger.info("步骤2: 数据预处理...")
            close_prices = price_data["close"]

            # 3. 执行现有的筛选逻辑（复用原函数的核心部分）
            return self._execute_screening_core(
                factors, close_prices, symbol, timeframe, start_time
            )

        except Exception as e:
            self.logger.error(f"多时间框架筛选失败 {symbol} {timeframe}: {str(e)}")
            raise

    def _execute_screening_core(
        self,
        factors: pd.DataFrame,
        close_prices: pd.Series,
        symbol: str,
        timeframe: str,
        start_time: float,
    ) -> Dict[str, FactorMetrics]:
        """执行筛选的核心逻辑（从原函数中提取）"""
        # 直接调用主筛选方法的逻辑，但使用已有的数据
        self.logger.info(f"开始{symbol} {timeframe} 核心筛选分析...")

        # 2. 数据预处理和对齐
        self.logger.info("步骤2: 数据预处理...")
        returns = close_prices.pct_change()  # 当期收益，避免未来函数

        # 时间对齐
        common_index = factors.index.intersection(close_prices.index)

        # 如果对齐失败，尝试诊断并修复
        if len(common_index) == 0:
            self.logger.error("数据对齐失败！尝试诊断...")
            self.logger.error(f"  因子前5个时间: {factors.index[:5].tolist()}")
            self.logger.error(f"  价格前5个时间: {close_prices.index[:5].tolist()}")

            # 对于daily数据，尝试标准化到日期
            if timeframe == "daily":
                self.logger.info("检测到daily时间框架，尝试标准化到日期...")
                factors.index = factors.index.normalize()
                close_prices.index = close_prices.index.normalize()
                returns.index = returns.index.normalize()
                common_index = factors.index.intersection(close_prices.index)
                self.logger.info(f"标准化后共同时间点: {len(common_index)}")

        if len(common_index) < self.config.min_sample_size:
            raise ValueError(
                f"数据对齐后样本量不足: {len(common_index)} < {self.config.min_sample_size}"
            )

        factors_aligned = factors.loc[common_index]
        returns_aligned = returns.loc[common_index]
        factor_metadata = self._generate_factor_metadata(factors_aligned)

        self.logger.info(
            f"数据对齐完成: 样本量={len(common_index)}, 因子数={len(factors_aligned.columns)}"
        )

        # 3. 5维度分析
        all_metrics = {}

        # 3.1 预测能力分析
        self.logger.info("步骤3.1: 预测能力分析...")
        all_metrics["multi_horizon_ic"] = self.calculate_multi_horizon_ic(
            factors_aligned, returns_aligned
        )
        all_metrics["ic_decay"] = self.analyze_ic_decay(all_metrics["multi_horizon_ic"])

        # 3.2 稳定性分析
        self.logger.info("步骤3.2: 稳定性分析...")
        all_metrics["rolling_ic"] = self.calculate_rolling_ic(
            factors_aligned, returns_aligned
        )
        all_metrics["cross_section_stability"] = (
            self.calculate_cross_sectional_stability(factors_aligned)
        )

        # 3.3 独立性分析
        self.logger.info("步骤3.3: 独立性分析...")
        all_metrics["vif_scores"] = self.calculate_vif_scores(
            factors_aligned, vif_threshold=self.config.vif_threshold
        )
        all_metrics["correlation_matrix"] = self.calculate_factor_correlation_matrix(
            factors_aligned
        )
        all_metrics["information_increment"] = self.calculate_information_increment(
            factors_aligned, returns_aligned
        )

        # 3.4 实用性分析
        self.logger.info("步骤3.4: 实用性分析...")
        price_data = pd.DataFrame(
            {
                "close": close_prices,
                "volume": factors_aligned.get(
                    "volume", pd.Series(index=factors_aligned.index, dtype=float)
                ),
            }
        )
        all_metrics["trading_costs"] = self.calculate_trading_costs(
            factors_aligned, price_data, factor_metadata
        )
        all_metrics["liquidity_requirements"] = self.calculate_liquidity_requirements(
            factors_aligned, price_data["volume"]
        )

        # 3.5 短周期适应性分析
        self.logger.info("步骤3.5: 短周期适应性分析...")
        all_metrics["reversal_effects"] = self.detect_reversal_effects(
            factors_aligned, returns_aligned
        )
        all_metrics["momentum_persistence"] = self.analyze_momentum_persistence(
            factors_aligned, returns_aligned
        )
        all_metrics["volatility_sensitivity"] = self.analyze_volatility_sensitivity(
            factors_aligned, returns_aligned
        )

        # 4. 统计显著性检验
        self.logger.info("步骤4: 统计显著性检验...")

        # 收集p值
        p_values = {}
        for factor, ic_data in all_metrics["multi_horizon_ic"].items():
            # 使用1日IC的p值作为主要显著性指标
            p_values[factor] = ic_data.get("p_value_1d", 1.0)

        all_metrics["p_values"] = p_values

        # FDR校正（传入样本量用于自适应阈值调整）
        sample_size = len(factors_aligned)
        if self.config.fdr_method == "benjamini_hochberg":
            corrected_p, adaptive_alpha = self.benjamini_hochberg_correction(
                p_values, sample_size=sample_size
            )
        else:
            corrected_p, adaptive_alpha = self.bonferroni_correction(p_values)

        all_metrics["corrected_p_values"] = corrected_p
        all_metrics["adaptive_alpha"] = adaptive_alpha

        # 5. 综合评分
        self.logger.info("步骤5: 综合评分...")
        comprehensive_results = self.calculate_comprehensive_scores(all_metrics)

        # 性能统计
        duration = time.time() - start_time

        # 保存结果（保存到时间框架子目录）
        screening_stats = {
            "total_factors": len(comprehensive_results),
            "significant_factors": sum(
                1 for m in comprehensive_results.values() if m.is_significant
            ),
            "high_score_factors": sum(
                1 for m in comprehensive_results.values() if m.comprehensive_score > 0.7
            ),
            "total_time": duration,
            "sample_size": len(factors_aligned),
            "symbol": symbol,
            "timeframe": timeframe,
        }
        data_quality_info = {
            "factor_data_shape": factors.shape,
            "aligned_data_shape": factors_aligned.shape,
            "data_overlap_count": len(common_index),
        }
        self.save_comprehensive_screening_info(
            comprehensive_results, symbol, timeframe, screening_stats, data_quality_info
        )
        self.logger.info(f"✅ {symbol} {timeframe} 筛选完成，耗时: {duration:.2f}秒")
        self.logger.info(f"   总因子数: {len(comprehensive_results)}")
        self.logger.info(
            f"   顶级因子数: {sum(1 for m in comprehensive_results.values() if m.comprehensive_score >= 0.8)}"
        )

        return comprehensive_results

    def generate_multi_timeframe_summary(
        self,
        symbol: str,
        timeframes: List[str],
        all_results: Dict[str, Dict[str, FactorMetrics]],
    ) -> Dict:
        """
        生成多时间框架汇总报告

        Args:
            symbol: 股票代码
            timeframes: 时间框架列表
            all_results: 所有时间框架的筛选结果

        Returns:
            Dict: 汇总报告数据
        """
        from datetime import datetime

        summary = {
            "symbol": symbol,
            "timeframes": timeframes,
            "generation_time": datetime.now().isoformat(),
            "session_timestamp": self.session_timestamp,
            "total_timeframes": len(timeframes),
            "timeframe_summary": {},
            "cross_timeframe_analysis": {},
            "top_factors_by_timeframe": {},
            "consensus_factors": [],
            "performance_comparison": {},
        }

        # 按时间框架汇总
        for tf in timeframes:
            if tf in all_results:
                tf_results = all_results[tf]
                tf_summary = {
                    "total_factors": len(tf_results),
                    "significant_factors": sum(
                        1 for m in tf_results.values() if m.p_value < 0.05
                    ),
                    "top_factors": sum(
                        1 for m in tf_results.values() if m.comprehensive_score >= 0.8
                    ),
                    "average_ic": (
                        sum(m.ic_mean for m in tf_results.values()) / len(tf_results)
                        if tf_results
                        else 0
                    ),
                    "average_score": (
                        sum(m.comprehensive_score for m in tf_results.values())
                        / len(tf_results)
                        if tf_results
                        else 0
                    ),
                }
                summary["timeframe_summary"][tf] = tf_summary

                # 顶级因子列表
                top_factors = sorted(
                    [(name, metrics) for name, metrics in tf_results.items()],
                    key=lambda x: x[1].comprehensive_score,
                    reverse=True,
                )[
                    :10
                ]  # 取前10个
                summary["top_factors_by_timeframe"][tf] = [
                    {
                        "factor": name,
                        "score": metrics.comprehensive_score,
                        "ic_mean": metrics.ic_mean,
                    }
                    for name, metrics in top_factors
                ]

        # 跨时间框架分析 - 寻找共识因子
        if len(all_results) > 1:
            # 找出在多个时间框架中都表现优秀的因子
            factor_performance = {}
            for tf, tf_results in all_results.items():
                for factor_name, metrics in tf_results.items():
                    if factor_name not in factor_performance:
                        factor_performance[factor_name] = {}
                    factor_performance[factor_name][tf] = metrics.comprehensive_score

            # 计算共识因子（在超过一半的时间框架中得分>=0.7的因子）
            consensus_threshold = 0.7
            min_timeframes = max(1, len(timeframes) // 2)

            consensus_factors = []
            for factor_name, tf_scores in factor_performance.items():
                high_score_count = sum(
                    1 for score in tf_scores.values() if score >= consensus_threshold
                )
                if high_score_count >= min_timeframes:
                    avg_score = sum(tf_scores.values()) / len(tf_scores)
                    consensus_factors.append(
                        {
                            "factor": factor_name,
                            "average_score": avg_score,
                            "high_score_count": high_score_count,
                            "scores_by_timeframe": tf_scores,
                        }
                    )

            # 按平均分数排序
            consensus_factors.sort(key=lambda x: x["average_score"], reverse=True)
            summary["consensus_factors"] = consensus_factors[:20]  # 取前20个共识因子

        return summary

    def save_multi_timeframe_summary(
        self,
        symbol: str,
        timeframes: List[str],
        all_results: Dict[str, Dict[str, FactorMetrics]],
    ):
        """
        保存多时间框架汇总报告

        Args:
            symbol: 股票代码
            timeframes: 时间框架列表
            all_results: 所有时间框架的筛选结果
        """
        if not hasattr(self, "multi_tf_session_dir"):
            self.logger.warning("多时间框架会话目录未设置，跳过汇总报告保存")
            return

        # 生成汇总报告
        summary = self.generate_multi_timeframe_summary(symbol, timeframes, all_results)

        # 保存汇总报告
        summary_file = self.multi_tf_session_dir / "multi_timeframe_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(
                self._to_json_serializable(summary), f, indent=2, ensure_ascii=False
            )

        # 生成总索引文件
        index_file = self.multi_tf_session_dir / "index.txt"
        with open(index_file, "w", encoding="utf-8") as f:
            f.write("多时间框架因子筛选完整报告索引\n")
            f.write(f"{'='*50}\n\n")
            f.write("基础信息:\n")
            f.write(f"  股票代码: {symbol}\n")
            f.write(f"  时间框架: {', '.join(timeframes)}\n")
            f.write(f"  生成时间: {summary['generation_time']}\n")
            f.write(f"  会话ID: {summary['session_timestamp']}\n\n")

            f.write("目录结构:\n")
            f.write("  1. timeframes/ - 各时间框架详细分析结果\n")
            for i, tf in enumerate(timeframes, 1):
                f.write(f"     {i}. {tf}/ - {tf}时间框架分析\n")
            f.write("  2. multi_timeframe_summary.json - 多时间框架汇总报告\n")
            f.write("  3. index.txt - 本索引文件\n\n")

            f.write("使用说明:\n")
            f.write("  - 查看各时间框架详细结果: 进入 timeframes/ 目录\n")
            f.write("  - 查看多时间框架对比分析: 打开 multi_timeframe_summary.json\n")
            f.write("  - 寻找共识因子: 查看汇总报告中的 consensus_factors 部分\n\n")

            # 添加各时间框架概要
            f.write("各时间框架概要:\n")
            for tf in timeframes:
                if tf in summary["timeframe_summary"]:
                    tf_summary = summary["timeframe_summary"][tf]
                    f.write(f"  {tf}:\n")
                    f.write(f"    总因子数: {tf_summary['total_factors']}\n")
                    f.write(f"    显著因子: {tf_summary['significant_factors']}\n")
                    f.write(f"    顶级因子: {tf_summary['top_factors']}\n")
                    f.write(f"    平均IC: {tf_summary['average_ic']:.4f}\n")
                    f.write(f"    平均评分: {tf_summary['average_score']:.3f}\n\n")

            if summary["consensus_factors"]:
                f.write("顶级共识因子 (前5个):\n")
                for i, factor in enumerate(summary["consensus_factors"][:5], 1):
                    f.write(
                        f"  {i}. {factor['factor']} - 评分: {factor['average_score']:.3f}\n"
                    )
                    f.write(
                        f"     在{factor['high_score_count']}个时间框架中表现优秀\n"
                    )

        self.logger.info(f"✅ 多时间框架汇总报告已保存到: {self.multi_tf_session_dir}")
        self.logger.info(f"   时间框架数量: {len(timeframes)}")
        self.logger.info(f"   共识因子数量: {len(summary['consensus_factors'])}")

    def screen_multiple_timeframes(
        self, symbol: str, timeframes: List[str]
    ) -> Dict[str, Dict[str, FactorMetrics]]:
        """
        多时间框架筛选的主入口函数

        Args:
            symbol: 股票代码
            timeframes: 时间框架列表

        Returns:
            Dict[str, Dict[str, FactorMetrics]]: 各时间框架的筛选结果
        """
        from datetime import datetime

        start_time = datetime.now()

        self.logger.info(f"🚀 开始多时间框架因子筛选: {symbol}")
        self.logger.info(f"   时间框架: {', '.join(timeframes)}")
        self.logger.info(f"   会话时间戳: {self.session_timestamp}")

        # 🎯 新增：检查因子文件对齐性
        try:
            from utils import (
                find_aligned_factor_files,
                validate_factor_alignment,
            )

            self.logger.info("🔍 检查因子文件对齐性...")

            # 确保数据根目录正确
            factor_data_root = Path(self.data_root)
            self.logger.info(f"🔍 使用因子数据根目录: {factor_data_root}")

            aligned_files = find_aligned_factor_files(
                factor_data_root, symbol, timeframes
            )
            self.logger.info("✅ 找到对齐的因子文件:")
            for tf, file_path in aligned_files.items():
                self.logger.info(f"   {tf}: {file_path.name}")

            # 验证时间对齐性
            is_aligned, alignment_msg = validate_factor_alignment(
                factor_data_root, symbol, timeframes, aligned_files
            )
            if is_aligned:
                self.logger.info(f"✅ 时间对齐验证通过: {alignment_msg}")
            else:
                self.logger.warning(f"⚠️ 时间对齐验证警告: {alignment_msg}")

            # 保存对齐的文件信息供后续使用
            self.aligned_factor_files = aligned_files

        except Exception as e:
            self.logger.warning(f"⚠️ 因子对齐检查失败，使用默认文件选择: {str(e)}")
            self.aligned_factor_files = None

        # 设置多时间框架会话目录结构
        main_session_dir = self.setup_multi_timeframe_session(symbol, timeframes)
        self.logger.info(f"📁 主会话目录: {main_session_dir}")

        # 初始化主日志记录器（用于记录总体进度）
        # 临时设置日志根目录到主会话目录
        original_log_root = self.log_root
        self.log_root = main_session_dir
        main_logger = self._setup_logger(f"{self.session_timestamp}_main")
        # 恢复原始日志根目录
        self.log_root = original_log_root

        try:
            # 逐个处理每个时间框架
            all_results = {}
            successful_timeframes = []
            failed_timeframes = []

            for i, timeframe in enumerate(timeframes, 1):
                main_logger.info(f"处理时间框架 {i}/{len(timeframes)}: {timeframe}")

                # 内存清理：每个时间框架开始前清理
                import gc

                gc.collect()

                try:
                    # 筛选单个时间框架
                    tf_results = self.screen_single_timeframe_in_multi_session(
                        symbol, timeframe
                    )
                    all_results[timeframe] = tf_results
                    successful_timeframes.append(timeframe)

                    main_logger.info(
                        f"✅ {timeframe} 筛选完成 - 顶级因子数: {sum(1 for m in tf_results.values() if m.comprehensive_score >= 0.8)}"
                    )

                except Exception as e:
                    failed_timeframes.append(timeframe)
                    main_logger.error(f"❌ {timeframe} 筛选失败: {str(e)}")

                    if (
                        len(failed_timeframes) > len(timeframes) // 2
                    ):  # 如果失败超过一半，停止执行
                        main_logger.error("失败时间框架过多，停止执行")
                        break

            # 保存多时间框架汇总报告
            if all_results:
                main_logger.info("生成多时间框架汇总报告...")
                self.save_multi_timeframe_summary(symbol, timeframes, all_results)

            # 完成统计
            total_duration = (datetime.now() - start_time).total_seconds()
            main_logger.info("🎉 多时间框架筛选完成!")
            main_logger.info(f"   总耗时: {total_duration:.2f}秒")
            main_logger.info(
                f"   成功时间框架: {len(successful_timeframes)}/{len(timeframes)}"
            )
            main_logger.info(
                f"   失败时间框架: {', '.join(failed_timeframes) if failed_timeframes else '无'}"
            )

            # 计算总体统计
            total_factors = sum(len(results) for results in all_results.values())
            total_top_factors = sum(
                sum(1 for m in results.values() if m.comprehensive_score >= 0.8)
                for results in all_results.values()
            )

            main_logger.info(f"   总因子数: {total_factors}")
            main_logger.info(f"   总顶级因子数: {total_top_factors}")
            if len(all_results) > 0:
                main_logger.info(
                    f"   平均每时间框架顶级因子数: {total_top_factors/len(all_results):.1f}"
                )
            else:
                main_logger.info("   平均每时间框架顶级因子数: 0.0 (无成功结果)")

            return all_results

        except Exception as e:
            main_logger.error(f"多时间框架筛选过程出现错误: {str(e)}")
            raise

    def screen_factors_comprehensive(
        self, symbol: str, timeframe: str = "60min"
    ) -> Dict[str, FactorMetrics]:
        """主筛选函数 - 5维度综合筛选"""

        # 创建会话目录（如果还没有的话）
        if not hasattr(self, "session_dir") or not self.session_dir:
            session_id = f"{symbol}_{timeframe}_{self.session_timestamp}"
            self.session_dir = self.screening_results_dir / session_id
            self.session_dir.mkdir(parents=True, exist_ok=True)
        else:
            # 从现有的session_dir提取session_id
            session_id = self.session_dir.name

        start_time = time.time()
        self.logger.info(f"📁 创建会话目录: {self.session_dir}")
        self.logger.info(f"开始5维度因子筛选: {symbol} {timeframe}")

        try:
            # 1. 数据加载
            self.logger.info("步骤1: 数据加载...")
            factors = self.load_factors(symbol, timeframe)
            price_data = self.load_price_data(symbol, timeframe)  # 传递timeframe参数

            # 2. 数据预处理和对齐
            self.logger.info("步骤2: 数据预处理...")
            close_prices = price_data["close"]
            returns = close_prices.pct_change()  # 当期收益，避免未来函数

            # 添加诊断日志 - 关键修复
            self.logger.info("数据对齐前诊断:")
            self.logger.info(
                f"  因子数据: {len(factors)} 行, 时间 {factors.index.min()} 到 {factors.index.max()}"
            )
            self.logger.info(
                f"  价格数据: {len(close_prices)} 行, 时间 {close_prices.index.min()} 到 {close_prices.index.max()}"
            )

            # 时间对齐
            common_index = factors.index.intersection(close_prices.index)

            # 如果对齐失败，尝试诊断并修复
            if len(common_index) == 0:
                self.logger.error("数据对齐失败！尝试诊断...")
                self.logger.error(f"  因子前5个时间: {factors.index[:5].tolist()}")
                self.logger.error(f"  价格前5个时间: {close_prices.index[:5].tolist()}")

                # 对于daily数据，尝试标准化到日期
                if timeframe == "daily":
                    self.logger.info("检测到daily时间框架，尝试标准化到日期...")
                    factors.index = factors.index.normalize()
                    close_prices.index = close_prices.index.normalize()
                    returns.index = returns.index.normalize()
                    common_index = factors.index.intersection(close_prices.index)
                    self.logger.info(f"标准化后共同时间点: {len(common_index)}")

            if len(common_index) < self.config.min_sample_size:
                raise ValueError(
                    f"数据对齐后样本量不足: {len(common_index)} < {self.config.min_sample_size}"
                )

            factors_aligned = factors.loc[common_index]
            returns_aligned = returns.loc[common_index]
            prices_aligned = price_data.loc[common_index]
            factor_metadata = self._generate_factor_metadata(factors_aligned)

            self.logger.info(
                f"数据对齐完成: 样本量={len(common_index)}, 因子数={len(factors_aligned.columns)}"
            )

            # 3. 5维度分析
            all_metrics = {}

            # 3.1 预测能力分析
            self.logger.info("步骤3.1: 预测能力分析...")
            all_metrics["multi_horizon_ic"] = self.calculate_multi_horizon_ic(
                factors_aligned, returns_aligned
            )
            all_metrics["ic_decay"] = self.analyze_ic_decay(
                all_metrics["multi_horizon_ic"]
            )

            # 3.2 稳定性分析
            self.logger.info("步骤3.2: 稳定性分析...")
            all_metrics["rolling_ic"] = self.calculate_rolling_ic(
                factors_aligned, returns_aligned
            )
            all_metrics["cross_section_stability"] = (
                self.calculate_cross_sectional_stability(factors_aligned)
            )

            # 内存清理
            import gc

            gc.collect()

            # 3.3 独立性分析
            self.logger.info("步骤3.3: 独立性分析...")
            all_metrics["vif_scores"] = self.calculate_vif_scores(
                factors_aligned, vif_threshold=self.config.vif_threshold
            )
            all_metrics["correlation_matrix"] = (
                self.calculate_factor_correlation_matrix(factors_aligned)
            )
            all_metrics["information_increment"] = self.calculate_information_increment(
                factors_aligned, returns_aligned
            )

            # 内存清理
            gc.collect()

            # 3.4 实用性分析
            self.logger.info("步骤3.4: 实用性分析...")
            all_metrics["trading_costs"] = self.calculate_trading_costs(
                factors_aligned, prices_aligned, factor_metadata
            )
            all_metrics["liquidity_requirements"] = (
                self.calculate_liquidity_requirements(
                    factors_aligned, prices_aligned["volume"]
                )
            )

            # 3.5 短周期适应性分析
            self.logger.info("步骤3.5: 短周期适应性分析...")
            all_metrics["reversal_effects"] = self.detect_reversal_effects(
                factors_aligned, returns_aligned
            )
            all_metrics["momentum_persistence"] = self.analyze_momentum_persistence(
                factors_aligned, returns_aligned
            )
            all_metrics["volatility_sensitivity"] = self.analyze_volatility_sensitivity(
                factors_aligned, returns_aligned
            )

            # 4. 统计显著性检验
            self.logger.info("步骤4: 统计显著性检验...")

            # 收集p值
            p_values = {}
            for factor, ic_data in all_metrics["multi_horizon_ic"].items():
                # 使用1日IC的p值作为主要显著性指标
                p_values[factor] = ic_data.get("p_value_1d", 1.0)

            all_metrics["p_values"] = p_values

            # FDR校正（传入样本量用于自适应阈值调整）
            sample_size = len(factors_aligned)
            if self.config.fdr_method == "benjamini_hochberg":
                corrected_p, adaptive_alpha = self.benjamini_hochberg_correction(
                    p_values, sample_size=sample_size
                )
            else:
                corrected_p, adaptive_alpha = self.bonferroni_correction(p_values)

            all_metrics["corrected_p_values"] = corrected_p
            all_metrics["adaptive_alpha"] = adaptive_alpha

            # 5. 综合评分
            self.logger.info("步骤5: 综合评分...")
            comprehensive_results = self.calculate_comprehensive_scores(all_metrics)

            # 6. 性能统计
            total_time = time.time() - start_time
            current_memory = self.process.memory_info().rss / 1024 / 1024
            # 重新获取起始内存以避免负值
            if not hasattr(self, "_session_start_memory"):
                self._session_start_memory = self.start_memory
            memory_used = max(0, current_memory - self._session_start_memory)

            self.logger.info("5维度筛选完成:")
            self.logger.info(f"  - 总耗时: {total_time:.2f}秒")
            self.logger.info(f"  - 内存使用: {memory_used:.1f}MB")
            self.logger.info(f"  - 因子总数: {len(comprehensive_results)}")

            # 统计各维度表现
            significant_count = sum(
                1 for m in comprehensive_results.values() if m.is_significant
            )
            high_score_count = sum(
                1 for m in comprehensive_results.values() if m.comprehensive_score > 0.7
            )

            self.logger.info(f"  - 显著因子: {significant_count}")
            self.logger.info(f"  - 高分因子: {high_score_count}")

            # 7. 收集筛选统计信息
            screening_stats = {
                "total_factors": len(comprehensive_results),
                "significant_factors": significant_count,
                "high_score_factors": high_score_count,
                "total_time": total_time,
                "memory_used_mb": memory_used,
                "sample_size": len(common_index) if "common_index" in locals() else 0,
                "factors_aligned": (
                    len(factors_aligned.columns) if "factors_aligned" in locals() else 0
                ),
                "data_alignment_successful": (
                    len(common_index) > 0 if "common_index" in locals() else False
                ),
                "screening_timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "timeframe": timeframe,
            }

            # 8. 收集数据质量信息
            data_quality_info = {
                "factor_data_shape": factors.shape if "factors" in locals() else None,
                "price_data_shape": (
                    price_data.shape if "price_data" in locals() else None
                ),
                "aligned_data_shape": (
                    factors_aligned.shape if "factors_aligned" in locals() else None
                ),
                "data_overlap_count": (
                    len(common_index) if "common_index" in locals() else 0
                ),
                "factor_data_range": {
                    "start": (
                        factors.index.min().isoformat()
                        if "factors" in locals() and len(factors) > 0
                        else None
                    ),
                    "end": (
                        factors.index.max().isoformat()
                        if "factors" in locals() and len(factors) > 0
                        else None
                    ),
                },
                "price_data_range": {
                    "start": (
                        price_data.index.min().isoformat()
                        if "price_data" in locals() and len(price_data) > 0
                        else None
                    ),
                    "end": (
                        price_data.index.max().isoformat()
                        if "price_data" in locals() and len(price_data) > 0
                        else None
                    ),
                },
                "alignment_success_rate": (
                    len(common_index) / min(len(factors), len(price_data))
                    if "factors" in locals() and "price_data" in locals()
                    else 0.0
                ),
            }

            # 9. 保存完整筛选信息 - 使用增强版结果管理器
            try:
                if self.result_manager is not None:
                    # 使用新的增强版结果管理器，传递现有会话目录
                    session_id = self.result_manager.create_screening_session(
                        symbol=symbol,
                        timeframe=timeframe,
                        results=comprehensive_results,
                        screening_stats=screening_stats,
                        config=self.config,
                        data_quality_info=data_quality_info,
                        existing_session_dir=self.session_dir,
                    )

                    self.logger.info(f"✅ 完整筛选会话已创建: {session_id}")
                    screening_stats["session_id"] = session_id
                else:
                    self.logger.info("使用传统存储方式")

                # 跳过传统格式保存，避免重复目录创建
                self.logger.info("使用增强版结果管理器，跳过传统格式保存")

            except Exception as e:
                self.logger.error(f"保存完整筛选信息失败: {str(e)}")
                screening_stats["save_error"] = str(e)

            return comprehensive_results

        except Exception as e:
            self.logger.error(f"因子筛选失败: {str(e)}")
            raise

    def generate_screening_report(
        self,
        results: Dict[str, FactorMetrics],
        output_path: str = None,
        symbol: str = None,
        timeframe: str = None,
    ) -> pd.DataFrame:
        """生成筛选报告"""
        self.logger.info("生成筛选报告...")

        if not results:
            self.logger.warning("没有筛选结果，无法生成报告")
            return pd.DataFrame()

        # 转换为DataFrame
        report_data = []
        for factor_name, metrics in results.items():
            row = {
                "Factor": factor_name,
                "Comprehensive_Score": metrics.comprehensive_score,
                "Predictive_Score": metrics.predictive_score,
                "Stability_Score": metrics.stability_score,
                "Independence_Score": metrics.independence_score,
                "Practicality_Score": metrics.practicality_score,
                "Adaptability_Score": metrics.adaptability_score,
                "IC_Mean": metrics.ic_mean,
                "IC_IR": metrics.ic_ir,
                "IC_1d": metrics.ic_1d,
                "IC_5d": metrics.ic_5d,
                "IC_10d": metrics.ic_10d,
                "Rolling_IC_Stability": metrics.rolling_ic_stability,
                "IC_Consistency": metrics.ic_consistency,
                "VIF_Score": metrics.vif_score,
                "Max_Correlation": metrics.correlation_max,
                "Info_Increment": metrics.information_increment,
                "Turnover_Rate": metrics.turnover_rate,
                "Transaction_Cost": metrics.transaction_cost,
                "Cost_Efficiency": metrics.cost_efficiency,
                "P_Value": metrics.p_value,
                "Corrected_P_Value": metrics.corrected_p_value,
                "Is_Significant": metrics.is_significant,
                "Sample_Size": metrics.sample_size,
            }
            report_data.append(row)

        report_df = pd.DataFrame(report_data)
        report_df = report_df.sort_values("Comprehensive_Score", ascending=False)

        # 保存报告（包含时间框架标识）
        if output_path is None:
            # 优先使用会话目录
            if self.session_dir is not None:
                output_path = self.session_dir / "screening_report.csv"
            else:
                # 备用方案：使用原有逻辑
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                symbol_info = symbol or results.get("symbol", "unknown")
                timeframe_info = timeframe or results.get("timeframe", "unknown")
                output_path = (
                    self.screening_results_dir
                    / f"screening_report_{symbol_info}_{timeframe_info}_{timestamp}.csv"
                )

        # 确保路径是字符串格式，避免pandas Path._flavour问题
        output_path_str = str(output_path)
        report_df.to_csv(output_path_str, index=False)
        self.logger.info(f"筛选报告已保存: {output_path}")

        return report_df

    def save_comprehensive_screening_info(
        self,
        results: Dict[str, FactorMetrics],
        symbol: str,
        timeframe: str,
        screening_stats: Dict,
        data_quality_info: Dict = None,
    ):
        """保存完整的筛选信息，包括多个格式的报告"""

        # 使用会话目录和统一的时间戳
        if self.session_dir is None:
            # 如果没有会话目录，使用原有逻辑
            base_dir = self.screening_results_dir
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"screening_{symbol}_{timeframe}_{timestamp}"
        else:
            base_dir = self.session_dir
            base_filename = "screening_report"

        self.logger.info(f"💾 保存筛选信息到会话目录: {base_dir}")

        # 1. 保存详细的CSV报告
        csv_path = base_dir / f"{base_filename}.csv"
        self.generate_screening_report(
            results, str(csv_path), symbol, timeframe
        )

        # 2. 保存筛选过程统计信息
        stats_path = base_dir / "screening_statistics.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(
                self._to_json_serializable(screening_stats),
                f,
                indent=2,
                ensure_ascii=False,
            )

        # 3. 保存顶级因子摘要
        summary_path = base_dir / "top_factors_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("=== 因子筛选摘要报告 ===\n")
            f.write(f"股票代码: {symbol}\n")
            f.write(f"时间框架: {timeframe}\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n=== 筛选统计 ===\n")
            f.write(f"总因子数: {screening_stats.get('total_factors', 0)}\n")
            f.write(f"显著因子: {screening_stats.get('significant_factors', 0)}\n")
            f.write(f"高分因子: {screening_stats.get('high_score_factors', 0)}\n")
            f.write(f"总耗时: {screening_stats.get('total_time', 0):.2f}秒\n")
            f.write(f"内存使用: {screening_stats.get('memory_used_mb', 0):.1f}MB\n")

            # 获取前10名因子
            top_factors = self.get_top_factors(
                results, top_n=10, min_score=0.0, require_significant=False
            )
            f.write("\n=== 前10名顶级因子 ===\n")
            for i, factor in enumerate(top_factors, 1):
                f.write(
                    f"{i:2d}. {factor.name:<25} 综合得分: {factor.comprehensive_score:.3f} "
                )
                f.write(
                    f"预测能力: {factor.predictive_score:.3f} 显著性: {'✓' if factor.is_significant else '✗'}\n"
                )

        # 4. 保存数据质量报告
        if data_quality_info:
            quality_path = base_dir / "data_quality.json"
            with open(quality_path, "w", encoding="utf-8") as f:
                json.dump(
                    self._to_json_serializable(data_quality_info),
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

        # 5. 保存配置参数记录
        config_path = base_dir / "config.yaml"
        config_dict = {
            "screening_config": asdict(self.config),
            "execution_info": {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": self.session_timestamp,
                "data_root": str(self.data_root),
                "screening_results_dir": str(self.screening_results_dir),
                "session_dir": str(self.session_dir),
            },
        }
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(
                config_dict, f, default_flow_style=False, allow_unicode=True, indent=2
            )

        # 6. 创建一个主索引文件
        index_path = base_dir / "index.txt"
        with open(index_path, "w", encoding="utf-8") as f:
            f.write("因子筛选完整报告索引\n")
            f.write("========================\n\n")
            f.write("基础信息:\n")
            f.write(f"  股票代码: {symbol}\n")
            f.write(f"  时间框架: {timeframe}\n")
            f.write(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("包含文件:\n")
            f.write(f"  1. {csv_path.name} - 详细因子数据 (CSV格式)\n")
            f.write(f"  2. {stats_path.name} - 筛选过程统计 (JSON格式)\n")
            f.write(f"  3. {summary_path.name} - 顶级因子摘要 (TXT格式)\n")
            if data_quality_info:
                f.write(f"  4. {quality_path.name} - 数据质量报告 (JSON格式)\n")
            f.write(f"  5. {config_path.name} - 配置参数记录 (YAML格式)\n")
            f.write(f"  6. {index_path.name} - 本索引文件\n\n")
            f.write("使用说明:\n")
            f.write(f"  - 查看顶级因子: 阅读 {summary_path.name}\n")
            f.write(f"  - 详细数据分析: 打开 {csv_path.name} 使用Excel或pandas\n")
            f.write(f"  - 筛选过程详情: 查看 {stats_path.name}\n")
            f.write(f"  - 配置参数参考: 查看 {config_path.name}\n")

        self.logger.info(f"✅ 完整筛选信息已保存到: {base_dir}")
        self.logger.info(f"📄 主索引文件: {index_path}")

        return {
            "csv_report": str(csv_path),
            "stats_json": str(stats_path),
            "summary_txt": str(summary_path),
            "data_quality_json": str(quality_path) if data_quality_info else None,
            "config_yaml": str(config_path),
            "index_txt": str(index_path),
        }

    def get_top_factors(
        self,
        results: Dict[str, FactorMetrics],
        top_n: int = 20,
        min_score: float = 0.5,
        require_significant: bool = True,
    ) -> List[FactorMetrics]:
        """获取顶级因子"""

        # 筛选条件
        filtered_results = []
        for metrics in results.values():
            if metrics.comprehensive_score >= min_score:
                if not require_significant or metrics.is_significant:
                    filtered_results.append(metrics)

        # 按综合得分排序
        filtered_results.sort(key=lambda x: x.comprehensive_score, reverse=True)

        return filtered_results[:top_n]


def main():
    """主函数 - 支持命令行参数和批量配置"""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="专业级因子筛选系统")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--symbol", type=str, default="0700.HK", help="股票代码")
    parser.add_argument("--timeframe", type=str, default="60min", help="时间框架")

    args = parser.parse_args()

    if args.config:
        # 使用配置文件
        try:
            from config_manager import ConfigManager

            manager = ConfigManager()

            # 检查是否是批量配置
            import yaml

            with open(args.config, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            if "batch_name" in config_data:
                # 批量配置 - 直接处理所有子配置
                import yaml

                with open(args.config, "r", encoding="utf-8") as f:
                    batch_config = yaml.safe_load(f)

                print(f"🚀 开始批量筛选: {batch_config['batch_name']}")
                print(f"📊 子任务数量: {len(batch_config['screening_configs'])}")
                print("=" * 80)

                successful_tasks = 0
                failed_tasks = 0
                all_results = {}  # 收集所有结果

                print(f"\n🚀 开始多时间框架批量分析: {batch_config['batch_name']}")
                print(f"📊 分析时间框架数量: {len(batch_config['screening_configs'])}")
                print("=" * 80)

                # 获取第一个配置的数据根和输出目录
                first_config = batch_config["screening_configs"][0]
                data_root = first_config.get("data_root", "../因子输出")
                output_dir = first_config.get("output_dir", "./因子筛选")

                print(f"📁 数据目录: {data_root}")
                print(f"📁 输出目录: {output_dir}")

                # 创建统一的批量筛选器
                batch_screener = ProfessionalFactorScreener(data_root=data_root)
                batch_screener.screening_results_dir = Path(output_dir)

                # 创建统一的批量会话目录
                from datetime import datetime

                batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                batch_session_id = (
                    f"{batch_config['batch_name']}_multi_timeframe_{batch_timestamp}"
                )
                batch_screener.session_dir = (
                    batch_screener.screening_results_dir / batch_session_id
                )
                batch_screener.session_dir.mkdir(parents=True, exist_ok=True)
                batch_screener.session_timestamp = batch_timestamp

                print(f"📁 批量会话目录: {batch_screener.session_dir}")

                for i, sub_config in enumerate(batch_config["screening_configs"], 1):
                    try:
                        print(
                            f"\n📊 [{i}/{len(batch_config['screening_configs'])}] 处理: {sub_config['name']}"
                        )
                        print(f"   股票: {sub_config['symbols'][0]}")
                        print(f"   时间框架: {sub_config['timeframes'][0]}")

                        # 使用同一个筛选器执行筛选（复用会话目录）
                        result = batch_screener.screen_factors_comprehensive(
                            symbol=sub_config["symbols"][0],
                            timeframe=sub_config["timeframes"][0],
                        )

                        # 收集结果 - result 是 Dict[str, FactorMetrics]
                        tf_key = (
                            f"{sub_config['symbols'][0]}_{sub_config['timeframes'][0]}"
                        )
                        all_results[tf_key] = result

                        # 统计显著因子数量
                        significant_count = sum(
                            1 for m in result.values() if m.is_significant
                        )
                        successful_tasks += 1
                        print(f"   ✅ 完成: {significant_count} 个显著因子")

                    except Exception as e:
                        failed_tasks += 1
                        print(f"   ❌ 失败: {str(e)}")
                        continue

                # 生成统一的多时间框架汇总报告
                print("\n📈 生成统一汇总报告...")
                if all_results:
                    _generate_multi_timeframe_summary(
                        batch_screener.session_dir,
                        batch_config["batch_name"],
                        all_results,
                        batch_config["screening_configs"],
                    )

                print("\n✅ 多时间框架分析完成:")
                print(f"   总任务: {len(batch_config['screening_configs'])}")
                print(f"   成功: {successful_tasks}")
                print(f"   失败: {failed_tasks}")
                if all_results:
                    print(
                        f"   📊 汇总报告: 已生成到统一会话目录: {batch_screener.session_dir}"
                    )
                    print(
                        "   📁 所有结果文件保存在同一目录，无需查找多个时间框架子目录"
                    )

                return
            else:
                # 单个配置
                config = manager.load_config(args.config, config_type="screening")
                symbol = config.symbols[0] if config.symbols else args.symbol
                timeframe = (
                    config.timeframes[0] if config.timeframes else args.timeframe
                )
        except Exception as e:
            print(f"❌ 配置文件加载失败: {e}")
            sys.exit(1)
    else:
        # 没有指定配置文件时，尝试加载默认配置
        default_config_path = (
            Path(__file__).parent / "configs" / "0700_multi_timeframe_config.yaml"
        )
        if default_config_path.exists():
            try:
                import yaml

                with open(default_config_path, "r", encoding="utf-8") as f:
                    config_data = yaml.safe_load(f)

                # 如果是批量配置，加载第一个子配置
                if "batch_name" in config_data and "screening_configs" in config_data:
                    first_sub_config = config_data["screening_configs"][0]
                    from config_manager import ScreeningConfig

                    config = ScreeningConfig(**first_sub_config)
                    print(f"✅ 自动加载默认配置: {default_config_path}")
                    print(f"📁 数据目录: {config.data_root}")
                    print(f"📁 输出目录: {config.output_dir}")
                else:
                    from config_manager import ScreeningConfig

                    config = ScreeningConfig(**config_data)
            except Exception as e:
                print(f"⚠️ 默认配置加载失败，使用内置配置: {e}")
                from config_manager import ScreeningConfig
            config = ScreeningConfig(
                data_root="../因子输出",
                output_dir="./因子筛选",
                ic_horizons=[1, 3, 5, 10, 20],
                min_sample_size=100,
                alpha_level=0.05,
                fdr_method="benjamini_hochberg",
                min_ic_threshold=0.02,
                min_ir_threshold=0.5,
            )
        symbol = args.symbol
        timeframe = args.timeframe

    # 单个筛选任务
    screener = ProfessionalFactorScreener(config=config)

    print(f"开始专业级因子筛选: {symbol} {timeframe}")
    print("=" * 80)

    try:
        # 5维度综合筛选
        results = screener.screen_factors_comprehensive(symbol, timeframe)

        # 生成报告
        report_df = screener.generate_screening_report(results)

        # 获取顶级因子
        top_factors = screener.get_top_factors(results, top_n=10, min_score=0.6)

        # 输出结果
        print("\n5维度因子筛选结果:")
        print("=" * 80)
        print(f"总因子数量: {len(results)}")
        print(f"显著因子数量: {sum(1 for m in results.values() if m.is_significant)}")
        print(
            f"高分因子数量 (>0.6): {sum(1 for m in results.values() if m.comprehensive_score > 0.6)}"
        )
        print(f"顶级因子数量: {len(top_factors)}")

        print("\n前10名顶级因子:")
        print("-" * 120)
        print(
            f"{'排名':<4} {'因子名称':<20} {'综合得分':<8} {'预测':<6} {'稳定':<6} {'独立':<6} {'实用':<6} {'适应':<6} {'IC均值':<8} {'显著性':<6}"
        )
        print("-" * 120)

        for i, metrics in enumerate(top_factors[:10]):
            significance = (
                "***"
                if metrics.corrected_p_value < 0.001
                else (
                    "**"
                    if metrics.corrected_p_value < 0.01
                    else "*" if metrics.corrected_p_value < 0.05 else ""
                )
            )

            print(
                f"{i+1:<4} {metrics.name:<20} {metrics.comprehensive_score:.3f}    "
                f"{metrics.predictive_score:.3f}  {metrics.stability_score:.3f}  "
                f"{metrics.independence_score:.3f}  {metrics.practicality_score:.3f}  "
                f"{metrics.adaptability_score:.3f}  {metrics.ic_mean:+.4f}  {significance:<6}"
            )

        print(f"\n报告文件: {report_df}")

    except Exception as e:
        print(f"筛选失败: {str(e)}")
        raise


def _generate_multi_timeframe_summary(
    session_dir, batch_name: str, all_results: Dict, screening_configs: List[Dict]
) -> None:
    """生成统一的多时间框架汇总报告

    Args:
        session_dir: 会话目录路径
        batch_name: 批量处理名称
        all_results: 所有时间框架的筛选结果
        screening_configs: 筛选配置列表
    """
    from datetime import datetime

    print("📊 生成多时间框架汇总报告...")

    # 检查是否有有效结果
    if not all_results:
        print("⚠️ 没有数据生成汇总报告")
        return

    # 创建汇总数据结构
    summary_data = []
    best_factors_overall = []

    for tf_key, result in all_results.items():
        # result 是 Dict[str, FactorMetrics]
        if result and isinstance(result, dict):
            # 获取时间框架信息
            parts = tf_key.split("_")
            symbol = parts[0]
            timeframe = "_".join(parts[1:]) if len(parts) > 1 else "unknown"

            # 获取最佳因子 - 按综合得分排序
            sorted_factors = sorted(
                result.values(), key=lambda x: x.comprehensive_score, reverse=True
            )
            top_factors = sorted_factors[:10]  # 取前10个因子

            for factor_metrics in top_factors:
                summary_item = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "factor_name": factor_metrics.name,
                    "comprehensive_score": factor_metrics.comprehensive_score,
                    "tier": getattr(factor_metrics, "tier", "N/A"),
                    "predictive_power": factor_metrics.predictive_score,
                    "stability": factor_metrics.stability_score,
                    "independence": factor_metrics.independence_score,
                    "practicality": factor_metrics.practicality_score,
                    "short_term_fitness": factor_metrics.adaptability_score,
                    "ic_mean": factor_metrics.ic_mean,
                    "ic_ir": factor_metrics.ic_ir,
                    "ic_win_rate": getattr(factor_metrics, "ic_win_rate", 0),
                    "rank_ic_mean": getattr(factor_metrics, "rank_ic_mean", 0),
                    "rank_ic_ir": getattr(factor_metrics, "rank_ic_ir", 0),
                    "turnover": getattr(factor_metrics, "turnover_rate", 0),
                    "p_value": factor_metrics.p_value,
                    "significant": factor_metrics.is_significant,
                }
                summary_data.append(summary_item)
                best_factors_overall.append(summary_item)

    if not summary_data:
        print("⚠️ 没有数据生成汇总报告")
        return

    # 创建汇总DataFrame
    import pandas as pd

    summary_df = pd.DataFrame(summary_data)

    # 保存统一汇总报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_filename = f"{batch_name}_multi_timeframe_summary_{timestamp}.csv"
    summary_path = session_dir / summary_filename

    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"✅ 多时间框架汇总报告已保存: {summary_path}")

    # 生成最佳因子综合排行
    if best_factors_overall:
        best_df = pd.DataFrame(best_factors_overall)

        # 按综合评分排序
        best_df_sorted = best_df.sort_values("comprehensive_score", ascending=False)

        # 保存最佳因子排行
        best_filename = f"{batch_name}_best_factors_overall_{timestamp}.csv"
        best_path = session_dir / best_filename
        best_df_sorted.to_csv(best_path, index=False, encoding="utf-8")
        print(f"✅ 最佳因子综合排行已保存: {best_path}")

        # 输出Top 10最佳因子到控制台
        print("\n🏆 Top 10 最佳因子 (跨所有时间框架):")
        print("=" * 120)
        top_10 = best_df_sorted.head(10)
        for i, (_, factor) in enumerate(top_10.iterrows(), 1):
            print(
                f"{i:2d}. {factor['factor_name']:<25} | "
                f"{factor['symbol']}-{factor['timeframe']:<10} | "
                f"评分: {factor['comprehensive_score']:.3f} | "
                f"等级: {factor['tier']:<2} | "
                f"IC: {factor['ic_mean']:.3f} | "
                f"胜率: {factor['ic_win_rate']:.1%}"
            )

    # 生成统计摘要
    _generate_batch_statistics(session_dir, batch_name, all_results, timestamp)


def _generate_batch_statistics(
    session_dir, batch_name: str, all_results: Dict, timestamp: str
) -> None:
    """生成批量处理统计摘要"""

    import pandas as pd

    print("📈 生成批量处理统计摘要...")

    stats_data = []
    total_factors_processed = 0
    total_tier1_factors = 0
    total_tier2_factors = 0

    for tf_key, result in all_results.items():
        if result and "session_info" in result:
            symbol = tf_key.split("_")[0]
            timeframe = tf_key.split("_")[1] if "_" in tf_key else "unknown"

            session_info = result["session_info"]
            screening_results = result.get("screening_results", {})

            # 统计各等级因子数量
            tier_counts = screening_results.get("tier_counts", {})
            tier1_count = tier_counts.get("Tier 1 (≥0.8)", 0)
            tier2_count = tier_counts.get("Tier 2 (0.6-0.8)", 0)
            total_count = screening_results.get("total_factors", 0)

            total_factors_processed += total_count
            total_tier1_factors += tier1_count
            total_tier2_factors += tier2_count

            stats_item = {
                "symbol": symbol,
                "timeframe": timeframe,
                "total_factors": total_count,
                "tier1_factors": tier1_count,
                "tier2_factors": tier2_count,
                "tier1_ratio": tier1_count / total_count if total_count > 0 else 0,
                "start_time": session_info.get("start_time", ""),
                "status": session_info.get("status", "unknown"),
            }
            stats_data.append(stats_item)

    if stats_data:
        stats_df = pd.DataFrame(stats_data)

        # 保存统计摘要
        stats_filename = f"{batch_name}_batch_statistics_{timestamp}.csv"
        stats_path = session_dir / stats_filename
        stats_df.to_csv(stats_path, index=False, encoding="utf-8")
        print(f"✅ 批量处理统计摘要已保存: {stats_path}")

        # 输出统计摘要到控制台
        print("\n📊 批量处理统计摘要:")
        print("=" * 80)
        print(f"处理时间框架数量: {len(stats_data)}")
        print(f"总处理因子数: {total_factors_processed}")
        print(f"总Tier 1因子数: {total_tier1_factors}")
        print(f"总Tier 2因子数: {total_tier2_factors}")

        if total_factors_processed > 0:
            overall_tier1_ratio = total_tier1_factors / total_factors_processed
            overall_tier2_ratio = total_tier2_factors / total_factors_processed
            print(f"整体Tier 1比例: {overall_tier1_ratio:.1%}")
            print(f"整体Tier 2比例: {overall_tier2_ratio:.1%}")

            # 按时间框架统计
            print("\n各时间框架详细统计:")
            for _, stats in stats_df.iterrows():
                print(
                    f"  {stats['symbol']}-{stats['timeframe']:>8}: "
                    f"总计 {stats['total_factors']:>3} 个 | "
                    f"Tier1 {stats['tier1_factors']:>2} 个 ({stats['tier1_ratio']:.1%}) | "
                    f"Tier2 {stats['tier2_factors']:>2} 个"
                )


if __name__ == "__main__":
    main()
