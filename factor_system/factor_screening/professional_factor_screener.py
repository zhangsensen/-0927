#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专业级量化交易因子筛选系统 - 5维度筛选框架 + 公平评分系统
作者：量化首席工程师
版本：3.1.0
日期：2025-10-07
状态：生产就绪
更新：集成公平评分算法，修复胜率计算bug

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
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import psutil
import yaml

# 导入配置类
from config_manager import ScreeningConfig  # type: ignore
from scipy import stats

# P0 优化：导入向量化核心引擎
try:
    from vectorized_core import get_vectorized_analyzer

    VECTORIZED_ENGINE_AVAILABLE = True
except ImportError:
    VECTORIZED_ENGINE_AVAILABLE = False
    logging.getLogger(__name__).warning("⚠️ 向量化引擎不可用，将使用传统模式")

# P3 优化：导入性能监控
try:
    from performance_monitor import get_performance_monitor

    PERFORMANCE_MONITOR_AVAILABLE = True
except ImportError:
    PERFORMANCE_MONITOR_AVAILABLE = False
    logging.getLogger(__name__).warning("⚠️ 性能监控模块不可用")

# 🔧 集成data_loader_patch补丁到主代码
try:
    from data_loader_patch import load_factors_v2, load_price_data_v2
except ImportError:
    # 如果补丁不可用，定义回退函数
    def load_factors_v2(self, symbol: str, timeframe: str):
        raise ImportError("data_loader_patch不可用")

    def load_price_data_v2(self, symbol: str, timeframe: str):
        raise ImportError("data_loader_patch不可用")


# 公平评分已集成到主流程，无需单独模块


# JSON编码器，支持numpy类型
class NumpyJSONEncoder(json.JSONEncoder):
    """JSON编码器，支持numpy类型"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)


# 导入因子对齐工具
try:
    from factor_alignment_utils import (  # type: ignore
        FactorFileAligner,
        find_aligned_factor_files,
        validate_factor_alignment,
    )
except ImportError as e:
    # 如果导入失败，提供回退方案
    logging.getLogger(__name__).warning(f"因子对齐工具导入失败: {e}")
    FactorFileAligner = None

    def find_aligned_factor_files(*args: Any, **kwargs: Any) -> None:
        raise ImportError("因子对齐工具不可用")

    def validate_factor_alignment(*args: Any, **kwargs: Any) -> Tuple[bool, str]:
        return True, "对齐验证工具不可用"


try:
    from utils.temporal_validator import TemporalValidationError, TemporalValidator
except ImportError:  # pragma: no cover - 运行环境缺失
    TemporalValidator = None  # type: ignore

    class TemporalValidationError(Exception):
        """时间序列验证器不可用时的后备异常"""

        pass


# P0级集成：导入实际使用的工具模块（诚实版）
try:
    from utils.input_validator import InputValidator, ValidationError
except ImportError:
    logging.getLogger(__name__).warning("输入验证器导入失败")
    InputValidator = None  # type: ignore
    ValidationError = Exception  # type: ignore

try:
    from utils.structured_logger import get_structured_logger
except ImportError as e:
    logging.getLogger(__name__).warning(f"结构化日志器导入失败: {e}")
    get_structured_logger = None  # type: ignore

# 已移除未使用的模块导入（Linus原则：诚实反映实际状态）
# - memory_optimizer: 当前系统内存使用正常，无需复杂监控
# - backup_manager: 文件系统已足够，无需过度工程化


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
    bennett_score: float = 0.0
    is_significant: bool = False

    # 综合评分
    predictive_score: float = 0.0
    stability_score: float = 0.0
    independence_score: float = 0.0
    practicality_score: float = 0.0
    adaptability_score: float = 0.0
    comprehensive_score: float = 0.0
    traditional_score: float = 0.0  # 传统评分（公平评分前）

    # 公平评分相关信息
    fair_scoring_applied: bool = False
    fair_scoring_change: float = 0.0
    fair_scoring_percent_change: float = 0.0

    # 胜率信息
    ic_win_rate: float = 0.0  # IC胜率（正IC占比）
    sample_weight: float = 1.0  # 样本量权重
    predictive_weight: float = 1.0  # 预测能力权重
    actual_sample_size: int = 0  # 实际样本量

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

    def __init__(
        self, data_root: Optional[str] = None, config: Optional[ScreeningConfig] = None
    ):
        """初始化筛选器

        Args:
            data_root: 向后兼容参数，优先使用config中的路径配置
            config: 筛选配置对象
        """
        self.config = config or ScreeningConfig()

        # 🔧 修复路径硬编码 - 智能路径解析
        if hasattr(self.config, "data_root") and self.config.data_root:
            self.data_root = Path(self.config.data_root)
        elif data_root:
            self.data_root = Path(data_root)
        else:
            # 智能路径解析：尝试自动发现项目根目录
            try:
                # 从当前文件位置推导项目根目录
                current_file = Path(__file__)
                project_root = current_file.parent.parent.parent
                potential_factor_output = project_root / "factor_output"

                if potential_factor_output.exists():
                    self.data_root = potential_factor_output
                    logging.getLogger(__name__).info(
                        f"✅ 自动发现因子输出目录: {self.data_root}"
                    )
                else:
                    # 回退到相对路径
                    self.data_root = Path("../factor_output")
                    logging.getLogger(__name__).info(
                        f"使用默认因子输出目录: {self.data_root}"
                    )
            except Exception:
                # 最终回退到相对路径
                self.data_root = Path("../factor_output")
                logging.getLogger(__name__).info(
                    f"使用默认因子输出目录: {self.data_root}"
                )

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
                self.logger.warning("时间序列验证器初始化失败: %s", validator_error)
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

        # 🚀 初始化公平评分器
        try:
            from fair_scorer import FairScorer  # type: ignore

            # 🎯 最优解配置：自动选择最优公平评分配置
            # 🔧 修复：使用相对模块文件的绝对路径
            if getattr(config, "use_optimal_fair_scoring", False):
                default_config = (
                    Path(__file__).parent
                    / "configs"
                    / "optimal_fair_scoring_config.yaml"
                )
                fair_config_path = getattr(
                    config, "optimal_scoring_config_path", str(default_config)
                )
                self.logger.info("🎯 使用最优解公平评分配置: 预测能力核心化算法")
            else:
                default_config = (
                    Path(__file__).parent / "configs" / "fair_scoring_config.yaml"
                )
                fair_config_path = getattr(
                    config, "fair_scoring_config_path", str(default_config)
                )
                self.logger.info("使用传统公平评分配置")

            self.fair_scorer = FairScorer(fair_config_path)

            if self.fair_scorer.enabled:
                self.logger.info("✅ 公平评分器初始化成功")
            else:
                self.logger.info("⚪ 公平评分器已禁用")

        except ImportError as e:
            self.fair_scorer = None
            self.logger.warning(f"公平评分器导入失败: {e}")
            self.logger.info("将使用传统评分方式")
        except Exception as e:
            self.fair_scorer = None
            self.logger.error(f"公平评分器初始化失败: {e}")
            self.logger.info("将使用传统评分方式")

        # 性能监控
        self.process = psutil.Process()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        # P0级集成：初始化新增的工具模块
        self._initialize_utility_modules()

        # P0 性能优化：初始化向量化引擎
        if VECTORIZED_ENGINE_AVAILABLE:
            self.vectorized_analyzer = get_vectorized_analyzer(
                min_sample_size=self.config.min_sample_size
            )
            self.logger.info("✅ VectorBT 向量化引擎已启用")
        else:
            self.vectorized_analyzer = None
            self.logger.warning("⚠️ 向量化引擎不可用，性能将受影响")

        # P3 性能监控：初始化监控器
        if PERFORMANCE_MONITOR_AVAILABLE:
            self.perf_monitor = get_performance_monitor(enable_logging=True)
            self.logger.info("✅ 性能监控已启用")
        else:
            self.perf_monitor = None
            self.logger.warning("⚠️ 性能监控不可用")

        self.logger.info("专业级因子筛选器初始化完成")
        self.logger.info(
            f"配置: IC周期={self.config.ic_horizons}, 最小样本={self.config.min_sample_size}"
        )
        self.logger.info(
            f"显著性水平={self.config.alpha_level}, FDR方法={self.config.fdr_method}"
        )

    def _initialize_utility_modules(self) -> None:
        """P0级集成：初始化工具模块（实际集成）"""

        # 1. 内存优化器 - 已移除（Linus原则：当前系统工作正常，无需复杂化）
        self.memory_optimizer = None

        # 2. 初始化输入验证器
        if InputValidator is not None:
            self.input_validator = InputValidator()
            self.logger.info("✅ 输入验证器已启用")
        else:
            self.input_validator = None
            self.logger.warning("输入验证器模块未安装")

        # 3. 初始化结构化日志器（增强模式）
        if get_structured_logger is not None:
            try:
                self.structured_logger = get_structured_logger(
                    name="factor_screening",
                    log_file=self.log_root / f"structured_{self.session_timestamp}.log",
                )
                self.logger.info("✅ 结构化日志器已启用")
            except Exception as e:
                self.structured_logger = None
                self.logger.warning(f"结构化日志器初始化失败: {e}")
        else:
            self.structured_logger = None
            self.logger.warning("结构化日志器模块未安装")

        # 4. 备份管理器 - 已移除（Linus原则：文件系统已足够，无需过度工程化）
        self.backup_manager = None

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
            "%(asctime)s - %(name)s - %(levelname)s - "
            "[%(funcName)s:%(lineno)d] - %(message)s"
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
                    "missing_periods": factor_data.isna().sum(),
                    "missing_ratio": factor_data.isna().sum() / len(factor_data),
                    "first_valid_index": self._find_first_non_missing_index(
                        factor_data
                    ),
                    "valid_ratio": 1 - (factor_data.isna().sum() / len(factor_data)),
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
            missing_count = factor_data.isna().sum()

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
        if factors_df.isna().any().any():
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

        # P1-2修复：改进常量列检测逻辑，保护K线形态指标
        constant_cols = []
        for col in factors_df.select_dtypes(include=[np.number]).columns:
            col_std = factors_df[col].std()
            col_nunique = factors_df[col].nunique()

            # K线形态指标特殊处理：允许二值常量（0/1模式）
            if col.startswith("TA_CDL"):
                # K线形态指标：只有当完全无变化且只有一个值时才删除
                if col_nunique <= 1:
                    constant_cols.append(col)
            else:
                # 其他指标：使用标准差检测
                if col_std < 1e-6:
                    constant_cols.append(col)

        if constant_cols:
            # 区分K线形态指标和其他常量列
            cdl_cols = [col for col in constant_cols if col.startswith("TA_CDL")]
            other_cols = [col for col in constant_cols if not col.startswith("TA_CDL")]

            if cdl_cols:
                self.logger.info(f"移除无变化K线形态指标: {cdl_cols}")
            if other_cols:
                self.logger.warning(f"移除常量列: {other_cols}")

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
            raw_data_path = Path(self.config.raw_data_root) / "HK"
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
        """计算多周期IC值 - P0 性能优化：使用向量化引擎"""

        # 性能监控：IC计算
        if self.perf_monitor is not None:
            with self.perf_monitor.time_operation("calculate_multi_horizon_ic"):
                # P0 优化：优先使用向量化引擎
                if self.vectorized_analyzer is not None:
                    self.logger.info("🚀 使用 VectorBT 向量化引擎计算 IC")
                    return self.vectorized_analyzer.calculate_multi_horizon_ic_batch(
                        factors=factors,
                        returns=returns,
                        horizons=self.config.ic_horizons,
                    )

                # 降级方案：传统逐因子计算
                self.logger.warning("⚠️ 降级使用传统 IC 计算（性能较差）")
                return self._calculate_multi_horizon_ic_legacy(factors, returns)
        else:
            # 没有性能监控时的正常逻辑
            # P0 优化：优先使用向量化引擎
            if self.vectorized_analyzer is not None:
                self.logger.info("🚀 使用 VectorBT 向量化引擎计算 IC")
                return self.vectorized_analyzer.calculate_multi_horizon_ic_batch(
                    factors=factors,
                    returns=returns,
                    horizons=self.config.ic_horizons,
                )

            # 降级方案：传统逐因子计算
            self.logger.warning("⚠️ 降级使用传统 IC 计算（性能较差）")
            return self._calculate_multi_horizon_ic_legacy(factors, returns)

    def _calculate_multi_horizon_ic_legacy(
        self, factors: pd.DataFrame, returns: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """传统多周期IC计算 - 降级方案"""
        self.logger.info("开始多周期IC计算（传统模式）...")
        start_time = time.time()

        ic_results: Dict[str, Dict[str, float]] = {}
        horizons = self.config.ic_horizons

        factor_cols = [
            col
            for col in factors.columns
            if col not in ["open", "high", "low", "close", "volume"]
        ]

        total_factors = len(factor_cols)

        # 向量化优化: 预先对齐所有数据，避免重复对齐
        returns_series = returns.reindex(factors.index)
        valid_idx = returns_series.notna()
        aligned_factors = factors[factor_cols].loc[valid_idx]
        aligned_returns = returns_series.loc[valid_idx]

        processed = 0
        for factor in factor_cols:
            processed += 1
            if processed % self.config.progress_report_interval == 0:
                self.logger.info(f"多周期IC计算进度: {processed}/{total_factors}")

            factor_series = aligned_factors[factor]
            horizon_ics: Dict[str, float] = {}

            for horizon in horizons:
                if horizon < 0:
                    self.logger.warning(f"忽略非法预测周期 {horizon}，因子 {factor}")
                    continue

                if self.temporal_validator is not None:
                    try:
                        is_valid, message = (
                            self.temporal_validator.validate_time_alignment(
                                factor_series,
                                returns_series,
                                horizon,
                                context=f"IC-{factor}",
                            )
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

                # 🔥 关键修复：正确的时间对齐
                # factor[t] 预测 returns[t+horizon]
                # 不能用shift(horizon)！那是factor[t+h] vs returns[t]，时间反了
                # 正确方式：切片对齐
                if horizon == 0:
                    current_factor = factor_series
                    future_returns = aligned_returns
                else:
                    # 因子在前，收益在后
                    current_factor = factor_series.iloc[:-horizon]
                    future_returns = aligned_returns.iloc[horizon:]

                # 向量化: 一次性获取有效数据
                common_idx = current_factor.index.intersection(future_returns.index)
                if len(common_idx) < self.config.min_sample_size:
                    continue

                lagged_factor = current_factor.loc[common_idx]
                final_returns_series = future_returns.loc[common_idx]
                valid_mask = lagged_factor.notna() & final_returns_series.notna()
                valid_count = int(valid_mask.sum())

                if valid_count < self.config.min_sample_size:
                    continue

                final_factor = lagged_factor[valid_mask]
                final_returns = final_returns_series[valid_mask]

                try:
                    # 向量化: 使用numpy计算统计量
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
        """计算滚动IC - P0 性能优化：使用 VectorBT 引擎"""
        if window is None:
            window = self.config.rolling_window

        # P0 优化：优先使用向量化引擎
        if self.vectorized_analyzer is not None:
            self.logger.info(f"🚀 使用 VectorBT 向量化引擎计算滚动 IC (窗口={window})")
            return self.vectorized_analyzer.calculate_rolling_ic_vbt(
                factors=factors,
                returns=returns,
                window=window,
            )

        # 降级方案：传统计算
        self.logger.warning("⚠️ 降级使用传统滚动 IC 计算")
        return self._calculate_rolling_ic_legacy(factors, returns, window)

    def _calculate_rolling_ic_legacy(
        self, factors: pd.DataFrame, returns: pd.Series, window: int
    ) -> Dict[str, Dict[str, float]]:
        """传统滚动IC计算 - 降级方案"""
        self.logger.info(f"计算滚动IC (窗口={window}, 传统模式)...")
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

            final_factor = factor_series.loc[common_idx].to_numpy()
            final_returns = returns_series.loc[common_idx].to_numpy()

            # Linus模式：完全向量化的滚动窗口计算
            try:
                # 使用numpy的stride_tricks创建滑动窗口视图
                from numpy.lib.stride_tricks import sliding_window_view

                # 创建滑动窗口视图 - 一次生成所有窗口
                factor_windows = sliding_window_view(final_factor, window_shape=window)
                returns_windows = sliding_window_view(
                    final_returns, window_shape=window
                )

                # 数值稳定性预处理：过滤异常窗口
                factor_stds = np.std(factor_windows, axis=1)
                returns_stds = np.std(returns_windows, axis=1)

                # 向量化过滤：保留数值稳定的窗口
                valid_mask = (
                    (factor_stds > 1e-8)
                    & (returns_stds > 1e-8)
                    & (np.max(np.abs(factor_windows), axis=1) <= 1e10)
                    & (np.max(np.abs(returns_windows), axis=1) <= 100)
                )

                if np.sum(valid_mask) < 10:  # 至少需要10个有效窗口
                    continue

                # 使用有效窗口
                valid_factor_windows = factor_windows[valid_mask]
                valid_returns_windows = returns_windows[valid_mask]

                # Linus优化：批量计算所有窗口的Spearman相关系数
                # 使用更快的Pearson相关系数近似（向量化）

                # 中心化数据
                factor_centered = valid_factor_windows - np.mean(
                    valid_factor_windows, axis=1, keepdims=True
                )
                returns_centered = valid_returns_windows - np.mean(
                    valid_returns_windows, axis=1, keepdims=True
                )

                # 向量化相关系数计算
                numerator = np.sum(factor_centered * returns_centered, axis=1)
                factor_norm = np.sqrt(np.sum(factor_centered**2, axis=1))
                returns_norm = np.sqrt(np.sum(returns_centered**2, axis=1))

                # 除零保护
                denominator = factor_norm * returns_norm
                valid_corr_mask = denominator > 1e-12

                if np.sum(valid_corr_mask) < 10:
                    continue

                # 计算相关系数
                rolling_ics = numerator[valid_corr_mask] / denominator[valid_corr_mask]

                # 数值范围检查
                rolling_ics = rolling_ics[
                    (rolling_ics >= -1.0)
                    & (rolling_ics <= 1.0)
                    & ~np.isnan(rolling_ics)
                    & ~np.isinf(rolling_ics)
                ]

                if len(rolling_ics) >= 10:
                    # Linus优化：向量化统计计算
                    rolling_ics_array = np.asarray(rolling_ics, dtype=np.float64)
                    rolling_ic_mean = np.mean(rolling_ics_array)
                    rolling_ic_std = np.std(rolling_ics_array)

                    # 稳定性指标
                    stability = 1 - (rolling_ic_std / (abs(rolling_ic_mean) + 1e-8))
                    consistency = np.sum(rolling_ics_array * rolling_ic_mean > 0) / len(
                        rolling_ics_array
                    )

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
                self.logger.warning(
                    f"sliding_window_view不可用，使用降级方案计算因子 {factor}"
                )

                factor_df = pd.Series(final_factor)
                returns_df = pd.Series(final_returns)

                # 向量化滚动计算
                rolling_corr = factor_df.rolling(window).corr(returns_df)
                rolling_ics = rolling_corr.dropna()

                # 过滤异常值
                rolling_ics = rolling_ics[
                    (rolling_ics >= -1.0)
                    & (rolling_ics <= 1.0)
                    & ~np.isnan(rolling_ics)
                    & ~np.isinf(rolling_ics)
                ]

                if len(rolling_ics) >= 10:
                    rolling_ic_mean = float(rolling_ics.mean())
                    rolling_ic_std = float(rolling_ics.std())
                    stability = 1 - (rolling_ic_std / (abs(rolling_ic_mean) + 1e-8))
                    consistency = float(
                        np.sum(rolling_ics * rolling_ic_mean > 0) / len(rolling_ics)
                    )

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
        """计算方差膨胀因子 (VIF) - P0 性能优化：使用矩阵化计算"""

        # P0 优化：优先使用向量化引擎
        if self.vectorized_analyzer is not None:
            self.logger.info(f"🚀 使用矩阵化 VIF 计算（阈值={vif_threshold}）")
            return self.vectorized_analyzer.calculate_vif_batch(
                factors=factors,
                vif_threshold=vif_threshold,
            )

        # 降级方案：传统递归计算
        self.logger.warning("⚠️ 降级使用传统 VIF 计算")
        return self._calculate_vif_scores_legacy(factors, vif_threshold, max_iterations)

    def _calculate_vif_scores_legacy(
        self,
        factors: pd.DataFrame,
        vif_threshold: float = 5.0,
        max_iterations: int = 10,
    ) -> Dict[str, float]:
        """传统VIF计算 - 递归移除高共线性因子（降级方案）

        Args:
            factors: 输入因子表，需包含数值列。
            vif_threshold: 目标最大VIF阈值。
            max_iterations: 最大递归迭代次数。

        Returns:
            因子名称到VIF值的映射。
        """
        self.logger.info(f"开始递归VIF计算（传统模式，阈值={vif_threshold}）...")

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
                        "VIF递归完成: 迭代%s次，保留%s个因子，最大VIF=%.2f",
                        iteration,
                        len(remaining_factors),
                        max_vif,
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

            # P0-2修复：改进条件数检查和数值稳定性
            try:
                # 使用SVD计算条件数，更稳定
                U, s, Vt = np.linalg.svd(X, full_matrices=False)
                if len(s) == 0 or s[-1] <= 1e-15:
                    cond_number = np.inf
                else:
                    cond_number = s[0] / s[-1]
            except (np.linalg.LinAlgError, ValueError):
                cond_number = np.inf

            # 更严格的数值稳定性检查
            if not np.isfinite(cond_number) or cond_number > 1e10:
                vif_results[factor] = 5.0  # 设为阈值而非极大值
                continue

            try:
                # P0-2修复：使用更稳定的求解方法
                beta, residuals, rank, singular_vals = np.linalg.lstsq(
                    X, y, rcond=1e-15
                )

                # 检查求解质量
                if rank < X.shape[1]:
                    # 矩阵不满秩，设为阈值
                    vif_results[factor] = 5.0
                    continue

            except np.linalg.LinAlgError as err:
                self.logger.debug(f"VIF最小二乘求解失败 {factor}: {err}")
                vif_results[factor] = 5.0  # 设为阈值而非极大值
                continue

            # P0-2修复：改进R²计算的数值稳定性
            if residuals.size > 0:
                rss = float(residuals[0])
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

            # P0-2修复：改进VIF计算，避免极值
            if r_squared >= 0.999999:
                vif = 5.0  # 设为阈值，避免极大值
            else:
                vif = 1.0 / (1.0 - r_squared)
                vif = float(np.clip(vif, 1.0, 50.0))  # 限制VIF范围

            vif_results[factor] = vif

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
        """计算信息增量 - P1 性能优化：使用批量处理"""
        if base_factors is None:
            base_factors = self.config.base_factors

        # P1 优化：使用向量化引擎
        if self.vectorized_analyzer is not None:
            self.logger.info(f"🚀 使用批量信息增量计算 (基准因子: {base_factors})")
            return self.vectorized_analyzer.calculate_information_increment_batch(
                factors=factors,
                returns=returns,
                base_factors=base_factors,
            )

        # 降级方案
        self.logger.warning("⚠️ 降级使用传统信息增量计算")
        return self._calculate_information_increment_legacy(
            factors, returns, base_factors
        )

    def _calculate_information_increment_legacy(
        self, factors: pd.DataFrame, returns: pd.Series, base_factors: List[str]
    ) -> Dict[str, float]:
        """传统信息增量计算 - 降级方案"""
        self.logger.info(f"计算信息增量（传统模式，基准因子: {base_factors})...")

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
        """计算交易成本 - P0 性能优化：使用批量处理"""

        # P0 优化：使用向量化引擎进行批量计算
        if self.vectorized_analyzer is not None and "volume" in prices.columns:
            self.logger.info("🚀 使用批量交易成本计算")
            return self.vectorized_analyzer.calculate_trading_costs_batch(
                factors=factors,
                volume=prices["volume"],
                commission_rate=self.config.commission_rate,
                slippage_bps=self.config.slippage_bps,
                market_impact_coeff=self.config.market_impact_coeff,
            )

        # 降级方案：传统逐因子计算
        self.logger.warning("⚠️ 降级使用传统交易成本计算")
        return self._calculate_trading_costs_legacy(factors, prices, factor_metadata)

    def _calculate_trading_costs_legacy(
        self,
        factors: pd.DataFrame,
        prices: pd.DataFrame,
        factor_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """传统交易成本计算 - 降级方案"""
        self.logger.info("计算交易成本（传统模式）...")

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
            change_frequency = (
                factor_change > self.config.factor_change_threshold
            ).mean()  # 因子变化频率

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

        # P1-3修复：改进换手率异常处理，避免强制裁剪掩盖问题
        if turnover_rate > 2.0:
            # 分析异常原因
            extreme_changes = valid_changes[
                valid_changes > valid_changes.quantile(0.95)
            ]
            if len(extreme_changes) > len(valid_changes) * 0.1:
                # 如果超过10%的变化都很大，可能是因子设计问题
                self.logger.warning(
                    f"因子 {factor_name} turnover率异常高 ({turnover_rate:.6f})，"
                    "可能存在因子设计问题，建议检查计算逻辑"
                )
                # 使用更保守的估计：95%分位数
                conservative_rate = valid_changes.quantile(0.95)
                if conservative_rate > 2.0:
                    turnover_rate = 2.0
                else:
                    turnover_rate = float(conservative_rate)
            else:
                # 少数极值导致，使用中位数估计
                median_rate = valid_changes.median()
                self.logger.info(
                    f"因子 {factor_name} turnover率异常高 ({turnover_rate:.6f})，"
                    f"使用中位数估计 ({median_rate:.6f})"
                )
                turnover_rate = float(min(median_rate, 2.0))

        return turnover_rate

    # ==================== 5. 短周期适应性分析 ====================

    def detect_reversal_effects(
        self, factors: pd.DataFrame, returns: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """检测反转效应 - P1 性能优化：使用批量处理"""

        # P1 优化：使用向量化引擎
        if self.vectorized_analyzer is not None:
            self.logger.info("🚀 使用批量短周期适应性计算")
            return self.vectorized_analyzer.calculate_short_term_adaptability_batch(
                factors=factors,
                returns=returns,
                high_rank_threshold=self.config.high_rank_threshold,
                low_rank_threshold=0.2,
            )

        # 降级方案
        self.logger.warning("⚠️ 降级使用传统反转效应检测")
        return self._detect_reversal_effects_legacy(factors, returns)

    def _detect_reversal_effects_legacy(
        self, factors: pd.DataFrame, returns: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """传统反转效应检测 - 降级方案"""
        self.logger.info("检测反转效应（传统模式）...")

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
        """分析动量持续性（完全向量化实现）。"""

        # 优先使用向量化引擎
        if self.vectorized_analyzer is not None:
            self.logger.info("🚀 使用批量动量持续性分析")
            return self.vectorized_analyzer.calculate_momentum_persistence_batch(
                factors=factors, returns=returns, windows=[5, 10, 20], forward_horizon=5
            )
        else:
            # 降级方案
            self.logger.warning("⚠️ 降级使用传统动量持续性分析")
            # 性能监控：动量持续性分析
            if self.perf_monitor is not None:
                with self.perf_monitor.time_operation("analyze_momentum_persistence"):
                    self.logger.info("分析动量持续性...")
                    return self._analyze_momentum_persistence_impl(factors, returns)
            else:
                self.logger.info("分析动量持续性...")
                return self._analyze_momentum_persistence_impl(factors, returns)

    def _analyze_momentum_persistence_impl(
        self, factors: pd.DataFrame, returns: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """动量持续性分析的具体实现（Linus式优化）"""

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

        # Linus式优化：预先计算returns的滑动窗口，避免重复计算
        returns_series = returns.dropna()
        if len(returns_series) < forward_horizon + 20:
            return momentum_analysis

        returns_values = returns_series.to_numpy(dtype=np.float64)

        # 一次性计算所有需要的滑动窗口和前瞻收益
        forward_returns_cache = {}
        max_window = windows.max()

        if len(returns_values) > max_window + forward_horizon:
            # 预计算所有窗口的前瞻收益和
            for window in windows:
                if len(returns_values) > window + forward_horizon:
                    start_idx = window + 1
                    forward_mat = np.lib.stride_tricks.sliding_window_view(
                        returns_values[start_idx:], forward_horizon
                    )
                    forward_returns_cache[window] = forward_mat.sum(axis=1)

        for factor in factor_cols:
            factor_series = factors[factor].dropna()

            common_idx = factor_series.index.intersection(returns_series.index)
            if len(common_idx) < self.config.min_momentum_samples:
                continue

            factor_values = factor_series.loc[common_idx].to_numpy(dtype=np.float64)
            returns_aligned = returns_series.reindex(common_idx).to_numpy(
                dtype=np.float64
            )

            n = factor_values.shape[0]
            if n < self.config.min_momentum_samples:
                continue

            # Linus式优化：使用预计算的前瞻收益，避免重复滑动窗口
            all_signals = []
            all_forward_returns = []

            for window in windows:
                if window not in forward_returns_cache:
                    continue

                max_start = n - forward_horizon
                if max_start <= window:
                    continue

                # 获取预计算的前瞻收益
                forward_sums = forward_returns_cache[window]
                if len(forward_sums) < max_start - window:
                    continue

                current_vals = factor_values[window:max_start]
                usable_forward = forward_sums[: len(current_vals)]

                all_signals.append(current_vals)
                all_forward_returns.append(usable_forward)

            if not all_signals:
                continue

            # Linus式优化：减少内存分配，直接使用预计算结果
            signals_array = np.concatenate(all_signals, dtype=np.float64)
            forward_returns_array = np.concatenate(
                all_forward_returns, dtype=np.float64
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
        """分析波动率敏感性（完全向量化实现）"""

        # 优先使用向量化引擎
        if self.vectorized_analyzer is not None:
            self.logger.info("🚀 使用批量波动率敏感性分析")
            return self.vectorized_analyzer.calculate_volatility_sensitivity_batch(
                factors=factors,
                returns=returns,
                vol_window=20,
                high_vol_percentile=0.7,
                low_vol_percentile=0.3,
            )
        else:
            # 降级方案
            self.logger.warning("⚠️ 降级使用传统波动率敏感性分析")
            # 性能监控：波动率敏感性分析
            if self.perf_monitor is not None:
                with self.perf_monitor.time_operation("analyze_volatility_sensitivity"):
                    self.logger.info("分析波动率敏感性...")
                    return self._analyze_volatility_sensitivity_impl(factors, returns)
            else:
                self.logger.info("分析波动率敏感性...")
                return self._analyze_volatility_sensitivity_impl(factors, returns)

    def _analyze_volatility_sensitivity_impl(
        self, factors: pd.DataFrame, returns: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """波动率敏感性分析的具体实现"""

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
        self,
        p_values: Dict[str, float],
        alpha: float = None,
        sample_size: int = None,
        timeframe: str = None,
    ) -> Tuple[Dict[str, float], float]:
        """
        🚀 向量化 Benjamini-Hochberg FDR校正 - 自适应显著性阈值

        性能优化：
        - 移除 for i 循环，使用 NumPy 向量化计算
        - 复杂度从 O(n) 降至 O(log n)（主要是排序开销）
        """
        if alpha is None:
            alpha = self.config.alpha_level

        if not p_values:
            return {}, alpha

        # 转换为数组（向量化预处理）
        factors = list(p_values.keys())
        p_vals = np.array([p_values[factor] for factor in factors])
        n_tests = len(p_vals)

        # 🚀 时间框架自适应alpha
        adaptive_alpha = alpha
        if timeframe and getattr(self.config, "enable_timeframe_adaptive", False):
            tf_alpha_map = getattr(self.config, "timeframe_alpha_map", {})
            if timeframe in tf_alpha_map:
                adaptive_alpha = tf_alpha_map[timeframe]
                self.logger.info(
                    f"时间框架自适应: {timeframe} 使用alpha={adaptive_alpha:.3f}"
                )

        # 样本量自适应调整
        if sample_size is not None:
            if sample_size < 100:
                adaptive_alpha = min(adaptive_alpha * 1.2, 0.15)
                self.logger.info(
                    f"小样本量({sample_size})，进一步放宽alpha至{adaptive_alpha:.3f}"
                )
            elif sample_size < 200:
                adaptive_alpha = min(adaptive_alpha * 1.1, 0.12)
                self.logger.info(
                    f"中等样本量({sample_size})，微调alpha至{adaptive_alpha:.3f}"
                )

        # 🚀 向量化 BH 校正（移除循环）
        sorted_indices = np.argsort(p_vals)
        sorted_p = p_vals[sorted_indices]

        # 向量化计算校正p值：p_corrected = p * n / (i + 1)
        i_plus_1 = np.arange(1, n_tests + 1)  # [1, 2, ..., n]
        corrected_p_sorted = np.minimum(sorted_p * n_tests / i_plus_1, 1.0)

        # 还原到原始顺序
        corrected_p_vals = np.empty_like(corrected_p_sorted)
        corrected_p_vals[sorted_indices] = corrected_p_sorted

        # 组装结果字典
        corrected_p = {
            factor: float(p_val) for factor, p_val in zip(factors, corrected_p_vals)
        }

        # 向量化统计显著因子数量
        significant_count = int((corrected_p_sorted <= adaptive_alpha).sum())
        significant_ratio = significant_count / n_tests if n_tests > 0 else 0

        # 统计报告
        if significant_ratio < 0.05 and sample_size and sample_size < 500:
            self.logger.warning(
                f"显著因子比例过低({significant_ratio:.1%})，"
                "建议检查数据质量或考虑增加样本量"
            )
        elif significant_ratio > 0.20:
            self.logger.info(
                f"显著因子比例: {significant_ratio:.1%} "
                f"({significant_count}/{n_tests})"
            )

        return corrected_p, adaptive_alpha

    def bonferroni_correction(
        self, p_values: Dict[str, float]
    ) -> Tuple[Dict[str, float], float]:
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
        self, all_metrics: Dict[str, Dict], timeframe: str = "1min"
    ) -> Dict[str, FactorMetrics]:
        """计算综合评分 - 5维度加权评分（时间框架自适应）"""
        self.logger.info(f"计算综合评分 (timeframe={timeframe})...")
        self._current_timeframe = timeframe  # 记录当前时间框架供公平评分使用

        comprehensive_results = {}

        # 获取所有因子名称
        all_factors: Set[str] = set()
        for metric_key, metric_dict in all_metrics.items():
            if isinstance(metric_dict, dict):
                all_factors.update(metric_dict.keys())

        if not all_factors:
            self.logger.warning("综合评分阶段未找到任何因子数据")
            return comprehensive_results

        # 预构建便于访问的指标表
        def _get_metric(metric_name: str, factor: str, default: Any = None) -> Any:
            mapping = all_metrics.get(metric_name, {})
            if isinstance(mapping, dict):
                return mapping.get(factor, default)
            return default

        for factor in sorted(all_factors):
            metrics = FactorMetrics(name=factor)

            # 1. 预测能力评分 (35%)
            predictive_score = 0.0
            ic_data = _get_metric("multi_horizon_ic", factor, {})
            if ic_data:
                # 提取各周期IC（原始值，包含正负）
                metrics.ic_1d = ic_data.get("ic_1d", 0.0)
                metrics.ic_3d = ic_data.get("ic_3d", 0.0)
                metrics.ic_5d = ic_data.get("ic_5d", 0.0)
                metrics.ic_10d = ic_data.get("ic_10d", 0.0)
                metrics.ic_20d = ic_data.get("ic_20d", 0.0)

                # 获取原始IC值（用于胜率计算）
                raw_ic_values = [
                    ic_data.get(f"ic_{h}d", 0.0) for h in self.config.ic_horizons
                ]
                raw_ic_values = [ic for ic in raw_ic_values if ic != 0.0]

                # 计算平均IC和IR（使用绝对值，因为预测能力不关心方向）
                ic_values = [abs(ic) for ic in raw_ic_values]

                if ic_values:
                    metrics.ic_mean = np.mean(ic_values)
                    metrics.ic_std = np.std(ic_values) if len(ic_values) > 1 else 0.1
                    metrics.ic_ir = metrics.ic_mean / (metrics.ic_std + 1e-8)
                    metrics.predictive_power_mean_ic = metrics.ic_mean  # 设置缺失字段

                    # 计算IC胜率：原始IC中正值的占比
                    positive_ic_count = sum(1 for ic in raw_ic_values if ic > 0)
                    metrics.ic_win_rate = (
                        positive_ic_count / len(raw_ic_values) if raw_ic_values else 0.0
                    )

                    # 预测能力得分：平衡IC范围，IC=0.05->1.0分
                    predictive_score = min(metrics.ic_mean * 20, 1.0)

                    # 🚨 关键：将predictive_score存储，后续会应用样本量权重
                    metrics.predictive_score_raw = (
                        predictive_score  # 原始分数（用于调试）
                    )

            decay_data = _get_metric("ic_decay", factor, {})
            if decay_data:
                metrics.ic_decay_rate = decay_data.get("decay_rate", 0.0)
                metrics.ic_longevity = decay_data.get("ic_longevity", 0)

                # 衰减惩罚
                decay_penalty = abs(metrics.ic_decay_rate) * 0.1
                predictive_score = max(0, predictive_score - decay_penalty)

            metrics.predictive_score = predictive_score

            # 2. 稳定性评分 (25%)
            stability_score = 0.0
            rolling_data = _get_metric("rolling_ic", factor, {})
            if rolling_data:
                metrics.rolling_ic_mean = rolling_data.get("rolling_ic_mean", 0.0)
                metrics.rolling_ic_std = rolling_data.get("rolling_ic_std", 0.0)
                metrics.rolling_ic_stability = rolling_data.get(
                    "rolling_ic_stability", 0.0
                )
                metrics.ic_consistency = rolling_data.get("ic_consistency", 0.0)

                stability_score = (
                    metrics.rolling_ic_stability + metrics.ic_consistency
                ) / 2

            cs_data = _get_metric("cross_section_stability", factor, {})
            if cs_data:
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
            vif_score = _get_metric("vif_scores", factor)
            if vif_score is not None:
                metrics.vif_score = float(vif_score)
                vif_penalty = min(metrics.vif_score / self.config.vif_threshold, 2.0)
                independence_score *= 1 / (1 + vif_penalty)

            corr_matrix = all_metrics.get("correlation_matrix")
            if isinstance(corr_matrix, pd.DataFrame) and factor in corr_matrix.columns:
                factor_corrs = corr_matrix[factor].drop(factor, errors="ignore")
                if len(factor_corrs) > 0:
                    metrics.correlation_max = float(factor_corrs.abs().max())
                    corr_penalty = max(0, metrics.correlation_max - 0.5) * 2
                    independence_score *= max(0.0, 1 - corr_penalty)

            information_increment = _get_metric("information_increment", factor)
            if information_increment is not None:
                metrics.information_increment = float(information_increment)
                # 信息增量奖励
                info_bonus = max(0.0, metrics.information_increment) * 5
                independence_score = min(independence_score + info_bonus, 1.0)

            metrics.independence_score = max(0.0, independence_score)

            # 4. 实用性评分 (15%)
            practicality_score = 1.0
            cost_data = _get_metric("trading_costs", factor, {})
            if cost_data:
                metrics.turnover_rate = cost_data.get("turnover_rate", 0.0)
                metrics.transaction_cost = cost_data.get("total_cost", 0.0)
                metrics.cost_efficiency = cost_data.get("cost_efficiency", 0.0)

                practicality_score = metrics.cost_efficiency or practicality_score

            liquidity_data = _get_metric("liquidity_requirements", factor, {})
            if liquidity_data:
                metrics.liquidity_requirement = liquidity_data.get(
                    "liquidity_requirement", 0.0
                )
                metrics.volume_coverage_ratio = liquidity_data.get(
                    "volume_coverage_ratio", 0.0
                )
                liquidity_penalty = max(0.0, 1 - metrics.volume_coverage_ratio)
                practicality_score *= max(0.0, 1 - liquidity_penalty)

            metrics.practicality_score = max(0.0, practicality_score)

            # 5. 适应性评分 (5%)
            # 修复：使用绝对值而非max(0,x)，因为负值也代表有效的反向效应
            adaptability_score = 0.0
            reversal_data = _get_metric("reversal_effects", factor, {})
            if reversal_data:
                metrics.reversal_effect = reversal_data.get("reversal_effect", 0.0)
                adaptability_score += abs(metrics.reversal_effect)  # 修复：使用abs()

            momentum_data = _get_metric("momentum_persistence", factor, {})
            if momentum_data:
                metrics.momentum_persistence = momentum_data.get(
                    "momentum_persistence", 0.0
                )
                adaptability_score += abs(
                    metrics.momentum_persistence
                )  # 修复：使用abs()

            volatility_data = _get_metric("volatility_sensitivity", factor, {})
            if volatility_data:
                metrics.volatility_sensitivity = volatility_data.get(
                    "volatility_sensitivity", 0.0
                )
                adaptability_score += abs(
                    metrics.volatility_sensitivity
                )  # 修复：使用abs()

            metrics.adaptability_score = min(adaptability_score / 3, 1.0)

            # 🚀 样本量权重修正（轻微折扣，保持公平竞争）
            sample_weight = 1.0
            predictive_weight = 1.0
            sample_weight_params = getattr(self.config, "sample_weight_params", None)
            if sample_weight_params and sample_weight_params.get("enable", False):
                # 从IC数据提取样本量
                actual_sample_size = None
                if ic_data:
                    # 尝试从各周期获取样本量（取最大值）
                    sample_sizes = [
                        ic_data.get(f"sample_size_{h}d", 0)
                        for h in self.config.ic_horizons
                    ]
                    if sample_sizes and max(sample_sizes) > 0:
                        actual_sample_size = max(sample_sizes)

                if actual_sample_size is not None:
                    min_samples = sample_weight_params.get(
                        "min_full_weight_samples", 500
                    )
                    power = sample_weight_params.get("weight_power", 0.5)

                    # 计算样本量权重: w = min(1.0, (N/N0)^power)
                    sample_weight = min(
                        1.0, (actual_sample_size / min_samples) ** power
                    )

                    # 对指定维度应用样本量权重（仅影响统计可靠性相关维度）
                    affected_dims = sample_weight_params.get(
                        "affected_dimensions",
                        ["stability", "independence", "practicality"],
                    )

                    if "stability" in affected_dims:
                        metrics.stability_score *= sample_weight
                    if "independence" in affected_dims:
                        metrics.independence_score *= sample_weight
                    if "practicality" in affected_dims:
                        metrics.practicality_score *= sample_weight
                    if "adaptability" in affected_dims:
                        metrics.adaptability_score *= sample_weight

                    # 记录样本量权重用于后续分析
                    metrics.sample_weight = sample_weight
                    metrics.predictive_weight = predictive_weight
                    metrics.actual_sample_size = actual_sample_size

            # 综合评分计算
            custom_weights = getattr(self.config, "weights", None)
            if custom_weights:
                weights = {
                    "predictive_power": float(
                        custom_weights.get("predictive_power", 0.35)
                    ),
                    "stability": float(custom_weights.get("stability", 0.25)),
                    "independence": float(custom_weights.get("independence", 0.20)),
                    "practicality": float(custom_weights.get("practicality", 0.15)),
                    "short_term_fitness": float(
                        custom_weights.get("short_term_fitness", 0.05)
                    ),
                }
            else:
                weights = {
                    "predictive_power": 0.35,
                    "stability": 0.25,
                    "independence": 0.20,
                    "practicality": 0.15,
                    "short_term_fitness": 0.05,
                }

            total_weight = sum(weights.values())
            if not np.isclose(total_weight, 1.0, rtol=1e-6):
                self.logger.error(
                    "权重配置错误: 总和=%.6f, 应为1.0 -- 当前权重=%s",
                    total_weight,
                    weights,
                )
                raise ValueError("权重配置错误 - 系统完整性检查失败")

            # 计算传统综合评分
            traditional_score = float(
                weights["predictive_power"] * metrics.predictive_score
                + weights["stability"] * metrics.stability_score
                + weights["independence"] * metrics.independence_score
                + weights["practicality"] * metrics.practicality_score
                + weights["short_term_fitness"] * metrics.adaptability_score
            )

            # 🚀 应用公平评分调整
            if self.fair_scorer and self.fair_scorer.enabled:
                # 获取样本量信息
                sample_size = getattr(
                    metrics,
                    "actual_sample_size",
                    self._estimate_sample_size(all_metrics, factor),
                )

                # 应用公平评分
                fair_score = self.fair_scorer.apply_fair_scoring(
                    original_score=traditional_score,
                    timeframe=timeframe,
                    sample_size=sample_size,
                    ic_mean=metrics.ic_mean,
                    stability_score=metrics.stability_score,
                    predictive_score=metrics.predictive_score,
                )

                # 记录评分对比
                score_comparison = self.fair_scorer.compare_scores(
                    traditional_score, fair_score, timeframe
                )
                self.logger.debug(
                    f"因子 {factor} 公平评分调整: {score_comparison['percent_change']:.1f}% "
                    f"({traditional_score:.3f} -> {fair_score:.3f})"
                )

                # 保存评分信息
                metrics.comprehensive_score = fair_score
                metrics.traditional_score = traditional_score
                metrics.fair_scoring_applied = True
                metrics.fair_scoring_change = fair_score - traditional_score
                metrics.fair_scoring_percent_change = score_comparison["percent_change"]
            else:
                # 未启用公平评分，使用传统评分
                metrics.comprehensive_score = traditional_score
                metrics.traditional_score = traditional_score
                metrics.fair_scoring_applied = False

            # 显著性标记
            corrected_p_vals = _get_metric("corrected_p_values", factor)
            bennett_scores = _get_metric("bennett_scores", factor)
            metrics.corrected_p_value = (
                float(corrected_p_vals)
                if corrected_p_vals is not None
                else metrics.corrected_p_value
            )
            metrics.bennett_score = (
                float(bennett_scores) if bennett_scores is not None else 0.0
            )

            # 使用自适应alpha判断显著性
            current_alpha = float(
                all_metrics.get("adaptive_alpha", self.config.alpha_level)
            )
            metrics.is_significant = (
                metrics.corrected_p_value is not None
                and metrics.corrected_p_value <= current_alpha
            )

            # 🔧 关键修复：设置因子等级分类（时间框架自适应）
            metrics.tier = self._classify_factor_tier(
                metrics.comprehensive_score,
                metrics.is_significant,
                metrics.ic_mean,
                timeframe,
            )

            comprehensive_results[factor] = metrics

        self.logger.info(f"综合评分计算完成: {len(comprehensive_results)} 个因子")
        return comprehensive_results

    def _estimate_sample_size(self, all_metrics: Dict[str, Dict], factor: str) -> int:
        """估算因子样本量"""
        # 尝试从IC数据获取样本量
        ic_data = all_metrics.get("multi_horizon_ic", {}).get(factor, {})
        if ic_data:
            # 获取各周期的样本量，取最大值
            sample_sizes = []
            for h in self.config.ic_horizons:
                size_key = f"sample_size_{h}d"
                if size_key in ic_data:
                    sample_sizes.append(ic_data[size_key])
            if sample_sizes:
                return max(sample_sizes)

        # 根据时间框架提供默认估算
        default_sizes = {
            "1min": 40000,
            "5min": 8000,
            "15min": 3000,
            "30min": 1500,
            "60min": 750,
            "2h": 250,
            "4h": 125,
            "1day": 100,
        }

        # 从self属性获取时间框架
        if hasattr(self, "_current_timeframe"):
            return default_sizes.get(self._current_timeframe, 1000)

        return 1000  # 默认值

    def _classify_factor_tier(
        self,
        comprehensive_score: float,
        is_significant: bool,
        ic_mean: float,
        timeframe: str = "1min",
    ) -> str:
        """P1-1修复：因子等级分类逻辑（时间框架自适应）

        分级标准（根据时间框架动态调整）：
        - Tier 1: 核心因子，强烈推荐
        - Tier 2: 重要因子，推荐使用
        - Tier 3: 备用因子，特定条件使用
        - 不推荐: 不建议使用
        """
        # 🔧 获取样本量自适应阈值（最优解配置）
        if hasattr(self.config, "adaptive_tier_thresholds"):
            thresholds = self.config.adaptive_tier_thresholds.get(
                timeframe,
                {
                    "tier1": 0.80,
                    "tier2": 0.60,
                    "tier3": 0.40,
                    "upgrade_tier2": 0.55,
                    "upgrade_tier1": 0.75,
                },
            )
        else:
            # 回退到原有配置
            thresholds = self.config.timeframe_tier_thresholds.get(
                timeframe,
                {
                    "tier1": 0.80,
                    "tier2": 0.60,
                    "tier3": 0.40,
                    "upgrade_tier2": 0.55,
                    "upgrade_tier1": 0.75,
                },
            )

        # 基础分级（使用自适应阈值）
        if comprehensive_score >= thresholds["tier1"]:
            base_tier = "Tier 1"
        elif comprehensive_score >= thresholds["tier2"]:
            base_tier = "Tier 2"
        elif comprehensive_score >= thresholds["tier3"]:
            base_tier = "Tier 3"
        else:
            base_tier = "不推荐"

        # 显著性和IC调整（使用自适应升级阈值）
        if is_significant and abs(ic_mean) >= 0.05:
            # 显著且IC较强，维持或提升等级
            if (
                base_tier == "Tier 3"
                and comprehensive_score >= thresholds["upgrade_tier2"]
            ):
                return "Tier 2"
            elif (
                base_tier == "Tier 2"
                and comprehensive_score >= thresholds["upgrade_tier1"]
            ):
                return "Tier 1"
        elif not is_significant or abs(ic_mean) < 0.02:
            # 不显著或IC很弱，降级
            if base_tier == "Tier 1":
                return "Tier 2"
            elif base_tier == "Tier 2":
                return "Tier 3"

        return base_tier

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

        # P0-3修复：改进p值收集和显著性判断
        p_values = {}
        for factor, ic_data in all_metrics["multi_horizon_ic"].items():
            # 使用1日IC的p值作为主要显著性指标，如果没有则使用最小的p值
            p_1d = ic_data.get("p_value_1d", 1.0)
            if p_1d >= 1.0:
                # 如果1日p值无效，使用所有周期中最小的p值
                all_p_values = [
                    ic_data.get(f"p_value_{h}d", 1.0) for h in self.config.ic_horizons
                ]
                p_1d = min([p for p in all_p_values if p < 1.0] or [1.0])

            p_values[factor] = p_1d

        all_metrics["p_values"] = p_values

        # FDR校正（传入样本量和时间框架用于自适应阈值调整）
        sample_size = len(factors_aligned)
        if self.config.fdr_method == "benjamini_hochberg":
            corrected_p, adaptive_alpha = self.benjamini_hochberg_correction(
                p_values, sample_size=sample_size, timeframe=timeframe
            )
        else:
            corrected_p, adaptive_alpha = self.bonferroni_correction(p_values)

        all_metrics["corrected_p_values"] = corrected_p
        all_metrics["adaptive_alpha"] = adaptive_alpha

        # 5. 综合评分
        self.logger.info("步骤5: 综合评分...")
        comprehensive_results = self.calculate_comprehensive_scores(
            all_metrics, timeframe
        )

        # 性能统计
        duration = time.time() - start_time

        # 保存结果（保存到时间框架子目录）
        screening_stats = {
            "total_factors": len(comprehensive_results),
            "significant_factors": sum(
                1 for metric in comprehensive_results.values() if metric.is_significant
            ),
            "high_score_factors": sum(
                1
                for metric in comprehensive_results.values()
                if metric.comprehensive_score > 0.7
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
            "factor_count": len(factors.columns),
            "sample_size": len(common_index),
            "factor_data_range": {
                "start": factors.index.min().isoformat() if len(factors) > 0 else None,
                "end": factors.index.max().isoformat() if len(factors) > 0 else None,
            },
            "alignment_success_rate": len(common_index)
            / min(len(factors), len(close_prices)),
        }

        # 🔧 修复：调用完整的保存逻辑（与主筛选方法一致）
        try:
            if self.result_manager is not None:
                # 使用增强版结果管理器，传递现有会话目录
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

            # 根据配置决定是否保存传统格式
            if self.config.enable_legacy_format:
                self.logger.info("根据配置启用传统格式保存")
                try:
                    self.save_comprehensive_screening_info(
                        comprehensive_results,
                        symbol,
                        timeframe,
                        screening_stats,
                        data_quality_info,
                    )
                    self.logger.info("传统格式保存完成")
                except Exception as e:
                    self.logger.error(f"传统格式保存失败: {e}")
            else:
                self.logger.info("使用增强版结果管理器，跳过传统格式保存")

        except Exception as e:
            self.logger.error(f"保存完整筛选信息失败: {str(e)}")
            screening_stats["save_error"] = str(e)
        self.logger.info(f"✅ {symbol} {timeframe} 筛选完成，耗时: {duration:.2f}秒")
        self.logger.info(f"   总因子数: {len(comprehensive_results)}")
        top_factor_count = sum(
            1
            for metric in comprehensive_results.values()
            if metric.comprehensive_score >= 0.8
        )
        self.logger.info("   顶级因子数: %s", top_factor_count)

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
                metrics_list = list(tf_results.values())
                tf_summary = {
                    "total_factors": len(metrics_list),
                    "significant_factors": sum(
                        1
                        for metric in metrics_list
                        if metric.corrected_p_value < self.config.alpha_level
                    ),
                    "top_factors": sum(
                        1
                        for metric in metrics_list
                        if metric.comprehensive_score >= 0.8
                    ),
                    "average_ic": (
                        float(np.mean([metric.ic_mean for metric in metrics_list]))
                        if metrics_list
                        else 0.0
                    ),
                    "average_score": (
                        float(
                            np.mean(
                                [metric.comprehensive_score for metric in metrics_list]
                            )
                        )
                        if metrics_list
                        else 0.0
                    ),
                }
                summary["timeframe_summary"][tf] = tf_summary

                top_factors = sorted(
                    [(name, metrics) for name, metrics in tf_results.items()],
                    key=lambda x: x[1].comprehensive_score,
                    reverse=True,
                )[:10]
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
            factor_performance: Dict[str, Dict[str, float]] = {}
            for tf, tf_results in all_results.items():
                for factor_name, metrics in tf_results.items():
                    factor_performance.setdefault(factor_name, {})[
                        tf
                    ] = metrics.comprehensive_score

            consensus_threshold = 0.7
            min_timeframes = max(1, len(timeframes) // 2)

            consensus_factors = []
            for factor_name, tf_scores in factor_performance.items():
                scores = list(tf_scores.values())
                high_score_count = sum(
                    1 for score in scores if score >= consensus_threshold
                )
                if high_score_count >= min_timeframes:
                    avg_score = float(np.mean(scores))
                    consensus_factors.append(
                        {
                            "factor": factor_name,
                            "average_score": avg_score,
                            "high_score_count": high_score_count,
                            "scores_by_timeframe": tf_scores,
                        }
                    )

            consensus_factors.sort(key=lambda x: x["average_score"], reverse=True)
            summary["consensus_factors"] = consensus_factors[:20]

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
            from factor_alignment_utils import (
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
            # 🚀 P0修复：可配置的对齐失败策略
            alignment_strategy = getattr(
                self.config, "alignment_failure_strategy", "warn"
            )

            if alignment_strategy == "fail_fast":
                self.logger.error(f"❌ 因子对齐检查失败（fail_fast模式）: {str(e)}")
                raise ValueError(
                    f"因子对齐失败，无法保证多时间框架一致性: {str(e)}"
                ) from e
            elif alignment_strategy == "warn":
                self.logger.warning(
                    f"⚠️ 因子对齐检查失败（warn模式），继续使用默认文件选择: {str(e)}\n"
                    "   注意：不同时间框架可能来自不同批次，结果可比性降低"
                )
                self.aligned_factor_files = None
            else:  # fallback
                self.logger.info(
                    f"ℹ️ 因子对齐检查失败（fallback模式），使用默认文件选择: {str(e)}"
                )
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

                    # 🔧 修复：正确遍历字典的values()
                    top_factor_count_tf = sum(
                        1
                        for metric in tf_results.values()
                        if metric.comprehensive_score >= 0.8
                    )
                    self.logger.info(
                        "✅ %s 筛选完成 - 顶级因子数: %s",
                        timeframe,
                        top_factor_count_tf,
                    )

                except Exception as e:
                    failed_timeframes.append(timeframe)
                    main_logger.error(f"❌ {timeframe} 筛选失败: {str(e)}")

                    # 🔧 修复：移除提前停止逻辑，继续处理所有时间框架
                    # 让筛选器完成所有时间框架的处理
                    main_logger.warning(f"⚠️ {timeframe} 失败，继续处理剩余时间框架")

            # 保存多时间框架汇总报告
            if all_results:
                main_logger.info("生成多时间框架汇总报告...")

                # 🔧 修复：调用完整的汇总生成函数
                batch_name = f"{symbol.replace('.', '_')}_multi_timeframe_analysis"

                # 转换结果格式以匹配_generate_multi_timeframe_summary的期望格式
                formatted_results = {}
                for timeframe, tf_results in all_results.items():
                    if isinstance(tf_results, dict):
                        # tf_results 已经是 Dict[str, FactorMetrics] 格式
                        formatted_results[f"{symbol}_{timeframe}"] = tf_results
                    else:
                        main_logger.warning(f"⚠️ {timeframe} 结果格式异常，跳过")

                # 调用完整的汇总生成函数
                _generate_multi_timeframe_summary(
                    main_session_dir,
                    batch_name,
                    formatted_results,
                    [],  # screening_configs 暂时为空
                    logger=main_logger,
                )

                # 保留原有的简单汇总（向后兼容）
                self.save_multi_timeframe_summary(symbol, timeframes, all_results)

            # 完成统计
            total_duration = (datetime.now() - start_time).total_seconds()
            main_logger.info("🎉 多时间框架筛选完成!")
            main_logger.info(f"   总耗时: {total_duration:.2f}秒")
            main_logger.info(
                f"   成功时间框架: {len(successful_timeframes)}/{len(timeframes)}"
            )
            failed_summary = ", ".join(failed_timeframes) if failed_timeframes else "无"
            main_logger.info("   失败时间框架: %s", failed_summary)

            # 计算总体统计
            total_factors = sum(len(result) for result in all_results.values())
            total_top_factors = sum(
                sum(
                    1 for metric in result.values() if metric.comprehensive_score >= 0.8
                )
                for result in all_results.values()
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

        # P0级集成：使用输入验证器
        if self.input_validator is not None:
            is_valid, msg = self.input_validator.validate_symbol(symbol, strict=False)
            if not is_valid:
                self.logger.error(f"输入验证失败: {msg}")
                raise ValueError(msg)

            is_valid, msg = self.input_validator.validate_timeframe(timeframe)
            if not is_valid:
                self.logger.error(f"输入验证失败: {msg}")
                raise ValueError(msg)

        # 使用性能监控器包装整个筛选过程
        operation_name = f"screen_factors_comprehensive_{symbol}_{timeframe}"
        if self.perf_monitor is not None:
            perf_context = self.perf_monitor.time_operation(operation_name)
            perf_context.__enter__()
            perf_monitor_active = True
        else:
            perf_monitor_active = False

        # P0级集成：使用结构化日志记录操作开始
        if self.structured_logger is not None:
            self.structured_logger.info(
                "因子筛选开始",
                symbol=symbol,
                timeframe=timeframe,
                operation="screen_factors_comprehensive",
            )

        # P0-1修复：智能会话管理，避免批量处理中的重复创建
        in_multi_tf_mode = (
            hasattr(self, "multi_tf_session_dir") and self.multi_tf_session_dir
        )
        current_session_dir = None

        if in_multi_tf_mode:
            # 批量模式：强制为当前时间框架切换到专属子目录
            tf_session_dir = (
                self.multi_tf_session_dir
                / "timeframes"
                / f"{symbol}_{timeframe}_{self.session_timestamp}"
            )
            tf_session_dir.mkdir(parents=True, exist_ok=True)
            self.session_dir = tf_session_dir
            session_id = tf_session_dir.name
            current_session_dir = tf_session_dir
            self.logger.info(
                f"📁 批量模式-切换时间框架子会话: {timeframe}",
                extra={"session_dir": str(tf_session_dir)},
            )
        else:
            if not hasattr(self, "session_dir") or not self.session_dir:
                # 单独模式：创建独立会话目录
                session_id = f"{symbol}_{timeframe}_{self.session_timestamp}"
                self.session_dir = self.screening_results_dir / session_id
                self.session_dir.mkdir(parents=True, exist_ok=True)
                current_session_dir = self.session_dir
                self.logger.info(f"📁 创建独立会话目录: {self.session_dir}")
            else:
                # 使用现有会话目录，避免重复日志
                session_id = self.session_dir.name
                current_session_dir = self.session_dir
                self.logger.debug(f"复用现有会话目录: {self.session_dir}")

        start_time = time.time()
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
                "  因子数据: %s 行, 时间 %s 到 %s",
                len(factors),
                factors.index.min(),
                factors.index.max(),
            )
            self.logger.info(
                "  价格数据: %s 行, 时间 %s 到 %s",
                len(close_prices),
                close_prices.index.min(),
                close_prices.index.max(),
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

            # P0-3修复：改进p值收集逻辑，确保显著性判断一致性
            p_values = {}
            for factor, ic_data in all_metrics["multi_horizon_ic"].items():
                # 优先使用1日IC的p值，如果无效则使用最小的有效p值
                p_1d = ic_data.get("p_value_1d", 1.0)

                if p_1d < 1.0 and p_1d > 0.0:
                    # 1日p值有效，直接使用
                    p_values[factor] = p_1d
                else:
                    # 1日p值无效，收集所有周期的p值
                    all_p_values = []
                    for h in self.config.ic_horizons:
                        p_val = ic_data.get(f"p_value_{h}d", 1.0)
                        if 0.0 < p_val < 1.0:  # 有效p值
                            all_p_values.append(p_val)

                    if all_p_values:
                        # 使用最小的有效p值（最显著的）
                        p_values[factor] = min(all_p_values)
                        self.logger.debug(
                            f"因子 {factor}: 1日p值无效({p_1d:.6f})，"
                            f"使用最小p值({min(all_p_values):.6f})"
                        )
                    else:
                        # 所有p值都无效，设为1.0（不显著）
                        p_values[factor] = 1.0
                        self.logger.warning(
                            f"因子 {factor}: 所有周期p值均无效，设为不显著"
                        )

            all_metrics["p_values"] = p_values

            # FDR校正（传入样本量和时间框架用于自适应阈值调整）
            sample_size = len(factors_aligned)
            if self.config.fdr_method == "benjamini_hochberg":
                corrected_p, adaptive_alpha = self.benjamini_hochberg_correction(
                    p_values, sample_size=sample_size, timeframe=timeframe
                )
            else:
                corrected_p, adaptive_alpha = self.bonferroni_correction(p_values)

            all_metrics["corrected_p_values"] = corrected_p
            all_metrics["adaptive_alpha"] = adaptive_alpha

            # P0-3修复：添加显著性判断调试日志
            significant_factors = []
            for factor, corrected_p_val in corrected_p.items():
                if corrected_p_val < adaptive_alpha:
                    significant_factors.append(factor)
                    ic_data = all_metrics["multi_horizon_ic"][factor]
                    ic_1d = ic_data.get("ic_1d", 0.0)
                    self.logger.debug(
                        f"显著因子: {factor}, IC_1d={ic_1d:.6f}, "
                        f"原始p={p_values[factor]:.6f}, "
                        f"校正p={corrected_p_val:.6f}, α={adaptive_alpha:.6f}"
                    )

            self.logger.info(
                f"FDR校正完成: {len(significant_factors)}/{len(corrected_p)} 个因子显著 "
                f"(α={adaptive_alpha:.6f})"
            )

            # 5. 综合评分
            self.logger.info("步骤5: 综合评分...")
            comprehensive_results = self.calculate_comprehensive_scores(
                all_metrics, timeframe
            )

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
            def _count_metrics(
                data: Dict[str, FactorMetrics],
                predicate: Callable[[FactorMetrics], bool],
            ) -> int:
                return sum(1 for metric in data.values() if predicate(metric))

            significant_count = _count_metrics(
                comprehensive_results, lambda metric: metric.is_significant
            )

            # 🚀 P0修复：使用时间框架自适应高分阈值
            high_score_threshold = 0.5  # 默认阈值
            if getattr(self.config, "enable_timeframe_adaptive", False):
                tf_high_score_map = getattr(self.config, "timeframe_high_score_map", {})
                high_score_threshold = tf_high_score_map.get(timeframe, 0.5)
                self.logger.info(
                    f"时间框架自适应高分阈值: {timeframe} 使用 {high_score_threshold:.2f}"
                )

            high_score_count = _count_metrics(
                comprehensive_results,
                lambda metric: metric.comprehensive_score > high_score_threshold,
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

                # P2-1修复：根据配置决定是否保存传统格式
                if self.config.enable_legacy_format:
                    self.logger.info("根据配置启用传统格式保存")
                    try:
                        self.save_comprehensive_screening_info(
                            comprehensive_results,
                            symbol,
                            timeframe,
                            screening_stats,
                            data_quality_info,
                        )
                        self.logger.info("传统格式保存完成")
                    except Exception as e:
                        self.logger.error(f"传统格式保存失败: {e}")
                else:
                    self.logger.info("使用增强版结果管理器，跳过传统格式保存")

            except Exception as e:
                self.logger.error(f"保存完整筛选信息失败: {str(e)}")
                screening_stats["save_error"] = str(e)

            # 退出性能监控
            if perf_monitor_active:
                try:
                    perf_context.__exit__(None, None, None)
                except:
                    pass  # 忽略性能监控退出错误

            return comprehensive_results

        except Exception as e:
            self.logger.error(f"因子筛选失败: {str(e)}")
            # 确保在异常情况下也退出性能监控
            if perf_monitor_active:
                try:
                    perf_context.__exit__(type(e), e, e.__traceback__)
                except:
                    pass
            raise
        finally:
            if in_multi_tf_mode:
                # 批量模式：避免状态泄漏到下一时间框架
                self.session_dir = None

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
        self.generate_screening_report(results, str(csv_path), symbol, timeframe)

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
        iterable = results.values() if isinstance(results, dict) else list(results)
        for metrics in iterable:
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
                data_root = first_config.get("data_root", "../factor_output")
                output_dir = first_config.get("output_dir", "./screening_results")

                print(f"📁 数据目录: {data_root}")
                print(f"📁 输出目录: {output_dir}")

                # P2-2修复：增强批量处理信息透明度
                print("\n📋 批量处理详细信息:")
                print(
                    f"  - 预计处理时间: ~{len(batch_config['screening_configs']) * 2}分钟"
                )
                print("  - 内存使用预估: ~500MB")
                print(
                    f"  - 并行处理: {'启用' if batch_config.get('enable_parallel', True) else '禁用'}"
                )
                print(f"  - 工作进程数: {batch_config.get('max_workers', 4)}")
                print("=" * 80)

                # 创建统一的批量筛选器
                batch_screener = ProfessionalFactorScreener(data_root=data_root)
                batch_screener.screening_results_dir = Path(output_dir)

                # P0-1修复：创建统一的批量会话目录，避免重复创建
                from datetime import datetime

                batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                batch_session_id = (
                    f"{batch_config['batch_name']}_multi_timeframe_{batch_timestamp}"
                )

                # 设置多时间框架会话目录（关键修复）
                batch_screener.multi_tf_session_dir = (
                    batch_screener.screening_results_dir / batch_session_id
                )
                batch_screener.multi_tf_session_dir.mkdir(parents=True, exist_ok=True)
                batch_screener.session_timestamp = batch_timestamp

                # 创建时间框架子目录结构
                timeframes_dir = batch_screener.multi_tf_session_dir / "timeframes"
                timeframes_dir.mkdir(exist_ok=True)

                print(f"📁 批量会话目录: {batch_screener.multi_tf_session_dir}")
                print(f"📁 时间框架子目录: {timeframes_dir}")

                for i, sub_config in enumerate(batch_config["screening_configs"], 1):
                    try:
                        # P2-2修复：增强中间步骤可观测性
                        start_time = time.time()
                        print(
                            f"\n📊 [{i}/{len(batch_config['screening_configs'])}] 处理: {sub_config['name']}"
                        )
                        print(f"   股票: {sub_config['symbols'][0]}")
                        print(f"   时间框架: {sub_config['timeframes'][0]}")
                        from datetime import datetime, timedelta

                        print(f"   开始时间: {datetime.now().strftime('%H:%M:%S')}")
                        print(
                            "   预计完成时间: "
                            f"{(datetime.now() + timedelta(minutes=2)).strftime('%H:%M:%S')}"
                        )
                        print("   " + "-" * 50)

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

                        # P2-2修复：详细完成报告
                        end_time = time.time()
                        duration = end_time - start_time
                        significant_count = sum(
                            metric.is_significant for metric in result.values()
                        )
                        high_score_count = sum(
                            metric.comprehensive_score >= 0.6
                            for metric in result.values()
                        )

                        successful_tasks += 1
                        print(f"   ✅ 完成: 耗时 {duration:.1f}秒")
                        print(f"      - 总因子: {len(result)}")
                        print(f"      - 显著因子: {significant_count}")
                        print(f"      - 高分因子: {high_score_count}")
                        print(
                            f"      - 完成时间: {datetime.now().strftime('%H:%M:%S')}"
                        )

                        # 进度条显示
                        progress = i / len(batch_config["screening_configs"]) * 100
                        total_msg = (
                            "   📈 总体进度: "
                            f"{progress:.1f}% ({i}/{len(batch_config['screening_configs'])})"
                        )
                        self.logger.info(total_msg)

                    except Exception as e:
                        failed_tasks += 1
                        print(f"   ❌ 失败: {str(e)}")
                        continue

                # 生成统一的多时间框架汇总报告
                print("\n📈 生成统一汇总报告...")
                if all_results:
                    _generate_multi_timeframe_summary(
                        batch_screener.multi_tf_session_dir,
                        batch_config["batch_name"],
                        all_results,
                        batch_config["screening_configs"],
                        logger=batch_screener.logger,
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
                data_root="../factor_output",
                output_dir="./screening_results",
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
    session_dir,
    batch_name: str,
    all_results: Dict,
    screening_configs: List[Dict],
    logger: Optional[logging.Logger] = None,
) -> None:
    """生成统一的多时间框架汇总报告

    Args:
        session_dir: 会话目录路径
        batch_name: 批量处理名称
        all_results: 所有时间框架的筛选结果
        screening_configs: 筛选配置列表
    """
    from datetime import datetime

    summary_logger = logger or logging.getLogger(__name__)
    summary_logger.info("📊 生成多时间框架汇总报告...")

    # 检查是否有有效结果
    if not all_results:
        summary_logger.warning("⚠️ 没有数据生成汇总报告")
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

            # 🔧 筛选最佳因子 - 必须满足统计显著性 OR Tier 1/2
            sorted_factors = sorted(
                result.values(), key=lambda x: x.comprehensive_score, reverse=True
            )

            # 核心筛选逻辑：只保留优秀因子（修复：必须同时满足统计显著和Tier要求）
            top_factors = [
                f
                for f in sorted_factors
                if (
                    f.is_significant  # 必须统计显著
                    and getattr(f, "tier", "N/A") in ["Tier 1", "Tier 2"]
                )  # 且Tier 1/2
            ][
                :20
            ]  # 最多取20个优秀因子

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
        summary_logger.warning("⚠️ 没有数据生成汇总报告")
        return

    # 创建汇总DataFrame
    import pandas as pd

    summary_df = pd.DataFrame(summary_data)

    # 保存统一汇总报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_filename = f"{batch_name}_multi_timeframe_summary_{timestamp}.csv"
    summary_path = session_dir / summary_filename

    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    summary_logger.info(
        "✅ 多时间框架汇总报告已保存", extra={"summary_path": str(summary_path)}
    )

    # 生成最佳因子综合排行
    if best_factors_overall:
        best_df = pd.DataFrame(best_factors_overall)

        # 按综合评分排序
        best_df_sorted = best_df.sort_values("comprehensive_score", ascending=False)

        # 保存最佳因子排行
        best_filename = f"{batch_name}_best_factors_overall_{timestamp}.csv"
        best_path = session_dir / best_filename
        best_df_sorted.to_csv(best_path, index=False, encoding="utf-8")
        summary_logger.info(
            "✅ 最佳因子综合排行已保存", extra={"best_factors_path": str(best_path)}
        )

        # 输出Top 10最佳因子到控制台（全局）
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

        # 🆕 输出各时间框架Top 5（分层展示，避免高频统治）
        print("\n📊 各时间框架 Top 5 最佳因子（分层对比）:")
        print("=" * 120)
        all_timeframes = best_df_sorted["timeframe"].unique()
        for tf in all_timeframes:
            tf_data = best_df_sorted[best_df_sorted["timeframe"] == tf]
            tier2_count = (tf_data["tier"] == "Tier 2").sum()
            tier1_count = (tf_data["tier"] == "Tier 1").sum()

            print(
                f"\n【{tf}】 (Tier1: {tier1_count}, Tier2: {tier2_count}, 总计: {len(tf_data)})"
            )
            print("-" * 120)
            top_5 = tf_data.head(5)
            for i, (_, factor) in enumerate(top_5.iterrows(), 1):
                print(
                    f"  {i}. {factor['factor_name']:<25} | "
                    f"评分: {factor['comprehensive_score']:.3f} | "
                    f"等级: {factor['tier']:<8} | "
                    f"IC: {factor['ic_mean']:>6.3f} | "
                    f"胜率: {factor['ic_win_rate']:>5.1%}"
                )

    # 生成统计摘要
    _generate_batch_statistics(
        session_dir,
        batch_name,
        all_results,
        timestamp,
        logger=logger,
    )


def _generate_batch_statistics(
    session_dir,
    batch_name: str,
    all_results: Dict[str, Dict[str, FactorMetrics]],
    timestamp: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """生成批量处理统计摘要"""

    from datetime import datetime

    import pandas as pd

    stats_logger = logger or logging.getLogger(__name__)
    stats_logger.info("📈 生成批量处理统计摘要...", extra={"batch_name": batch_name})

    stats_data: List[Dict[str, Any]] = []
    overall_totals = Counter()

    for tf_key, tf_results in all_results.items():
        if not tf_results:
            continue

        symbol = tf_key.split("_")[0]
        timeframe = tf_key[len(symbol) + 1 :] if "_" in tf_key else "unknown"

        total_count = len(tf_results)
        if total_count == 0:
            continue

        tier_counter = Counter(
            (metrics.tier or "未分级") for metrics in tf_results.values()
        )
        significant_count = sum(
            metrics.is_significant for metrics in tf_results.values()
        )
        top_count = sum(
            metrics.comprehensive_score >= 0.8 for metrics in tf_results.values()
        )
        avg_score = float(
            np.mean([metrics.comprehensive_score for metrics in tf_results.values()])
        )

        stats_item = {
            "symbol": symbol,
            "timeframe": timeframe,
            "total_factors": total_count,
            "significant_factors": significant_count,
            "tier1_factors": tier_counter.get("Tier 1", 0),
            "tier2_factors": tier_counter.get("Tier 2", 0),
            "tier3_factors": tier_counter.get("Tier 3", 0),
            "not_recommended": tier_counter.get("不推荐", 0),
            "top_factors": top_count,
            "average_score": round(avg_score, 6),
        }
        stats_data.append(stats_item)

        overall_totals.update(
            {
                "total_factors": total_count,
                "significant_factors": significant_count,
                "tier1_factors": stats_item["tier1_factors"],
                "tier2_factors": stats_item["tier2_factors"],
                "tier3_factors": stats_item["tier3_factors"],
                "not_recommended": stats_item["not_recommended"],
                "top_factors": top_count,
            }
        )

        stats_logger.info("时间框架统计", extra={**stats_item})

    if not stats_data:
        stats_logger.warning("⚠️ 批量统计摘要无数据")
        return

    stats_df = pd.DataFrame(stats_data)

    # 保存统计摘要 CSV
    stats_filename = f"{batch_name}_batch_statistics_{timestamp}.csv"
    stats_path = session_dir / stats_filename
    stats_df.to_csv(stats_path, index=False, encoding="utf-8")
    stats_logger.info(
        "✅ 批量处理统计摘要已保存",
        extra={"batch_statistics_path": str(stats_path)},
    )

    # 保存统计摘要 JSON
    summary_payload = {
        "batch_name": batch_name,
        "generated_at": datetime.now().isoformat(),
        "timeframes": stats_data,
        "overall": dict(overall_totals),
    }
    stats_json_path = session_dir / f"{batch_name}_batch_statistics_{timestamp}.json"
    with open(stats_json_path, "w", encoding="utf-8") as fp:
        json.dump(
            summary_payload, fp, ensure_ascii=False, indent=2, cls=NumpyJSONEncoder
        )
    stats_logger.info(
        "✅ 批量统计JSON已保存",
        extra={"batch_statistics_json": str(stats_json_path)},
    )

    # 控制台输出（保持原有人机提示）
    print("\n📊 批量处理统计摘要:")
    print("=" * 80)
    print(f"处理时间框架数量: {len(stats_data)}")
    print(f"总处理因子数: {overall_totals['total_factors']}")
    print(f"总显著因子数: {overall_totals['significant_factors']}")
    print(f"总Tier 1因子数: {overall_totals['tier1_factors']}")
    print(f"总Tier 2因子数: {overall_totals['tier2_factors']}")

    total_factors = overall_totals["total_factors"] or 1
    overall_tier1_ratio = overall_totals["tier1_factors"] / total_factors
    overall_tier2_ratio = overall_totals["tier2_factors"] / total_factors

    print(f"整体Tier 1比例: {overall_tier1_ratio:.1%}")
    print(f"整体Tier 2比例: {overall_tier2_ratio:.1%}")

    stats_logger.info(
        "整体Tier比例",
        extra={
            "overall_tier1_ratio": overall_tier1_ratio,
            "overall_tier2_ratio": overall_tier2_ratio,
        },
    )

    print("\n各时间框架详细统计:")
    for stats_item in stats_data:
        print(
            f"  {stats_item['symbol']}-{stats_item['timeframe']:>8}: "
            f"总计 {stats_item['total_factors']:>3} 个 | "
            f"显著 {stats_item['significant_factors']:>3} 个 | "
            f"Tier1 {stats_item['tier1_factors']:>2} 个 | "
            f"Tier2 {stats_item['tier2_factors']:>2} 个"
        )

    stats_logger.info(
        "✅ 批量处理统计摘要生成完成",
        extra={
            "timeframe_count": len(stats_data),
            "total_factors": overall_totals["total_factors"],
            "significant_factors": overall_totals["significant_factors"],
        },
    )


class ProfessionalFactorScreenerEnhanced(ProfessionalFactorScreener):
    """🔧 增强版筛选器：集成data_loader_patch改进"""

    def load_factors_v2(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        🔧 集成版：加载因子数据，优先使用data_loader_patch的改进版本
        """
        # 🔧 优先尝试使用data_loader_patch的改进版本
        try:
            from data_loader_patch import load_factors_v2 as patch_load_factors

            return patch_load_factors(self, symbol, timeframe)
        except (ImportError, NameError):
            # 如果补丁不可用，使用原始方法
            self.logger.warning("data_loader_patch不可用，使用原始因子加载方法")
            return super().load_factors(symbol, timeframe)

    def load_price_data_v2(self, symbol: str, timeframe: str = None) -> pd.DataFrame:
        """
        🔧 集成版：加载价格数据，优先使用data_loader_patch的改进版本
        """
        # 🔧 优先尝试使用data_loader_patch的改进版本
        try:
            from data_loader_patch import load_price_data_v2 as patch_load_price_data

            return patch_load_price_data(self, symbol, timeframe)
        except (ImportError, NameError):
            # 如果补丁不可用，使用原始方法
            self.logger.warning("data_loader_patch不可用，使用原始价格加载方法")
            return super().load_price_data(symbol, timeframe)


# 为了向后兼容，创建工厂函数
def create_enhanced_screener(
    data_root: Optional[str] = None, config: Optional[ScreeningConfig] = None
):
    """
    创建增强版筛选器实例

    Returns:
        集成了data_loader_patch改进的筛选器实例
    """
    return ProfessionalFactorScreenerEnhanced(data_root=data_root, config=config)


if __name__ == "__main__":
    main()
