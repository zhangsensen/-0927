#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""并行回测配置加载和验证模块

为向量化并行引擎提供完整的配置抽象支持
"""

import multiprocessing as mp
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ParallelBacktestConfig:
    """并行回测配置数据类 - 扩展原有配置支持并行计算"""

    # === 数据路径配置 ===
    panel_file: str
    price_dir: str
    screening_file: str
    output_dir: str = (
        "/Users/zhangshenshen/深度量化0927/etf_rotation_system/data/results/backtest"
    )

    # === 并行计算配置 ===
    n_workers: int = field(default_factory=lambda: max(1, mp.cpu_count() - 1))
    chunk_size: int = 20
    enable_cache: bool = True
    log_level: str = "INFO"

    # === 因子配置 ===
    top_k: int = 10
    factors: List[str] = field(default_factory=list)

    # === 回测参数配置 ===
    # 注意：top_n_list 应从 YAML 配置读取，此处不设置默认值（在_dict_to_config中处理）
    top_n_list: List[int] = field(default_factory=lambda: [])
    rebalance_freq_list: list = field(default_factory=lambda: [5, 10, 20])
    fees: float = 0.003  # A股 ETF: 佣金0.2% + 印花税0.1% = 0.3% 往返
    init_cash: float = 1000000

    # === A股 ETF 成本模型 ===
    commission_rate: float = 0.002  # 佣金 0.2%
    stamp_duty_rate: float = 0.001  # 印花税 0.1% (仅卖出时)
    slippage_amount: float = 0.0001  # 滑点 0.01% 成交额

    # === 权重网格配置 ===
    weight_grid_points: List[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    weight_sum_range: List[float] = field(default_factory=lambda: [0.8, 1.2])
    max_combinations: int = 1000  # 🔧 修复: 默认1000而不是10000

    # === 复合因子计算配置 ===
    standardization_method: str = "zscore"
    enable_score_cache: bool = True
    numerical_epsilon: float = 1e-8

    # === 向量化优化配置 ===
    max_memory_usage_gb: float = 16.0
    enable_gc: bool = True
    checkpoint_interval: int = 5000
    use_float32: bool = False
    batch_processing_size: int = 1000

    # === 性能优化配置 ===
    omp_num_threads: int = 1
    veclib_maximum_threads: int = 1
    mkl_num_threads: int = 1
    enable_progress_bar: bool = True

    # === 输出配置 ===
    save_top_results: int = 50
    save_best_config: bool = True
    save_detailed_results: bool = True
    results_prefix: str = "parallel_backtest_results"
    best_config_prefix: str = "parallel_best_strategy"

    # === 调试和日志配置 ===
    verbose: bool = False
    log_errors: bool = True
    save_intermediate: bool = False
    log_to_file: bool = False
    log_dir: str = "/tmp"
    console_output: bool = True

    # === 约束配置 ===
    min_trade_days: int = 252
    max_single_weight: float = 0.8
    min_effective_symbols: int = 3

    # === 指标配置 ===
    primary_metric: str = "sharpe_ratio"
    periods_per_year: int = 252
    min_sharpe_ratio: float = 0.5
    max_drawdown_threshold: float = -30

    # === 场景预设名称 ===
    current_preset: Optional[str] = None


@dataclass
class FastConfig:
    """零开销配置类 - 编译时常量，消除运行时解析开销"""

    # === 数据路径配置 ===
    panel_file: str = (
        "/Users/zhangshenshen/深度量化0927/etf_rotation_system/data/results/panels/panel_20251024_203314/panel.parquet"
    )
    price_dir: str = "/Users/zhangshenshen/深度量化0927/raw/ETF/daily"
    screening_file: str = (
        "/Users/zhangshenshen/深度量化0927/etf_rotation_system/data/results/screening/screening_20251024_203438/passed_factors.csv"
    )
    output_dir: str = (
        "/Users/zhangshenshen/深度量化0927/etf_rotation_system/data/results/backtest"
    )

    # === 并行计算配置 ===
    n_workers: int = max(1, mp.cpu_count() - 1)
    chunk_size: int = 20
    enable_cache: bool = True
    log_level: str = "INFO"

    # === 因子配置 ===
    top_k: int = 9  # 更新到9个验证通过的因子
    factors: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.factors = self._load_passed_factors(self.screening_file)

    @staticmethod
    def _load_passed_factors(screening_file: str) -> list:
        import os

        import pandas as pd

        if screening_file and os.path.exists(screening_file):
            try:
                df = pd.read_csv(screening_file)
                factors = df.iloc[:, 0].dropna().tolist()
                # 去除标题行（如有）
                if factors and factors[0].lower().startswith("factor"):
                    factors = factors[1:]
                if len(factors) > 0:
                    print(f"[INFO] 自动加载筛选因子: {factors}")
                    return factors
            except Exception as e:
                print(f"[WARN] 读取筛选因子失败: {e}")
        # 兜底：返回最新验证的9个因子
        print("[WARN] 筛选结果文件缺失或异常，回退到最新验证的9因子列表")
        return [
            "PRICE_POSITION_60D",
            "MOM_ACCEL",
            "DISTANCE_TO_52W_HIGH",
            "VOLATILITY_REGIME_SHIFT",
            "WR_14",
            "DRAWDOWN_RECOVERY_SPEED",
            "VOL_VOLATILITY_20",
            "VOLUME_PRICE_TREND",
            "SHADOW_RATIO",
        ]

    # === 回测参数配置 ===
    # 注意：top_n_list 应从 YAML 配置读取，如YAML缺失则使用默认值
    top_n_list: List[int] = field(default_factory=lambda: [3, 5, 8, 9])
    rebalance_freq_list: list = field(default_factory=lambda: [5, 10, 20])
    fees: float = 0.003  # A股 ETF: 0.3% 往返手续费
    init_cash: float = 1000000

    # === A股 ETF 成本模型 ===
    commission_rate: float = 0.002  # 佣金 0.2%
    stamp_duty_rate: float = 0.001  # 印花税 0.1%
    slippage_amount: float = 0.0001  # 滑点 0.01%

    # === 权重网格配置 ===
    weight_grid_points: List[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    weight_sum_range: List[float] = field(default_factory=lambda: [0.8, 1.2])
    max_combinations: int = 10000

    # === 复合因子计算配置 ===
    standardization_method: str = "zscore"
    enable_score_cache: bool = True
    numerical_epsilon: float = 1e-8

    # === 向量化优化配置 ===
    max_memory_usage_gb: float = 16.0
    enable_gc: bool = True
    checkpoint_interval: int = 5000
    use_float32: bool = False
    batch_processing_size: int = 1000

    # === 性能优化配置 ===
    omp_num_threads: int = 1
    veclib_maximum_threads: int = 1
    mkl_num_threads: int = 1
    enable_progress_bar: bool = True

    # === 输出配置 ===
    save_top_results: int = 50
    save_best_config: bool = True
    save_detailed_results: bool = True
    results_prefix: str = "parallel_backtest_results"
    best_config_prefix: str = "parallel_best_strategy"

    # === 调试和日志配置 ===
    verbose: bool = False
    log_errors: bool = True
    save_intermediate: bool = False
    log_to_file: bool = False
    log_dir: str = "/tmp"
    console_output: bool = True

    # === 约束配置 ===
    min_trade_days: int = 252
    max_single_weight: float = 0.8
    min_effective_symbols: int = 3

    # === 指标配置 ===
    primary_metric: str = "sharpe_ratio"
    periods_per_year: int = 252
    min_sharpe_ratio: float = 0.5
    max_drawdown_threshold: float = -30


class ParallelConfigLoader:
    """并行回测配置加载器"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置加载器

        Args:
            config_path: 配置文件路径，默认为当前目录下的parallel_backtest_config.yaml
        """
        if config_path is None:
            config_path = Path(__file__).parent / "parallel_backtest_config.yaml"

        self.config_path = Path(config_path)
        self.presets = {}
        self.config = None

        # 配置缓存 - 避免重复解析YAML文件
        self._config_cache = {}
        self._config_file_mtime = None

    def _safe_float_convert(self, value: Any) -> float:
        """安全地将值转换为浮点数，处理YAML科学计数法字符串问题"""
        if isinstance(value, float):
            return value
        elif isinstance(value, (int,)):
            return float(value)
        elif isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                # 如果转换失败，尝试处理科学计数法
                try:
                    import decimal

                    return float(decimal.Decimal(value))
                except:
                    return float(value.replace("e", "E"))  # 尝试修复科学计数法格式
        else:
            return float(value)  # 最后尝试直接转换

    def load_config(self, preset_name: Optional[str] = None) -> ParallelBacktestConfig:
        """
        加载配置文件（带缓存优化）

        Args:
            preset_name: 预设名称，如果提供则应用预设

        Returns:
            配置对象
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        # 检查文件修改时间和缓存
        current_mtime = self.config_path.stat().st_mtime
        cache_key = f"{self.config_path}_{preset_name or 'default'}"

        if cache_key in self._config_cache and self._config_file_mtime == current_mtime:
            # 使用缓存的配置
            config_data = self._config_cache[cache_key]
        else:
            # 加载并缓存配置
            with open(self.config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            # 缓存原始配置数据
            self._config_cache[cache_key] = config_data
            self._config_file_mtime = current_mtime

        # 应用预设（如果指定）
        if (
            preset_name
            and "presets" in config_data
            and preset_name in config_data["presets"]
        ):
            preset_config = config_data["presets"][preset_name]
            # 深度合并预设配置
            config_data = self._deep_merge_configs(config_data, preset_config)
            config_data["current_preset"] = preset_name

        # 转换为配置对象
        self.config = self._dict_to_config(config_data)
        self._validate_config()

        return self.config

    def _dict_to_config(self, config_data: Dict[str, Any]) -> ParallelBacktestConfig:
        """将字典转换为配置对象"""
        # 提取各部分配置
        data_paths = config_data.get("data_paths", {})
        factor_config = config_data.get("factor_config", {})
        backtest_config = config_data.get("backtest_config", {})
        # 🔧 修复: 支持从backtest_config.weight_grid读取（新版YAML）或从顶层weight_grid读取（旧版兼容）
        weight_grid = backtest_config.get(
            "weight_grid", config_data.get("weight_grid", {})
        )
        composite_config = config_data.get("composite_config", {})
        parallel_config = config_data.get("parallel_config", {})
        vectorization_config = config_data.get("vectorization_config", {})
        performance_config = config_data.get("performance_config", {})
        output_config = config_data.get("output_config", {})
        debug_config = config_data.get("debug_config", {})
        constraints = config_data.get("constraints", {})
        metrics = config_data.get("metrics", {})

        return ParallelBacktestConfig(
            # 数据路径
            panel_file=data_paths.get("panel_file", ""),
            price_dir=data_paths.get("price_dir", ""),
            screening_file=data_paths.get("screening_file", ""),
            output_dir=data_paths.get(
                "output_dir",
                "/Users/zhangshenshen/深度量化0927/etf_rotation_system/data/results/backtest",
            ),
            # 并行计算配置
            n_workers=parallel_config.get("n_workers", max(1, mp.cpu_count() - 1)),
            chunk_size=parallel_config.get("chunk_size", 20),
            enable_cache=parallel_config.get("enable_cache", True),
            log_level=parallel_config.get("log_level", "INFO"),
            # 因子配置
            top_k=factor_config.get("top_k", 10),
            factors=factor_config.get("factors", []),
            # 回测参数 - top_n_list 必须从 YAML 配置提供，不使用硬编码默认值
            top_n_list=backtest_config.get("top_n_list", []),
            rebalance_freq_list=backtest_config.get("rebalance_freq_list", [5, 10, 20]),
            fees=self._safe_float_convert(backtest_config.get("fees", 0.001)),
            init_cash=self._safe_float_convert(
                backtest_config.get("init_cash", 1000000)
            ),
            # 权重网格
            weight_grid_points=[
                self._safe_float_convert(x)
                for x in weight_grid.get(
                    "grid_points",
                    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                )
            ],
            weight_sum_range=[
                self._safe_float_convert(x)
                for x in weight_grid.get("weight_sum_range", [0.8, 1.2])
            ],
            max_combinations=weight_grid.get(
                "max_combinations", 1000
            ),  # 🔧 修复: 默认1000
            # 复合因子计算
            standardization_method=composite_config.get(
                "standardization_method", "zscore"
            ),
            enable_score_cache=composite_config.get("enable_score_cache", True),
            numerical_epsilon=self._safe_float_convert(
                composite_config.get("numerical_epsilon", 1e-8)
            ),
            # 向量化优化
            max_memory_usage_gb=self._safe_float_convert(
                vectorization_config.get("max_memory_usage_gb", 16.0)
            ),
            enable_gc=vectorization_config.get("enable_gc", True),
            checkpoint_interval=vectorization_config.get("checkpoint_interval", 5000),
            use_float32=vectorization_config.get("use_float32", False),
            batch_processing_size=vectorization_config.get(
                "batch_processing_size", 1000
            ),
            # 性能优化
            omp_num_threads=performance_config.get("omp_num_threads", 1),
            veclib_maximum_threads=performance_config.get("veclib_maximum_threads", 1),
            mkl_num_threads=performance_config.get("mkl_num_threads", 1),
            enable_progress_bar=performance_config.get("enable_progress_bar", True),
            # 输出配置
            save_top_results=output_config.get("save_top_results", 50),
            save_best_config=output_config.get("save_best_config", True),
            save_detailed_results=output_config.get("save_detailed_results", True),
            results_prefix=output_config.get(
                "results_prefix", "parallel_backtest_results"
            ),
            best_config_prefix=output_config.get(
                "best_config_prefix", "parallel_best_strategy"
            ),
            # 调试配置
            verbose=debug_config.get("verbose", False),
            log_errors=debug_config.get("log_errors", True),
            save_intermediate=debug_config.get("save_intermediate", False),
            log_to_file=debug_config.get("log_to_file", False),
            log_dir=debug_config.get("log_dir", "/tmp"),
            console_output=debug_config.get("console_output", True),
            # 约束配置
            min_trade_days=constraints.get("min_trade_days", 252),
            max_single_weight=constraints.get("max_single_weight", 0.8),
            min_effective_symbols=constraints.get("min_effective_symbols", 3),
            # 指标配置
            primary_metric=metrics.get("primary_metric", "sharpe_ratio"),
            periods_per_year=metrics.get("periods_per_year", 252),
            min_sharpe_ratio=metrics.get("min_sharpe_ratio", 0.5),
            max_drawdown_threshold=metrics.get("max_drawdown_threshold", -30),
            # 当前预设
            current_preset=config_data.get("current_preset"),
        )

    def _validate_config(self) -> None:
        """快速验证配置的关键参数"""
        if not self.config:
            raise ValueError("配置未加载")

        # 只验证关键路径和参数，减少验证开销
        if not self.config.panel_file:
            raise ValueError("panel_file 不能为空")
        if self.config.n_workers <= 0:
            raise ValueError("n_workers 必须大于0")
        if self.config.max_combinations <= 0:
            raise ValueError("max_combinations 必须大于0")
        if self.config.weight_sum_range[0] >= self.config.weight_sum_range[1]:
            raise ValueError("weight_sum_range[0] 必须小于 weight_sum_range[1]")

        # 验证Top-N列表
        if not self.config.top_n_list:
            raise ValueError("top_n_list 不能为空")
        if any(n <= 0 for n in self.config.top_n_list):
            raise ValueError("top_n_list 中的值必须大于0")

    def _deep_merge_configs(
        self, base_config: Dict[str, Any], preset_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """深度合并配置字典"""
        result = base_config.copy()

        for key, value in preset_config.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def get_available_presets(self) -> List[str]:
        """获取可用的预设列表"""
        if not self.config_path.exists():
            return []

        with open(self.config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        return list(config_data.get("presets", {}).keys())

    def save_config(
        self, config: ParallelBacktestConfig, output_path: Optional[str] = None
    ) -> None:
        """保存配置到文件"""
        if output_path is None:
            output_path = self.config_path

        # 转换为字典格式
        config_dict = {
            "data_paths": {
                "panel_file": config.panel_file,
                "price_dir": config.price_dir,
                "screening_file": config.screening_file,
                "output_dir": config.output_dir,
            },
            "parallel_config": {
                "n_workers": config.n_workers,
                "chunk_size": config.chunk_size,
                "enable_cache": config.enable_cache,
                "log_level": config.log_level,
            },
            "factor_config": {"top_k": config.top_k, "factors": config.factors},
            "backtest_config": {
                "top_n_list": config.top_n_list,
                "rebalance_freq": config.rebalance_freq,
                "fees": config.fees,
                "init_cash": config.init_cash,
            },
            "weight_grid": {
                "grid_points": config.weight_grid_points,
                "weight_sum_range": config.weight_sum_range,
                "max_combinations": config.max_combinations,
            },
            "composite_config": {
                "standardization_method": config.standardization_method,
                "enable_score_cache": config.enable_score_cache,
                "numerical_epsilon": config.numerical_epsilon,
            },
            "vectorization_config": {
                "max_memory_usage_gb": config.max_memory_usage_gb,
                "enable_gc": config.enable_gc,
                "checkpoint_interval": config.checkpoint_interval,
                "use_float32": config.use_float32,
                "batch_processing_size": config.batch_processing_size,
            },
            "performance_config": {
                "omp_num_threads": config.omp_num_threads,
                "veclib_maximum_threads": config.veclib_maximum_threads,
                "mkl_num_threads": config.mkl_num_threads,
                "enable_progress_bar": config.enable_progress_bar,
            },
            "output_config": {
                "save_top_results": config.save_top_results,
                "save_best_config": config.save_best_config,
                "save_detailed_results": config.save_detailed_results,
                "results_prefix": config.results_prefix,
                "best_config_prefix": config.best_config_prefix,
            },
            "debug_config": {
                "verbose": config.verbose,
                "log_errors": config.log_errors,
                "save_intermediate": config.save_intermediate,
                "log_to_file": config.log_to_file,
                "log_dir": config.log_dir,
                "console_output": config.console_output,
            },
            "constraints": {
                "min_trade_days": config.min_trade_days,
                "max_single_weight": config.max_single_weight,
                "min_effective_symbols": config.min_effective_symbols,
            },
            "metrics": {
                "primary_metric": config.primary_metric,
                "periods_per_year": config.periods_per_year,
                "min_sharpe_ratio": config.min_sharpe_ratio,
                "max_drawdown_threshold": config.max_drawdown_threshold,
            },
            "current_preset": config.current_preset,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(
                config_dict, f, default_flow_style=False, allow_unicode=True, indent=2
            )


def load_parallel_config_from_args(args) -> ParallelBacktestConfig:
    """从命令行参数加载配置"""
    loader = ParallelConfigLoader()

    # 如果指定了配置文件
    config_file = getattr(args, "config_file", None)
    preset_name = getattr(args, "preset", None)

    if config_file:
        loader.config_path = Path(config_file)

    return loader.load_config(preset_name)


def create_default_parallel_config(output_path: str) -> None:
    """创建默认的并行配置文件"""
    default_config = {
        "data_paths": {
            "panel_file": "/Users/zhangshenshen/深度量化0927/etf_rotation_system/data/results/panels/panel_20251022_155341/panel.parquet",
            "price_dir": "/Users/zhangshenshen/深度量化0927/raw/ETF/daily",
            "screening_file": "/Users/zhangshenshen/深度量化0927/etf_rotation_system/data/results/screening/screening_20251022_155407/passed_factors.csv",
            "output_dir": "/Users/zhangshenshen/深度量化0927/etf_rotation_system/data/results/backtest",
        },
        "parallel_config": {
            "n_workers": 9,
            "chunk_size": 20,
            "enable_cache": True,
            "log_level": "INFO",
        },
        "factor_config": {
            "top_k": 8,
            "factors": [
                "PRICE_POSITION_60D",
                "MOM_ACCEL",
                "VOLATILITY_120D",
                "VOL_VOLATILITY_20",
                "VOLUME_PRICE_TREND",
                "RSI_6",
                "INTRADAY_POSITION",
                "INTRA_DAY_RANGE",
            ],
        },
        "backtest_config": {
            "top_n_list": [2, 3, 4, 5, 6, 7],  # 从YAML读取配置，默认为示例值
            "rebalance_freq": 20,
            "fees": 0.001,
            "init_cash": 1000000,
        },
        "weight_grid": {
            "grid_points": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "weight_sum_range": [0.8, 1.2],
            "max_combinations": 1000,  # 🔧 修复: 默认1000
        },
        "composite_config": {
            "standardization_method": "zscore",
            "enable_score_cache": True,
            "numerical_epsilon": 1e-8,
        },
        "vectorization_config": {
            "max_memory_usage_gb": 16.0,
            "enable_gc": True,
            "checkpoint_interval": 5000,
            "use_float32": False,
            "batch_processing_size": 1000,
        },
        "performance_config": {
            "omp_num_threads": 1,
            "veclib_maximum_threads": 1,
            "mkl_num_threads": 1,
            "enable_progress_bar": True,
        },
        "presets": {
            "quick_test": {
                "weight_grid": {
                    "grid_points": [0.0, 0.5, 1.0],
                    "max_combinations": 100,
                },
                "backtest_config": {"top_n_list": [3, 5]},
            },
            "comprehensive": {
                "weight_grid": {
                    "grid_points": [
                        0.0,
                        0.05,
                        0.1,
                        0.15,
                        0.2,
                        0.25,
                        0.3,
                        0.35,
                        0.4,
                        0.45,
                        0.5,
                        0.55,
                        0.6,
                        0.65,
                        0.7,
                        0.75,
                        0.8,
                        0.85,
                        0.9,
                        0.95,
                        1.0,
                    ],
                    "max_combinations": 50000,
                    "weight_sum_range": [0.7, 1.3],
                },
                "backtest_config": {
                    "top_n_list": [2, 3, 4, 5, 6, 7]
                },  # 覆盖默认预设的Top-N范围
            },
        },
        "current_preset": None,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(
            default_config, f, default_flow_style=False, allow_unicode=True, indent=2
        )


def load_fast_config() -> FastConfig:
    """零开销配置加载 - 编译时常量，无运行时解析

    Returns:
        FastConfig: 预编译的配置对象，零开销访问
    """
    return FastConfig()


def load_fast_config_from_args(args) -> FastConfig:
    """从命令行参数加载快速配置（WFO增强版）

    Args:
        args: 命令行参数对象（可以是list如['-c', 'config.yaml']）

    Returns:
        FastConfig: 从YAML配置文件加载的配置对象或默认配置
    """
    # 兼容args为list的情况
    if isinstance(args, list):
        config_path = args[1] if len(args) > 1 else "parallel_backtest_config.yaml"
        preset_name = None
    else:
        config_path = (
            getattr(args, "config_file", None) or "parallel_backtest_config.yaml"
        )
        preset_name = getattr(args, "preset", None)

    # 尝试从YAML文件加载配置
    loader = ParallelConfigLoader(config_path)

    try:
        parallel_config = loader.load_config(preset_name=preset_name)

        # 读取原始YAML（用于WFO）
        from pathlib import Path

        import yaml

        with open(Path(config_path), "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        # 将ParallelBacktestConfig转换为FastConfig，确保top_n_list被正确加载
        fast_config = FastConfig()
        fast_config.top_n_list = parallel_config.top_n_list
        fast_config.rebalance_freq_list = parallel_config.rebalance_freq_list
        fast_config.panel_file = parallel_config.panel_file
        fast_config.price_dir = parallel_config.price_dir
        fast_config.screening_file = parallel_config.screening_file
        fast_config.output_dir = parallel_config.output_dir
        fast_config.n_workers = parallel_config.n_workers
        fast_config.fees = parallel_config.fees
        fast_config.init_cash = parallel_config.init_cash
        fast_config.weight_grid_points = parallel_config.weight_grid_points
        fast_config.max_combinations = parallel_config.max_combinations
        fast_config.save_top_results = parallel_config.save_top_results

        # 从backtest_config中提取新增的优化参数
        backtest_config = raw_config.get("backtest_config", {})
        fast_config.run_is = backtest_config.get("run_is", True)
        fast_config.run_oos = backtest_config.get("run_oos", True)
        fast_config.save_top_n = backtest_config.get("save_top_n", None)

        # 存储原始配置（供WFO使用）
        fast_config.raw_config = raw_config

        return fast_config
    except Exception as e:
        print(f"[WARNING] 加载配置文件失败: {e}，使用默认配置")
        return FastConfig()
