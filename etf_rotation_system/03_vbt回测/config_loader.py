#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VBT回测配置加载和验证模块

提供配置文件加载、验证、预设应用等功能
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class BacktestConfig:
    """回测配置数据类"""

    # 数据路径
    panel_file: str
    price_dir: str
    screening_file: str
    output_dir: str = "etf_rotation_system/03_vbt回测/results"

    # 因子配置
    top_k: int = 10
    factors: List[str] = field(default_factory=list)

    # 回测参数
    top_n_list: List[int] = field(default_factory=lambda: [3, 5, 8, 10])
    rebalance_freq: int = 20
    fees: float = 0.001
    init_cash: float = 1000000

    # 权重网格
    weight_grid_points: List[float] = field(
        default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0]
    )
    weight_sum_range: List[float] = field(default_factory=lambda: [0.7, 1.3])
    max_combinations: int = 10000

    # 复合因子计算
    standardization_method: str = "zscore"
    enable_score_cache: bool = True

    # 性能优化
    omp_num_threads: int = 2
    veclib_maximum_threads: int = 2
    mkl_num_threads: int = 2
    batch_size: int = 1000
    enable_progress_bar: bool = True

    # 输出配置
    save_top_results: int = 50
    save_best_config: bool = True
    save_detailed_results: bool = True
    results_prefix: str = "backtest_results"
    best_config_prefix: str = "best_strategy"

    # 调试配置
    verbose: bool = False
    log_errors: bool = True
    save_intermediate: bool = False
    # 日志输出配置
    log_to_file: bool = False
    log_dir: str = "/tmp"
    log_prefix: str = "backtest_log"
    console_output: bool = True

    # 约束配置
    min_trade_days: int = 252
    max_single_weight: float = 0.8
    min_effective_symbols: int = 3

    # 指标配置
    primary_metric: str = "sharpe_ratio"
    periods_per_year: int = 252
    min_sharpe_ratio: float = 0.5
    max_drawdown_threshold: float = -30


class ConfigLoader:
    """配置加载器"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置加载器

        Args:
            config_path: 配置文件路径，默认为当前目录下的backtest_config.yaml
        """
        if config_path is None:
            config_path = Path(__file__).parent / "backtest_config.yaml"

        self.config_path = Path(config_path)
        self.presets = {}
        self.config = None

    def load_config(
        self,
        preset_name: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> BacktestConfig:
        """
        加载配置文件

        Args:
            preset_name: 要应用的预设名称
            overrides: 配置覆盖字典

        Returns:
            BacktestConfig对象
        """
        # 加载YAML配置
        with open(self.config_path, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)

        # 应用预设
        if preset_name and "presets" in yaml_config:
            if preset_name not in yaml_config["presets"]:
                raise ValueError(
                    f"预设 '{preset_name}' 不存在。可用预设: {list(yaml_config['presets'].keys())}"
                )

            # 深度合并预设配置
            self._deep_merge(yaml_config, yaml_config["presets"][preset_name])

        # 应用命令行覆盖
        if overrides:
            self._deep_merge(yaml_config, overrides)

        # 转换为BacktestConfig对象
        self.config = self._yaml_to_config(yaml_config)

        # 验证配置
        self._validate_config()

        return self.config

    def _deep_merge(self, base_dict: Dict, update_dict: Dict):
        """深度合并字典"""
        for key, value in update_dict.items():
            if (
                key in base_dict
                and isinstance(base_dict[key], dict)
                and isinstance(value, dict)
            ):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value

    def _yaml_to_config(self, yaml_config: Dict) -> BacktestConfig:
        """将YAML配置转换为BacktestConfig对象"""
        # 路径配置
        data_paths = yaml_config.get("data_paths", {})

        # 权重网格配置
        weight_grid = yaml_config.get("weight_grid", {})

        # 回测配置
        backtest_config = yaml_config.get("backtest_config", {})

        # 复合因子配置
        composite_config = yaml_config.get("composite_config", {})

        # 性能配置
        performance_config = yaml_config.get("performance_config", {})

        # 输出配置
        output_config = yaml_config.get("output_config", {})

        # 调试配置
        debug_config = yaml_config.get("debug_config", {})

        # 约束配置
        constraints = yaml_config.get("constraints", {})

        # 指标配置
        metrics = yaml_config.get("metrics", {})

        # 因子配置
        factor_config = yaml_config.get("factor_config", {})

        return BacktestConfig(
            # 数据路径
            panel_file=data_paths.get("panel_file", ""),
            price_dir=data_paths.get("price_dir", ""),
            screening_file=data_paths.get("screening_file", ""),
            output_dir=data_paths.get(
                "output_dir", "etf_rotation_system/03_vbt回测/results"
            ),
            # 因子配置
            top_k=factor_config.get("top_k", 10),
            factors=factor_config.get("factors", []),
            # 回测参数
            top_n_list=backtest_config.get("top_n_list", [3, 5, 8, 10]),
            rebalance_freq=backtest_config.get("rebalance_freq", 20),
            fees=backtest_config.get("fees", 0.001),
            init_cash=backtest_config.get("init_cash", 1000000),
            # 权重网格
            weight_grid_points=weight_grid.get(
                "grid_points", [0.0, 0.25, 0.5, 0.75, 1.0]
            ),
            weight_sum_range=weight_grid.get("weight_sum_range", [0.7, 1.3]),
            max_combinations=weight_grid.get("max_combinations", 10000),
            # 复合因子计算
            standardization_method=composite_config.get(
                "standardization_method", "zscore"
            ),
            enable_score_cache=composite_config.get("enable_score_cache", True),
            # 性能优化
            omp_num_threads=performance_config.get("omp_num_threads", 2),
            veclib_maximum_threads=performance_config.get("veclib_maximum_threads", 2),
            mkl_num_threads=performance_config.get("mkl_num_threads", 2),
            batch_size=performance_config.get("batch_size", 1000),
            enable_progress_bar=performance_config.get("enable_progress_bar", True),
            # 输出配置
            save_top_results=output_config.get("save_top_results", 50),
            save_best_config=output_config.get("save_best_config", True),
            save_detailed_results=output_config.get("save_detailed_results", True),
            results_prefix=output_config.get("results_prefix", "backtest_results"),
            best_config_prefix=output_config.get("best_config_prefix", "best_strategy"),
            # 调试配置
            verbose=debug_config.get("verbose", False),
            log_errors=debug_config.get("log_errors", True),
            save_intermediate=debug_config.get("save_intermediate", False),
            # 日志输出配置
            log_to_file=debug_config.get("log_to_file", False),
            log_dir=debug_config.get("log_dir", "/tmp"),
            log_prefix=debug_config.get("log_prefix", "backtest_log"),
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
        )

    def _validate_config(self):
        """验证配置参数"""
        if not self.config:
            raise ValueError("配置未加载")

        # 验证数据路径
        if not self.config.panel_file:
            raise ValueError("panel_file 不能为空")

        if not self.config.price_dir:
            raise ValueError("price_dir 不能为空")

        if not self.config.screening_file:
            raise ValueError("screening_file 不能为空")

        # 验证数值参数
        if self.config.top_k <= 0:
            raise ValueError("top_k 必须大于0")

        if self.config.fees < 0:
            raise ValueError("fees 不能为负数")

        if self.config.init_cash <= 0:
            raise ValueError("init_cash 必须大于0")

        if self.config.rebalance_freq <= 0:
            raise ValueError("rebalance_freq 必须大于0")

        # 验证权重网格
        if not self.config.weight_grid_points:
            raise ValueError("weight_grid_points 不能为空")

        if not all(0 <= w <= 1 for w in self.config.weight_grid_points):
            raise ValueError("weight_grid_points 中的权重必须在0-1之间")

        if self.config.weight_sum_range[0] >= self.config.weight_sum_range[1]:
            raise ValueError("weight_sum_range 的最小值必须小于最大值")

        if self.config.max_combinations <= 0:
            raise ValueError("max_combinations 必须大于0")

        # 验证方法参数
        if self.config.standardization_method not in ["zscore", "rank"]:
            raise ValueError("standardization_method 必须是 'zscore' 或 'rank'")

        if self.config.primary_metric not in [
            "sharpe_ratio",
            "total_return",
            "max_drawdown",
        ]:
            raise ValueError(
                "primary_metric 必须是 'sharpe_ratio', 'total_return' 或 'max_drawdown'"
            )

        # 验证文件存在性
        panel_path = Path(self.config.panel_file)
        if not panel_path.exists():
            print(f"警告: panel文件不存在: {panel_path}")

        screening_path = Path(self.config.screening_file)
        if not screening_path.exists():
            print(f"警告: screening文件不存在: {screening_path}")

        price_dir = Path(self.config.price_dir)
        if not price_dir.exists():
            print(f"警告: 价格数据目录不存在: {price_dir}")

    def apply_environment_config(self):
        """应用性能环境配置"""
        if not self.config:
            raise ValueError("配置未加载")

        os.environ.setdefault("OMP_NUM_THREADS", str(self.config.omp_num_threads))
        os.environ.setdefault(
            "VECLIB_MAXIMUM_THREADS", str(self.config.veclib_maximum_threads)
        )
        os.environ.setdefault("MKL_NUM_THREADS", str(self.config.mkl_num_threads))

    def list_presets(self) -> List[str]:
        """列出所有可用的预设"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f)

            return list(yaml_config.get("presets", {}).keys())
        except Exception as e:
            print(f"无法读取预设列表: {e}")
            return []

    def save_current_config(self, output_path: str):
        """保存当前配置到文件"""
        if not self.config:
            raise ValueError("配置未加载")

        config_dict = {
            "data_paths": {
                "panel_file": self.config.panel_file,
                "price_dir": self.config.price_dir,
                "screening_file": self.config.screening_file,
                "output_dir": self.config.output_dir,
            },
            "factor_config": {
                "top_k": self.config.top_k,
                "factors": self.config.factors,
            },
            "backtest_config": {
                "top_n_list": self.config.top_n_list,
                "rebalance_freq": self.config.rebalance_freq,
                "fees": self.config.fees,
                "init_cash": self.config.init_cash,
            },
            "weight_grid": {
                "grid_points": self.config.weight_grid_points,
                "weight_sum_range": self.config.weight_sum_range,
                "max_combinations": self.config.max_combinations,
            },
            "composite_config": {
                "standardization_method": self.config.standardization_method,
                "enable_score_cache": self.config.enable_score_cache,
            },
            "performance_config": {
                "omp_num_threads": self.config.omp_num_threads,
                "veclib_maximum_threads": self.config.veclib_maximum_threads,
                "mkl_num_threads": self.config.mkl_num_threads,
                "batch_size": self.config.batch_size,
                "enable_progress_bar": self.config.enable_progress_bar,
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(
                config_dict, f, default_flow_style=False, allow_unicode=True, indent=2
            )


def load_config_from_args(args) -> BacktestConfig:
    """
    从命令行参数加载配置

    Args:
        args: 命令行参数对象

    Returns:
        BacktestConfig对象
    """
    # 使用命令行参数中的配置文件路径
    config_path = getattr(args, "config", None)
    loader = ConfigLoader(config_path)

    # 构建覆盖字典
    overrides = {}

    # 命令行参数覆盖
    if hasattr(args, "preset") and args.preset:
        preset_name = args.preset
    else:
        preset_name = None

    # 直接参数覆盖
    if hasattr(args, "panel") and args.panel:
        overrides.setdefault("data_paths", {})["panel_file"] = args.panel

    if hasattr(args, "price_dir") and args.price_dir:
        overrides.setdefault("data_paths", {})["price_dir"] = args.price_dir

    if hasattr(args, "screening") and args.screening:
        overrides.setdefault("data_paths", {})["screening_file"] = args.screening

    if hasattr(args, "output_dir") and args.output_dir:
        overrides.setdefault("data_paths", {})["output_dir"] = args.output_dir

    if hasattr(args, "max_combos") and args.max_combos:
        overrides.setdefault("weight_grid", {})["max_combinations"] = args.max_combos

    if hasattr(args, "top_k") and args.top_k:
        overrides.setdefault("factor_config", {})["top_k"] = args.top_k

    # 加载配置
    config = loader.load_config(preset_name=preset_name, overrides=overrides)

    # 应用环境配置
    loader.apply_environment_config()

    return config


if __name__ == "__main__":
    # 测试配置加载
    loader = ConfigLoader()

    print("可用预设:")
    for preset in loader.list_presets():
        print(f"  - {preset}")

    print("\n加载默认配置...")
    config = loader.load_config()
    print(f"  权重网格: {config.weight_grid_points}")
    print(f"  最大组合数: {config.max_combinations}")
    print(f"  Top-N列表: {config.top_n_list}")

    print("\n加载快速测试预设...")
    quick_config = loader.load_config(preset_name="quick_test")
    print(f"  权重网格: {quick_config.weight_grid_points}")
    print(f"  最大组合数: {quick_config.max_combinations}")
    print(f"  Top-N列表: {quick_config.top_n_list}")
