#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置加载器 - 纯向量化回测
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class BacktestConfig:
    """回测配置"""

    # 数据路径
    factor_dir: str
    ohlcv_dir: str
    output_dir: str

    # 并行配置
    n_workers: int = 8
    chunk_size: int = 100
    log_level: str = "INFO"

    # 因子配置
    factors: List[str] = field(default_factory=list)
    combination_sizes: List[int] = field(default_factory=lambda: [1, 2, 3])
    max_combinations: int = 10000
    weight_grid: List[List[float]] = field(default_factory=list)

    # 回测配置
    top_n_list: List[int] = field(default_factory=lambda: [5, 10, 15])
    rebalance_freq_list: List[int] = field(default_factory=lambda: [5, 10, 20])
    rebalance_mode: str = "time"  # "time" | "event"
    fees: float = 0.0005
    init_cash: float = 1000000
    allow_zero_position: bool = False

    # 信号配置
    signal_method: str = "composite_score"
    standardization: str = "zscore"
    invert_factors: List[str] = field(default_factory=list)
    # 因子筛选阈值
    min_abs_ic: float = 0.01
    min_ic_ir: float = 0.02
    min_positive_rate: float = 0.55
    min_significant_rate: float = 0.1
    min_observations: int = 80
    fdr_q: float = 0.1
    max_factor_correlation: float = 0.8
    fallback_top_k: int = 20
    ic_significant_threshold: float = 0.02

    # 性能指标
    performance_metrics: List[str] = field(
        default_factory=lambda: ["sharpe_ratio", "annual_return", "max_drawdown"]
    )

    # 输出配置
    save_top_n: int = 100
    save_equity_curves: bool = True
    save_trades: bool = False
    output_format: str = "csv"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "BacktestConfig":
        """从YAML文件加载配置"""
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        factor_cfg = config_dict["factor_config"]
        selection_cfg = factor_cfg.get("selection_thresholds", {})
        backtest_cfg = config_dict["backtest_config"]

        # 读取调仓模式并进行兼容性映射：event → 按日评估（T+1落地），无需额外代码改动
        rebalance_mode = backtest_cfg.get("rebalance_mode", "time")
        rebalance_freq_list = backtest_cfg["rebalance_freq_list"]
        if rebalance_mode == "event":
            # 事件驱动=信号日评估。将频率列表覆盖为[1]，表示每日评估并仅在权重变化时产生换手
            rebalance_freq_list = [1]

        return cls(
            # 数据路径
            factor_dir=config_dict["data_paths"]["factor_dir"],
            ohlcv_dir=config_dict["data_paths"]["ohlcv_dir"],
            output_dir=config_dict["data_paths"]["output_dir"],
            # 并行配置
            n_workers=config_dict["parallel_config"]["n_workers"],
            chunk_size=config_dict["parallel_config"]["chunk_size"],
            log_level=config_dict["parallel_config"]["log_level"],
            # 因子配置
            factors=factor_cfg["factors"],
            combination_sizes=factor_cfg["combination_sizes"],
            max_combinations=factor_cfg["max_combinations"],
            weight_grid=factor_cfg["weight_grid"],
            min_abs_ic=selection_cfg.get("min_abs_ic", 0.01),
            min_ic_ir=selection_cfg.get("min_ic_ir", 0.02),
            min_positive_rate=selection_cfg.get("min_positive_rate", 0.55),
            min_significant_rate=selection_cfg.get("min_significant_rate", 0.1),
            min_observations=selection_cfg.get("min_observations", 80),
            fdr_q=selection_cfg.get("fdr_q", 0.1),
            max_factor_correlation=factor_cfg.get("max_correlation", 0.8),
            fallback_top_k=selection_cfg.get("fallback_top_k", 20),
            ic_significant_threshold=selection_cfg.get("significance_threshold", 0.02),
            # 回测配置
            top_n_list=backtest_cfg["top_n_list"],
            rebalance_freq_list=rebalance_freq_list,
            rebalance_mode=rebalance_mode,
            fees=backtest_cfg["fees"],
            init_cash=backtest_cfg["init_cash"],
            allow_zero_position=backtest_cfg["allow_zero_position"],
            # 信号配置
            signal_method=config_dict["signal_config"]["method"],
            standardization=config_dict["signal_config"]["standardization"],
            invert_factors=config_dict["signal_config"]["invert_factors"],
            # 性能指标
            performance_metrics=config_dict["performance_metrics"],
            # 输出配置
            save_top_n=config_dict["output_config"]["save_top_n"],
            save_equity_curves=config_dict["output_config"]["save_equity_curves"],
            save_trades=config_dict["output_config"]["save_trades"],
            output_format=config_dict["output_config"]["format"],
        )

    def get_absolute_paths(self, base_dir: Path) -> "BacktestConfig":
        """转换相对路径为绝对路径"""
        self.factor_dir = str((base_dir / self.factor_dir).resolve())
        self.ohlcv_dir = str((base_dir / self.ohlcv_dir).resolve())
        self.output_dir = str((base_dir / self.output_dir).resolve())
        return self

    def validate(self):
        """验证配置合理性"""
        errors = []

        # 检查路径
        if not Path(self.factor_dir).exists():
            errors.append(f"因子目录不存在: {self.factor_dir}")
        if not Path(self.ohlcv_dir).exists():
            errors.append(f"OHLCV目录不存在: {self.ohlcv_dir}")

        # 检查数值范围
        if self.fees < 0 or self.fees > 0.1:
            errors.append(f"费用率异常: {self.fees}")
        if self.init_cash <= 0:
            errors.append(f"初始资金必须>0: {self.init_cash}")
        if any(n <= 0 for n in self.top_n_list):
            errors.append(f"Top-N必须>0: {self.top_n_list}")
        if any(f <= 0 for f in self.rebalance_freq_list):
            errors.append(f"调仓频率必须>0: {self.rebalance_freq_list}")

        # 检查组合大小
        if max(self.combination_sizes) > 10:
            errors.append(f"因子组合过大（建议≤5）: {max(self.combination_sizes)}")

        if self.fdr_q < 0 or self.fdr_q > 1:
            errors.append(f"FDR阈值需在[0,1]: {self.fdr_q}")
        if self.min_observations <= 0:
            errors.append(f"最小样本数需>0: {self.min_observations}")
        if self.max_factor_correlation <= 0 or self.max_factor_correlation > 1:
            errors.append(f"最大因子相关性需在(0,1]: {self.max_factor_correlation}")
        if self.fallback_top_k <= 0:
            errors.append(f"回退因子数量需>0: {self.fallback_top_k}")

        if errors:
            raise ValueError("配置验证失败:\\n" + "\\n".join(errors))

        return True
