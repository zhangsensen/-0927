#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WFO批量实验工具
==============

设计理念:
- 因子只算1次，WFO参数扫描N次
- 自动保存所有实验结果
- 支持参数网格搜索

使用示例:
  python scripts/wfo_experiments.py --config configs/wfo_grid.yaml
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.constrained_walk_forward_optimizer import ConstrainedWalkForwardOptimizer
from core.cross_section_processor import CrossSectionProcessor
from core.factor_selector import create_default_selector
from core.precise_factor_library_v2 import PreciseFactorLibrary
from utils.factor_cache import FactorCache


class WFOExperiments:
    """WFO批量实验"""

    def __init__(self, ohlcv: Dict[str, pd.DataFrame], output_dir: Path):
        """
        初始化实验

        Args:
            ohlcv: OHLCV数据
            output_dir: 输出目录
        """
        self.ohlcv = ohlcv
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 缓存管理器
        self.cache = FactorCache()

        # 因子数据（延迟加载）
        self.factors_dict = None
        self.standardized_factors = None

        # 实验结果
        self.experiment_results = []

    def _prepare_factors(self):
        """准备因子数据（使用缓存）"""
        print("\n" + "=" * 80)
        print("准备因子数据")
        print("=" * 80)

        # 尝试加载原始因子缓存
        lib = PreciseFactorLibrary()
        self.factors_dict = self.cache.load_factors(
            self.ohlcv, PreciseFactorLibrary, stage="raw"
        )

        if self.factors_dict is None:
            print("⏱️  计算原始因子...")
            factors_df = lib.compute_all_factors(self.ohlcv)

            # 转换为字典格式
            self.factors_dict = {}
            for factor_name in factors_df.columns.get_level_values(0).unique():
                self.factors_dict[factor_name] = factors_df[factor_name]

            # 保存缓存
            self.cache.save_factors(
                self.factors_dict, self.ohlcv, PreciseFactorLibrary, stage="raw"
            )
            print(f"✅ 因子计算完成: {len(self.factors_dict)}个")
        else:
            print(f"✅ 使用缓存因子: {len(self.factors_dict)}个")

        # 尝试加载标准化因子缓存
        self.standardized_factors = self.cache.load_factors(
            self.ohlcv, PreciseFactorLibrary, stage="standardized"
        )

        if self.standardized_factors is None:
            print("⏱️  标准化因子...")
            processor = CrossSectionProcessor(verbose=False)
            self.standardized_factors = processor.process_all_factors(self.factors_dict)

            # 保存缓存
            self.cache.save_factors(
                self.standardized_factors,
                self.ohlcv,
                PreciseFactorLibrary,
                stage="standardized",
            )
            print(f"✅ 标准化完成: {len(self.standardized_factors)}个")
        else:
            print(f"✅ 使用缓存标准化因子: {len(self.standardized_factors)}个")

    def run_single_experiment(
        self,
        exp_name: str,
        is_period: int = 252,
        oos_period: int = 60,
        step_size: int = 20,
        target_factor_count: int = 5,
    ) -> dict:
        """
        运行单个WFO实验

        Args:
            exp_name: 实验名称
            is_period: 样本内天数
            oos_period: 样本外天数
            step_size: 滚动步长
            target_factor_count: 目标因子数

        Returns:
            实验结果字典
        """
        print(f"\n{'='*80}")
        print(f"实验: {exp_name}")
        print(
            f"参数: IS={is_period}, OOS={oos_period}, step={step_size}, factors={target_factor_count}"
        )
        print(f"{'='*80}")

        # 准备数据
        factor_names = list(self.standardized_factors.keys())
        dates = self.ohlcv["close"].index
        symbols = self.ohlcv["close"].columns

        # 转换为3D数组
        T, N, F = len(dates), len(symbols), len(factor_names)
        factors_3d = np.full((T, N, F), np.nan)

        for f_idx, fname in enumerate(factor_names):
            factors_3d[:, :, f_idx] = self.standardized_factors[fname].values

        # 计算收益率
        returns_df = self.ohlcv["close"].pct_change()

        # 运行WFO
        selector = create_default_selector()
        optimizer = ConstrainedWalkForwardOptimizer(selector=selector, verbose=False)

        wfo_df, _ = optimizer.run_constrained_wfo(
            factors_data=factors_3d,
            returns=returns_df.values,
            factor_names=factor_names,
            is_period=is_period,
            oos_period=oos_period,
            step_size=step_size,
            target_factor_count=target_factor_count,
        )

        # 统计结果
        avg_oos_ic = wfo_df["oos_ic_mean"].mean()
        avg_ic_drop = wfo_df["ic_drop"].mean()
        num_windows = len(wfo_df)

        # 因子选中频率
        all_selected = ",".join(wfo_df["selected_factors"].tolist()).split(",")
        factor_freq = pd.Series(all_selected).value_counts()
        top_factor = factor_freq.index[0] if len(factor_freq) > 0 else "N/A"

        print(
            f"✅ 完成: {num_windows}窗口, OOS IC={avg_oos_ic:.4f}, 衰减={avg_ic_drop:.4f}"
        )
        print(f"   TOP因子: {top_factor} ({factor_freq.iloc[0]}/{num_windows})")

        # 保存结果
        result_file = self.output_dir / f"{exp_name}_wfo_results.csv"
        wfo_df.to_csv(result_file, index=False)

        return {
            "experiment": exp_name,
            "is_period": is_period,
            "oos_period": oos_period,
            "step_size": step_size,
            "target_factor_count": target_factor_count,
            "num_windows": num_windows,
            "avg_oos_ic": avg_oos_ic,
            "avg_ic_drop": avg_ic_drop,
            "top_factor": top_factor,
            "top_factor_freq": (
                float(factor_freq.iloc[0]) / num_windows if len(factor_freq) > 0 else 0
            ),
            "result_file": str(result_file.name),
        }

    def run_grid_search(self, param_grid: dict):
        """
        运行参数网格搜索

        Args:
            param_grid: 参数网格
                {
                    'is_period': [126, 252, 504],
                    'oos_period': [30, 60, 120],
                    'step_size': [10, 20, 40],
                    'target_factor_count': [3, 5, 8]
                }
        """
        # 准备因子（只算1次）
        self._prepare_factors()

        # 生成所有参数组合
        from itertools import product

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        total_experiments = 1
        for values in param_values:
            total_experiments *= len(values)

        print(f"\n🚀 开始网格搜索: {total_experiments}个实验")

        exp_idx = 0
        for params in product(*param_values):
            exp_idx += 1
            param_dict = dict(zip(param_names, params))
            exp_name = f"exp_{exp_idx:03d}"

            result = self.run_single_experiment(exp_name=exp_name, **param_dict)

            self.experiment_results.append(result)

        # 保存汇总结果
        self._save_summary()

    def _save_summary(self):
        """保存实验汇总"""
        summary_df = pd.DataFrame(self.experiment_results)

        # 排序（按OOS IC降序）
        summary_df = summary_df.sort_values("avg_oos_ic", ascending=False)

        # 保存
        summary_file = self.output_dir / "experiments_summary.csv"
        summary_df.to_csv(summary_file, index=False)

        # 保存JSON
        summary_json = self.output_dir / "experiments_summary.json"
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(self.experiment_results, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*80}")
        print(f"✅ 实验完成，汇总结果已保存")
        print(f"{'='*80}")
        print(f"   总实验数: {len(self.experiment_results)}")
        print(f"   汇总文件: {summary_file.name}")

        # 显示TOP 5
        print(f"\n   TOP 5 参数组合:")
        for idx, row in summary_df.head(5).iterrows():
            print(
                f"     #{row.name+1:2d} | OOS IC={row['avg_oos_ic']:.4f} | "
                f"IS={row['is_period']}, OOS={row['oos_period']}, "
                f"step={row['step_size']}, factors={row['target_factor_count']}"
            )


def load_grid_config(config_file: Path, grid_name: str = "basic_grid") -> dict:
    """
    从YAML加载参数网格配置

    Args:
        config_file: 配置文件路径
        grid_name: 网格名称

    Returns:
        参数网格字典
    """
    import yaml

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if grid_name not in config:
        raise ValueError(
            f"找不到网格配置: {grid_name}，可用选项: {list(config.keys())}"
        )

    return config[grid_name]


def main():
    """主函数"""
    import argparse

    from scripts.production_backtest import ProductionBacktest

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="WFO批量实验工具")
    parser.add_argument(
        "--grid",
        type=str,
        default="basic_grid",
        help="参数网格名称 (basic_grid, full_grid, conservative, etc.)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "wfo_grid.yaml",
        help="配置文件路径",
    )

    args = parser.parse_args()

    # 加载参数网格
    print(f"加载参数网格: {args.grid}")
    param_grid = load_grid_config(args.config, args.grid)

    total_exps = 1
    for values in param_grid.values():
        total_exps *= len(values)
    print(f"总实验数: {total_exps}")

    # 加载数据
    print("\n加载ETF数据...")
    project_root = Path(__file__).parent.parent
    backtest = ProductionBacktest(output_base_dir=project_root / "results")
    backtest.load_data()

    # 创建实验
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "results" / f"wfo_experiments_{timestamp}"

    experiments = WFOExperiments(ohlcv=backtest.ohlcv, output_dir=output_dir)

    # 运行网格搜索
    experiments.run_grid_search(param_grid)


if __name__ == "__main__":
    main()
