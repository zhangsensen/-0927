#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""简化的并行回测引擎 - 恢复原始性能

基于原始parallel_backtest_engine.py，只添加最小化的配置支持
确保性能不受影响
"""

import json
import logging
import multiprocessing as mp
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt
import yaml
from tqdm import tqdm


class SimpleParallelBacktestEngine:
    """简化的并行回测引擎 - 保持原始高性能"""

    def __init__(self, config_file: str = None):
        """
        初始化引擎

        Args:
            config_file: 可选的配置文件路径
        """
        # 默认配置（与原始版本相同）
        self.n_workers = max(1, mp.cpu_count() - 1)
        self.chunk_size = 20
        self.enable_cache = True
        self.log_level = "INFO"

        # 如果提供配置文件，则加载配置
        if config_file:
            self._load_simple_config(config_file)

        # 设置日志
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def _load_simple_config(self, config_file: str):
        """加载简化配置，只覆盖必要的参数"""
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # 只加载必要的配置项
            if "parallel_config" in config:
                parallel_config = config["parallel_config"]
                self.n_workers = parallel_config.get("n_workers", self.n_workers)
                self.chunk_size = parallel_config.get("chunk_size", self.chunk_size)
                self.enable_cache = parallel_config.get(
                    "enable_cache", self.enable_cache
                )
                self.log_level = parallel_config.get("log_level", self.log_level)

            # 加载数据路径
            self.data_paths = config.get("data_paths", {})

        except Exception as e:
            print(f"配置文件加载失败，使用默认配置: {e}")

    def generate_weight_combinations(
        self, factor_names: List[str], max_combinations: int = 10000
    ) -> List[Dict[str, float]]:
        """
        生成权重组合 - 确定性网格搜索（对齐configurable引擎）
        消除随机性，保证可复现
        """
        import itertools

        weight_options = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        n_factors = len(factor_names)

        # 使用确定性网格生成
        combinations = []
        combo_generator = itertools.product(weight_options, repeat=n_factors)

        for combo in combo_generator:
            weight_sum = sum(combo)

            # 权重和约束
            if 0.8 <= weight_sum <= 1.2:
                weights = dict(zip(factor_names, combo))
                combinations.append(weights)

                if len(combinations) >= max_combinations:
                    break

        return combinations

    def _process_weight_chunk(
        self, args: Tuple[List[Dict], pd.DataFrame, List[str], Dict, List[int]]
    ) -> List[Dict]:
        """
        处理权重块（与原始版本相同的向量化逻辑）
        """
        weight_combinations, factor_data, factor_names, backtest_config, top_n_list = (
            args
        )

        results = []

        # 预计算（与原始版本相同）
        factor_matrix = factor_data[factor_names].values
        factor_matrix_3d = factor_matrix.reshape(len(factor_data), len(factor_names), 1)

        for weight_dict in weight_combinations:
            try:
                # 构建权重数组（与原始版本相同）
                weight_array = np.array(
                    [weight_dict.get(factor, 0.0) for factor in factor_names]
                )

                # 向量化计算（与原始版本相同）
                scores_3d = np.einsum("cf,dsf->cds", weight_array, factor_matrix_3d)
                scores = scores_3d.squeeze()

                # 标准化分数
                scores = (scores - scores.mean()) / (scores.std() + 1e-8)

                for top_n in top_n_list:
                    try:
                        # 选择顶部资产
                        selected_indices = np.argpartition(-scores, top_n)[:top_n]
                        selected_indices = selected_indices[
                            np.argsort(-scores[selected_indices])
                        ]

                        # 计算权重
                        weights = np.zeros(len(factor_data.columns[3:]))
                        weights[selected_indices] = 1.0 / top_n

                        # 创建价格矩阵
                        price_matrix = factor_data.iloc[:, 3:].values

                        # 执行回测
                        portfolio = vbt.Portfolio.from_orders(
                            price_matrix,
                            np.tile(weights, (len(factor_data), 1)),
                            init_cash=backtest_config["init_cash"],
                            fees=backtest_config["fees"],
                        )

                        # 计算性能指标
                        total_return = portfolio.total_return() * 100
                        sharpe_ratio = portfolio.sharpe_ratio() * np.sqrt(252)
                        max_drawdown = portfolio.max_drawdown() * 100
                        final_value = portfolio.final_value()
                        turnover = portfolio.trades.count() / len(factor_data) * 100

                        results.append(
                            {
                                "weights": str(weight_dict),
                                "top_n": top_n,
                                "total_return": total_return,
                                "sharpe_ratio": sharpe_ratio,
                                "max_drawdown": max_drawdown,
                                "final_value": final_value,
                                "turnover": turnover,
                            }
                        )

                    except Exception as e:
                        self.logger.warning(
                            f"回测失败: {weight_dict}, top_n={top_n}: {e}"
                        )
                        continue

            except Exception as e:
                self.logger.warning(f"权重处理失败: {weight_dict}: {e}")
                continue

        return results

    def run_backtest(
        self,
        panel_file: str,
        price_dir: str,
        screening_file: str,
        output_dir: str = None,
        max_combinations: int = 10000,
    ) -> pd.DataFrame:
        """
        运行回测（与原始版本相同的逻辑）
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print("=" * 80)
        print("ETF轮动回测引擎 - 简化配置并行计算版本")
        print("=" * 80)
        print(f"时间戳: {timestamp}")
        print(f"工作进程数: {self.n_workers}")
        print(f"块大小: {self.chunk_size}")

        # 加载数据
        print(f"面板: {panel_file}")
        print(f"筛选: {screening_file}")

        panel_data = pd.read_parquet(panel_file)
        screening_data = pd.read_csv(screening_file)

        # 获取因子（与原始版本相同）
        top_factors = screening_data.nlargest(8, "ic_mean")["factor"].tolist()
        print(f"因子数: {len(top_factors)}")
        print(f"因子: {top_factors}")

        # 准备数据（与原始版本相同）
        factor_data = panel_data[
            top_factors + ["open", "high", "low", "close", "volume"]
        ].copy()
        factor_data = factor_data.dropna()

        # 生成权重组合
        print(f"\n开始简化配置并行回测: {max_combinations}个权重组合")
        start_time = time.time()

        weight_combinations = self.generate_weight_combinations(
            top_factors, max_combinations
        )

        # 回测配置 - A股 ETF 成本模型
        backtest_config = {
            "init_cash": 1000000,
            "fees": 0.003,  # A股 ETF: 佣金0.2% + 印花税0.1% = 0.3% 往返
        }
        top_n_list = [3, 5, 8]

        # 并行处理（与原始版本相同）
        chunk_size = self.chunk_size
        chunks = [
            weight_combinations[i : i + chunk_size]
            for i in range(0, len(weight_combinations), chunk_size)
        ]

        args_list = [
            (chunk, factor_data, top_factors, backtest_config, top_n_list)
            for chunk in chunks
        ]

        with mp.Pool(self.n_workers) as pool:
            results = []
            with tqdm(total=len(chunks), desc="并行处理") as pbar:
                for chunk_results in pool.imap_unordered(
                    self._process_weight_chunk, args_list
                ):
                    results.extend(chunk_results)
                    pbar.update(1)

        # 转换为DataFrame
        results_df = pd.DataFrame(results)

        if not results_df.empty:
            # 按夏普比率排序
            results_df = results_df.sort_values("sharpe_ratio", ascending=False)

            # 输出结果
            total_time = time.time() - start_time
            print(f"\n简化配置并行回测完成，总耗时: {total_time:.2f}秒")
            print(
                f"最优策略: {eval(results_df.iloc[0]['weights'])}, top_n={results_df.iloc[0]['top_n']}, sharpe={results_df.iloc[0]['sharpe_ratio']:.3f}"
            )

            # 计算性能指标
            total_strategies = len(results_df)
            speed = total_strategies / total_time
            print(f"\n🎯 简化配置并行优化总结:")
            print(f"处理时间: {total_time:.2f}秒")
            print(f"处理速度: {speed:.1f}策略/秒")
            print(f"工作进程: {self.n_workers}个")
            print(f"最优夏普比率: {results_df.iloc[0]['sharpe_ratio']:.3f}")
            print(f"最优收益率: {results_df.iloc[0]['total_return']:.2f}%")
        else:
            print("没有有效的回测结果")
            total_time = time.time() - start_time
            print(f"简化配置并行回测完成，总耗时: {total_time:.2f}秒")

        # 保存结果（创建时间戳文件夹）
        if output_dir and not results_df.empty:
            output_path = Path(output_dir)
            timestamp_folder = output_path / f"backtest_{timestamp}"
            timestamp_folder.mkdir(parents=True, exist_ok=True)

            # 保存CSV结果
            csv_file = timestamp_folder / "results.csv"
            results_df.to_csv(csv_file, index=False)
            print(f"结果保存至: {csv_file}")

            # 保存最优配置
            best_config = {
                "timestamp": timestamp,
                "engine_type": "simple_parallel",
                "weights": results_df.iloc[0]["weights"],
                "top_n": int(results_df.iloc[0]["top_n"]),
                "performance": {
                    "total_return": float(results_df.iloc[0]["total_return"]),
                    "sharpe_ratio": float(results_df.iloc[0]["sharpe_ratio"]),
                    "max_drawdown": float(results_df.iloc[0]["max_drawdown"]),
                },
                "factors": top_factors,
                "timing": {
                    "total_time": total_time,
                    "strategies_tested": len(results_df),
                    "speed_per_second": speed,
                },
            }

            config_file = timestamp_folder / "best_config.json"
            with open(config_file, "w") as f:
                json.dump(best_config, f, indent=2)
            print(f"最优配置保存至: {config_file}")

            # 保存日志文件
            log_file = timestamp_folder / "backtest.log"
            with open(log_file, "w") as f:
                f.write(f"ETF轮动回测引擎 - 简化配置并行计算版本\n")
                f.write(f"时间戳: {timestamp}\n")
                f.write(f"工作进程数: {self.n_workers}\n")
                f.write(f"块大小: {self.chunk_size}\n")
                f.write(f"最大组合数: {max_combinations}\n")
                f.write(f"面板: {panel_file}\n")
                f.write(f"筛选: {screening_file}\n\n")
                f.write(f"简化配置并行回测完成，总耗时: {total_time:.2f}秒\n")
                f.write(
                    f"最优策略: {eval(results_df.iloc[0]['weights'])}, top_n={int(results_df.iloc[0]['top_n'])}, sharpe={results_df.iloc[0]['sharpe_ratio']:.3f}\n"
                )
            print(f"日志文件保存至: {log_file}")

        return results_df


def main():
    """主函数"""
    import sys

    if len(sys.argv) < 4:
        print(
            "用法: python simple_parallel_backtest_engine.py <panel> <price_dir> <screening_csv> [output_dir] [config_file]"
        )
        print("示例:")
        print("  python simple_parallel_backtest_engine.py \\")
        print("    ../data/panels/panel.parquet \\")
        print("    ../../raw/ETF/daily \\")
        print("    ../data/screening/passed_factors.csv \\")
        print("    ./results \\")
        print("    parallel_backtest_config.yaml  # 可选配置文件")
        return

    panel_file = sys.argv[1]
    price_dir = sys.argv[2]
    screening_file = sys.argv[3]
    output_dir = (
        sys.argv[4]
        if len(sys.argv) > 4
        else "/Users/zhangshenshen/深度量化0927/etf_rotation_system/data/results/backtest"
    )
    config_file = sys.argv[5] if len(sys.argv) > 5 else None

    # 创建引擎
    engine = SimpleParallelBacktestEngine(config_file)

    # 运行回测
    results = engine.run_backtest(
        panel_file=panel_file,
        price_dir=price_dir,
        screening_file=screening_file,
        output_dir=output_dir,
        max_combinations=10000,
    )

    print(f"\n✅ 简化配置并行回测完成！共生成 {len(results)} 个策略结果")


if __name__ == "__main__":
    main()
