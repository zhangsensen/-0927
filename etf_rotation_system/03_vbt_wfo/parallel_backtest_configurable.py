#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""配置化并行回测引擎 - 向量化版本

结合高性能向量化计算与完整配置抽象的ETF轮动回测引擎
"""

import argparse
import itertools
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import vectorbt as vbt

    HAS_VBT = True
except ImportError:
    HAS_VBT = False
    print("警告: 未安装 vectorbt，请运行: pip install vectorbt")

from config_loader_parallel import FastConfig, load_fast_config_from_args


class ConfigurableParallelBacktestEngine:
    """配置化并行回测引擎 - 完全向量化实现"""

    def __init__(self, config: FastConfig):
        """
        初始化配置化并行回测引擎

        Args:
            config: 回测配置对象
        """
        self.config = config

        # 设置环境变量（基于配置）
        os.environ.setdefault("OMP_NUM_THREADS", str(config.omp_num_threads))
        os.environ.setdefault(
            "VECLIB_MAXIMUM_THREADS", str(config.veclib_maximum_threads)
        )
        os.environ.setdefault("MKL_NUM_THREADS", str(config.mkl_num_threads))

        # 设置日志
        logging.basicConfig(
            level=getattr(logging, config.log_level.upper()),
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def _validate_config(self):
        """启动时验证配置合理性"""
        # ...existing code...

    def _validate_config(self):
        """启动时验证配置合理性"""
        # 检查权重网格与因子数的适配性
        n_grid = len(self.config.weight_grid_points)

        # 检查文件路径
        if not Path(self.config.panel_file).exists():
            self.logger.warning(f"⚠️  面板文件不存在: {self.config.panel_file}")

        if not Path(self.config.price_dir).exists():
            self.logger.warning(f"⚠️  价格目录不存在: {self.config.price_dir}")

        # 检查筛选文件（如果提供）
        if self.config.screening_file and not Path(self.config.screening_file).exists():
            self.logger.warning(f"⚠️  筛选文件不存在: {self.config.screening_file}")

        # 检查并行配置
        import multiprocessing

        max_cores = multiprocessing.cpu_count()
        if self.config.n_workers > max_cores:
            self.logger.warning(
                f"⚠️  配置的工作进程数({self.config.n_workers})超过CPU核心数({max_cores})"
            )

    def _load_factor_panel(self) -> pd.DataFrame:
        """加载因子面板"""
        self.logger.info(f"加载因子面板: {self.config.panel_file}")
        panel = pd.read_parquet(self.config.panel_file)
        if not isinstance(panel.index, pd.MultiIndex):
            raise ValueError("面板必须是 (symbol, date) MultiIndex")

        # 统计面板信息
        n_symbols = panel.index.get_level_values(0).nunique()
        dates = panel.index.get_level_values(1).unique()
        date_range = (
            f"{dates.min().strftime('%Y-%m-%d')}~{dates.max().strftime('%Y-%m-%d')}"
        )
        self.logger.info(
            f"面板形状: {panel.shape}, 日期: {date_range} ({len(dates)}日), 标的: {n_symbols}个"
        )

        return panel

    def _load_price_data(self) -> pd.DataFrame:
        """加载价格数据 - 带缓存优化"""
        import glob

        # 缓存路径
        cache_dir = Path(self.config.output_dir) / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / "prices.parquet"

        # 检查源文件最新修改时间
        source_files = sorted(glob.glob(f"{self.config.price_dir}/*.parquet"))
        if not source_files:
            raise ValueError(f"未找到价格数据文件: {self.config.price_dir}")

        latest_mtime = max(Path(f).stat().st_mtime for f in source_files)

        # 检查缓存是否有效
        cache_hit = False
        if cache_file.exists():
            cache_mtime = cache_file.stat().st_mtime
            if cache_mtime >= latest_mtime:
                self.logger.info(f"命中价格缓存: {cache_file}")
                pivot = pd.read_parquet(cache_file)
                cache_hit = True

        if not cache_hit:
            # 重新加载并缓存
            self.logger.info(f"刷新价格缓存，加载 {len(source_files)} 个文件...")
            prices = []
            for f in source_files:
                df = pd.read_parquet(f)
                symbol = f.split("/")[-1].split("_")[0]
                df["symbol"] = symbol
                df["date"] = pd.to_datetime(df["trade_date"])
                prices.append(df[["date", "close", "symbol"]])

            price_df = pd.concat(prices, ignore_index=True)
            pivot = price_df.pivot(index="date", columns="symbol", values="close")
            pivot = pivot.sort_index()

            # � 不填充：保留 NaN 用于标记停牌/退市日期
            # 原因：
            #   1. ETF 停牌日期就是 NaN，反映现实
            #   2. 权重计算时 NaN×权重=NaN，自动排除停牌日期
            #   3. 无需人工填充，规避虚假价格风险

            # 写入缓存
            pivot.to_parquet(cache_file, compression="snappy")
            self.logger.info(f"价格缓存已更新: {cache_file}")

        # 统计价格数据质量（增强版）
        missing_pct = pivot.isna().sum().sum() / (pivot.shape[0] * pivot.shape[1])
        cache_status = "缓存命中" if cache_hit else "重新加载"

        # 🔍 新增：详细缺失值统计
        max_consecutive_gaps = {}
        for col in pivot.columns:
            mask = pivot[col].isna()
            if mask.any():
                consecutive = (mask != mask.shift()).cumsum()
                gap_lens = consecutive[mask].value_counts()
                max_consecutive_gaps[col] = (
                    gap_lens.index.max() if len(gap_lens) > 0 else 0
                )

        if max_consecutive_gaps:
            worst_symbol = max(max_consecutive_gaps, key=max_consecutive_gaps.get)
            worst_gap = max_consecutive_gaps[worst_symbol]
            self.logger.warning(
                f"⚠️  最长连续缺失: {worst_symbol} = {worst_gap}天 "
                f"(limit=3, 超出的将保持NaN)"
            )

        self.logger.info(
            f"价格矩阵: {pivot.shape}, 总缺失率: {missing_pct:.2%}, 状态: {cache_status}"
        )

        return pivot

    def _load_top_factors(self) -> List[str]:
        """从筛选结果加载Top K因子 - 添加IC/IR过滤"""
        df = pd.read_csv(self.config.screening_file)
        col_name = "factor" if "factor" in df.columns else "panel_factor"

        # 添加IC/IR质量过滤（如果列存在）
        if "ic_mean" in df.columns and "ic_ir" in df.columns:
            original_count = len(df)
            # 过滤低质量因子
            df = df[
                (df["ic_mean"].abs() >= 0.01)  # IC绝对值至少0.01
                & (df["ic_ir"].abs() >= 0.05)  # IR绝对值至少0.05
            ]
            filtered_count = len(df)
            if filtered_count < original_count:
                self.logger.info(
                    f"IC/IR过滤: {original_count}个 → {filtered_count}个因子 "
                    f"(移除{original_count - filtered_count}个低质因子)"
                )

        # 如果配置中指定了因子列表，使用配置的因子
        if self.config.factors:
            factors = self.config.factors
            # 验证因子是否存在于筛选结果中
            available_factors = df[col_name].tolist()
            missing_factors = [f for f in factors if f not in available_factors]
            if missing_factors:
                self.logger.warning(f"以下因子不在筛选结果中: {missing_factors}")
                factors = [f for f in factors if f in available_factors]
            self.logger.info(f"使用配置的 {len(factors)} 个因子: {factors}")
        else:
            factors = df.head(self.config.top_k)[col_name].tolist()
            self.logger.info(f"加载Top {len(factors)}因子: {factors}")

            # 打印因子质量信息（如果有）
            if "ic_mean" in df.columns and "ic_ir" in df.columns:
                for factor in factors[:5]:  # 只显示前5个
                    row = df[df[col_name] == factor].iloc[0]
                    self.logger.info(
                        f"  {factor}: IC={row['ic_mean']:.3f}, IR={row['ic_ir']:.3f}"
                    )

        return factors

    def _calculate_composite_score(
        self,
        panel: pd.DataFrame,
        factors: List[str],
        weights: Dict[str, float],
    ) -> pd.DataFrame:
        """计算复合因子得分 - 完全向量化实现"""
        # 重塑为 (date, symbol) 结构
        factor_data = panel[factors].unstack(level="symbol")

        # 向量化标准化
        if self.config.standardization_method == "zscore":
            normalized = (
                factor_data
                - factor_data.mean(axis=1, skipna=True).values[:, np.newaxis]
            ) / (
                factor_data.std(axis=1, skipna=True).values[:, np.newaxis]
                + self.config.numerical_epsilon
            )
        else:  # rank
            normalized = factor_data.rank(axis=1, pct=True) * 2 - 1

        # 获取维度信息
        n_dates, n_total = normalized.shape
        n_factors = len(factors)
        n_symbols = n_total // n_factors

        # 修复：unstack后列序是 (factor1,sym1), (factor1,sym2), ..., (factor2,sym1), ...
        # 需要转置为 (sym1,factor1), (sym1,factor2), ..., (sym2,factor1), ... 才能正确reshape
        # 方法：reshape为 (n_dates, n_factors, n_symbols) 然后转置为 (n_dates, n_symbols, n_factors)
        reshaped = normalized.values.reshape(n_dates, n_factors, n_symbols).transpose(
            0, 2, 1
        )

        # 向量化加权求和
        weight_array = np.array([weights.get(f, 0) for f in factors])
        scores_array = np.sum(
            reshaped * weight_array[np.newaxis, np.newaxis, :], axis=2
        )

        # 创建结果DataFrame
        symbols = [col[1] for col in normalized.columns[::n_factors]]
        scores = pd.DataFrame(scores_array, index=normalized.index, columns=symbols)

        # 🔧 修正未来函数：信号延迟1天（使用T-1日因子决策T日持仓）
        scores = scores.shift(1)

        return scores

    def _build_target_weights(self, scores: pd.DataFrame, top_n: int) -> pd.DataFrame:
        """构建Top-N目标权重"""
        ranks = scores.rank(axis=1, ascending=False, method="first")
        selection = ranks <= top_n
        weights = selection.astype(float)
        weights = weights.div(weights.sum(axis=1), axis=0).fillna(0.0)
        return weights

    def _process_weight_chunk(
        self,
        weight_chunk: List[Tuple[float, ...]],
        factors: List[str],
        panel: pd.DataFrame,
        prices: pd.DataFrame,
        top_n_list: List[int],
        rebalance_freq: int,
    ) -> List[Dict[str, Any]]:
        """处理一个权重组合块 - 完全向量化消除所有循环"""
        results = []

        try:
            # === 完全向量化步骤1: 批量计算所有权重组合的得分矩阵 ===
            # 重塑因子数据为3D矩阵: (dates, symbols, factors)
            factor_data = panel[factors].unstack(level="symbol")
            n_dates, n_total = factor_data.shape
            n_factors = len(factors)
            n_symbols = n_total // n_factors

            # 标准化因子数据 (一次性计算)
            factor_mean = factor_data.mean(axis=1, skipna=True).values[:, np.newaxis]
            factor_std = (
                factor_data.std(axis=1, skipna=True).values[:, np.newaxis] + 1e-8
            )
            normalized = (factor_data - factor_mean) / factor_std

            # 重塑为3D矩阵用于批量矩阵乘法
            factor_matrix_3d = normalized.values.reshape(n_dates, n_symbols, n_factors)

            # === 完全向量化步骤2: 批量计算所有权重组合的得分 ===
            weight_array = np.array(weight_chunk)  # (n_combinations, n_factors)
            n_combinations = len(weight_chunk)

            # 使用矩阵乘法批量计算所有得分: (n_combinations, n_dates, n_symbols)
            scores_3d = np.einsum("cf,dsf->cds", weight_array, factor_matrix_3d)

            # === 完全向量化步骤3: 批量处理所有Top-N回测 ===
            symbol_list = [col[1] for col in normalized.columns[::n_factors]]
            date_index = normalized.index

            # 🔧 修复：禁用信号质量阈值，只使用TopN排名筛选
            # 原阈值0.5过高，导致TopN参数失效（所有TopN结果相同）
            # Z-score标准化后，0.5表示超过0.5个标准差，过于严格
            min_score_threshold = -999.0  # 实际禁用

            # 为每个Top-N值批量处理所有权重组合
            for top_n in top_n_list:
                try:
                    # 向量化构建所有权重组合的目标权重矩阵
                    # 🔧 修复：仅使用相对排名，移除绝对阈值筛选
                    ranks_3d = (
                        np.argsort(np.argsort(-scores_3d, axis=2), axis=2) + 1
                    )  # 排名从1开始

                    # 仅使用TopN排名筛选
                    selection_3d = ranks_3d <= top_n
                    weights_3d = selection_3d.astype(float)

                    # 归一化权重 (每行和为1)
                    # 如果某天所有信号都不满足阈值，权重为全0（空仓）
                    weight_sums = weights_3d.sum(axis=2, keepdims=True)
                    weight_sums[weight_sums == 0] = 1  # 避免除零（空仓日自动变为全0）
                    weights_3d = weights_3d / weight_sums

                    # 批量回测所有权重组合
                    chunk_results = self._vectorized_batch_backtest(
                        scores_3d,
                        weights_3d,
                        prices,
                        symbol_list,
                        date_index,
                        top_n,
                        rebalance_freq,
                        weight_chunk,
                        factors,
                    )
                    results.extend(chunk_results)

                except Exception as e:
                    if self.config.log_errors:
                        self.logger.error(f"处理Top-N={top_n}时出错: {e}")
                        if self.config.verbose:
                            import traceback

                            self.logger.debug(traceback.format_exc())
                    continue  # 跳过失败的Top-N值

        except Exception as e:
            if self.config.log_errors:
                self.logger.error(f"处理权重块时出错: {e}")
                if self.config.verbose:
                    import traceback

                    self.logger.debug(traceback.format_exc())
            pass  # 跳过失败的块

        return results

    def _vectorized_batch_backtest(
        self,
        scores_3d: np.ndarray,
        weights_3d: np.ndarray,
        prices: pd.DataFrame,
        symbol_list: List[str],
        date_index: pd.DatetimeIndex,
        top_n: int,
        rebalance_freq: int,
        weight_chunk: List[Tuple[float, ...]],
        factors: List[str],
    ) -> List[Dict[str, Any]]:
        """完全向量化批量回测 - 消除所有循环"""
        n_combinations, n_dates, n_symbols = scores_3d.shape

        # 对齐价格数据
        common_dates = prices.index.intersection(date_index)
        price_aligned = prices.loc[common_dates, symbol_list]

        # 找到时间索引对齐
        date_mask = np.isin(date_index, common_dates)
        scores_aligned = scores_3d[:, date_mask, :]
        weights_aligned = weights_3d[:, date_mask, :]

        # 计算收益率矩阵 (只计算一次，复用所有组合)
        # 修复pct_change的FutureWarning
        returns = (
            price_aligned.pct_change(fill_method=None).fillna(0.0).values
        )  # (n_dates_aligned, n_symbols)
        n_aligned_dates = len(common_dates)

        # 向量化调仓处理
        rebalance_indices = np.arange(n_aligned_dates)[
            np.arange(n_aligned_dates) % rebalance_freq == 0
        ]
        if len(rebalance_indices) == 0:
            rebalance_indices = [0]

        # 创建调仓权重矩阵
        final_weights = np.zeros_like(weights_aligned)
        final_weights[:, rebalance_indices, :] = weights_aligned[
            :, rebalance_indices, :
        ]

        # 向前填充权重 (完全向量化)
        if len(rebalance_indices) > 0:
            # 使用np.maximum.accumulate进行向量化填充
            # 创建调仓矩阵，调仓日为1，其他为0
            rebalance_matrix = np.zeros((n_aligned_dates,), dtype=int)
            rebalance_matrix[rebalance_indices] = 1

            # 向前传播最近的调仓索引
            cumsum_rebalance = np.maximum.accumulate(
                np.arange(n_aligned_dates) * rebalance_matrix
            )

            # 将0值替换为最近的调仓索引
            valid_mask = cumsum_rebalance > 0
            last_valid = np.maximum.accumulate(
                np.where(valid_mask, cumsum_rebalance, 0)
            )
            cumsum_rebalance = np.where(valid_mask, cumsum_rebalance, last_valid)

            # 向量化填充权重
            final_weights = final_weights[:, cumsum_rebalance, :]

        # 计算投资组合收益 (完全向量化)
        # 🔧 修复双重延迟BUG:
        # scores已经在第276行shift(1)延迟，weights[T]已经对应T日持仓
        # 不应该再使用prev_weights，直接用final_weights计算收益
        # 原错误逻辑: prev_weights[:, 1:, :] = final_weights[:, :-1, :] 导致额外延迟1天
        # 正确逻辑: T日权重 × T日收益率 = T日组合收益
        portfolio_returns = np.sum(
            final_weights * returns[np.newaxis, :, :], axis=2
        )  # (n_combinations, n_dates)

        # 交易成本计算 (完全向量化)
        # 🔧 修复：权重变化应该与portfolio_returns的维度对齐
        # 由于portfolio_returns现在使用final_weights（无额外延迟），需要正确计算换手
        weight_changes = np.abs(final_weights[:, 1:, :] - final_weights[:, :-1, :]).sum(
            axis=2
        )
        turnover = 0.5 * weight_changes  # (n_combinations, n_dates-1)
        trading_costs = self.config.fees * turnover  # 从配置读取费用率

        # 净收益：第一天无交易成本，后续天数扣除成本
        net_returns = portfolio_returns.copy()
        net_returns[:, 1:] = portfolio_returns[:, 1:] - trading_costs

        # 计算统计指标 (完全向量化)
        init_cash = self.config.init_cash  # 从配置读取初始资金
        equity_matrix = (1 + net_returns).cumprod(axis=1) * init_cash

        # 最终结果统计
        final_values = equity_matrix[:, -1]
        total_returns = (final_values / init_cash - 1) * 100

        # 夏普比率计算 (使用 nanmean/nanstd 忽略停牌日期的 NaN)
        mean_returns = np.nanmean(net_returns, axis=1)
        std_returns = np.nanstd(net_returns, axis=1)
        sharpe_ratios = np.where(
            std_returns > 0,
            mean_returns / std_returns * np.sqrt(self.config.periods_per_year),
            0,
        )

        # 最大回撤计算
        running_max = np.maximum.accumulate(equity_matrix, axis=1)
        drawdowns = (equity_matrix / running_max - 1) * 100
        max_drawdowns = drawdowns.min(axis=1)

        # 换手率
        total_turnover = turnover.sum(axis=1)

        # 构建结果列表 (完全向量化)
        weight_dicts = [dict(zip(factors, chunk)) for chunk in weight_chunk]

        # 使用列表推导式批量构建结果，包含rebalance_freq
        results = [
            {
                "weights": str(weight_dicts[i]),
                "top_n": top_n,
                "rebalance_freq": rebalance_freq,
                "total_return": float(total_returns[i]),
                "sharpe_ratio": float(sharpe_ratios[i]),
                "max_drawdown": float(max_drawdowns[i]),
                "final_value": float(final_values[i]),
                "turnover": float(total_turnover[i]),
            }
            for i in range(n_combinations)
        ]

        return results

    def _generate_weight_combinations(self) -> List[Tuple[float, ...]]:
        """生成有效的权重组合 - 流式化避免指数级内存占用"""
        valid_combos = []
        weight_sum_min, weight_sum_max = self.config.weight_sum_range
        max_combos = self.config.max_combinations

        # 计算理论组合数
        n_grid_points = len(self.config.weight_grid_points)
        n_factors = len(self.config.factors)
        theoretical_combos = n_grid_points**n_factors

        self.logger.info(
            f"权重网格: {n_grid_points}点 × {n_factors}因子, 理论组合: {theoretical_combos:,}"
        )

        # 自适应权重约束警告
        avg_weight_if_equal = 1.0 / n_factors
        if n_factors > 10 and weight_sum_max > 1.5:
            self.logger.warning(
                f"⚠️  因子数({n_factors})较多，但权重和上限({weight_sum_max})较宽松"
            )
            self.logger.warning(
                f"    平均权重={avg_weight_if_equal:.3f}，建议调整weight_sum_range=[0.9, 1.1]"
            )

        if theoretical_combos > 1e9:
            # 组合数 > 10亿：使用Dirichlet智能采样
            self.logger.warning(
                f"⚠️  理论组合数 {theoretical_combos:.2e}，采用Dirichlet智能采样"
            )
            np.random.seed(42)
            seen = set()  # 使用set加速去重

            # 先使用Dirichlet生成符合权重和约束的组合
            target_sum = (weight_sum_min + weight_sum_max) / 2  # 目标权重和（中点）
            alpha = np.ones(n_factors) * 2.0  # Dirichlet参数（控制分散度）

            for attempt in range(max_combos * 20):  # 增加到20倍过采样
                # 生成归一化权重向量
                raw_weights = np.random.dirichlet(alpha)
                raw_weights *= target_sum  # 缩放到目标权重和

                # 映射到最近的网格点
                combo = tuple(
                    [
                        min(self.config.weight_grid_points, key=lambda x: abs(x - w))
                        for w in raw_weights
                    ]
                )
                weight_sum = sum(combo)

                if weight_sum_min <= weight_sum <= weight_sum_max:
                    if combo not in seen:
                        seen.add(combo)
                        valid_combos.append(combo)

                    if len(valid_combos) >= max_combos:
                        break

                # 每5000次尝试报告进度
                if (attempt + 1) % 5000 == 0:
                    self.logger.info(
                        f"  采样进度: {attempt+1:,}, 有效: {len(valid_combos):,}"
                    )

            self.logger.info(
                f"采样完成: {len(valid_combos):,} 个有效组合 (命中率 {len(valid_combos)/(attempt+1)*100:.2f}%)"
            )
        else:
            # 组合数 ≤ 10亿：直接遍历（单个计算快速）
            self.logger.info(f"理论组合数 {theoretical_combos:.2e}，直接遍历生成")
            combo_generator = itertools.product(
                self.config.weight_grid_points, repeat=n_factors
            )

            for combo in combo_generator:
                weight_sum = sum(combo)

                if weight_sum_min <= weight_sum <= weight_sum_max:
                    valid_combos.append(combo)

                    if len(valid_combos) >= max_combos:
                        break

        filter_rate = (
            (1 - len(valid_combos) / theoretical_combos) * 100
            if theoretical_combos > 0
            else 0
        )
        self.logger.info(
            f"有效组合: {len(valid_combos):,} (过滤率: {filter_rate:.3f}%), 权重和: [{weight_sum_min}, {weight_sum_max}]"
        )
        return valid_combos

    def _chunk_weight_combinations(
        self, weight_combos: List[Tuple[float, ...]]
    ) -> List[List[Tuple[float, ...]]]:
        """将权重组合分块"""
        chunks = []
        for i in range(0, len(weight_combos), self.config.chunk_size):
            chunk = weight_combos[i : i + self.config.chunk_size]
            chunks.append(chunk)
        return chunks

    def parallel_grid_search(
        self, panel=None, prices=None, factors=None
    ) -> pd.DataFrame:
        """并行网格搜索权重组合 - 支持多周期调仓，数据预加载和缓存

        Args:
            panel: 外部传入的因子面板(WFO场景用于IS/OOS数据切分)
            prices: 外部传入的价格矩阵(WFO场景用于IS/OOS数据切分)
            factors: 外部传入的因子列表
        """

        try:
            mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass

        self.logger.info("开始配置化并行网格搜索...")
        start_time = time.time()

        # 加载数据 - 支持外部传入(WFO场景)
        self.logger.info("=== 数据加载阶段 ===")

        stage_start = time.time()
        if panel is None:
            panel = self._load_factor_panel()
            self.logger.info(
                f"✓ 因子面板从文件加载完成，耗时: {time.time() - stage_start:.2f}秒"
            )
        else:
            self.logger.info(f"✓ 使用外部传入的因子面板 (WFO模式)，形状: {panel.shape}")

        stage_start = time.time()
        if prices is None:
            prices = self._load_price_data()
            self.logger.info(
                f"✓ 价格矩阵从文件加载完成，耗时: {time.time() - stage_start:.2f}秒"
            )
        else:
            self.logger.info(
                f"✓ 使用外部传入的价格矩阵 (WFO模式)，形状: {prices.shape}"
            )

        stage_start = time.time()
        if factors is None:
            factors = self._load_top_factors()
        # frozen dataclass需要用replace创建新实例
        from dataclasses import replace

        self.config = replace(self.config, factors=factors)
        self.logger.info(f"✓ 因子列表加载完成，耗时: {time.time() - stage_start:.2f}秒")
        self.logger.info(f"  实际使用因子数: {len(factors)}")

        # 生成权重组合
        self.logger.info("\n=== 权重组合生成 ===")
        stage_start = time.time()
        weight_combos = self._generate_weight_combinations()
        self.logger.info(f"✓ 权重组合生成完成，耗时: {time.time() - stage_start:.2f}秒")

        # 分块
        chunks = self._chunk_weight_combinations(weight_combos)
        total_tasks = len(chunks)
        total_rebalance_freqs = len(self.config.rebalance_freq_list)
        total_strategies = (
            len(weight_combos) * len(self.config.top_n_list) * total_rebalance_freqs
        )
        strategies_per_worker = total_strategies / self.config.n_workers

        self.logger.info(f"\n=== 并行执行 ===")
        self.logger.info(
            f"任务分块: {total_tasks}块 × {self.config.chunk_size}组合/块, {self.config.n_workers}进程并行"
        )
        self.logger.info(
            f"预计处理: {len(weight_combos):,}组合 × {len(self.config.top_n_list)}个Top-N × {total_rebalance_freqs}个调仓周期 = {total_strategies:,}策略"
        )
        self.logger.info(f"调仓周期: {self.config.rebalance_freq_list}日")
        self.logger.info(f"每进程负载: ~{strategies_per_worker:.0f}策略")

        # 对每个rebalance_freq执行回测
        all_results = []
        for rebalance_freq in self.config.rebalance_freq_list:
            self.logger.info(f"\n--- 开始回测调仓周期: {rebalance_freq}日 ---")
            freq_start = time.time()

            # 创建工作函数（对当前rebalance_freq）
            work_func = partial(
                self._process_weight_chunk,
                factors=factors,
                panel=panel,
                prices=prices,
                top_n_list=self.config.top_n_list,
                rebalance_freq=rebalance_freq,
            )

            # 并行执行当前rebalance_freq的所有块
            freq_results = []
            try:
                with mp.Pool(processes=self.config.n_workers) as pool:
                    # 使用tqdm显示进度
                    results_iter = pool.imap_unordered(work_func, chunks)

                    progress_bar = tqdm(
                        results_iter,
                        total=total_tasks,
                        desc=f"并行处理 ({self.config.n_workers}进程, rebalance={rebalance_freq}日)",
                        disable=not self.config.enable_progress_bar,
                    )

                    for chunk_results in progress_bar:
                        freq_results.extend(chunk_results)

            except Exception as e:
                self.logger.error(f"调仓周期{rebalance_freq}日的并行处理失败: {e}")
                raise

            # 记录当前rebalance_freq的结果统计
            freq_time = time.time() - freq_start
            self.logger.info(
                f"✓ 调仓周期{rebalance_freq}日完成: {len(freq_results):,}结果, 耗时: {freq_time:.2f}秒"
            )
            all_results.extend(freq_results)

        # 处理全部结果
        processing_time = time.time() - start_time
        n_failed = total_strategies - len(all_results)
        self.logger.info(
            f"✓ 所有调仓周期处理完成: {len(all_results):,}结果, 失败: {n_failed}, 总耗时: {processing_time:.2f}秒"
        )

        # 转换为DataFrame并排序
        df = pd.DataFrame(all_results)
        if len(df) > 0:
            df = df.sort_values("sharpe_ratio", ascending=False)

            # 计算性能指标
            total_strategies = (
                len(weight_combos)
                * len(self.config.top_n_list)
                * len(self.config.rebalance_freq_list)
            )
            speed = total_strategies / processing_time
            estimated_sequential_time = total_strategies / 142  # 基线速度142策/秒
            speedup = estimated_sequential_time / processing_time
            efficiency = speedup / self.config.n_workers * 100

            self.logger.info(
                f"✓ 速度: {speed:.1f}策略/秒, 加速比: {speedup:.1f}x, 并行效率: {efficiency:.1f}%"
            )

            # 结果统计
            self.logger.info(f"\n=== 结果统计 ===")
            best = df.iloc[0]
            self.logger.info(
                f"最优策略: sharpe={best['sharpe_ratio']:.3f}, 收益={best['total_return']:.2f}%, 回撤={best['max_drawdown']:.2f}%"
            )

            # 有效策略统计
            valid_strategies = df[df["sharpe_ratio"] > 0]
            good_strategies = df[df["sharpe_ratio"] > 0.5]
            self.logger.info(
                f"有效策略: {len(valid_strategies):,} (sharpe>0.5: {len(good_strategies):,})"
            )

            if len(df) >= 10:
                top10_sharpe_range = f"[{df.iloc[9]['sharpe_ratio']:.3f}, {df.iloc[0]['sharpe_ratio']:.3f}]"
                self.logger.info(f"Top10夏普范围: {top10_sharpe_range}")

        return df

    def backtest_specific_strategies(
        self, strategy_params: List[Dict], panel: pd.DataFrame, prices: pd.DataFrame
    ) -> pd.DataFrame:
        """回测指定的策略列表(用于WFO的OOS验证)

        Args:
            strategy_params: IS阶段选出的策略参数列表
                [{'weights': {...}, 'top_n': 3, 'rebalance_freq': 10}, ...]
            panel: OOS期的因子面板
            prices: OOS期的价格矩阵

        Returns:
            DataFrame with OOS performance metrics
        """
        try:
            mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass

        self.logger.info(f"开始回测{len(strategy_params)}个指定策略 (OOS验证模式)...")
        start_time = time.time()

        # 加载因子列表
        factors = self._load_top_factors()
        from dataclasses import replace

        self.config = replace(self.config, factors=factors)

        # 按调仓频率分组以便并行处理
        freq_groups = {}
        for params in strategy_params:
            freq = params["rebalance_freq"]
            if freq not in freq_groups:
                freq_groups[freq] = []
            freq_groups[freq].append(params)

        self.logger.info(f"策略按{len(freq_groups)}个调仓频率分组")

        # 对每个频率组并行回测
        all_results = []
        for freq_idx, (freq, params_list) in enumerate(freq_groups.items(), 1):
            self.logger.info(
                f"\n处理频率组 {freq_idx}/{len(freq_groups)}: 调仓={freq}天, 策略数={len(params_list)}"
            )

            # 构造成chunk格式复用现有架构
            # 每个chunk包含该频率的所有权重组合
            weights_list = [p["weights"] for p in params_list]
            top_n_list = [p["top_n"] for p in params_list]

            # 准备worker函数参数
            work_func = partial(
                self._process_specific_strategies,
                factors=factors,
                panel=panel,
                prices=prices,
                rebalance_freq=freq,
            )

            # 将策略参数分组打包
            strategy_chunks = []
            for params in params_list:
                strategy_chunks.append(
                    {"weights": params["weights"], "top_n": params["top_n"]}
                )

            # 分块并行处理
            chunk_size = self.config.chunk_size
            chunks = [
                strategy_chunks[i : i + chunk_size]
                for i in range(0, len(strategy_chunks), chunk_size)
            ]

            # 并行回测
            chunk_start_count = len(all_results)
            with mp.Pool(processes=self.config.n_workers) as pool:
                chunk_results = pool.map(work_func, chunks)
                for chunk_result in chunk_results:
                    all_results.extend(chunk_result)

            # ✅ 修复：简化日志，直接计算本次增量
            completed_count = len(all_results) - chunk_start_count
            self.logger.info(f"  ✓ 完成{completed_count}个策略回测")

        # 整理结果
        total_time = time.time() - start_time

        if not all_results:
            self.logger.error(f"\n指定策略回测失败：没有返回任何结果！")
            return pd.DataFrame()  # 返回空DataFrame

        df = pd.DataFrame(all_results)
        df = df.sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)

        self.logger.info(f"\n指定策略回测完成，耗时: {total_time:.2f}秒")
        self.logger.info(f"有效策略数: {len(df[df['sharpe_ratio'] > 0])}/{len(df)}")

        if len(df) > 0:
            best = df.iloc[0]
            self.logger.info(
                f"最优策略: sharpe={best['sharpe_ratio']:.3f}, "
                f"收益={best['total_return']:.2f}%, "
                f"回撤={best['max_drawdown']:.2f}%"
            )

        return df

    def _process_specific_strategies(
        self,
        strategy_chunk: List[Dict],
        factors: List[str],
        panel: pd.DataFrame,
        prices: pd.DataFrame,
        rebalance_freq: int,
    ) -> List[Dict]:
        """处理指定的策略chunk (OOS验证用) - 复用_process_weight_chunk"""
        # 将weights从dict转为tuple (保证顺序与factors一致)
        weight_list = []
        for s in strategy_chunk:
            weights_dict = s["weights"]
            # 按照factors顺序构造tuple
            weight_tuple = tuple(weights_dict[f] for f in factors)
            weight_list.append(weight_tuple)

        # ✅ 修复：使用配置的全部top_n值，而不是从chunk提取
        # 这样每个chunk都会测试所有top_n，保证结果数量一致
        top_n_list = self.config.top_n_list

        # 调用现有的_process_weight_chunk方法
        try:
            results = self._process_weight_chunk(
                weight_chunk=weight_list,
                factors=factors,
                panel=panel,
                prices=prices,
                top_n_list=top_n_list,
                rebalance_freq=rebalance_freq,
            )
            # 直接返回所有结果，不做筛选 (因为weights已经对应)
            return results
        except Exception as e:
            # ✅ 修复：强制记录错误，不依赖config.log_errors
            self.logger.error(f"OOS回测chunk失败 (len={len(strategy_chunk)}): {e}")
            import traceback

            self.logger.error(traceback.format_exc())
            return []

    def run_parallel_backtest(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """运行完整的配置化并行回测"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        preset_info = (
            f" (预设: {self.config.current_preset})"
            if self.config.current_preset
            else ""
        )

        print("=" * 80)
        print("ETF轮动回测引擎 - 配置化并行计算版本")
        print("=" * 80)
        print(f"时间戳: {timestamp}")
        print(f"预设: {self.config.current_preset or '默认'}")
        print(f"工作进程数: {self.config.n_workers}")
        print(f"块大小: {self.config.chunk_size}")
        print(f"最大组合数: {self.config.max_combinations}")
        print(f"内存限制: {self.config.max_memory_usage_gb}GB")
        print(f"面板: {self.config.panel_file}")
        print(f"筛选: {self.config.screening_file}")

        # 并行网格搜索
        print(
            f"\n开始配置化并行回测: {self.config.max_combinations}个权重组合{preset_info}"
        )
        start_time = time.time()

        results = self.parallel_grid_search()

        total_time = time.time() - start_time
        print(f"\n配置化并行回测完成，总耗时: {total_time:.2f}秒")

        # 保存结果 - 创建时间戳文件夹，保存Top N配置
        output_path = Path(self.config.output_dir)
        timestamp_folder = output_path / f"backtest_{timestamp}"
        timestamp_folder.mkdir(parents=True, exist_ok=True)

        # 限制保存Top N，减少文件大小
        top_results = results.head(self.config.save_top_results)
        csv_file = timestamp_folder / "results.csv"
        top_results.to_csv(csv_file, index=False)
        print(
            f"结果保存至: {csv_file} (Top{self.config.save_top_results}/{len(results)}策略)"
        )

        # 输出Top N
        top_n = min(self.config.save_top_results, len(results))
        print(f"\nTop {top_n} 策略:")
        print(results.head(top_n).to_string(index=False))

        # 保存最优策略配置
        if len(results) > 0:
            best = results.iloc[0]
            best_config = {
                "timestamp": timestamp,
                "engine_type": "configurable_parallel",
                "preset": self.config.current_preset,
                "config": {
                    "n_workers": self.config.n_workers,
                    "chunk_size": self.config.chunk_size,
                    "max_combinations": self.config.max_combinations,
                    "top_n_list": self.config.top_n_list,
                    "rebalance_freq_list": self.config.rebalance_freq_list,
                    "fees": self.config.fees,
                    "init_cash": self.config.init_cash,
                    "weight_grid_points": self.config.weight_grid_points,
                    "weight_sum_range": self.config.weight_sum_range,
                },
                "weights": best["weights"],
                "top_n": int(best["top_n"]),
                "rebalance_freq": int(
                    best.get("rebalance_freq", self.config.rebalance_freq_list[0])
                ),
                "performance": {
                    "total_return": float(best["total_return"]),
                    "sharpe_ratio": float(best["sharpe_ratio"]),
                    "max_drawdown": float(best["max_drawdown"]),
                },
                "factors": self._load_top_factors(),
                "timing": {
                    "total_time": total_time,
                    "strategies_tested": len(results),
                    "speed_per_second": len(results) / total_time,
                },
                "data_source": {
                    "panel": self.config.panel_file,
                    "screening": self.config.screening_file,
                    "price_dir": self.config.price_dir,
                },
            }

            if self.config.save_best_config:
                config_file = timestamp_folder / "best_config.json"
                with open(config_file, "w") as f:
                    json.dump(best_config, f, indent=2, ensure_ascii=False)
                print(f"最优配置保存至: {config_file}")

            # 保存日志文件
            log_file = timestamp_folder / "backtest.log"
            with open(log_file, "w") as f:
                f.write(f"ETF轮动回测引擎 - 配置化并行计算版本\n")
                f.write(f"时间戳: {timestamp}\n")
                f.write(f"预设: {self.config.current_preset or '默认'}\n")
                f.write(f"\n{'='*80}\n")
                f.write(f"📊 回测配置\n")
                f.write(f"{'='*80}\n")
                f.write(
                    f"工作进程: {self.config.n_workers} | 块大小: {self.config.chunk_size} | 最大组合: {self.config.max_combinations:,}\n"
                )
                f.write(
                    f"因子数: {len(best_config.get('factors', []))} | Top-N范围: {self.config.top_n_list} | 调仓周期: {self.config.rebalance_freq_list}日\n"
                )
                f.write(
                    f"费率模型: 佣金0.2% + 印花税0.1% + 滑点0.01% = {self.config.fees*100:.1f}% 往返\n"
                )
                f.write(
                    f"权重网格: {len(self.config.weight_grid_points)}个点 (理论{len(best_config.get('factors', []))**len(self.config.weight_grid_points):,}组合)\n"
                )

                f.write(f"\n{'='*80}\n")
                f.write(f"📁 数据源\n")
                f.write(f"{'='*80}\n")
                panel_name = Path(self.config.panel_file).parent.name
                screening_name = Path(self.config.screening_file).parent.name
                f.write(f"因子面板: {panel_name}\n")
                f.write(f"因子筛选: {screening_name}\n")
                f.write(f"价格数据: {Path(self.config.price_dir).name}\n")

                f.write(f"\n{'='*80}\n")
                f.write(f"⚡ 执行统计\n")
                f.write(f"{'='*80}\n")
                f.write(f"总耗时: {total_time:.2f}秒 ({total_time/60:.1f}分钟)\n")
                f.write(f"总策略: {len(results):,}个\n")
                f.write(f"处理速度: {len(results)/total_time:.1f}策略/秒\n")
                f.write(
                    f"结果保存: Top {self.config.save_top_results} / {len(results):,}\n"
                )

                f.write(f"\n{'='*80}\n")
                f.write(f"🏆 最优策略 (Rank 1)\n")
                f.write(f"{'='*80}\n")
                f.write(f"夏普比率: {best['sharpe_ratio']:.4f}\n")
                f.write(f"总收益率: {best['total_return']:.2f}%\n")
                f.write(f"最大回撤: {best['max_drawdown']:.2f}%\n")
                f.write(
                    f"Calmar比率: {best['total_return'] / abs(best['max_drawdown']):.2f}\n"
                )
                f.write(f"持仓数量: {int(best['top_n'])}只\n")

                # 权重分析
                import ast

                weights = (
                    best["weights"]
                    if isinstance(best["weights"], dict)
                    else ast.literal_eval(best["weights"])
                )
                sorted_weights = sorted(
                    [(k, v) for k, v in weights.items() if v > 0],
                    key=lambda x: x[1],
                    reverse=True,
                )
                f.write(f"\n权重分配:\n")
                for factor, weight in sorted_weights:
                    f.write(f"  • {factor}: {weight:.2f}\n")

                f.write(f"\n{'='*80}\n")
                f.write(f"📈 性能分布\n")
                f.write(f"{'='*80}\n")
                f.write(f"平均Sharpe: {results['sharpe_ratio'].mean():.4f}\n")
                f.write(f"中位数Sharpe: {results['sharpe_ratio'].median():.4f}\n")
                f.write(f"最高Sharpe: {results['sharpe_ratio'].max():.4f}\n")
                f.write(f"最低Sharpe: {results['sharpe_ratio'].min():.4f}\n")
                f.write(
                    f"正期望(Sharpe>0): {len(results[results['sharpe_ratio'] > 0]):,} ({len(results[results['sharpe_ratio'] > 0])/len(results)*100:.1f}%)\n"
                )
                f.write(
                    f"优秀策略(Sharpe>0.5): {len(results[results['sharpe_ratio'] > 0.5]):,} ({len(results[results['sharpe_ratio'] > 0.5])/len(results)*100:.1f}%)\n"
                )
                f.write(
                    f"高质量(Sharpe>0.7): {len(results[results['sharpe_ratio'] > 0.7]):,} ({len(results[results['sharpe_ratio'] > 0.7])/len(results)*100:.1f}%)\n"
                )

                f.write(f"\n{'='*80}\n")
                f.write(f"✅ Top 5 策略\n")
                f.write(f"{'='*80}\n")
                for idx in range(min(5, len(results))):
                    row = results.iloc[idx]
                    f.write(
                        f"#{idx+1}: Sharpe={row['sharpe_ratio']:.4f} | Return={row['total_return']:.2f}% | DD={row['max_drawdown']:.2f}% | Top_N={int(row['top_n'])}\n"
                    )

        return results, best_config if len(results) > 0 else {}


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="配置化并行ETF轮动回测引擎",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用默认配置
  python parallel_backtest_configurable.py

  # 使用指定配置文件
  python parallel_backtest_configurable.py --config-file my_config.yaml

  # 使用预设场景
  python parallel_backtest_configurable.py --preset comprehensive

  # 创建默认配置文件
  python parallel_backtest_configurable.py --create-config
        """,
    )

    parser.add_argument(
        "--config-file",
        type=str,
        help="配置文件路径 (默认: parallel_backtest_config.yaml)",
    )

    parser.add_argument("--preset", type=str, help="使用的预设场景名称")

    parser.add_argument("--create-config", action="store_true", help="创建默认配置文件")

    args = parser.parse_args()

    # 创建默认配置文件
    if args.create_config:
        from config_loader_parallel import create_default_parallel_config

        config_path = args.config_file or "parallel_backtest_config.yaml"
        create_default_parallel_config(config_path)
        print(f"默认配置文件已创建: {config_path}")
        return

    # 加载配置（零开销快速配置）
    try:
        config = load_fast_config_from_args(args)
    except Exception as e:
        print(f"配置加载失败: {e}")
        sys.exit(1)

    # 创建引擎并运行
    engine = ConfigurableParallelBacktestEngine(config)

    try:
        results, best_config = engine.run_parallel_backtest()

        print("\n🎯 配置化并行优化总结:")
        print(f"处理时间: {best_config['timing']['total_time']:.2f}秒")
        print(f"处理速度: {best_config['timing']['speed_per_second']:.1f}策略/秒")
        print(f"工作进程: {best_config['config']['n_workers']}个")
        print(f"最大组合: {best_config['config']['max_combinations']}")
        print(f"当前预设: {best_config['preset'] or '默认'}")
        print(f"最优夏普比率: {best_config['performance']['sharpe_ratio']:.3f}")
        print(f"最优收益率: {best_config['performance']['total_return']:.2f}%")

    except Exception as e:
        print(f"回测执行失败: {e}")
        if config.log_errors:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
