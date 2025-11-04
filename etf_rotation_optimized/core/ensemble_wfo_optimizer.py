"""
Ensemble WFO Optimizer | 集成Walk-Forward优化器

基于ConstrainedWalkForwardOptimizer扩展,实现:
1. 智能因子组合采样 (1000个5因子组合)
2. 批量向量化评估
3. Top10集成预测
4. 抗过拟合机制

作者: AI Agent
日期: 2025-01-XX
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from core.constrained_walk_forward_optimizer import (
    ConstrainedWalkForwardOptimizer,
)
from core.ensemble_sampler import EnsembleSampler
from core.factor_weighting import FactorWeighting
from core.ic_calculator_numba import ICCalculatorNumba as ICCalculator

logger = logging.getLogger(__name__)


@dataclass
class EnsembleWindowResult:
    """单窗口集成结果"""

    window_index: int
    """窗口索引"""

    is_start: int
    is_end: int
    oos_start: int
    oos_end: int
    """时间范围"""

    n_sampled_combos: int
    """采样的组合数量"""

    is_ic_scores: Dict[str, float]
    """IS阶段各因子IC"""

    top10_combos: List[Tuple[str, ...]]
    """Top10最优组合"""

    top10_is_ics: List[float]
    """Top10 IS阶段IC"""

    oos_ensemble_ic: float
    """OOS阶段集成IC"""

    oos_ensemble_sharpe: float
    """OOS阶段集成Sharpe"""

    oos_single_ics: Dict[Tuple[str, ...], float]
    """OOS阶段各组合IC (用于验证)"""


class EnsembleWFOOptimizer(ConstrainedWalkForwardOptimizer):
    """
    集成Walk-Forward优化器

    继承ConstrainedWalkForwardOptimizer,增强为:
    - 每个窗口采样1000个5因子组合
    - 批量评估所有组合的IS性能
    - 选择Top10组合进行OOS集成
    - 使用梯度衰减权重抗过拟合
    """

    def __init__(
        self,
        constraints_config: Dict,
        n_samples: int = 1000,
        combo_size: int = 5,
        top_k: int = 10,
        weighting_scheme: str = "gradient_decay",
        gradient_decay_rate: float = 0.5,
        random_seed: int = 42,
        verbose: bool = True,
    ):
        """
        初始化集成WFO优化器

        Args:
            constraints_config: 约束配置 (family_quotas + mutual_exclusions)
            n_samples: 每窗口采样组合数 (默认1000)
            combo_size: 每组合因子数 (默认5)
            top_k: 集成的最优组合数 (默认10, 抗过拟合)
            weighting_scheme: 加权方案 ('equal'/'ic_weighted'/'gradient_decay')
            gradient_decay_rate: 梯度衰减率 (默认0.5)
            random_seed: 随机种子 (确保可复现)
            verbose: 是否打印日志
        """
        # 调用父类初始化
        super().__init__(selector=None, verbose=verbose)

        # 集成组件
        self.sampler = EnsembleSampler(constraints_config, random_seed=random_seed)
        self.weighter = FactorWeighting()
        self.ic_calculator = ICCalculator()

        # 超参数
        self.n_samples = n_samples
        self.combo_size = combo_size
        self.top_k = top_k
        self.weighting_scheme = weighting_scheme
        self.gradient_decay_rate = gradient_decay_rate
        self.random_seed = random_seed

        # 结果存储
        self.window_results: List[EnsembleWindowResult] = []

        if verbose:
            logger.info("=" * 80)
            logger.info("🚀 Ensemble WFO Optimizer 初始化")
            logger.info("=" * 80)
            logger.info(f"采样策略: {n_samples}个组合, 每组合{combo_size}因子")
            logger.info(f"集成策略: Top{top_k}, 权重方案={weighting_scheme}")
            logger.info(f"随机种子: {random_seed}")
            logger.info("=" * 80)

    def run_ensemble_wfo(
        self,
        factors_data: np.ndarray,
        returns: np.ndarray,
        factor_names: List[str],
        is_period: int = 100,
        oos_period: int = 20,
        step_size: int = 20,
    ) -> pd.DataFrame:
        """
        运行集成Walk-Forward优化

        Args:
            factors_data: 因子数据 (time_steps, num_assets, num_factors)
            returns: 收益率 (time_steps, num_assets)
            factor_names: 因子名称列表
            is_period: IS窗口长度
            oos_period: OOS窗口长度
            step_size: 滑动步长

        Returns:
            汇总DataFrame: 每窗口OOS性能指标
        """
        num_time_steps, num_assets, num_factors = factors_data.shape

        if self.verbose:
            logger.info("\n" + "=" * 80)
            logger.info("开始 Ensemble WFO 优化")
            logger.info("=" * 80)
            logger.info(
                f"数据形状: {num_time_steps} 日期 × {num_assets} 资产 × {num_factors} 因子"
            )
            logger.info(f"窗口设置: IS={is_period}, OOS={oos_period}, step={step_size}")

        # 划分窗口
        windows = self._partition_windows(
            num_time_steps, is_period, oos_period, step_size
        )

        if self.verbose:
            logger.info(f"总窗口数: {len(windows)}")
            logger.info("=" * 80)

        # 逐窗口优化
        import time

        from .performance_monitor import PerformanceMonitor

        window_times = []
        for window_idx, (is_start, is_end, oos_start, oos_end) in enumerate(windows):
            window_start_time = time.time()

            if self.verbose:
                logger.info(
                    f"\n{'─'*80}\n【窗口 {window_idx+1}/{len(windows)}】"
                    f"IS: [{is_start}, {is_end}), OOS: [{oos_start}, {oos_end})\n{'─'*80}"
                )

            # 进度心跳
            if window_idx % 10 == 0 and window_idx > 0:
                avg_time = np.mean(window_times[-10:])
                eta = avg_time * (len(windows) - window_idx)
                logger.info(
                    f"🔄 进度: {window_idx}/{len(windows)} 窗口完成 "
                    f"({window_idx/len(windows)*100:.1f}%) | "
                    f"平均耗时: {avg_time:.1f}s | ETA: {eta/60:.1f}min"
                )

            # 运行单窗口优化
            with PerformanceMonitor.timer(f"窗口{window_idx+1}"):
                window_result = self._run_single_window(
                    factors_data,
                    returns,
                    factor_names,
                    is_start,
                    is_end,
                    oos_start,
                    oos_end,
                    window_idx,
                )

            self.window_results.append(window_result)

            window_elapsed = time.time() - window_start_time
            window_times.append(window_elapsed)

            if self.verbose:
                logger.info(
                    f"✓ 窗口{window_idx+1} 完成: "
                    f"OOS IC={window_result.oos_ensemble_ic:.4f}, "
                    f"Sharpe={window_result.oos_ensemble_sharpe:.2f} | "
                    f"耗时: {window_elapsed:.1f}s"
                )

        # 生成汇总报告
        summary_df = self._generate_summary_report()

        if self.verbose:
            logger.info("\n" + "=" * 80)
            logger.info("Ensemble WFO 优化完成")
            logger.info("=" * 80)
            logger.info(f"总窗口数: {len(self.window_results)}")
            logger.info(f"平均OOS IC: {summary_df['oos_ensemble_ic'].mean():.4f}")
            logger.info(
                f"平均OOS Sharpe: {summary_df['oos_ensemble_sharpe'].mean():.2f}"
            )
            logger.info("=" * 80)

        return summary_df

    def _run_single_window(
        self,
        factors_data: np.ndarray,
        returns: np.ndarray,
        factor_names: List[str],
        is_start: int,
        is_end: int,
        oos_start: int,
        oos_end: int,
        window_idx: int,
    ) -> EnsembleWindowResult:
        """
        运行单窗口优化 - 核心6步流程

        Steps:
        1. IS数据切片 (T-1对齐)
        2. 计算IS IC评分
        3. 智能采样1000个组合
        4. 批量评估所有组合IS性能
        5. 选择Top10组合
        6. OOS集成预测 + 性能评估

        Returns:
            单窗口集成结果
        """
        # Step 1: IS数据切片 + T-1对齐
        from .data_contract import align_factor_to_return

        is_factors_raw = factors_data[is_start:is_end]
        is_returns_raw = returns[is_start:is_end]

        # T-1对齐: 因子[t] 预测 收益[t+1]
        is_factors, is_returns = align_factor_to_return(is_factors_raw, is_returns_raw)

        # Step 2: 计算IS阶段各因子IC
        is_ic_scores = self._compute_window_ic(is_factors, is_returns, factor_names)

        if self.verbose:
            logger.info(
                f"Step 2: IS IC计算完成, "
                f"平均IC={np.mean(list(is_ic_scores.values())):.4f}"
            )

        # Step 3: 智能采样1000个组合
        sampled_combos = self.sampler.sample_combinations(
            n_samples=self.n_samples,
            factor_pool=factor_names,
            ic_scores=is_ic_scores,
            combo_size=self.combo_size,
        )

        if self.verbose:
            logger.info(f"Step 3: 采样完成, {len(sampled_combos)} 个组合")

        # Step 4: 批量评估所有组合的IS性能
        combo_is_ics = self._batch_evaluate_combos(
            sampled_combos, is_factors, is_returns, factor_names
        )

        if self.verbose:
            logger.info(
                f"Step 4: 批量评估完成, "
                f"IS IC范围=[{min(combo_is_ics):.4f}, {max(combo_is_ics):.4f}]"
            )

        # Step 5: 选择Top10组合 (抗过拟合)
        top_indices = np.argsort(combo_is_ics)[-self.top_k :][::-1]
        top10_combos = [sampled_combos[i] for i in top_indices]
        top10_is_ics = [combo_is_ics[i] for i in top_indices]

        if self.verbose:
            logger.info(
                f"Step 5: Top{self.top_k}选择完成, "
                f"IS IC范围=[{min(top10_is_ics):.4f}, {max(top10_is_ics):.4f}]"
            )

        # Step 6: OOS集成预测 - T-1对齐
        from .data_contract import align_factor_to_return

        # 预计算因子名称到索引的映射（避免重复查找）
        factor_name_to_idx = {name: idx for idx, name in enumerate(factor_names)}

        # Step 6: OOS集成预测 + 性能评估（向量化优化）
        # 6.1 提取OOS窗口数据并T-1对齐
        oos_factors_raw = factors_data[oos_start:oos_end]
        oos_returns_raw = returns[oos_start:oos_end]

        # T-1对齐：因子[t] 预测 收益[t+1]
        oos_factors, oos_returns = align_factor_to_return(
            oos_factors_raw, oos_returns_raw
        )

        # 6.2 批量提取Top10组合的因子索引
        top10_indices = np.array(
            [[factor_name_to_idx[f] for f in combo] for combo in top10_combos]
        )  # (10, 5)

        # 向量化提取因子数据: (T, N, K) → (10, T, N, 5)
        # 使用高级索引一次性提取所有组合
        top10_factors = oos_factors[:, :, top10_indices.T]  # (T, N, 5, 10)
        top10_factors = np.transpose(top10_factors, (3, 0, 1, 2))  # (10, T, N, 5)

        # 向量化计算每个组合的信号（等权平均）
        ensemble_signals_array = np.mean(top10_factors, axis=3)  # (10, T, N)

        # 向量化计算每个组合的OOS IC
        oos_single_ics = {}
        for i, combo in enumerate(top10_combos):
            combo_ic = self._compute_signal_ic(ensemble_signals_array[i], oos_returns)
            oos_single_ics[combo] = combo_ic

        # 计算集成权重 (基于IS IC排序)
        # 使用简化方式: 直接根据IC计算权重,无需FactorWeighting
        if self.weighting_scheme == "equal":
            combo_weights = np.ones(self.top_k) / self.top_k
        elif self.weighting_scheme == "ic_weighted":
            ic_array = np.array(top10_is_ics)
            combo_weights = ic_array / ic_array.sum()
        elif self.weighting_scheme == "gradient_decay":
            # 梯度衰减: w_i = exp(-decay_rate*i) / Z
            # 从配置读取衰减率，默认0.5
            decay_rate = getattr(self, "gradient_decay_rate", 0.5)
            ranks = np.arange(self.top_k)
            weights = np.exp(-decay_rate * ranks)
            combo_weights = weights / weights.sum()
        else:
            combo_weights = np.ones(self.top_k) / self.top_k

        # 加权集成: ensemble_signal = Σ(w_i * signal_i)
        final_signal = np.tensordot(
            combo_weights, ensemble_signals_array, axes=([0], [0])
        )  # (T, N)

        # 6.3 计算OOS性能
        oos_ensemble_ic = self._compute_signal_ic(final_signal, oos_returns)

        # 计算Sharpe (按日横截面相关的时序稳定性) - 向量化
        T_oos = min(len(final_signal), len(oos_returns))
        sig = final_signal[:T_oos]
        ret = oos_returns[:T_oos]

        # 行内标准差与有效掩码
        sig_std = np.nanstd(sig, axis=1)
        ret_std = np.nanstd(ret, axis=1)
        valid_mask = (sig_std > 1e-10) & (ret_std > 1e-10)

        if np.any(valid_mask):
            sig_mean = np.nanmean(sig, axis=1, keepdims=True)
            ret_mean = np.nanmean(ret, axis=1, keepdims=True)
            sig_norm = (sig - sig_mean) / (sig_std[:, None] + 1e-10)
            ret_norm = (ret - ret_mean) / (ret_std[:, None] + 1e-10)
            ic_series = np.nanmean(sig_norm * ret_norm, axis=1)
            ic_series = ic_series[valid_mask]
            ic_std = np.nanstd(ic_series)
            oos_ensemble_sharpe = (
                float(np.nanmean(ic_series) / ic_std)
                if ic_series.size > 0 and ic_std > 1e-12
                else 0.0
            )
        else:
            oos_ensemble_sharpe = 0.0

        if self.verbose:
            logger.info(
                f"Step 6: OOS集成完成, IC={oos_ensemble_ic:.4f}, "
                f"Sharpe={oos_ensemble_sharpe:.2f}"
            )

        # 返回窗口结果
        return EnsembleWindowResult(
            window_index=window_idx,
            is_start=is_start,
            is_end=is_end,
            oos_start=oos_start,
            oos_end=oos_end,
            n_sampled_combos=len(sampled_combos),
            is_ic_scores=is_ic_scores,
            top10_combos=top10_combos,
            top10_is_ics=top10_is_ics,
            oos_ensemble_ic=oos_ensemble_ic,
            oos_ensemble_sharpe=oos_ensemble_sharpe,
            oos_single_ics=oos_single_ics,
        )

    def _batch_evaluate_combos(
        self,
        combos: List[Tuple[str, ...]],
        is_factors: np.ndarray,
        is_returns: np.ndarray,
        factor_names: List[str],
    ) -> List[float]:
        """
        批量评估所有组合的IS IC性能 - 全向量化版本

        Args:
            combos: 因子组合列表
            is_factors: IS因子数据 (T, N, K)
            is_returns: IS收益数据 (T, N)
            factor_names: 因子名称列表

        Returns:
            每个组合的IS IC列表
        """
        from .performance_monitor import PerformanceMonitor

        with PerformanceMonitor.timer("批量组合评估"):
            # 预计算因子索引映射 - O(K)一次性完成
            factor_idx_map = {name: idx for idx, name in enumerate(factor_names)}

            # 批量提取所有组合的索引 - O(C*5)
            combo_indices = np.array(
                [[factor_idx_map[f] for f in combo] for combo in combos]
            )  # (C, 5)

            # 向量化提取所有组合因子 - O(1)高级索引
            # (T, N, K) → (C, T, N, 5)
            all_combo_factors = is_factors[:, :, combo_indices.T]  # (T, N, 5, C)
            all_combo_factors = np.transpose(
                all_combo_factors, (3, 0, 1, 2)
            )  # (C, T, N, 5)

            # 向量化等权合并 - O(1)
            all_signals = np.mean(all_combo_factors, axis=3)  # (C, T, N)

            # 向量化IC计算
            combo_ics = self._compute_batch_ic(all_signals, is_returns)

        return combo_ics

    def _compute_signal_ic(self, signal: np.ndarray, returns: np.ndarray) -> float:
        """
        计算信号与收益的IC (Information Coefficient) - 向量化版本

        Args:
            signal: 预测信号 (T, N)
            returns: 实际收益 (T, N)

        Returns:
            平均IC
        """
        T = min(len(signal), len(returns))

        ic_series = []

        for t in range(T):
            signal_t = signal[t]
            return_t = returns[t]

            # 删除NaN值
            valid_mask = ~np.isnan(signal_t) & ~np.isnan(return_t)
            signal_valid = signal_t[valid_mask]
            return_valid = return_t[valid_mask]

            # 至少需要2个有效数据点来计算相关
            if len(signal_valid) >= 10:
                if np.std(signal_valid) > 1e-10 and np.std(return_valid) > 1e-10:
                    # 标准化
                    signal_norm = (signal_valid - np.mean(signal_valid)) / (
                        np.std(signal_valid) + 1e-10
                    )
                    return_norm = (return_valid - np.mean(return_valid)) / (
                        np.std(return_valid) + 1e-10
                    )

                    # 计算相关系数
                    ic_t = np.corrcoef(signal_norm, return_norm)[0, 1]
                    if not np.isnan(ic_t):
                        ic_series.append(ic_t)

        return np.mean(ic_series) if ic_series else 0.0

    def _compute_batch_ic(
        self, signals: np.ndarray, returns: np.ndarray
    ) -> List[float]:
        """
        批量计算多个信号的IC - 全向量化（处理NaN）

        Args:
            signals: (C, T, N) C个组合的信号
            returns: (T, N) 收益

        Returns:
            C个IC值
        """
        from .performance_monitor import PerformanceMonitor

        with PerformanceMonitor.timer("批量IC计算"):
            eps = 1e-10
            # 形状
            C, T, N = signals.shape

            # (C, T, 1) 与 (1, T, 1)
            sig_mean = np.nanmean(signals, axis=2, keepdims=True)
            sig_std = np.nanstd(signals, axis=2, keepdims=True)
            ret_mean = np.nanmean(returns, axis=1, keepdims=True)
            ret_std = np.nanstd(returns, axis=1, keepdims=True)

            # 标准化，NaN保持
            sig_norm = (signals - sig_mean) / (sig_std + eps)  # (C, T, N)
            ret_norm = (returns - ret_mean) / (ret_std + eps)  # (T, N)

            # 逐日横截面相关（皮尔逊近似Spearman的秩相关）
            ic_matrix = np.nanmean(sig_norm * ret_norm[None, :, :], axis=2)  # (C, T)

            # 有效性掩码：有效样本数>=10 且 std>0
            valid_count = np.sum(
                ~np.isnan(signals) & ~np.isnan(returns[None, :, :]), axis=2
            )  # (C, T)
            sig_std_2d = sig_std.squeeze(-1)  # (C, T)
            ret_std_1d = ret_std.squeeze(-1)  # (T,)
            valid_mask = (
                (valid_count >= 10) & (sig_std_2d > eps) & (ret_std_1d[None, :] > eps)
            )

            ic_masked = np.where(valid_mask, ic_matrix, np.nan)  # (C, T)
            combo_ics = np.nanmean(ic_masked, axis=1)  # (C,)

        return np.nan_to_num(combo_ics, nan=0.0).astype(float).tolist()

    def _generate_summary_report(self) -> pd.DataFrame:
        """
        生成汇总报告DataFrame

        Returns:
            每窗口性能指标汇总
        """
        records = []

        for result in self.window_results:
            records.append(
                {
                    "window_index": result.window_index,
                    "is_start": result.is_start,
                    "is_end": result.is_end,
                    "oos_start": result.oos_start,
                    "oos_end": result.oos_end,
                    "n_sampled_combos": result.n_sampled_combos,
                    "top10_mean_is_ic": (
                        float(np.mean(result.top10_is_ics))
                        if len(result.top10_is_ics) > 0
                        else 0.0
                    ),
                    "oos_ensemble_ic": result.oos_ensemble_ic,
                    "oos_ensemble_sharpe": result.oos_ensemble_sharpe,
                    "top10_combos": str(result.top10_combos[:3]),
                }
            )

        return pd.DataFrame(records)

    def save_results(self, output_dir: Path):
        """
        保存结果到文件

        Args:
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 保存汇总报告
        summary_df = self._generate_summary_report()
        summary_path = output_dir / "ensemble_wfo_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        if self.verbose:
            logger.info(f"✓ 汇总报告已保存: {summary_path}")

        # 2. 保存详细窗口结果 (JSON格式)
        import json

        detailed_results = []
        for result in self.window_results:
            detailed_results.append(
                {
                    "window_index": result.window_index,
                    "time_range": {
                        "is_start": result.is_start,
                        "is_end": result.is_end,
                        "oos_start": result.oos_start,
                        "oos_end": result.oos_end,
                    },
                    "sampling": {"n_combos": result.n_sampled_combos},
                    "top10_combos": [list(c) for c in result.top10_combos],
                    "top10_is_ics": result.top10_is_ics,
                    "oos_metrics": {
                        "ensemble_ic": result.oos_ensemble_ic,
                        "ensemble_sharpe": result.oos_ensemble_sharpe,
                    },
                }
            )

        detailed_path = output_dir / "ensemble_wfo_detailed.json"
        with open(detailed_path, "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)

        if self.verbose:
            logger.info(f"✓ 详细结果已保存: {detailed_path}")
