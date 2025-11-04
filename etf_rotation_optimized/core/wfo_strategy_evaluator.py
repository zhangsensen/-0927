"""
WFO策略评估器（独立模块）

职责：
- 评估单个策略的收益和KPI
- 支持并行化调用
- 无状态，纯函数式
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .wfo_multi_strategy_selector import StrategySpec


class WFOStrategyEvaluator:
    """策略评估器：纯函数式，支持并行

    ⚡ Linus优化: 共享selector实例，避免重复创建
    """

    # 类级别共享实例（线程安全，因为只读操作）
    _shared_selector = None

    @classmethod
    def _get_selector(cls):
        """获取共享selector实例"""
        if cls._shared_selector is None:
            from .wfo_multi_strategy_selector import WFOMultiStrategySelector

            cls._shared_selector = WFOMultiStrategySelector()
        return cls._shared_selector

    @staticmethod
    def evaluate_single_strategy(
        spec: StrategySpec,
        results_list: List,
        factors: np.ndarray,
        returns: np.ndarray,
        factor_names: List[str],
        dates: Optional[pd.DatetimeIndex] = None,
    ) -> Dict:
        """评估单个策略，返回字典"""
        # 使用共享实例（避免重复创建）
        selector = WFOStrategyEvaluator._get_selector()

        # 1. 拼接信号
        signals = selector._stitch_signals_for_spec(
            spec, results_list, factors, factor_names
        )

        # 2. Z分数过滤
        if spec.z_threshold is not None:
            signals = selector._apply_z_threshold(signals, float(spec.z_threshold))

        # 3. 计算收益和换手
        daily_ret, daily_to = selector._topn_tplus1_returns_and_turnover(
            signals, returns, spec.top_n
        )
        if dates is not None and len(dates) == len(daily_ret):
            daily_ret.index = dates
            daily_to.index = dates

        # 4. 计算KPI
        kpi = selector._compute_kpis(daily_ret)
        avg_turnover = float(daily_to.mean()) if len(daily_to) else 0.0

        # 5. 计算覆盖率（P1修复: 只统计OOS段）- 完全向量化实现
        sig_prev = signals[:-1]  # (T-1, N) - 前一天的信号
        ret_today = returns[1:]  # (T-1, N) - 当天的收益

        # 找出OOS段（有信号的日期）
        has_signal = ~np.all(np.isnan(sig_prev), axis=1)  # (T-1,)
        total_oos_days = int(np.sum(has_signal))

        if total_oos_days > 0:
            # 计算每天有多少有效数据点
            valid_mask = ~(np.isnan(sig_prev) | np.isnan(ret_today))  # (T-1, N)
            n_valid = np.sum(valid_mask, axis=1)  # (T-1,)

            # 可交易日：有信号且有效数据>=top_n
            tradable_mask = has_signal & (n_valid >= spec.top_n)
            tradable_days = int(np.sum(tradable_mask))
            coverage = float(tradable_days / total_oos_days)
        else:
            coverage = 0.0

        # 6. 组装结果
        rec = {
            "factors": "|".join(spec.factors),
            "n_factors": len(spec.factors),
            "tau": spec.tau,
            "top_n": spec.top_n,
            "z_threshold": (
                "" if spec.z_threshold is None else float(spec.z_threshold)
            ),
            "coverage": coverage,
            "avg_turnover": avg_turnover,
            **kpi,
            "score": selector._score(kpi, avg_turnover, coverage),
            "_key": spec.key(),
        }

        return rec, daily_ret

    @staticmethod
    def evaluate_chunk(
        chunk: List[StrategySpec],
        results_list: List,
        factors: np.ndarray,
        returns: np.ndarray,
        factor_names: List[str],
        dates: Optional[pd.DatetimeIndex] = None,
    ) -> List[Tuple[Dict, pd.Series]]:
        """评估一批策略（用于并行）"""
        results = []
        for spec in chunk:
            rec, daily_ret = WFOStrategyEvaluator.evaluate_single_strategy(
                spec, results_list, factors, returns, factor_names, dates
            )
            results.append((rec, daily_ret))
        return results
