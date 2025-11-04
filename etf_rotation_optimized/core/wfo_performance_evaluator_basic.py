"""
WFO 性能评估器（基础版）

将 WFO 窗口级别的因子加权信号，串接为连续的 OOS 信号序列，
在严格 T+1 约束下构建 Top-N 等权持仓，计算净值与完整 KPI，
并将结果落盘到 Pipeline.wfo_dir。

注意:
- 严格避免前视：使用信号[t-1] 来决定 t 日持仓。
- 窗口拼接：按各窗口 OOS 段拼接，不重复、不间断（若有间隙则空仓）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class WFOPerformanceOutputs:
    daily_returns: pd.Series
    equity_curve: pd.Series
    kpis: Dict[str, float]


class WfoPerformanceEvaluator:
    """WFO性能评估器（基础版，仅T+1 Top-N等权）"""

    def __init__(self, top_n: int = 6):
        self.top_n = int(top_n)

    @staticmethod
    def _stitch_oos_signals(
        results_list,
        factors: np.ndarray,  # (T, N, K)
        returns: np.ndarray,  # (T, N)
        factor_names: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        依据各窗口选中因子与权重，生成每个窗口的 OOS 加权信号并按时间拼接。

        返回:
            stitched_signals: (T, N) OOS 区间内有效，其它位置为 NaN
            returns:          (T, N) 原样返回以对齐
        """
        T, N, K = factors.shape
        name_to_idx = {n: i for i, n in enumerate(factor_names)}
        stitched = np.full((T, N), np.nan, dtype=float)

        for r in results_list:
            # 取窗口 OOS 段
            s, e = int(r.oos_start), int(r.oos_end)
            if s >= e or s < 0 or e > T:
                continue

            # 选中因子索引 & 权重
            idxs = [name_to_idx[f] for f in r.selected_factors if f in name_to_idx]
            if not idxs:
                continue
            w = np.array(
                [r.factor_weights[f] for f in r.selected_factors if f in name_to_idx],
                dtype=float,
            )
            if w.size == 0:
                continue

            # OOS 加权信号: factors[s:e, :, idxs] -> (e-s, N, F)
            oos_fac = factors[s:e, :, idxs]
            # tensordot over factor dim -> (e-s, N)
            oos_sig = np.tensordot(oos_fac, w, axes=([2], [0]))
            stitched[s:e, :] = oos_sig

        return stitched, returns

    def _topn_tplus1_returns(
        self, signals: np.ndarray, returns: np.ndarray
    ) -> pd.Series:
        """
        在严格 T+1 下，用 t-1 日信号选择 t 日 Top-N，等权持有，计算组合日收益。
        空仓日记为 0。
        """
        T, N = returns.shape
        daily_ret = np.zeros(T, dtype=float)

        for t in range(1, T):  # 从第1天开始，因为需要 t-1 信号
            sig_prev = signals[t - 1]
            ret_today = returns[t]

            # 仅使用同时非 NaN 的资产
            mask = ~(np.isnan(sig_prev) | np.isnan(ret_today))
            if not np.any(mask):
                daily_ret[t] = 0.0
                continue

            # 选 Top-N
            valid_idx = np.where(mask)[0]
            ranked = valid_idx[np.argsort(sig_prev[mask])[::-1]]
            topk = ranked[: self.top_n]
            if topk.size == 0:
                daily_ret[t] = 0.0
                continue

            # 等权平均收益
            daily_ret[t] = float(np.nanmean(ret_today[topk]))

        # 将第0天收益置零（无持仓）
        daily_ret[0] = 0.0
        return pd.Series(daily_ret)

    @staticmethod
    def _compute_kpis(daily_returns: pd.Series) -> Dict[str, float]:
        r = daily_returns.fillna(0.0).values
        if r.size < 2:
            return {
                k: 0.0
                for k in [
                    "total_return",
                    "annual_return",
                    "annual_volatility",
                    "sharpe_ratio",
                    "max_drawdown",
                    "win_rate",
                ]
            }

        # 累计净值与回撤
        equity = np.cumprod(1 + r)
        running_max = np.maximum.accumulate(equity)
        dd = (equity - running_max) / (running_max + 1e-12)

        total_return = float(equity[-1] - 1.0)
        ann_ret = float((equity[-1]) ** (252.0 / max(1, len(r))) - 1.0)
        ann_vol = float(np.std(r) * np.sqrt(252.0))
        sharpe = float(ann_ret / ann_vol) if ann_vol > 1e-12 else 0.0
        win = float(np.mean(r > 0))

        return {
            "total_return": total_return,
            "annual_return": ann_ret,
            "annual_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": float(np.min(dd)) if dd.size else 0.0,
            "win_rate": win,
        }

    def evaluate_and_save(
        self,
        results_list,
        factors: np.ndarray,
        returns: np.ndarray,
        factor_names: List[str],
        dates: Optional[pd.DatetimeIndex],
        out_dir,
    ) -> WFOPerformanceOutputs:
        # 生成拼接的 OOS 加权信号
        signals, aligned_returns = self._stitch_oos_signals(
            results_list, factors, returns, factor_names
        )

        # 计算严格 T+1 Top-N 等权组合收益
        daily_returns = self._topn_tplus1_returns(signals, aligned_returns)
        equity_curve = (1.0 + daily_returns.fillna(0.0)).cumprod()

        # 对齐索引
        if dates is not None and len(dates) == len(daily_returns):
            daily_returns.index = dates
            equity_curve.index = dates

        # KPI
        kpis = self._compute_kpis(daily_returns)

        # 落盘
        pd.DataFrame({"return": daily_returns}).to_csv(
            out_dir / "wfo_returns_event_driven.csv"
        )
        pd.DataFrame({"equity": equity_curve}).to_csv(
            out_dir / "wfo_equity_event_driven.csv"
        )
        pd.DataFrame([kpis]).to_csv(out_dir / "wfo_kpi_event_driven.csv", index=False)

        return WFOPerformanceOutputs(
            daily_returns=daily_returns, equity_curve=equity_curve, kpis=kpis
        )
