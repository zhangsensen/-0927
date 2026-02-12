"""
因子质检分析器 | Factor Quality Analyzer
================================================================================
Layer 2: 10 维度单因子评估，输出 FactorQualityReport。

维度:
  1. IC 分析 (mean_ic, std_ic, IC_IR, t_stat, p_value, hit_rate)
  2. 单调性 (三等分组收益, monotonicity_score)
  3. 稳定性 (滚动IC 120天窗口均值/正IC率)
  4. IC 衰减 (ic_by_horizon {1,3,5,10,20}d, 最优持有期)
  5. 换手率 (rank_autocorrelation lag=FREQ)
  6. NaN 覆盖 (nan_rate, 截面均值/最小有效数)
  7. 市场环境 (ic_bull, ic_bear, ic_sideways)
  8. A股/QDII拆分 (ic_ashare, ic_qdii)
  9. 方向一致性 (IC符号是否匹配metadata.direction)
  10. 综合评分 (quality_score, passed)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# 常量
DEFAULT_FREQ = 5  # 默认改为生产级频率，保持向后兼容可通过参数覆盖
IC_HORIZONS = [1, 3, 5, 10, 20]
QDII_CODES = {"513100", "513500", "159920", "513050", "513130"}
ROLLING_IC_WINDOW = 120
MIN_VALID_OBS = 5
PASS_THRESHOLD = 2.0
MAX_NAN_RATE = 0.30


@dataclass
class FactorQualityReport:
    """单因子质检报告"""

    factor_name: str

    # IC 分析
    mean_ic: float = 0.0
    std_ic: float = 0.0
    ic_ir: float = 0.0
    t_stat: float = 0.0
    p_value: float = 1.0
    hit_rate: float = 0.0
    n_obs: int = 0

    # 单调性
    monotonicity_score: float = 0.0
    tercile_returns: List[float] = field(default_factory=list)

    # 稳定性
    rolling_ic_mean: float = 0.0
    rolling_ic_positive_rate: float = 0.0

    # IC 衰减
    ic_by_horizon: Dict[int, float] = field(default_factory=dict)
    best_horizon: int = 3

    # 换手率
    rank_autocorrelation: float = 0.0

    # NaN 覆盖
    nan_rate: float = 0.0
    min_valid_per_date: int = 0
    mean_valid_per_date: float = 0.0

    # 市场环境
    ic_bull: float = 0.0
    ic_bear: float = 0.0
    ic_sideways: float = 0.0

    # A股/QDII 拆分
    ic_ashare: float = 0.0
    ic_qdii: float = 0.0

    # 方向一致性
    direction_consistent: bool = False

    # 综合
    quality_score: float = 0.0
    passed: bool = False

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, (np.floating, np.integer)):
                d[k] = float(v)
            elif isinstance(v, dict):
                d[k] = {
                    str(kk): float(vv)
                    if isinstance(vv, (np.floating, np.integer))
                    else vv
                    for kk, vv in v.items()
                }
            elif isinstance(v, list):
                d[k] = [
                    float(x) if isinstance(x, (np.floating, np.integer)) else x
                    for x in v
                ]
            else:
                d[k] = v
        return d


def spearman_ic_series(factor_df: pd.DataFrame, return_df: pd.DataFrame) -> pd.Series:
    """
    逐日计算横截面 Spearman IC（向量化 rank + corrwith）。

    提取自 factor_alpha_analysis.py，改用 pandas 向量化实现。
    """
    common_idx = factor_df.index.intersection(return_df.index)
    common_cols = factor_df.columns.intersection(return_df.columns)
    f = factor_df.loc[common_idx, common_cols]
    r = return_df.loc[common_idx, common_cols]

    # 截面 rank (axis=1)
    f_rank = f.rank(axis=1)
    r_rank = r.rank(axis=1)

    # 有效个数掩码
    valid = f.notna() & r.notna()
    n_valid = valid.sum(axis=1)
    date_mask = n_valid >= MIN_VALID_OBS

    f_rank = f_rank[date_mask]
    r_rank = r_rank[date_mask]

    # 逐行 Pearson(rank, rank) = Spearman，向量化
    # demean
    f_dm = f_rank.sub(f_rank.mean(axis=1), axis=0)
    r_dm = r_rank.sub(r_rank.mean(axis=1), axis=0)

    # covariance / (std_f * std_r)
    cov = (f_dm * r_dm).sum(axis=1)
    std_f = (f_dm**2).sum(axis=1).pow(0.5)
    std_r = (r_dm**2).sum(axis=1).pow(0.5)
    denom = std_f * std_r
    denom = denom.replace(0, np.nan)

    ic = cov / denom
    ic = ic.dropna()
    ic.name = "IC"
    return ic


def compute_forward_returns(close: pd.DataFrame, periods: int) -> pd.DataFrame:
    """前瞻 N 日收益率 (用于分析)"""
    return close.shift(-periods) / close - 1


class FactorQualityAnalyzer:
    """
    因子质检分析器

    对单因子进行 10 维度评估，输出 FactorQualityReport。
    支持完整模式和快速预筛模式。
    """

    def __init__(
        self,
        close: pd.DataFrame,
        regime_series: Optional[pd.Series] = None,
        freq: int = DEFAULT_FREQ,
    ):
        """
        参数:
            close: 收盘价 DataFrame (index=date, columns=symbols)
            regime_series: 市场环境序列 (index=date, values='bull'|'bear'|'sideways')
            freq: 调仓频率（天数）
        """
        self.close = close
        self.freq = freq
        self.regime_series = regime_series

        # 预计算前瞻收益
        self.fwd_ret = compute_forward_returns(close, freq)

        # 分池
        all_cols = close.columns.tolist()
        self.qdii_cols = [c for c in all_cols if c in QDII_CODES]
        self.ashare_cols = [c for c in all_cols if c not in QDII_CODES]

    def analyze(
        self,
        factor_name: str,
        factor_df: pd.DataFrame,
        direction: str = "high_is_good",
        production_ready: bool = True,
    ) -> FactorQualityReport:
        """
        完整 10 维度质检。

        参数:
            factor_name: 因子名称
            factor_df: 因子值 DataFrame (index=date, columns=symbols)
            direction: 'high_is_good' | 'low_is_good' | 'neutral'
            production_ready: 因子元数据中的 production_ready 标志

        返回:
            FactorQualityReport
        """
        report = FactorQualityReport(factor_name=factor_name)

        # --- 1. IC 分析 ---
        ic_s = spearman_ic_series(factor_df, self.fwd_ret)
        if len(ic_s) < 30:
            logger.warning(
                "Factor %s: IC series too short (%d), skipping", factor_name, len(ic_s)
            )
            return report

        report.mean_ic = float(ic_s.mean())
        report.std_ic = float(ic_s.std())
        report.ic_ir = report.mean_ic / report.std_ic if report.std_ic > 1e-10 else 0.0
        report.hit_rate = float((ic_s > 0).mean())
        report.n_obs = len(ic_s)
        report.t_stat = (
            report.mean_ic / (report.std_ic / np.sqrt(len(ic_s)))
            if report.std_ic > 1e-10
            else 0.0
        )
        report.p_value = float(
            2 * (1 - stats.t.cdf(abs(report.t_stat), df=len(ic_s) - 1))
        )

        # --- 2. 单调性 (三等分组) ---
        report.monotonicity_score, report.tercile_returns = self._monotonicity(
            factor_df
        )

        # --- 3. 稳定性 (滚动 IC) ---
        if len(ic_s) >= ROLLING_IC_WINDOW:
            rolling_ic_mean = ic_s.rolling(ROLLING_IC_WINDOW).mean().dropna()
            if len(rolling_ic_mean) > 0:
                report.rolling_ic_mean = float(rolling_ic_mean.mean())
                report.rolling_ic_positive_rate = float((rolling_ic_mean > 0).mean())

        # --- 4. IC 衰减 ---
        for h in IC_HORIZONS:
            fwd_h = compute_forward_returns(self.close, h)
            ic_h = spearman_ic_series(factor_df, fwd_h)
            report.ic_by_horizon[h] = float(ic_h.mean()) if len(ic_h) > 0 else 0.0
        if report.ic_by_horizon:
            report.best_horizon = max(
                report.ic_by_horizon, key=lambda k: abs(report.ic_by_horizon[k])
            )

        # --- 5. 换手率 (rank autocorrelation) ---
        report.rank_autocorrelation = self._rank_autocorrelation(factor_df)

        # --- 6. NaN 覆盖 ---
        valid_counts = factor_df.notna().sum(axis=1)
        total_cols = factor_df.shape[1]
        report.nan_rate = (
            float(1 - valid_counts.mean() / total_cols) if total_cols > 0 else 1.0
        )
        report.min_valid_per_date = (
            int(valid_counts.min()) if len(valid_counts) > 0 else 0
        )
        report.mean_valid_per_date = (
            float(valid_counts.mean()) if len(valid_counts) > 0 else 0.0
        )

        # --- 7. 市场环境 IC ---
        if self.regime_series is not None:
            report.ic_bull, report.ic_bear, report.ic_sideways = self._regime_ic(
                factor_df, ic_s
            )

        # --- 8. A股/QDII 拆分 ---
        if self.qdii_cols:
            ic_qdii = spearman_ic_series(
                factor_df[self.qdii_cols], self.fwd_ret[self.qdii_cols]
            )
            report.ic_qdii = float(ic_qdii.mean()) if len(ic_qdii) > 0 else 0.0
        if self.ashare_cols:
            ic_ashare = spearman_ic_series(
                factor_df[self.ashare_cols], self.fwd_ret[self.ashare_cols]
            )
            report.ic_ashare = float(ic_ashare.mean()) if len(ic_ashare) > 0 else 0.0

        # --- 9. 方向一致性 ---
        if direction == "high_is_good":
            report.direction_consistent = report.mean_ic > 0
        elif direction == "low_is_good":
            report.direction_consistent = report.mean_ic < 0
        else:  # neutral
            report.direction_consistent = True

        # --- 10. 综合评分 ---
        report.quality_score = self._compute_score(report, production_ready)
        report.passed = (
            report.quality_score >= PASS_THRESHOLD and report.nan_rate <= MAX_NAN_RATE
        )

        return report

    def quick_ic_screen(
        self, factor_name: str, factor_df: pd.DataFrame
    ) -> Tuple[float, float]:
        """
        快速 IC 预筛 — 仅计算 mean_ic 和 p_value。

        用于挖掘阶段大量候选的快速过滤。

        返回:
            (mean_ic, p_value)
        """
        ic_s = spearman_ic_series(factor_df, self.fwd_ret)
        if len(ic_s) < 30:
            return 0.0, 1.0

        mean_ic = float(ic_s.mean())
        std_ic = float(ic_s.std())
        if std_ic < 1e-10:
            return mean_ic, 1.0

        t_stat = mean_ic / (std_ic / np.sqrt(len(ic_s)))
        p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=len(ic_s) - 1)))
        return mean_ic, p_value

    # ── 内部方法 ──────────────────────────────────────────────

    def _monotonicity(self, factor_df: pd.DataFrame) -> Tuple[float, List[float]]:
        """
        三等分组单调性检验（向量化）。

        用截面 rank percentile 分 3 组，计算各组平均前瞻收益的时序均值。
        单调性 = |rank_corr([1,2,3], group_returns)|。
        """
        fwd = self.fwd_ret
        common_idx = factor_df.index.intersection(fwd.index)
        common_cols = factor_df.columns.intersection(fwd.columns)

        f_aligned = factor_df.loc[common_idx, common_cols]
        r_aligned = fwd.loc[common_idx, common_cols]

        # 截面 rank percentile: [0, 1]
        rank_pct = f_aligned.rank(axis=1, pct=True)

        # 有效掩码: 因子和收益均非 NaN
        valid = f_aligned.notna() & r_aligned.notna()
        valid_count = valid.sum(axis=1)
        date_mask = valid_count >= 6  # 至少每组 2 只

        r_vals = r_aligned[date_mask]
        rk_vals = rank_pct[date_mask]

        # 分组: bottom [0, 1/3), mid [1/3, 2/3), top [2/3, 1]
        tercile_means = []
        for lo, hi in [(0.0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.01)]:
            grp_mask = (rk_vals >= lo) & (rk_vals < hi) & r_vals.notna()
            grp_ret = r_vals.where(grp_mask)
            daily_mean = grp_ret.mean(axis=1)
            tercile_means.append(
                float(daily_mean.mean()) if daily_mean.notna().any() else 0.0
            )

        if any(x != 0 for x in tercile_means):
            corr_val, _ = stats.spearmanr([1, 2, 3], tercile_means)
            mono = abs(float(corr_val)) if np.isfinite(corr_val) else 0.0
        else:
            mono = 0.0

        return mono, tercile_means

    def _rank_autocorrelation(self, factor_df: pd.DataFrame) -> float:
        """因子排名自相关 (lag=freq)，向量化"""
        ranked = factor_df.rank(axis=1, pct=True)
        lagged = ranked.shift(self.freq)

        # 展平并计算整体 Spearman 相关
        r_flat = ranked.values.ravel()
        l_flat = lagged.values.ravel()
        mask = np.isfinite(r_flat) & np.isfinite(l_flat)

        if mask.sum() < 100:
            return 0.0

        corr_val, _ = stats.spearmanr(r_flat[mask], l_flat[mask])
        return float(corr_val) if np.isfinite(corr_val) else 0.0

    def _regime_ic(
        self, factor_df: pd.DataFrame, ic_s: pd.Series
    ) -> Tuple[float, float, float]:
        """按市场环境拆分 IC"""
        regime = self.regime_series
        common_idx = ic_s.index.intersection(regime.index)
        if len(common_idx) == 0:
            return 0.0, 0.0, 0.0

        ic_aligned = ic_s.loc[common_idx]
        regime_aligned = regime.loc[common_idx]

        ic_bull = (
            float(ic_aligned[regime_aligned == "bull"].mean())
            if (regime_aligned == "bull").any()
            else 0.0
        )
        ic_bear = (
            float(ic_aligned[regime_aligned == "bear"].mean())
            if (regime_aligned == "bear").any()
            else 0.0
        )
        ic_sideways = (
            float(ic_aligned[regime_aligned == "sideways"].mean())
            if (regime_aligned == "sideways").any()
            else 0.0
        )

        return ic_bull, ic_bear, ic_sideways

    @staticmethod
    def _compute_score(report: FactorQualityReport, production_ready: bool) -> float:
        """
        综合评分公式：

        +2.0  if IC p < 0.01  (or +1.0 if p < 0.05)
        +1.0  if 单调性 ≥ 0.8
        +1.0  if 滚动IC正率 > 55%
        +0.5  if rank_autocorr > 0.7 (低换手)
        +1.0  if IC方向与metadata一致
        -3.0  if production_ready=False
        """
        score = 0.0

        if report.p_value < 0.01:
            score += 2.0
        elif report.p_value < 0.05:
            score += 1.0

        if report.monotonicity_score >= 0.8:
            score += 1.0

        if report.rolling_ic_positive_rate > 0.55:
            score += 1.0

        if report.rank_autocorrelation > 0.7:
            score += 0.5

        if report.direction_consistent:
            score += 1.0

        if not production_ready:
            score -= 3.0

        return score
