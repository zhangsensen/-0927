"""
因子挖掘管道 | Factor Discovery Pipeline
================================================================================
Layer 3: 三种自动因子挖掘方法 + 统一 FDR 校正。

方法:
  3a. AlgebraicSearch — 两两代数组合 (+, -, ×, ÷, max, min)
  3b. WindowOptimizer — 窗口参数变体搜索
  3c. TransformSearch — 数学变换 (rank, log1p, sqrt, sign, abs)

统一 FDR: Benjamini-Hochberg 校正，控制假发现率 ≤ 5%。
"""

import logging
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from .quality import FactorQualityAnalyzer
from .registry import FactorEntry

logger = logging.getLogger(__name__)


class AlgebraicSearch:
    """
    代数搜索：对现有因子做两两组合

    算子: +, -, ×, ÷, max, min
    C(N,2) × 6 种候选
    """

    OPERATORS = {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "div": lambda a, b: a / b.replace(0, np.nan),
        "max": lambda a, b: np.maximum(a, b),
        "min": lambda a, b: np.minimum(a, b),
    }

    def __init__(self, ic_threshold: float = 0.02):
        self.ic_threshold = ic_threshold

    def search(
        self,
        factors_dict: Dict[str, pd.DataFrame],
        analyzer: FactorQualityAnalyzer,
    ) -> Tuple[List[FactorEntry], Dict[str, pd.DataFrame], List[float]]:
        """
        执行代数搜索。

        参数:
            factors_dict: {factor_name: DataFrame} 已有因子值
            analyzer: 质检分析器 (用于 quick_ic_screen)

        返回:
            (entries, new_factors_dict, p_values)
        """
        factor_names = sorted(factors_dict.keys())
        n = len(factor_names)
        logger.info("AlgebraicSearch: %d factors → %d pairs × 6 ops = %d candidates",
                     n, n * (n - 1) // 2, n * (n - 1) // 2 * 6)

        entries = []
        new_factors = {}
        p_values = []

        for f1, f2 in combinations(factor_names, 2):
            df1 = factors_dict[f1]
            df2 = factors_dict[f2]

            for op_name, op_fn in sorted(self.OPERATORS.items()):
                new_name = f"{f1}__{op_name}__{f2}"

                try:
                    new_df = op_fn(df1, df2)
                except Exception:
                    continue

                # 检查有效性: 至少 50% 非 NaN
                valid_rate = new_df.notna().mean().mean()
                if valid_rate < 0.5:
                    continue

                # 快速 IC 预筛
                mean_ic, p_val = analyzer.quick_ic_screen(new_name, new_df)

                if abs(mean_ic) < self.ic_threshold:
                    continue

                entry = FactorEntry(
                    name=new_name,
                    source="algebraic",
                    expression=f"{f1} {op_name} {f2}",
                    parent_factors=[f1, f2],
                    metadata={"operator": op_name, "mean_ic_quick": mean_ic},
                )
                entries.append(entry)
                new_factors[new_name] = new_df
                p_values.append(p_val)

        logger.info("AlgebraicSearch: %d candidates passed IC threshold (|IC| > %.3f)",
                     len(entries), self.ic_threshold)
        return entries, new_factors, p_values


class WindowOptimizer:
    """
    窗口优化：对基础因子类型测试不同窗口参数。

    基于因子元数据中的 window 字段，测试
    [5, 10, 20, 40, 60, 120] 中与默认窗口不同的变体。
    """

    CANDIDATE_WINDOWS = [5, 10, 20, 40, 60, 120]

    # 因子名称模板 → (因子基名, 默认窗口)
    # 只对窗口敏感的因子做变体
    WINDOW_FACTORS = {
        "MOM": ("MOM", 20),
        "SLOPE": ("SLOPE", 20),
        "RET_VOL": ("RET_VOL", 20),
        "VOL_RATIO": ("VOL_RATIO", 20),
        "PV_CORR": ("PV_CORR", 20),
        "SHARPE_RATIO": ("SHARPE_RATIO", 20),
        "TSMOM": ("TSMOM", 60),
        "BREAKOUT": ("BREAKOUT", 20),
    }

    def search(
        self,
        close: pd.DataFrame,
        volume: pd.DataFrame,
        analyzer: FactorQualityAnalyzer,
    ) -> Tuple[List[FactorEntry], Dict[str, pd.DataFrame], List[float]]:
        """
        窗口变体搜索。

        参数:
            close: 收盘价
            volume: 成交量
            analyzer: 质检分析器

        返回:
            (entries, new_factors_dict, p_values)
        """
        entries = []
        new_factors = {}
        p_values = []

        for base_name, (_, default_win) in sorted(self.WINDOW_FACTORS.items()):
            for win in self.CANDIDATE_WINDOWS:
                if win == default_win:
                    continue

                new_name = f"{base_name}_{win}D"
                new_df = self._compute_variant(base_name, close, volume, win)
                if new_df is None:
                    continue

                valid_rate = new_df.notna().mean().mean()
                if valid_rate < 0.5:
                    continue

                mean_ic, p_val = analyzer.quick_ic_screen(new_name, new_df)

                entry = FactorEntry(
                    name=new_name,
                    source="window_variant",
                    expression=f"{base_name}(window={win})",
                    parent_factors=[],
                    metadata={"base_factor": base_name, "window": win, "mean_ic_quick": mean_ic},
                )
                entries.append(entry)
                new_factors[new_name] = new_df
                p_values.append(p_val)

        logger.info("WindowOptimizer: %d window variants generated", len(entries))
        return entries, new_factors, p_values

    @staticmethod
    def _compute_variant(
        base_name: str,
        close: pd.DataFrame,
        volume: pd.DataFrame,
        window: int,
    ) -> Optional[pd.DataFrame]:
        """根据基础因子类型和窗口计算变体"""
        try:
            if base_name == "MOM":
                return close.pct_change(window)
            elif base_name == "SLOPE":
                # 线性回归斜率 (简化: 用 rolling corr with time index)
                t = pd.DataFrame(
                    np.arange(len(close)).reshape(-1, 1).repeat(close.shape[1], axis=1),
                    index=close.index,
                    columns=close.columns,
                    dtype=float,
                )
                return close.rolling(window).corr(t) * close.rolling(window).std() / t.rolling(window).std()
            elif base_name == "RET_VOL":
                return close.pct_change().rolling(window).std() * np.sqrt(252)
            elif base_name == "VOL_RATIO":
                vol_ma = volume.rolling(window).mean()
                vol_ma_long = volume.rolling(window * 3).mean()
                return vol_ma / vol_ma_long.replace(0, np.nan)
            elif base_name == "PV_CORR":
                returns = close.pct_change()
                return returns.rolling(window).corr(volume.pct_change())
            elif base_name == "SHARPE_RATIO":
                returns = close.pct_change()
                mu = returns.rolling(window).mean() * 252
                sigma = returns.rolling(window).std() * np.sqrt(252)
                return mu / sigma.replace(0, np.nan)
            elif base_name == "TSMOM":
                return close / close.shift(window) - 1
            elif base_name == "BREAKOUT":
                return (close - close.rolling(window).min()) / (
                    close.rolling(window).max() - close.rolling(window).min()
                ).replace(0, np.nan)
            else:
                return None
        except Exception as e:
            logger.debug("Failed to compute %s(window=%d): %s", base_name, window, e)
            return None


class TransformSearch:
    """
    变换搜索：对每个因子施加数学变换。

    变换: rank, log1p, sqrt, sign, abs
    """

    TRANSFORMS = {
        "rank": lambda df: df.rank(axis=1, pct=True),
        "log1p": lambda df: np.sign(df) * np.log1p(df.abs()),
        "sqrt": lambda df: np.sign(df) * np.sqrt(df.abs()),
        "sign": lambda df: np.sign(df),
        "abs": lambda df: df.abs(),
    }

    def search(
        self,
        factors_dict: Dict[str, pd.DataFrame],
        analyzer: FactorQualityAnalyzer,
    ) -> Tuple[List[FactorEntry], Dict[str, pd.DataFrame], List[float]]:
        """
        变换搜索。

        参数:
            factors_dict: {factor_name: DataFrame} 已有因子
            analyzer: 质检分析器

        返回:
            (entries, new_factors_dict, p_values)
        """
        entries = []
        new_factors = {}
        p_values = []

        for factor_name in sorted(factors_dict.keys()):
            df = factors_dict[factor_name]

            for t_name, t_fn in sorted(self.TRANSFORMS.items()):
                new_name = f"{factor_name}__{t_name}"

                try:
                    new_df = t_fn(df)
                except Exception:
                    continue

                valid_rate = new_df.notna().mean().mean()
                if valid_rate < 0.5:
                    continue

                mean_ic, p_val = analyzer.quick_ic_screen(new_name, new_df)

                entry = FactorEntry(
                    name=new_name,
                    source="transform",
                    expression=f"{t_name}({factor_name})",
                    parent_factors=[factor_name],
                    metadata={"transform": t_name, "mean_ic_quick": mean_ic},
                )
                entries.append(entry)
                new_factors[new_name] = new_df
                p_values.append(p_val)

        logger.info("TransformSearch: %d transform candidates generated", len(entries))
        return entries, new_factors, p_values


class FactorDiscoveryPipeline:
    """
    因子发现统一管道

    整合三种搜索方法 + BH-FDR 校正。
    """

    def __init__(
        self,
        analyzer: FactorQualityAnalyzer,
        close: pd.DataFrame,
        volume: pd.DataFrame,
        fdr_alpha: float = 0.05,
        algebraic_ic_threshold: float = 0.02,
    ):
        self.analyzer = analyzer
        self.close = close
        self.volume = volume
        self.fdr_alpha = fdr_alpha
        self.algebraic_search = AlgebraicSearch(ic_threshold=algebraic_ic_threshold)
        self.window_optimizer = WindowOptimizer()
        self.transform_search = TransformSearch()

    def run(
        self,
        factors_dict: Dict[str, pd.DataFrame],
        enable_algebraic: bool = True,
        enable_window: bool = True,
        enable_transform: bool = True,
    ) -> Tuple[List[FactorEntry], Dict[str, pd.DataFrame]]:
        """
        执行因子发现全流程。

        参数:
            factors_dict: 已有因子值 {name: DataFrame}
            enable_algebraic: 是否启用代数搜索
            enable_window: 是否启用窗口优化
            enable_transform: 是否启用变换搜索

        返回:
            (fdr_passed_entries, fdr_passed_factors_dict)
        """
        all_entries = []
        all_factors = {}
        all_p_values = []

        # 3a. 代数搜索
        if enable_algebraic:
            logger.info("=== Stage 3a: Algebraic Search ===")
            entries, factors, pvals = self.algebraic_search.search(
                factors_dict, self.analyzer
            )
            all_entries.extend(entries)
            all_factors.update(factors)
            all_p_values.extend(pvals)

        # 3b. 窗口优化
        if enable_window:
            logger.info("=== Stage 3b: Window Optimization ===")
            entries, factors, pvals = self.window_optimizer.search(
                self.close, self.volume, self.analyzer
            )
            all_entries.extend(entries)
            all_factors.update(factors)
            all_p_values.extend(pvals)

        # 3c. 变换搜索
        if enable_transform:
            logger.info("=== Stage 3c: Transform Search ===")
            entries, factors, pvals = self.transform_search.search(
                factors_dict, self.analyzer
            )
            all_entries.extend(entries)
            all_factors.update(factors)
            all_p_values.extend(pvals)

        logger.info("Total candidates before FDR: %d", len(all_entries))

        if not all_entries:
            return [], {}

        # 统一 BH-FDR 校正
        passed_entries, passed_factors = self._apply_fdr(
            all_entries, all_factors, all_p_values
        )

        logger.info("Passed FDR (alpha=%.2f): %d / %d",
                     self.fdr_alpha, len(passed_entries), len(all_entries))

        return passed_entries, passed_factors

    def _apply_fdr(
        self,
        entries: List[FactorEntry],
        factors: Dict[str, pd.DataFrame],
        p_values: List[float],
    ) -> Tuple[List[FactorEntry], Dict[str, pd.DataFrame]]:
        """Benjamini-Hochberg FDR 校正"""
        p_arr = np.array(p_values, dtype=float)

        # 处理 NaN p-values
        nan_mask = ~np.isfinite(p_arr)
        p_arr[nan_mask] = 1.0

        reject, _, _, _ = multipletests(p_arr, alpha=self.fdr_alpha, method="fdr_bh")

        passed_entries = []
        passed_factors = {}

        for i, (entry, rejected) in enumerate(zip(entries, reject)):
            if rejected:
                passed_entries.append(entry)
                if entry.name in factors:
                    passed_factors[entry.name] = factors[entry.name]

        return passed_entries, passed_factors
