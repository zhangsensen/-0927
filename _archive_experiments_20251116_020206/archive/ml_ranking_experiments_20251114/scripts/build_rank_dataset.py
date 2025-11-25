#!/usr/bin/env python3
"""
构建排序校准训练数据集：
1. 扫描 experiments/results/run_* 目录，读取最新的 all_combos.parquet。
2. 匹配 results_combo_wfo 下对应 run_ts 的全量真实回测 CSV。
3. 对齐组合，提取/扩展 WFO 特征 + 回测指标，输出 data/calibrator_dataset.parquet。

使用示例：
    python scripts/build_rank_dataset.py
    python scripts/build_rank_dataset.py --run-ts 20251111_022309 --output data/rank_ds.parquet
"""

from __future__ import annotations

import argparse
import ast
import math
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import linregress  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    linregress = None

COMMISSION_RATE = 0.000005


@dataclass
class RunResource:
    run_dir: Path
    backtest_csvs: List[Path]

    @property
    def run_ts(self) -> str:
        return self.run_dir.name.replace("run_", "")


def list_run_dirs(base: Path) -> List[Path]:
    if not base.exists():
        return []
    runs = [d for d in base.glob("run_*") if d.is_dir()]
    return sorted({d.resolve() for d in runs}, reverse=True)


BACKTEST_PATTERNS = [
    "top*_profit_backtest_slip0bps_*.csv",
    "top*_profit_backtest_*.csv",
    "top*_backtest_slip0bps_*.csv",
    "top*_backtest_*.csv",
]


def find_backtest_csvs(run_ts: str, rb_base: Path) -> List[Path]:
    if not rb_base.exists():
        return []
    candidates: Dict[Path, int] = {}
    for sub in rb_base.glob(f"{run_ts}_*"):
        if not sub.is_dir():
            continue
        for pattern in BACKTEST_PATTERNS:
            for csv in sub.glob(pattern):
                name = csv.name
                try:
                    top_str = name.split("_")[0].replace("top", "")
                    top_n = int(top_str)
                except ValueError:
                    top_n = -1
                path = csv.resolve()
                # 使用较大的 top_n 覆盖重复的文件
                if path not in candidates or top_n > candidates[path]:
                    candidates[path] = top_n
    if not candidates:
        return []
    ordered = sorted(candidates.items(), key=lambda x: (x[1], str(x[0])), reverse=True)
    return [path for path, _ in ordered]


def safe_literal_list(val, expected: str) -> List[float]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if isinstance(val, (list, tuple, np.ndarray)):
        seq = list(val)
    else:
        try:
            seq = ast.literal_eval(str(val))
        except (ValueError, SyntaxError):
            return []
    if not isinstance(seq, (list, tuple)):
        return []
    if expected == "float":
        return [float(x) for x in seq if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if expected == "int":
        return [int(x) for x in seq if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return []


def compute_list_features(values: Sequence[float]) -> dict:
    if not values:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "median": np.nan,
            "p10": np.nan,
            "p90": np.nan,
            "first": np.nan,
            "last": np.nan,
            "trend": np.nan,
            "skew": np.nan,
            "kurt": np.nan,
            "last3_mean": np.nan,
            "last3_std": np.nan,
            "last3_trend": np.nan,
            "ewm_alpha0_3_last": np.nan,
            "count": 0,
        }
    arr = np.asarray(values, dtype=float)
    ser = pd.Series(arr)
    last3 = arr[-3:] if arr.size >= 3 else arr
    stats = {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "first": float(arr[0]),
        "last": float(arr[-1]),
        "trend": float(arr[-1] - arr[0]),
        "skew": float(ser.skew()) if len(arr) > 2 else 0.0,
        "kurt": float(ser.kurt()) if len(arr) > 3 else 0.0,
        "last3_mean": float(np.mean(last3)),
        "last3_std": float(np.std(last3, ddof=1)) if len(last3) > 1 else 0.0,
        "last3_trend": float(last3[-1] - last3[0]) if len(last3) > 1 else 0.0,
        "ewm_alpha0_3_last": float(ser.ewm(alpha=0.3).mean().iloc[-1]),
        "count": int(len(arr)),
    }
    return stats


def _monotonicity(arr: np.ndarray) -> float:
    if arr.size <= 1:
        return 0.0
    t = np.arange(arr.size, dtype=float)
    denom = np.std(arr) * np.std(t)
    if denom == 0:
        return 0.0
    return float(np.corrcoef(arr, t)[0, 1])


def _sign_reversals(arr: np.ndarray) -> int:
    if arr.size <= 1:
        return 0
    signs = np.sign(arr)
    changes = np.diff(signs)
    return int(np.sum(changes != 0))


def _positive_streak(arr: np.ndarray) -> int:
    best = 0
    current = 0
    for val in arr:
        if val > 0:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def _trend_metrics(arr: np.ndarray) -> Tuple[float, float]:
    if arr.size <= 1 or np.allclose(arr, arr[0]):
        return 0.0, 1.0
    t = np.arange(arr.size, dtype=float)
    if linregress is None:
        corr = _monotonicity(arr)
        return corr, float("nan")
    res = linregress(t, arr)
    return float(res.rvalue), float(res.pvalue)


def _acceleration(arr: np.ndarray) -> float:
    if arr.size <= 2:
        return 0.0
    second_diff = np.diff(arr, n=2)
    if second_diff.size == 0:
        return 0.0
    return float(np.mean(second_diff))


def _ratio_safe(num: float, denom: float) -> float:
    if math.isclose(denom, 0.0):
        denom = 1e-9
    return float(num / denom)


def _last_vs_first_ratio(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    first = np.mean(arr[: min(3, arr.size)])
    last = np.mean(arr[-min(3, arr.size) :])
    return _ratio_safe(last, first if not math.isclose(first, 0.0) else 1e-9)


def _coeff_variation(arr: np.ndarray) -> float:
    if arr.size <= 1:
        return 0.0
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    if math.isclose(mean, 0.0):
        mean = 1e-9
    return std / abs(mean)


def enrich_wfo_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    ic_lists = out.get("oos_ic_list")
    if ic_lists is not None:
        ic_values = ic_lists.apply(lambda x: safe_literal_list(x, "float"))
        feats = ic_values.apply(compute_list_features)
        keys = sorted({k for d in feats for k in d.keys()})
        for key in keys:
            out[f"oos_ic_{key}"] = feats.apply(lambda d: d.get(key, np.nan))
        out["oos_ic_range"] = out["oos_ic_max"] - out["oos_ic_min"]
        out["oos_ic_cv"] = out["oos_ic_std"] / (out["oos_ic_mean"].replace(0, np.nan))

        def _extra_ic_metrics(values: List[float]) -> Dict[str, float]:
            if not values:
                return {
                    "monotonicity": 0.0,
                    "reversals": 0.0,
                    "positive_streak": 0.0,
                    "trend_strength": 0.0,
                    "trend_pvalue": float("nan"),
                    "acceleration": 0.0,
                    "last_vs_first_ratio": 0.0,
                }
            arr = np.asarray(values, dtype=float)
            mono = _monotonicity(arr)
            rev = _sign_reversals(arr)
            streak = _positive_streak(arr)
            trend_r, trend_p = _trend_metrics(arr)
            accel = _acceleration(arr)
            ratio = _last_vs_first_ratio(arr)
            return {
                "monotonicity": mono,
                "reversals": float(rev),
                "positive_streak": float(streak),
                "trend_strength": float(trend_r),
                "trend_pvalue": float(trend_p),
                "acceleration": accel,
                "last_vs_first_ratio": ratio,
            }

        ic_extra = ic_values.apply(_extra_ic_metrics)
        for key in ["monotonicity", "reversals", "positive_streak", "trend_strength", "trend_pvalue", "acceleration", "last_vs_first_ratio"]:
            out[f"oos_ic_{key}"] = ic_extra.apply(lambda d: d.get(key, np.nan))

    ir_lists = out.get("oos_ir_list")
    if ir_lists is not None:
        ir_values = ir_lists.apply(lambda x: safe_literal_list(x, "float"))
        feats = ir_values.apply(compute_list_features)
        keys = sorted({k for d in feats for k in d.keys()})
        for key in keys:
            out[f"oos_ir_{key}"] = feats.apply(lambda d: d.get(key, np.nan))
        def _ir_stability(values: List[float]) -> float:
            if not values:
                return 0.0
            arr = np.asarray(values, dtype=float)
            if arr.size <= 1:
                return 0.0
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1))
            if math.isclose(std, 0.0):
                std = 1e-9
            return abs(mean) / (std + 1e-9)

        out["oos_ir_stability_score"] = ir_values.apply(_ir_stability)

    pos_lists = out.get("positive_rate_list")
    if pos_lists is not None:
        feats = pos_lists.apply(lambda x: compute_list_features(safe_literal_list(x, "float")))
        keys = sorted({k for d in feats for k in d.keys()})
        for key in keys:
            out[f"pos_rate_{key}"] = feats.apply(lambda d: d.get(key, np.nan))

    freq_lists = out.get("best_freq_list")
    if freq_lists is not None:
        stats = freq_lists.apply(lambda x: safe_literal_list(x, "int"))
        out["best_freq_mode"] = stats.apply(lambda lst: Counter(lst).most_common(1)[0][0] if lst else np.nan)
        out["best_freq_unique"] = stats.apply(lambda lst: len(set(lst)))
        out["best_freq_var"] = stats.apply(lambda lst: float(np.var(lst, ddof=1)) if len(lst) > 1 else 0.0)

    out["ic_over_std"] = out["mean_oos_ic"] / (out["oos_ic_std"].replace(0, np.nan))
    return out


def add_standardized_targets(df: pd.DataFrame) -> pd.DataFrame:
    if "run_ts" not in df.columns:
        return df

    out = df.copy()
    metric_cols = [
        ("annual_ret_net", "annual_ret"),
        ("sharpe_net", "sharpe"),
        ("calmar_net", None),
    ]
    group = out.groupby("run_ts")
    for net_col, fallback in metric_cols:
        if net_col not in out.columns:
            if fallback and fallback in out.columns:
                out[net_col] = out[fallback]
            else:
                continue
        mean = group[net_col].transform("mean")
        std = group[net_col].transform("std").replace(0, np.nan)
        z_col = f"{net_col}_z"
        pct_col = f"{net_col}_pct"
        decile_col = f"{net_col}_decile"
        out[z_col] = (out[net_col] - mean) / std
        out[z_col] = out[z_col].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        out[pct_col] = group[net_col].rank(pct=True, method="average")
        out[decile_col] = np.clip((out[pct_col] * 10).astype(int), 0, 10)
    return out


def augment_backtest(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "annual_ret_net" not in out.columns and "annual_ret" in out.columns:
        out["annual_ret_net"] = out["annual_ret"]
    if "total_ret_net" not in out.columns and "total_ret" in out.columns:
        out["total_ret_net"] = out["total_ret"]
    if "sharpe_net" not in out.columns and "sharpe" in out.columns:
        out["sharpe_net"] = out["sharpe"]
    if "max_dd_net" not in out.columns and "max_dd" in out.columns:
        out["max_dd_net"] = out["max_dd"]
    if "max_dd_net" in out.columns:
        out["calmar_net"] = out.apply(
            lambda r: r["annual_ret_net"] / abs(r["max_dd_net"]) if r["max_dd_net"] < -1e-8 else np.nan,
            axis=1,
        )
    if {"annual_ret_net", "vol"}.issubset(out.columns):
        out["return_vol_ratio"] = out["annual_ret_net"] / out["vol"].replace(0, np.nan)
    if "avg_turnover" in out.columns:
        out["ret_turnover_ratio"] = out["annual_ret_net"] / (out["avg_turnover"] + 1e-6)
    return out


def add_real_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def safe_div(num: pd.Series, denom: pd.Series, default: float = np.nan) -> pd.Series:
        denom_safe = denom.replace(0, np.nan)
        res = num / denom_safe
        return res.replace([np.inf, -np.inf], default)

    if {"annual_ret", "annual_ret_net"}.issubset(out.columns):
        gross = out["annual_ret"].astype(float)
        net = out["annual_ret_net"].astype(float)
        delta = gross - net
        denom = gross.replace(0, np.nan)
        out["ret_net_minus_gross"] = net - gross
        out["ret_net_over_gross"] = safe_div(net, gross, default=0.0)
        out["ret_cost_ratio"] = safe_div(delta.abs(), gross.abs() + 1e-6, default=0.0)
        out["cost_drag"] = safe_div(delta, denom, default=0.0)
        out["breakeven_turnover_est"] = net.abs() / (COMMISSION_RATE + 1e-9)

    if "avg_turnover" in out.columns:
        avg_turnover = out["avg_turnover"].astype(float).abs()
        out["avg_turnover_rate"] = avg_turnover
        turnover_std = out.get("turnover_std", pd.Series(0.0, index=out.index)).astype(float)
        out["turnover_stability"] = safe_div(turnover_std, avg_turnover + 1e-6, default=0.0)

    if "best_rebalance_freq" in out.columns:
        freq_map = {5: 0.2, 10: 0.4, 20: 0.6, 60: 0.8, 120: 1.0}
        out["rebalance_frequency_score"] = out["best_rebalance_freq"].map(freq_map).fillna(0.5)

    if {"sharpe", "sharpe_net"}.issubset(out.columns):
        out["sharpe_net_minus_gross"] = out["sharpe_net"] - out["sharpe"]

    if {"max_dd", "max_dd_net"}.issubset(out.columns):
        max_dd = out["max_dd"].astype(float)
        max_dd_net = out["max_dd_net"].astype(float)
        out["max_dd_net_minus_gross"] = max_dd_net - max_dd
        out["dd_ratio_net_over_gross"] = safe_div(max_dd_net.abs(), max_dd.abs() + 1e-6, default=0.0)

    if {"annual_ret_net", "avg_turnover"}.issubset(out.columns):
        out["ret_turnover_eff"] = safe_div(out["annual_ret_net"], out["avg_turnover"].abs() + 1e-6, default=0.0)

    if {"annual_ret_net", "n_rebalance"}.issubset(out.columns):
        out["ret_rebalance_eff"] = safe_div(out["annual_ret_net"], out["n_rebalance"].replace(0, np.nan), default=0.0)

    if {"avg_win", "avg_loss"}.issubset(out.columns):
        out["win_loss_ratio"] = safe_div(out["avg_win"], out["avg_loss"].abs() + 1e-6, default=0.0)

    if {"annual_ret_net", "max_dd_net"}.issubset(out.columns):
        out["calmar_like_ratio"] = safe_div(out["annual_ret_net"], out["max_dd_net"].abs() + 1e-6, default=0.0)

    if {"annual_ret_net", "return_vol_ratio"}.issubset(out.columns):
        out["ret_vs_vol_gap"] = out["annual_ret_net"] - out["return_vol_ratio"]

    run_cols = [
        "avg_turnover",
        "n_rebalance",
        "avg_n_holdings",
        "win_rate",
        "profit_factor",
        "ret_turnover_eff",
        "ret_rebalance_eff",
        "win_loss_ratio",
        "ret_net_minus_gross",
        "sharpe_net_minus_gross",
    ]

    if "run_ts" in out.columns:
        group = out.groupby("run_ts")
        for col in run_cols:
            if col not in out.columns:
                continue
            mean = group[col].transform("mean")
            std = group[col].transform("std").replace(0, np.nan)
            z_col = f"{col}_z"
            pct_col = f"{col}_pct"
            out[z_col] = ((out[col] - mean) / std).replace([np.inf, -np.inf], 0.0).fillna(0.0)
            out[pct_col] = group[col].rank(pct=True, method="average")
        if "ret_turnover_eff" in out.columns:
            out["ret_turnover_eff_rank"] = group["ret_turnover_eff"].rank(pct=True, method="average")

    expected_cols = [
        "ret_net_minus_gross",
        "ret_net_over_gross",
        "ret_cost_ratio",
        "cost_drag",
        "breakeven_turnover_est",
        "avg_turnover_rate",
        "turnover_stability",
        "rebalance_frequency_score",
        "sharpe_net_minus_gross",
        "max_dd_net_minus_gross",
        "dd_ratio_net_over_gross",
        "ret_turnover_eff",
        "ret_rebalance_eff",
        "win_loss_ratio",
        "calmar_like_ratio",
        "ret_vs_vol_gap",
        "ret_turnover_eff_rank",
    ]
    for col in expected_cols:
        if col not in out.columns:
            out[col] = np.nan

    return out


@lru_cache(maxsize=1)
def _load_market_context(etf_prices_path: Path) -> pd.DataFrame:
    if not etf_prices_path.exists():
        raise FileNotFoundError(f"缺少市场价格模板: {etf_prices_path}")
    prices = pd.read_csv(etf_prices_path)
    if {"date", "symbol", "close"} - set(prices.columns):
        raise ValueError("etf_prices_template.csv 必须包含 date,symbol,close 列")
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values("date")
    # 取第一个symbol作为市场代表，如果有多只则取均值
    pivot = prices.pivot_table(index="date", columns="symbol", values="close", aggfunc="mean")
    ref = pivot.mean(axis=1)
    ref = ref.ffill().dropna()
    df = pd.DataFrame({"close": ref})
    df["ret"] = df["close"].pct_change().fillna(0.0)
    df["ma20"] = df["close"].rolling(20, min_periods=1).mean()
    df["ma60"] = df["close"].rolling(60, min_periods=1).mean()
    df["is_bull"] = (df["ma20"] >= df["ma60"]).astype(int)
    df["vol20"] = df["ret"].rolling(20, min_periods=5).std().fillna(0.0)
    vol_q33 = df["vol20"].quantile(0.33)
    vol_q67 = df["vol20"].quantile(0.67)
    df["is_high_vol"] = (df["vol20"] >= vol_q67).astype(int)
    df["is_low_vol"] = (df["vol20"] <= vol_q33).astype(int)
    return df


def _resample_series(series: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return np.asarray([])
    if series.size == 0:
        return np.zeros(n)
    if series.size == n:
        return series
    idx = np.linspace(0, series.size - 1, num=n)
    return np.interp(idx, np.arange(series.size), series)


def add_market_regime_features(df: pd.DataFrame, etf_prices_path: Path) -> pd.DataFrame:
    """市场分regime特征生成器（需至少60日历史数据）"""
    # 校验数据充分性
    try:
        context = _load_market_context(etf_prices_path)
        if len(context) < 60:
            print(f"⚠️  市场数据不足({len(context)}天 < 60天)，跳过市场regime特征生成")
            return df  # 直接返回原始数据，不添加特征
    except (FileNotFoundError, ValueError) as e:
        print(f"⚠️  市场数据加载失败: {e}，跳过市场regime特征生成")
        return df
    
    ret_series = context["ret"].values
    bull_series = context["is_bull"].values.astype(float)
    high_vol_series = context["is_high_vol"].values.astype(float)
    low_vol_series = context["is_low_vol"].values.astype(float)
    out = df.copy()

    def _metrics(values: List[float]) -> Dict[str, float]:
        if not values:
            return {
                "ic_bull_mean": np.nan,
                "ic_bear_mean": np.nan,
                "ic_bull_bear_diff": np.nan,
                "ic_bull_bear_ratio": np.nan,
                "ic_high_vol_mean": np.nan,
                "ic_low_vol_mean": np.nan,
                "ic_vol_regime_ratio": np.nan,
                "vol_regime_stability": np.nan,
                "market_corr": np.nan,
                "market_beta": np.nan,
                "alpha_vs_market": np.nan,
            }
        arr = np.asarray(values, dtype=float)
        n = arr.size
        returns = _resample_series(ret_series, n)
        bulls = _resample_series(bull_series, n)
        highs = _resample_series(high_vol_series, n)
        lows = _resample_series(low_vol_series, n)

        bull_mask = bulls >= 0.5
        bear_mask = ~bull_mask
        high_mask = highs >= 0.5
        low_mask = lows >= 0.5

        bull_vals = arr[bull_mask] if bull_mask.any() else np.array([])
        bear_vals = arr[bear_mask] if bear_mask.any() else np.array([])
        high_vals = arr[high_mask] if high_mask.any() else np.array([])
        low_vals = arr[low_mask] if low_mask.any() else np.array([])

        ic_bull_mean = float(np.mean(bull_vals)) if bull_vals.size else np.nan
        ic_bear_mean = float(np.mean(bear_vals)) if bear_vals.size else np.nan
        ic_high_mean = float(np.mean(high_vals)) if high_vals.size else np.nan
        ic_low_mean = float(np.mean(low_vals)) if low_vals.size else np.nan

        diff = ic_bull_mean - ic_bear_mean if not np.isnan(ic_bull_mean) and not np.isnan(ic_bear_mean) else np.nan
        ratio = np.nan
        if not np.isnan(ic_bear_mean) and not math.isclose(ic_bear_mean, 0.0):
            ratio = ic_bull_mean / ic_bear_mean

        vol_ratio = np.nan
        if not np.isnan(ic_low_mean) and not math.isclose(ic_low_mean, 0.0):
            vol_ratio = ic_high_mean / ic_low_mean

        cv = _coeff_variation(highs) if highs.size > 1 else 0.0
        vol_stability = 1.0 / (1.0 + abs(cv))

        corr = float(np.corrcoef(arr, returns)[0, 1]) if np.std(arr) > 0 and np.std(returns) > 0 else 0.0
        cov = float(np.cov(arr, returns)[0, 1]) if returns.size > 1 else 0.0
        var_m = float(np.var(returns))
        beta = cov / var_m if var_m > 0 else 0.0
        alpha = float(np.mean(arr)) - beta * float(np.mean(returns))

        return {
            "ic_bull_mean": ic_bull_mean,
            "ic_bear_mean": ic_bear_mean,
            "ic_bull_bear_diff": diff,
            "ic_bull_bear_ratio": ratio,
            "ic_high_vol_mean": ic_high_mean,
            "ic_low_vol_mean": ic_low_mean,
            "ic_vol_regime_ratio": vol_ratio,
            "vol_regime_stability": vol_stability,
            "market_corr": corr,
            "market_beta": beta,
            "alpha_vs_market": alpha,
        }

    ic_lists = out.get("oos_ic_list")
    if ic_lists is None:
        return out

    ic_values = ic_lists.apply(lambda x: safe_literal_list(x, "float"))
    metrics = ic_values.apply(_metrics)
    for key in [
        "ic_bull_mean",
        "ic_bear_mean",
        "ic_bull_bear_diff",
        "ic_bull_bear_ratio",
        "ic_high_vol_mean",
        "ic_low_vol_mean",
        "ic_vol_regime_ratio",
        "vol_regime_stability",
        "market_corr",
        "market_beta",
        "alpha_vs_market",
    ]:
        out[f"market_{key}"] = metrics.apply(lambda d: d.get(key, np.nan))
    return out


FACTOR_CATEGORY_RULES: Dict[str, Tuple[str, ...]] = {
    "momentum": (
        "RSI",
        "MACD",
        "MOM",
        "ROC",
        "CCI",
        "PSY",
        "Momentum",
    ),
    "volatility": (
        "ATR",
        "BB",
        "VOL",
        "MAX_DD",
        "STD",
        "HV",
    ),
    "trend": (
        "ADX",
        "VORTEX",
        "AROON",
        "DMI",
        "MA",
        "EMA",
        "SLOPE",
        "TREND",
        "SAR",
    ),
    "volume": (
        "CMF",
        "OBV",
        "MFI",
        "PV_CORR",
        "VOLR",
        "AD",
    ),
}


def _factor_category(factor: str) -> str:
    token = factor.upper()
    for category, prefixes in FACTOR_CATEGORY_RULES.items():
        for prefix in prefixes:
            if token.startswith(prefix) or f"_{prefix}" in token:
                return category
    return "other"


def _extract_period(factor: str) -> Optional[float]:
    parts = factor.split("_")
    for part in reversed(parts):
        try:
            if part.endswith("D"):
                return float(part[:-1])
            return float(part)
        except ValueError:
            continue
    return None


def add_combo_composition_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    combos = out["combo"].fillna("")
    combo_tokens = combos.str.split(" \+ ")

    def _composition(tokens: Iterable[str]) -> Dict[str, float]:
        tokens = [t.strip() for t in tokens if t]
        if not tokens:
            return {
                "n_momentum_factors": 0,
                "n_volatility_factors": 0,
                "n_trend_factors": 0,
                "n_volume_factors": 0,
                "n_other_factors": 0,
                "factor_diversity_score": 0.0,
                "avg_factor_period": np.nan,
                "period_std": np.nan,
                "period_range": np.nan,
                "freq_consistency": np.nan,
                "freq_mode_pct": np.nan,
                "freq_entropy": np.nan,
            }
        counts = {"momentum": 0, "volatility": 0, "trend": 0, "volume": 0, "other": 0}
        periods: List[float] = []
        for token in tokens:
            cat = _factor_category(token)
            counts[cat] += 1
            period = _extract_period(token)
            if period is not None:
                periods.append(period)
        unique_categories = sum(1 for v in counts.values() if v > 0)
        combo_size = max(len(tokens), 1)
        diversity = unique_categories / combo_size

        period_mean = float(np.mean(periods)) if periods else np.nan
        period_std = float(np.std(periods, ddof=1)) if len(periods) > 1 else 0.0 if periods else np.nan
        period_range = float(np.max(periods) - np.min(periods)) if len(periods) > 0 else np.nan

        return {
            "n_momentum_factors": counts["momentum"],
            "n_volatility_factors": counts["volatility"],
            "n_trend_factors": counts["trend"],
            "n_volume_factors": counts["volume"],
            "n_other_factors": counts["other"],
            "factor_diversity_score": diversity,
            "avg_factor_period": period_mean,
            "period_std": period_std,
            "period_range": period_range,
        }

    comp_series = combo_tokens.apply(_composition)
    for key in [
        "n_momentum_factors",
        "n_volatility_factors",
        "n_trend_factors",
        "n_volume_factors",
        "n_other_factors",
        "factor_diversity_score",
        "avg_factor_period",
        "period_std",
        "period_range",
    ]:
        out[key] = comp_series.apply(lambda d: d.get(key, np.nan))

    if "best_freq_list" in out.columns:
        freq_vals = out["best_freq_list"].apply(lambda x: safe_literal_list(x, "int"))

        def _freq_stats(values: List[int]) -> Dict[str, float]:
            if not values:
                return {"freq_consistency": np.nan, "freq_mode_pct": np.nan, "freq_entropy": np.nan}
            arr = np.asarray(values, dtype=float)
            counts = Counter(arr)
            total = arr.size
            mode_freq = counts.most_common(1)[0][1]
            unique = len(counts)
            probs = np.array(list(counts.values()), dtype=float) / total
            entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
            return {
                "freq_consistency": 1.0 - unique / total,
                "freq_mode_pct": mode_freq / total,
                "freq_entropy": entropy,
            }

        freq_stats = freq_vals.apply(_freq_stats)
        for key in ["freq_consistency", "freq_mode_pct", "freq_entropy"]:
            out[key] = freq_stats.apply(lambda d: d.get(key, np.nan))

    return out


def add_extreme_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def series_or_nan(name: str) -> pd.Series:
        if name in out.columns:
            return out[name].astype(float)
        return pd.Series(np.nan, index=out.index)

    winning_days = series_or_nan("winning_days").clip(lower=0)
    losing_days = series_or_nan("losing_days").clip(lower=0)
    total_days = (winning_days + losing_days).replace(0, np.nan)
    max_consecutive_losses = series_or_nan("max_consecutive_losses")
    avg_win = series_or_nan("avg_win")
    avg_loss = series_or_nan("avg_loss")
    annual_ret_net = series_or_nan("annual_ret_net")
    total_ret_net = series_or_nan("total_ret_net")
    max_dd_net = series_or_nan("max_dd_net").replace(0, np.nan)

    out["max_consecutive_loss_days"] = max_consecutive_losses
    out["max_dd_duration"] = max_consecutive_losses

    # Drawdown recovery proxies
    base_return = total_ret_net.where(total_ret_net.notna(), annual_ret_net)
    out["dd_recovery_ratio"] = base_return / (max_dd_net.abs() + 1e-6)
    out["dd_frequency"] = losing_days / total_days

    out["avg_consecutive_loss"] = losing_days / (max_consecutive_losses.replace(0, np.nan))
    out["loss_clustering_score"] = max_consecutive_losses / (losing_days.replace(0, np.nan) + 1e-6)

    downside_var = (losing_days * (avg_loss**2)).fillna(0.0) / total_days
    downside_dev = np.sqrt(downside_var)
    out["downside_deviation"] = downside_dev
    out["sortino_ratio_derived"] = annual_ret_net / (downside_dev.replace(0, np.nan) + 1e-6)

    tail_ratio = avg_win.abs() / (avg_loss.abs() + 1e-6)
    out["tail_ratio"] = tail_ratio

    p_win = winning_days / total_days
    p_loss = losing_days / total_days
    mean_ret = p_win * avg_win + p_loss * avg_loss
    var_ret = p_win * (avg_win - mean_ret) ** 2 + p_loss * (avg_loss - mean_ret) ** 2
    var_ret = var_ret.replace(0, np.nan)
    skew_num = p_win * (avg_win - mean_ret) ** 3 + p_loss * (avg_loss - mean_ret) ** 3
    kurt_num = p_win * (avg_win - mean_ret) ** 4 + p_loss * (avg_loss - mean_ret) ** 4
    out["return_skewness"] = skew_num / (var_ret ** 1.5 + 1e-6)
    out["return_kurtosis"] = kurt_num / (var_ret**2 + 1e-6)

    sanitise_cols = [
        "dd_recovery_ratio",
        "dd_frequency",
        "avg_consecutive_loss",
        "loss_clustering_score",
        "downside_deviation",
        "sortino_ratio_derived",
        "tail_ratio",
        "return_skewness",
        "return_kurtosis",
    ]
    for col in sanitise_cols:
        if col in out.columns:
            out[col] = out[col].replace([np.inf, -np.inf], np.nan)

    return out


def gather_resources(run_ts_list: Sequence[str], results_root: Path, rb_root: Path) -> List[RunResource]:
    pairs: List[RunResource] = []
    for run_ts in run_ts_list:
        run_dir = results_root / f"run_{run_ts}"
        if not run_dir.exists():
            print(f"[WARN] run_{run_ts} 不存在，跳过")
            continue
        csv_list = find_backtest_csvs(run_ts, rb_root)
        if not csv_list:
            print(f"[WARN] 未找到匹配的回测CSV (run_ts={run_ts})，跳过")
            continue
        pairs.append(RunResource(run_dir=run_dir.resolve(), backtest_csvs=csv_list))
    return pairs


def resolve_run_ts(args_runs: Optional[str], results_root: Path) -> List[str]:
    if args_runs:
        parts = [x.strip() for x in args_runs.split(",") if x.strip()]
        return parts
    dirs = list_run_dirs(results_root)
    return [d.name.replace("run_", "") for d in dirs]


def main():
    parser = argparse.ArgumentParser(description="构建排序校准训练数据集")
    parser.add_argument("--run-ts", type=str, help="指定逗号分隔的 run_ts 列表，默认全部可用 run")
    parser.add_argument("--output", type=str, default="data/calibrator_dataset.parquet", help="输出文件路径")
    parser.add_argument(
        "--features-file",
        type=str,
        help="可选：仅保留白名单特征（文件每行一个列名）；始终保留标签与分组必要列",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    results_root = repo_root / "results"
    rb_root = repo_root / "results_combo_wfo"
    output_path = (repo_root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_ts_list = resolve_run_ts(args.run_ts, results_root)
    print("=" * 100)
    print("构建排序训练数据")
    print("=" * 100)
    print(f"候选 run_ts 数量: {len(run_ts_list)}")

    resources = gather_resources(run_ts_list, results_root, rb_root)
    if not resources:
        raise RuntimeError("没有找到可用的 run + 回测文件，请先运行全量回测。")
    print(f"有效 run 数量: {len(resources)}")

    frames = []
    for res in resources:
        print("-" * 80)
        print(f"处理 run: {res.run_dir.name}")
        print(f"  WFO: {res.run_dir / 'all_combos.parquet'}")

        wfo_df = pd.read_parquet(res.run_dir / "all_combos.parquet")
        wfo_feat = enrich_wfo_features(wfo_df)

        for csv in res.backtest_csvs:
            print(f"  回测: {csv}")
            bt_df = pd.read_csv(csv)
            bt_feat = augment_backtest(bt_df)

            merged = pd.merge(wfo_feat, bt_feat, on="combo", how="inner", suffixes=("_wfo", "_bt"))
            merged.insert(0, "run_ts", res.run_ts)
            if "combo_size_wfo" in merged.columns:
                merged.rename(columns={"combo_size_wfo": "combo_size"}, inplace=True)
            if "combo_size_bt" in merged.columns:
                merged.drop(columns=["combo_size_bt"], inplace=True)
            if "combo_size" in merged.columns:
                combo_size_col = merged.pop("combo_size")
                merged.insert(1, "combo_size", combo_size_col)
            print(f"    对齐组合数: {len(merged)}")
            frames.append(merged)

    dataset = pd.concat(frames, axis=0, ignore_index=True)
    dataset = add_standardized_targets(dataset)
    dataset = add_real_derived_features(dataset)
    dataset = add_market_regime_features(dataset, repo_root / "data" / "etf_prices_template.csv")
    dataset = add_combo_composition_features(dataset)
    dataset = add_extreme_risk_features(dataset)
    if "run_tag" in dataset.columns:
        dataset = dataset.drop(columns=["run_tag"])

    # 若提供特征白名单，仅导出可推理侧使用的特征 + 目标列/必要列
    if args.features_file:
        feat_path = (repo_root / args.features_file).resolve() if not Path(args.features_file).is_absolute() else Path(args.features_file)
        if not feat_path.exists():
            raise FileNotFoundError(f"features-file 不存在: {feat_path}")
        whitelist = [ln.strip() for ln in feat_path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.strip().startswith("#")]
        keep_core = {
            "run_ts",
            "combo",
            # 训练目标及其标准化/派生
            "annual_ret_net",
            "sharpe_net_z",
            "calmar_net",
            "annual_ret",
            "sharpe_net",
            "max_dd_net",
            "calmar_net_z",
            "annual_ret_net_z",
        }
        # 仅保留存在于数据集中的列
        keep_cols = [c for c in whitelist if c in dataset.columns]
        keep_cols = list(dict.fromkeys(keep_cols))  # 去重但保序
        final_cols = list(dict.fromkeys([*keep_core, *keep_cols]))
        existing = [c for c in final_cols if c in dataset.columns]
        dataset = dataset[existing].copy()

    dataset.to_parquet(output_path, index=False)

    print("=" * 100)
    print(f"✅ 数据集已保存: {output_path}")
    print(f"样本数: {len(dataset)}")
    print(f"特征总数: {dataset.shape[1]}")
    print("关键指标列示例: annual_ret_net, sharpe_net, calmar_net, ret_turnover_ratio")
    print("=" * 100)


if __name__ == "__main__":
    main()

