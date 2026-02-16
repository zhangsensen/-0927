#!/usr/bin/env python3
"""
P0b + P1: Residual Orthogonalization + Rank EMA Smoothing
==========================================================
After P0 showed simple crosses fail orthogonality (OHLCV parent dominates),
try two alternative approaches to make non-OHLCV signals Exp4-compatible:

P0b: Residual Orthogonalization
  - For each non-OHLCV factor: regress out ALL active OHLCV influence
  - Keep residual ε = non-OHLCV - Σ(β_i * OHLCV_i)
  - Residual is orthogonal by construction

P1: Rank EMA Smoothing
  - Apply EMA(span=5/10/20) to raw cross-sectional ranks
  - Increases rank autocorrelation for Exp4 stability
  - Test both raw non-OHLCV factors and residual factors

Output: results/p0b_residual_ema/
"""

import json
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import yaml

from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("p0b_p1")

SEP = "=" * 80
THIN = "-" * 80
CONFIG_PATH = PROJECT_ROOT / "configs" / "combo_wfo_config.yaml"
NON_OHLCV_DIR = PROJECT_ROOT / "results" / "non_ohlcv_factors"
FREQ = 5
TRAIN_END = pd.Timestamp("2025-04-30")
HO_START = pd.Timestamp("2025-05-01")
MIN_CROSS_SECTION = 5


# ── IC + Rank Stability ──────────────────────────────────


def compute_ic_series(factor_df: pd.DataFrame, fwd_ret: pd.DataFrame) -> pd.Series:
    """Cross-sectional Spearman rank IC."""
    common = factor_df.index.intersection(fwd_ret.index)
    cols = factor_df.columns.intersection(fwd_ret.columns)
    f = factor_df.loc[common, cols]
    r = fwd_ret.loc[common, cols]
    f_rank = f.rank(axis=1)
    r_rank = r.rank(axis=1)
    valid = (f.notna() & r.notna()).sum(axis=1) >= MIN_CROSS_SECTION
    f_dm = f_rank.sub(f_rank.mean(axis=1), axis=0)
    r_dm = r_rank.sub(r_rank.mean(axis=1), axis=0)
    num = (f_dm * r_dm).sum(axis=1)
    den = np.sqrt((f_dm ** 2).sum(axis=1) * (r_dm ** 2).sum(axis=1)).replace(0, np.nan)
    return (num / den).loc[valid].dropna()


def rank_autocorrelation(factor_df: pd.DataFrame, freq: int = FREQ) -> float:
    """Rank autocorrelation at lag=freq."""
    rp = factor_df.rank(axis=1, pct=True)
    lagged = rp.shift(freq)
    mask = rp.notna() & lagged.notna()
    curr = rp.values[mask.values]
    prev = lagged.values[mask.values]
    if len(curr) < 100:
        return 0.0
    corr, _ = stats.spearmanr(curr, prev)
    return float(corr) if np.isfinite(corr) else 0.0


def max_orth_corr(factor_df: pd.DataFrame, active: dict[str, pd.DataFrame]) -> tuple[float, str]:
    """Max |rank correlation| with any active factor."""
    rk_new = factor_df.rank(axis=1, pct=True)
    best = 0.0
    best_name = ""
    for name, adf in active.items():
        common = rk_new.index.intersection(adf.index)
        cols = rk_new.columns.intersection(adf.columns)
        if len(common) < 100 or len(cols) < 5:
            continue
        rk_a = adf.loc[common, cols].rank(axis=1, pct=True)
        rk_n = rk_new.loc[common, cols]
        mask = rk_n.notna() & rk_a.notna()
        a, b = rk_n.values[mask.values], rk_a.values[mask.values]
        if len(a) < 100:
            continue
        c, _ = stats.spearmanr(a, b)
        if np.isfinite(c) and abs(c) > best:
            best = abs(c)
            best_name = name
    return best, best_name


def ic_stats(ic_series: pd.Series) -> dict:
    """IC summary split by train/holdout."""
    if len(ic_series) == 0:
        return {"full_IC": 0, "full_IR": 0, "train_IC": 0, "ho_IC": 0, "n": 0}
    train = ic_series[ic_series.index <= TRAIN_END]
    ho = ic_series[ic_series.index >= HO_START]

    def _s(s):
        return (float(s.mean()), float(s.mean() / s.std()) if len(s) > 1 and s.std() > 1e-10 else 0.0)

    fi, fr = _s(ic_series)
    ti, _ = _s(train)
    hi, _ = _s(ho)
    return {"full_IC": fi, "full_IR": fr, "train_IC": ti, "ho_IC": hi, "n": len(ic_series)}


# ── P0b: Residual Orthogonalization ──────────────────────


def orthogonalize_residual(
    target_df: pd.DataFrame,
    regressors: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Cross-sectional residual: target - Σ(β_i * regressor_i) per date.

    Uses OLS per-date with all regressors jointly.
    """
    common = target_df.index
    for df in regressors.values():
        common = common.intersection(df.index)
    cols = target_df.columns
    for df in regressors.values():
        cols = cols.intersection(df.columns)

    target = target_df.loc[common, cols]
    reg_dfs = {k: v.loc[common, cols] for k, v in regressors.items()}

    residual = pd.DataFrame(np.nan, index=common, columns=cols)
    reg_names = sorted(reg_dfs.keys())

    for dt in common:
        y = target.loc[dt].values.astype(float)
        X = np.column_stack([reg_dfs[r].loc[dt].values.astype(float) for r in reg_names])

        valid = np.all(np.isfinite(np.column_stack([y.reshape(-1, 1), X])), axis=1)
        if valid.sum() < MIN_CROSS_SECTION:
            continue

        y_v = y[valid]
        X_v = X[valid]
        # Add intercept
        X_v = np.column_stack([np.ones(len(y_v)), X_v])

        try:
            beta, _, _, _ = np.linalg.lstsq(X_v, y_v, rcond=None)
            y_hat = X_v @ beta
            eps = y_v - y_hat
        except np.linalg.LinAlgError:
            continue

        res_full = np.full(len(y), np.nan)
        res_full[valid] = eps
        residual.loc[dt] = res_full

    return residual


# ── P1: Rank EMA Smoothing ───────────────────────────────


def rank_ema_smooth(factor_df: pd.DataFrame, span: int) -> pd.DataFrame:
    """Apply EMA smoothing to cross-sectional ranks.

    Steps:
    1. Cross-sectional rank percentile [0,1]
    2. EMA(span) along time axis per ETF
    3. Result maintains cross-sectional spread but smoother over time
    """
    ranked = factor_df.rank(axis=1, pct=True)
    smoothed = ranked.ewm(span=span, adjust=False).mean()
    return smoothed


# ── Main ─────────────────────────────────────────────────


def main():
    t0 = time.time()
    output_dir = PROJECT_ROOT / "results" / "p0b_residual_ema"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{SEP}")
    print("P0b + P1: Residual Orthogonalization + Rank EMA Smoothing")
    print(SEP)

    # Load config
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    ohlcv_names = sorted([f for f in config.get("active_factors", [])
                          if not f.startswith(("SHARE_", "MARGIN_"))])
    nonohlcv_names = sorted([f for f in config.get("active_factors", [])
                             if f.startswith(("SHARE_", "MARGIN_"))])

    # Load data
    print(f"\n[1/5] Loading data...")
    data_cfg = config["data"]
    loader = DataLoader(data_dir=data_cfg.get("data_dir"), cache_dir=data_cfg.get("cache_dir"))
    ohlcv = loader.load_ohlcv(
        etf_codes=data_cfg.get("symbols"),
        start_date=data_cfg.get("start_date"),
        end_date=data_cfg.get("end_date"),
    )
    close = ohlcv["close"]
    fwd_ret = close.shift(-FREQ) / close - 1
    print(f"  {close.shape[1]} ETFs, {len(close)} days")

    # Compute OHLCV factors (standardized)
    print(f"\n[2/5] Computing factors...")
    lib = PreciseFactorLibrary()
    raw = lib.compute_all_factors(prices=ohlcv)
    proc = CrossSectionProcessor(verbose=False)
    ohlcv_sub = {n: raw[n] for n in ohlcv_names if n in raw}
    std_ohlcv = proc.process_all_factors(ohlcv_sub)

    # Load non-OHLCV
    raw_nonohlcv = {}
    for name in nonohlcv_names:
        path = NON_OHLCV_DIR / f"{name}.parquet"
        if path.exists():
            df = pd.read_parquet(path).reindex(index=close.index, columns=close.columns)
            raw_nonohlcv[name] = df
    std_nonohlcv = proc.process_all_factors(raw_nonohlcv)
    print(f"  OHLCV: {len(std_ohlcv)}, non-OHLCV: {len(std_nonohlcv)}")

    # For orthogonality: check vs OHLCV-only active factors
    # (non-OHLCV factors are themselves active, so including them would be self-correlation)
    orth_reference = dict(std_ohlcv)  # OHLCV-only for fair orth comparison

    # ── P0b: Residual orthogonalization ───────────────────
    print(f"\n[3/5] P0b: Orthogonalizing non-OHLCV factors (regress out OHLCV)...")
    print(THIN)

    # Use top-5 most rank-stable OHLCV as regressors (to avoid overfitting with all 17)
    top_ohlcv_by_stability = sorted(
        std_ohlcv.items(),
        key=lambda x: rank_autocorrelation(x[1]),
        reverse=True,
    )[:5]
    regressor_names = [n for n, _ in top_ohlcv_by_stability]
    regressors = {n: df for n, df in top_ohlcv_by_stability}
    print(f"  Regressors (top 5 by rank stability): {regressor_names}")

    residual_factors = {}
    for name in sorted(std_nonohlcv.keys()):
        res = orthogonalize_residual(std_nonohlcv[name], regressors)
        residual_factors[f"RESID_{name}"] = res
        valid_rate = float(res.notna().mean().mean())
        print(f"  RESID_{name}: valid={valid_rate:.1%}")

    # ── P1: Rank EMA smoothing ────────────────────────────
    print(f"\n[4/5] P1: Rank EMA smoothing (span=5, 10, 20)...")
    print(THIN)

    ema_factors = {}
    for span in [5, 10, 20]:
        for name in sorted(std_nonohlcv.keys()):
            smoothed = rank_ema_smooth(std_nonohlcv[name], span=span)
            ema_name = f"EMA{span}_{name}"
            ema_factors[ema_name] = smoothed

        # Also smooth residuals
        for name in sorted(residual_factors.keys()):
            smoothed = rank_ema_smooth(residual_factors[name], span=span)
            ema_name = f"EMA{span}_{name}"
            ema_factors[ema_name] = smoothed

    print(f"  Generated: {len(ema_factors)} EMA variants")

    # ── Screen all candidates ─────────────────────────────
    print(f"\n[5/5] Screening all candidates (IC + rank stability + orthogonality)...")
    print(THIN)

    candidates = {}
    candidates.update(residual_factors)  # P0b residuals
    candidates.update(ema_factors)       # P1 EMA variants
    # Also include raw non-OHLCV as baseline
    for name, df in std_nonohlcv.items():
        candidates[f"RAW_{name}"] = df

    results = []
    for name in sorted(candidates.keys()):
        df = candidates[name]
        ic = compute_ic_series(df, fwd_ret)
        if len(ic) < 100:
            continue

        ics = ic_stats(ic)
        rac = rank_autocorrelation(df, FREQ)
        orth, orth_parent = max_orth_corr(df, orth_reference)

        results.append({
            "name": name,
            "type": "residual" if "RESID" in name else ("ema" if "EMA" in name else "raw"),
            "full_IC": ics["full_IC"],
            "full_IR": ics["full_IR"],
            "train_IC": ics["train_IC"],
            "ho_IC": ics["ho_IC"],
            "rank_autocorr": rac,
            "max_orth_corr": orth,
            "most_correlated": orth_parent,
            "pass_ic": abs(ics["full_IC"]) >= 0.03,
            "pass_rank": rac >= 0.70,
            "pass_orth": orth <= 0.60,
            "SURVIVOR": abs(ics["full_IC"]) >= 0.03 and rac >= 0.70 and orth <= 0.60,
        })

    rdf = pd.DataFrame(results).sort_values("full_IC", ascending=False, key=abs)

    # ── Results ───────────────────────────────────────────
    print(f"\n{SEP}")
    print("RESULTS SUMMARY")
    print(SEP)

    for typ in ["raw", "residual", "ema"]:
        subset = rdf[rdf["type"] == typ]
        if len(subset) == 0:
            continue
        ic_pass = subset["pass_ic"].sum()
        rank_pass = (subset["pass_ic"] & subset["pass_rank"]).sum()
        survivors = subset["SURVIVOR"].sum()
        print(f"\n  {typ.upper():10s}: {len(subset):3d} total, "
              f"{ic_pass:3d} IC-pass, {rank_pass:3d} rank-pass, {survivors:3d} SURVIVORS")

    survivors_df = rdf[rdf["SURVIVOR"]]
    print(f"\n{THIN}")
    print(f"TOTAL SURVIVORS: {len(survivors_df)}")
    print(THIN)

    if len(survivors_df) > 0:
        for _, r in survivors_df.iterrows():
            sign = "+" if r["full_IC"] > 0 else ""
            print(
                f"  {r['name']:55s}  IC={sign}{r['full_IC']:.4f}  "
                f"IR={r['full_IR']:.3f}  rank_ac={r['rank_autocorr']:.3f}  "
                f"orth={r['max_orth_corr']:.3f}  HO={r['ho_IC']:.4f}"
            )

    # Near-misses (relax orth to 0.70)
    near_miss = rdf[(rdf["pass_ic"]) & (rdf["pass_rank"]) & (rdf["max_orth_corr"] <= 0.70)]
    if len(near_miss) > 0:
        print(f"\nNEAR-MISSES (orth<0.70): {len(near_miss)}")
        for _, r in near_miss.head(20).iterrows():
            sign = "+" if r["full_IC"] > 0 else ""
            print(
                f"  {r['name']:55s}  IC={sign}{r['full_IC']:.4f}  "
                f"rank_ac={r['rank_autocorr']:.3f}  orth={r['max_orth_corr']:.3f}  "
                f"HO={r['ho_IC']:.4f}"
            )

    # Rank stability improvement analysis
    print(f"\n{THIN}")
    print("RANK STABILITY IMPROVEMENT (EMA effect)")
    print(THIN)
    for base_name in sorted(nonohlcv_names):
        raw_ac = rank_autocorrelation(std_nonohlcv[base_name], FREQ)
        print(f"\n  {base_name} (raw rank_ac={raw_ac:.3f}):")
        for span in [5, 10, 20]:
            ema_name = f"EMA{span}_{base_name}"
            if ema_name in candidates:
                ema_ac = rank_autocorrelation(candidates[ema_name], FREQ)
                print(f"    EMA(span={span:2d}): rank_ac={ema_ac:.3f}  (Δ={ema_ac - raw_ac:+.3f})")

    # Orthogonality comparison: residual vs raw
    print(f"\n{THIN}")
    print("ORTHOGONALITY: RESIDUAL vs RAW")
    print(THIN)
    for base_name in sorted(nonohlcv_names):
        raw_name = f"RAW_{base_name}"
        resid_name = f"RESID_{base_name}"
        raw_row = rdf[rdf["name"] == raw_name]
        resid_row = rdf[rdf["name"] == resid_name]
        if len(raw_row) > 0 and len(resid_row) > 0:
            raw_orth = raw_row.iloc[0]["max_orth_corr"]
            resid_orth = resid_row.iloc[0]["max_orth_corr"]
            raw_ic = raw_row.iloc[0]["full_IC"]
            resid_ic = resid_row.iloc[0]["full_IC"]
            print(f"  {base_name:20s}  raw: orth={raw_orth:.3f} IC={raw_ic:+.4f}  |  "
                  f"resid: orth={resid_orth:.3f} IC={resid_ic:+.4f}  (Δorth={resid_orth-raw_orth:+.3f})")

    # Save
    rdf.to_csv(output_dir / "screen_results.csv", index=False)
    if len(survivors_df) > 0:
        survivors_df.to_csv(output_dir / "survivors.csv", index=False)
        # Save survivor factor values
        surv_dir = output_dir / "survivor_factors"
        surv_dir.mkdir(exist_ok=True)
        for _, r in survivors_df.iterrows():
            name = r["name"]
            if name in candidates:
                candidates[name].to_parquet(surv_dir / f"{name}.parquet")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    return rdf


if __name__ == "__main__":
    main()
