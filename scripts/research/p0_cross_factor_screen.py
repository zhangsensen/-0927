#!/usr/bin/env python3
"""
P0: OHLCV × Non-OHLCV Cross-Factor IC Screening + Rank Stability Analysis
==========================================================================
Core hypothesis: OHLCV factors provide rank stability; non-OHLCV factors
provide orthogonal alpha. Algebraic crosses may inherit both properties,
surviving Exp4's delta_rank=0.10 hysteresis wall.

Pipeline:
  1. Load 17 OHLCV factors (standardized) + 6 non-OHLCV factors (from parquet)
  2. Generate 17 × 6 × 6 = 612 cross candidates (add, sub, mul, div, max, min)
  3. Screen: |IC| > 0.03, rank_autocorr_5d >= 0.70
  4. Orthogonality check vs active factors
  5. Save survivors for WFO/VEC validation

Output: results/p0_cross_factors/
"""

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

# ── Project path ──────────────────────────────────────────
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
logger = logging.getLogger("p0_cross")

SEP = "=" * 80
THIN = "-" * 80

# ── Config ──────────────────────────────────────────────
CONFIG_PATH = PROJECT_ROOT / "configs" / "combo_wfo_config.yaml"
NON_OHLCV_DIR = PROJECT_ROOT / "results" / "non_ohlcv_factors"
TRAIN_END = "2025-04-30"
HO_START = "2025-05-01"
FREQ = 5

# Algebraic operators (same as discovery.py)
OPERATORS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b.replace(0, np.nan),
    "max": lambda a, b: np.maximum(a, b),
    "min": lambda a, b: np.minimum(a, b),
}

# Screening thresholds
IC_THRESHOLD = 0.03       # |IC| > 0.03
RANK_AUTOCORR_MIN = 0.70  # rank stability for Exp4 compatibility
ORTH_MAX = 0.60            # max rank correlation with any active factor
VALID_RATE_MIN = 0.50      # minimum non-NaN rate
MIN_CROSS_SECTION = 5      # min ETFs for IC calculation


# ── Core functions ──────────────────────────────────────


def compute_ic_series(factor_df: pd.DataFrame, fwd_ret: pd.DataFrame) -> pd.Series:
    """Cross-sectional Spearman rank IC (vectorized)."""
    common_dates = factor_df.index.intersection(fwd_ret.index)
    common_codes = factor_df.columns.intersection(fwd_ret.columns)

    f = factor_df.loc[common_dates, common_codes]
    r = fwd_ret.loc[common_dates, common_codes]

    # Rank cross-sectionally
    f_rank = f.rank(axis=1)
    r_rank = r.rank(axis=1)

    # Mask dates with enough valid obs
    valid_count = (f.notna() & r.notna()).sum(axis=1)
    valid_dates = valid_count >= MIN_CROSS_SECTION

    # Demean
    f_dm = f_rank.sub(f_rank.mean(axis=1), axis=0)
    r_dm = r_rank.sub(r_rank.mean(axis=1), axis=0)

    # Pearson on ranks = Spearman
    num = (f_dm * r_dm).sum(axis=1)
    den = np.sqrt((f_dm ** 2).sum(axis=1) * (r_dm ** 2).sum(axis=1))
    den = den.replace(0, np.nan)
    ic = (num / den).loc[valid_dates]

    return ic.dropna()


def compute_rank_autocorrelation(factor_df: pd.DataFrame, freq: int = FREQ) -> float:
    """Rank autocorrelation at lag=freq (Exp4 stability proxy)."""
    rank_pct = factor_df.rank(axis=1, pct=True)
    lagged = rank_pct.shift(freq)

    # Flatten valid pairs
    mask = rank_pct.notna() & lagged.notna()
    current = rank_pct.values[mask.values]
    previous = lagged.values[mask.values]

    if len(current) < 100:
        return 0.0

    corr, _ = stats.spearmanr(current, previous)
    return float(corr) if np.isfinite(corr) else 0.0


def compute_orthogonality(
    factor_df: pd.DataFrame,
    active_factors: dict[str, pd.DataFrame],
) -> tuple[float, str]:
    """Max |rank correlation| with any active factor. Returns (max_corr, most_correlated)."""
    rank_new = factor_df.rank(axis=1, pct=True)
    max_corr = 0.0
    most_corr_name = ""

    for name, active_df in active_factors.items():
        common_dates = rank_new.index.intersection(active_df.index)
        common_codes = rank_new.columns.intersection(active_df.columns)
        if len(common_dates) < 100 or len(common_codes) < 5:
            continue

        rank_active = active_df.loc[common_dates, common_codes].rank(axis=1, pct=True)
        rank_n = rank_new.loc[common_dates, common_codes]

        mask = rank_n.notna() & rank_active.notna()
        a = rank_n.values[mask.values]
        b = rank_active.values[mask.values]

        if len(a) < 100:
            continue

        corr, _ = stats.spearmanr(a, b)
        if np.isfinite(corr) and abs(corr) > max_corr:
            max_corr = abs(corr)
            most_corr_name = name

    return max_corr, most_corr_name


def ic_summary(ic_series: pd.Series, train_end: str, ho_start: str) -> dict:
    """IC stats split by train/holdout."""
    if len(ic_series) == 0:
        return {"full_IC": 0, "full_IR": 0, "train_IC": 0, "ho_IC": 0, "n": 0}

    # Handle both string and datetime index
    idx = ic_series.index
    if hasattr(idx[0], 'strftime'):
        # DatetimeIndex
        train_dt = pd.Timestamp(train_end)
        ho_dt = pd.Timestamp(ho_start)
        train_mask = idx <= train_dt
        ho_mask = idx >= ho_dt
    else:
        idx_str = idx.astype(str)
        train_mask = idx_str <= train_end.replace("-", "")
        ho_mask = idx_str >= ho_start.replace("-", "")

    def _s(s):
        if len(s) == 0:
            return 0.0, 0.0
        return float(s.mean()), float(s.mean() / s.std()) if s.std() > 1e-10 else 0.0

    full_ic, full_ir = _s(ic_series)
    train_ic, _ = _s(ic_series[train_mask])
    ho_ic, _ = _s(ic_series[ho_mask])

    return {
        "full_IC": full_ic,
        "full_IR": full_ir,
        "train_IC": train_ic,
        "ho_IC": ho_ic,
        "n": len(ic_series),
    }


# ── Main pipeline ───────────────────────────────────────


def main():
    t0 = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "results" / "p0_cross_factors"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{SEP}")
    print("P0: OHLCV × Non-OHLCV Cross-Factor Screening")
    print(f"Hypothesis: OHLCV rank stability + non-OHLCV alpha → Exp4-compatible factors")
    print(f"Thresholds: |IC|>{IC_THRESHOLD}, rank_autocorr≥{RANK_AUTOCORR_MIN}, orth<{ORTH_MAX}")
    print(SEP)

    # ── 1. Load config ────────────────────────────────────
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    active_factor_names = sorted(config.get("active_factors", []))
    ohlcv_factor_names = [f for f in active_factor_names if not f.startswith(("SHARE_", "MARGIN_"))]
    nonohlcv_factor_names = [f for f in active_factor_names if f.startswith(("SHARE_", "MARGIN_"))]

    print(f"\n[1/7] Config loaded")
    print(f"  OHLCV active: {len(ohlcv_factor_names)} factors")
    print(f"  Non-OHLCV active: {len(nonohlcv_factor_names)} factors")
    print(f"  Cross candidates: {len(ohlcv_factor_names)} × {len(nonohlcv_factor_names)} × 6 = "
          f"{len(ohlcv_factor_names) * len(nonohlcv_factor_names) * 6}")

    # ── 2. Load data ──────────────────────────────────────
    print(f"\n[2/7] Loading data...")
    data_cfg = config["data"]
    loader = DataLoader(
        data_dir=data_cfg.get("data_dir"),
        cache_dir=data_cfg.get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=data_cfg.get("symbols"),
        start_date=data_cfg.get("start_date"),
        end_date=data_cfg.get("end_date"),
    )
    close = ohlcv["close"]
    print(f"  {close.shape[1]} ETFs, {len(close)} trading days")
    print(f"  Date range: {close.index[0].date()} ~ {close.index[-1].date()}")

    # ── 3. Compute OHLCV factors ──────────────────────────
    print(f"\n[3/7] Computing OHLCV factors...")
    lib = PreciseFactorLibrary()
    raw_factors = lib.compute_all_factors(prices=ohlcv)

    # Cross-section standardize
    proc = CrossSectionProcessor(verbose=False)
    ohlcv_subset = {name: raw_factors[name] for name in ohlcv_factor_names if name in raw_factors}
    std_ohlcv = proc.process_all_factors(ohlcv_subset)
    print(f"  Standardized: {len(std_ohlcv)} OHLCV factors")

    # ── 4. Load non-OHLCV factors from precomputed parquets ─
    print(f"\n[4/7] Loading non-OHLCV factors from {NON_OHLCV_DIR}...")
    raw_nonohlcv = {}
    for name in nonohlcv_factor_names:
        path = NON_OHLCV_DIR / f"{name}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            # Align to close index/columns
            raw_nonohlcv[name] = df.reindex(index=close.index, columns=close.columns)
        else:
            logger.warning(f"  Missing: {path}")
    std_nonohlcv = proc.process_all_factors(raw_nonohlcv)
    print(f"  Loaded: {len(std_nonohlcv)} non-OHLCV factors")

    if not std_nonohlcv:
        print("\nERROR: No non-OHLCV factors found. Run precompute first:")
        print("  uv run python scripts/precompute_non_ohlcv_factors.py")
        return

    # ── 5. Forward returns ────────────────────────────────
    print(f"\n[5/7] Computing {FREQ}-day forward returns...")
    fwd_ret = close.shift(-FREQ) / close - 1
    print(f"  Valid observations: {fwd_ret.notna().sum().sum():,}")

    # ── 6. Generate crosses & screen ──────────────────────
    print(f"\n[6/7] Generating OHLCV × Non-OHLCV crosses...")
    print(THIN)

    results = []
    survivors = {}
    total = len(std_ohlcv) * len(std_nonohlcv) * len(OPERATORS)
    processed = 0
    passed_ic = 0
    passed_rank = 0

    for ohlcv_name in sorted(std_ohlcv.keys()):
        df_ohlcv = std_ohlcv[ohlcv_name]

        for nonohlcv_name in sorted(std_nonohlcv.keys()):
            df_nonohlcv = std_nonohlcv[nonohlcv_name]

            for op_name, op_fn in sorted(OPERATORS.items()):
                processed += 1
                cross_name = f"{ohlcv_name}__{op_name}__{nonohlcv_name}"

                try:
                    cross_df = op_fn(df_ohlcv, df_nonohlcv)
                except Exception:
                    continue

                # Valid rate check
                valid_rate = float(cross_df.notna().mean().mean())
                if valid_rate < VALID_RATE_MIN:
                    continue

                # IC screening
                ic_series = compute_ic_series(cross_df, fwd_ret)
                if len(ic_series) < 100:
                    continue

                ic_stats = ic_summary(ic_series, TRAIN_END, HO_START)
                full_ic = ic_stats["full_IC"]

                if abs(full_ic) < IC_THRESHOLD:
                    continue
                passed_ic += 1

                # Rank autocorrelation (Exp4 stability)
                rank_ac = compute_rank_autocorrelation(cross_df, FREQ)

                row = {
                    "name": cross_name,
                    "ohlcv_parent": ohlcv_name,
                    "nonohlcv_parent": nonohlcv_name,
                    "operator": op_name,
                    "full_IC": full_ic,
                    "full_IR": ic_stats["full_IR"],
                    "train_IC": ic_stats["train_IC"],
                    "ho_IC": ic_stats["ho_IC"],
                    "rank_autocorr_5d": rank_ac,
                    "valid_rate": valid_rate,
                    "n_ic_days": ic_stats["n"],
                    "pass_ic": abs(full_ic) >= IC_THRESHOLD,
                    "pass_rank": rank_ac >= RANK_AUTOCORR_MIN,
                }
                results.append(row)

                if rank_ac >= RANK_AUTOCORR_MIN:
                    passed_rank += 1
                    survivors[cross_name] = cross_df

                # Progress
                if processed % 100 == 0:
                    print(f"  [{processed}/{total}] IC pass: {passed_ic}, "
                          f"IC+Rank pass: {passed_rank}")

    print(f"\n  Total processed: {processed}")
    print(f"  Passed IC (|IC|>{IC_THRESHOLD}): {passed_ic}")
    print(f"  Passed IC + Rank (autocorr≥{RANK_AUTOCORR_MIN}): {passed_rank}")

    if not results:
        print("\nNo candidates passed IC threshold. Exiting.")
        return

    # Build results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("full_IC", ascending=False, key=abs)

    # ── 7. Orthogonality check for survivors ──────────────
    all_active = {**std_ohlcv, **std_nonohlcv}
    if survivors:
        print(f"\n[7/7] Orthogonality check for {len(survivors)} survivors...")
        orth_results = []
        for sname, sdf in sorted(survivors.items()):
            max_corr, most_corr = compute_orthogonality(sdf, all_active)
            orth_results.append({
                "name": sname,
                "max_orth_corr": max_corr,
                "most_correlated_with": most_corr,
                "pass_orth": max_corr <= ORTH_MAX,
            })

        orth_df = pd.DataFrame(orth_results).set_index("name")
        results_df = results_df.set_index("name")
        results_df = results_df.join(orth_df, how="left")
        results_df["SURVIVOR"] = (
            results_df["pass_ic"].fillna(False)
            & results_df["pass_rank"].fillna(False)
            & results_df.get("pass_orth", pd.Series(True, index=results_df.index)).fillna(True)
        )
        results_df = results_df.reset_index()
    else:
        results_df["max_orth_corr"] = np.nan
        results_df["most_correlated_with"] = ""
        results_df["pass_orth"] = False
        results_df["SURVIVOR"] = False
        print(f"\n[7/7] No survivors to check orthogonality.")

    # ── Print Summary ─────────────────────────────────────
    print(f"\n{SEP}")
    print("CROSS-FACTOR SCREENING RESULTS")
    print(SEP)

    final_survivors = results_df[results_df["SURVIVOR"]]
    print(f"\nSURVIVORS: {len(final_survivors)} / {len(results_df)} pass all 3 gates")
    print(f"  Gate 1 (|IC|>{IC_THRESHOLD}): {results_df['pass_ic'].sum()}")
    print(f"  Gate 2 (rank_autocorr≥{RANK_AUTOCORR_MIN}): {(results_df['pass_ic'] & results_df['pass_rank']).sum()}")
    print(f"  Gate 3 (orth<{ORTH_MAX}): {len(final_survivors)}")

    if len(final_survivors) > 0:
        print(f"\n{THIN}")
        print("Top Survivors (sorted by |IC|):")
        print(THIN)
        top = final_survivors.sort_values("full_IC", ascending=False, key=abs).head(30)
        for _, row in top.iterrows():
            sign = "+" if row["full_IC"] > 0 else ""
            print(
                f"  {row['name']:55s}  "
                f"IC={sign}{row['full_IC']:.4f}  "
                f"IR={row['full_IR']:.3f}  "
                f"rank_ac={row['rank_autocorr_5d']:.3f}  "
                f"orth={row.get('max_orth_corr', 0):.3f}  "
                f"HO_IC={row['ho_IC']:.4f}"
            )

    # ── Analysis: rank stability inheritance ──────────────
    print(f"\n{THIN}")
    print("RANK STABILITY INHERITANCE ANALYSIS")
    print(THIN)

    # Compare parent OHLCV rank_autocorr vs cross rank_autocorr
    ohlcv_rank_ac = {}
    for name, df in std_ohlcv.items():
        ohlcv_rank_ac[name] = compute_rank_autocorrelation(df, FREQ)

    nonohlcv_rank_ac = {}
    for name, df in std_nonohlcv.items():
        nonohlcv_rank_ac[name] = compute_rank_autocorrelation(df, FREQ)

    print("\nParent factor rank autocorrelation (5-day):")
    print("  OHLCV factors:")
    for name in sorted(ohlcv_rank_ac, key=ohlcv_rank_ac.get, reverse=True):
        print(f"    {name:35s}  rank_ac={ohlcv_rank_ac[name]:.3f}")
    print("  Non-OHLCV factors:")
    for name in sorted(nonohlcv_rank_ac, key=nonohlcv_rank_ac.get, reverse=True):
        print(f"    {name:35s}  rank_ac={nonohlcv_rank_ac[name]:.3f}")

    # Operator effectiveness
    if len(results_df) > 0:
        ic_pass = results_df[results_df["pass_ic"]]
        print(f"\nOperator breakdown (IC-passing crosses):")
        for op in sorted(OPERATORS.keys()):
            op_rows = ic_pass[ic_pass["operator"] == op]
            rank_pass = op_rows[op_rows["pass_rank"]].shape[0] if len(op_rows) > 0 else 0
            avg_ic = op_rows["full_IC"].abs().mean() if len(op_rows) > 0 else 0
            avg_rank = op_rows["rank_autocorr_5d"].mean() if len(op_rows) > 0 else 0
            print(f"    {op:5s}: {len(op_rows):3d} IC-pass, {rank_pass:3d} rank-pass, "
                  f"avg|IC|={avg_ic:.4f}, avg_rank_ac={avg_rank:.3f}")

    # Non-OHLCV parent effectiveness
    if len(results_df) > 0:
        print(f"\nNon-OHLCV parent contribution:")
        for name in sorted(nonohlcv_rank_ac.keys()):
            parent_rows = results_df[results_df["nonohlcv_parent"] == name]
            ic_p = parent_rows["pass_ic"].sum()
            rank_p = (parent_rows["pass_ic"] & parent_rows["pass_rank"]).sum()
            surv = parent_rows.get("SURVIVOR", pd.Series(False)).sum()
            print(f"    {name:20s}: {ic_p:3d} IC-pass, {rank_p:3d} rank-pass, {surv:3d} survivors")

    # ── Save results ──────────────────────────────────────
    print(f"\n{SEP}")
    print(f"Saving results to {output_dir}/")
    print(SEP)

    results_df.to_csv(output_dir / "cross_screen_all.csv", index=False)
    print(f"  cross_screen_all.csv ({len(results_df)} rows)")

    if len(final_survivors) > 0:
        final_survivors.to_csv(output_dir / "survivors.csv", index=False)
        print(f"  survivors.csv ({len(final_survivors)} rows)")

        # Save survivor factor values as parquet for WFO/VEC
        survivor_dir = output_dir / "survivor_factors"
        survivor_dir.mkdir(exist_ok=True)
        for _, row in final_survivors.iterrows():
            name = row["name"]
            if name in survivors:
                survivors[name].to_parquet(survivor_dir / f"{name}.parquet")
        print(f"  survivor_factors/ ({len(final_survivors)} parquet files)")

    # Parent rank stability
    rank_ac_data = {
        "ohlcv": ohlcv_rank_ac,
        "nonohlcv": nonohlcv_rank_ac,
    }
    import json
    with open(output_dir / "parent_rank_stability.json", "w") as f:
        json.dump(rank_ac_data, f, indent=2)
    print(f"  parent_rank_stability.json")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({elapsed / 60:.1f}min)")

    return results_df


if __name__ == "__main__":
    results = main()
