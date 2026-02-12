#!/usr/bin/env python3
"""
Phase-0: Factor Pre-Filter Diagnostic
======================================
Analyzes 536 candidate factors to determine optimal pre-filter thresholds.

Sections:
  C) Quality metrics distributions (rank_autocorr, NaN, rolling IC, etc.)
  A) Orthogonality with active 17 factors (cross-sectional rank correlation)
  B) Bucket compatibility (bucket coverage & diversity)
  Σ) Combined multi-gate cascade + recommendations

Usage:
  uv run python scripts/phase0_factor_diagnostic.py
"""

import json
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ── Project paths ────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

RESULTS_DIR = PROJECT / "results" / "factor_mining_20260205_204210"
QUALITY_PARQUET = RESULTS_DIR / "quality_reports.parquet"
REGISTRY_JSON = RESULTS_DIR / "factor_registry.json"

# ── Active 17 factors (production v5.0) ──────────────────────
ACTIVE_17 = sorted([
    "ADX_14D", "AMIHUD_ILLIQUIDITY", "BREAKOUT_20D", "CALMAR_RATIO_60D",
    "CORRELATION_TO_MARKET_20D", "GK_VOL_RATIO_20D", "MAX_DD_60D", "MOM_20D",
    "OBV_SLOPE_10D", "PRICE_POSITION_20D", "PRICE_POSITION_120D", "PV_CORR_20D",
    "SHARPE_RATIO_20D", "SLOPE_20D", "UP_DOWN_VOL_RATIO_20D", "VOL_RATIO_20D",
    "VORTEX_14D",
])

SEP = "=" * 72
THIN = "-" * 72

logging.basicConfig(level=logging.WARNING)


# ── Utility ──────────────────────────────────────────────────

def print_dist(series: pd.Series, name: str,
               thresholds: list = None, higher_is_better: bool = True):
    """Print percentile distribution + threshold pass rates."""
    s = series.dropna()
    print(f"\n  {name}  (n={len(s)}/{len(series)})")
    pcts = s.quantile([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    print(f"    mean={s.mean():.4f}  std={s.std():.4f}")
    print(f"    p5={pcts[0.05]:.4f}  p10={pcts[0.10]:.4f}  p25={pcts[0.25]:.4f}  "
          f"p50={pcts[0.50]:.4f}  p75={pcts[0.75]:.4f}  "
          f"p90={pcts[0.90]:.4f}  p95={pcts[0.95]:.4f}")
    if thresholds:
        for t in thresholds:
            if higher_is_better:
                n_pass = int((s >= t).sum())
                print(f"      >= {t:.2f}: {n_pass:4d} / {len(s)} ({n_pass/len(s)*100:5.1f}%)")
            else:
                n_pass = int((s <= t).sum())
                print(f"      <= {t:.2f}: {n_pass:4d} / {len(s)} ({n_pass/len(s)*100:5.1f}%)")


# ── Section C: Existing quality metrics ──────────────────────

def section_c(qr: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{SEP}")
    print("  C) EXISTING QUALITY METRICS (from quality_reports.parquet)")
    print(SEP)

    print_dist(qr["rank_autocorrelation"],
               "C1: Rank Autocorrelation (Exp4 hysteresis compat)",
               [0.5, 0.6, 0.7, 0.8, 0.9])

    print_dist(qr["nan_rate"],
               "C2: NaN Rate (data completeness)",
               [0.05, 0.10, 0.20, 0.30], higher_is_better=False)

    print_dist(qr["rolling_ic_positive_rate"],
               "C3: Rolling IC Positive Rate (temporal stability)",
               [0.50, 0.55, 0.60, 0.65, 0.70])

    print_dist(qr["p_value"],
               "C4: IC p-value (statistical significance)",
               [0.01, 0.05, 0.10], higher_is_better=False)

    print_dist(qr["monotonicity_score"],
               "C5: Monotonicity Score (tercile spread quality)",
               [0.6, 0.7, 0.8, 0.9, 1.0])

    print_dist(qr["quality_score"],
               "C6: Composite Quality Score",
               [2.0, 3.0, 4.0, 5.0])

    # ── Multi-gate analysis ──
    print(f"\n{THIN}")
    print("  C7: Multi-Gate Pass Rates")
    print(THIN)

    gates = pd.DataFrame({
        "ic_sig":       (qr["p_value"] < 0.05).values,
        "ic_stable":    (qr["rolling_ic_positive_rate"] >= 0.55).values,
        "rank_autocorr": (qr["rank_autocorrelation"] >= 0.7).values,
        "nan_ok":       (qr["nan_rate"] <= 0.20).values,
        "monotonic":    (qr["monotonicity_score"] >= 0.8).values,
    }, index=qr["factor_name"].values)

    print(f"\n  Individual gates (out of {len(gates)}):")
    for col in gates.columns:
        n = int(gates[col].sum())
        print(f"    {col:20s}: {n:4d} ({n/len(gates)*100:5.1f}%)")

    all_pass = gates.all(axis=1)
    print(f"    {'ALL 5':20s}: {int(all_pass.sum()):4d} ({all_pass.mean()*100:5.1f}%)")

    # Cascade elimination (most selective first)
    print(f"\n  Cascade elimination order:")
    remaining = pd.Series(True, index=gates.index)
    gate_order = ["nan_ok", "ic_sig", "monotonic", "ic_stable", "rank_autocorr"]
    for g in gate_order:
        remaining = remaining & gates[g]
        print(f"    + {g:20s}: {int(remaining.sum()):4d} remain")

    return gates


# ── Section A: Orthogonality with active 17 ──────────────────

def section_a(qr: pd.DataFrame, registry: dict) -> pd.Series:
    print(f"\n{SEP}")
    print("  A) ORTHOGONALITY WITH ACTIVE 17 FACTORS")
    print(SEP)

    from etf_strategy.core.cross_section_processor import CrossSectionProcessor
    from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary

    # ── Load OHLCV directly (bypass DataLoader amount-column bug) ──
    print("\n  Loading OHLCV data...")
    t0 = time.time()
    data_dir = PROJECT / "raw" / "ETF" / "daily"
    parquet_files = sorted(data_dir.glob("*_daily_*.parquet"))

    dicts = {k: {} for k in ["close", "high", "low", "open", "volume", "amount"]}
    col_map = {"close": "adj_close", "high": "adj_high",
               "low": "adj_low", "open": "adj_open"}

    for fpath in parquet_files:
        code = fpath.stem.split("_")[0].split(".")[0]
        df = pd.read_parquet(fpath)
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        df = df.set_index("trade_date")
        df = df["2020-01-01":"2025-12-31"]
        for target, source in col_map.items():
            if source in df.columns:
                dicts[target][code] = df[source]
        vol_col = "vol" if "vol" in df.columns else "volume"
        if vol_col in df.columns:
            dicts["volume"][code] = df[vol_col]
        if "amount" in df.columns:
            dicts["amount"][code] = df["amount"]

    ohlcv = {}
    for col in ["close", "high", "low", "open", "volume", "amount"]:
        if dicts[col]:
            frame = pd.DataFrame(dicts[col])
            ohlcv[col] = frame[sorted(frame.columns)]

    close = ohlcv["close"]
    print(f"  {close.shape[1]} ETFs × {len(close)} days  ({time.time()-t0:.1f}s)")

    # ── Compute 38 base factors ──
    print("  Computing base factors...")
    t0 = time.time()
    lib = PreciseFactorLibrary()
    raw = lib.compute_all_factors(prices=ohlcv)
    factor_names = sorted(lib.list_factors().keys())

    base_factors = {}
    for name in factor_names:
        base_factors[name] = raw[name]
    print(f"  {len(base_factors)} base factors  ({time.time()-t0:.1f}s)")

    # ── Standardize ──
    print("  Standardizing (CrossSectionProcessor)...")
    t0 = time.time()
    proc = CrossSectionProcessor(verbose=False)
    std_factors = proc.process_all_factors(base_factors)
    print(f"  Done  ({time.time()-t0:.1f}s)")

    # ── Compute algebraic factors ──
    print("  Computing algebraic factors...")
    t0 = time.time()
    op_map = {
        "add": np.add,
        "sub": np.subtract,
        "mul": np.multiply,
        "max": np.maximum,
        "min": np.minimum,
    }

    all_factors = dict(std_factors)
    n_alg = 0
    n_skip = 0

    for name, info in sorted(registry.items()):
        if info["source"] != "algebraic":
            continue
        parents = info.get("parent_factors", [])
        if len(parents) != 2:
            n_skip += 1
            continue

        p_a, p_b = parents
        if p_a not in std_factors or p_b not in std_factors:
            n_skip += 1
            continue

        # Parse operator from name
        op_key = None
        for oname in op_map:
            if f"__{oname}__" in name:
                op_key = oname
                break
        if op_key is None:
            n_skip += 1
            continue

        all_factors[name] = op_map[op_key](std_factors[p_a], std_factors[p_b])
        n_alg += 1

    print(f"  {n_alg} algebraic computed, {n_skip} skipped  ({time.time()-t0:.1f}s)")
    print(f"  Total: {len(all_factors)} factors")

    # ── Compute rank correlations ──
    print("  Computing cross-sectional rank correlations with active 17...")
    t0 = time.time()

    # Common dates/symbols from close (all factors share this universe)
    common_dates = close.index
    common_symbols = close.columns

    T = len(common_dates)
    S = len(common_symbols)

    all_names = sorted(all_factors.keys())
    N = len(all_names)
    K = len(ACTIVE_17)

    # Build ranked 3D array: (T, S, N)
    # Use pandas .rank(axis=1) for vectorized per-date ranking
    print(f"    Ranking {N} factors × {T} dates × {S} symbols...")
    all_ranks = np.full((T, S, N), np.nan, dtype=np.float32)
    mid_rank = (S + 1) / 2.0  # neutral fill for NaN

    for i, name in enumerate(all_names):
        df = all_factors[name]
        # Reindex to common universe
        aligned = df.reindex(index=common_dates, columns=common_symbols)
        ranked = aligned.rank(axis=1, method="average", na_option="keep")
        # Fill NaN with mid-rank (neutral — no correlation contribution)
        ranked = ranked.fillna(mid_rank)
        all_ranks[:, :, i] = ranked.values
        if (i + 1) % 100 == 0:
            print(f"      {i+1}/{N}...")

    # Active factor indices
    active_idx = [all_names.index(n) for n in ACTIVE_17 if n in all_names]
    assert len(active_idx) == K, f"Found {len(active_idx)}/{K} active factors"

    # Vectorized correlation: for each date t, compute (N, K) correlation matrix
    # Then average over T
    print(f"    Computing correlations ({T} dates)...")

    corr_sum = np.zeros((N, K), dtype=np.float64)
    for t in range(T):
        # All factor ranks for this date: (S, N)
        r_all = all_ranks[t, :, :]       # (S, N)
        r_act = r_all[:, active_idx]      # (S, K)

        # Center (subtract column mean)
        r_all_c = r_all - r_all.mean(axis=0, keepdims=True)
        r_act_c = r_act - r_act.mean(axis=0, keepdims=True)

        # Normalize
        r_all_n = np.sqrt(np.sum(r_all_c ** 2, axis=0, keepdims=True))
        r_act_n = np.sqrt(np.sum(r_act_c ** 2, axis=0, keepdims=True))

        r_all_c = np.where(r_all_n > 1e-10, r_all_c / r_all_n, 0)
        r_act_c = np.where(r_act_n > 1e-10, r_act_c / r_act_n, 0)

        # Cross-correlation: (N, K)
        corr_t = r_all_c.T @ r_act_c
        corr_sum += corr_t

    mean_corr = corr_sum / T  # (N, K)
    elapsed = time.time() - t0
    print(f"  Correlation computed  ({elapsed:.1f}s)")

    # ── Results ──
    # max |corr| with active 17 for each factor
    max_abs_corr = np.max(np.abs(mean_corr), axis=1)  # (N,)
    mac_series = pd.Series(max_abs_corr, index=all_names)

    # For active factors, exclude self-correlation
    for j, act_name in enumerate(ACTIVE_17):
        if act_name in all_names:
            i = all_names.index(act_name)
            row = np.abs(mean_corr[i, :])
            row[j] = 0  # zero out self
            mac_series[act_name] = row.max()

    print_dist(mac_series, "A1: Max |rank_corr| with Active 17",
               [0.3, 0.4, 0.5, 0.6, 0.7, 0.8], higher_is_better=False)

    # By source
    hc_names = [n for n in all_names if registry.get(n, {}).get("source") == "hand_crafted"]
    alg_names = [n for n in all_names if registry.get(n, {}).get("source") == "algebraic"]

    hc_mac = mac_series[hc_names].dropna()
    alg_mac = mac_series[alg_names].dropna()
    print(f"\n  By source:")
    print(f"    Hand-crafted (n={len(hc_mac)}):"
          f"  mean={hc_mac.mean():.3f}  p50={hc_mac.median():.3f}  p90={hc_mac.quantile(0.9):.3f}")
    print(f"    Algebraic    (n={len(alg_mac)}):"
          f"  mean={alg_mac.mean():.3f}  p50={alg_mac.median():.3f}  p90={alg_mac.quantile(0.9):.3f}")

    # Top 20 most orthogonal (excluding active 17)
    non_active = mac_series.drop(labels=ACTIVE_17, errors="ignore")
    top_orth = non_active.nsmallest(20)
    print(f"\n  Top 20 most ORTHOGONAL non-active factors:")
    for name, val in top_orth.items():
        src = registry.get(name, {}).get("source", "?")
        qs = registry.get(name, {}).get("quality_score", 0)
        print(f"    {name:50s} max|r|={val:.3f}  qs={qs:.1f}  src={src}")

    # Top 20 most REDUNDANT
    top_red = non_active.nlargest(20)
    print(f"\n  Top 20 most REDUNDANT non-active factors:")
    for name, val in top_red.items():
        src = registry.get(name, {}).get("source", "?")
        qs = registry.get(name, {}).get("quality_score", 0)
        print(f"    {name:50s} max|r|={val:.3f}  qs={qs:.1f}  src={src}")

    # Active 17 inter-correlations (reference)
    print(f"\n  Active 17 max|corr| with OTHER active factors (reference):")
    for j, act_name in enumerate(ACTIVE_17):
        if act_name in all_names:
            i = all_names.index(act_name)
            row = np.abs(mean_corr[i, :])
            row[j] = 0  # exclude self
            max_other = row.max()
            best_j = row.argmax()
            best_name = ACTIVE_17[best_j]
            print(f"    {act_name:35s} max|r|={max_other:.3f}  (vs {best_name})")

    return mac_series


# ── Section B: Bucket compatibility ──────────────────────────

def section_b(qr: pd.DataFrame, registry: dict) -> dict:
    print(f"\n{SEP}")
    print("  B) BUCKET COMPATIBILITY")
    print(SEP)

    from etf_strategy.core.factor_buckets import FACTOR_TO_BUCKET

    # Assign buckets to ALL factors
    bucket_map = {}  # name → set of bucket names
    for name, info in registry.items():
        if name in FACTOR_TO_BUCKET:
            bucket_map[name] = {FACTOR_TO_BUCKET[name]}
        elif info["source"] == "algebraic":
            parents = info.get("parent_factors", [])
            parent_buckets = set()
            for p in parents:
                if p in FACTOR_TO_BUCKET:
                    parent_buckets.add(FACTOR_TO_BUCKET[p])
            bucket_map[name] = parent_buckets if parent_buckets else {"UNMAPPED"}
        else:
            bucket_map[name] = {"UNMAPPED"}

    # Bucket counts (a factor can appear in multiple buckets if cross-bucket)
    print("\n  B1: Bucket assignment for ALL 536 factors")
    bucket_counts = Counter()
    for buckets in bucket_map.values():
        for b in buckets:
            bucket_counts[b] += 1

    for bucket in sorted(bucket_counts.keys()):
        print(f"    {bucket:30s}: {bucket_counts[bucket]:4d}")

    # Single vs cross-bucket
    cross = sum(1 for b in bucket_map.values() if len(b) > 1)
    single = sum(1 for b in bucket_map.values() if len(b) == 1)
    print(f"\n    Single-bucket: {single}  Cross-bucket: {cross}")

    # For quality-passed factors only
    passed_names = set(qr[qr["passed"]]["factor_name"])

    print(f"\n  B2: Bucket distribution (quality-PASSED, n={len(passed_names)}):")
    bucket_counts_passed = Counter()
    for name in passed_names:
        if name in bucket_map:
            for b in bucket_map[name]:
                bucket_counts_passed[b] += 1

    for bucket in sorted(bucket_counts.keys()):
        total = bucket_counts[bucket]
        passed = bucket_counts_passed.get(bucket, 0)
        print(f"    {bucket:30s}: {passed:4d} / {total} "
              f"({passed/total*100:.0f}% pass)")

    # Per-bucket primary assignment (for combo estimation)
    # Algebraic factor → assign to "primary" bucket (first parent's bucket)
    primary_bucket = {}
    for name, buckets in bucket_map.items():
        if len(buckets) == 1:
            primary_bucket[name] = next(iter(buckets))
        elif len(buckets) > 1:
            # Cross-bucket: assign to least-populated bucket (diversify)
            primary_bucket[name] = min(buckets, key=lambda b: bucket_counts[b])
        else:
            primary_bucket[name] = "UNMAPPED"

    # For passed factors, group by primary bucket
    print(f"\n  B3: Per-bucket counts (passed factors, primary assignment):")
    by_bucket = defaultdict(list)
    for name in passed_names:
        b = primary_bucket.get(name, "UNMAPPED")
        by_bucket[b].append(name)

    for bucket in sorted(by_bucket.keys()):
        n = len(by_bucket[bucket])
        print(f"    {bucket:30s}: {n:4d}")

    # Diversity estimate: how many 4-combos from passed factors hit ≥3 buckets?
    print(f"\n  B4: Bucket diversity with NEW factors")
    # Count new factors (non-active-17) per bucket
    for bucket in sorted(by_bucket.keys()):
        new = [n for n in by_bucket[bucket] if n not in ACTIVE_17]
        existing = [n for n in by_bucket[bucket] if n in ACTIVE_17]
        print(f"    {bucket:30s}: {len(existing)} active + {len(new)} new")

    return bucket_map, primary_bucket


# ── Combined analysis ────────────────────────────────────────

def combined_analysis(qr: pd.DataFrame, mac_series: pd.Series,
                      registry: dict, primary_bucket: dict):
    print(f"\n{SEP}")
    print("  Σ) COMBINED MULTI-GATE CASCADE")
    print(SEP)

    df = qr.set_index("factor_name").copy()
    df["max_abs_corr_17"] = df.index.map(mac_series)

    # Exclude active 17 from candidate analysis
    candidates = df.drop(labels=[n for n in ACTIVE_17 if n in df.index],
                         errors="ignore")
    print(f"\n  Candidates (excl. active 17): {len(candidates)}")

    # Define gates with varying orthogonality thresholds
    for orth_thresh in [0.5, 0.6, 0.7, 0.8]:
        gates = pd.DataFrame({
            "ic_sig":        candidates["p_value"] < 0.05,
            "ic_stable":     candidates["rolling_ic_positive_rate"] >= 0.55,
            "rank_autocorr": candidates["rank_autocorrelation"] >= 0.7,
            "nan_ok":        candidates["nan_rate"] <= 0.20,
            "monotonic":     candidates["monotonicity_score"] >= 0.8,
            "orthogonal":    candidates["max_abs_corr_17"] <= orth_thresh,
        })

        all_pass = gates.all(axis=1)
        n_pass = int(all_pass.sum())
        print(f"\n  Orth ≤ {orth_thresh:.1f}:  {n_pass:3d} survivors  "
              f"({n_pass/len(candidates)*100:.1f}%)")

        if n_pass > 0 and n_pass <= 80:
            # Show bucket distribution of survivors
            survivor_names = all_pass[all_pass].index.tolist()
            survivor_buckets = Counter()
            for name in survivor_names:
                b = primary_bucket.get(name, "UNMAPPED")
                survivor_buckets[b] += 1
            bucket_str = ", ".join(f"{k}={v}" for k, v in sorted(survivor_buckets.items()))
            print(f"    Buckets: {bucket_str}")

    # ── Recommended threshold analysis ──
    print(f"\n{THIN}")
    print("  RECOMMENDED THRESHOLDS")
    print(THIN)

    # Use orth <= 0.6 as baseline recommendation
    gates = pd.DataFrame({
        "ic_sig":        candidates["p_value"] < 0.05,
        "ic_stable":     candidates["rolling_ic_positive_rate"] >= 0.55,
        "rank_autocorr": candidates["rank_autocorrelation"] >= 0.7,
        "nan_ok":        candidates["nan_rate"] <= 0.20,
        "monotonic":     candidates["monotonicity_score"] >= 0.8,
        "orthogonal":    candidates["max_abs_corr_17"] <= 0.6,
    })

    # Cascade with each gate's marginal elimination
    print(f"\n  Cascade (starting from {len(candidates)} non-active candidates):")
    remaining = pd.Series(True, index=candidates.index)
    gate_order = ["nan_ok", "ic_sig", "monotonic", "rank_autocorr",
                  "ic_stable", "orthogonal"]
    for g in gate_order:
        before = int(remaining.sum())
        remaining = remaining & gates[g]
        after = int(remaining.sum())
        eliminated = before - after
        print(f"    + {g:20s}: {after:4d} remain  (-{eliminated})")

    # ── List survivors ──
    survivors = remaining[remaining].index.tolist()
    print(f"\n  SURVIVORS ({len(survivors)}):")
    print(f"  {'Name':50s} {'QS':>4s} {'IC':>7s} {'p':>6s} {'RollIC+':>7s} "
          f"{'RkAuto':>6s} {'Orth':>6s} {'Bucket':>20s}")
    print(f"  {'-'*50} {'----':>4s} {'-------':>7s} {'------':>6s} "
          f"{'-------':>7s} {'------':>6s} {'------':>6s} {'-'*20:>20s}")

    for name in sorted(survivors):
        row = candidates.loc[name]
        qs = row.get("quality_score", 0)
        ic = row.get("mean_ic", 0)
        pv = row.get("p_value", 1)
        rip = row.get("rolling_ic_positive_rate", 0)
        ra = row.get("rank_autocorrelation", 0)
        orth = row.get("max_abs_corr_17", 1)
        bucket = primary_bucket.get(name, "?")
        print(f"  {name:50s} {qs:4.1f} {ic:+.4f} {pv:.4f} {rip:.3f}   "
              f"{ra:.3f}  {orth:.3f}  {bucket}")

    return survivors


# ── Main ─────────────────────────────────────────────────────

def main():
    t_start = time.time()

    print(f"\n{SEP}")
    print("  PHASE-0: FACTOR MINING PRE-FILTER DIAGNOSTIC")
    print(f"  536 candidates → optimal gate thresholds → WFO-ready shortlist")
    print(f"{SEP}")

    # Load existing results
    qr = pd.read_parquet(QUALITY_PARQUET)
    with open(REGISTRY_JSON) as f:
        registry = json.load(f)

    n_hc = sum(1 for v in registry.values() if v["source"] == "hand_crafted")
    n_alg = sum(1 for v in registry.values() if v["source"] == "algebraic")
    n_passed = int(qr["passed"].sum())
    print(f"\n  Registry: {len(registry)} factors ({n_hc} hand-crafted + {n_alg} algebraic)")
    print(f"  Quality-passed: {n_passed} / {len(qr)}")

    # Run sections
    gates = section_c(qr)
    mac_series = section_a(qr, registry)
    bucket_map, primary_bucket = section_b(qr, registry)
    survivors = combined_analysis(qr, mac_series, registry, primary_bucket)

    elapsed = time.time() - t_start
    print(f"\n{SEP}")
    print(f"  Done in {elapsed:.0f}s. Survivors: {len(survivors)} factors")
    print(f"{SEP}")


if __name__ == "__main__":
    main()
