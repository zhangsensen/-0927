"""
Rank Stability Pre-filter (Stage 0)
====================================
Computes cross-sectional rank stability metrics for all factors at FREQ=5
rebalance intervals. Factors with stable ranks are compatible with Exp4
hysteresis (delta_rank=0.10, min_hold_days=9); volatile-rank factors get
"locked" into bad positions.

Metrics:
  - rank_autocorr_5d: Spearman correlation of rank vectors between consecutive
    rebalance days. Averaged over all pairs.
  - top2_persistence: Fraction of top-2 ETFs that remain in top-2 after one
    rebalance period. Averaged over all periods.
  - rank_volatility: Std of each ETF's rank across time, averaged over ETFs.

Gate: rank_autocorr_5d >= 0.65

Usage:
    uv run python scripts/rank_stability_analysis.py
    uv run python scripts/rank_stability_analysis.py --config configs/combo_wfo_config.yaml
    uv run python scripts/rank_stability_analysis.py --npz results/factor_mining_.../survivors_3d.npz

Author: Sensen
Date: 2026-02-12
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# ── Project root ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / "src"))

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.factor_cache import FactorCache

logger = logging.getLogger(__name__)

# ── Known factor groups (for summary validation) ─────────────
S1_FACTORS = ["ADX_14D", "OBV_SLOPE_10D", "SHARPE_RATIO_20D", "SLOPE_20D"]
CHAMPION_FACTORS = ["AMIHUD_ILLIQUIDITY", "PRICE_POSITION_20D", "PV_CORR_20D", "SLOPE_20D"]

GATE_THRESHOLD = 0.65


# ── Core metric computation ──────────────────────────────────

def compute_rank_stability_metrics(
    factor_matrix: np.ndarray,
    rebalance_indices: np.ndarray,
    pos_size: int = 2,
) -> Dict[str, float]:
    """Compute rank stability metrics for a single factor.

    Args:
        factor_matrix: (T, N) array of standardized factor values.
                       Higher = better rank.
        rebalance_indices: 1D array of rebalance day indices into T-axis.
        pos_size: Number of top positions (for persistence metric).

    Returns:
        {rank_autocorr_5d, top2_persistence, rank_volatility}
    """
    T, N = factor_matrix.shape

    # Filter rebalance indices within bounds
    reb_idx = rebalance_indices[rebalance_indices < T]
    if len(reb_idx) < 2:
        return {
            "rank_autocorr_5d": np.nan,
            "top2_persistence": np.nan,
            "rank_volatility": np.nan,
        }

    # ── Compute cross-sectional ranks at each rebalance day ──
    # ranks: (n_rebalance, N) — rank 1 = lowest, rank N = highest
    ranks_list = []
    for idx in reb_idx:
        row = factor_matrix[idx, :]
        # Handle NaN: assign mid-rank to NaN values (they won't be selected)
        valid = ~np.isnan(row)
        rank = np.full(N, np.nan)
        if valid.sum() > 1:
            rank[valid] = stats.rankdata(row[valid], method="average")
        ranks_list.append(rank)
    ranks = np.array(ranks_list)  # (n_reb, N)

    n_reb = len(reb_idx)

    # ── 1. rank_autocorr_5d: Spearman corr between consecutive rebalance days ──
    autocorrs = []
    for i in range(n_reb - 1):
        r1 = ranks[i]
        r2 = ranks[i + 1]
        valid = ~(np.isnan(r1) | np.isnan(r2))
        if valid.sum() > 3:
            corr, _ = stats.spearmanr(r1[valid], r2[valid])
            if not np.isnan(corr):
                autocorrs.append(corr)
    rank_autocorr = float(np.mean(autocorrs)) if autocorrs else np.nan

    # ── 2. top2_persistence: fraction of top-K that remain in top-K ──
    persistences = []
    for i in range(n_reb - 1):
        r1 = ranks[i]
        r2 = ranks[i + 1]
        valid1 = ~np.isnan(r1)
        valid2 = ~np.isnan(r2)
        if valid1.sum() >= pos_size and valid2.sum() >= pos_size:
            # Top-K = highest ranks
            top_k_prev = set(np.argsort(r1)[-pos_size:])
            top_k_curr = set(np.argsort(r2)[-pos_size:])
            overlap = len(top_k_prev & top_k_curr)
            persistences.append(overlap / pos_size)
    top2_persistence = float(np.mean(persistences)) if persistences else np.nan

    # ── 3. rank_volatility: std of each ETF's rank across time, avg over ETFs ──
    # Use rank01 (normalized to [0,1]) for comparability
    n_valid = np.sum(~np.isnan(ranks), axis=1, keepdims=True)  # per rebalance
    rank01 = np.where(
        n_valid > 1,
        (ranks - 1) / np.maximum(n_valid - 1, 1),
        np.nan,
    )
    # Per-ETF std across rebalance days, ignoring NaN
    with np.errstate(invalid="ignore"):
        etf_stds = np.nanstd(rank01, axis=0)
    rank_volatility = float(np.nanmean(etf_stds))

    return {
        "rank_autocorr_5d": rank_autocorr,
        "top2_persistence": top2_persistence,
        "rank_volatility": rank_volatility,
    }


# ── Rebalance schedule (mirrors pipeline logic) ──────────────

def build_rebalance_indices(
    total_periods: int,
    lookback: int,
    freq: int,
) -> np.ndarray:
    """Generate rebalance day indices (same convention as pipeline)."""
    from etf_strategy.core.utils.rebalance import generate_rebalance_schedule

    return generate_rebalance_schedule(
        total_periods=total_periods,
        lookback_window=lookback,
        freq=freq,
    )


# ── Load base factors ────────────────────────────────────────

def load_base_factors(
    config: dict,
) -> Tuple[Dict[str, np.ndarray], List[str], int, int]:
    """Load base factors via FactorCache.

    Returns:
        factor_matrices: {factor_name: (T, N) ndarray}
        etf_codes: list of ETF codes
        T: number of trading days
        N: number of ETFs
    """
    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )

    factor_cache = FactorCache(
        cache_dir=Path(config["data"].get("cache_dir") or ".cache")
    )
    cached = factor_cache.get_or_compute(
        ohlcv=ohlcv,
        config=config,
        data_dir=loader.data_dir,
    )

    std_factors = cached["std_factors"]
    factor_names = cached["factor_names"]
    dates = cached["dates"]
    etf_codes = cached["etf_codes"]

    T = len(dates)
    N = len(etf_codes)

    # Convert to {name: (T, N)} dict
    factor_matrices = {}
    for fname in factor_names:
        factor_matrices[fname] = std_factors[fname].values  # (T, N)

    return factor_matrices, etf_codes, T, N


# ── Load algebraic factors from NPZ ──────────────────────────

def load_algebraic_factors_npz(
    npz_path: Path,
    base_etf_codes: List[str],
    base_T: int,
) -> Dict[str, np.ndarray]:
    """Load algebraic factors from survivors_3d.npz.

    Returns:
        {factor_name: (T_aligned, N_aligned) ndarray}
        Aligned to base ETF codes; padded with NaN where dates don't match.
    """
    if not npz_path.exists():
        print(f"  NPZ not found: {npz_path}, skipping algebraic factors")
        return {}

    data = np.load(npz_path)
    extra_names = list(data["factor_names"])
    extra_symbols = list(data["symbols"])
    extra_data = data["data"]  # (T_extra, N_extra, F)

    T_extra, N_extra, F = extra_data.shape
    print(f"  NPZ loaded: {F} factors x {T_extra} dates x {N_extra} symbols")

    # Symbol alignment: reindex to base_etf_codes
    sym_idx = []
    for code in base_etf_codes:
        if code in extra_symbols:
            sym_idx.append(extra_symbols.index(code))
        else:
            sym_idx.append(-1)  # will be NaN

    factor_matrices = {}
    for fi, fname in enumerate(extra_names):
        # Start with NaN-filled array matching base dimensions
        # Use min(T_extra, base_T) for date alignment (trim or pad)
        out = np.full((base_T, len(base_etf_codes)), np.nan, dtype=np.float32)
        T_use = min(T_extra, base_T)
        for ni, si in enumerate(sym_idx):
            if si >= 0:
                out[:T_use, ni] = extra_data[:T_use, si, fi]
        factor_matrices[fname] = out

    return factor_matrices


# ── Find latest NPZ from factor mining results ──────────────

def find_latest_survivors_npz(results_dir: Path) -> Optional[Path]:
    """Find the most recent survivors_3d.npz from factor mining runs."""
    mining_dirs = sorted(results_dir.glob("factor_mining_*"))
    for d in reversed(mining_dirs):
        npz = d / "survivors_3d.npz"
        if npz.exists():
            # Prefer the one with more factors (algebraic, not single-factor)
            data = np.load(npz)
            if len(data["factor_names"]) > 1:
                return npz
    # Fallback: any survivors NPZ
    for d in reversed(mining_dirs):
        npz = d / "survivors_3d.npz"
        if npz.exists():
            return npz
    return None


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Rank Stability Pre-filter (Stage 0)"
    )
    parser.add_argument(
        "--config",
        default="configs/combo_wfo_config.yaml",
        help="Config YAML path",
    )
    parser.add_argument(
        "--npz",
        default=None,
        help="Path to algebraic factors NPZ (auto-detected if omitted)",
    )
    parser.add_argument(
        "--gate",
        type=float,
        default=GATE_THRESHOLD,
        help=f"rank_autocorr gate threshold (default {GATE_THRESHOLD})",
    )
    parser.add_argument(
        "--output",
        default="results/rank_stability_report.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    t0 = time.time()

    # ── 1. Load config ──
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    freq = config.get("backtest", {}).get("freq", 5)
    pos_size = config.get("backtest", {}).get("pos_size", 2)
    lookback = config.get("backtest", {}).get("lookback", 252)
    active_factors = config.get("active_factors", [])

    print("=" * 72)
    print("  RANK STABILITY PRE-FILTER (Stage 0)")
    print("=" * 72)
    print(f"  FREQ={freq}, POS_SIZE={pos_size}, LOOKBACK={lookback}")
    print(f"  Gate: rank_autocorr_5d >= {args.gate}")
    print(f"  Active factors: {len(active_factors)}")
    print()

    # ── 2. Load base factors ──
    print("[1/4] Loading base factors...")
    factor_matrices, etf_codes, T, N = load_base_factors(config)
    print(f"  Base: {len(factor_matrices)} factors x {T} days x {N} ETFs")

    # ── 3. Build rebalance schedule ──
    reb_indices = build_rebalance_indices(T, lookback, freq)
    print(f"  Rebalance days: {len(reb_indices)} (first={reb_indices[0]}, last={reb_indices[-1]})")

    # ── 4. Load algebraic factors ──
    algebraic_matrices = {}
    npz_path = None
    if args.npz:
        npz_path = Path(args.npz)
        if not npz_path.is_absolute():
            npz_path = ROOT / npz_path
    else:
        npz_path = find_latest_survivors_npz(ROOT / "results")

    if npz_path and npz_path.exists():
        print(f"\n[2/4] Loading algebraic factors from {npz_path.name}...")
        algebraic_matrices = load_algebraic_factors_npz(npz_path, etf_codes, T)
        print(f"  Algebraic: {len(algebraic_matrices)} factors loaded")
    else:
        print("\n[2/4] No algebraic factors NPZ found, skipping.")

    # ── 5. Compute metrics ──
    print(f"\n[3/4] Computing rank stability metrics...")
    all_results = []

    # Base factors
    for fname in sorted(factor_matrices.keys()):
        mat = factor_matrices[fname]
        metrics = compute_rank_stability_metrics(mat, reb_indices, pos_size)
        is_active = fname in active_factors
        all_results.append({
            "factor_name": fname,
            "source": "base",
            "is_active": is_active,
            **metrics,
        })

    # Algebraic factors
    n_alg = len(algebraic_matrices)
    for i, (fname, mat) in enumerate(sorted(algebraic_matrices.items())):
        if (i + 1) % 20 == 0 or (i + 1) == n_alg:
            print(f"    Algebraic: {i+1}/{n_alg}")
        metrics = compute_rank_stability_metrics(mat, reb_indices, pos_size)
        all_results.append({
            "factor_name": fname,
            "source": "algebraic",
            "is_active": False,
            **metrics,
        })

    df = pd.DataFrame(all_results)
    df["pass_gate"] = df["rank_autocorr_5d"] >= args.gate

    # ── 6. Save report ──
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, float_format="%.4f")

    elapsed = time.time() - t0
    print(f"\n[4/4] Report saved: {output_path} ({len(df)} factors, {elapsed:.1f}s)")

    # ── 7. Summary ──
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)

    # S1 factors
    s1_df = df[df["factor_name"].isin(S1_FACTORS)]
    if not s1_df.empty:
        print("\n  S1 factors (production baseline):")
        print("  " + "-" * 68)
        print(f"  {'Factor':<30s} {'AutoCorr':>10s} {'TopK_Pers':>10s} {'RankVol':>10s} {'Gate':>6s}")
        print("  " + "-" * 68)
        for _, row in s1_df.sort_values("rank_autocorr_5d", ascending=False).iterrows():
            gate_str = "PASS" if row["pass_gate"] else "FAIL"
            print(
                f"  {row['factor_name']:<30s} "
                f"{row['rank_autocorr_5d']:>10.4f} "
                f"{row['top2_persistence']:>10.4f} "
                f"{row['rank_volatility']:>10.4f} "
                f"{gate_str:>6s}"
            )
        print(
            f"  {'[S1 MEAN]':<30s} "
            f"{s1_df['rank_autocorr_5d'].mean():>10.4f} "
            f"{s1_df['top2_persistence'].mean():>10.4f} "
            f"{s1_df['rank_volatility'].mean():>10.4f}"
        )

    # Champion factors
    champ_df = df[df["factor_name"].isin(CHAMPION_FACTORS)]
    if not champ_df.empty:
        print("\n  Champion factors (Exp4-incompatible reference):")
        print("  " + "-" * 68)
        print(f"  {'Factor':<30s} {'AutoCorr':>10s} {'TopK_Pers':>10s} {'RankVol':>10s} {'Gate':>6s}")
        print("  " + "-" * 68)
        for _, row in champ_df.sort_values("rank_autocorr_5d", ascending=False).iterrows():
            gate_str = "PASS" if row["pass_gate"] else "FAIL"
            print(
                f"  {row['factor_name']:<30s} "
                f"{row['rank_autocorr_5d']:>10.4f} "
                f"{row['top2_persistence']:>10.4f} "
                f"{row['rank_volatility']:>10.4f} "
                f"{gate_str:>6s}"
            )
        print(
            f"  {'[CHAMP MEAN]':<30s} "
            f"{champ_df['rank_autocorr_5d'].mean():>10.4f} "
            f"{champ_df['top2_persistence'].mean():>10.4f} "
            f"{champ_df['rank_volatility'].mean():>10.4f}"
        )

    # S1 vs Champion comparison
    if not s1_df.empty and not champ_df.empty:
        s1_mean = s1_df["rank_autocorr_5d"].mean()
        ch_mean = champ_df["rank_autocorr_5d"].mean()
        delta = s1_mean - ch_mean
        print(f"\n  S1 vs Champion autocorr delta: {delta:+.4f} "
              f"({'S1 more stable' if delta > 0 else 'Champion more stable'})")

    # All active factors
    active_df = df[df["is_active"]]
    if not active_df.empty:
        print(f"\n  All active factors ({len(active_df)}):")
        print("  " + "-" * 68)
        print(f"  {'Factor':<30s} {'AutoCorr':>10s} {'TopK_Pers':>10s} {'RankVol':>10s} {'Gate':>6s}")
        print("  " + "-" * 68)
        for _, row in active_df.sort_values("rank_autocorr_5d", ascending=False).iterrows():
            gate_str = "PASS" if row["pass_gate"] else "FAIL"
            print(
                f"  {row['factor_name']:<30s} "
                f"{row['rank_autocorr_5d']:>10.4f} "
                f"{row['top2_persistence']:>10.4f} "
                f"{row['rank_volatility']:>10.4f} "
                f"{gate_str:>6s}"
            )

    # Algebraic top/bottom 10
    alg_df = df[df["source"] == "algebraic"].dropna(subset=["rank_autocorr_5d"])
    if not alg_df.empty:
        print(f"\n  Algebraic factors: {len(alg_df)} total")

        top10 = alg_df.nlargest(10, "rank_autocorr_5d")
        print("\n  TOP 10 most stable algebraic factors:")
        print("  " + "-" * 68)
        print(f"  {'Factor':<45s} {'AutoCorr':>10s} {'TopK_Pers':>10s} {'Gate':>6s}")
        print("  " + "-" * 68)
        for _, row in top10.iterrows():
            gate_str = "PASS" if row["pass_gate"] else "FAIL"
            name_trunc = row["factor_name"][:44]
            print(
                f"  {name_trunc:<45s} "
                f"{row['rank_autocorr_5d']:>10.4f} "
                f"{row['top2_persistence']:>10.4f} "
                f"{gate_str:>6s}"
            )

        bot10 = alg_df.nsmallest(10, "rank_autocorr_5d")
        print("\n  BOTTOM 10 least stable algebraic factors:")
        print("  " + "-" * 68)
        print(f"  {'Factor':<45s} {'AutoCorr':>10s} {'TopK_Pers':>10s} {'Gate':>6s}")
        print("  " + "-" * 68)
        for _, row in bot10.iterrows():
            gate_str = "PASS" if row["pass_gate"] else "FAIL"
            name_trunc = row["factor_name"][:44]
            print(
                f"  {name_trunc:<45s} "
                f"{row['rank_autocorr_5d']:>10.4f} "
                f"{row['top2_persistence']:>10.4f} "
                f"{gate_str:>6s}"
            )

    # Gate summary
    print("\n  " + "=" * 68)
    n_pass = int(df["pass_gate"].sum())
    n_fail = int((~df["pass_gate"]).sum())
    n_nan = int(df["rank_autocorr_5d"].isna().sum())
    print(f"  GATE SUMMARY (threshold={args.gate}):")
    print(f"    PASS: {n_pass} / {len(df)} ({100*n_pass/max(len(df),1):.1f}%)")
    print(f"    FAIL: {n_fail} / {len(df)} ({100*n_fail/max(len(df),1):.1f}%)")
    if n_nan > 0:
        print(f"    NaN:  {n_nan}")

    # Break down by source
    for src in ["base", "algebraic"]:
        src_df = df[df["source"] == src]
        if not src_df.empty:
            sp = int(src_df["pass_gate"].sum())
            sf = int((~src_df["pass_gate"]).sum())
            print(f"    {src:>12s}: {sp} pass / {sf} fail")

    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()
