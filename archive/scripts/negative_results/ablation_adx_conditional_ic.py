#!/usr/bin/env python3
"""Phase 0: ADX Conditional IC Ablation — Regime-Dependent Factor Predictability

Hypothesis: ADX_14D has regime-dependent IC (positive in low-vol, zero/negative
in mid/high-vol environments). If confirmed, ADX contributes alpha only in calm
markets and adds noise when volatility is elevated.

Methodology:
  1. Classify each trading day into a volatility regime using the production
     regime gate (510300 20D realized vol, thresholds 25/30/40 pct).
  2. On each FREQ=5 rebalance day, compute:
       - Rank IC (Spearman) between factor cross-section and forward 5D/10D returns
       - Top-bottom quartile spread (mean return of Q4 minus Q1)
  3. Aggregate IC stats per regime bucket: mean, std, t-stat, positive rate.
  4. Compare ADX_14D against S1 peers (OBV_SLOPE_10D, SHARPE_RATIO_20D, SLOPE_20D).

Usage:
    uv run python scripts/ablation_adx_conditional_ic.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats as sp_stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Project paths ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.factor_cache import FactorCache
from etf_strategy.regime_gate import compute_volatility_gate_raw

# ── Constants ─────────────────────────────────────────────────────────────
FREQ = 5
LOOKBACK = 252

# Factors to analyse (S1 production set)
TARGET_FACTORS = [
    "ADX_14D",
    "OBV_SLOPE_10D",
    "SHARPE_RATIO_20D",
    "SLOPE_20D",
]

# Forward-return horizons
FWD_HORIZONS = [5, 10]

# Regime gate exposure values -> human-readable bucket names
GATE_BUCKET_MAP = {
    1.0: "low_vol",
    0.7: "mid_vol",
    0.4: "high_vol",
    0.1: "extreme_vol",
}

SEP = "=" * 90
THIN = "-" * 90


# ── Helpers ───────────────────────────────────────────────────────────────

def classify_regime(gate_val: float) -> str:
    """Map a raw (unshifted) gate exposure value to a regime bucket name."""
    # Match to the closest known bucket value
    best_name = "unknown"
    best_dist = float("inf")
    for val, name in GATE_BUCKET_MAP.items():
        dist = abs(gate_val - val)
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name


def compute_regime_series(
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    backtest_config: dict,
) -> pd.Series:
    """Compute the RAW (unshifted) regime gate per date, mapped to bucket names.

    We use the raw gate (no shift) so that the regime label at date t reflects
    volatility conditions known up to t (the shift is for trading signals, not
    for analytical labelling).
    """
    vol_cfg = backtest_config.get("regime_gate", {}).get("volatility", {})
    proxy_symbol = str(vol_cfg.get("proxy_symbol", "510300"))
    window = int(vol_cfg.get("window", 20))
    thresholds_pct = tuple(vol_cfg.get("thresholds_pct", [25, 30, 40]))
    exposures = tuple(vol_cfg.get("exposures", [1.0, 0.7, 0.4, 0.1]))

    raw_gate = compute_volatility_gate_raw(
        close_df,
        proxy_symbol=proxy_symbol,
        window=window,
        thresholds_pct=thresholds_pct,
        exposures=exposures,
    )

    # Align to target dates and classify
    raw_gate = raw_gate.reindex(dates).fillna(float(exposures[0]))
    regime_labels = raw_gate.apply(classify_regime)
    return regime_labels


def spearman_ic(factor_vals: np.ndarray, ret_vals: np.ndarray) -> float:
    """Cross-sectional Spearman IC for a single date. Returns NaN if too few obs."""
    mask = np.isfinite(factor_vals) & np.isfinite(ret_vals)
    n_valid = int(mask.sum())
    if n_valid < 5:
        return np.nan
    corr, _ = sp_stats.spearmanr(factor_vals[mask], ret_vals[mask])
    return corr if np.isfinite(corr) else np.nan


def quartile_spread(
    factor_vals: np.ndarray, ret_vals: np.ndarray
) -> float:
    """Top-quartile minus bottom-quartile mean return, conditioned on factor rank.

    Returns NaN if too few observations.
    """
    mask = np.isfinite(factor_vals) & np.isfinite(ret_vals)
    n_valid = int(mask.sum())
    if n_valid < 8:  # need at least 2 per quartile
        return np.nan

    fv = factor_vals[mask]
    rv = ret_vals[mask]

    q25 = np.percentile(fv, 25)
    q75 = np.percentile(fv, 75)

    bottom_mask = fv <= q25
    top_mask = fv >= q75

    if bottom_mask.sum() < 1 or top_mask.sum() < 1:
        return np.nan

    spread = float(np.mean(rv[top_mask]) - np.mean(rv[bottom_mask]))
    return spread


def ic_summary(ic_array: np.ndarray) -> dict:
    """Compute IC summary statistics from an array of per-date IC values."""
    valid = ic_array[np.isfinite(ic_array)]
    n = len(valid)
    if n < 3:
        return {
            "n_days": n,
            "ic_mean": np.nan,
            "ic_std": np.nan,
            "ic_ir": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "positive_rate": np.nan,
        }

    mean_ic = float(np.mean(valid))
    std_ic = float(np.std(valid, ddof=1))
    ic_ir = mean_ic / std_ic if std_ic > 1e-10 else 0.0
    t_stat = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 1e-10 else 0.0
    p_value = float(2 * (1 - sp_stats.t.cdf(abs(t_stat), df=n - 1)))
    positive_rate = float(np.mean(valid > 0))

    return {
        "n_days": n,
        "ic_mean": mean_ic,
        "ic_std": std_ic,
        "ic_ir": ic_ir,
        "t_stat": t_stat,
        "p_value": p_value,
        "positive_rate": positive_rate,
    }


def significance_stars(p: float) -> str:
    """Return significance stars for a p-value."""
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print(SEP)
    print("  Phase 0: ADX Conditional IC — Regime-Dependent Factor Predictability")
    print(SEP)

    # ── 1. Load config ────────────────────────────────────────────────────
    config_path = ROOT / "configs" / "combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    backtest_config = config.get("backtest", {})

    # ── 2. Load data ──────────────────────────────────────────────────────
    print("\n[1/5] Loading OHLCV data...")
    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )
    close_df = ohlcv["close"]
    print(f"  {len(close_df)} trading days x {len(close_df.columns)} ETFs")
    print(f"  Date range: {close_df.index[0].date()} ~ {close_df.index[-1].date()}")

    # ── 3. Compute factors (cached) ───────────────────────────────────────
    print("\n[2/5] Loading/computing factors...")
    factor_cache = FactorCache(
        cache_dir=Path(config["data"].get("cache_dir") or ".cache")
    )
    cached = factor_cache.get_or_compute(
        ohlcv=ohlcv, config=config, data_dir=loader.data_dir
    )
    factors_3d = cached["factors_3d"]  # (T, N, F)
    factor_names = cached["factor_names"]  # sorted alphabetically
    dates = cached["dates"]
    etf_codes = cached["etf_codes"]

    T, N, F = factors_3d.shape
    print(f"  factors_3d shape: ({T}, {N}, {F})")
    print(f"  Factor names ({F}): {factor_names}")

    # Resolve target factor indices
    factor_idx_map = {name: i for i, name in enumerate(factor_names)}
    target_indices = {}
    for fname in TARGET_FACTORS:
        if fname in factor_idx_map:
            target_indices[fname] = factor_idx_map[fname]
            print(f"    {fname} -> index {factor_idx_map[fname]}")
        else:
            print(f"    WARNING: {fname} not found in factor_names, skipping")

    if not target_indices:
        print("ERROR: No target factors found. Aborting.")
        return

    # ── 4. Compute regime series ──────────────────────────────────────────
    print("\n[3/5] Computing regime classification...")
    regime_series = compute_regime_series(close_df, dates, backtest_config)

    # Regime day counts
    regime_counts = regime_series.value_counts().sort_index()
    print(f"  Regime distribution:")
    for bucket, count in regime_counts.items():
        pct = count / len(regime_series) * 100
        print(f"    {bucket:<12s}: {count:>5d} days ({pct:5.1f}%)")

    # ── 5. Compute forward returns ────────────────────────────────────────
    print("\n[4/5] Computing forward returns...")
    # close_df aligned to dates
    close_aligned = close_df.reindex(dates)[etf_codes].copy()
    close_vals = close_aligned.values  # (T, N)

    fwd_returns = {}
    for horizon in FWD_HORIZONS:
        # fwd_ret[t] = close[t+horizon] / close[t] - 1
        fwd = np.full((T, N), np.nan)
        for t in range(T - horizon):
            c_now = close_vals[t, :]
            c_fut = close_vals[t + horizon, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                fwd[t, :] = c_fut / c_now - 1
        fwd_returns[horizon] = fwd
        n_valid = np.isfinite(fwd).sum()
        print(f"  {horizon}D forward returns: {n_valid:,} valid cells")

    # ── 6. Build rebalance schedule ───────────────────────────────────────
    # Rebalance every FREQ days, starting after LOOKBACK warmup
    rebalance_dates_idx = list(range(LOOKBACK, T, FREQ))
    n_rebal = len(rebalance_dates_idx)
    print(f"  Rebalance days: {n_rebal} (FREQ={FREQ}, LOOKBACK={LOOKBACK})")

    # ── 7. Compute conditional IC + quartile spread ───────────────────────
    print("\n[5/5] Computing conditional IC and quartile spreads...")

    # Storage: results[factor_name][horizon] = dict of regime -> list of IC values
    ic_per_regime = {}  # factor -> horizon -> regime -> [ic_values]
    spread_per_regime = {}  # factor -> horizon -> regime -> [spread_values]

    regime_arr = regime_series.values  # (T,)

    for fname, fidx in target_indices.items():
        ic_per_regime[fname] = {}
        spread_per_regime[fname] = {}
        for horizon in FWD_HORIZONS:
            ic_per_regime[fname][horizon] = {b: [] for b in GATE_BUCKET_MAP.values()}
            spread_per_regime[fname][horizon] = {b: [] for b in GATE_BUCKET_MAP.values()}

    for rb_t in rebalance_dates_idx:
        regime = regime_arr[rb_t]
        if regime not in GATE_BUCKET_MAP.values():
            continue

        for fname, fidx in target_indices.items():
            factor_vals = factors_3d[rb_t, :, fidx]  # (N,)

            for horizon in FWD_HORIZONS:
                if rb_t + horizon >= T:
                    continue
                ret_vals = fwd_returns[horizon][rb_t, :]  # (N,)

                # Rank IC
                ic_val = spearman_ic(factor_vals, ret_vals)
                if np.isfinite(ic_val):
                    ic_per_regime[fname][horizon][regime].append(ic_val)

                # Quartile spread
                qs = quartile_spread(factor_vals, ret_vals)
                if np.isfinite(qs):
                    spread_per_regime[fname][horizon][regime].append(qs)

    # ── 8. Print results ──────────────────────────────────────────────────
    regime_order = ["low_vol", "mid_vol", "high_vol", "extreme_vol"]

    for horizon in FWD_HORIZONS:
        print()
        print(SEP)
        print(f"  RANK IC ANALYSIS — {horizon}D FORWARD RETURNS")
        print(SEP)

        # Header
        print(
            f"\n  {'Factor':<22s} {'Regime':<14s}"
            f" {'N':>5s}"
            f" {'IC_Mean':>8s}"
            f" {'IC_Std':>7s}"
            f" {'t_stat':>7s}"
            f" {'Sig':>4s}"
            f" {'IC_IR':>7s}"
            f" {'PosRate':>8s}"
            f" {'Q_Spread':>9s}"
        )
        print(f"  {'─'*22} {'─'*14} {'─'*5} {'─'*8} {'─'*7} {'─'*7} {'─'*4} {'─'*7} {'─'*8} {'─'*9}")

        for fname in TARGET_FACTORS:
            if fname not in target_indices:
                continue

            for i, regime in enumerate(regime_order):
                ic_vals = np.array(ic_per_regime[fname][horizon][regime])
                spread_vals = np.array(spread_per_regime[fname][horizon][regime])

                summary = ic_summary(ic_vals)
                mean_spread = float(np.mean(spread_vals)) if len(spread_vals) > 0 else np.nan

                # Format factor name only on first regime row
                fname_display = fname if i == 0 else ""
                stars = significance_stars(summary["p_value"])

                ic_mean_str = f"{summary['ic_mean']:>+8.4f}" if np.isfinite(summary["ic_mean"]) else f"{'N/A':>8s}"
                ic_std_str = f"{summary['ic_std']:>7.4f}" if np.isfinite(summary["ic_std"]) else f"{'N/A':>7s}"
                t_str = f"{summary['t_stat']:>+7.2f}" if np.isfinite(summary["t_stat"]) else f"{'N/A':>7s}"
                ir_str = f"{summary['ic_ir']:>7.3f}" if np.isfinite(summary["ic_ir"]) else f"{'N/A':>7s}"
                pos_str = f"{summary['positive_rate']:>7.1%}" if np.isfinite(summary["positive_rate"]) else f"{'N/A':>7s}"
                spr_str = f"{mean_spread:>+9.4f}" if np.isfinite(mean_spread) else f"{'N/A':>9s}"

                print(
                    f"  {fname_display:<22s} {regime:<14s}"
                    f" {summary['n_days']:>5d}"
                    f" {ic_mean_str}"
                    f" {ic_std_str}"
                    f" {t_str}"
                    f" {stars:>4s}"
                    f" {ir_str}"
                    f" {pos_str}"
                    f" {spr_str}"
                )

            # Separator between factors
            print(f"  {'─'*22} {'─'*14} {'─'*5} {'─'*8} {'─'*7} {'─'*7} {'─'*4} {'─'*7} {'─'*8} {'─'*9}")

    # ── 9. Cross-regime IC comparison (compact) ───────────────────────────
    print()
    print(SEP)
    print("  CROSS-REGIME IC COMPARISON (IC Mean by Factor x Regime)")
    print(SEP)

    for horizon in FWD_HORIZONS:
        print(f"\n  {horizon}D Forward Returns:")
        header_parts = [f"  {'Factor':<22s}"]
        for regime in regime_order:
            header_parts.append(f" {regime:>12s}")
        header_parts.append(f" {'All':>10s}")
        header_parts.append(f" {'LowVol-HiVol':>13s}")
        print("".join(header_parts))
        print(f"  {'─'*22}" + " ".join([f"{'─'*12}"] * 4) + f" {'─'*10} {'─'*13}")

        for fname in TARGET_FACTORS:
            if fname not in target_indices:
                continue

            parts = [f"  {fname:<22s}"]
            regime_means = {}
            all_ics = []

            for regime in regime_order:
                ic_vals = np.array(ic_per_regime[fname][horizon][regime])
                valid = ic_vals[np.isfinite(ic_vals)]
                if len(valid) > 0:
                    m = float(np.mean(valid))
                    regime_means[regime] = m
                    all_ics.extend(valid.tolist())
                    stars = significance_stars(ic_summary(ic_vals)["p_value"])
                    parts.append(f" {m:>+10.4f}{stars:>2s}")
                else:
                    regime_means[regime] = np.nan
                    parts.append(f" {'N/A':>12s}")

            # All-regime mean
            if all_ics:
                all_mean = float(np.mean(all_ics))
                parts.append(f" {all_mean:>+10.4f}")
            else:
                parts.append(f" {'N/A':>10s}")

            # Low-vol minus high-vol delta
            low_m = regime_means.get("low_vol", np.nan)
            high_m = regime_means.get("high_vol", np.nan)
            if np.isfinite(low_m) and np.isfinite(high_m):
                delta = low_m - high_m
                parts.append(f" {delta:>+13.4f}")
            else:
                parts.append(f" {'N/A':>13s}")

            print("".join(parts))

    # ── 10. Quartile spread comparison ────────────────────────────────────
    print()
    print(SEP)
    print("  QUARTILE SPREAD COMPARISON (Top Q4 - Bottom Q1 Mean Return)")
    print(SEP)

    for horizon in FWD_HORIZONS:
        print(f"\n  {horizon}D Forward Returns:")
        header_parts = [f"  {'Factor':<22s}"]
        for regime in regime_order:
            header_parts.append(f" {regime:>12s}")
        header_parts.append(f" {'All':>12s}")
        print("".join(header_parts))
        print(f"  {'─'*22}" + " ".join([f"{'─'*12}"] * 4) + f" {'─'*12}")

        for fname in TARGET_FACTORS:
            if fname not in target_indices:
                continue

            parts = [f"  {fname:<22s}"]
            all_spreads = []

            for regime in regime_order:
                sp_vals = np.array(spread_per_regime[fname][horizon][regime])
                valid = sp_vals[np.isfinite(sp_vals)]
                if len(valid) > 0:
                    m = float(np.mean(valid))
                    all_spreads.extend(valid.tolist())
                    parts.append(f" {m*100:>+11.3f}%")
                else:
                    parts.append(f" {'N/A':>12s}")

            # All regimes
            if all_spreads:
                parts.append(f" {np.mean(all_spreads)*100:>+11.3f}%")
            else:
                parts.append(f" {'N/A':>12s}")

            print("".join(parts))

    # ── 11. ADX sign-flip detection ───────────────────────────────────────
    print()
    print(SEP)
    print("  ADX_14D REGIME SIGN-FLIP DETECTION")
    print(SEP)

    if "ADX_14D" in target_indices:
        for horizon in FWD_HORIZONS:
            print(f"\n  {horizon}D horizon:")
            sign_changes = []
            prev_sign = None
            for regime in regime_order:
                ic_vals = np.array(ic_per_regime["ADX_14D"][horizon][regime])
                valid = ic_vals[np.isfinite(ic_vals)]
                if len(valid) == 0:
                    sign_str = "N/A"
                    curr_sign = None
                else:
                    m = float(np.mean(valid))
                    summary = ic_summary(ic_vals)
                    is_sig = summary["p_value"] < 0.10 if np.isfinite(summary["p_value"]) else False
                    if m > 0.005 and is_sig:
                        curr_sign = "+"
                    elif m < -0.005 and is_sig:
                        curr_sign = "-"
                    else:
                        curr_sign = "0"
                    sign_str = f"IC={m:+.4f} (sign={curr_sign}, p={summary['p_value']:.3f})"

                flip_marker = ""
                if prev_sign is not None and curr_sign is not None:
                    if prev_sign != curr_sign and prev_sign != "0" and curr_sign != "0":
                        flip_marker = " *** SIGN FLIP ***"
                    elif prev_sign != curr_sign:
                        flip_marker = " (IC weakens/disappears)"

                print(f"    {regime:<14s}: {sign_str}{flip_marker}")
                prev_sign = curr_sign

    # ── 12. Summary verdict ───────────────────────────────────────────────
    print()
    print(SEP)
    print("  VERDICT")
    print(SEP)

    if "ADX_14D" in target_indices:
        print("\n  ADX_14D regime-dependent IC pattern:")
        for horizon in FWD_HORIZONS:
            low_ics = np.array(ic_per_regime["ADX_14D"][horizon]["low_vol"])
            low_summary = ic_summary(low_ics)
            mid_ics = np.array(ic_per_regime["ADX_14D"][horizon]["mid_vol"])
            mid_summary = ic_summary(mid_ics)
            high_ics = np.array(ic_per_regime["ADX_14D"][horizon]["high_vol"])
            high_summary = ic_summary(high_ics)
            ext_ics = np.array(ic_per_regime["ADX_14D"][horizon]["extreme_vol"])
            ext_summary = ic_summary(ext_ics)

            low_m = low_summary["ic_mean"]
            mid_m = mid_summary["ic_mean"]
            high_m = high_summary["ic_mean"]
            ext_m = ext_summary["ic_mean"]

            print(f"\n    {horizon}D horizon:")
            print(f"      low_vol  IC = {low_m:+.4f}" if np.isfinite(low_m) else "      low_vol  IC = N/A")
            print(f"      mid_vol  IC = {mid_m:+.4f}" if np.isfinite(mid_m) else "      mid_vol  IC = N/A")
            print(f"      high_vol IC = {high_m:+.4f}" if np.isfinite(high_m) else "      high_vol IC = N/A")
            print(f"      extreme  IC = {ext_m:+.4f}" if np.isfinite(ext_m) else "      extreme  IC = N/A")

            # Check hypothesis: positive in low, zero/negative in mid+high
            hypothesis_confirmed = False
            if np.isfinite(low_m) and np.isfinite(high_m):
                low_positive = low_m > 0.01 and low_summary["p_value"] < 0.10
                high_weak = high_m < 0.01 or high_summary["p_value"] > 0.10

                if low_positive and high_weak:
                    hypothesis_confirmed = True
                    print(f"      --> HYPOTHESIS CONFIRMED: ADX has positive IC in low-vol"
                          f" ({low_m:+.4f}, p={low_summary['p_value']:.3f})"
                          f" but weak/zero in high-vol ({high_m:+.4f}, p={high_summary['p_value']:.3f})")
                else:
                    print(f"      --> HYPOTHESIS NOT CONFIRMED:"
                          f" low_vol IC {'positive' if low_m > 0 else 'non-positive'}"
                          f" (p={low_summary['p_value']:.3f}),"
                          f" high_vol IC {'weak' if abs(high_m) < 0.01 else 'present'}"
                          f" (p={high_summary['p_value']:.3f})")

    # Comparison with other factors
    print("\n  Factor robustness across regimes (5D horizon, IC mean):")
    for fname in TARGET_FACTORS:
        if fname not in target_indices:
            continue
        regime_ics = []
        for regime in regime_order:
            ic_vals = np.array(ic_per_regime[fname][5][regime])
            valid = ic_vals[np.isfinite(ic_vals)]
            if len(valid) > 0:
                regime_ics.append(float(np.mean(valid)))
            else:
                regime_ics.append(np.nan)

        valid_ics = [x for x in regime_ics if np.isfinite(x)]
        if len(valid_ics) >= 2:
            ic_range = max(valid_ics) - min(valid_ics)
            sign_consistent = all(x > 0 for x in valid_ics) or all(x <= 0 for x in valid_ics)
            robustness = "ROBUST" if sign_consistent and ic_range < 0.03 else (
                "MODERATE" if sign_consistent else "REGIME-DEPENDENT"
            )
        else:
            ic_range = np.nan
            robustness = "INSUFFICIENT DATA"

        ic_strs = [f"{x:+.4f}" if np.isfinite(x) else "N/A" for x in regime_ics]
        print(f"    {fname:<22s}: [{', '.join(ic_strs)}]  range={ic_range:.4f}  {robustness}")

    print()
    print(SEP)
    print("  Phase 0 ablation complete.")
    print(SEP)
    print()


if __name__ == "__main__":
    main()
