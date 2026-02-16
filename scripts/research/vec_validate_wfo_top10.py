#!/usr/bin/env python3
"""
VEC Holdout Validation for WFO Top Combos

Loads top-10 combos from WFO output (by execution_score or cum_oos_return),
adds S1 and C2 baselines, runs VEC backtest with production params
(F5, T1_OPEN, Exp4 hysteresis, regime gate, med cost), and reports
holdout-period (2025-05-01 to end_date) metrics.

Usage:
    uv run python scripts/research/vec_validate_wfo_top10.py [WFO_DIR_OR_PARQUET]

If no argument given, uses the latest results/run_* directory.

Environment variables:
    FROZEN_PARAMS_MODE=warn   (required if pool differs from frozen v5.0)
    WFO_CONFIG_PATH           (optional, override config path)
"""
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

import numpy as np
import pandas as pd
import yaml

from etf_strategy.core.cost_model import build_cost_array, load_cost_model
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.factor_cache import FactorCache
from etf_strategy.core.frozen_params import FrozenETFPool
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import (
    generate_rebalance_schedule,
    shift_timing_signal,
)
from etf_strategy.regime_gate import compute_regime_gate_arr, gate_stats

# Import VEC backtest from batch script
sys.path.insert(0, str(ROOT / "scripts"))
from batch_vec_backtest import run_vec_backtest

# ─────────────────────────────────────────────────────────────────────────────
# Baseline combos
# ─────────────────────────────────────────────────────────────────────────────
S1_COMBO = "ADX_14D + OBV_SLOPE_10D + SHARPE_RATIO_20D + SLOPE_20D"
C2_COMBO = "AMIHUD_ILLIQUIDITY + CALMAR_RATIO_60D + CORRELATION_TO_MARKET_20D"

TOP_N = 10  # Number of WFO combos to validate


def find_wfo_input(arg=None):
    """Locate WFO results: parquet file or run directory."""
    if arg:
        p = Path(arg)
        if not p.is_absolute():
            p = ROOT / p
        if p.is_file() and p.suffix == ".parquet":
            return p
        if p.is_dir():
            # Prefer top_combos, then all_combos, then top100_by_ic
            for name in ["top_combos.parquet", "all_combos.parquet", "top100_by_ic.parquet"]:
                candidate = p / name
                if candidate.exists():
                    return candidate
            raise FileNotFoundError(f"No parquet found in {p}")
        raise FileNotFoundError(f"Not found: {p}")

    # Auto-discover latest run_* directory (non-symlink)
    run_dirs = sorted(
        [
            d
            for d in (ROOT / "results").glob("run_*")
            if d.is_dir() and not d.is_symlink()
        ]
    )
    if not run_dirs:
        raise FileNotFoundError("No results/run_* directories found")
    latest = run_dirs[-1]
    for name in ["top_combos.parquet", "all_combos.parquet", "top100_by_ic.parquet"]:
        candidate = latest / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No parquet found in {latest}")


def load_top_combos(parquet_path, n=TOP_N):
    """Load top-N combos by execution_score (or cum_oos_return fallback)."""
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} combos from {parquet_path.name}")

    # Sort by best available metric
    if "execution_score" in df.columns:
        sort_col = "execution_score"
        print(f"Sorting by: execution_score (hysteresis-aware)")
    elif "cum_oos_return" in df.columns:
        sort_col = "cum_oos_return"
        print(f"Sorting by: cum_oos_return (fallback)")
    elif "mean_oos_return" in df.columns:
        sort_col = "mean_oos_return"
        print(f"Sorting by: mean_oos_return (fallback)")
    else:
        sort_col = "mean_oos_ic"
        print(f"Sorting by: mean_oos_ic (last resort)")

    df = df.sort_values(by=sort_col, ascending=False).reset_index(drop=True)
    top = df.head(n).copy()

    # Add baselines if not already present
    combos = set(top["combo"].tolist())
    baselines = []
    for label, combo_str in [("S1", S1_COMBO), ("C2", C2_COMBO)]:
        if combo_str not in combos:
            # Check if it exists in the full results
            match = df[df["combo"] == combo_str]
            if not match.empty:
                baselines.append(match.iloc[[0]])
                print(f"  Added baseline {label} from WFO results (rank #{match.index[0]+1})")
            else:
                # Create a stub row
                stub = pd.DataFrame(
                    [{
                        "combo": combo_str,
                        sort_col: np.nan,
                        "combo_size": len(combo_str.split(" + ")),
                    }]
                )
                baselines.append(stub)
                print(f"  Added baseline {label} (not in WFO results, stub)")

    if baselines:
        top = pd.concat([top] + baselines, ignore_index=True)

    return top, sort_col


def prepare_data(config):
    """Load OHLCV, compute factors, build timing/regime arrays."""
    symbols = config["data"]["symbols"]

    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=symbols,
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )

    # Compute base factors via FactorCache
    factor_cache = FactorCache(
        cache_dir=Path(config["data"].get("cache_dir") or ".cache")
    )
    cached = factor_cache.get_or_compute(
        ohlcv=ohlcv,
        config=config,
        data_dir=loader.data_dir,
    )

    factors_3d = cached["factors_3d"]
    factor_names = list(cached["factor_names"])
    dates = cached["dates"]
    etf_codes = cached["etf_codes"]
    T = len(dates)
    N = len(etf_codes)

    # Price arrays
    close_prices = ohlcv["close"][etf_codes].ffill().fillna(1.0).values
    open_prices = ohlcv["open"][etf_codes].ffill().fillna(1.0).values
    high_prices = ohlcv["high"][etf_codes].ffill().fillna(1.0).values
    low_prices = ohlcv["low"][etf_codes].ffill().fillna(1.0).values

    # Timing
    timing_config = config.get("backtest", {}).get("timing", {})
    extreme_threshold = timing_config.get("extreme_threshold", -0.4)
    extreme_position = timing_config.get("extreme_position", 0.3)
    timing_module = LightTimingModule(
        extreme_threshold=extreme_threshold,
        extreme_position=extreme_position,
    )
    timing_arr_raw = (
        timing_module.compute_position_ratios(ohlcv["close"])
        .reindex(dates)
        .fillna(1.0)
        .values
    )
    timing_arr = shift_timing_signal(timing_arr_raw)

    # Regime gate
    gate_arr = compute_regime_gate_arr(
        ohlcv["close"], dates, backtest_config=config.get("backtest", {})
    )
    timing_arr = (timing_arr.astype(np.float64) * gate_arr.astype(np.float64)).astype(
        np.float64
    )

    gate_enabled = bool(config.get("backtest", {}).get("regime_gate", {}).get("enabled", False))
    if gate_enabled:
        s = gate_stats(gate_arr)
        print(f"  Regime gate: mean={s['mean']:.2f}, min={s['min']:.2f}, max={s['max']:.2f}")

    # Cost model
    cost_model = load_cost_model(config)
    qdii_set = set(FrozenETFPool().qdii_codes)
    cost_arr = build_cost_array(cost_model, list(etf_codes), qdii_set)
    tier = cost_model.active_tier
    print(f"  Cost: A-share={tier.a_share*10000:.0f}bp, QDII={tier.qdii*10000:.0f}bp")

    return {
        "factors_3d": factors_3d,
        "factor_names": factor_names,
        "dates": dates,
        "etf_codes": etf_codes,
        "close_prices": close_prices,
        "open_prices": open_prices,
        "high_prices": high_prices,
        "low_prices": low_prices,
        "timing_arr": timing_arr,
        "cost_arr": cost_arr,
        "ohlcv": ohlcv,
        "T": T,
        "N": N,
    }


def load_extra_factors(data, combos_df, config):
    """Load non-OHLCV factors referenced in combos but missing from base factor set.

    Follows the same pattern as batch_vec_backtest.py for loading from
    results/non_ohlcv_factors/*.parquet or combo_wfo.extra_factors.factors_dir.
    """
    factor_names = list(data["factor_names"])
    factors_3d = data["factors_3d"]
    dates = data["dates"]
    etf_codes = data["etf_codes"]

    # Collect all factors referenced in combos
    combo_factors = set()
    for c in combos_df["combo"].tolist():
        combo_factors.update(f.strip() for f in c.split(" + "))

    missing = sorted(combo_factors - set(factor_names))
    if not missing:
        return data

    # Determine extra factors directory
    ext_dir_cfg = config.get("combo_wfo", {}).get("extra_factors", {}).get("factors_dir", "")
    if ext_dir_cfg:
        ext_dir = Path(ext_dir_cfg)
        if not ext_dir.is_absolute():
            ext_dir = ROOT / ext_dir
    else:
        ext_dir = ROOT / "results" / "non_ohlcv_factors"

    if not ext_dir.exists():
        print(f"  WARNING: Extra factors dir not found: {ext_dir}")
        print(f"  Missing factors: {missing}")
        return data

    cs_processor = CrossSectionProcessor(
        lower_percentile=config.get("cross_section", {}).get("winsorize_lower", 0.025),
        upper_percentile=config.get("cross_section", {}).get("winsorize_upper", 0.975),
        verbose=False,
    )

    loaded = 0
    for fname in missing:
        fpath = ext_dir / f"{fname}.parquet"
        if not fpath.exists():
            print(f"  WARNING: {fname}.parquet not found in {ext_dir}")
            continue

        fdf = pd.read_parquet(fpath)
        fdf.index = pd.to_datetime(fdf.index)
        fdf = fdf.reindex(dates)
        fdf = (
            fdf[etf_codes]
            if all(c in fdf.columns for c in etf_codes)
            else fdf.reindex(columns=etf_codes)
        )
        farr = fdf.values
        valid_ratio = np.isfinite(farr).sum() / farr.size if farr.size > 0 else 0

        if valid_ratio > 0.01:
            processed = cs_processor.process_all_factors({fname: fdf})
            fstd = processed[fname].values
            fstd = np.where(np.isfinite(fstd), fstd, 0.0)
            factors_3d = np.concatenate(
                [factors_3d, fstd[:, :, np.newaxis]], axis=2
            )
            factor_names.append(fname)
            loaded += 1
        else:
            print(f"  WARNING: {fname}: insufficient data ({valid_ratio*100:.1f}%)")

    if loaded > 0:
        print(f"  Loaded {loaded} extra factors -> total {len(factor_names)} factors")

    data = {**data}
    data["factors_3d"] = factors_3d
    data["factor_names"] = factor_names
    data["T"] = factors_3d.shape[0]
    data["N"] = factors_3d.shape[1]
    return data


def run_single_combo_vec(combo_str, data, config):
    """Run VEC backtest for a single combo with production params."""
    factor_names = data["factor_names"]
    factor_index_map = {name: idx for idx, name in enumerate(factor_names)}

    factors = [f.strip() for f in combo_str.split(" + ")]
    missing = [f for f in factors if f not in factor_index_map]
    if missing:
        return None, f"Missing factors: {missing}"

    factor_indices = [factor_index_map[f] for f in factors]

    backtest_config = config.get("backtest", {})
    risk_config = backtest_config.get("risk_control", {})

    eq_curve, ret, wr, pf, trades, rounding, risk = run_vec_backtest(
        data["factors_3d"],
        data["close_prices"],
        data["open_prices"],
        data["high_prices"],
        data["low_prices"],
        data["timing_arr"],
        factor_indices,
        freq=backtest_config.get("freq", 5),
        pos_size=backtest_config.get("pos_size", 2),
        initial_capital=float(backtest_config.get("initial_capital", 1_000_000)),
        commission_rate=float(backtest_config.get("commission_rate", 0.0002)),
        lookback=backtest_config.get("lookback") or backtest_config.get("lookback_window", 252),
        cost_arr=data["cost_arr"],
        use_t1_open=True,
        delta_rank=0.10,
        min_hold_days=9,
        trailing_stop_pct=risk_config.get("trailing_stop_pct", 0.08),
        stop_on_rebalance_only=risk_config.get("stop_check_on_rebalance_only", False),
        profit_ladders=risk_config.get("profit_ladders", []),
        circuit_breaker_day=risk_config.get("circuit_breaker", {}).get("max_drawdown_day", 0.0),
        circuit_breaker_total=risk_config.get("circuit_breaker", {}).get("max_drawdown_total", 0.0),
        circuit_recovery_days=risk_config.get("circuit_breaker", {}).get("recovery_days", 5),
        cooldown_days=risk_config.get("cooldown_days", 0),
        leverage_cap=risk_config.get("leverage_cap", 1.0),
    )

    return {
        "equity_curve": eq_curve,
        "total_return": ret,
        "win_rate": wr,
        "profit_factor": pf,
        "num_trades": trades,
        "risk_metrics": risk,
    }, None


def compute_holdout_metrics(equity_curve, dates, training_end="2025-04-30"):
    """Compute holdout-period metrics from equity curve."""
    training_end_dt = pd.Timestamp(training_end)
    ho_start = None
    for i, d in enumerate(dates):
        if d > training_end_dt:
            ho_start = i
            break

    if ho_start is None:
        return {
            "ho_return": np.nan, "ho_mdd": np.nan, "ho_sharpe": np.nan,
            "ho_worst_month": np.nan, "ho_trades": np.nan,
        }

    ho_curve = equity_curve[ho_start:]

    # Find first meaningful value (after capital deploys)
    init_val = ho_curve[0]
    first_active = 0
    for i in range(len(ho_curve)):
        if ho_curve[i] != init_val:
            first_active = max(0, i - 1)
            break
    ho_curve = ho_curve[first_active:]

    if len(ho_curve) < 2 or ho_curve[0] <= 0:
        return {
            "ho_return": np.nan, "ho_mdd": np.nan, "ho_sharpe": np.nan,
            "ho_worst_month": np.nan,
        }

    ho_return = (ho_curve[-1] / ho_curve[0]) - 1.0

    # Max drawdown
    peak = ho_curve[0]
    max_dd = 0.0
    for v in ho_curve:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    # Sharpe
    daily_rets = np.diff(ho_curve) / ho_curve[:-1]
    daily_rets = daily_rets[np.isfinite(daily_rets)]
    if len(daily_rets) > 1 and np.std(daily_rets) > 1e-8:
        sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Worst month
    ho_dates = dates[ho_start:]
    ho_dates = ho_dates[first_active:]
    monthly_rets = []
    if len(ho_dates) >= 20 and len(ho_curve) == len(ho_dates):
        month_start_val = ho_curve[0]
        current_month = ho_dates[0].month if hasattr(ho_dates[0], "month") else pd.Timestamp(ho_dates[0]).month
        for i in range(1, len(ho_curve)):
            d = ho_dates[i] if i < len(ho_dates) else ho_dates[-1]
            m = d.month if hasattr(d, "month") else pd.Timestamp(d).month
            if m != current_month:
                mret = (ho_curve[i - 1] / month_start_val) - 1.0 if month_start_val > 0 else 0
                monthly_rets.append(mret)
                month_start_val = ho_curve[i - 1]
                current_month = m
        if month_start_val > 0:
            mret = (ho_curve[-1] / month_start_val) - 1.0
            monthly_rets.append(mret)

    worst_month = min(monthly_rets) if monthly_rets else np.nan

    return {
        "ho_return": ho_return,
        "ho_mdd": max_dd,
        "ho_sharpe": sharpe,
        "ho_worst_month": worst_month,
    }


def main():
    print("=" * 90)
    print("VEC Holdout Validation: WFO Top Combos + S1/C2 Baselines")
    print("=" * 90)

    # 1. Find WFO input
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    wfo_path = find_wfo_input(arg)
    print(f"\nWFO source: {wfo_path}")
    print(f"WFO run dir: {wfo_path.parent}")

    # 2. Load config
    config_path = Path(
        os.environ.get("WFO_CONFIG_PATH", str(ROOT / "configs/combo_wfo_config.yaml"))
    )
    with open(config_path) as f:
        config = yaml.safe_load(f)
    print(f"Config: {config_path}")

    backtest_config = config.get("backtest", {})
    training_end = config["data"].get("training_end_date", "2025-04-30")
    print(f"Training end: {training_end}")
    print(f"Params: FREQ={backtest_config.get('freq')}, POS_SIZE={backtest_config.get('pos_size')}, "
          f"LOOKBACK={backtest_config.get('lookback')}")

    # 3. Load top combos
    top_combos, sort_col = load_top_combos(wfo_path, n=TOP_N)
    print(f"\nCombos to validate: {len(top_combos)}")
    for i, row in top_combos.iterrows():
        wfo_val = row.get(sort_col, np.nan)
        is_baseline = row["combo"] in (S1_COMBO, C2_COMBO)
        tag = " [BASELINE]" if is_baseline else ""
        wfo_str = f"{wfo_val:+.4f}" if pd.notna(wfo_val) else "N/A"
        print(f"  [{i:>2}] {wfo_str}  {row['combo']}{tag}")

    # 4. Prepare data
    print(f"\n--- Loading data ---")
    data = prepare_data(config)
    print(f"  Base: T={data['T']}, N={data['N']}, factors={len(data['factor_names'])}")

    # 5. Load extra factors (non-OHLCV)
    data = load_extra_factors(data, top_combos, config)
    print(f"  Final: T={data['T']}, N={data['N']}, factors={len(data['factor_names'])}")

    # 6. Run VEC backtest for each combo
    print(f"\n--- Running VEC backtests (F5, T1_OPEN, Exp4 dr=0.10 mh=9, regime ON, med cost) ---")
    results = []
    for i, row in top_combos.iterrows():
        combo_str = row["combo"]
        short = combo_str if len(combo_str) <= 60 else combo_str[:57] + "..."
        print(f"  [{i:>2}] {short}...", end=" ", flush=True)

        result, error = run_single_combo_vec(combo_str, data, config)
        if error:
            print(f"SKIP ({error})")
            results.append({
                "rank": i,
                "combo": combo_str,
                "wfo_score": row.get(sort_col, np.nan),
                "error": error,
            })
            continue

        ho = compute_holdout_metrics(result["equity_curve"], data["dates"], training_end)
        print(f"HO: {ho['ho_return']*100:+.1f}%, MDD: {ho['ho_mdd']*100:.1f}%, "
              f"Sharpe: {ho['ho_sharpe']:.2f}, Trades: {result['num_trades']}")

        results.append({
            "rank": i,
            "combo": combo_str,
            "wfo_score": row.get(sort_col, np.nan),
            "full_return": result["total_return"],
            "num_trades": result["num_trades"],
            "win_rate": result["win_rate"],
            "ho_return": ho["ho_return"],
            "ho_mdd": ho["ho_mdd"],
            "ho_sharpe": ho["ho_sharpe"],
            "ho_worst_month": ho.get("ho_worst_month", np.nan),
            "sharpe_full": result["risk_metrics"]["sharpe_ratio"],
            "mdd_full": result["risk_metrics"]["max_drawdown"],
        })

    # 7. Results table
    df_results = pd.DataFrame(results)
    df_valid = df_results.dropna(subset=["ho_return"]).copy()

    if df_valid.empty:
        print("\nNo valid results.")
        return

    # Sort by HO Sharpe
    df_valid = df_valid.sort_values("ho_sharpe", ascending=False).reset_index(drop=True)

    print(f"\n{'='*90}")
    print(f"VEC HOLDOUT VALIDATION RESULTS (sorted by HO Sharpe)")
    print(f"{'='*90}")
    print(f"{'#':<3} {'WFO':>6} {'HO Ret':>8} {'HO MDD':>8} {'HO Shrp':>8} {'WrstMo':>8} {'Trades':>7} {'Combo'}")
    print(f"{'-'*3} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*40}")

    for i, row in df_valid.iterrows():
        combo = row["combo"]
        is_s1 = combo == S1_COMBO
        is_c2 = combo == C2_COMBO
        tag = " *S1*" if is_s1 else (" *C2*" if is_c2 else "")
        short_combo = combo if len(combo) <= 50 else combo[:47] + "..."
        wfo_str = f"{row['wfo_score']:+.3f}" if pd.notna(row["wfo_score"]) else "  N/A"
        wm = f"{row['ho_worst_month']*100:+.1f}%" if pd.notna(row.get("ho_worst_month")) else "   N/A"

        print(
            f"{row['rank']:>3} "
            f"{wfo_str:>6} "
            f"{row['ho_return']*100:>+7.1f}% "
            f"{row['ho_mdd']*100:>7.1f}% "
            f"{row['ho_sharpe']:>8.2f} "
            f"{wm:>8} "
            f"{row['num_trades']:>7.0f} "
            f"{short_combo}{tag}"
        )

    # 8. Ranking comparison
    print(f"\n{'='*90}")
    print(f"WFO vs VEC RANKING COMPARISON")
    print(f"{'='*90}")

    # Exclude baselines for rank comparison
    df_wfo_ranked = df_valid[~df_valid["combo"].isin([S1_COMBO, C2_COMBO])].copy()
    if not df_wfo_ranked.empty:
        df_wfo_ranked["wfo_rank"] = df_wfo_ranked["wfo_score"].rank(ascending=False, method="min").astype(int)
        df_wfo_ranked["vec_rank"] = df_wfo_ranked["ho_sharpe"].rank(ascending=False, method="min").astype(int)
        df_wfo_ranked["rank_delta"] = df_wfo_ranked["wfo_rank"] - df_wfo_ranked["vec_rank"]

        print(f"{'WFO#':>5} {'VEC#':>5} {'Delta':>6} {'WFO Score':>10} {'HO Shrp':>8} {'Combo'}")
        print(f"{'-'*5} {'-'*5} {'-'*6} {'-'*10} {'-'*8} {'-'*40}")

        for _, row in df_wfo_ranked.sort_values("wfo_rank").iterrows():
            short = row["combo"] if len(row["combo"]) <= 50 else row["combo"][:47] + "..."
            wfo_str = f"{row['wfo_score']:+.4f}" if pd.notna(row["wfo_score"]) else "N/A"
            print(
                f"{row['wfo_rank']:>5} "
                f"{row['vec_rank']:>5} "
                f"{row['rank_delta']:>+6} "
                f"{wfo_str:>10} "
                f"{row['ho_sharpe']:>8.2f} "
                f"{short}"
            )

        # Rank correlation
        if len(df_wfo_ranked) >= 3:
            from scipy.stats import spearmanr, kendalltau
            rho, p_rho = spearmanr(df_wfo_ranked["wfo_rank"], df_wfo_ranked["vec_rank"])
            tau, p_tau = kendalltau(df_wfo_ranked["wfo_rank"], df_wfo_ranked["vec_rank"])
            print(f"\nSpearman rho: {rho:.3f} (p={p_rho:.3f})")
            print(f"Kendall tau:  {tau:.3f} (p={p_tau:.3f})")
            if rho > 0.5:
                print("-> WFO ranking is MODERATELY predictive of VEC holdout")
            elif rho > 0.0:
                print("-> WFO ranking has WEAK correlation with VEC holdout")
            else:
                print("-> WFO ranking does NOT predict VEC holdout (or inverted)")

    # 9. Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / f"vec_wfo_validation_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_results.to_parquet(out_dir / "vec_validation_results.parquet", index=False)
    print(f"\nResults saved: {out_dir}")

    # 10. Summary
    print(f"\n{'='*90}")
    print(f"SUMMARY")
    print(f"{'='*90}")

    s1_row = df_valid[df_valid["combo"] == S1_COMBO]
    c2_row = df_valid[df_valid["combo"] == C2_COMBO]
    best_row = df_valid.iloc[0] if not df_valid.empty else None

    if not s1_row.empty:
        s1 = s1_row.iloc[0]
        print(f"S1 baseline: HO {s1['ho_return']*100:+.1f}%, MDD {s1['ho_mdd']*100:.1f}%, Sharpe {s1['ho_sharpe']:.2f}")
    if not c2_row.empty:
        c2 = c2_row.iloc[0]
        print(f"C2 baseline: HO {c2['ho_return']*100:+.1f}%, MDD {c2['ho_mdd']*100:.1f}%, Sharpe {c2['ho_sharpe']:.2f}")
    if best_row is not None:
        print(f"Best combo:  HO {best_row['ho_return']*100:+.1f}%, MDD {best_row['ho_mdd']*100:.1f}%, "
              f"Sharpe {best_row['ho_sharpe']:.2f}")
        print(f"  -> {best_row['combo']}")

    # Count combos beating S1 and C2
    if not s1_row.empty:
        s1_sharpe = s1_row.iloc[0]["ho_sharpe"]
        n_beat_s1 = len(df_valid[df_valid["ho_sharpe"] > s1_sharpe])
        print(f"\nCombos beating S1 Sharpe ({s1_sharpe:.2f}): {n_beat_s1}/{len(df_valid)}")
    if not c2_row.empty:
        c2_sharpe = c2_row.iloc[0]["ho_sharpe"]
        n_beat_c2 = len(df_valid[df_valid["ho_sharpe"] > c2_sharpe])
        print(f"Combos beating C2 Sharpe ({c2_sharpe:.2f}): {n_beat_c2}/{len(df_valid)}")


if __name__ == "__main__":
    main()
