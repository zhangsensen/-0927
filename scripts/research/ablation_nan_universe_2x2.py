#!/usr/bin/env python3
"""
2×2 Ablation Matrix: Decompose S1 performance drop into two independent effects.

Axes:
  - Universe: GLOBAL (all 49 ETFs ranked) vs A_SHARE_ONLY (QDII excluded from ranking)
  - Aggregation: valid_count (÷ valid factors) vs fixed_denom (NaN→0, ÷ total factors)

Methodology:
  - valid_count: original factors_3d with NaN → kernel divides by count of non-NaN
  - fixed_denom: replace NaN with 0 → kernel divides by total count (all "valid")
  - A_SHARE_ONLY: QDII columns set to NaN (→ -inf in kernel, excluded from ranking)
  - GLOBAL: all ETFs participate

This decomposes S1 HO change from +42.7% (sealed) to +14.4% (current) into:
  - Universe effect: GLOBAL→A_SHARE (was the sealed result contaminated by QDII?)
  - Aggregation effect: raw_sum→valid_count (was NaN penalty critical for S1?)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import yaml
import numpy as np
import pandas as pd

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.factor_cache import FactorCache
from etf_strategy.core.frozen_params import load_frozen_config, FrozenETFPool
from etf_strategy.core.cost_model import load_cost_model, build_cost_array
from etf_strategy.core.execution_model import load_execution_model
from etf_strategy.core.utils.rebalance import shift_timing_signal
from etf_strategy.regime_gate import compute_regime_gate_arr
from batch_vec_backtest import run_vec_backtest

STRATEGIES = {
    "S1": ["ADX_14D", "OBV_SLOPE_10D", "SHARPE_RATIO_20D", "SLOPE_20D"],
    "C2": ["AMIHUD_ILLIQUIDITY", "CALMAR_RATIO_60D", "CORRELATION_TO_MARKET_20D"],
}


def compute_period_metrics(eq, start, end):
    """Compute return, MDD, Sharpe, worst month from equity curve slice."""
    eq_slice = eq[start:end + 1]
    if len(eq_slice) < 2 or eq_slice[0] == 0:
        return {"return": 0.0, "mdd": 0.0, "sharpe": 0.0, "worst_month": 0.0}
    ret = eq_slice[-1] / eq_slice[0] - 1
    peak = eq_slice[0]
    mdd = 0.0
    for v in eq_slice:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > mdd:
            mdd = dd
    daily_rets = np.diff(eq_slice) / eq_slice[:-1]
    daily_rets = daily_rets[np.isfinite(daily_rets)]
    sharpe = 0.0
    if len(daily_rets) > 1 and np.std(daily_rets) > 0:
        sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)
    worst_month = 0.0
    for m in range(len(daily_rets) // 21):
        m_ret = np.prod(1 + daily_rets[m * 21:(m + 1) * 21]) - 1
        if m_ret < worst_month:
            worst_month = m_ret
    return {"return": ret, "mdd": mdd, "sharpe": sharpe, "worst_month": worst_month}


def prepare_factors(factors_3d_orig, qdii_mask, agg_mode, univ_mode):
    """Prepare factors_3d for a specific (aggregation, universe) combination.

    agg_mode:
      - "valid_count": NaN preserved → kernel divides by valid count
      - "fixed_denom": NaN→0 for non-QDII → kernel divides by total count

    univ_mode:
      - "GLOBAL": all ETFs participate
      - "A_SHARE_ONLY": QDII columns set to NaN → kernel gives -inf
    """
    f3d = factors_3d_orig.copy()

    if agg_mode == "fixed_denom":
        if univ_mode == "A_SHARE_ONLY":
            # Step 1: replace non-QDII NaN with 0 (fixed denominator for A-share)
            for n in range(f3d.shape[1]):
                if not qdii_mask[n]:
                    nan_mask = np.isnan(f3d[:, n, :])
                    f3d[:, n, :][nan_mask] = 0.0
            # Step 2: explicitly exclude QDII (original data has valid values!)
            f3d[:, qdii_mask, :] = np.nan
        else:
            # GLOBAL: replace ALL NaN with 0
            nan_mask = np.isnan(f3d)
            f3d[nan_mask] = 0.0
    elif agg_mode == "valid_count":
        if univ_mode == "A_SHARE_ONLY":
            # Set QDII to NaN → -inf in kernel
            f3d[:, qdii_mask, :] = np.nan
        # else: GLOBAL with valid_count = original behavior

    return f3d


def main():
    # ── Load config ──
    config_path = ROOT / "configs/combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    frozen = load_frozen_config(config, config_path=str(config_path))
    bt_cfg = config["backtest"]
    FREQ = bt_cfg["freq"]
    POS_SIZE = bt_cfg["pos_size"]
    LOOKBACK = bt_cfg.get("lookback") or bt_cfg.get("lookback_window")
    INITIAL_CAPITAL = float(bt_cfg["initial_capital"])
    COMMISSION_RATE = float(bt_cfg["commission_rate"])
    hyst_cfg = bt_cfg.get("hysteresis", {})
    DELTA_RANK = hyst_cfg.get("delta_rank", 0.0)
    MIN_HOLD_DAYS = hyst_cfg.get("min_hold_days", 0)

    exec_model = load_execution_model(config)
    cost_model = load_cost_model(config)
    qdii_set = set(FrozenETFPool().qdii_codes)

    print(f"🔒 v{frozen.version} | FREQ={FREQ} POS={POS_SIZE} dr={DELTA_RANK} mh={MIN_HOLD_DAYS}")
    print(f"📋 Exec={exec_model.mode} Cost={cost_model.mode}/{cost_model.tier}")

    # ── Load data ──
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
        ohlcv=ohlcv, config=config, data_dir=loader.data_dir
    )
    factors_3d_orig = cached["factors_3d"]
    factor_names = list(cached["factor_names"])
    dates = cached["dates"]
    etf_codes = list(cached["etf_codes"])
    T, N = len(dates), len(etf_codes)

    COST_ARR = build_cost_array(cost_model, etf_codes, qdii_set)

    # Timing + regime gate
    from etf_strategy.core.market_timing import LightTimingModule
    timing_config = bt_cfg.get("timing", {})
    tm = LightTimingModule(
        extreme_threshold=timing_config.get("extreme_threshold", -0.4),
        extreme_position=timing_config.get("extreme_position", 0.3),
    )
    timing_arr_raw = tm.compute_position_ratios(ohlcv["close"]).reindex(dates).fillna(1.0).values
    timing_arr = shift_timing_signal(timing_arr_raw)
    gate_arr = compute_regime_gate_arr(ohlcv["close"], dates, backtest_config=bt_cfg)
    timing_arr = (timing_arr.astype(np.float64) * gate_arr.astype(np.float64))

    close_prices = ohlcv["close"][etf_codes].ffill().fillna(1.0).values
    open_prices = ohlcv["open"][etf_codes].ffill().fillna(1.0).values
    high_prices = ohlcv["high"][etf_codes].ffill().fillna(1.0).values
    low_prices = ohlcv["low"][etf_codes].ffill().fillna(1.0).values

    qdii_mask = np.array([c in qdii_set for c in etf_codes], dtype=np.bool_)

    # Holdout period
    holdout_start = pd.Timestamp("2025-05-01")
    ho_idx = 0
    for i, d in enumerate(dates):
        if pd.Timestamp(d) >= holdout_start:
            ho_idx = i
            break

    # NaN diagnostic
    n_nan_total = np.isnan(factors_3d_orig).sum()
    n_cells = factors_3d_orig.size
    print(f"\n📊 Factor NaN: {n_nan_total}/{n_cells} ({n_nan_total/n_cells*100:.1f}%)")

    # ── 2×2 Matrix ──
    AGG_MODES = ["valid_count", "fixed_denom"]
    UNIV_MODES = ["GLOBAL", "A_SHARE_ONLY"]

    for strat_name, factor_list in STRATEGIES.items():
        print(f"\n{'='*75}")
        print(f"  Strategy: {strat_name} ({' + '.join(factor_list)})")
        print(f"{'='*75}")

        factor_indices = [factor_names.index(f) for f in factor_list]

        # Count NaN for this combo
        combo_factors = factors_3d_orig[:, :, factor_indices]
        nan_per_factor = {}
        for fi, fn in enumerate(factor_list):
            nan_rate = np.isnan(combo_factors[:, :, fi]).sum() / (T * N)
            nan_per_factor[fn] = nan_rate
        print(f"  NaN rates: {', '.join(f'{k}={v*100:.1f}%' for k,v in nan_per_factor.items())}")

        results = {}
        for agg_mode in AGG_MODES:
            for univ_mode in UNIV_MODES:
                label = f"{agg_mode}|{univ_mode}"
                print(f"\n  🏃 [{label}]...", end=" ", flush=True)

                f3d = prepare_factors(factors_3d_orig, qdii_mask, agg_mode, univ_mode)

                eq, ret, wr, pf, trades, _, risk = run_vec_backtest(
                    f3d, close_prices, open_prices, high_prices, low_prices,
                    timing_arr, factor_indices,
                    freq=FREQ, pos_size=POS_SIZE, initial_capital=INITIAL_CAPITAL,
                    commission_rate=COMMISSION_RATE, lookback=LOOKBACK,
                    cost_arr=COST_ARR, use_t1_open=exec_model.is_t1_open,
                    delta_rank=DELTA_RANK, min_hold_days=MIN_HOLD_DAYS,
                )

                ho = compute_period_metrics(eq, ho_idx, T - 1)
                full = compute_period_metrics(eq, LOOKBACK, T - 1)
                results[label] = {"ho": ho, "full": full, "trades": trades}
                print(f"HO={ho['return']*100:+.1f}% MDD={ho['mdd']*100:.1f}% Sharpe={ho['sharpe']:.2f} Trades={trades}")

        # ── Print 2×2 matrix ──
        print(f"\n  {'─'*75}")
        print(f"  2×2 ABLATION MATRIX — {strat_name} Holdout (2025-05 ~ 2026-02)")
        print(f"  {'─'*75}")
        print(f"  {'':30s} {'GLOBAL':>18s} {'A_SHARE_ONLY':>18s} {'Δ Universe':>14s}")
        print(f"  {'─'*75}")

        for agg_mode in AGG_MODES:
            g = results[f"{agg_mode}|GLOBAL"]["ho"]
            a = results[f"{agg_mode}|A_SHARE_ONLY"]["ho"]
            d_univ = a["return"] - g["return"]
            label = "valid_count (÷valid)" if agg_mode == "valid_count" else "fixed_denom (NaN→0)"
            print(f"  {label:<30s} {g['return']*100:>+17.1f}% {a['return']*100:>+17.1f}% {d_univ*100:>+13.1f}pp")

        # Aggregation delta
        print(f"  {'─'*75}")
        for univ_mode in UNIV_MODES:
            vc = results[f"valid_count|{univ_mode}"]["ho"]
            fd = results[f"fixed_denom|{univ_mode}"]["ho"]
            d_agg = fd["return"] - vc["return"]
            print(f"  {'Δ Aggregation (' + univ_mode + ')':<30s} {'':>18s} {'':>18s} {d_agg*100:>+13.1f}pp")

        # Full metrics table
        print(f"\n  {'─'*75}")
        print(f"  {'Config':<35s} {'HO Ret':>8s} {'HO MDD':>8s} {'Sharpe':>8s} {'WM':>8s} {'Trades':>7s}")
        print(f"  {'─'*75}")
        for key in sorted(results.keys()):
            r = results[key]
            ho = r["ho"]
            print(f"  {key:<35s} {ho['return']*100:>+7.1f}% {ho['mdd']*100:>7.1f}% {ho['sharpe']:>8.2f} {ho['worst_month']*100:>+7.1f}% {r['trades']:>7d}")

        # Decomposition summary
        print(f"\n  📊 Decomposition:")
        base = results["valid_count|GLOBAL"]["ho"]["return"]
        target = results["fixed_denom|A_SHARE_ONLY"]["ho"]["return"]
        via_agg = results["fixed_denom|GLOBAL"]["ho"]["return"] - base
        via_univ = results["fixed_denom|A_SHARE_ONLY"]["ho"]["return"] - results["fixed_denom|GLOBAL"]["ho"]["return"]
        total = target - base
        print(f"    Base (valid_count|GLOBAL):     {base*100:+.1f}%")
        print(f"    + Aggregation effect:           {via_agg*100:+.1f}pp")
        print(f"    + Universe effect:              {via_univ*100:+.1f}pp")
        print(f"    = Target (fixed_denom|A_SHARE): {target*100:+.1f}% (total Δ={total*100:+.1f}pp)")
        interaction = total - via_agg - via_univ
        if abs(interaction) > 0.005:
            print(f"    ⚠️ Interaction term:            {interaction*100:+.1f}pp")


if __name__ == "__main__":
    main()
