#!/usr/bin/env python3
"""
Alpha Source Decomposition: C2 regime gate ON vs OFF

Decomposes C2's holdout return into:
  1. Factor selection alpha (C2_gateON - Random_gateON)
  2. Regime timing beta  (Random_gateON - EqualWeight_gateOFF)
  3. Interaction term     (remainder)

Test matrix:
  a) C2 gate ON           — baseline (production config)
  b) C2 gate OFF          — timing_arr = ones (no regime scaling)
  c) EqualWeight gate ON  — hold all A-share ETFs equally, with regime gate
  d) EqualWeight gate OFF — hold all A-share ETFs equally, no gate (buy-and-hold)
  e) Random-2 gate ON     — randomly select 2 ETFs with Exp4 hysteresis, 20 seeds
  f) Random-2 gate OFF    — randomly select 2 ETFs without regime gate, 20 seeds

Decomposition:
  Pure timing beta      = EW_gateON - EW_gateOFF
  Selection alpha       = C2_gateON - Random_gateON_median
  Timing x Selection    = remainder (C2_gateON - C2_gateOFF - Pure_timing_beta)
  C2 gate contribution  = C2_gateON - C2_gateOFF
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import yaml
import pandas as pd
import numpy as np
from datetime import datetime

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.factor_cache import FactorCache
from etf_strategy.core.frozen_params import load_frozen_config, FrozenETFPool
from etf_strategy.core.cost_model import load_cost_model, build_cost_array
from etf_strategy.core.execution_model import load_execution_model
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import shift_timing_signal
from etf_strategy.regime_gate import compute_regime_gate_arr

# Import run_vec_backtest from batch_vec_backtest
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "batch_vec", str(ROOT / "scripts/batch_vec_backtest.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
run_vec_backtest = _mod.run_vec_backtest

# ── C2 factor combination ──
C2_FACTORS = ["AMIHUD_ILLIQUIDITY", "CALMAR_RATIO_60D", "CORRELATION_TO_MARKET_20D"]

N_RANDOM_SEEDS = 20


def compute_holdout_metrics(equity_curve: np.ndarray, dates, holdout_start: str):
    """Compute holdout-period metrics from equity curve."""
    dates_list = [str(d.date()) if hasattr(d, "date") else str(d) for d in dates]
    try:
        ho_idx = next(i for i, d in enumerate(dates_list) if d >= holdout_start)
    except StopIteration:
        return {}

    ho_eq = equity_curve[ho_idx:]
    if len(ho_eq) < 10:
        return {}

    ho_ret = ho_eq[-1] / ho_eq[0] - 1.0
    running_max = np.maximum.accumulate(ho_eq)
    drawdowns = (ho_eq - running_max) / running_max
    ho_mdd = -drawdowns.min()

    # Sharpe (annualized, daily)
    daily_rets = np.diff(ho_eq) / ho_eq[:-1]
    if len(daily_rets) > 1 and np.std(daily_rets) > 1e-10:
        ho_sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)
    else:
        ho_sharpe = 0.0

    # Worst month (approximate: 21 trading days)
    worst_month = 0.0
    for i in range(0, len(ho_eq) - 21, 5):
        m_ret = ho_eq[i + 21] / ho_eq[i] - 1.0
        if m_ret < worst_month:
            worst_month = m_ret

    return {
        "ho_return": ho_ret,
        "ho_mdd": ho_mdd,
        "ho_sharpe": ho_sharpe,
        "ho_worst_month": worst_month,
    }


def compute_full_period_metrics(equity_curve: np.ndarray):
    """Compute full-period metrics from equity curve."""
    full_ret = equity_curve[-1] / equity_curve[0] - 1.0 if len(equity_curve) > 0 else 0.0
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max
    full_mdd = -drawdowns.min()

    daily_rets = np.diff(equity_curve) / equity_curve[:-1]
    if len(daily_rets) > 1 and np.std(daily_rets) > 1e-10:
        full_sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)
    else:
        full_sharpe = 0.0

    return {
        "full_return": full_ret,
        "full_mdd": full_mdd,
        "full_sharpe": full_sharpe,
    }


def compute_equal_weight_portfolio(close_df, timing_arr, dates, a_share_mask, holdout_start):
    """Compute equal-weight buy-and-hold of all A-share ETFs with timing scaling.

    Args:
        close_df: DataFrame (T, N) close prices aligned to dates
        timing_arr: (T,) timing/regime scaling array
        dates: DatetimeIndex
        a_share_mask: boolean array (N,) True for A-share ETFs
        holdout_start: holdout period start date string

    Returns:
        dict with equity curve and metrics
    """
    prices = close_df.values  # (T, N)
    T, N = prices.shape

    # Daily returns for A-share ETFs only
    daily_rets = np.zeros((T, N))
    for t in range(1, T):
        for n in range(N):
            if a_share_mask[n] and prices[t - 1, n] > 0 and not np.isnan(prices[t, n]):
                daily_rets[t, n] = prices[t, n] / prices[t - 1, n] - 1.0

    # Equal weight across A-share ETFs
    n_a_share = a_share_mask.sum()
    if n_a_share == 0:
        raise ValueError("No A-share ETFs found")

    # Portfolio daily return = mean of A-share ETF returns, scaled by timing
    port_daily_ret = np.zeros(T)
    for t in range(1, T):
        a_rets = daily_rets[t, a_share_mask]
        valid = np.isfinite(a_rets)
        if valid.sum() > 0:
            port_daily_ret[t] = np.mean(a_rets[valid]) * timing_arr[t]

    # Build equity curve
    equity = np.ones(T, dtype=np.float64)
    for t in range(1, T):
        equity[t] = equity[t - 1] * (1.0 + port_daily_ret[t])

    # Scale to initial capital
    equity *= 1_000_000.0

    ho_metrics = compute_holdout_metrics(equity, dates, holdout_start)
    full_metrics = compute_full_period_metrics(equity)

    return {
        "equity_curve": equity,
        **full_metrics,
        **ho_metrics,
    }


def run_random_selection_vec(
    factors_3d, close_prices, open_prices, high_prices, low_prices,
    timing_arr, factor_names,
    freq, pos_size, initial_capital, commission_rate, lookback,
    cost_arr, use_t1_open, delta_rank, min_hold_days,
    trailing_stop_pct, stop_on_rebalance_only, leverage_cap,
    dates, holdout_start, seed,
):
    """Run VEC with a random factor injected into factors_3d.

    Creates a synthetic random score column and uses it as the sole factor
    for selection. This measures what random ETF selection + regime timing achieves.
    """
    T, N, K = factors_3d.shape
    rng = np.random.RandomState(seed)

    # Create random scores: (T, N) uniform random, cross-sectionally independent
    random_scores = rng.randn(T, N)

    # Append as new factor column
    factors_3d_ext = np.concatenate(
        [factors_3d, random_scores[:, :, np.newaxis]], axis=2
    )
    random_factor_idx = K  # index of the new random column

    eq_curve, ret, wr, pf, trades, rounding, risk = run_vec_backtest(
        factors_3d_ext, close_prices, open_prices, high_prices, low_prices,
        timing_arr, [random_factor_idx],
        freq=freq, pos_size=pos_size, initial_capital=initial_capital,
        commission_rate=commission_rate, lookback=lookback,
        cost_arr=cost_arr,
        trailing_stop_pct=trailing_stop_pct,
        stop_on_rebalance_only=stop_on_rebalance_only,
        leverage_cap=leverage_cap,
        use_t1_open=use_t1_open,
        delta_rank=delta_rank,
        min_hold_days=min_hold_days,
    )

    ho_metrics = compute_holdout_metrics(eq_curve, dates, holdout_start)
    full_metrics = compute_full_period_metrics(eq_curve)

    return {
        "seed": seed,
        "equity_curve": eq_curve,
        "full_return": ret,
        "trades": trades,
        **full_metrics,
        **ho_metrics,
    }


def main():
    print("=" * 90)
    print("Alpha Source Decomposition: C2 Regime Gate ON vs OFF")
    print("=" * 90)

    # ── 1. Load config ──
    import os
    config_path = Path(
        os.environ.get("WFO_CONFIG_PATH", str(ROOT / "configs/combo_wfo_config.yaml"))
    )
    with open(config_path) as f:
        config = yaml.safe_load(f)

    frozen = load_frozen_config(config, config_path=str(config_path))
    print(f"  Frozen config: version={frozen.version}")

    backtest_config = config.get("backtest", {})
    FREQ = backtest_config["freq"]
    POS_SIZE = backtest_config["pos_size"]
    LOOKBACK = backtest_config.get("lookback") or backtest_config.get("lookback_window")
    INITIAL_CAPITAL = float(backtest_config["initial_capital"])
    COMMISSION_RATE = float(backtest_config["commission_rate"])
    print(f"  FREQ={FREQ}, POS_SIZE={POS_SIZE}, LOOKBACK={LOOKBACK}")

    # Execution model
    exec_model = load_execution_model(config)
    USE_T1_OPEN = exec_model.is_t1_open
    print(f"  EXECUTION: {exec_model.mode}")

    # Cost model
    cost_model = load_cost_model(config)
    qdii_set = set(FrozenETFPool().qdii_codes)

    # Hysteresis
    hyst_config = backtest_config.get("hysteresis", {})
    DELTA_RANK = float(hyst_config.get("delta_rank", 0.0))
    MIN_HOLD_DAYS = int(hyst_config.get("min_hold_days", 0))
    print(f"  HYSTERESIS: delta_rank={DELTA_RANK}, min_hold_days={MIN_HOLD_DAYS}")

    # Risk control
    risk_config = backtest_config.get("risk_control", {})
    trailing_stop_pct = risk_config.get("trailing_stop_pct", 0.08)
    leverage_cap = risk_config.get("leverage_cap", 1.0)
    stop_on_rebalance_only = risk_config.get("stop_check_on_rebalance_only", False)

    # Holdout
    holdout_start = config["data"].get("holdout_start", "2025-05-01")
    print(f"  Holdout start: {holdout_start}")

    # ── 2. Load data & compute factors ──
    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )

    factor_cache = FactorCache(cache_dir=Path(config["data"].get("cache_dir") or ".cache"))
    cached = factor_cache.get_or_compute(ohlcv=ohlcv, config=config, data_dir=loader.data_dir)

    factor_names = list(cached["factor_names"])
    dates = cached["dates"]
    etf_codes = cached["etf_codes"]
    factors_3d = cached["factors_3d"].copy()  # (T, N, K)
    T, N, K = factors_3d.shape
    print(f"  Data: T={T}, N={N}, K={K} factors")

    # C2 factor indices
    factor_index_map = {name: idx for idx, name in enumerate(factor_names)}
    C2_INDICES = [factor_index_map[f] for f in C2_FACTORS]
    print(f"  C2 factor indices: {dict(zip(C2_FACTORS, C2_INDICES))}")

    # Cost array
    COST_ARR = build_cost_array(cost_model, list(etf_codes), qdii_set)
    tier = cost_model.active_tier
    print(f"  COST: A={tier.a_share*10000:.0f}bp, QDII={tier.qdii*10000:.0f}bp")

    # Prices
    close_df = ohlcv["close"][etf_codes].ffill().fillna(1.0)
    close_prices = close_df.values
    open_prices = ohlcv["open"][etf_codes].ffill().fillna(1.0).values
    high_prices = ohlcv["high"][etf_codes].ffill().fillna(1.0).values
    low_prices = ohlcv["low"][etf_codes].ffill().fillna(1.0).values

    # ── 3. Compute timing arrays ──
    timing_config = backtest_config.get("timing", {})
    timing_type = timing_config.get("type", "light_timing")
    extreme_threshold = timing_config.get("extreme_threshold", -0.4)
    extreme_position = timing_config.get("extreme_position", 0.3)

    if timing_type == "light_timing":
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
    else:
        timing_arr_raw = np.ones(T, dtype=np.float64)

    timing_shifted = shift_timing_signal(timing_arr_raw)

    # Gate ON: timing * regime gate
    gate_arr = compute_regime_gate_arr(
        ohlcv["close"], dates, backtest_config=backtest_config
    )
    timing_gate_on = timing_shifted * gate_arr

    # Gate OFF: timing only (no regime scaling)
    timing_gate_off = np.ones(T, dtype=np.float64)

    print(f"  Timing: type={timing_type}, gate mean={gate_arr.mean():.3f}")
    print(f"  Gate ON mean exposure: {timing_gate_on.mean():.3f}")

    # A-share mask for equal-weight
    qdii_codes = set(config.get("universe", {}).get("qdii_tickers", []))
    a_share_mask = np.array([code not in qdii_codes for code in etf_codes], dtype=bool)
    n_a_share = a_share_mask.sum()
    print(f"  A-share ETFs: {n_a_share}, QDII: {N - n_a_share}")

    # ── Common VEC params ──
    vec_params = dict(
        freq=FREQ, pos_size=POS_SIZE, initial_capital=INITIAL_CAPITAL,
        commission_rate=COMMISSION_RATE, lookback=LOOKBACK,
        cost_arr=COST_ARR,
        trailing_stop_pct=trailing_stop_pct,
        stop_on_rebalance_only=stop_on_rebalance_only,
        leverage_cap=leverage_cap,
        use_t1_open=USE_T1_OPEN,
        delta_rank=DELTA_RANK,
        min_hold_days=MIN_HOLD_DAYS,
    )

    results = {}

    # ── 4a. C2 gate ON (baseline) ──
    print(f"\n{'='*70}")
    print("(a) C2 gate ON (production baseline)")
    print(f"{'='*70}")
    eq_c2_on, ret, wr, pf, trades, _, risk = run_vec_backtest(
        factors_3d, close_prices, open_prices, high_prices, low_prices,
        timing_gate_on, C2_INDICES, **vec_params,
    )
    ho = compute_holdout_metrics(eq_c2_on, dates, holdout_start)
    fp = compute_full_period_metrics(eq_c2_on)
    results["C2_gate_ON"] = {"equity": eq_c2_on, "trades": trades, **fp, **ho}
    print(f"  Full: {ret*100:.1f}%, HO: {ho.get('ho_return',0)*100:.1f}%, "
          f"MDD: {ho.get('ho_mdd',0)*100:.1f}%, Sharpe: {ho.get('ho_sharpe',0):.2f}, "
          f"Trades: {trades}")

    # ── 4b. C2 gate OFF ──
    print(f"\n{'='*70}")
    print("(b) C2 gate OFF (no regime scaling)")
    print(f"{'='*70}")
    eq_c2_off, ret, wr, pf, trades, _, risk = run_vec_backtest(
        factors_3d, close_prices, open_prices, high_prices, low_prices,
        timing_gate_off, C2_INDICES, **vec_params,
    )
    ho = compute_holdout_metrics(eq_c2_off, dates, holdout_start)
    fp = compute_full_period_metrics(eq_c2_off)
    results["C2_gate_OFF"] = {"equity": eq_c2_off, "trades": trades, **fp, **ho}
    print(f"  Full: {ret*100:.1f}%, HO: {ho.get('ho_return',0)*100:.1f}%, "
          f"MDD: {ho.get('ho_mdd',0)*100:.1f}%, Sharpe: {ho.get('ho_sharpe',0):.2f}, "
          f"Trades: {trades}")

    # ── 4c. Equal-Weight gate ON ──
    print(f"\n{'='*70}")
    print("(c) Equal-Weight all A-share ETFs, gate ON")
    print(f"{'='*70}")
    ew_on = compute_equal_weight_portfolio(
        close_df, timing_gate_on, dates, a_share_mask, holdout_start
    )
    results["EW_gate_ON"] = ew_on
    print(f"  Full: {ew_on['full_return']*100:.1f}%, HO: {ew_on.get('ho_return',0)*100:.1f}%, "
          f"MDD: {ew_on.get('ho_mdd',0)*100:.1f}%, Sharpe: {ew_on.get('ho_sharpe',0):.2f}")

    # ── 4d. Equal-Weight gate OFF ──
    print(f"\n{'='*70}")
    print("(d) Equal-Weight all A-share ETFs, gate OFF (buy-and-hold)")
    print(f"{'='*70}")
    timing_no_gate = np.ones(T, dtype=np.float64)
    ew_off = compute_equal_weight_portfolio(
        close_df, timing_no_gate, dates, a_share_mask, holdout_start
    )
    results["EW_gate_OFF"] = ew_off
    print(f"  Full: {ew_off['full_return']*100:.1f}%, HO: {ew_off.get('ho_return',0)*100:.1f}%, "
          f"MDD: {ew_off.get('ho_mdd',0)*100:.1f}%, Sharpe: {ew_off.get('ho_sharpe',0):.2f}")

    # ── 4e. Random-2 gate ON (20 seeds) ──
    print(f"\n{'='*70}")
    print(f"(e) Random-2 selection, gate ON ({N_RANDOM_SEEDS} seeds)")
    print(f"{'='*70}")
    random_on_results = []
    for seed in range(N_RANDOM_SEEDS):
        r = run_random_selection_vec(
            factors_3d, close_prices, open_prices, high_prices, low_prices,
            timing_gate_on, factor_names,
            dates=dates, holdout_start=holdout_start, seed=seed,
            **vec_params,
        )
        random_on_results.append(r)
        if seed % 5 == 0:
            print(f"  seed={seed}: HO={r.get('ho_return',0)*100:.1f}%, "
                  f"MDD={r.get('ho_mdd',0)*100:.1f}%")

    # Median results
    ho_rets_on = [r.get("ho_return", 0) for r in random_on_results]
    ho_mdds_on = [r.get("ho_mdd", 0) for r in random_on_results]
    ho_sharpes_on = [r.get("ho_sharpe", 0) for r in random_on_results]
    full_rets_on = [r.get("full_return", 0) for r in random_on_results]
    median_idx_on = int(np.argsort(ho_rets_on)[len(ho_rets_on) // 2])
    results["Random2_gate_ON"] = {
        "ho_return_median": float(np.median(ho_rets_on)),
        "ho_return_mean": float(np.mean(ho_rets_on)),
        "ho_return_std": float(np.std(ho_rets_on)),
        "ho_mdd_median": float(np.median(ho_mdds_on)),
        "ho_sharpe_median": float(np.median(ho_sharpes_on)),
        "full_return_median": float(np.median(full_rets_on)),
        "all_ho_returns": ho_rets_on,
        "equity": random_on_results[median_idx_on]["equity_curve"],
    }
    print(f"  Median HO: {np.median(ho_rets_on)*100:.1f}% "
          f"(mean={np.mean(ho_rets_on)*100:.1f}%, std={np.std(ho_rets_on)*100:.1f}%)")
    print(f"  Median MDD: {np.median(ho_mdds_on)*100:.1f}%, "
          f"Sharpe: {np.median(ho_sharpes_on):.2f}")

    # ── 4f. Random-2 gate OFF (20 seeds) ──
    print(f"\n{'='*70}")
    print(f"(f) Random-2 selection, gate OFF ({N_RANDOM_SEEDS} seeds)")
    print(f"{'='*70}")
    random_off_results = []
    for seed in range(N_RANDOM_SEEDS):
        r = run_random_selection_vec(
            factors_3d, close_prices, open_prices, high_prices, low_prices,
            timing_gate_off, factor_names,
            dates=dates, holdout_start=holdout_start, seed=seed,
            **vec_params,
        )
        random_off_results.append(r)
        if seed % 5 == 0:
            print(f"  seed={seed}: HO={r.get('ho_return',0)*100:.1f}%, "
                  f"MDD={r.get('ho_mdd',0)*100:.1f}%")

    ho_rets_off = [r.get("ho_return", 0) for r in random_off_results]
    ho_mdds_off = [r.get("ho_mdd", 0) for r in random_off_results]
    ho_sharpes_off = [r.get("ho_sharpe", 0) for r in random_off_results]
    full_rets_off = [r.get("full_return", 0) for r in random_off_results]
    median_idx_off = int(np.argsort(ho_rets_off)[len(ho_rets_off) // 2])
    results["Random2_gate_OFF"] = {
        "ho_return_median": float(np.median(ho_rets_off)),
        "ho_return_mean": float(np.mean(ho_rets_off)),
        "ho_return_std": float(np.std(ho_rets_off)),
        "ho_mdd_median": float(np.median(ho_mdds_off)),
        "ho_sharpe_median": float(np.median(ho_sharpes_off)),
        "full_return_median": float(np.median(full_rets_off)),
        "all_ho_returns": ho_rets_off,
        "equity": random_off_results[median_idx_off]["equity_curve"],
    }
    print(f"  Median HO: {np.median(ho_rets_off)*100:.1f}% "
          f"(mean={np.mean(ho_rets_off)*100:.1f}%, std={np.std(ho_rets_off)*100:.1f}%)")
    print(f"  Median MDD: {np.median(ho_mdds_off)*100:.1f}%, "
          f"Sharpe: {np.median(ho_sharpes_off):.2f}")

    # ══════════════════════════════════════════════════════════════════════════
    # 5. DECOMPOSITION
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("ALPHA SOURCE DECOMPOSITION (Holdout Period)")
    print("=" * 90)

    c2_on_ho = results["C2_gate_ON"].get("ho_return", 0)
    c2_off_ho = results["C2_gate_OFF"].get("ho_return", 0)
    ew_on_ho = results["EW_gate_ON"].get("ho_return", 0)
    ew_off_ho = results["EW_gate_OFF"].get("ho_return", 0)
    rand_on_ho = results["Random2_gate_ON"]["ho_return_median"]
    rand_off_ho = results["Random2_gate_OFF"]["ho_return_median"]

    # Component breakdown
    pure_timing_beta = ew_on_ho - ew_off_ho
    c2_gate_contribution = c2_on_ho - c2_off_ho
    selection_alpha = c2_on_ho - rand_on_ho
    random_gate_lift = rand_on_ho - rand_off_ho
    interaction = c2_gate_contribution - pure_timing_beta

    print(f"\n  {'Variant':<35} {'HO Return':>10} {'HO MDD':>10} {'HO Sharpe':>10}")
    print(f"  {'-'*65}")
    print(f"  {'C2 gate ON (production)':<35} {c2_on_ho*100:>9.1f}% "
          f"{results['C2_gate_ON'].get('ho_mdd',0)*100:>9.1f}% "
          f"{results['C2_gate_ON'].get('ho_sharpe',0):>9.2f}")
    print(f"  {'C2 gate OFF':<35} {c2_off_ho*100:>9.1f}% "
          f"{results['C2_gate_OFF'].get('ho_mdd',0)*100:>9.1f}% "
          f"{results['C2_gate_OFF'].get('ho_sharpe',0):>9.2f}")
    print(f"  {'EqualWeight gate ON':<35} {ew_on_ho*100:>9.1f}% "
          f"{results['EW_gate_ON'].get('ho_mdd',0)*100:>9.1f}% "
          f"{results['EW_gate_ON'].get('ho_sharpe',0):>9.2f}")
    print(f"  {'EqualWeight gate OFF (B&H)':<35} {ew_off_ho*100:>9.1f}% "
          f"{results['EW_gate_OFF'].get('ho_mdd',0)*100:>9.1f}% "
          f"{results['EW_gate_OFF'].get('ho_sharpe',0):>9.2f}")
    print(f"  {'Random-2 gate ON (median)':<35} {rand_on_ho*100:>9.1f}% "
          f"{results['Random2_gate_ON']['ho_mdd_median']*100:>9.1f}% "
          f"{results['Random2_gate_ON']['ho_sharpe_median']:>9.2f}")
    print(f"  {'Random-2 gate OFF (median)':<35} {rand_off_ho*100:>9.1f}% "
          f"{results['Random2_gate_OFF']['ho_mdd_median']*100:>9.1f}% "
          f"{results['Random2_gate_OFF']['ho_sharpe_median']:>9.2f}")

    print(f"\n  {'DECOMPOSITION':=^65}")
    print(f"  C2 total HO return:                       {c2_on_ho*100:>+8.1f}pp")
    print(f"  = Market beta (EW gate OFF):               {ew_off_ho*100:>+8.1f}pp")
    print(f"  + Pure timing beta (EW ON - EW OFF):       {pure_timing_beta*100:>+8.1f}pp")
    print(f"  + Selection alpha (C2 ON - Random ON):     {selection_alpha*100:>+8.1f}pp")
    print(f"  + Timing x selection interaction:           {interaction*100:>+8.1f}pp")
    accounted = ew_off_ho + pure_timing_beta + selection_alpha + interaction
    residual = c2_on_ho - accounted
    print(f"  = Sum of components:                       {accounted*100:>+8.1f}pp")
    if abs(residual) > 0.001:
        print(f"  Residual (rounding):                       {residual*100:>+8.1f}pp")

    print(f"\n  {'ALTERNATIVE DECOMPOSITION (via Random baseline)':=^65}")
    print(f"  Random gate lift (Rand ON - Rand OFF):     {random_gate_lift*100:>+8.1f}pp")
    print(f"  C2 gate contribution (C2 ON - C2 OFF):    {c2_gate_contribution*100:>+8.1f}pp")
    print(f"  C2 factor selection (C2 OFF - Rand OFF):   {(c2_off_ho - rand_off_ho)*100:>+8.1f}pp")

    # Risk assessment
    print(f"\n  {'RISK ASSESSMENT':=^65}")
    alpha_pct = (selection_alpha / c2_on_ho * 100) if c2_on_ho > 0.001 else 0
    timing_pct = (pure_timing_beta / c2_on_ho * 100) if c2_on_ho > 0.001 else 0
    market_pct = (ew_off_ho / c2_on_ho * 100) if c2_on_ho > 0.001 else 0
    print(f"  Market beta share:     {market_pct:>6.1f}%  of C2 total")
    print(f"  Timing beta share:     {timing_pct:>6.1f}%  of C2 total")
    print(f"  Selection alpha share: {alpha_pct:>6.1f}%  of C2 total")

    if selection_alpha * 100 < 10:
        print(f"\n  WARNING: Selection alpha < 10pp ({selection_alpha*100:.1f}pp)")
        print(f"  C2 is primarily a timing strategy, not a stock-picking strategy")
    else:
        print(f"\n  Selection alpha is substantial ({selection_alpha*100:.1f}pp)")
        print(f"  C2 has genuine factor selection value beyond regime timing")

    # ── 6. Save results ──
    out_dir = ROOT / "results" / f"alpha_decomposition_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Summary CSV
    summary_rows = []
    for name, data in results.items():
        row = {"variant": name}
        if "ho_return_median" in data:
            row["ho_return"] = data["ho_return_median"]
            row["ho_mdd"] = data["ho_mdd_median"]
            row["ho_sharpe"] = data["ho_sharpe_median"]
            row["full_return"] = data["full_return_median"]
        else:
            row["ho_return"] = data.get("ho_return", 0)
            row["ho_mdd"] = data.get("ho_mdd", 0)
            row["ho_sharpe"] = data.get("ho_sharpe", 0)
            row["full_return"] = data.get("full_return", 0)
        row["trades"] = data.get("trades", 0)
        summary_rows.append(row)
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(out_dir / "decomposition_summary.csv", index=False)

    # Decomposition values
    decomp = {
        "c2_on_ho": c2_on_ho,
        "c2_off_ho": c2_off_ho,
        "ew_on_ho": ew_on_ho,
        "ew_off_ho": ew_off_ho,
        "rand_on_ho_median": rand_on_ho,
        "rand_off_ho_median": rand_off_ho,
        "pure_timing_beta": pure_timing_beta,
        "selection_alpha": selection_alpha,
        "c2_gate_contribution": c2_gate_contribution,
        "interaction": interaction,
        "random_gate_lift": random_gate_lift,
        "c2_factor_selection_no_gate": c2_off_ho - rand_off_ho,
        "alpha_pct_of_total": alpha_pct,
        "timing_pct_of_total": timing_pct,
        "market_pct_of_total": market_pct,
    }
    pd.Series(decomp).to_csv(out_dir / "decomposition_values.csv")

    # Random seed details
    rand_details = pd.DataFrame([
        {"seed": r["seed"], "gate": "ON", "ho_return": r.get("ho_return", 0),
         "ho_mdd": r.get("ho_mdd", 0), "trades": r.get("trades", 0)}
        for r in random_on_results
    ] + [
        {"seed": r["seed"], "gate": "OFF", "ho_return": r.get("ho_return", 0),
         "ho_mdd": r.get("ho_mdd", 0), "trades": r.get("trades", 0)}
        for r in random_off_results
    ])
    rand_details.to_csv(out_dir / "random_seed_details.csv", index=False)

    # Equity curves
    eq_df = pd.DataFrame(index=range(T))
    eq_df["date"] = [str(d) for d in dates]
    eq_df["C2_gate_ON"] = results["C2_gate_ON"]["equity"]
    eq_df["C2_gate_OFF"] = results["C2_gate_OFF"]["equity"]
    eq_df["EW_gate_ON"] = results["EW_gate_ON"]["equity_curve"]
    eq_df["EW_gate_OFF"] = results["EW_gate_OFF"]["equity_curve"]
    eq_df["Random2_gate_ON_median"] = results["Random2_gate_ON"]["equity"]
    eq_df["Random2_gate_OFF_median"] = results["Random2_gate_OFF"]["equity"]
    eq_df.to_csv(out_dir / "equity_curves.csv", index=False)

    print(f"\n  Results saved to {out_dir}")

    return results, decomp


if __name__ == "__main__":
    results, decomp = main()
