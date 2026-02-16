#!/usr/bin/env python3
"""
WHEN/HOW Dimension Research — Stage 0 + Stage 1

Stage 0: Ensemble failure correlation (HOW)
  - Run VEC for composite_1 and core_4f
  - Analyze quarterly return correlation
  - Kill criterion: rho > 0.7 AND P(both_fail|one_fail) > 0.8

Stage 1: Cross-sectional return dispersion orthogonality (WHEN)
  - Compute 20-day return dispersion across 49 ETFs
  - Check orthogonality with existing regime gate (|rho| > 0.5 → KILL)
  - Test predictive power on composite_1 alpha (quartile monotonicity)
  - Train/holdout split validation (Rule 4)

Output: research report with kill/proceed decision for each stage.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats as sp_stats

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.factor_cache import FactorCache
from etf_strategy.core.frozen_params import load_frozen_config
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import (
    shift_timing_signal,
    generate_rebalance_schedule,
)
from etf_strategy.regime_gate import (
    compute_regime_gate_arr,
    compute_volatility_gate_raw,
    gate_stats,
)
from etf_strategy.core.execution_model import load_execution_model
from etf_strategy.core.cost_model import load_cost_model, build_cost_array
from etf_strategy.core.frozen_params import FrozenETFPool

# Import the VEC backtest from batch script
from batch_vec_backtest import run_vec_backtest, calculate_atr


def load_data_and_config():
    """Load data exactly as batch_vec_backtest.py does."""
    config_path = ROOT / "configs/combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    frozen = load_frozen_config(config, config_path=str(config_path))
    print(f"[OK] Frozen params: version={frozen.version}")

    backtest_config = config.get("backtest", {})
    FREQ = backtest_config["freq"]
    POS_SIZE = backtest_config["pos_size"]
    LOOKBACK = backtest_config.get("lookback") or backtest_config.get("lookback_window")
    INITIAL_CAPITAL = float(backtest_config["initial_capital"])
    COMMISSION_RATE = float(backtest_config["commission_rate"])

    exec_model = load_execution_model(config)
    USE_T1_OPEN = exec_model.is_t1_open

    cost_model = load_cost_model(config)
    qdii_set = set(FrozenETFPool().qdii_codes)

    # Load OHLCV
    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )

    # Compute factors (cached)
    factor_cache = FactorCache(
        cache_dir=Path(config["data"].get("cache_dir") or ".cache")
    )
    cached = factor_cache.get_or_compute(
        ohlcv=ohlcv, config=config, data_dir=loader.data_dir
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

    # Cost array
    COST_ARR = build_cost_array(cost_model, list(etf_codes), qdii_set)

    # Timing
    timing_config = backtest_config.get("timing", {})
    extreme_threshold = timing_config.get("extreme_threshold", -0.4)
    extreme_position = timing_config.get("extreme_position", 0.3)
    timing_module = LightTimingModule(
        extreme_threshold=extreme_threshold, extreme_position=extreme_position
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
        ohlcv["close"], dates, backtest_config=backtest_config
    )
    timing_arr = (timing_arr.astype(np.float64) * gate_arr.astype(np.float64)).astype(
        np.float64
    )

    # Risk control
    risk_config = backtest_config.get("risk_control", {})
    stop_method = risk_config.get("stop_method", "fixed")
    use_atr_stop = stop_method == "atr"
    trailing_stop_pct = risk_config.get("trailing_stop_pct", 0.08)
    atr_window = risk_config.get("atr_window", 14)
    atr_multiplier = risk_config.get("atr_multiplier", 3.0)
    stop_on_rebalance_only = risk_config.get("stop_check_on_rebalance_only", False)
    profit_ladders = risk_config.get("profit_ladders", [])
    circuit_breaker_config = risk_config.get("circuit_breaker", {})
    circuit_breaker_day = circuit_breaker_config.get("max_drawdown_day", 0.0)
    circuit_breaker_total = circuit_breaker_config.get("max_drawdown_total", 0.0)
    circuit_recovery_days = circuit_breaker_config.get("recovery_days", 5)
    cooldown_days = risk_config.get("cooldown_days", 0)
    leverage_cap = risk_config.get("leverage_cap", 1.0)

    if use_atr_stop:
        atr_arr = calculate_atr(high_prices, low_prices, close_prices, window=atr_window)
    else:
        atr_arr = None

    # Hysteresis
    hyst_config = backtest_config.get("hysteresis", {})
    DELTA_RANK = float(hyst_config.get("delta_rank", 0.0))
    MIN_HOLD_DAYS = int(hyst_config.get("min_hold_days", 0))

    return {
        "config": config,
        "factors_3d": factors_3d,
        "factor_names": factor_names,
        "dates": dates,
        "etf_codes": etf_codes,
        "ohlcv": ohlcv,
        "close_prices": close_prices,
        "open_prices": open_prices,
        "high_prices": high_prices,
        "low_prices": low_prices,
        "timing_arr": timing_arr,
        "gate_arr": gate_arr,
        "COST_ARR": COST_ARR,
        "FREQ": FREQ,
        "POS_SIZE": POS_SIZE,
        "LOOKBACK": LOOKBACK,
        "INITIAL_CAPITAL": INITIAL_CAPITAL,
        "COMMISSION_RATE": COMMISSION_RATE,
        "USE_T1_OPEN": USE_T1_OPEN,
        "DELTA_RANK": DELTA_RANK,
        "MIN_HOLD_DAYS": MIN_HOLD_DAYS,
        # Risk params
        "use_atr_stop": use_atr_stop,
        "trailing_stop_pct": trailing_stop_pct,
        "atr_arr": atr_arr,
        "atr_multiplier": atr_multiplier,
        "stop_on_rebalance_only": stop_on_rebalance_only,
        "profit_ladders": profit_ladders,
        "circuit_breaker_day": circuit_breaker_day,
        "circuit_breaker_total": circuit_breaker_total,
        "circuit_recovery_days": circuit_recovery_days,
        "cooldown_days": cooldown_days,
        "leverage_cap": leverage_cap,
    }


def run_combo_vec(ctx, factors, signs, icirs):
    """Run VEC backtest for a single combo, return equity curve."""
    factor_index_map = {name: idx for idx, name in enumerate(ctx["factor_names"])}
    factor_indices = [factor_index_map[f] for f in factors]

    # Apply sign * weight to factors_3d
    n_factors = len(factors)
    abs_icirs = [abs(ic) for ic in icirs]
    total = sum(abs_icirs)
    weights = [aic / total for aic in abs_icirs] if total > 0 else [1.0 / n_factors] * n_factors

    needs_transform = any(
        s != 1 or w != 1.0 / n_factors for s, w in zip(signs, weights)
    )
    if needs_transform:
        factors_3d_local = ctx["factors_3d"].copy()
        for i, (sign, weight) in enumerate(zip(signs, weights)):
            if sign != 1 or weight != 1.0:
                factors_3d_local[:, :, factor_indices[i]] *= sign * weight * n_factors
    else:
        factors_3d_local = ctx["factors_3d"]

    eq_curve, ret, wr, pf, trades, rounding, risk = run_vec_backtest(
        factors_3d_local,
        ctx["close_prices"],
        ctx["open_prices"],
        ctx["high_prices"],
        ctx["low_prices"],
        ctx["timing_arr"],
        factor_indices,
        freq=ctx["FREQ"],
        pos_size=ctx["POS_SIZE"],
        initial_capital=ctx["INITIAL_CAPITAL"],
        commission_rate=ctx["COMMISSION_RATE"],
        lookback=ctx["LOOKBACK"],
        cost_arr=ctx["COST_ARR"],
        use_t1_open=ctx["USE_T1_OPEN"],
        delta_rank=ctx["DELTA_RANK"],
        min_hold_days=ctx["MIN_HOLD_DAYS"],
        # Risk params
        use_atr_stop=ctx["use_atr_stop"],
        trailing_stop_pct=ctx["trailing_stop_pct"],
        atr_arr=ctx["atr_arr"],
        atr_multiplier=ctx["atr_multiplier"],
        stop_on_rebalance_only=ctx["stop_on_rebalance_only"],
        profit_ladders=ctx["profit_ladders"],
        circuit_breaker_day=ctx["circuit_breaker_day"],
        circuit_breaker_total=ctx["circuit_breaker_total"],
        circuit_recovery_days=ctx["circuit_recovery_days"],
        cooldown_days=ctx["cooldown_days"],
        leverage_cap=ctx["leverage_cap"],
    )

    return {
        "equity_curve": eq_curve,
        "total_return": ret,
        "win_rate": wr,
        "profit_factor": pf,
        "num_trades": trades,
        "risk": risk,
    }


# ═══════════════════════════════════════════════════════════════════
# Stage 0: Ensemble Failure Correlation
# ═══════════════════════════════════════════════════════════════════


def stage0_ensemble_correlation(ctx, eq_composite, eq_core4f):
    """Analyze quarterly return correlation between two strategies."""
    print("\n" + "=" * 70)
    print("STAGE 0: Ensemble Failure Correlation (HOW)")
    print("=" * 70)

    dates = ctx["dates"]
    LOOKBACK = ctx["LOOKBACK"]
    training_end = pd.Timestamp(ctx["config"]["data"]["training_end_date"])

    # Convert equity curves to pandas Series
    eq_a = pd.Series(eq_composite, index=dates)
    eq_b = pd.Series(eq_core4f, index=dates)

    # Skip lookback warmup period
    valid_start = dates[LOOKBACK]
    eq_a = eq_a[eq_a.index >= valid_start]
    eq_b = eq_b[eq_b.index >= valid_start]

    # Split into train/holdout
    train_a = eq_a[eq_a.index <= training_end]
    train_b = eq_b[eq_b.index <= training_end]
    ho_a = eq_a[eq_a.index > training_end]
    ho_b = eq_b[eq_b.index > training_end]

    # Compute quarterly (63 trading days ≈ 1 quarter) returns
    def compute_segment_returns(eq_series, segment_days=63):
        """Split equity curve into non-overlapping segments, return period returns."""
        values = eq_series.values
        idx = eq_series.index
        segments = []
        segment_labels = []
        i = 0
        while i + segment_days <= len(values):
            seg_ret = values[i + segment_days - 1] / values[i] - 1.0
            segments.append(seg_ret)
            segment_labels.append(idx[i])
            i += segment_days
        # Handle remainder if > half a segment
        if len(values) - i > segment_days // 2 and i < len(values):
            seg_ret = values[-1] / values[i] - 1.0
            segments.append(seg_ret)
            segment_labels.append(idx[i])
        return np.array(segments), segment_labels

    # Train quarterly
    train_ret_a, train_labels = compute_segment_returns(train_a)
    train_ret_b, _ = compute_segment_returns(train_b)
    n_train = min(len(train_ret_a), len(train_ret_b))
    train_ret_a = train_ret_a[:n_train]
    train_ret_b = train_ret_b[:n_train]
    train_labels = train_labels[:n_train]

    # Holdout — use shorter segments (21 days ≈ 1 month) for more granularity
    ho_ret_a, ho_labels = compute_segment_returns(ho_a, segment_days=21)
    ho_ret_b, _ = compute_segment_returns(ho_b, segment_days=21)
    n_ho = min(len(ho_ret_a), len(ho_ret_b))
    ho_ret_a = ho_ret_a[:n_ho]
    ho_ret_b = ho_ret_b[:n_ho]
    ho_labels = ho_labels[:n_ho]

    # Combine all segments for overall analysis
    all_ret_a = np.concatenate([train_ret_a, ho_ret_a])
    all_ret_b = np.concatenate([train_ret_b, ho_ret_b])

    # --- Compute correlation ---
    if len(all_ret_a) >= 3:
        rho_all, pval_all = sp_stats.pearsonr(all_ret_a, all_ret_b)
    else:
        rho_all, pval_all = np.nan, np.nan

    if len(train_ret_a) >= 3:
        rho_train, pval_train = sp_stats.pearsonr(train_ret_a, train_ret_b)
    else:
        rho_train, pval_train = np.nan, np.nan

    if len(ho_ret_a) >= 3:
        rho_ho, pval_ho = sp_stats.pearsonr(ho_ret_a, ho_ret_b)
    else:
        rho_ho, pval_ho = np.nan, np.nan

    # --- 2×2 contingency table (train, quarterly) ---
    win_a_train = train_ret_a > 0
    win_b_train = train_ret_b > 0
    a_cell = np.sum(win_a_train & win_b_train)
    b_cell = np.sum(win_a_train & ~win_b_train)
    c_cell = np.sum(~win_a_train & win_b_train)
    d_cell = np.sum(~win_a_train & ~win_b_train)

    wr_a_train = np.mean(win_a_train) if len(win_a_train) > 0 else 0
    wr_b_train = np.mean(win_b_train) if len(win_b_train) > 0 else 0

    # Same for holdout
    if n_ho > 0:
        win_a_ho = ho_ret_a > 0
        win_b_ho = ho_ret_b > 0
        a_ho = np.sum(win_a_ho & win_b_ho)
        b_ho = np.sum(win_a_ho & ~win_b_ho)
        c_ho = np.sum(~win_a_ho & win_b_ho)
        d_ho = np.sum(~win_a_ho & ~win_b_ho)
        wr_a_ho = np.mean(win_a_ho)
        wr_b_ho = np.mean(win_b_ho)
    else:
        a_ho = b_ho = c_ho = d_ho = 0
        wr_a_ho = wr_b_ho = 0

    # P(both_fail)
    n_total_train = n_train
    p_both_fail_train = d_cell / n_total_train if n_total_train > 0 else 0
    p_theoretical_indep = (1 - wr_a_train) * (1 - wr_b_train)
    # P(both_fail|one_fail)
    one_fail_train = b_cell + c_cell + d_cell
    p_both_fail_given_one = d_cell / one_fail_train if one_fail_train > 0 else 0

    # --- Print Results ---
    print(f"\n--- Train Period ({n_train} quarterly segments) ---")
    print(f"  composite_1 win rate: {wr_a_train:.1%}")
    print(f"  core_4f     win rate: {wr_b_train:.1%}")
    print(f"  Pearson rho: {rho_train:.3f} (p={pval_train:.3f})")
    print(f"\n  Contingency table (quarterly):")
    print(f"                  core_4f WIN  core_4f LOSE")
    print(f"  composite_1 WIN    {a_cell:3d}          {b_cell:3d}")
    print(f"  composite_1 LOSE   {c_cell:3d}          {d_cell:3d}")
    print(f"\n  P(both_fail) = {p_both_fail_train:.3f}")
    print(f"  P(both_fail) under independence = {p_theoretical_indep:.3f}")
    print(f"  P(both_fail|at_least_one_fail) = {p_both_fail_given_one:.3f}")

    if n_ho > 0:
        p_both_fail_ho = d_ho / n_ho if n_ho > 0 else 0
        one_fail_ho = b_ho + c_ho + d_ho
        p_both_fail_given_one_ho = d_ho / one_fail_ho if one_fail_ho > 0 else 0
        print(f"\n--- Holdout Period ({n_ho} monthly segments) ---")
        print(f"  composite_1 win rate: {wr_a_ho:.1%}")
        print(f"  core_4f     win rate: {wr_b_ho:.1%}")
        print(f"  Pearson rho: {rho_ho:.3f} (p={pval_ho:.3f})")
        print(f"\n  Contingency table (monthly):")
        print(f"                  core_4f WIN  core_4f LOSE")
        print(f"  composite_1 WIN    {a_ho:3d}          {b_ho:3d}")
        print(f"  composite_1 LOSE   {c_ho:3d}          {d_ho:3d}")
        print(f"\n  P(both_fail) = {p_both_fail_ho:.3f}")
        print(f"  P(both_fail|at_least_one_fail) = {p_both_fail_given_one_ho:.3f}")

    print(f"\n--- Overall ({len(all_ret_a)} segments) ---")
    print(f"  Pearson rho: {rho_all:.3f} (p={pval_all:.3f})")

    # --- Kill Decision ---
    print(f"\n--- Stage 0 Kill Criteria ---")
    kill = False
    if rho_train > 0.7 and p_both_fail_given_one > 0.8:
        print(f"  KILL: rho={rho_train:.3f} > 0.7 AND P(both_fail|one_fail)={p_both_fail_given_one:.3f} > 0.8")
        print(f"  => Strategies are risk-redundant, ensemble has NO value")
        kill = True
    elif rho_train > 0.5:
        print(f"  MARGINAL: rho={rho_train:.3f} in (0.5, 0.7)")
        print(f"  => Ensemble has limited diversification benefit")
    else:
        print(f"  PROCEED: rho={rho_train:.3f} < 0.5")
        print(f"  => Strategies have meaningful decorrelation → proceed to Stage 2a")

    # Compute ensemble potential: if we picked the better strategy each period
    if n_train > 0:
        oracle_train = np.maximum(train_ret_a, train_ret_b)
        equal_blend_train = 0.5 * train_ret_a + 0.5 * train_ret_b
        print(f"\n--- Ensemble Potential (Train) ---")
        print(f"  composite_1 mean quarterly: {np.mean(train_ret_a):.3%}")
        print(f"  core_4f     mean quarterly: {np.mean(train_ret_b):.3%}")
        print(f"  Equal blend mean quarterly: {np.mean(equal_blend_train):.3%}")
        print(f"  Oracle (best) mean quarterly: {np.mean(oracle_train):.3%}")
        print(f"  Blend Sharpe (quarterly): {np.mean(equal_blend_train) / (np.std(equal_blend_train) + 1e-9):.3f}")
        print(f"  composite_1 Sharpe (quarterly): {np.mean(train_ret_a) / (np.std(train_ret_a) + 1e-9):.3f}")
        print(f"  core_4f Sharpe (quarterly): {np.mean(train_ret_b) / (np.std(train_ret_b) + 1e-9):.3f}")

    return {
        "rho_train": rho_train,
        "rho_ho": rho_ho,
        "rho_all": rho_all,
        "p_both_fail_train": p_both_fail_train,
        "p_both_fail_given_one_train": p_both_fail_given_one,
        "kill": kill,
        "contingency_train": (a_cell, b_cell, c_cell, d_cell),
        "n_train_segments": n_train,
        "n_ho_segments": n_ho,
    }


# ═══════════════════════════════════════════════════════════════════
# Stage 1: Cross-Sectional Return Dispersion
# ═══════════════════════════════════════════════════════════════════


def stage1_dispersion_analysis(ctx, eq_composite):
    """Check if cross-sectional return dispersion is orthogonal to regime gate
    and predictive of selection alpha."""
    print("\n" + "=" * 70)
    print("STAGE 1: Cross-Sectional Return Dispersion (WHEN)")
    print("=" * 70)

    dates = ctx["dates"]
    ohlcv = ctx["ohlcv"]
    etf_codes = ctx["etf_codes"]
    LOOKBACK = ctx["LOOKBACK"]
    FREQ = ctx["FREQ"]
    training_end = pd.Timestamp(ctx["config"]["data"]["training_end_date"])

    # 1. Compute cross-sectional return dispersion
    close_df = ohlcv["close"][etf_codes].reindex(dates)
    # Use 20-day returns (monthly horizon)
    ret_20d = close_df.pct_change(20)
    dispersion_20d = ret_20d.std(axis=1)  # cross-sectional std per day
    dispersion_20d.name = "dispersion_20d"

    # Also try 5-day (matching FREQ=5)
    ret_5d = close_df.pct_change(5)
    dispersion_5d = ret_5d.std(axis=1)
    dispersion_5d.name = "dispersion_5d"

    print(f"\n--- Dispersion Statistics ---")
    for disp, label in [(dispersion_5d, "5D"), (dispersion_20d, "20D")]:
        valid = disp.dropna()
        print(f"  {label}: mean={valid.mean():.4f}, std={valid.std():.4f}, "
              f"min={valid.min():.4f}, max={valid.max():.4f}")

    # 2. Compute regime gate signal (raw, unshifted)
    backtest_config = ctx["config"].get("backtest", {})
    gate_cfg = backtest_config.get("regime_gate", {}).get("volatility", {})
    regime_vol_raw = compute_volatility_gate_raw(
        ohlcv["close"][etf_codes],
        proxy_symbol=gate_cfg.get("proxy_symbol", "510300"),
        window=gate_cfg.get("window", 20),
        thresholds_pct=tuple(gate_cfg.get("thresholds_pct", [25, 30, 40])),
        exposures=tuple(gate_cfg.get("exposures", [1.0, 0.7, 0.4, 0.1])),
    )

    # For orthogonality, we want the continuous vol signal, not the binned exposure
    # Recompute the raw volatility measure
    proxy_close = close_df["510300"] if "510300" in close_df.columns else close_df.mean(axis=1)
    proxy_rets = proxy_close.pct_change()
    hv_20 = proxy_rets.rolling(window=20, min_periods=20).std() * np.sqrt(252) * 100
    hv_5d = hv_20.shift(5)
    regime_vol_continuous = (hv_20 + hv_5d) / 2

    # 3. Orthogonality check
    print(f"\n--- Orthogonality: Dispersion vs Regime Vol ---")

    # Align and drop NaN
    for disp, label in [(dispersion_5d, "5D"), (dispersion_20d, "20D")]:
        combined = pd.DataFrame({
            "disp": disp,
            "regime_vol": regime_vol_continuous,
        }).dropna()
        if len(combined) >= 20:
            rho, pval = sp_stats.pearsonr(combined["disp"], combined["regime_vol"])
            rho_s, pval_s = sp_stats.spearmanr(combined["disp"], combined["regime_vol"])
            print(f"  Dispersion {label} vs regime_vol: Pearson={rho:.3f} (p={pval:.2e}), "
                  f"Spearman={rho_s:.3f} (p={pval_s:.2e})")
            if abs(rho) > 0.5:
                print(f"    -> WARNING: |rho|={abs(rho):.3f} > 0.5 — same dimension as regime gate (Rule 31)")

    # 4. Predictive power: dispersion → next-period strategy returns
    print(f"\n--- Predictive Power: Dispersion → composite_1 Alpha ---")

    # Get equity curve as series
    eq_series = pd.Series(eq_composite, index=dates)
    valid_start = dates[LOOKBACK]
    eq_series = eq_series[eq_series.index >= valid_start]

    # Compute per-rebalance returns (FREQ=5 day returns)
    rebalance_schedule = generate_rebalance_schedule(len(dates), LOOKBACK, FREQ)
    rebal_dates = [dates[i] for i in rebalance_schedule]

    # Strategy returns per period
    strategy_rets = []
    disp_at_rebal_5d = []
    disp_at_rebal_20d = []
    rebal_dates_used = []

    for i in range(len(rebalance_schedule) - 1):
        t_start = rebalance_schedule[i]
        t_end = rebalance_schedule[i + 1]
        if t_start < LOOKBACK:
            continue
        # Strategy return this period
        r = eq_composite[t_end] / eq_composite[t_start] - 1.0
        strategy_rets.append(r)
        # Dispersion at signal time (t_start, shifted by 1 for lookahead prevention)
        sig_date = dates[max(t_start - 1, 0)]
        d5 = dispersion_5d.get(sig_date, np.nan)
        d20 = dispersion_20d.get(sig_date, np.nan)
        disp_at_rebal_5d.append(d5)
        disp_at_rebal_20d.append(d20)
        rebal_dates_used.append(dates[t_start])

    strategy_rets = np.array(strategy_rets)
    disp_at_rebal_5d = np.array(disp_at_rebal_5d)
    disp_at_rebal_20d = np.array(disp_at_rebal_20d)
    rebal_dates_arr = np.array(rebal_dates_used)

    # Split train/holdout
    is_train = np.array([d <= training_end for d in rebal_dates_used])
    is_ho = ~is_train

    for disp_vals, label in [(disp_at_rebal_5d, "5D"), (disp_at_rebal_20d, "20D")]:
        print(f"\n  === Dispersion {label} ===")

        for period_name, mask in [("Train", is_train), ("Holdout", is_ho), ("All", np.ones(len(strategy_rets), dtype=bool))]:
            rets_p = strategy_rets[mask]
            disp_p = disp_vals[mask]
            valid = ~np.isnan(disp_p) & ~np.isnan(rets_p)
            rets_v = rets_p[valid]
            disp_v = disp_p[valid]

            if len(rets_v) < 10:
                print(f"    {period_name}: insufficient data ({len(rets_v)} obs)")
                continue

            # Correlation
            rho_pd, pval_pd = sp_stats.pearsonr(disp_v, rets_v)
            rho_sp, pval_sp = sp_stats.spearmanr(disp_v, rets_v)

            # Quartile analysis
            quartiles = np.percentile(disp_v, [25, 50, 75])
            q_labels = np.digitize(disp_v, quartiles)  # 0, 1, 2, 3

            q_means = []
            q_counts = []
            for q in range(4):
                q_mask = q_labels == q
                q_means.append(np.mean(rets_v[q_mask]) if q_mask.sum() > 0 else np.nan)
                q_counts.append(q_mask.sum())

            # Check monotonicity
            valid_means = [m for m in q_means if not np.isnan(m)]
            if len(valid_means) >= 3:
                is_mono_up = all(valid_means[i] <= valid_means[i+1] for i in range(len(valid_means)-1))
                is_mono_down = all(valid_means[i] >= valid_means[i+1] for i in range(len(valid_means)-1))
                mono_str = "UP" if is_mono_up else ("DOWN" if is_mono_down else "NONE")
            else:
                mono_str = "INSUFFICIENT"

            print(f"    {period_name} ({len(rets_v)} obs):")
            print(f"      Pearson rho(disp,ret) = {rho_pd:.3f} (p={pval_pd:.3f})")
            print(f"      Spearman rho = {rho_sp:.3f} (p={pval_sp:.3f})")
            print(f"      Quartile returns (Q1=low_disp → Q4=high_disp):")
            for q in range(4):
                print(f"        Q{q+1}: mean_ret={q_means[q]:+.3%} (n={q_counts[q]})")
            print(f"      Monotonicity: {mono_str}")

    # 5. Kill decision
    print(f"\n--- Stage 1 Kill Criteria ---")

    # Check orthogonality with 20D dispersion (primary)
    combined = pd.DataFrame({
        "disp": dispersion_20d,
        "regime_vol": regime_vol_continuous,
    }).dropna()
    rho_orth = sp_stats.pearsonr(combined["disp"], combined["regime_vol"])[0] if len(combined) >= 20 else np.nan

    kill_orthogonal = abs(rho_orth) > 0.5 if not np.isnan(rho_orth) else False

    # Check monotonicity in train
    train_mask_valid = is_train & ~np.isnan(disp_at_rebal_20d) & ~np.isnan(strategy_rets)
    train_rets_v = strategy_rets[train_mask_valid]
    train_disp_v = disp_at_rebal_20d[train_mask_valid]
    if len(train_rets_v) >= 10:
        quartiles_t = np.percentile(train_disp_v, [25, 50, 75])
        q_labels_t = np.digitize(train_disp_v, quartiles_t)
        q_means_train = [np.mean(train_rets_v[q_labels_t == q]) for q in range(4)]
        valid_q = [m for m in q_means_train if not np.isnan(m)]
        kill_no_mono = not (
            all(valid_q[i] <= valid_q[i+1] for i in range(len(valid_q)-1))
            or all(valid_q[i] >= valid_q[i+1] for i in range(len(valid_q)-1))
        ) if len(valid_q) >= 3 else True
    else:
        kill_no_mono = True

    # Check train/holdout consistency
    ho_mask_valid = is_ho & ~np.isnan(disp_at_rebal_20d) & ~np.isnan(strategy_rets)
    ho_rets_v = strategy_rets[ho_mask_valid]
    ho_disp_v = disp_at_rebal_20d[ho_mask_valid]
    if len(ho_rets_v) >= 8 and len(train_rets_v) >= 10:
        rho_train_disp = sp_stats.spearmanr(train_disp_v, train_rets_v)[0]
        rho_ho_disp = sp_stats.spearmanr(ho_disp_v, ho_rets_v)[0]
        kill_reversal = (rho_train_disp * rho_ho_disp < 0)  # sign flip
    else:
        kill_reversal = False  # insufficient data, don't kill
        rho_train_disp = rho_ho_disp = np.nan

    if kill_orthogonal:
        print(f"  KILL (orthogonality): |rho(disp, regime_vol)|={abs(rho_orth):.3f} > 0.5")
        print(f"  => Dispersion is same dimension as existing regime gate (Rule 31)")
    elif kill_no_mono:
        print(f"  KILL (monotonicity): No monotonic relationship between dispersion and returns")
        print(f"  => Dispersion has no predictive power for selection alpha")
    elif kill_reversal:
        print(f"  KILL (reversal): Train rho={rho_train_disp:.3f}, HO rho={rho_ho_disp:.3f} (sign flip)")
        print(f"  => Direction reversal between train and holdout → noise fitting")
    else:
        print(f"  PROCEED: Orthogonal (|rho|={abs(rho_orth):.3f}), "
              f"monotonic, no reversal")
        print(f"  => Dispersion may be a valid WHEN signal → proceed to Stage 2b")

    return {
        "rho_orthogonality": rho_orth,
        "kill_orthogonal": kill_orthogonal,
        "kill_no_mono": kill_no_mono,
        "kill_reversal": kill_reversal,
        "rho_train_disp": rho_train_disp if 'rho_train_disp' in dir() else np.nan,
        "rho_ho_disp": rho_ho_disp if 'rho_ho_disp' in dir() else np.nan,
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════
# Stage 2a: Ensemble Variants (Capital Split + Score Blend)
# ═══════════════════════════════════════════════════════════════════


def stage2a_ensemble_variants(ctx, eq_composite_ps2, eq_core4f_ps2):
    """Test ensemble variants with POS_SIZE=2 equity curves via capital split."""
    print("\n" + "=" * 70)
    print("STAGE 2a: Ensemble Validation (Capital Split)")
    print("=" * 70)

    dates = ctx["dates"]
    LOOKBACK = ctx["LOOKBACK"]
    training_end = pd.Timestamp(ctx["config"]["data"]["training_end_date"])

    # Run both strategies with POS_SIZE=1
    COMPOSITE_1 = {
        "factors": ["ADX_14D", "BREAKOUT_20D", "MARGIN_BUY_RATIO", "PRICE_POSITION_120D", "SHARE_CHG_5D"],
        "signs": [1, 1, -1, 1, -1],
        "icirs": [0.440, 18.579, -5.101, 4.542, -2.306],
    }
    CORE_4F = {
        "factors": ["MARGIN_CHG_10D", "PRICE_POSITION_120D", "SHARE_CHG_20D", "SLOPE_20D"],
        "signs": [-1, 1, -1, 1],
        "icirs": [-3.387, 4.542, -1.807, 3.125],
    }

    # Override POS_SIZE to 1 for each sub-strategy
    ctx_ps1 = dict(ctx)
    ctx_ps1["POS_SIZE"] = 1

    print("\n  Running composite_1 (POS_SIZE=1)...")
    res_c1_ps1 = run_combo_vec(ctx_ps1, **COMPOSITE_1)
    print(f"    return={res_c1_ps1['total_return']:.3%}, "
          f"sharpe={res_c1_ps1['risk']['sharpe_ratio']:.3f}, "
          f"mdd={res_c1_ps1['risk']['max_drawdown']:.3%}")

    print("  Running core_4f (POS_SIZE=1)...")
    res_c4_ps1 = run_combo_vec(ctx_ps1, **CORE_4F)
    print(f"    return={res_c4_ps1['total_return']:.3%}, "
          f"sharpe={res_c4_ps1['risk']['sharpe_ratio']:.3f}, "
          f"mdd={res_c4_ps1['risk']['max_drawdown']:.3%}")

    # Capital Split: equal-weight blend of equity curves
    eq_c1 = res_c1_ps1["equity_curve"]
    eq_c4 = res_c4_ps1["equity_curve"]
    eq_blend = 0.5 * eq_c1 + 0.5 * eq_c4  # 50/50 capital allocation

    # Compute blend metrics
    T = len(eq_blend)
    start_day = LOOKBACK
    # Daily returns from blend
    daily_rets = np.diff(eq_blend[start_day:]) / eq_blend[start_day:-1]

    # Total return
    blend_total_ret = eq_blend[-1] / eq_blend[start_day] - 1.0
    trading_days = T - start_day
    years = trading_days / 252.0

    # Annual return
    blend_ann_ret = (1.0 + blend_total_ret) ** (1.0 / years) - 1.0 if years > 0 else 0.0

    # Sharpe
    blend_vol = np.std(daily_rets) * np.sqrt(252) if len(daily_rets) > 0 else 1e-9
    blend_sharpe = blend_ann_ret / blend_vol if blend_vol > 0 else 0.0

    # Max drawdown
    cummax = np.maximum.accumulate(eq_blend[start_day:])
    drawdowns = (eq_blend[start_day:] - cummax) / cummax
    blend_mdd = abs(float(np.min(drawdowns)))

    # Calmar
    blend_calmar = blend_ann_ret / blend_mdd if blend_mdd > 0 else 0.0

    # --- Train / Holdout split ---
    train_end_idx = None
    for i, d in enumerate(dates):
        if d <= training_end:
            train_end_idx = i
    if train_end_idx is None:
        train_end_idx = len(dates) - 1

    # Train metrics
    train_eq = eq_blend[start_day:train_end_idx + 1]
    train_ret = train_eq[-1] / train_eq[0] - 1.0 if len(train_eq) > 1 else 0.0
    train_daily = np.diff(train_eq) / train_eq[:-1] if len(train_eq) > 1 else np.array([0.0])
    train_years = len(train_daily) / 252.0
    train_ann_ret = (1.0 + train_ret) ** (1.0 / train_years) - 1.0 if train_years > 0 else 0.0
    train_vol = np.std(train_daily) * np.sqrt(252) if len(train_daily) > 0 else 1e-9
    train_sharpe = train_ann_ret / train_vol if train_vol > 0 else 0.0
    train_cummax = np.maximum.accumulate(train_eq)
    train_dd = (train_eq - train_cummax) / train_cummax
    train_mdd = abs(float(np.min(train_dd)))

    # Holdout metrics
    ho_eq = eq_blend[train_end_idx:]
    ho_ret = ho_eq[-1] / ho_eq[0] - 1.0 if len(ho_eq) > 1 else 0.0
    ho_daily = np.diff(ho_eq) / ho_eq[:-1] if len(ho_eq) > 1 else np.array([0.0])
    ho_years = len(ho_daily) / 252.0
    ho_ann_ret = (1.0 + ho_ret) ** (1.0 / ho_years) - 1.0 if ho_years > 0 else 0.0
    ho_vol = np.std(ho_daily) * np.sqrt(252) if len(ho_daily) > 0 else 1e-9
    ho_sharpe = ho_ann_ret / ho_vol if ho_vol > 0 else 0.0
    ho_cummax = np.maximum.accumulate(ho_eq)
    ho_dd = (ho_eq - ho_cummax) / ho_cummax
    ho_mdd = abs(float(np.min(ho_dd)))

    # Reference: standalone composite_1 PS=2 metrics
    c1_ps2_eq = eq_composite_ps2
    c1_daily = np.diff(c1_ps2_eq[start_day:]) / c1_ps2_eq[start_day:-1]
    c1_total = c1_ps2_eq[-1] / c1_ps2_eq[start_day] - 1.0
    c1_ann = (1.0 + c1_total) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    c1_vol = np.std(c1_daily) * np.sqrt(252)
    c1_sharpe = c1_ann / c1_vol if c1_vol > 0 else 0.0
    c1_cummax = np.maximum.accumulate(c1_ps2_eq[start_day:])
    c1_dd = (c1_ps2_eq[start_day:] - c1_cummax) / c1_cummax
    c1_mdd = abs(float(np.min(c1_dd)))

    print(f"\n--- Full Period Results ---")
    print(f"  {'Metric':<20} {'composite_1(PS=2)':<20} {'Blend(2×PS=1)':<20} {'Delta':<12}")
    print(f"  {'Total Return':<20} {c1_total:>18.1%} {blend_total_ret:>18.1%} {blend_total_ret - c1_total:>+10.1%}")
    print(f"  {'Annual Return':<20} {c1_ann:>18.1%} {blend_ann_ret:>18.1%} {blend_ann_ret - c1_ann:>+10.1%}")
    print(f"  {'Sharpe':<20} {c1_sharpe:>18.3f} {blend_sharpe:>18.3f} {blend_sharpe - c1_sharpe:>+10.3f}")
    print(f"  {'Max DD':<20} {c1_mdd:>18.1%} {blend_mdd:>18.1%} {blend_mdd - c1_mdd:>+10.1%}")
    print(f"  {'Calmar':<20} {c1_ann / c1_mdd if c1_mdd > 0 else 0:>18.3f} {blend_calmar:>18.3f}")

    print(f"\n--- Train Period ---")
    print(f"  Blend: ret={train_ret:.1%}, sharpe={train_sharpe:.3f}, mdd={train_mdd:.1%}")

    print(f"\n--- Holdout Period ---")
    print(f"  Blend: ret={ho_ret:.1%}, sharpe={ho_sharpe:.3f}, mdd={ho_mdd:.1%}")

    # Kill criteria
    print(f"\n--- Stage 2a Kill Criteria ---")
    # Compare to standalone composite_1 (the better Sharpe strategy)
    sharpe_threshold = c1_sharpe + 0.10
    mdd_threshold = c1_mdd + 0.03

    kill_sharpe = blend_sharpe < sharpe_threshold
    kill_mdd = blend_mdd > mdd_threshold

    if kill_sharpe:
        print(f"  KILL (Sharpe): Blend {blend_sharpe:.3f} < threshold {sharpe_threshold:.3f} "
              f"(standalone + 0.10)")
    if kill_mdd:
        print(f"  KILL (MDD): Blend {blend_mdd:.1%} > threshold {mdd_threshold:.1%} "
              f"(standalone + 3pp)")
    if not kill_sharpe and not kill_mdd:
        print(f"  PROCEED: Blend Sharpe {blend_sharpe:.3f} >= {sharpe_threshold:.3f} "
              f"and MDD {blend_mdd:.1%} <= {mdd_threshold:.1%}")

    # Also check: does blend beat standalone on any risk-adjusted metric?
    print(f"\n--- Value Assessment ---")
    if blend_sharpe > c1_sharpe:
        print(f"  Blend Sharpe ({blend_sharpe:.3f}) > standalone ({c1_sharpe:.3f}): YES (+{blend_sharpe - c1_sharpe:.3f})")
    else:
        print(f"  Blend Sharpe ({blend_sharpe:.3f}) > standalone ({c1_sharpe:.3f}): NO ({blend_sharpe - c1_sharpe:+.3f})")

    if blend_mdd < c1_mdd:
        print(f"  Blend MDD ({blend_mdd:.1%}) < standalone ({c1_mdd:.1%}): YES (improvement)")
    else:
        print(f"  Blend MDD ({blend_mdd:.1%}) < standalone ({c1_mdd:.1%}): NO (worse)")

    if blend_calmar > c1_ann / c1_mdd:
        print(f"  Blend Calmar ({blend_calmar:.3f}) > standalone ({c1_ann / c1_mdd:.3f}): YES")
    else:
        print(f"  Blend Calmar ({blend_calmar:.3f}) > standalone ({c1_ann / c1_mdd:.3f}): NO")

    return {
        "blend_total_ret": blend_total_ret,
        "blend_sharpe": blend_sharpe,
        "blend_mdd": blend_mdd,
        "blend_calmar": blend_calmar,
        "standalone_sharpe": c1_sharpe,
        "standalone_mdd": c1_mdd,
        "kill_sharpe": kill_sharpe,
        "kill_mdd": kill_mdd,
        "ho_ret": ho_ret,
        "ho_sharpe": ho_sharpe,
        "ho_mdd": ho_mdd,
    }


def main():
    print("=" * 70)
    print("WHEN/HOW Dimension Research — Stage 0 + Stage 1 + Stage 2a")
    print(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load data
    print("\n[1/5] Loading data and computing factors...")
    ctx = load_data_and_config()
    print(f"  Data: {len(ctx['dates'])} days × {len(ctx['etf_codes'])} ETFs × "
          f"{len(ctx['factor_names'])} factors")

    # Define the two strategies
    COMPOSITE_1 = {
        "factors": ["ADX_14D", "BREAKOUT_20D", "MARGIN_BUY_RATIO", "PRICE_POSITION_120D", "SHARE_CHG_5D"],
        "signs": [1, 1, -1, 1, -1],
        "icirs": [0.440, 18.579, -5.101, 4.542, -2.306],
    }
    CORE_4F = {
        "factors": ["MARGIN_CHG_10D", "PRICE_POSITION_120D", "SHARE_CHG_20D", "SLOPE_20D"],
        "signs": [-1, 1, -1, 1],
        "icirs": [-3.387, 4.542, -1.807, 3.125],
    }

    # Run VEC for both strategies with POS_SIZE=2 (standard)
    print("\n[2/5] Running VEC backtest for composite_1 (POS_SIZE=2)...")
    res_c1 = run_combo_vec(ctx, **COMPOSITE_1)
    print(f"  composite_1: return={res_c1['total_return']:.3%}, "
          f"sharpe={res_c1['risk']['sharpe_ratio']:.3f}, "
          f"mdd={res_c1['risk']['max_drawdown']:.3%}, "
          f"trades={res_c1['num_trades']}")

    print("\n[3/5] Running VEC backtest for core_4f (POS_SIZE=2)...")
    res_c4 = run_combo_vec(ctx, **CORE_4F)
    print(f"  core_4f: return={res_c4['total_return']:.3%}, "
          f"sharpe={res_c4['risk']['sharpe_ratio']:.3f}, "
          f"mdd={res_c4['risk']['max_drawdown']:.3%}, "
          f"trades={res_c4['num_trades']}")

    # Stage 0: Ensemble correlation
    print("\n[4/5] Running Stage 0 + Stage 1 analysis...")
    s0 = stage0_ensemble_correlation(ctx, res_c1["equity_curve"], res_c4["equity_curve"])

    # Stage 1: Dispersion analysis
    s1 = stage1_dispersion_analysis(ctx, res_c1["equity_curve"])

    # Stage 2a: Ensemble variants (if Stage 0 not killed)
    s2a = None
    if not s0["kill"]:
        print("\n[5/5] Running Stage 2a: Ensemble validation...")
        s2a = stage2a_ensemble_variants(ctx, res_c1["equity_curve"], res_c4["equity_curve"])
    else:
        print("\n[5/5] Stage 2a skipped (Stage 0 killed)")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\nStage 0 (Ensemble HOW):")
    print(f"  Train rho = {s0['rho_train']:.3f}")
    print(f"  P(both_fail) = {s0['p_both_fail_train']:.3f}")
    print(f"  Decision: {'KILL' if s0['kill'] else 'PROCEED (marginal)'}")

    print(f"\nStage 1 (Dispersion WHEN):")
    print(f"  Orthogonality |rho| = {abs(s1['rho_orthogonality']):.3f}")
    killed_reasons = []
    if s1["kill_orthogonal"]:
        killed_reasons.append("orthogonality")
    if s1["kill_no_mono"]:
        killed_reasons.append("no monotonicity")
    if s1["kill_reversal"]:
        killed_reasons.append("train/HO reversal")
    if killed_reasons:
        print(f"  Decision: KILL ({', '.join(killed_reasons)})")
    else:
        print(f"  Decision: PROCEED")

    if s2a:
        print(f"\nStage 2a (Ensemble Capital Split):")
        print(f"  Blend Sharpe = {s2a['blend_sharpe']:.3f} (standalone = {s2a['standalone_sharpe']:.3f})")
        print(f"  Blend MDD = {s2a['blend_mdd']:.1%} (standalone = {s2a['standalone_mdd']:.1%})")
        print(f"  HO: ret={s2a['ho_ret']:.1%}, sharpe={s2a['ho_sharpe']:.3f}, mdd={s2a['ho_mdd']:.1%}")
        killed_2a = s2a.get("kill_sharpe", False) or s2a.get("kill_mdd", False)
        print(f"  Decision: {'KILL' if killed_2a else 'PROCEED'}")

    print(f"\n--- Final Verdict ---")
    all_killed = True
    if not s0["kill"]:
        if s2a and not (s2a.get("kill_sharpe", False) or s2a.get("kill_mdd", False)):
            all_killed = False
            print(f"  Ensemble (Capital Split) PASSED → candidate for v9.0")
    if not (s1["kill_orthogonal"] or s1["kill_no_mono"] or s1["kill_reversal"]):
        all_killed = False
        print(f"  Dispersion WHEN PASSED → candidate for v9.0")
    if all_killed:
        print(f"  ALL DIRECTIONS KILLED → credible negative conclusion (Rule 29)")
        print(f"  v8.0 = ceiling with current data and methodology")
        print(f"  Recommended: pure maintenance mode + shadow monitoring")


if __name__ == "__main__":
    main()
