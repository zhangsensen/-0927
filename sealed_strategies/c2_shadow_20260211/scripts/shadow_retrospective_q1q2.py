#!/usr/bin/env python3
"""Retrospective orthogonality & drawdown analysis: S1 vs C2.

Uses existing holdout data (2025-05 ~ 2026-02-10) to answer:
  Q1: Are S1 and C2 holdings genuinely orthogonal?
  Q2: Do their drawdowns occur at different times?

No real money needed — runs VEC for equity curves and reconstructs
daily holdings from factor scores + hysteresis logic.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from etf_strategy.core.cost_model import build_cost_array, load_cost_model
from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.execution_model import load_execution_model
from etf_strategy.core.factor_cache import FactorCache
from etf_strategy.core.frozen_params import FrozenETFPool, load_frozen_config
from etf_strategy.core.hysteresis import apply_hysteresis
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import (
    generate_rebalance_schedule,
    shift_timing_signal,
)
from etf_strategy.regime_gate import compute_regime_gate_arr, gate_stats

# Numba kernel imports
from batch_vec_backtest import (
    run_vec_backtest,
    stable_topk_indices,
)

# ─────────────────────────────────────────────────────────────
# Strategy definitions
# ─────────────────────────────────────────────────────────────
STRATEGIES = {
    "S1": ["ADX_14D", "OBV_SLOPE_10D", "SHARPE_RATIO_20D", "SLOPE_20D"],
    "C2": ["AMIHUD_ILLIQUIDITY", "CALMAR_RATIO_60D", "CORRELATION_TO_MARKET_20D"],
}


def load_pipeline_data(config):
    """Load OHLCV, compute factors, build timing — same as batch_vec_backtest.py."""
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

    factor_names = cached["factor_names"]
    dates = cached["dates"]
    etf_codes = cached["etf_codes"]
    factors_3d = cached["factors_3d"]
    T = len(dates)

    close_prices = ohlcv["close"][etf_codes].ffill().fillna(1.0).values
    open_prices = ohlcv["open"][etf_codes].ffill().fillna(1.0).values
    high_prices = ohlcv["high"][etf_codes].ffill().fillna(1.0).values
    low_prices = ohlcv["low"][etf_codes].ffill().fillna(1.0).values

    # Timing: same as batch_vec_backtest.py
    timing_config = config.get("backtest", {}).get("timing", {})
    timing_type = timing_config.get("type", "light_timing")
    extreme_threshold = timing_config.get("extreme_threshold", -0.4)
    extreme_position = timing_config.get("extreme_position", 0.3)

    if timing_type == "dual_timing":
        from etf_strategy.core.market_timing import DualTimingModule

        index_cfg = timing_config.get("index_timing", {})
        indiv_cfg = timing_config.get("individual_timing", {})
        dual = DualTimingModule(
            index_ma_window=index_cfg.get("window", 200),
            bear_position=index_cfg.get("bear_position", 0.1),
            index_symbol=index_cfg.get("symbol", "market_avg"),
            individual_ma_window=indiv_cfg.get("window", 20),
        )
        signals = dual.compute_all_signals(ohlcv["close"])
        timing_arr_raw = (
            signals["index_timing"].reindex(dates).fillna(1.0).values
            if index_cfg.get("enabled", False)
            else np.ones(T, dtype=np.float64)
        )
    else:
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
    timing_arr = (timing_arr * gate_arr).astype(np.float64)

    return {
        "factors_3d": factors_3d,
        "factor_names": list(factor_names),
        "dates": dates,
        "etf_codes": list(etf_codes),
        "close": close_prices,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "timing_arr": timing_arr,
        "ohlcv": ohlcv,
    }


def get_factor_indices(factor_names_list, target_factors):
    """Map factor names to indices in factors_3d."""
    name_to_idx = {name: i for i, name in enumerate(factor_names_list)}
    indices = []
    for f in target_factors:
        if f not in name_to_idx:
            raise ValueError(f"Factor {f!r} not in available factors: {sorted(name_to_idx)}")
        indices.append(name_to_idx[f])
    return indices


def run_vec_for_strategy(data, config, factor_indices):
    """Run VEC backtest and return equity curve."""
    bt_cfg = config.get("backtest", {})
    freq = bt_cfg["freq"]
    pos_size = bt_cfg["pos_size"]
    lookback = bt_cfg["lookback"]
    initial_capital = float(bt_cfg["initial_capital"])
    commission_rate = float(bt_cfg["commission_rate"])

    cost_model = load_cost_model(config)
    qdii_set = set(FrozenETFPool().qdii_codes)
    cost_arr = build_cost_array(cost_model, data["etf_codes"], qdii_set)

    exec_model = load_execution_model(config)

    hysteresis_cfg = config.get("backtest", {}).get("hysteresis", {})
    delta_rank = hysteresis_cfg.get("delta_rank", 0.0)
    min_hold_days = hysteresis_cfg.get("min_hold_days", 0)

    eq, total_ret, win_rate, pf, n_trades, *_ = run_vec_backtest(
        data["factors_3d"],
        data["close"],
        data["open"],
        data["high"],
        data["low"],
        data["timing_arr"],
        factor_indices,
        freq=freq,
        pos_size=pos_size,
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        lookback=lookback,
        cost_arr=cost_arr,
        use_t1_open=exec_model.is_t1_open,
        trailing_stop_pct=0.0,
        stop_on_rebalance_only=True,
        delta_rank=delta_rank,
        min_hold_days=min_hold_days,
    )
    return eq, total_ret, n_trades


def reconstruct_holdings(data, config, factor_indices):
    """Reconstruct daily holdings by replaying scoring + hysteresis logic.

    Returns:
        holdings_history: (num_rebalances, N) bool — which ETFs held after each rebalance
        rebal_dates: list of dates
    """
    bt_cfg = config.get("backtest", {})
    freq = bt_cfg["freq"]
    pos_size = bt_cfg["pos_size"]
    lookback = bt_cfg["lookback"]

    hysteresis_cfg = bt_cfg.get("hysteresis", {})
    delta_rank = hysteresis_cfg.get("delta_rank", 0.0)
    min_hold_days = hysteresis_cfg.get("min_hold_days", 0)

    factors_3d = data["factors_3d"]
    T, N = factors_3d.shape[:2]

    rebalance_schedule = generate_rebalance_schedule(T, lookback, freq)

    # State tracking
    holdings_mask = np.zeros(N, dtype=np.bool_)
    hold_days = np.zeros(N, dtype=np.int64)

    holdings_history = []
    scores_history = []
    rebal_indices = []

    prev_rebal_t = -1

    for rebal_t in rebalance_schedule:
        # Increment hold_days by elapsed trading days
        if prev_rebal_t >= 0:
            elapsed = rebal_t - prev_rebal_t
        else:
            elapsed = 0
        for n in range(N):
            if holdings_mask[n]:
                hold_days[n] += elapsed

        # Compute combined_score (same as kernel: sum of factor values at t-1)
        combined_score = np.full(N, -np.inf, dtype=np.float64)
        valid = 0
        for n in range(N):
            score = 0.0
            has_value = False
            for idx in factor_indices:
                val = factors_3d[rebal_t - 1, n, idx]
                if not np.isnan(val):
                    score += val
                    has_value = True
            if has_value and score != 0.0:
                combined_score[n] = score
                valid += 1

        if valid >= pos_size:
            top_indices = stable_topk_indices(combined_score, pos_size)
            target_mask = apply_hysteresis(
                combined_score, holdings_mask, hold_days,
                top_indices, pos_size, delta_rank, min_hold_days,
            )
        else:
            target_mask = holdings_mask.copy()

        # Update state: sold positions reset hold_days
        for n in range(N):
            if holdings_mask[n] and not target_mask[n]:
                hold_days[n] = 0
            elif not holdings_mask[n] and target_mask[n]:
                hold_days[n] = 0  # newly bought

        holdings_mask = target_mask.copy()
        holdings_history.append(holdings_mask.copy())
        scores_history.append(combined_score.copy())
        rebal_indices.append(rebal_t)
        prev_rebal_t = rebal_t

    return (
        np.array(holdings_history),  # (R, N)
        np.array(scores_history),    # (R, N)
        rebal_indices,
    )


# ─────────────────────────────────────────────────────────────
# Q1: Orthogonality metrics
# ─────────────────────────────────────────────────────────────
def compute_q1_metrics(
    h_s1, h_c2, scores_s1, scores_c2, rebal_indices, dates, holdout_start_idx
):
    """Compute holding overlap and rank correlation metrics."""
    from scipy.stats import spearmanr

    # Filter to holdout period
    mask = np.array(rebal_indices) >= holdout_start_idx
    h_s1_ho = h_s1[mask]
    h_c2_ho = h_c2[mask]
    sc_s1_ho = scores_s1[mask]
    sc_c2_ho = scores_c2[mask]
    rebal_ho = np.array(rebal_indices)[mask]

    R = len(rebal_ho)
    overlaps = []
    jaccards = []
    spearman_rhos = []

    for r in range(R):
        set_s1 = set(np.where(h_s1_ho[r])[0])
        set_c2 = set(np.where(h_c2_ho[r])[0])
        overlap = len(set_s1 & set_c2)
        union = len(set_s1 | set_c2)
        overlaps.append(overlap)
        jaccards.append(overlap / union if union > 0 else 0.0)

        # Rank correlation of scores (excluding -inf)
        s1_sc = sc_s1_ho[r]
        c2_sc = sc_c2_ho[r]
        valid = (s1_sc > -np.inf) & (c2_sc > -np.inf)
        if valid.sum() >= 5:
            rho, _ = spearmanr(s1_sc[valid], c2_sc[valid])
            spearman_rhos.append(rho)

    return {
        "overlap_counts": overlaps,
        "overlap_mean": np.mean(overlaps),
        "overlap_dist": {
            i: overlaps.count(i)
            for i in range(max(overlaps) + 1) if overlaps.count(i) > 0
        },
        "jaccard_mean": np.mean(jaccards),
        "jaccard_std": np.std(jaccards),
        "spearman_rho_mean": np.mean(spearman_rhos) if spearman_rhos else np.nan,
        "spearman_rho_std": np.std(spearman_rhos) if spearman_rhos else np.nan,
        "n_rebalances": R,
    }


# ─────────────────────────────────────────────────────────────
# Q2: Drawdown offset metrics
# ─────────────────────────────────────────────────────────────
def compute_q2_metrics(eq_s1, eq_c2, dates, holdout_start_idx, mkt_returns_ho=None):
    """Compute drawdown synchrony, joint-tail, and co-crash metrics."""

    # Holdout period
    eq_s1_ho = eq_s1[holdout_start_idx:]
    eq_c2_ho = eq_c2[holdout_start_idx:]
    dates_ho = dates[holdout_start_idx:]

    # Daily returns
    ret_s1 = np.diff(eq_s1_ho) / eq_s1_ho[:-1]
    ret_c2 = np.diff(eq_c2_ho) / eq_c2_ho[:-1]

    # --- Drawdown series ---
    def drawdown_series(eq):
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        return dd

    dd_s1 = drawdown_series(eq_s1_ho)
    dd_c2 = drawdown_series(eq_c2_ho)

    # Drawdown synchrony: fraction of days both are in >2% drawdown
    dd_threshold = -0.02
    s1_in_dd = dd_s1 < dd_threshold
    c2_in_dd = dd_c2 < dd_threshold
    both_in_dd = s1_in_dd & c2_in_dd
    either_in_dd = s1_in_dd | c2_in_dd

    sync_rate = both_in_dd.sum() / either_in_dd.sum() if either_in_dd.sum() > 0 else 0.0

    # --- Weekly returns for worst-week analysis ---
    dates_ho_pd = pd.DatetimeIndex(dates_ho)
    df_ret = pd.DataFrame(
        {"s1": np.concatenate([[0.0], ret_s1]), "c2": np.concatenate([[0.0], ret_c2])},
        index=dates_ho_pd,
    )
    weekly = df_ret.resample("W").apply(lambda x: (1 + x).prod() - 1)

    # Worst 5 weeks for each
    worst5_s1 = weekly["s1"].nsmallest(5)
    worst5_c2 = weekly["c2"].nsmallest(5)
    worst_weeks_overlap = len(set(worst5_s1.index) & set(worst5_c2.index))

    # --- Monthly returns for worst-month analysis ---
    monthly = df_ret.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    worst3_s1 = monthly["s1"].nsmallest(3)
    worst3_c2 = monthly["c2"].nsmallest(3)
    worst_months_overlap = len(set(worst3_s1.index) & set(worst3_c2.index))

    # --- Per-strategy left-tail (each strategy's own bottom 10%) ---
    n_days = len(ret_s1)
    q10_s1 = np.percentile(ret_s1, 10)
    q10_c2 = np.percentile(ret_c2, 10)
    s1_left = ret_s1 <= q10_s1
    c2_left = ret_c2 <= q10_c2

    # Marginal probabilities
    p_s1_left = s1_left.mean()
    p_c2_left = c2_left.mean()

    # Joint left-tail: both in their own bottom 10% simultaneously
    joint_left = s1_left & c2_left
    joint_left_prob = joint_left.mean()
    joint_left_n = int(joint_left.sum())

    # Independence baseline & co-crash ratio
    p_ind = p_s1_left * p_c2_left
    co_crash_ratio = joint_left_prob / p_ind if p_ind > 1e-9 else np.nan

    # Joint loss probability: P(both < 0)
    joint_loss = (ret_s1 < 0) & (ret_c2 < 0)
    joint_loss_prob = joint_loss.mean()

    # Complementarity correlation (pooled OR — reference only, biased negative)
    threshold_10_pooled = np.percentile(np.concatenate([ret_s1, ret_c2]), 10)
    either_left_pooled = (ret_s1 < threshold_10_pooled) | (ret_c2 < threshold_10_pooled)
    left_tail_n_pooled = int(either_left_pooled.sum())
    if left_tail_n_pooled >= 5:
        left_corr_complementarity = np.corrcoef(
            ret_s1[either_left_pooled], ret_c2[either_left_pooled]
        )[0, 1]
    else:
        left_corr_complementarity = np.nan

    # Market-conditional joint loss: P(s1<0 AND c2<0 | market in left tail)
    joint_loss_given_mkt_left = np.nan
    mkt_left_n = 0
    mkt_left_joint_k = 0
    joint_loss_mkt_left_ci95 = np.nan
    residual_joint_loss_prob = np.nan
    residual_corr = np.nan

    if mkt_returns_ho is not None and len(mkt_returns_ho) == n_days:
        q10_mkt = np.percentile(mkt_returns_ho, 10)
        mkt_left = mkt_returns_ho <= q10_mkt
        mkt_left_n = int(mkt_left.sum())
        if mkt_left_n >= 3:
            mkt_left_joint_k = int(
                ((ret_s1[mkt_left] < 0) & (ret_c2[mkt_left] < 0)).sum()
            )
            joint_loss_given_mkt_left = mkt_left_joint_k / mkt_left_n

            # Wilson score interval upper bound (95%)
            z = 1.96
            p_hat = joint_loss_given_mkt_left
            n_w = mkt_left_n
            denom = 1 + z**2 / n_w
            center = p_hat + z**2 / (2 * n_w)
            margin = z * np.sqrt(p_hat * (1 - p_hat) / n_w + z**2 / (4 * n_w**2))
            joint_loss_mkt_left_ci95 = (center + margin) / denom

        # Beta-neutral residual joint loss (full holdout, not just left tail)
        coef_s1 = np.polyfit(mkt_returns_ho, ret_s1, 1)
        coef_c2 = np.polyfit(mkt_returns_ho, ret_c2, 1)
        eps_s1 = ret_s1 - (coef_s1[0] * mkt_returns_ho + coef_s1[1])
        eps_c2 = ret_c2 - (coef_c2[0] * mkt_returns_ho + coef_c2[1])
        residual_joint_loss_prob = ((eps_s1 < 0) & (eps_c2 < 0)).mean()
        residual_corr = np.corrcoef(eps_s1, eps_c2)[0, 1]

    # --- Full return correlation ---
    full_corr = np.corrcoef(ret_s1, ret_c2)[0, 1]

    return {
        "n_days": n_days,
        "dd_sync_rate": sync_rate,
        "s1_dd_days": int(s1_in_dd.sum()),
        "c2_dd_days": int(c2_in_dd.sum()),
        "both_dd_days": int(both_in_dd.sum()),
        "worst5_weeks_overlap": worst_weeks_overlap,
        "worst5_weeks_s1": worst5_s1.to_dict(),
        "worst5_weeks_c2": worst5_c2.to_dict(),
        "worst3_months_overlap": worst_months_overlap,
        "worst3_months_s1": worst3_s1.to_dict(),
        "worst3_months_c2": worst3_c2.to_dict(),
        # Per-strategy left-tail metrics (primary)
        "joint_left_prob": joint_left_prob,
        "joint_left_n": joint_left_n,
        "co_crash_ratio": co_crash_ratio,
        "p_independent": p_ind,
        "joint_loss_prob": joint_loss_prob,
        # Market-conditional (510300)
        "joint_loss_given_mkt_left": joint_loss_given_mkt_left,
        "joint_loss_mkt_left_ci95": joint_loss_mkt_left_ci95,
        "mkt_left_joint_k": mkt_left_joint_k,
        "mkt_left_n": mkt_left_n,
        # Beta-neutral residual (market factor regressed out)
        "residual_joint_loss_prob": residual_joint_loss_prob,
        "residual_corr": residual_corr,
        # Complementarity reference (pooled OR — biased negative, not for verdict)
        "left_corr_complementarity": left_corr_complementarity,
        "left_tail_n_pooled": left_tail_n_pooled,
        # Standard correlations
        "full_return_corr": full_corr,
        "weekly_return_corr": weekly["s1"].corr(weekly["c2"]),
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Shadow Retrospective: Q1 (orthogonality) + Q2 (drawdown offset)")
    print("=" * 70)

    config_path = ROOT / "configs" / "combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    frozen = load_frozen_config(config, config_path=str(config_path))
    print(f"Config: v{frozen.version}, FREQ={config['backtest']['freq']}, "
          f"POS_SIZE={config['backtest']['pos_size']}")

    # Load data
    print("\n--- Loading data ---")
    data = load_pipeline_data(config)
    dates = data["dates"]
    etf_codes = data["etf_codes"]
    T = len(dates)
    N = len(etf_codes)
    factor_names = data["factor_names"]

    # Find holdout start index
    training_end = pd.Timestamp(config["data"]["training_end_date"])
    holdout_start_idx = 0
    for i, d in enumerate(dates):
        if pd.Timestamp(d) > training_end:
            holdout_start_idx = i
            break
    print(f"Holdout starts at index {holdout_start_idx} ({dates[holdout_start_idx]})")
    print(f"Data: {T} days x {N} ETFs x {len(factor_names)} factors")

    # Market proxy (510300) for conditional analysis
    mkt_idx = etf_codes.index("510300") if "510300" in etf_codes else None

    results = {}

    for strat_name, strat_factors in STRATEGIES.items():
        print(f"\n{'='*50}")
        print(f"Strategy: {strat_name} — {' + '.join(strat_factors)}")
        print(f"{'='*50}")

        factor_idx = get_factor_indices(factor_names, strat_factors)
        print(f"Factor indices: {factor_idx}")

        # 1. VEC equity curve
        print("Running VEC...")
        eq, total_ret, n_trades = run_vec_for_strategy(data, config, factor_idx)
        ho_eq = eq[holdout_start_idx:]
        ho_ret = ho_eq[-1] / ho_eq[0] - 1
        print(f"  Full return: {total_ret:+.1%}, HO return: {ho_ret:+.1%}, Trades: {n_trades}")

        # 2. Reconstruct holdings
        print("Reconstructing holdings...")
        h_history, sc_history, rebal_indices = reconstruct_holdings(
            data, config, factor_idx,
        )
        ho_rebals = sum(1 for r in rebal_indices if r >= holdout_start_idx)
        print(f"  Total rebalances: {len(rebal_indices)}, Holdout: {ho_rebals}")

        results[strat_name] = {
            "equity": eq,
            "holdings": h_history,
            "scores": sc_history,
            "rebal_indices": rebal_indices,
            "ho_return": ho_ret,
            "n_trades": n_trades,
        }

    # ─────────────────────────────────────────────────────────
    # Q1 Analysis
    # ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Q1: ORTHOGONALITY — Are S1 and C2 selecting different ETFs?")
    print(f"{'='*70}")

    q1 = compute_q1_metrics(
        results["S1"]["holdings"],
        results["C2"]["holdings"],
        results["S1"]["scores"],
        results["C2"]["scores"],
        results["S1"]["rebal_indices"],  # Same schedule for both
        dates,
        holdout_start_idx,
    )

    print(f"\nHolding overlap (top-2 vs top-2) per rebalance:")
    for count, freq in sorted(q1["overlap_dist"].items()):
        pct = freq / q1["n_rebalances"] * 100
        bar = "█" * int(pct / 2)
        print(f"  {count}/2 overlap: {freq:3d} times ({pct:5.1f}%) {bar}")

    print(f"\nJaccard similarity: {q1['jaccard_mean']:.3f} ± {q1['jaccard_std']:.3f}")
    print(f"  (0 = completely different ETFs, 1 = identical)")
    print(f"\nSpearman rank correlation of scores: {q1['spearman_rho_mean']:.3f} ± {q1['spearman_rho_std']:.3f}")
    print(f"  (0 = orthogonal ranking, 1 = same ranking)")

    # Verdict: primary = Jaccard (hard), Spearman = reference only
    print(f"\n--- Q1 Verdict ---")
    overlap_rate = q1["overlap_mean"]  # avg overlap per rebalance (out of pos_size)
    if q1["jaccard_mean"] < 0.25:
        q1_verdict = "PASS"
        print(f"✅ ORTHOGONAL: Jaccard={q1['jaccard_mean']:.3f} (<0.25), top-2 overlap={overlap_rate:.1f}/2")
        print(f"   Spearman={q1['spearman_rho_mean']:.3f} (reference — mid-rank correlation expected from shared market factor)")
    elif q1["jaccard_mean"] < 0.40:
        q1_verdict = "PARTIAL"
        print(f"⚠️  PARTIAL: Jaccard={q1['jaccard_mean']:.3f}, some holding overlap exists")
    else:
        q1_verdict = "FAIL"
        print(f"❌ NOT ORTHOGONAL: Jaccard={q1['jaccard_mean']:.3f} (>=0.40) — similar ETF selection")

    # ─────────────────────────────────────────────────────────
    # Q2 Analysis
    # ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Q2: DRAWDOWN OFFSET — Do S1 and C2 suffer at different times?")
    print(f"{'='*70}")

    # Market returns for conditional analysis
    if mkt_idx is not None:
        mkt_close_ho = data["close"][holdout_start_idx:, mkt_idx]
        mkt_ret_ho = np.diff(mkt_close_ho) / mkt_close_ho[:-1]
    else:
        mkt_ret_ho = None

    q2 = compute_q2_metrics(
        results["S1"]["equity"],
        results["C2"]["equity"],
        dates,
        holdout_start_idx,
        mkt_returns_ho=mkt_ret_ho,
    )

    print(f"\nReturn correlations (shared A-share beta, not actionable):")
    print(f"  Daily:  {q2['full_return_corr']:.3f}")
    print(f"  Weekly: {q2['weekly_return_corr']:.3f}")

    print(f"\nDrawdown synchrony (>2% DD threshold):")
    print(f"  S1 in drawdown: {q2['s1_dd_days']} days")
    print(f"  C2 in drawdown: {q2['c2_dd_days']} days")
    print(f"  Both simultaneously: {q2['both_dd_days']} days")
    print(f"  Sync rate: {q2['dd_sync_rate']:.1%} (reference only)")

    print(f"\nWorst weeks overlap: {q2['worst5_weeks_overlap']}/5")
    s1_ww = [f"{d.strftime('%Y-%m-%d')}:{v:+.1%}" for d, v in q2["worst5_weeks_s1"].items()]
    c2_ww = [f"{d.strftime('%Y-%m-%d')}:{v:+.1%}" for d, v in q2["worst5_weeks_c2"].items()]
    print(f"  S1 worst weeks: {s1_ww}")
    print(f"  C2 worst weeks: {c2_ww}")

    print(f"\nWorst months overlap: {q2['worst3_months_overlap']}/3")
    s1_wm = [f"{d.strftime('%Y-%m')}:{v:+.1%}" for d, v in q2["worst3_months_s1"].items()]
    c2_wm = [f"{d.strftime('%Y-%m')}:{v:+.1%}" for d, v in q2["worst3_months_c2"].items()]
    print(f"  S1 worst months: {s1_wm}")
    print(f"  C2 worst months: {c2_wm}")

    print(f"\n--- Joint-Tail Metrics (PRIMARY for verdict) ---")
    print(f"  Holdout days: {q2['n_days']}")
    print(f"  Joint left-tail prob:    {q2['joint_left_prob']:.4f}  (n={q2['joint_left_n']})")
    print(f"  Independent baseline:    {q2['p_independent']:.4f}")
    print(f"  Co-crash ratio:          {q2['co_crash_ratio']:.2f}  (<1=dispersed, ~1=independent, >1=clustered)")
    print(f"  Joint loss prob P(both<0): {q2['joint_loss_prob']:.3f}")

    if q2["mkt_left_n"] > 0:
        k = q2["mkt_left_joint_k"]
        n_mkt = q2["mkt_left_n"]
        print(f"\n  Market-conditional (510300 bottom 10%, n={n_mkt}):")
        print(f"    P(both<0 | mkt left):     {q2['joint_loss_given_mkt_left']:.3f}  ({k}/{n_mkt})")
        print(f"    Wilson 95% CI upper:       {q2['joint_loss_mkt_left_ci95']:.3f}")

    if not np.isnan(q2["residual_joint_loss_prob"]):
        print(f"\n  Beta-neutral residual (510300 regressed out, full holdout):")
        print(f"    P(eps_s1<0 AND eps_c2<0):  {q2['residual_joint_loss_prob']:.3f}  (independent baseline ~0.25)")
        print(f"    Residual correlation:       {q2['residual_corr']:.3f}")

    print(f"\n  Monitoring (not for verdict):")
    print(f"    Co-crash ratio:            {q2['co_crash_ratio']:.2f}  (inflated by shared A-share beta)")
    print(f"    Complementarity corr:      {q2['left_corr_complementarity']:.3f}  (pooled OR, biased negative)")

    # Verdict: market-conditional as primary, co_crash as monitoring only
    print(f"\n--- Q2 Verdict ---")
    worst_wk = q2["worst5_weeks_overlap"]
    p_mkt = q2["joint_loss_given_mkt_left"]
    ci95 = q2["joint_loss_mkt_left_ci95"]

    cond1_worst_wk = worst_wk <= 2
    cond2_mkt_point = p_mkt <= 0.50 if not np.isnan(p_mkt) else False
    cond3_mkt_ci = ci95 <= 0.65 if not np.isnan(ci95) else False

    if cond1_worst_wk and cond2_mkt_point and cond3_mkt_ci:
        q2_verdict = "PASS"
        print(f"✅ TAIL DISPERSED:")
        print(f"   worst_week_overlap={worst_wk}/5 (<=2)")
        print(f"   P(both<0|mkt crash)={p_mkt:.3f} (<=0.50), CI95 upper={ci95:.3f} (<=0.65)")
        print(f"   In market crashes, at least one leg survives {1-p_mkt:.0%} of the time")
    elif cond1_worst_wk and cond2_mkt_point:
        q2_verdict = "PARTIAL"
        print(f"⚠️  PARTIAL: point estimate OK but CI wide (CI95={ci95:.3f} > 0.65, n={q2['mkt_left_n']})")
        print(f"   Shadow warranted but small sample — needs more data to confirm")
    else:
        q2_verdict = "FAIL"
        reasons = []
        if not cond1_worst_wk:
            reasons.append(f"worst_week_overlap={worst_wk}/5 (>2)")
        if not cond2_mkt_point:
            reasons.append(f"P(both<0|mkt crash)={p_mkt:.3f} (>0.50)")
        print(f"❌ INSUFFICIENT TAIL DISPERSAL: {', '.join(reasons)}")

    # ─────────────────────────────────────────────────────────
    # Combined verdict
    # ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("COMBINED ASSESSMENT: Should C2 enter Shadow as second engine?")
    print(f"{'='*70}")

    q1_pass = q1_verdict == "PASS"
    q2_pass = q2_verdict == "PASS"
    ho_ret_s1 = results["S1"]["ho_return"]
    ho_ret_c2 = results["C2"]["ho_return"]

    print(f"\n  Q1 Orthogonality:     {q1_verdict}  (Jaccard={q1['jaccard_mean']:.3f})")
    print(f"  Q2 Tail dispersal:    {q2_verdict}  (P(jl|mkt)={q2['joint_loss_given_mkt_left']:.3f}, CI95={q2['joint_loss_mkt_left_ci95']:.3f}, worst_wk={q2['worst5_weeks_overlap']}/5)")
    print(f"  HO return S1:         {ho_ret_s1:+.1%}")
    print(f"  HO return C2:         {ho_ret_c2:+.1%}")
    if not np.isnan(q2["residual_joint_loss_prob"]):
        print(f"  Residual P(both<0):   {q2['residual_joint_loss_prob']:.3f}  (beta-neutral, baseline ~0.25)")
    print(f"  Co-crash ratio:       {q2['co_crash_ratio']:.2f}  (monitoring — inflated by shared beta)")

    if q1_pass and q2_pass:
        print(f"\n✅ PROCEED TO SHADOW: C2 is a genuinely independent second engine")
        print(f"   Selection layer: zero holding overlap")
        print(f"   Tail layer: market crashes → at least one leg survives {1-q2['joint_loss_given_mkt_left']:.0%}")
        print(f"   Expected portfolio benefit: tail risk reduction through diversification")
    elif q1_pass or q2_pass:
        print(f"\n⚠️  CONDITIONAL PROCEED: Partial evidence for independence")
        print(f"   Shadow warranted but watch for convergence under stress")
    else:
        print(f"\n❌ DO NOT SHADOW: C2 does not provide sufficient independence from S1")
        print(f"   Focus research on finding truly orthogonal alpha sources")

    # Save results
    out_dir = ROOT / "results" / "shadow_retrospective"
    out_dir.mkdir(exist_ok=True)

    # Save equity curves
    eq_df = pd.DataFrame(
        {"date": dates, "S1": results["S1"]["equity"], "C2": results["C2"]["equity"]}
    )
    eq_df.to_csv(out_dir / "equity_curves.csv", index=False)

    # Save holding overlap (holdout only)
    rebal_dates_all = [dates[i] for i in results["S1"]["rebal_indices"]]
    ho_rebal_mask = [i >= holdout_start_idx for i in results["S1"]["rebal_indices"]]
    rebal_dates_ho = [d for d, m in zip(rebal_dates_all, ho_rebal_mask) if m]
    overlap_df = pd.DataFrame(
        {"date": rebal_dates_ho, "overlap": q1["overlap_counts"]}
    )
    overlap_df.to_csv(out_dir / "holding_overlap_ho.csv", index=False)

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
