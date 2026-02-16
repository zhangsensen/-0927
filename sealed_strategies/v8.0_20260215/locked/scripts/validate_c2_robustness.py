#!/usr/bin/env python3
"""C2 Robustness Validation: Gate 1 (Rolling Window) + Gate 2 (PnL Concentration).

Tests whether C2_3factor's advantage over S1 is robust or event-driven.

Gate 1: Rolling 126-day window metrics — does C2 beat S1 in most windows?
Gate 2: Per-rebalance PnL distribution — is C2's alpha driven by few big wins?

Both gates use F5+Exp4 production execution (FREQ=5, dr=0.10, mh=9).
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import yaml

from etf_strategy.core.cost_model import build_cost_array, load_cost_model
from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.execution_model import load_execution_model
from etf_strategy.core.factor_cache import FactorCache
from etf_strategy.core.frozen_params import FrozenETFPool
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import (
    generate_rebalance_schedule,
    shift_timing_signal,
)
from etf_strategy.regime_gate import compute_regime_gate_arr

from aligned_metrics import compute_aligned_metrics
from batch_vec_backtest import run_vec_backtest

# ── Strategies to compare ────────────────────────────────────────────────
STRATEGIES = {
    "S1": "ADX_14D + OBV_SLOPE_10D + SHARPE_RATIO_20D + SLOPE_20D",
    "C2": "AMIHUD_ILLIQUIDITY + CALMAR_RATIO_60D + CORRELATION_TO_MARKET_20D",
}

# Production execution config
EXEC_CFG = {"freq": 5, "delta_rank": 0.10, "min_hold_days": 9}
LOOKBACK = 252


def _compute_mdd(equity: np.ndarray) -> float:
    peak = equity[0]
    mdd = 0.0
    for v in equity:
        if not np.isfinite(v):
            continue
        if v > peak:
            peak = v
        if peak > 0:
            dd = (peak - v) / peak
            if dd > mdd:
                mdd = dd
    return mdd


def _worst_month(equity: np.ndarray, window: int = 21) -> float:
    if len(equity) < window + 1:
        return 0.0
    worst = 0.0
    for i in range(window, len(equity)):
        ret = equity[i] / equity[i - window] - 1.0 if equity[i - window] > 0 else 0.0
        if ret < worst:
            worst = ret
    return worst


def rolling_window_metrics(
    equity: np.ndarray, window: int = 126, step: int = 21
) -> list[dict]:
    """Compute metrics for overlapping rolling windows."""
    results = []
    for start in range(0, len(equity) - window, step):
        end = start + window
        win_eq = equity[start:end]
        if win_eq[0] <= 0:
            continue
        ret = win_eq[-1] / win_eq[0] - 1.0
        daily_rets = np.diff(win_eq) / win_eq[:-1]
        daily_rets = daily_rets[np.isfinite(daily_rets)]
        if len(daily_rets) > 1:
            sharpe = (
                np.mean(daily_rets) / np.std(daily_rets, ddof=1) * np.sqrt(252)
                if np.std(daily_rets, ddof=1) > 1e-10
                else 0.0
            )
        else:
            sharpe = 0.0
        mdd = _compute_mdd(win_eq)
        worst_mo = _worst_month(win_eq)
        results.append({
            "start_idx": start,
            "end_idx": end,
            "return": ret,
            "sharpe": sharpe,
            "mdd": mdd,
            "worst_month": worst_mo,
        })
    return results


def rebalance_period_pnl(
    equity: np.ndarray, total_T: int, holdout_start_idx: int
) -> np.ndarray:
    """Compute per-rebalance-period returns from holdout equity curve.

    Returns array of period returns (one per rebalance interval).
    """
    freq = EXEC_CFG["freq"]
    # Generate rebalance schedule for full period
    sched = generate_rebalance_schedule(total_T, lookback_window=LOOKBACK, freq=freq)
    # Filter to holdout-only rebalance days
    ho_sched = sched[sched >= holdout_start_idx]
    # Convert to relative indices within holdout equity
    ho_relative = ho_sched - holdout_start_idx

    period_rets = []
    for i in range(len(ho_relative) - 1):
        s = ho_relative[i]
        e = ho_relative[i + 1]
        if s < 0 or e >= len(equity) or equity[s] <= 0:
            continue
        period_ret = equity[e] / equity[s] - 1.0
        period_rets.append(period_ret)

    # Last period: from last rebalance to end
    if len(ho_relative) > 0:
        s = ho_relative[-1]
        if 0 <= s < len(equity) - 1 and equity[s] > 0:
            period_ret = equity[-1] / equity[s] - 1.0
            period_rets.append(period_ret)

    return np.array(period_rets)


def pnl_concentration_analysis(period_rets: np.ndarray) -> dict:
    """Gate 2: Analyze PnL contribution concentration."""
    if len(period_rets) == 0:
        return {"error": "no periods"}

    total_pnl = np.sum(period_rets)
    n = len(period_rets)

    # Sort by absolute contribution (descending)
    sorted_by_abs = np.sort(np.abs(period_rets))[::-1]
    # Top 20% periods
    top_k = max(1, int(np.ceil(n * 0.2)))
    # Sort by actual value to find top contributors to POSITIVE pnl
    sorted_desc = np.sort(period_rets)[::-1]
    top_20_sum = np.sum(sorted_desc[:top_k])

    # Positive and negative
    wins = period_rets[period_rets > 0]
    losses = period_rets[period_rets < 0]
    flat = period_rets[period_rets == 0]

    # Max single loss
    max_single_loss = np.min(period_rets) if len(period_rets) > 0 else 0.0

    # Max consecutive losing periods
    max_consec_loss = 0
    current_streak = 0
    for r in period_rets:
        if r < 0:
            current_streak += 1
            if current_streak > max_consec_loss:
                max_consec_loss = current_streak
        else:
            current_streak = 0

    # Max consecutive winning periods
    max_consec_win = 0
    current_streak = 0
    for r in period_rets:
        if r > 0:
            current_streak += 1
            if current_streak > max_consec_win:
                max_consec_win = current_streak
        else:
            current_streak = 0

    # Gini coefficient of absolute period returns (higher = more concentrated)
    if n > 1:
        abs_rets = np.abs(period_rets)
        abs_sorted = np.sort(abs_rets)
        index = np.arange(1, n + 1)
        gini = (2.0 * np.sum(index * abs_sorted) / (n * np.sum(abs_sorted))) - (n + 1) / n
    else:
        gini = 0.0

    return {
        "n_periods": n,
        "total_pnl": total_pnl,
        "n_wins": len(wins),
        "n_losses": len(losses),
        "n_flat": len(flat),
        "win_rate": len(wins) / n if n > 0 else 0,
        "avg_win": np.mean(wins) if len(wins) > 0 else 0,
        "avg_loss": np.mean(losses) if len(losses) > 0 else 0,
        "top_20pct_contribution": top_20_sum / total_pnl if abs(total_pnl) > 1e-8 else float("nan"),
        "top_20pct_n": top_k,
        "max_single_loss": max_single_loss,
        "max_consec_loss": max_consec_loss,
        "max_consec_win": max_consec_win,
        "best_period": np.max(period_rets),
        "worst_period": np.min(period_rets),
        "median_period": np.median(period_rets),
        "gini_coefficient": gini,
    }


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 90)
    print("C2 ROBUSTNESS VALIDATION (Gate 1 + Gate 2)")
    print(f"  Strategies: {list(STRATEGIES.keys())}")
    print(f"  Execution: F5_ON (freq={EXEC_CFG['freq']}, dr={EXEC_CFG['delta_rank']}, mh={EXEC_CFG['min_hold_days']})")
    print("=" * 90)

    # ── 1. Load config & data (same as validate_bucket_candidates.py) ──
    config_path = ROOT / "configs" / "combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    backtest_config = config.get("backtest", {})
    POS_SIZE = 2
    INITIAL_CAPITAL = float(backtest_config.get("initial_capital", 1_000_000))
    COMMISSION_RATE = float(backtest_config.get("commission_rate", 0.0002))

    exec_model = load_execution_model(config)
    cost_model = load_cost_model(config)
    qdii_set = set(FrozenETFPool().qdii_codes)

    risk_config = backtest_config.get("risk_control", {})
    trailing_stop_pct = risk_config.get("trailing_stop_pct", 0.0)
    stop_on_rebalance_only = risk_config.get("stop_check_on_rebalance_only", True)
    leverage_cap = risk_config.get("leverage_cap", 1.0)
    profit_ladders = risk_config.get("profit_ladders", [])

    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )

    # ── 2. Compute factors ──
    factor_cache = FactorCache(
        cache_dir=Path(config["data"].get("cache_dir") or ".cache")
    )
    cached = factor_cache.get_or_compute(
        ohlcv=ohlcv, config=config, data_dir=loader.data_dir
    )
    factors_3d = cached["factors_3d"]
    factor_names = cached["factor_names"]
    dates = cached["dates"]
    etf_codes = cached["etf_codes"]
    T = len(dates)
    N = len(etf_codes)

    close_prices = ohlcv["close"][etf_codes].ffill().fillna(1.0).values
    open_prices = ohlcv["open"][etf_codes].ffill().fillna(1.0).values
    high_prices = ohlcv["high"][etf_codes].ffill().fillna(1.0).values
    low_prices = ohlcv["low"][etf_codes].ffill().fillna(1.0).values
    COST_ARR = build_cost_array(cost_model, list(etf_codes), qdii_set)

    # ── 3. Timing + regime gate ──
    timing_cfg = backtest_config.get("timing", {})
    timing_module = LightTimingModule(
        extreme_threshold=float(timing_cfg.get("extreme_threshold", -0.1)),
        extreme_position=float(timing_cfg.get("extreme_position", 0.1)),
    )
    timing_raw = (
        timing_module.compute_position_ratios(ohlcv["close"])
        .reindex(dates)
        .fillna(1.0)
        .values
    )
    timing_arr = shift_timing_signal(timing_raw)
    gate_arr = compute_regime_gate_arr(
        ohlcv["close"], dates, backtest_config=backtest_config
    )
    timing_arr = timing_arr.astype(np.float64) * gate_arr.astype(np.float64)

    # ── 4. Train/holdout split ──
    training_end_date = pd.Timestamp(
        config["data"].get("training_end_date", "2025-04-30")
    )
    train_end_idx = 0
    for i, d in enumerate(dates):
        if pd.Timestamp(d) <= training_end_date:
            train_end_idx = i

    start_idx = LOOKBACK
    ho_start_idx = train_end_idx  # holdout starts here in full equity curve

    print(f"  Data: {T} days x {N} ETFs x {len(factor_names)} factors")
    print(f"  Train: {dates[0]} ~ {dates[train_end_idx]}")
    print(f"  Holdout: {dates[train_end_idx+1]} ~ {dates[T-1]}")
    print(f"  Holdout days: {T - train_end_idx - 1}")

    # ── 5. Resolve factor indices ──
    factor_index_map = {name: idx for idx, name in enumerate(factor_names)}
    strat_indices = {}
    for name, combo_str in STRATEGIES.items():
        factors = [f.strip() for f in combo_str.split("+")]
        indices = [factor_index_map[f] for f in factors]
        strat_indices[name] = indices

    # ── 6. Run VEC for both strategies ──
    print("\nRunning VEC backtests (F5_ON)...")
    equity_curves = {}
    for name, f_indices in strat_indices.items():
        eq, ret, wr, pf, trades, _, risk = run_vec_backtest(
            factors_3d, close_prices, open_prices, high_prices, low_prices,
            timing_arr, f_indices,
            freq=EXEC_CFG["freq"], pos_size=POS_SIZE,
            initial_capital=INITIAL_CAPITAL, commission_rate=COMMISSION_RATE,
            lookback=LOOKBACK, cost_arr=COST_ARR,
            trailing_stop_pct=trailing_stop_pct,
            stop_on_rebalance_only=stop_on_rebalance_only,
            leverage_cap=leverage_cap, profit_ladders=profit_ladders,
            use_t1_open=exec_model.is_t1_open,
            delta_rank=EXEC_CFG["delta_rank"],
            min_hold_days=EXEC_CFG["min_hold_days"],
        )
        equity_curves[name] = {
            "full": eq,
            "holdout": eq[ho_start_idx:],
            "trades": trades,
            "win_rate": wr,
            "risk": risk,
        }
        ho_eq = eq[ho_start_idx:]
        ho_m = compute_aligned_metrics(ho_eq, start_idx=0)
        ho_mdd = _compute_mdd(ho_eq)
        print(
            f"  {name:4s}: HO_ret={ho_m['aligned_return']*100:+6.1f}%  "
            f"MDD={ho_mdd*100:5.1f}%  "
            f"Sharpe={ho_m['aligned_sharpe']:.2f}  "
            f"trades={trades}"
        )

    # ══════════════════════════════════════════════════════════════════════
    # GATE 1: Rolling Window Robustness
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("GATE 1: ROLLING WINDOW ROBUSTNESS (126-day windows, 21-day step)")
    print("=" * 90)

    rolling = {}
    for name in STRATEGIES:
        ho_eq = equity_curves[name]["holdout"]
        rolling[name] = rolling_window_metrics(ho_eq, window=126, step=21)

    n_windows = min(len(rolling["S1"]), len(rolling["C2"]))
    print(f"\n  Windows: {n_windows}")

    # Window-by-window comparison
    c2_better_return = 0
    c2_better_sharpe = 0
    c2_better_mdd = 0
    c2_better_worst_mo = 0
    c2_worse_return = 0
    s1_worst_window = None
    c2_worst_window = None

    print(f"\n  {'Window':>6s}  {'S1 Ret':>8s}  {'C2 Ret':>8s}  {'S1 MDD':>8s}  {'C2 MDD':>8s}  {'S1 WM':>8s}  {'C2 WM':>8s}  {'Winner':>8s}")
    print("  " + "-" * 80)

    for i in range(n_windows):
        s1w = rolling["S1"][i]
        c2w = rolling["C2"][i]
        if c2w["return"] > s1w["return"]:
            c2_better_return += 1
        else:
            c2_worse_return += 1
        if c2w["sharpe"] > s1w["sharpe"]:
            c2_better_sharpe += 1
        if c2w["mdd"] < s1w["mdd"]:
            c2_better_mdd += 1
        if c2w["worst_month"] > s1w["worst_month"]:  # less negative = better
            c2_better_worst_mo += 1

        # Track worst windows
        if s1_worst_window is None or s1w["return"] < s1_worst_window["return"]:
            s1_worst_window = {**s1w, "window_idx": i}
        if c2_worst_window is None or c2w["return"] < c2_worst_window["return"]:
            c2_worst_window = {**c2w, "window_idx": i}

        winner = "C2" if c2w["return"] > s1w["return"] else "S1"
        print(
            f"  {i+1:>6d}  "
            f"{s1w['return']*100:>+7.1f}%  "
            f"{c2w['return']*100:>+7.1f}%  "
            f"{s1w['mdd']*100:>7.1f}%  "
            f"{c2w['mdd']*100:>7.1f}%  "
            f"{s1w['worst_month']*100:>+7.1f}%  "
            f"{c2w['worst_month']*100:>+7.1f}%  "
            f"{'  '+winner:>8s}"
        )

    print(f"\n  GATE 1 SUMMARY:")
    print(f"    C2 wins return:     {c2_better_return}/{n_windows} ({c2_better_return/n_windows*100:.0f}%)")
    print(f"    C2 wins Sharpe:     {c2_better_sharpe}/{n_windows} ({c2_better_sharpe/n_windows*100:.0f}%)")
    print(f"    C2 wins MDD:        {c2_better_mdd}/{n_windows} ({c2_better_mdd/n_windows*100:.0f}%)")
    print(f"    C2 wins worst month:{c2_better_worst_mo}/{n_windows} ({c2_better_worst_mo/n_windows*100:.0f}%)")
    print(f"\n    S1 worst window: #{s1_worst_window['window_idx']+1}, ret={s1_worst_window['return']*100:+.1f}%, MDD={s1_worst_window['mdd']*100:.1f}%")
    print(f"    C2 worst window: #{c2_worst_window['window_idx']+1}, ret={c2_worst_window['return']*100:+.1f}%, MDD={c2_worst_window['mdd']*100:.1f}%")

    # Gate 1 verdict
    g1_pass = c2_better_return / n_windows >= 0.5 and c2_worst_window["return"] >= s1_worst_window["return"] * 1.5
    # Pass if: C2 wins >50% of windows AND C2's worst window isn't 1.5x worse than S1's worst
    print(f"\n    GATE 1 VERDICT: {'PASS' if g1_pass else 'REVIEW'}")
    print(f"      (C2 wins majority={c2_better_return/n_windows*100:.0f}% >= 50%, "
          f"C2 worst={c2_worst_window['return']*100:+.1f}% vs S1 worst={s1_worst_window['return']*100:+.1f}%)")

    # ══════════════════════════════════════════════════════════════════════
    # GATE 2: PnL CONTRIBUTION CONCENTRATION
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("GATE 2: PnL CONTRIBUTION CONCENTRATION (per-rebalance period)")
    print("=" * 90)

    pnl_analysis = {}
    for name in STRATEGIES:
        ho_eq = equity_curves[name]["holdout"]
        period_rets = rebalance_period_pnl(ho_eq, T, ho_start_idx)
        analysis = pnl_concentration_analysis(period_rets)
        pnl_analysis[name] = {"period_rets": period_rets, "analysis": analysis}

    for name in STRATEGIES:
        a = pnl_analysis[name]["analysis"]
        print(f"\n  {name}:")
        print(f"    Rebalance periods:     {a['n_periods']}")
        print(f"    Win/Loss/Flat:         {a['n_wins']}/{a['n_losses']}/{a['n_flat']}")
        print(f"    Period win rate:       {a['win_rate']:.1%}")
        print(f"    Avg win:               {a['avg_win']*100:+.2f}%")
        print(f"    Avg loss:              {a['avg_loss']*100:+.2f}%")
        print(f"    Best period:           {a['best_period']*100:+.2f}%")
        print(f"    Worst period:          {a['worst_period']*100:+.2f}%")
        print(f"    Median period:         {a['median_period']*100:+.2f}%")
        print(f"    Top 20% contribution:  {a['top_20pct_contribution']:.1%} (n={a['top_20pct_n']})")
        print(f"    Max single loss:       {a['max_single_loss']*100:+.2f}%")
        print(f"    Max consec losses:     {a['max_consec_loss']}")
        print(f"    Max consec wins:       {a['max_consec_win']}")
        print(f"    Gini coefficient:      {a['gini_coefficient']:.3f}")

    # Gate 2 comparative analysis
    s1a = pnl_analysis["S1"]["analysis"]
    c2a = pnl_analysis["C2"]["analysis"]

    print(f"\n  GATE 2 COMPARISON:")
    print(f"    {'Metric':<25s}  {'S1':>10s}  {'C2':>10s}  {'Better':>8s}")
    print("    " + "-" * 60)
    metrics_to_compare = [
        ("Period win rate", f"{s1a['win_rate']:.1%}", f"{c2a['win_rate']:.1%}",
         "C2" if c2a["win_rate"] > s1a["win_rate"] else "S1"),
        ("Top 20% concentration", f"{s1a['top_20pct_contribution']:.1%}", f"{c2a['top_20pct_contribution']:.1%}",
         "C2" if c2a["top_20pct_contribution"] < s1a["top_20pct_contribution"] else "S1"),
        ("Max single loss", f"{s1a['max_single_loss']*100:+.2f}%", f"{c2a['max_single_loss']*100:+.2f}%",
         "C2" if c2a["max_single_loss"] > s1a["max_single_loss"] else "S1"),
        ("Max consec losses", f"{s1a['max_consec_loss']}", f"{c2a['max_consec_loss']}",
         "C2" if c2a["max_consec_loss"] <= s1a["max_consec_loss"] else "S1"),
        ("Gini coeff (lower=diverse)", f"{s1a['gini_coefficient']:.3f}", f"{c2a['gini_coefficient']:.3f}",
         "C2" if c2a["gini_coefficient"] < s1a["gini_coefficient"] else "S1"),
        ("Avg win / |Avg loss|", f"{abs(s1a['avg_win']/s1a['avg_loss']) if s1a['avg_loss'] != 0 else 0:.2f}",
         f"{abs(c2a['avg_win']/c2a['avg_loss']) if c2a['avg_loss'] != 0 else 0:.2f}",
         "C2" if (abs(c2a["avg_win"] / c2a["avg_loss"]) if c2a["avg_loss"] != 0 else 0) >
                 (abs(s1a["avg_win"] / s1a["avg_loss"]) if s1a["avg_loss"] != 0 else 0) else "S1"),
    ]
    for label, s1v, c2v, winner in metrics_to_compare:
        print(f"    {label:<25s}  {s1v:>10s}  {c2v:>10s}  {winner:>8s}")

    # Gate 2 verdict
    # FRAGILE if: top 20% contributes >80% of PnL
    c2_fragile = c2a["top_20pct_contribution"] > 0.80
    print(f"\n    GATE 2 VERDICT: {'FRAGILE — top 20% drives >80% of PnL' if c2_fragile else 'PASS — PnL not overly concentrated'}")

    # ══════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)

    print(f"\n  Gate 1 (Rolling Window): {'PASS' if g1_pass else 'REVIEW'}")
    print(f"  Gate 2 (PnL Concentration): {'FRAGILE' if c2_fragile else 'PASS'}")

    if g1_pass and not c2_fragile:
        print("\n  RECOMMENDATION: C2 passes both gates → proceed to Shadow phase")
    elif g1_pass and c2_fragile:
        print("\n  RECOMMENDATION: C2 wins most windows but PnL concentrated → Shadow with caution")
    elif not g1_pass and not c2_fragile:
        print("\n  RECOMMENDATION: C2 fails rolling window test → better as satellite, not primary")
    else:
        print("\n  RECOMMENDATION: C2 fails both gates → do NOT promote")

    # ── Save results ──
    out_dir = ROOT / "results" / f"c2_robustness_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save rolling window data
    rows = []
    for i in range(n_windows):
        rows.append({
            "window": i + 1,
            **{f"S1_{k}": v for k, v in rolling["S1"][i].items()},
            **{f"C2_{k}": v for k, v in rolling["C2"][i].items()},
        })
    pd.DataFrame(rows).to_csv(out_dir / "rolling_window_comparison.csv", index=False)

    # Save per-period PnL
    for name in STRATEGIES:
        rets = pnl_analysis[name]["period_rets"]
        pd.DataFrame({"period_return": rets}).to_csv(
            out_dir / f"{name}_period_pnl.csv", index=False
        )

    # Save summary
    summary = {
        "gate1_pass": g1_pass,
        "gate1_c2_win_rate": c2_better_return / n_windows,
        "gate1_n_windows": n_windows,
        "gate1_c2_worst_return": c2_worst_window["return"],
        "gate1_s1_worst_return": s1_worst_window["return"],
        "gate2_fragile": c2_fragile,
        "gate2_c2_top20_contribution": c2a["top_20pct_contribution"],
        "gate2_s1_top20_contribution": s1a["top_20pct_contribution"],
        "gate2_c2_gini": c2a["gini_coefficient"],
        "gate2_s1_gini": s1a["gini_coefficient"],
    }
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)

    print(f"\n  Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
