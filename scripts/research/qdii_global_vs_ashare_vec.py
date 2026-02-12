#!/usr/bin/env python3
"""
QDII Strategic Validation: A_SHARE_ONLY vs GLOBAL VEC comparison.

Answers: "Is there significant alpha being left on the table by blocking QDII?"

Methodology:
  - Run A: factors_3d with QDII columns set to NaN → simulates A_SHARE_ONLY
  - Run B: factors_3d as-is → GLOBAL (all 49 ETFs eligible)
  - Both use identical F5 + Exp4 + med cost + regime gate
  - Track QDII selection frequency and return attribution in GLOBAL mode
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import yaml
import numpy as np
import pandas as pd
from datetime import datetime

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.factor_cache import FactorCache
from etf_strategy.core.frozen_params import load_frozen_config, FrozenETFPool
from etf_strategy.core.cost_model import load_cost_model, build_cost_array
from etf_strategy.core.execution_model import load_execution_model
from etf_strategy.core.utils.rebalance import (
    shift_timing_signal,
    generate_rebalance_schedule,
)
from etf_strategy.regime_gate import compute_regime_gate_arr
from aligned_metrics import compute_aligned_metrics

# Import VEC runner
from batch_vec_backtest import run_vec_backtest


# ── Strategy definitions ──
STRATEGIES = {
    "S1": ["ADX_14D", "OBV_SLOPE_10D", "SHARPE_RATIO_20D", "SLOPE_20D"],
    "C2": ["AMIHUD_ILLIQUIDITY", "CALMAR_RATIO_60D", "CORRELATION_TO_MARKET_20D"],
}


def compute_qdii_selection_stats(
    factors_3d: np.ndarray,
    factor_indices: list[int],
    rebalance_schedule: np.ndarray,
    qdii_mask: np.ndarray,  # (N,) bool
    pos_size: int,
    delta_rank: float,
    min_hold_days: int,
    freq: int,
) -> dict:
    """Count how often QDII ETFs are selected in GLOBAL mode (Python-level sim)."""
    from etf_strategy.core.hysteresis import apply_hysteresis
    from batch_vec_backtest import stable_topk_indices

    T, N, _ = factors_3d.shape
    n_rebalances = len(rebalance_schedule)
    qdii_selections = 0
    total_selections = 0

    # Hysteresis state
    h_mask = np.zeros(N, dtype=np.bool_)
    hold_days = np.zeros(N, dtype=np.int64)
    prev_target = np.zeros(N, dtype=np.bool_)

    qdii_select_dates = []

    for ri, t in enumerate(rebalance_schedule):
        # Compute combined score (matching fixed VEC kernel)
        combined_score = np.full(N, -np.inf)
        for n in range(N):
            s = 0.0
            nv = 0
            for idx in factor_indices:
                val = factors_3d[t - 1, n, idx]
                if not np.isnan(val):
                    s += val
                    nv += 1
            if nv > 0:
                combined_score[n] = s / nv

        valid = np.sum(combined_score > -np.inf)
        if valid < pos_size:
            continue

        top_indices = stable_topk_indices(combined_score, pos_size)

        # Apply hysteresis
        if (delta_rank > 0 or min_hold_days > 0) and len(top_indices) >= pos_size:
            target_mask = apply_hysteresis(
                combined_score, h_mask, hold_days, top_indices,
                pos_size, delta_rank, min_hold_days,
            )
        else:
            target_mask = np.zeros(N, dtype=np.bool_)
            for i in range(len(top_indices)):
                if combined_score[top_indices[i]] > -np.inf:
                    target_mask[top_indices[i]] = True

        # Update hold_days
        for n in range(N):
            if target_mask[n]:
                if prev_target[n]:
                    hold_days[n] += freq
                else:
                    hold_days[n] = 0
            else:
                hold_days[n] = 0
        h_mask[:] = target_mask
        prev_target[:] = target_mask

        # Count
        selected = np.where(target_mask)[0]
        for idx in selected:
            total_selections += 1
            if qdii_mask[idx]:
                qdii_selections += 1
                qdii_select_dates.append(t)

    return {
        "total_selections": total_selections,
        "qdii_selections": qdii_selections,
        "qdii_rate": qdii_selections / max(total_selections, 1),
        "n_rebalances": n_rebalances,
        "qdii_rebalance_dates": qdii_select_dates,
    }


def run_comparison(config_path: str | Path = None):
    """Run A_SHARE_ONLY vs GLOBAL VEC comparison for S1 and C2."""
    # ── Load config ──
    if config_path is None:
        config_path = ROOT / "configs/combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    frozen = load_frozen_config(config, config_path=str(config_path))
    print(f"🔒 Frozen params: v{frozen.version}")

    bt_cfg = config["backtest"]
    FREQ = bt_cfg["freq"]
    POS_SIZE = bt_cfg["pos_size"]
    LOOKBACK = bt_cfg.get("lookback") or bt_cfg.get("lookback_window")
    INITIAL_CAPITAL = float(bt_cfg["initial_capital"])
    COMMISSION_RATE = float(bt_cfg["commission_rate"])

    # Hysteresis
    hyst_cfg = bt_cfg.get("hysteresis", {})
    DELTA_RANK = hyst_cfg.get("delta_rank", 0.0)
    MIN_HOLD_DAYS = hyst_cfg.get("min_hold_days", 0)

    # Execution & cost
    exec_model = load_execution_model(config)
    cost_model = load_cost_model(config)
    qdii_set = set(FrozenETFPool().qdii_codes)

    print(f"📋 FREQ={FREQ}, POS={POS_SIZE}, dr={DELTA_RANK}, mh={MIN_HOLD_DAYS}")
    print(f"📋 Exec={exec_model.mode}, Cost={cost_model.mode}/{cost_model.tier}")

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

    # ── Compute factors ──
    factor_cache = FactorCache(
        cache_dir=Path(config["data"].get("cache_dir") or ".cache")
    )
    cached = factor_cache.get_or_compute(
        ohlcv=ohlcv, config=config, data_dir=loader.data_dir
    )
    factors_3d = cached["factors_3d"]
    factor_names = list(cached["factor_names"])
    dates = cached["dates"]
    etf_codes = list(cached["etf_codes"])
    T = len(dates)
    N = len(etf_codes)

    # ── Cost array ──
    COST_ARR = build_cost_array(cost_model, etf_codes, qdii_set)

    # ── Timing / Regime gate ──
    from etf_strategy.core.market_timing import LightTimingModule
    timing_config = bt_cfg.get("timing", {})
    timing_module = LightTimingModule(
        extreme_threshold=timing_config.get("extreme_threshold", -0.4),
        extreme_position=timing_config.get("extreme_position", 0.3),
    )
    timing_arr_raw = (
        timing_module.compute_position_ratios(ohlcv["close"])
        .reindex(dates).fillna(1.0).values
    )
    timing_arr = shift_timing_signal(timing_arr_raw)
    gate_arr = compute_regime_gate_arr(
        ohlcv["close"], dates, backtest_config=bt_cfg
    )
    timing_arr = (timing_arr.astype(np.float64) * gate_arr.astype(np.float64))

    # ── Price arrays ──
    close_prices = ohlcv["close"][etf_codes].ffill().fillna(1.0).values
    open_prices = ohlcv["open"][etf_codes].ffill().fillna(1.0).values
    high_prices = ohlcv["high"][etf_codes].ffill().fillna(1.0).values
    low_prices = ohlcv["low"][etf_codes].ffill().fillna(1.0).values

    # ── QDII mask ──
    qdii_mask = np.array([code in qdii_set for code in etf_codes], dtype=np.bool_)
    qdii_indices = np.where(qdii_mask)[0]
    qdii_names = [etf_codes[i] for i in qdii_indices]
    print(f"\n🌍 QDII ETFs ({len(qdii_indices)}): {qdii_names}")
    print(f"🇨🇳 A-share ETFs: {N - len(qdii_indices)}")

    # ── Prepare A_SHARE_ONLY factors (QDII columns = NaN) ──
    factors_3d_ashare = factors_3d.copy()
    factors_3d_ashare[:, qdii_mask, :] = np.nan
    print(f"✅ A_SHARE_ONLY factors: QDII columns set to NaN")

    # ── Holdout period detection ──
    holdout_start = pd.Timestamp("2025-05-01")
    train_end_idx = 0
    for i, d in enumerate(dates):
        if pd.Timestamp(d) >= holdout_start:
            train_end_idx = i
            break
    print(f"\n📅 Train: {dates[0].date()} ~ {dates[train_end_idx-1].date()}")
    print(f"📅 Holdout: {dates[train_end_idx].date()} ~ {dates[-1].date()}")

    # ── Rebalance schedule (for diagnostics) ──
    rebalance_schedule = generate_rebalance_schedule(T, LOOKBACK, FREQ)

    # ── Run comparisons ──
    results = {}

    for strat_name, factor_list in STRATEGIES.items():
        print(f"\n{'='*70}")
        print(f"Strategy: {strat_name} ({' + '.join(factor_list)})")
        print(f"{'='*70}")

        # Resolve factor indices
        factor_indices = []
        for f in factor_list:
            if f in factor_names:
                factor_indices.append(factor_names.index(f))
            else:
                print(f"⚠️  Factor {f} not found! Skipping strategy.")
                break
        else:
            pass
        if len(factor_indices) != len(factor_list):
            continue

        print(f"   Factor indices: {factor_indices}")

        # ── QDII selection stats (GLOBAL mode) ──
        qdii_stats = compute_qdii_selection_stats(
            factors_3d, factor_indices, rebalance_schedule,
            qdii_mask, POS_SIZE, DELTA_RANK, MIN_HOLD_DAYS, FREQ,
        )
        print(f"\n   📊 QDII Selection (GLOBAL mode):")
        print(f"      Total selections: {qdii_stats['total_selections']}")
        print(f"      QDII selections:  {qdii_stats['qdii_selections']}")
        print(f"      QDII rate:        {qdii_stats['qdii_rate']*100:.1f}%")

        # ── Run VEC: A_SHARE_ONLY ──
        print(f"\n   🏃 Running VEC: A_SHARE_ONLY...")
        eq_a, ret_a, wr_a, pf_a, trades_a, _, risk_a = run_vec_backtest(
            factors_3d_ashare, close_prices, open_prices, high_prices, low_prices,
            timing_arr, factor_indices,
            freq=FREQ, pos_size=POS_SIZE, initial_capital=INITIAL_CAPITAL,
            commission_rate=COMMISSION_RATE, lookback=LOOKBACK,
            cost_arr=COST_ARR, use_t1_open=exec_model.is_t1_open,
            delta_rank=DELTA_RANK, min_hold_days=MIN_HOLD_DAYS,
        )

        # ── Run VEC: GLOBAL ──
        print(f"   🏃 Running VEC: GLOBAL...")
        eq_g, ret_g, wr_g, pf_g, trades_g, _, risk_g = run_vec_backtest(
            factors_3d, close_prices, open_prices, high_prices, low_prices,
            timing_arr, factor_indices,
            freq=FREQ, pos_size=POS_SIZE, initial_capital=INITIAL_CAPITAL,
            commission_rate=COMMISSION_RATE, lookback=LOOKBACK,
            cost_arr=COST_ARR, use_t1_open=exec_model.is_t1_open,
            delta_rank=DELTA_RANK, min_hold_days=MIN_HOLD_DAYS,
        )

        # ── Compute holdout metrics from equity curves ──
        lookback_idx = LOOKBACK
        ho_start = train_end_idx

        def compute_period_metrics(eq, start, end, label):
            eq_slice = eq[start:end+1]
            if len(eq_slice) < 2 or eq_slice[0] == 0:
                return {"return": 0.0, "mdd": 0.0, "sharpe": 0.0}
            ret = eq_slice[-1] / eq_slice[0] - 1
            # Max drawdown
            peak = eq_slice[0]
            mdd = 0.0
            for v in eq_slice:
                if v > peak:
                    peak = v
                dd = (peak - v) / peak
                if dd > mdd:
                    mdd = dd
            # Sharpe from daily returns
            daily_rets = np.diff(eq_slice) / eq_slice[:-1]
            daily_rets = daily_rets[np.isfinite(daily_rets)]
            if len(daily_rets) > 1 and np.std(daily_rets) > 0:
                sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)
            else:
                sharpe = 0.0
            # Worst month
            n_months = len(daily_rets) // 21
            worst_month = 0.0
            for m in range(n_months):
                m_ret = np.prod(1 + daily_rets[m*21:(m+1)*21]) - 1
                if m_ret < worst_month:
                    worst_month = m_ret
            return {"return": ret, "mdd": mdd, "sharpe": sharpe, "worst_month": worst_month}

        # Full period
        full_a = compute_period_metrics(eq_a, lookback_idx, T-1, "full")
        full_g = compute_period_metrics(eq_g, lookback_idx, T-1, "full")

        # Train period
        train_a = compute_period_metrics(eq_a, lookback_idx, ho_start-1, "train")
        train_g = compute_period_metrics(eq_g, lookback_idx, ho_start-1, "train")

        # Holdout period
        ho_a = compute_period_metrics(eq_a, ho_start, T-1, "holdout")
        ho_g = compute_period_metrics(eq_g, ho_start, T-1, "holdout")

        results[strat_name] = {
            "A_SHARE_ONLY": {"full": full_a, "train": train_a, "holdout": ho_a,
                             "trades": trades_a},
            "GLOBAL": {"full": full_g, "train": train_g, "holdout": ho_g,
                       "trades": trades_g},
            "qdii_stats": qdii_stats,
        }

        # ── Print results ──
        print(f"\n   {'─'*60}")
        print(f"   {'Metric':<25} {'A_SHARE_ONLY':>15} {'GLOBAL':>15} {'Delta':>12}")
        print(f"   {'─'*60}")

        for period_name, period_label in [("full", "Full"), ("train", "Train"), ("holdout", "Holdout")]:
            ra = results[strat_name]["A_SHARE_ONLY"][period_name]
            rg = results[strat_name]["GLOBAL"][period_name]
            delta_ret = rg["return"] - ra["return"]
            delta_mdd = rg["mdd"] - ra["mdd"]
            delta_sharpe = rg["sharpe"] - ra["sharpe"]

            print(f"   {period_label + ' Return':<25} {ra['return']*100:>+14.1f}% {rg['return']*100:>+14.1f}% {delta_ret*100:>+11.1f}pp")
            print(f"   {period_label + ' MDD':<25} {ra['mdd']*100:>14.1f}% {rg['mdd']*100:>14.1f}% {delta_mdd*100:>+11.1f}pp")
            print(f"   {period_label + ' Sharpe':<25} {ra['sharpe']:>15.2f} {rg['sharpe']:>15.2f} {delta_sharpe:>+12.2f}")
            if "worst_month" in ra:
                print(f"   {period_label + ' Worst Month':<25} {ra['worst_month']*100:>+14.1f}% {rg['worst_month']*100:>+14.1f}%")
            print()

        print(f"   Trades:                 {trades_a:>15} {trades_g:>15} {trades_g - trades_a:>+12}")

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"SUMMARY: QDII Unlock Value Assessment")
    print(f"{'='*70}")
    for strat_name, r in results.items():
        ho_a = r["A_SHARE_ONLY"]["holdout"]
        ho_g = r["GLOBAL"]["holdout"]
        qs = r["qdii_stats"]
        delta = ho_g["return"] - ho_a["return"]
        print(f"\n{strat_name}:")
        print(f"  QDII selection rate: {qs['qdii_rate']*100:.1f}% ({qs['qdii_selections']}/{qs['total_selections']})")
        print(f"  HO Return: A_SHARE {ho_a['return']*100:+.1f}% → GLOBAL {ho_g['return']*100:+.1f}% (Δ={delta*100:+.1f}pp)")
        print(f"  HO MDD:    A_SHARE {ho_a['mdd']*100:.1f}% → GLOBAL {ho_g['mdd']*100:.1f}%")
        print(f"  HO Sharpe: A_SHARE {ho_a['sharpe']:.2f} → GLOBAL {ho_g['sharpe']:.2f}")

        if delta > 0.15:
            print(f"  ⚡ SIGNIFICANT: +{delta*100:.1f}pp holdout alpha from QDII — proceed to beta/FX attribution")
        elif delta > 0.05:
            print(f"  📊 MODERATE: +{delta*100:.1f}pp — worth investigating, but check if cost-sensitive")
        elif delta > 0:
            print(f"  📉 MARGINAL: +{delta*100:.1f}pp — likely noise, not actionable")
        else:
            print(f"  ❌ NEGATIVE: {delta*100:.1f}pp — QDII hurts under current framework")


if __name__ == "__main__":
    run_comparison()
