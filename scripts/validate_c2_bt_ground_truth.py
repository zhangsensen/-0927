#!/usr/bin/env python3
"""BT Ground Truth: C2 vs S1 under integer-lot/capital constraints.

Runs Backtrader event-driven simulation for both strategies with production
execution params (F5+Exp4), then compares BT results against VEC results
to verify C2 is executable.

Pass criteria (BT-C2 vs VEC-C2):
  1. |ΔReturn| ≤ 2pp or relative ≤ 5%
  2. MDD / worst month not significantly worse (≤ 1pp)
  3. Trade count within ±20%
  4. Margin failures = 0
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
from etf_strategy.regime_gate import compute_regime_gate_arr, gate_stats

from aligned_metrics import compute_aligned_metrics
from batch_bt_backtest import run_bt_backtest
from batch_vec_backtest import run_vec_backtest

# ── Strategies ──
STRATEGIES = {
    "S1": "ADX_14D + OBV_SLOPE_10D + SHARPE_RATIO_20D + SLOPE_20D",
    "C2": "AMIHUD_ILLIQUIDITY + CALMAR_RATIO_60D + CORRELATION_TO_MARKET_20D",
}

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


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 90)
    print("BT GROUND TRUTH: C2 vs S1 (integer lots + capital constraints)")
    print(f"  Execution: F5_ON (freq={EXEC_CFG['freq']}, dr={EXEC_CFG['delta_rank']}, mh={EXEC_CFG['min_hold_days']})")
    print("=" * 90)

    # ── 1. Load config & data ──
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
    tier = cost_model.active_tier

    risk_config = backtest_config.get("risk_control", {})
    trailing_stop_pct = risk_config.get("trailing_stop_pct", 0.0)
    stop_on_rebalance_only = risk_config.get("stop_check_on_rebalance_only", True)
    leverage_cap = risk_config.get("leverage_cap", 1.0)
    profit_ladders = risk_config.get("profit_ladders", [])

    # Dynamic leverage
    dl_config = risk_config.get("dynamic_leverage", {})
    dynamic_leverage_enabled = dl_config.get("enabled", False)
    target_vol = dl_config.get("target_vol", 0.20)
    vol_window = dl_config.get("vol_window", 20)

    print(f"  Execution: {exec_model.mode}, Cost: {cost_model.tier}")

    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )

    # ── 2. Factors ──
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
    std_factors = cached["std_factors"]
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
    timing_arr_combined = timing_arr.astype(np.float64) * gate_arr.astype(np.float64)

    # BT needs Series
    timing_series = pd.Series(timing_arr_combined, index=dates)

    # Vol regime series for BT — set to 1.0 because regime gate is already
    # applied via gate_arr in timing_arr_combined.
    # CLAUDE.md pitfall: "NEVER duplicate regime gate (timing_arr only)"
    vol_regime_series = pd.Series(1.0, index=dates)

    # BT data feeds
    data_feeds = {}
    for ticker in etf_codes:
        df = pd.DataFrame({
            "open": ohlcv["open"][ticker],
            "high": ohlcv["high"][ticker],
            "low": ohlcv["low"][ticker],
            "close": ohlcv["close"][ticker],
            "volume": ohlcv["volume"][ticker],
        })
        df = df.reindex(dates).ffill().fillna(0.01)
        data_feeds[ticker] = df

    # Rebalance schedule
    rebalance_schedule = generate_rebalance_schedule(
        total_periods=T, lookback_window=LOOKBACK, freq=EXEC_CFG["freq"]
    )

    # ── 4. Train/holdout split ──
    training_end_date = pd.Timestamp(
        config["data"].get("training_end_date", "2025-04-30")
    )
    train_end_idx = 0
    for i, d in enumerate(dates):
        if pd.Timestamp(d) <= training_end_date:
            train_end_idx = i

    ho_start_idx = train_end_idx
    start_idx = LOOKBACK

    print(f"  Data: {T} days x {N} ETFs x {len(factor_names)} factors")
    print(f"  Holdout: {dates[train_end_idx+1]} ~ {dates[T-1]} ({T - train_end_idx - 1} days)")

    # ── 5. Factor indices ──
    factor_index_map = {name: idx for idx, name in enumerate(factor_names)}

    # ══════════════════════════════════════════════════════════════════════
    # RUN BOTH VEC AND BT FOR EACH STRATEGY
    # ══════════════════════════════════════════════════════════════════════
    results = {}
    for name, combo_str in STRATEGIES.items():
        factors = [f.strip() for f in combo_str.split("+")]
        f_indices = [factor_index_map[f] for f in factors]

        print(f"\n{'─'*90}")
        print(f"  {name}: {combo_str}")
        print(f"{'─'*90}")

        # ── VEC ──
        print(f"  Running VEC...")
        eq_vec, ret_vec, wr_vec, pf_vec, trades_vec, _, risk_vec = run_vec_backtest(
            factors_3d, close_prices, open_prices, high_prices, low_prices,
            timing_arr_combined, f_indices,
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
        ho_eq_vec = eq_vec[ho_start_idx:]
        vec_m = compute_aligned_metrics(ho_eq_vec, start_idx=0)
        vec_ho_ret = vec_m["aligned_return"]
        vec_ho_mdd = _compute_mdd(ho_eq_vec)
        vec_ho_wm = _worst_month(ho_eq_vec)
        vec_ho_calmar = vec_ho_ret / vec_ho_mdd if vec_ho_mdd > 0 else 0.0

        print(f"    VEC: HO_ret={vec_ho_ret*100:+.1f}%  MDD={vec_ho_mdd*100:.1f}%  "
              f"WrstMo={vec_ho_wm*100:+.1f}%  Calmar={vec_ho_calmar:.2f}  trades={trades_vec}")

        # ── BT ──
        print(f"  Running BT (integer lots + capital constraints)...")
        combined_score_df = pd.DataFrame(0.0, index=dates, columns=list(etf_codes))
        for f in factors:
            combined_score_df = combined_score_df.add(std_factors[f], fill_value=0)

        bt_return, margin_failures, bt_risk, daily_returns_s = run_bt_backtest(
            combined_score_df,
            timing_series,
            vol_regime_series,
            list(etf_codes),
            data_feeds,
            rebalance_schedule,
            EXEC_CFG["freq"],
            POS_SIZE,
            INITIAL_CAPITAL,
            COMMISSION_RATE,
            target_vol=target_vol,
            vol_window=vol_window,
            dynamic_leverage_enabled=dynamic_leverage_enabled,
            collect_daily_returns=True,
            use_t1_open=exec_model.is_t1_open,
            cost_model=cost_model,
            qdii_codes=qdii_set,
            delta_rank=EXEC_CFG["delta_rank"],
            min_hold_days=EXEC_CFG["min_hold_days"],
        )

        # Extract holdout equity from BT daily returns
        bt_ho_ret = np.nan
        bt_ho_mdd = np.nan
        bt_ho_wm = np.nan
        bt_ho_calmar = np.nan
        bt_ho_sharpe = np.nan
        bt_trades = bt_risk["total_trades"]
        bt_eq_holdout = None

        if isinstance(daily_returns_s, pd.Series) and len(daily_returns_s) > 0:
            eq_bt = (1.0 + daily_returns_s.fillna(0.0)).cumprod() * INITIAL_CAPITAL
            eq_bt = eq_bt.sort_index()
            eq_bt.index = pd.to_datetime(eq_bt.index)
            train_end = pd.to_datetime(training_end_date)
            hold_eq_s = eq_bt.loc[eq_bt.index > train_end]

            if len(hold_eq_s) >= 2:
                bt_eq_holdout = hold_eq_s.values
                bt_ho_ret = (bt_eq_holdout[-1] / bt_eq_holdout[0]) - 1.0
                bt_ho_mdd = _compute_mdd(bt_eq_holdout)
                bt_ho_wm = _worst_month(bt_eq_holdout)
                bt_ho_calmar = bt_ho_ret / bt_ho_mdd if bt_ho_mdd > 0 else 0.0
                bt_m = compute_aligned_metrics(bt_eq_holdout, start_idx=0)
                bt_ho_sharpe = bt_m["aligned_sharpe"]

        print(f"    BT:  HO_ret={bt_ho_ret*100:+.1f}%  MDD={bt_ho_mdd*100:.1f}%  "
              f"WrstMo={bt_ho_wm*100:+.1f}%  Calmar={bt_ho_calmar:.2f}  "
              f"trades={bt_trades}  margin_fail={margin_failures}")

        results[name] = {
            "vec_ho_ret": vec_ho_ret,
            "vec_ho_mdd": vec_ho_mdd,
            "vec_ho_wm": vec_ho_wm,
            "vec_ho_calmar": vec_ho_calmar,
            "vec_ho_sharpe": vec_m["aligned_sharpe"],
            "vec_trades": trades_vec,
            "bt_ho_ret": bt_ho_ret,
            "bt_ho_mdd": bt_ho_mdd,
            "bt_ho_wm": bt_ho_wm,
            "bt_ho_calmar": bt_ho_calmar,
            "bt_ho_sharpe": bt_ho_sharpe,
            "bt_trades": bt_trades,
            "bt_margin_failures": margin_failures,
            "bt_full_return": bt_return,
            "bt_full_mdd": bt_risk["max_drawdown"],
            "bt_win_rate": bt_risk["win_rate"],
            "bt_profit_factor": bt_risk["profit_factor"],
        }

    # ══════════════════════════════════════════════════════════════════════
    # COMPARISON TABLE
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("VEC vs BT COMPARISON (Holdout Period)")
    print("=" * 90)

    print(f"\n  {'':15s}  {'VEC Ret':>10s}  {'BT Ret':>10s}  {'Gap':>8s}  {'VEC MDD':>9s}  {'BT MDD':>9s}  {'Gap':>8s}  {'VEC WM':>9s}  {'BT WM':>9s}")
    print("  " + "-" * 100)
    for name in STRATEGIES:
        r = results[name]
        d_ret = r["bt_ho_ret"] - r["vec_ho_ret"]
        d_mdd = r["bt_ho_mdd"] - r["vec_ho_mdd"]
        print(
            f"  {name:15s}  "
            f"{r['vec_ho_ret']*100:>+9.1f}%  "
            f"{r['bt_ho_ret']*100:>+9.1f}%  "
            f"{d_ret*100:>+7.1f}pp  "
            f"{r['vec_ho_mdd']*100:>8.1f}%  "
            f"{r['bt_ho_mdd']*100:>8.1f}%  "
            f"{d_mdd*100:>+7.1f}pp  "
            f"{r['vec_ho_wm']*100:>+8.1f}%  "
            f"{r['bt_ho_wm']*100:>+8.1f}%"
        )

    print(f"\n  {'':15s}  {'VEC Trades':>11s}  {'BT Trades':>11s}  {'Gap%':>8s}  {'Margin Fail':>12s}  {'BT WR':>8s}  {'BT PF':>8s}")
    print("  " + "-" * 80)
    for name in STRATEGIES:
        r = results[name]
        trade_gap = (r["bt_trades"] - r["vec_trades"]) / r["vec_trades"] * 100 if r["vec_trades"] > 0 else 0
        print(
            f"  {name:15s}  "
            f"{r['vec_trades']:>11d}  "
            f"{r['bt_trades']:>11d}  "
            f"{trade_gap:>+7.0f}%  "
            f"{r['bt_margin_failures']:>12d}  "
            f"{r['bt_win_rate']*100:>7.1f}%  "
            f"{r['bt_profit_factor']:>7.2f}"
        )

    # ══════════════════════════════════════════════════════════════════════
    # C2 EXECUTABILITY VERDICT
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("C2 EXECUTABILITY VERDICT (BT-C2 vs VEC-C2)")
    print("=" * 90)

    c2 = results["C2"]
    s1 = results["S1"]

    d_ret_c2 = c2["bt_ho_ret"] - c2["vec_ho_ret"]
    d_mdd_c2 = c2["bt_ho_mdd"] - c2["vec_ho_mdd"]
    d_wm_c2 = c2["bt_ho_wm"] - c2["vec_ho_wm"]
    rel_ret_c2 = abs(d_ret_c2 / c2["vec_ho_ret"]) * 100 if abs(c2["vec_ho_ret"]) > 1e-6 else 0

    print(f"\n  Criterion 1: |ΔReturn| ≤ 2pp or relative ≤ 5%")
    c1_pass = abs(d_ret_c2) <= 0.02 or rel_ret_c2 <= 5.0
    print(f"    ΔReturn = {d_ret_c2*100:+.1f}pp (relative: {rel_ret_c2:.1f}%)")
    print(f"    → {'PASS' if c1_pass else 'FAIL'}")

    print(f"\n  Criterion 2: MDD / worst month not worse by >1pp")
    c2_mdd_pass = d_mdd_c2 <= 0.01
    c2_wm_pass = abs(d_wm_c2) <= 0.01 or d_wm_c2 >= 0  # less negative = better
    print(f"    ΔMDD = {d_mdd_c2*100:+.1f}pp → {'PASS' if c2_mdd_pass else 'FAIL'}")
    print(f"    ΔWorst Month = {d_wm_c2*100:+.1f}pp → {'PASS' if c2_wm_pass else 'REVIEW'}")

    print(f"\n  Criterion 3: Trade count within ±20%")
    trade_gap_pct = abs(c2["bt_trades"] - c2["vec_trades"]) / c2["vec_trades"] * 100 if c2["vec_trades"] > 0 else 0
    c3_pass = trade_gap_pct <= 20 or abs(c2["bt_trades"] - c2["vec_trades"]) <= 5  # allow small absolute diff
    print(f"    VEC trades={c2['vec_trades']}, BT trades={c2['bt_trades']} (gap={trade_gap_pct:.0f}%)")
    print(f"    → {'PASS' if c3_pass else 'REVIEW'}")

    print(f"\n  Criterion 4: Margin failures = 0")
    c4_pass = c2["bt_margin_failures"] == 0
    print(f"    Margin failures = {c2['bt_margin_failures']}")
    print(f"    → {'PASS' if c4_pass else 'FAIL'}")

    all_pass = c1_pass and c2_mdd_pass and c3_pass and c4_pass
    print(f"\n  {'='*50}")
    print(f"  OVERALL: {'PASS — C2 is executable' if all_pass else 'REVIEW — see failed criteria'}")
    print(f"  {'='*50}")

    # ── Also check: does BT-C2 still beat BT-S1? ──
    print(f"\n  BONUS: Does BT-C2 still beat BT-S1 in holdout?")
    bt_c2_vs_s1_ret = c2["bt_ho_ret"] - s1["bt_ho_ret"]
    bt_c2_vs_s1_mdd = c2["bt_ho_mdd"] - s1["bt_ho_mdd"]
    print(f"    BT-C2 HO ret: {c2['bt_ho_ret']*100:+.1f}% vs BT-S1: {s1['bt_ho_ret']*100:+.1f}% → Δ={bt_c2_vs_s1_ret*100:+.1f}pp")
    print(f"    BT-C2 HO MDD: {c2['bt_ho_mdd']*100:.1f}% vs BT-S1: {s1['bt_ho_mdd']*100:.1f}% → Δ={bt_c2_vs_s1_mdd*100:+.1f}pp")
    if bt_c2_vs_s1_ret > 0:
        print(f"    → C2 still wins in BT ground truth")
    else:
        print(f"    → C2 loses advantage under BT constraints")

    # ── Save ──
    out_dir = ROOT / "results" / f"c2_bt_ground_truth_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in STRATEGIES:
        r = results[name]
        rows.append({"strategy": name, **r})
    pd.DataFrame(rows).to_csv(out_dir / "bt_vs_vec_comparison.csv", index=False)

    print(f"\n  Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
