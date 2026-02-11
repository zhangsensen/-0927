#!/usr/bin/env python3
"""Validate top bucket-constrained candidates under F5+Exp4 production execution.

Compares new candidates against S1 baseline at identical execution params:
  FREQ=5, Exp4 ON (dr=0.10, mh=9), regime gate ON, T1_OPEN, med cost.

Left-tail priority metrics: HO MDD, worst month, HO Calmar.
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
from etf_strategy.core.utils.rebalance import shift_timing_signal
from etf_strategy.regime_gate import compute_regime_gate_arr

from aligned_metrics import compute_aligned_metrics
from batch_vec_backtest import run_vec_backtest

# ── Candidates to validate ────────────────────────────────────────────────
CANDIDATES = {
    "S1_baseline": "ADX_14D + OBV_SLOPE_10D + SHARPE_RATIO_20D + SLOPE_20D",
    "C1_top": "ADX_14D + AMIHUD_ILLIQUIDITY + GK_VOL_RATIO_20D + SHARPE_RATIO_20D",
    "C2_3factor": "AMIHUD_ILLIQUIDITY + CALMAR_RATIO_60D + CORRELATION_TO_MARKET_20D",
    "C3_5factor": "ADX_14D + AMIHUD_ILLIQUIDITY + GK_VOL_RATIO_20D + PRICE_POSITION_120D + SHARPE_RATIO_20D",
    "C4_pp20d": "ADX_14D + AMIHUD_ILLIQUIDITY + GK_VOL_RATIO_20D + PRICE_POSITION_20D",
    "C5_5f_pp": "ADX_14D + AMIHUD_ILLIQUIDITY + GK_VOL_RATIO_20D + PRICE_POSITION_120D + PRICE_POSITION_20D",
}

# Test F5+ON (production) and F3+OFF (research reference)
CONFIGS = {
    "F5_ON": {"freq": 5, "delta_rank": 0.10, "min_hold_days": 9},
    "F5_OFF": {"freq": 5, "delta_rank": 0.0, "min_hold_days": 0},
    "F3_OFF": {"freq": 3, "delta_rank": 0.0, "min_hold_days": 0},
}


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


def _worst_month(equity: np.ndarray, freq: int = 21) -> float:
    """Worst rolling month return."""
    if len(equity) < freq + 1:
        return 0.0
    worst = 0.0
    for i in range(freq, len(equity)):
        ret = equity[i] / equity[i - freq] - 1.0 if equity[i - freq] > 0 else 0.0
        if ret < worst:
            worst = ret
    return worst


def _half_year_returns(equity: np.ndarray, dates: list, start_idx: int) -> list:
    """Compute half-year returns for temporal stability check."""
    results = []
    # Group dates into ~126-day chunks
    chunk_size = 126
    eq_start = equity[0]
    chunk_start_val = eq_start
    chunk_start_i = 0

    for i in range(chunk_size, len(equity), chunk_size):
        if chunk_start_val > 0:
            ret = equity[i] / chunk_start_val - 1.0
        else:
            ret = 0.0
        actual_idx = start_idx + i
        label = str(dates[actual_idx]) if actual_idx < len(dates) else f"idx_{actual_idx}"
        results.append((label, ret))
        chunk_start_val = equity[i]
        chunk_start_i = i

    # Last partial chunk
    if chunk_start_i < len(equity) - 1 and chunk_start_val > 0:
        ret = equity[-1] / chunk_start_val - 1.0
        actual_idx = start_idx + len(equity) - 1
        label = str(dates[actual_idx]) if actual_idx < len(dates) else f"idx_{actual_idx}"
        results.append((label, ret))

    return results


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 90)
    print("BUCKET CANDIDATE VALIDATION (F5+Exp4 Production Execution)")
    print(f"  Candidates: {len(CANDIDATES)}")
    print(f"  Configs: {list(CONFIGS.keys())}")
    print(f"  Total runs: {len(CANDIDATES) * len(CONFIGS)}")
    print("=" * 90)

    # ── 1. Load config & data ──
    config_path = ROOT / "configs" / "combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    backtest_config = config.get("backtest", {})
    POS_SIZE = 2
    LOOKBACK = 252
    INITIAL_CAPITAL = float(backtest_config.get("initial_capital", 1_000_000))
    COMMISSION_RATE = float(backtest_config.get("commission_rate", 0.0002))

    exec_model = load_execution_model(config)
    cost_model = load_cost_model(config)
    tier = cost_model.active_tier
    qdii_set = set(FrozenETFPool().qdii_codes)

    risk_config = backtest_config.get("risk_control", {})
    trailing_stop_pct = risk_config.get("trailing_stop_pct", 0.0)
    stop_on_rebalance_only = risk_config.get("stop_check_on_rebalance_only", True)
    leverage_cap = risk_config.get("leverage_cap", 1.0)
    profit_ladders = risk_config.get("profit_ladders", [])

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

    print(f"  Data: {T} days x {N} ETFs x {len(factor_names)} factors")
    print(f"  Train: {dates[0]} ~ {dates[train_end_idx]}")
    print(f"  Holdout: {dates[train_end_idx+1]} ~ {dates[T-1]}")

    # ── 5. Resolve factor indices ──
    factor_index_map = {name: idx for idx, name in enumerate(factor_names)}
    candidate_indices = {}
    for name, combo_str in CANDIDATES.items():
        factors = [f.strip() for f in combo_str.split("+")]
        try:
            indices = [factor_index_map[f] for f in factors]
            candidate_indices[name] = indices
        except KeyError as e:
            print(f"  WARNING: {name}: factor {e} not found, skipping")

    # ── 6. Run validation matrix ──
    print(f"\nRunning {len(candidate_indices) * len(CONFIGS)} backtests...")
    results = []
    start_idx = LOOKBACK
    count = 0

    for cand_name, f_indices in candidate_indices.items():
        for cfg_name, cfg in CONFIGS.items():
            count += 1
            tag = f"[{count}] {cand_name} {cfg_name}"

            try:
                eq, ret, wr, pf, trades, _, risk = run_vec_backtest(
                    factors_3d, close_prices, open_prices, high_prices, low_prices,
                    timing_arr, f_indices,
                    freq=cfg["freq"], pos_size=POS_SIZE,
                    initial_capital=INITIAL_CAPITAL, commission_rate=COMMISSION_RATE,
                    lookback=LOOKBACK, cost_arr=COST_ARR,
                    trailing_stop_pct=trailing_stop_pct,
                    stop_on_rebalance_only=stop_on_rebalance_only,
                    leverage_cap=leverage_cap, profit_ladders=profit_ladders,
                    use_t1_open=exec_model.is_t1_open,
                    delta_rank=cfg["delta_rank"], min_hold_days=cfg["min_hold_days"],
                )

                # Split equity
                train_eq = eq[start_idx: train_end_idx + 1]
                holdout_eq = eq[train_end_idx:]

                train_m = compute_aligned_metrics(train_eq, start_idx=0)
                holdout_m = compute_aligned_metrics(holdout_eq, start_idx=0)
                ho_mdd = _compute_mdd(holdout_eq)
                ho_worst_mo = _worst_month(holdout_eq)

                # Half-year stability (holdout only)
                hy_rets = _half_year_returns(holdout_eq, dates, train_end_idx)
                n_pos_hy = sum(1 for _, r in hy_rets if r > 0)
                n_total_hy = len(hy_rets)

                ho_ret = holdout_m["aligned_return"]
                ho_calmar = ho_ret / ho_mdd if ho_mdd > 0 else 0.0

                results.append({
                    "candidate": cand_name,
                    "config": cfg_name,
                    "combo": CANDIDATES[cand_name],
                    "n_factors": len(f_indices),
                    "train_return": train_m["aligned_return"],
                    "train_sharpe": train_m["aligned_sharpe"],
                    "holdout_return": ho_ret,
                    "holdout_sharpe": holdout_m["aligned_sharpe"],
                    "holdout_mdd": ho_mdd,
                    "holdout_calmar": ho_calmar,
                    "holdout_worst_month": ho_worst_mo,
                    "half_year_pos_rate": n_pos_hy / n_total_hy if n_total_hy > 0 else 0,
                    "trades": trades,
                    "turnover": risk["turnover_ann"],
                    "cost_drag": risk["cost_drag"],
                    "win_rate": wr,
                })

                print(
                    f"  {tag:40s}"
                    f"  train={train_m['aligned_return']*100:+6.1f}%"
                    f"  HO={ho_ret*100:+6.1f}%"
                    f"  HO_MDD={ho_mdd*100:5.1f}%"
                    f"  HO_Calmar={ho_calmar:5.2f}"
                    f"  WrstMo={ho_worst_mo*100:+5.1f}%"
                    f"  trades={trades:>3}"
                )
            except Exception as e:
                print(f"  {tag}  ERROR: {e}")

    if not results:
        print("No results!")
        return

    df = pd.DataFrame(results)

    # ── 7. Decision table: F5_ON only (production comparison) ──
    print("\n" + "=" * 90)
    print("PRODUCTION COMPARISON (F5 + Exp4 ON)")
    print("=" * 90)
    f5on = df[df["config"] == "F5_ON"].sort_values("holdout_calmar", ascending=False)

    print(f"\n{'Candidate':15s} {'Combo':55s} {'HO Ret':>8s} {'HO MDD':>8s} {'HO Calmar':>10s} {'Wrst Mo':>8s} {'HY Pos%':>8s} {'Trades':>7s}")
    print("-" * 120)
    for _, r in f5on.iterrows():
        print(
            f"{r['candidate']:15s} "
            f"{r['combo']:55s} "
            f"{r['holdout_return']*100:>+7.1f}% "
            f"{r['holdout_mdd']*100:>7.1f}% "
            f"{r['holdout_calmar']:>9.2f} "
            f"{r['holdout_worst_month']*100:>+7.1f}% "
            f"{r['half_year_pos_rate']*100:>7.0f}% "
            f"{r['trades']:>6.0f}"
        )

    # ── 8. Left-tail ranking ──
    print("\n" + "=" * 90)
    print("LEFT-TAIL RANKING (F5_ON)")
    print("=" * 90)
    # Score: normalize each metric, weight toward left-tail
    # Lower MDD = better, less negative worst month = better, higher calmar = better
    f5on_copy = f5on.copy()
    f5on_copy["mdd_rank"] = f5on_copy["holdout_mdd"].rank(ascending=True)
    f5on_copy["wm_rank"] = f5on_copy["holdout_worst_month"].rank(ascending=False)
    f5on_copy["cal_rank"] = f5on_copy["holdout_calmar"].rank(ascending=False)
    f5on_copy["composite_rank"] = (
        0.35 * f5on_copy["mdd_rank"] +
        0.35 * f5on_copy["wm_rank"] +
        0.30 * f5on_copy["cal_rank"]
    )
    f5on_sorted = f5on_copy.sort_values("composite_rank")

    print(f"\n{'Rank':>4s} {'Candidate':15s} {'MDD_rk':>7s} {'WM_rk':>6s} {'Cal_rk':>7s} {'Composite':>10s}")
    print("-" * 55)
    for rank, (_, r) in enumerate(f5on_sorted.iterrows(), 1):
        print(
            f"{rank:>4d} {r['candidate']:15s} "
            f"{r['mdd_rank']:>6.1f} "
            f"{r['wm_rank']:>5.1f} "
            f"{r['cal_rank']:>6.1f} "
            f"{r['composite_rank']:>9.2f}"
        )

    # ── 9. F5_ON vs F5_OFF comparison (Exp4 effect per candidate) ──
    print("\n" + "=" * 90)
    print("EXP4 EFFECT (F5_ON - F5_OFF per candidate)")
    print("=" * 90)
    for cand in CANDIDATES:
        on_row = df[(df["candidate"] == cand) & (df["config"] == "F5_ON")]
        off_row = df[(df["candidate"] == cand) & (df["config"] == "F5_OFF")]
        if on_row.empty or off_row.empty:
            continue
        o, n = off_row.iloc[0], on_row.iloc[0]
        d_ret = n["holdout_return"] - o["holdout_return"]
        d_mdd = n["holdout_mdd"] - o["holdout_mdd"]
        d_trades = n["trades"] - o["trades"]
        print(
            f"  {cand:15s}"
            f"  D_HO_Ret={d_ret*100:+6.1f}pp"
            f"  D_HO_MDD={d_mdd*100:+6.1f}pp"
            f"  D_trades={d_trades:+4.0f}"
            f"  {'Exp4 HELPS' if d_ret > 0 and d_mdd < 0 else 'Exp4 MIXED' if d_ret > 0 or d_mdd < 0 else 'Exp4 HURTS'}"
        )

    # ── 10. Save ──
    out_dir = ROOT / "results" / f"bucket_candidate_validation_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "validation_results.csv", index=False)
    df.to_parquet(out_dir / "validation_results.parquet", index=False)
    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()
