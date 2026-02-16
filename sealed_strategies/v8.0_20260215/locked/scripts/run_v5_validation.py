#!/usr/bin/env python3
"""v5.0 Candidate Validation: Rolling OOS + Cost Sensitivity

3-line comparison for sealed S1/S2:
  A: F=3,  Exp4=ON (dr=0.10, mh=9) — current prod + Exp4
  B: F=5,  Exp4=ON (dr=0.10, mh=9) — v5.0 candidate
  C: F=20, Exp4=OFF                 — low-freq baseline

Validation outputs:
  1. Quarterly rolling OOS: per-window returns, positive rate, worst window
  2. Cost sensitivity: med vs low tier
  3. Per-trade statistics: avg PnL, variance, trade count per window

Usage:
    uv run python scripts/run_v5_validation.py
    uv run python scripts/run_v5_validation.py --segment M    # monthly segments
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
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

from batch_vec_backtest import run_vec_backtest

# ── Experiment configs ─────────────────────────────────────────────────────
SEALED = {
    "S1": "ADX_14D + OBV_SLOPE_10D + SHARPE_RATIO_20D + SLOPE_20D",
    "S2": "ADX_14D + OBV_SLOPE_10D + PRICE_POSITION_120D + SHARPE_RATIO_20D + SLOPE_20D",
}

CONFIGS = {
    "F3_ON":  {"freq": 3,  "delta_rank": 0.10, "min_hold_days": 9},
    "F5_ON":  {"freq": 5,  "delta_rank": 0.10, "min_hold_days": 9},
    "F20_OFF": {"freq": 20, "delta_rank": 0.0,  "min_hold_days": 0},
}

COST_TIERS = ["med", "low"]


# ── Segment helpers ────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Segment:
    label: str
    start_idx: int
    end_idx: int
    is_holdout: bool


def _build_segments(
    dates: pd.Index,
    start_idx: int,
    seg_type: str,
    training_end_ts: pd.Timestamp,
) -> list[Segment]:
    dt_idx = pd.DatetimeIndex(dates)
    if seg_type == "Q":
        periods = dt_idx.to_period("Q")
    elif seg_type == "M":
        periods = dt_idx.to_period("M")
    else:
        periods = dt_idx.to_period("Y")

    segments: list[Segment] = []
    cursor = start_idx
    while cursor < dt_idx.size - 1:
        p = periods[cursor]
        end = cursor + 1
        while end < dt_idx.size and periods[end] == p:
            end += 1
        if end - cursor > 1:
            is_ho = bool(dt_idx[cursor] > training_end_ts)
            segments.append(Segment(str(p), cursor, end, is_ho))
        cursor = end
    return segments


def _window_metrics(eq: np.ndarray, s: int, e: int) -> dict:
    w = eq[s:e]
    w = w[np.isfinite(w)]
    if w.size <= 1:
        return {"ret": 0.0, "mdd": 0.0, "sharpe": 0.0}
    ret = float((w[-1] - w[0]) / w[0]) if w[0] != 0 else 0.0
    # MDD
    peak = np.maximum.accumulate(w)
    dd = np.where(peak > 0, (w - peak) / peak, 0.0)
    mdd = float(abs(dd.min())) if dd.size else 0.0
    # Sharpe
    rets = np.diff(w) / w[:-1]
    rets = rets[np.isfinite(rets)]
    sharpe = 0.0
    if rets.size > 1:
        std = float(np.std(rets, ddof=1))
        if std > 1e-12:
            sharpe = float(np.mean(rets) / std * np.sqrt(252.0))
    return {"ret": ret, "mdd": mdd, "sharpe": sharpe}


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--segment", type=str, default="Q", help="Segment type: Q (quarterly), M (monthly)")
    return p.parse_args()


def main():
    args = _parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seg_type = args.segment.upper()

    total_runs = len(SEALED) * len(CONFIGS) * len(COST_TIERS)
    print("=" * 80)
    print("v5.0 Candidate Validation")
    print(f"  Strategies: {list(SEALED.keys())}")
    print(f"  Configs: {list(CONFIGS.keys())}")
    print(f"  Cost tiers: {COST_TIERS}")
    print(f"  Segments: {seg_type}")
    print(f"  Total VEC runs: {total_runs}")
    print("=" * 80)

    # ── 1. Load config ─────────────────────────────────────────────────────
    config_path = ROOT / "configs" / "combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    backtest_config = config.get("backtest", {})
    LOOKBACK = 252
    POS_SIZE = 2
    INITIAL_CAPITAL = float(backtest_config.get("initial_capital", 1_000_000))
    COMMISSION_RATE = float(backtest_config.get("commission_rate", 0.0002))

    exec_model = load_execution_model(config)
    USE_T1_OPEN = exec_model.is_t1_open

    risk_config = backtest_config.get("risk_control", {})
    trailing_stop_pct = risk_config.get("trailing_stop_pct", 0.0)
    stop_on_rebalance_only = risk_config.get("stop_check_on_rebalance_only", True)
    leverage_cap = risk_config.get("leverage_cap", 1.0)
    profit_ladders = risk_config.get("profit_ladders", [])

    qdii_set = set(FrozenETFPool().qdii_codes)
    print(f"  Execution: {exec_model.mode}")

    # ── 2. Load data & factors ─────────────────────────────────────────────
    print("\nLoading data...")
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
    factors_3d = cached["factors_3d"]
    factor_names = cached["factor_names"]
    dates = cached["dates"]
    etf_codes = cached["etf_codes"]
    T, N = len(dates), len(etf_codes)

    close_prices = ohlcv["close"][etf_codes].ffill().fillna(1.0).values
    open_prices = ohlcv["open"][etf_codes].ffill().fillna(1.0).values
    high_prices = ohlcv["high"][etf_codes].ffill().fillna(1.0).values
    low_prices = ohlcv["low"][etf_codes].ffill().fillna(1.0).values

    # ── 3. Timing + regime gate ────────────────────────────────────────────
    timing_cfg = backtest_config.get("timing", {})
    timing_module = LightTimingModule(
        extreme_threshold=float(timing_cfg.get("extreme_threshold", -0.1)),
        extreme_position=float(timing_cfg.get("extreme_position", 0.1)),
    )
    timing_raw = (
        timing_module.compute_position_ratios(ohlcv["close"])
        .reindex(dates).fillna(1.0).values
    )
    timing_arr = shift_timing_signal(timing_raw)
    gate_arr = compute_regime_gate_arr(
        ohlcv["close"], dates, backtest_config=backtest_config
    )
    timing_arr = (timing_arr.astype(np.float64) * gate_arr.astype(np.float64))

    print(f"  Data: {T} days x {N} ETFs x {len(factor_names)} factors")

    # ── 4. Resolve factor indices ──────────────────────────────────────────
    factor_index_map = {name: idx for idx, name in enumerate(factor_names)}
    sealed_indices: dict[str, list[int]] = {}
    for name, combo_str in SEALED.items():
        factors = [f.strip() for f in combo_str.split("+")]
        sealed_indices[name] = [factor_index_map[f] for f in factors]

    # ── 5. Build segments ──────────────────────────────────────────────────
    training_end_ts = pd.Timestamp(
        config["data"].get("training_end_date", "2025-04-30")
    )
    segments = _build_segments(dates, LOOKBACK, seg_type, training_end_ts)
    n_train_seg = sum(1 for s in segments if not s.is_holdout)
    n_ho_seg = sum(1 for s in segments if s.is_holdout)
    print(f"  Segments: {len(segments)} total ({n_train_seg} train, {n_ho_seg} holdout)")

    # ── 6. Build cost arrays for each tier ─────────────────────────────────
    cost_arrays: dict[str, np.ndarray] = {}
    for tier_name in COST_TIERS:
        config_copy = yaml.safe_load(yaml.dump(config))
        config_copy["backtest"]["cost_model"]["tier"] = tier_name
        cm = load_cost_model(config_copy)
        cost_arrays[tier_name] = build_cost_array(cm, list(etf_codes), qdii_set)
        t = cm.active_tier
        print(f"  Cost[{tier_name}]: A={t.a_share*10000:.0f}bp QDII={t.qdii*10000:.0f}bp")

    # ── 7. Run all combinations ────────────────────────────────────────────
    print(f"\nRunning {total_runs} VEC backtests...")
    all_results = []
    count = 0

    for strat_name, f_indices in sealed_indices.items():
        for cfg_name, cfg_params in CONFIGS.items():
            for tier_name in COST_TIERS:
                count += 1
                tag = f"[{count}/{total_runs}] {strat_name} {cfg_name} cost={tier_name}"

                try:
                    eq, ret, wr, pf, trades, _, risk = run_vec_backtest(
                        factors_3d,
                        close_prices,
                        open_prices,
                        high_prices,
                        low_prices,
                        timing_arr,
                        f_indices,
                        freq=cfg_params["freq"],
                        pos_size=POS_SIZE,
                        initial_capital=INITIAL_CAPITAL,
                        commission_rate=COMMISSION_RATE,
                        lookback=LOOKBACK,
                        cost_arr=cost_arrays[tier_name],
                        trailing_stop_pct=trailing_stop_pct,
                        stop_on_rebalance_only=stop_on_rebalance_only,
                        leverage_cap=leverage_cap,
                        profit_ladders=profit_ladders,
                        use_t1_open=USE_T1_OPEN,
                        delta_rank=cfg_params["delta_rank"],
                        min_hold_days=cfg_params["min_hold_days"],
                    )

                    # Compute per-segment metrics
                    seg_metrics = []
                    for seg in segments:
                        m = _window_metrics(eq, seg.start_idx, seg.end_idx)
                        m["label"] = seg.label
                        m["is_holdout"] = seg.is_holdout
                        seg_metrics.append(m)

                    # Aggregate
                    train_segs = [m for m in seg_metrics if not m["is_holdout"]]
                    ho_segs = [m for m in seg_metrics if m["is_holdout"]]
                    all_segs_rets = [m["ret"] for m in seg_metrics]
                    train_rets = [m["ret"] for m in train_segs]
                    ho_rets = [m["ret"] for m in ho_segs]

                    pos_rate_all = sum(1 for r in all_segs_rets if r > 0) / max(len(all_segs_rets), 1)
                    pos_rate_train = sum(1 for r in train_rets if r > 0) / max(len(train_rets), 1)
                    pos_rate_ho = sum(1 for r in ho_rets if r > 0) / max(len(ho_rets), 1)

                    row = {
                        "strategy": strat_name,
                        "config": cfg_name,
                        "cost_tier": tier_name,
                        "freq": cfg_params["freq"],
                        "exp4": "ON" if cfg_params["delta_rank"] > 0 else "OFF",
                        # Full period
                        "full_return": ret,
                        "full_sharpe": risk["sharpe_ratio"],
                        "full_mdd": risk["max_drawdown"],
                        "turnover": risk["turnover_ann"],
                        "cost_drag": risk["cost_drag"],
                        "trades": trades,
                        "win_rate": wr,
                        # Segment consistency
                        "seg_count": len(segments),
                        "seg_pos_rate_all": pos_rate_all,
                        "seg_pos_rate_train": pos_rate_train,
                        "seg_pos_rate_holdout": pos_rate_ho,
                        "seg_worst_ret": min(all_segs_rets) if all_segs_rets else 0,
                        "seg_median_ret": float(np.median(all_segs_rets)) if all_segs_rets else 0,
                        # Holdout aggregate
                        "ho_return": sum(ho_rets) if ho_rets else 0,  # approx
                        "ho_worst_seg": min(ho_rets) if ho_rets else 0,
                        "ho_best_seg": max(ho_rets) if ho_rets else 0,
                        "ho_median_sharpe": float(np.median([m["sharpe"] for m in ho_segs])) if ho_segs else 0,
                    }
                    all_results.append(row)

                    # Store segment detail for later
                    for m in seg_metrics:
                        m["strategy"] = strat_name
                        m["config"] = cfg_name
                        m["cost_tier"] = tier_name

                    h = f"HO:{pos_rate_ho*100:.0f}%pos"
                    print(f"  {tag}  ret={ret*100:+6.1f}%  TO={risk['turnover_ann']:5.1f}x  {h}  worst={min(all_segs_rets)*100:+.1f}%")

                except Exception as e:
                    print(f"  {tag}  ERROR: {e}")

    if not all_results:
        print("No results!")
        return

    df = pd.DataFrame(all_results)

    # ── 8. Rolling OOS Comparison Table ────────────────────────────────────
    print("\n" + "=" * 80)
    print("ROLLING OOS CONSISTENCY (quarterly segments)")
    print("=" * 80)

    for strat in sorted(df["strategy"].unique()):
        ds = df[df["strategy"] == strat]
        print(f"\n  {strat}: {SEALED.get(strat, '')}")
        print(
            f"  {'Config':<10} {'Cost':<5}"
            f" | {'FullRet%':>9} {'TO(x)':>7} {'CstDrg%':>8}"
            f" | {'SegPos%':>8} {'HO_Pos%':>8} {'WorstSeg%':>10}"
            f" | {'HO_Worst%':>10} {'HO_Best%':>9} {'HO_MdShrp':>10}"
        )
        print("  " + "-" * 110)

        for _, row in ds.sort_values(["config", "cost_tier"]).iterrows():
            print(
                f"  {row['config']:<10} {row['cost_tier']:<5}"
                f" | {row['full_return']*100:>+8.1f}% {row['turnover']:>6.1f}x {row['cost_drag']*100:>7.1f}%"
                f" | {row['seg_pos_rate_all']*100:>7.0f}% {row['seg_pos_rate_holdout']*100:>7.0f}% {row['seg_worst_ret']*100:>+9.1f}%"
                f" | {row['ho_worst_seg']*100:>+9.1f}% {row['ho_best_seg']*100:>+8.1f}% {row['ho_median_sharpe']:>+9.3f}"
            )

    # ── 9. Cost Sensitivity Analysis ───────────────────────────────────────
    print("\n" + "=" * 80)
    print("COST SENSITIVITY (med vs low)")
    print("=" * 80)

    for strat in sorted(df["strategy"].unique()):
        ds = df[df["strategy"] == strat]
        print(f"\n  {strat}:")
        print(
            f"  {'Config':<10}"
            f" | {'Ret@med':>9} {'Ret@low':>9} {'D_cost':>8}"
            f" | {'TO@med':>8} {'TO@low':>8}"
            f" | {'HO_Pos@med':>11} {'HO_Pos@low':>11}"
        )
        print("  " + "-" * 90)

        for cfg_name in CONFIGS:
            med = ds[(ds["config"] == cfg_name) & (ds["cost_tier"] == "med")]
            low = ds[(ds["config"] == cfg_name) & (ds["cost_tier"] == "low")]
            if med.empty or low.empty:
                continue
            m, l = med.iloc[0], low.iloc[0]
            d_ret = l["full_return"] - m["full_return"]
            print(
                f"  {cfg_name:<10}"
                f" | {m['full_return']*100:>+8.1f}% {l['full_return']*100:>+8.1f}% {d_ret*100:>+7.1f}%"
                f" | {m['turnover']:>7.1f}x {l['turnover']:>7.1f}x"
                f" | {m['seg_pos_rate_holdout']*100:>10.0f}% {l['seg_pos_rate_holdout']*100:>10.0f}%"
            )

    # ── 10. Decision Summary ───────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("DECISION SUMMARY")
    print("=" * 80)

    for strat in sorted(df["strategy"].unique()):
        ds = df[df["strategy"] == strat]
        print(f"\n  {strat}:")

        for cfg_name in CONFIGS:
            rows = ds[ds["config"] == cfg_name]
            if rows.empty:
                continue
            med_row = rows[rows["cost_tier"] == "med"]
            if med_row.empty:
                continue
            r = med_row.iloc[0]

            # Assess
            checks = []
            if r["seg_pos_rate_holdout"] >= 0.5:
                checks.append("HO_pos>=50%")
            else:
                checks.append("HO_pos<50%!")
            if r["seg_worst_ret"] > -0.15:
                checks.append("worst>-15%")
            else:
                checks.append(f"worst={r['seg_worst_ret']*100:.1f}%!")
            if r["turnover"] < 25:
                checks.append(f"TO={r['turnover']:.0f}x OK")
            else:
                checks.append(f"TO={r['turnover']:.0f}x HIGH")

            # Cost robustness
            low_row = rows[rows["cost_tier"] == "low"]
            if not low_row.empty:
                lr = low_row.iloc[0]
                if lr["seg_pos_rate_holdout"] >= r["seg_pos_rate_holdout"]:
                    checks.append("cost-robust")
                else:
                    checks.append("cost-sensitive!")

            status = " | ".join(checks)
            print(f"    {cfg_name:<10}: {status}")

    # ── 11. Save ───────────────────────────────────────────────────────────
    out_dir = ROOT / "results" / f"v5_validation_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "validation_summary.csv", index=False)
    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()
