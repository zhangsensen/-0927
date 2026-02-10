#!/usr/bin/env python3
"""FREQ × Exp4 Ablation Study (消融实验)

Scans FREQ ∈ {3, 4, 5, 7, 10, 20, 30} × Exp4 ∈ {OFF, ON} for sealed S1/S2,
computing train and holdout period metrics separately.

Determines whether reducing frequency or adding Exp4 filtering (hysteresis +
min_hold) contributes more to performance improvement.

Fixed params: T1_OPEN, med cost, POS_SIZE=2, LOOKBACK=252, A_SHARE_ONLY mode

Usage:
    uv run python scripts/run_freq_x_exp4_scan.py
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

# ── Experiment matrix ──────────────────────────────────────────────────────
FREQS = [3, 4, 5, 7, 10, 20, 30]
EXP4_CONFIGS = {
    "OFF": {"delta_rank": 0.0, "min_hold_days": 0},
    "ON": {"delta_rank": 0.10, "min_hold_days": 9},
}
SEALED = {
    "S1": "ADX_14D + OBV_SLOPE_10D + SHARPE_RATIO_20D + SLOPE_20D",
    "S2": "ADX_14D + OBV_SLOPE_10D + PRICE_POSITION_120D + SHARPE_RATIO_20D + SLOPE_20D",
}


def _compute_mdd(equity: np.ndarray) -> float:
    """Compute max drawdown from an equity array segment."""
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


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_runs = len(FREQS) * len(EXP4_CONFIGS) * len(SEALED)

    print("=" * 80)
    print("FREQ x Exp4 Ablation Study")
    print(f"  FREQ: {FREQS}")
    print(f"  Exp4: OFF / ON (dr=0.10, mh=9)")
    print(f"  Strategies: S1 (4F), S2 (5F)")
    print(f"  Matrix: {len(FREQS)} x {len(EXP4_CONFIGS)} x {len(SEALED)} = {total_runs} runs")
    print("=" * 80)

    # ── 1. Load config ─────────────────────────────────────────────────────
    config_path = ROOT / "configs" / "combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    backtest_config = config.get("backtest", {})
    POS_SIZE = 2
    LOOKBACK = 252
    INITIAL_CAPITAL = float(backtest_config.get("initial_capital", 1_000_000))
    COMMISSION_RATE = float(backtest_config.get("commission_rate", 0.0002))

    exec_model = load_execution_model(config)
    USE_T1_OPEN = exec_model.is_t1_open

    cost_model = load_cost_model(config)
    tier = cost_model.active_tier
    qdii_set = set(FrozenETFPool().qdii_codes)

    risk_config = backtest_config.get("risk_control", {})
    trailing_stop_pct = risk_config.get("trailing_stop_pct", 0.0)
    stop_on_rebalance_only = risk_config.get("stop_check_on_rebalance_only", True)
    leverage_cap = risk_config.get("leverage_cap", 1.0)
    profit_ladders = risk_config.get("profit_ladders", [])

    print(f"  Execution: {exec_model.mode}")
    print(f"  Cost: tier={cost_model.tier}, A={tier.a_share*10000:.0f}bp, QDII={tier.qdii*10000:.0f}bp")
    print(f"  POS_SIZE={POS_SIZE}, LOOKBACK={LOOKBACK}, CAPITAL={INITIAL_CAPITAL:,.0f}")

    # ── 2. Load data ───────────────────────────────────────────────────────
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

    # ── 3. Compute factors (cached) ────────────────────────────────────────
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

    # ── 4. Timing + regime gate ────────────────────────────────────────────
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
    timing_arr = (timing_arr.astype(np.float64) * gate_arr.astype(np.float64))

    print(f"  Data: {T} days x {N} ETFs x {len(factor_names)} factors")

    # ── 5. Train/holdout split index ───────────────────────────────────────
    training_end_date = pd.Timestamp(
        config["data"].get("training_end_date", "2025-04-30")
    )
    train_end_idx = 0
    for i, d in enumerate(dates):
        if pd.Timestamp(d) <= training_end_date:
            train_end_idx = i
    print(f"  Train:   idx 0..{train_end_idx}  ({dates[0]} ~ {dates[train_end_idx]})")
    print(f"  Holdout: idx {train_end_idx+1}..{T-1}  ({dates[train_end_idx+1]} ~ {dates[T-1]})")

    # ── 6. Resolve sealed strategy factor indices ──────────────────────────
    factor_index_map = {name: idx for idx, name in enumerate(factor_names)}
    sealed_indices: dict[str, list[int]] = {}
    for name, combo_str in SEALED.items():
        factors = [f.strip() for f in combo_str.split("+")]
        try:
            indices = [factor_index_map[f] for f in factors]
            sealed_indices[name] = indices
            print(f"  {name}: {combo_str} -> indices {indices}")
        except KeyError as e:
            print(f"  WARNING: {name}: factor {e} not in factor library, skipping")

    # ── 7. Run ablation matrix ─────────────────────────────────────────────
    print(f"\nRunning {total_runs} VEC backtests...")
    results = []
    count = 0
    start_idx = LOOKBACK  # VEC equity curve starts from this index

    for strategy_name, f_indices in sealed_indices.items():
        for freq in FREQS:
            for exp4_label, exp4_params in EXP4_CONFIGS.items():
                count += 1
                dr = exp4_params["delta_rank"]
                mh = exp4_params["min_hold_days"]

                tag = f"[{count}/{total_runs}] {strategy_name} F={freq:>2d} Exp4={exp4_label}"
                try:
                    eq, ret, wr, pf, trades, _, risk = run_vec_backtest(
                        factors_3d,
                        close_prices,
                        open_prices,
                        high_prices,
                        low_prices,
                        timing_arr,
                        f_indices,
                        freq=freq,
                        pos_size=POS_SIZE,
                        initial_capital=INITIAL_CAPITAL,
                        commission_rate=COMMISSION_RATE,
                        lookback=LOOKBACK,
                        cost_arr=COST_ARR,
                        trailing_stop_pct=trailing_stop_pct,
                        stop_on_rebalance_only=stop_on_rebalance_only,
                        leverage_cap=leverage_cap,
                        profit_ladders=profit_ladders,
                        use_t1_open=USE_T1_OPEN,
                        delta_rank=dr,
                        min_hold_days=mh,
                    )

                    # ── Split equity curve into train/holdout ──────────
                    train_eq = eq[start_idx : train_end_idx + 1]
                    holdout_eq = eq[train_end_idx:]  # overlap 1 point for return calc

                    train_m = compute_aligned_metrics(train_eq, start_idx=0)
                    holdout_m = compute_aligned_metrics(holdout_eq, start_idx=0)
                    holdout_mdd = _compute_mdd(holdout_eq)

                    results.append(
                        {
                            "strategy": strategy_name,
                            "freq": freq,
                            "exp4": exp4_label,
                            "dr": dr,
                            "mh": mh,
                            "full_return": ret,
                            "full_sharpe": risk["sharpe_ratio"],
                            "full_mdd": risk["max_drawdown"],
                            "train_return": train_m["aligned_return"],
                            "train_sharpe": train_m["aligned_sharpe"],
                            "holdout_return": holdout_m["aligned_return"],
                            "holdout_sharpe": holdout_m["aligned_sharpe"],
                            "holdout_mdd": holdout_mdd,
                            "turnover": risk["turnover_ann"],
                            "cost_drag": risk["cost_drag"],
                            "trades": trades,
                            "win_rate": wr,
                        }
                    )
                    h_ret = holdout_m["aligned_return"]
                    to = risk["turnover_ann"]
                    print(
                        f"  {tag}"
                        f"  ret={ret*100:+6.1f}%"
                        f"  holdout={h_ret*100:+6.1f}%"
                        f"  TO={to:5.1f}x"
                        f"  cd={risk['cost_drag']*100:5.1f}%"
                    )
                except Exception as e:
                    print(f"  {tag}  ERROR: {e}")

    if not results:
        print("\nNo results! Check that OBV_SLOPE_10D is in the factor library.")
        return

    df = pd.DataFrame(results)

    # ── 8. Decision table ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("DECISION TABLE")
    print("=" * 80)

    for strat in sorted(df["strategy"].unique()):
        ds = df[df["strategy"] == strat].copy()
        print(f"\n  {strat}: {SEALED.get(strat, '')}")
        header = (
            f"  {'FREQ':>4}"
            f" | {'HO_Ret%':>8}{'HO_Ret%':>9}"
            f" | {'HO_Shrp':>8}{'HO_Shrp':>9}"
            f" | {'HO_MDD%':>8}{'HO_MDD%':>9}"
            f" | {'TO(x)':>7}{'TO(x)':>8}"
            f" | {'CstDrg%':>8}{'CstDrg%':>9}"
            f" | {'Trades':>7}{'Trades':>8}"
        )
        sub_hdr = (
            f"  {'':>4}"
            f" | {'OFF':>8}{'ON':>9}"
            f" | {'OFF':>8}{'ON':>9}"
            f" | {'OFF':>8}{'ON':>9}"
            f" | {'OFF':>7}{'ON':>8}"
            f" | {'OFF':>8}{'ON':>9}"
            f" | {'OFF':>7}{'ON':>8}"
        )
        print(header)
        print(sub_hdr)
        print("  " + "-" * 112)

        for freq in FREQS:
            off = ds[(ds["freq"] == freq) & (ds["exp4"] == "OFF")]
            on = ds[(ds["freq"] == freq) & (ds["exp4"] == "ON")]
            if off.empty or on.empty:
                continue
            o, n = off.iloc[0], on.iloc[0]
            print(
                f"  {freq:>4}"
                f" | {o['holdout_return']*100:>+7.2f}%{n['holdout_return']*100:>+8.2f}%"
                f" | {o['holdout_sharpe']:>+7.3f}{n['holdout_sharpe']:>+8.3f}"
                f" | {o['holdout_mdd']*100:>7.2f}%{n['holdout_mdd']*100:>8.2f}%"
                f" | {o['turnover']:>6.1f}x{n['turnover']:>7.1f}x"
                f" | {o['cost_drag']*100:>7.1f}%{n['cost_drag']*100:>8.1f}%"
                f" | {o['trades']:>7.0f}{n['trades']:>8.0f}"
            )

    # ── 9. Marginal contribution ───────────────────────────────────────────
    print("\n" + "=" * 80)
    print("MARGINAL CONTRIBUTION ANALYSIS")
    print("=" * 80)
    print("  D_Exp4 = metric(freq, ON) - metric(freq, OFF)      [Exp4 effect at given freq]")
    print("  D_Freq = metric(freq, OFF) - metric(3, OFF)         [freq effect vs baseline]")

    for strat in sorted(df["strategy"].unique()):
        ds = df[df["strategy"] == strat]
        base = ds[(ds["freq"] == 3) & (ds["exp4"] == "OFF")]
        if base.empty:
            continue
        base_row = base.iloc[0]

        print(f"\n  {strat}:")
        print(
            f"  {'FREQ':>4}"
            f" | {'D_E4_Ret%':>10}{'D_E4_TO':>9}{'D_E4_Shrp':>11}"
            f" | {'D_Fq_Ret%':>10}{'D_Fq_TO':>9}"
        )
        print("  " + "-" * 58)

        d_exp4_rets = []
        d_freq_rets = []

        for freq in FREQS:
            off = ds[(ds["freq"] == freq) & (ds["exp4"] == "OFF")]
            on = ds[(ds["freq"] == freq) & (ds["exp4"] == "ON")]
            if off.empty or on.empty:
                continue
            o, n = off.iloc[0], on.iloc[0]

            d_e4_ret = n["holdout_return"] - o["holdout_return"]
            d_e4_to = n["turnover"] - o["turnover"]
            d_e4_shrp = n["holdout_sharpe"] - o["holdout_sharpe"]
            d_fq_ret = o["holdout_return"] - base_row["holdout_return"]
            d_fq_to = o["turnover"] - base_row["turnover"]

            d_exp4_rets.append(d_e4_ret)
            d_freq_rets.append(d_fq_ret)

            print(
                f"  {freq:>4}"
                f" | {d_e4_ret*100:>+9.2f}%{d_e4_to:>+8.1f}x{d_e4_shrp:>+10.3f}"
                f" | {d_fq_ret*100:>+9.2f}%{d_fq_to:>+8.1f}x"
            )

        # Summary
        if d_exp4_rets:
            avg_e4 = np.mean(d_exp4_rets)
            freq_spread = max(d_freq_rets) - min(d_freq_rets) if d_freq_rets else 0
            print(f"\n  Avg |D_Exp4| = {abs(avg_e4)*100:.2f}pp")
            print(f"  Freq spread  = {freq_spread*100:.2f}pp (max-min of D_Freq)")

    # ── 10. Decision recommendation ────────────────────────────────────────
    print("\n" + "=" * 80)
    print("DECISION RECOMMENDATION")
    print("=" * 80)

    for strat in sorted(df["strategy"].unique()):
        ds = df[df["strategy"] == strat]
        base = ds[(ds["freq"] == 3) & (ds["exp4"] == "OFF")]
        if base.empty:
            continue

        # Best by holdout return (prefer TO < 20x)
        viable = ds[ds["turnover"] < 20.0]
        if viable.empty:
            viable = ds

        best = viable.loc[viable["holdout_return"].idxmax()]

        print(f"\n  {strat}:")
        print(f"    Best (holdout return, TO<20x): FREQ={int(best['freq'])}, Exp4={best['exp4']}")
        print(
            f"    Holdout: {best['holdout_return']*100:+.2f}%"
            f"  Sharpe={best['holdout_sharpe']:+.3f}"
            f"  MDD={best['holdout_mdd']*100:.2f}%"
        )
        print(
            f"    Turnover: {best['turnover']:.1f}x"
            f"  Cost Drag: {best['cost_drag']*100:.1f}%"
            f"  Trades: {int(best['trades'])}"
        )

        # Dominance analysis
        d_exp4_rets = []
        d_freq_rets_all = []
        for freq in FREQS:
            off = ds[(ds["freq"] == freq) & (ds["exp4"] == "OFF")]
            on = ds[(ds["freq"] == freq) & (ds["exp4"] == "ON")]
            if not off.empty and not on.empty:
                d_exp4_rets.append(
                    on.iloc[0]["holdout_return"] - off.iloc[0]["holdout_return"]
                )
            if not off.empty:
                d_freq_rets_all.append(off.iloc[0]["holdout_return"])

        if d_exp4_rets and len(d_freq_rets_all) > 1:
            avg_exp4 = abs(np.mean(d_exp4_rets))
            freq_spread = max(d_freq_rets_all) - min(d_freq_rets_all)

            print(f"    Avg |D_Exp4|: {avg_exp4*100:.2f}pp")
            print(f"    Freq spread:  {freq_spread*100:.2f}pp")

            if avg_exp4 > freq_spread * 0.7:
                verdict = "Exp4 DOMINATES -> keep FREQ=3 + add Exp4 (v4.x path)"
            elif freq_spread > avg_exp4 * 2:
                verdict = "FREQ DOMINATES -> increase freq, skip Exp4 (v5.0 path)"
            else:
                verdict = "SYNERGISTIC -> combine FREQ change + Exp4 (combined path)"
            print(f"    >> {verdict}")

    # ── 11. Save ───────────────────────────────────────────────────────────
    out_dir = ROOT / "results" / f"freq_x_exp4_scan_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "decision_table.csv", index=False)
    print(f"\nResults saved to: {out_dir}")
    print(f"  decision_table.csv: {len(df)} rows")


if __name__ == "__main__":
    main()
