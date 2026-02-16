#!/usr/bin/env python3
"""
Exp4: Focused S1/S2 Hysteresis Comparison

Tests sealed strategies S1 (4F) and S2 (5F) against baseline and all 9 grid points.
Reports: return, sharpe, MDD, turnover, cost_drag — split by training and holdout.

Training:  2020-01-01 ~ 2025-04-30
Holdout:   2025-05-01 ~ 2025-12-12
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import yaml
import numpy as np
import pandas as pd
from datetime import datetime

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.factor_cache import FactorCache
from etf_strategy.core.frozen_params import load_frozen_config, FrozenETFPool
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import shift_timing_signal, generate_rebalance_schedule
from etf_strategy.regime_gate import compute_regime_gate_arr
from etf_strategy.core.execution_model import load_execution_model
from etf_strategy.core.cost_model import load_cost_model, build_cost_array
from batch_vec_backtest import run_vec_backtest

# ── Sealed strategies ────────────────────────────────────────────
S1_COMBO = "ADX_14D + OBV_SLOPE_10D + SHARPE_RATIO_20D + SLOPE_20D"
S2_COMBO = "ADX_14D + OBV_SLOPE_10D + PRICE_POSITION_120D + SHARPE_RATIO_20D + SLOPE_20D"

STRATEGIES = {"S1(4F)": S1_COMBO, "S2(5F)": S2_COMBO}

# ── Grid parameters ─────────────────────────────────────────────
DELTA_RANK_GRID = [0.10, 0.15, 0.20]
MIN_HOLD_GRID = [6, 9, 12]


def compute_period_metrics(equity_curve, start_idx, end_idx, initial_capital):
    """Compute return, sharpe, MDD for a sub-period of the equity curve."""
    if end_idx <= start_idx:
        return {"return": 0.0, "sharpe": 0.0, "max_dd": 0.0}

    eq = equity_curve[start_idx:end_idx + 1]
    period_return = (eq[-1] / eq[0]) - 1.0 if eq[0] > 0 else 0.0

    # Daily returns
    daily_ret = np.diff(eq) / eq[:-1]
    daily_ret = daily_ret[np.isfinite(daily_ret)]

    if len(daily_ret) > 1:
        sharpe = (np.mean(daily_ret) / np.std(daily_ret, ddof=1)) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Max drawdown
    running_max = np.maximum.accumulate(eq)
    drawdowns = (eq - running_max) / running_max
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    return {"return": period_return, "sharpe": sharpe, "max_dd": max_dd}


def load_data():
    """Load data, config, and prepare all shared arrays."""
    config_path = ROOT / "configs/combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    frozen = load_frozen_config(config, config_path=str(config_path))
    backtest_config = config.get("backtest", {})

    FREQ = backtest_config["freq"]
    POS_SIZE = backtest_config["pos_size"]
    LOOKBACK = backtest_config["lookback"]
    INITIAL_CAPITAL = float(backtest_config["initial_capital"])
    COMMISSION_RATE = float(backtest_config["commission_rate"])

    exec_model = load_execution_model(config)
    USE_T1_OPEN = exec_model.is_t1_open

    cost_model = load_cost_model(config)
    qdii_set = set(FrozenETFPool().qdii_codes)
    tier = cost_model.active_tier

    print(f"  FREQ={FREQ}, POS={POS_SIZE}, LOOKBACK={LOOKBACK}")
    print(f"  Execution: {exec_model.mode}")
    print(f"  Cost: {cost_model.mode}/{cost_model.tier} "
          f"(A={tier.a_share*10000:.0f}bp, QDII={tier.qdii*10000:.0f}bp)")

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

    # Compute factors
    factor_cache = FactorCache(
        cache_dir=Path(config["data"].get("cache_dir") or ".cache")
    )
    cached = factor_cache.get_or_compute(
        ohlcv=ohlcv, config=config, data_dir=loader.data_dir,
    )
    factor_names = cached["factor_names"]
    dates = cached["dates"]
    etf_codes = cached["etf_codes"]
    factors_3d = cached["factors_3d"]
    T = len(dates)
    N = len(etf_codes)

    COST_ARR = build_cost_array(cost_model, list(etf_codes), qdii_set)

    close_prices = ohlcv["close"][etf_codes].ffill().fillna(1.0).values
    open_prices = ohlcv["open"][etf_codes].ffill().fillna(1.0).values
    high_prices = ohlcv["high"][etf_codes].ffill().fillna(1.0).values
    low_prices = ohlcv["low"][etf_codes].ffill().fillna(1.0).values

    # Timing
    timing_config = backtest_config.get("timing", {})
    timing_module = LightTimingModule(
        extreme_threshold=timing_config.get("extreme_threshold", -0.1),
        extreme_position=timing_config.get("extreme_position", 0.1),
    )
    timing_arr_raw = (
        timing_module.compute_position_ratios(ohlcv["close"])
        .reindex(dates).fillna(1.0).values
    )
    timing_arr = shift_timing_signal(timing_arr_raw)

    # Regime gate
    gate_arr = compute_regime_gate_arr(
        ohlcv["close"], dates, backtest_config=backtest_config
    )
    timing_arr = (timing_arr.astype(np.float64) * gate_arr.astype(np.float64)).astype(
        np.float64
    )

    # Find training/holdout split index
    training_end = pd.Timestamp(config["data"]["training_end_date"])
    train_end_idx = 0
    for i, d in enumerate(dates):
        if pd.Timestamp(d) <= training_end:
            train_end_idx = i

    print(f"  Data: {T} days x {N} ETFs x {len(factor_names)} factors")
    print(f"  Training: idx 0..{train_end_idx} ({dates[0]}..{dates[train_end_idx]})")
    print(f"  Holdout:  idx {train_end_idx+1}..{T-1} ({dates[train_end_idx+1]}..{dates[T-1]})")

    return {
        "config": config,
        "factors_3d": factors_3d,
        "factor_names": factor_names,
        "dates": dates,
        "close_prices": close_prices,
        "open_prices": open_prices,
        "high_prices": high_prices,
        "low_prices": low_prices,
        "timing_arr": timing_arr,
        "FREQ": FREQ,
        "POS_SIZE": POS_SIZE,
        "LOOKBACK": LOOKBACK,
        "INITIAL_CAPITAL": INITIAL_CAPITAL,
        "COMMISSION_RATE": COMMISSION_RATE,
        "COST_ARR": COST_ARR,
        "USE_T1_OPEN": USE_T1_OPEN,
        "train_end_idx": train_end_idx,
    }


def run_single(data, factor_indices, delta_rank, min_hold_days):
    """Run VEC backtest and return equity_curve + full-period risk metrics."""
    eq, ret, wr, pf, trades, _, risk = run_vec_backtest(
        data["factors_3d"],
        data["close_prices"],
        data["open_prices"],
        data["high_prices"],
        data["low_prices"],
        data["timing_arr"],
        factor_indices,
        freq=data["FREQ"],
        pos_size=data["POS_SIZE"],
        initial_capital=data["INITIAL_CAPITAL"],
        commission_rate=data["COMMISSION_RATE"],
        lookback=data["LOOKBACK"],
        cost_arr=data["COST_ARR"],
        trailing_stop_pct=0.0,
        stop_on_rebalance_only=True,
        use_t1_open=data["USE_T1_OPEN"],
        delta_rank=delta_rank,
        min_hold_days=min_hold_days,
    )
    return eq, ret, wr, pf, trades, risk


def main():
    print("=" * 80)
    print("Exp4: S1/S2 Focused Hysteresis Comparison")
    print("=" * 80)

    # 1. Load data
    print("\n[1/3] Loading data...")
    data = load_data()
    factor_names = data["factor_names"]
    factor_index_map = {name: idx for idx, name in enumerate(factor_names)}

    train_end_idx = data["train_end_idx"]
    lookback = data["LOOKBACK"]
    T = len(data["dates"])
    initial_capital = data["INITIAL_CAPITAL"]

    # Resolve factor indices for S1/S2
    strategy_indices = {}
    for name, combo_str in STRATEGIES.items():
        factors = [f.strip() for f in combo_str.split(" + ")]
        try:
            indices = [factor_index_map[f] for f in factors]
            strategy_indices[name] = indices
            print(f"  {name}: {combo_str} -> indices {indices}")
        except KeyError as e:
            print(f"  ERROR: {name} factor {e} not found!")
            return

    # 2. Build grid
    grid_points = [(0.0, 0)]  # baseline
    for dr in DELTA_RANK_GRID:
        for mh in MIN_HOLD_GRID:
            grid_points.append((dr, mh))

    # 3. Run all combinations
    print(f"\n[2/3] Running {len(STRATEGIES)} strategies x {len(grid_points)} grid points...")

    results = []
    for strat_name, indices in strategy_indices.items():
        for i, (dr, mh) in enumerate(grid_points):
            label = "baseline" if dr == 0.0 and mh == 0 else f"dr={dr:.2f}_mh={mh}"

            eq, ret, wr, pf, trades, risk = run_single(data, indices, dr, mh)

            # Full-period metrics from VEC
            full = {
                "return": ret,
                "sharpe": risk["sharpe_ratio"],
                "max_dd": risk["max_drawdown"],
                "turnover": risk["turnover_ann"],
                "cost_drag": risk["cost_drag"],
                "trades": trades,
                "win_rate": wr,
                "profit_factor": pf,
            }

            # Training period metrics (from equity curve)
            train = compute_period_metrics(eq, lookback, train_end_idx, initial_capital)

            # Holdout period metrics (from equity curve)
            holdout = compute_period_metrics(eq, train_end_idx, T - 1, initial_capital)

            results.append({
                "strategy": strat_name,
                "delta_rank": dr,
                "min_hold_days": mh,
                "label": label,
                # Full period
                "full_return": full["return"],
                "full_sharpe": full["sharpe"],
                "full_mdd": full["max_dd"],
                "full_turnover": full["turnover"],
                "full_cost_drag": full["cost_drag"],
                "full_trades": full["trades"],
                # Training
                "train_return": train["return"],
                "train_sharpe": train["sharpe"],
                "train_mdd": train["max_dd"],
                # Holdout
                "hold_return": holdout["return"],
                "hold_sharpe": holdout["sharpe"],
                "hold_mdd": holdout["max_dd"],
            })

            print(f"  {strat_name} {label:<18} "
                  f"full={ret*100:+6.1f}% train={train['return']*100:+6.1f}% "
                  f"hold={holdout['return']*100:+6.1f}% "
                  f"turn={risk['turnover_ann']:.1f}x cost={risk['cost_drag']:.1%}")

    # 4. Display results
    df = pd.DataFrame(results)

    for strat_name in STRATEGIES:
        sdf = df[df["strategy"] == strat_name].copy()
        baseline = sdf[sdf["label"] == "baseline"].iloc[0]

        print(f"\n{'='*90}")
        print(f"  {strat_name}: {STRATEGIES[strat_name]}")
        print(f"{'='*90}")

        # Header
        print(f"\n{'Label':<18} {'Train':>8} {'T.Shp':>6} {'T.MDD':>7} | "
              f"{'Hold':>8} {'H.Shp':>6} {'H.MDD':>7} | "
              f"{'Turn':>7} {'Cost%':>7} {'Trd':>5}")
        print("-" * 90)

        for _, row in sdf.iterrows():
            print(f"{row['label']:<18} "
                  f"{row['train_return']*100:>+7.1f}% "
                  f"{row['train_sharpe']:>6.2f} "
                  f"{row['train_mdd']*100:>6.1f}% | "
                  f"{row['hold_return']*100:>+7.1f}% "
                  f"{row['hold_sharpe']:>6.2f} "
                  f"{row['hold_mdd']*100:>6.1f}% | "
                  f"{row['full_turnover']:>6.1f}x "
                  f"{row['full_cost_drag']*100:>6.1f}% "
                  f"{row['full_trades']:>5.0f}")

        # Delta vs baseline
        print(f"\n  Delta vs baseline:")
        bl_train = baseline["train_return"]
        bl_hold = baseline["hold_return"]
        bl_turn = baseline["full_turnover"]

        for _, row in sdf.iterrows():
            if row["label"] == "baseline":
                continue
            train_delta = (row["train_return"] - bl_train) * 100
            hold_delta = (row["hold_return"] - bl_hold) * 100
            turn_pct = (1 - row["full_turnover"] / bl_turn) * 100 if bl_turn > 0 else 0
            hold_retain = row["hold_return"] / bl_hold * 100 if bl_hold != 0 else 0

            print(f"    {row['label']:<18} "
                  f"train: {train_delta:+5.1f}pp  "
                  f"hold: {hold_delta:+5.1f}pp ({hold_retain:.0f}% retained)  "
                  f"turnover: {turn_pct:+.0f}%")

    # 5. Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / "results" / f"exp4_s1s2_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "s1_s2_comparison.csv", index=False)
    print(f"\nResults saved to: {output_dir}")

    # 6. Recommendation
    print(f"\n{'='*90}")
    print("RECOMMENDATION")
    print(f"{'='*90}")

    for strat_name in STRATEGIES:
        sdf = df[df["strategy"] == strat_name]
        baseline = sdf[sdf["label"] == "baseline"].iloc[0]
        bl_hold = baseline["hold_return"]
        bl_turn = baseline["full_turnover"]

        best = None
        for _, row in sdf.iterrows():
            if row["label"] == "baseline":
                continue
            # Acceptance: turnover down >=20%, holdout return >=80% of baseline
            if row["full_turnover"] > bl_turn * 0.8:
                continue
            if bl_hold > 0 and row["hold_return"] < bl_hold * 0.8:
                continue
            if best is None or row["hold_sharpe"] > best["hold_sharpe"]:
                best = row

        if best is not None:
            turn_red = (1 - best["full_turnover"] / bl_turn) * 100
            print(f"\n  {strat_name}: {best['label']}")
            print(f"    Turnover: {bl_turn:.1f}x -> {best['full_turnover']:.1f}x ({turn_red:.0f}% reduction)")
            print(f"    Holdout return: {bl_hold*100:+.1f}% -> {best['hold_return']*100:+.1f}%")
            print(f"    Holdout sharpe: {baseline['hold_sharpe']:.2f} -> {best['hold_sharpe']:.2f}")
        else:
            print(f"\n  {strat_name}: No grid point meets all criteria.")


if __name__ == "__main__":
    main()
