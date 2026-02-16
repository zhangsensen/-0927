#!/usr/bin/env python3
"""
Exp4: Hysteresis + Min Hold — 3×3 Grid Search

Grid:
  delta_rank:    0.10, 0.15, 0.20
  min_hold_days: 6,    9,    12

Evaluation: T1_OPEN + med cost (SPLIT_MARKET)
Baseline:   delta_rank=0, min_hold_days=0 (current production behavior)

Acceptance criteria:
  - turnover ≤ 15-20x (down from 27-31x)
  - med candidates > 1
  - cost_drag significantly lower
  - holdout return ≥ 80% of baseline
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
from tqdm import tqdm

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.factor_cache import FactorCache
from etf_strategy.core.frozen_params import load_frozen_config, FrozenETFPool
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import shift_timing_signal
from etf_strategy.regime_gate import compute_regime_gate_arr, gate_stats
from etf_strategy.core.execution_model import load_execution_model
from etf_strategy.core.cost_model import load_cost_model, build_cost_array
from batch_vec_backtest import run_vec_backtest


# ── Grid parameters ──────────────────────────────────────────────
DELTA_RANK_GRID = [0.10, 0.15, 0.20]
MIN_HOLD_GRID = [6, 9, 12]


def load_data_and_config():
    """Load data, config, and prepare all shared arrays (same as batch_vec_backtest)."""
    config_path = ROOT / "configs/combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    frozen = load_frozen_config(config, config_path=str(config_path))
    print(f"  Frozen params: v{frozen.version}")

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
        ohlcv=ohlcv,
        config=config,
        data_dir=loader.data_dir,
    )
    factor_names = cached["factor_names"]
    dates = cached["dates"]
    etf_codes = cached["etf_codes"]
    factors_3d = cached["factors_3d"]
    T = len(dates)
    N = len(etf_codes)

    # Cost array
    COST_ARR = build_cost_array(cost_model, list(etf_codes), qdii_set)

    # Price arrays
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

    print(f"  Data: {T} days x {N} ETFs x {len(factor_names)} factors")

    return {
        "config": config,
        "factors_3d": factors_3d,
        "factor_names": factor_names,
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
    }


def load_wfo_combos():
    """Load combo strings from latest WFO output."""
    wfo_dirs = sorted(
        d for d in (ROOT / "results").glob("run_*")
        if d.is_dir() and not d.is_symlink()
    )
    if not wfo_dirs:
        raise FileNotFoundError("No WFO output found in results/run_*")
    latest = wfo_dirs[-1]
    combos_path = latest / "top100_by_ic.parquet"
    if not combos_path.exists():
        combos_path = latest / "all_combos.parquet"
    if not combos_path.exists():
        # Try full_combo_results.csv
        combos_path = latest / "full_combo_results.csv"
    if not combos_path.exists():
        raise FileNotFoundError(f"No combo file in {latest}")
    if combos_path.suffix == ".csv":
        df = pd.read_csv(combos_path)
    else:
        df = pd.read_parquet(combos_path)
    print(f"  WFO combos: {len(df)} from {latest.name}")
    return df


def run_grid_point(data, combos_df, delta_rank, min_hold_days):
    """Run VEC backtest for all combos with given hysteresis params."""
    factor_names = data["factor_names"]
    factor_index_map = {name: idx for idx, name in enumerate(factor_names)}

    results = []
    for combo_str in combos_df["combo"].tolist():
        factors_in_combo = [f.strip() for f in combo_str.split(" + ")]
        try:
            factor_indices = [factor_index_map[f] for f in factors_in_combo]
        except KeyError:
            continue

        try:
            _, ret, wr, pf, trades, _, risk = run_vec_backtest(
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
                # ✅ Exp4 params
                delta_rank=delta_rank,
                min_hold_days=min_hold_days,
            )
            results.append({
                "combo": combo_str,
                "return": ret,
                "sharpe": risk["sharpe_ratio"],
                "max_dd": risk["max_drawdown"],
                "trades": trades,
                "turnover_ann": risk["turnover_ann"],
                "cost_drag": risk["cost_drag"],
                "win_rate": wr,
                "profit_factor": pf,
            })
        except Exception as e:
            print(f"  [WARN] {combo_str[:30]}... failed: {e}")
            continue

    return pd.DataFrame(results)


def main():
    print("=" * 80)
    print("Exp4: Hysteresis + Min Hold — Grid Search")
    print("=" * 80)

    # 1. Load data
    print("\n[1/3] Loading data...")
    data = load_data_and_config()

    # 2. Load WFO combos
    print("\n[2/3] Loading WFO combos...")
    combos_df = load_wfo_combos()

    # 3. Run grid
    print("\n[3/3] Running grid search...")

    # Build grid: baseline + 3x3
    grid_points = [(0.0, 0)]  # baseline
    for dr in DELTA_RANK_GRID:
        for mh in MIN_HOLD_GRID:
            grid_points.append((dr, mh))

    grid_results = []

    for i, (dr, mh) in enumerate(grid_points):
        label = "baseline" if dr == 0.0 and mh == 0 else f"dr={dr:.2f}_mh={mh}"
        print(f"\n  [{i+1}/{len(grid_points)}] {label}")

        df = run_grid_point(data, combos_df, dr, mh)

        if len(df) == 0:
            print(f"    No results!")
            continue

        # Aggregate metrics
        med_return = df["return"].median()
        med_sharpe = df["sharpe"].median()
        med_turnover = df["turnover_ann"].median()
        med_cost_drag = df["cost_drag"].median()
        avg_trades = df["trades"].mean()
        positive_count = (df["return"] > 0).sum()

        # Count "survivors" at med cost: return > 0 AND sharpe > 0
        survivors = ((df["return"] > 0) & (df["sharpe"] > 0)).sum()

        grid_results.append({
            "delta_rank": dr,
            "min_hold_days": mh,
            "label": label,
            "n_combos": len(df),
            "median_return": med_return,
            "median_sharpe": med_sharpe,
            "median_turnover": med_turnover,
            "median_cost_drag": med_cost_drag,
            "avg_trades": avg_trades,
            "positive_count": positive_count,
            "survivors": survivors,
        })

        print(f"    return={med_return*100:+.1f}%  sharpe={med_sharpe:.2f}  "
              f"turnover={med_turnover:.1f}x  cost_drag={med_cost_drag:.1%}  "
              f"survivors={survivors}/{len(df)}")

    # 4. Summary
    print("\n" + "=" * 80)
    print("GRID SEARCH RESULTS")
    print("=" * 80)

    df_grid = pd.DataFrame(grid_results)

    # Compare to baseline
    baseline = df_grid[df_grid["label"] == "baseline"].iloc[0] if "baseline" in df_grid["label"].values else None

    print(f"\n{'Label':<18} {'Return':>8} {'Sharpe':>8} {'Turnover':>10} "
          f"{'CostDrag':>10} {'Trades':>8} {'Pos':>5} {'Surv':>5} {'Ret%BL':>8}")
    print("-" * 95)

    for _, row in df_grid.iterrows():
        ret_pct = row["median_return"] * 100
        if baseline is not None and baseline["median_return"] != 0:
            ret_vs_bl = row["median_return"] / baseline["median_return"] * 100
        else:
            ret_vs_bl = 100.0

        print(f"{row['label']:<18} {ret_pct:>+7.1f}% {row['median_sharpe']:>8.2f} "
              f"{row['median_turnover']:>9.1f}x {row['median_cost_drag']:>9.1%} "
              f"{row['avg_trades']:>7.0f} {row['positive_count']:>5} "
              f"{row['survivors']:>5} {ret_vs_bl:>7.0f}%")

    # 5. Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / "results" / f"exp4_grid_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    df_grid.to_csv(output_dir / "grid_summary.csv", index=False)
    print(f"\nResults saved to: {output_dir}")

    # 6. Recommendation
    if baseline is not None:
        print("\n" + "=" * 80)
        print("RECOMMENDATION")
        print("=" * 80)

        bl_turnover = baseline["median_turnover"]
        best = None
        for _, row in df_grid.iterrows():
            if row["label"] == "baseline":
                continue
            # Must reduce turnover significantly
            if row["median_turnover"] > bl_turnover * 0.8:
                continue
            # Return must be >= 80% of baseline
            if baseline["median_return"] > 0 and row["median_return"] < baseline["median_return"] * 0.8:
                continue
            if best is None or row["survivors"] > best["survivors"]:
                best = row
            elif row["survivors"] == best["survivors"] and row["median_sharpe"] > best["median_sharpe"]:
                best = row

        if best is not None:
            print(f"  Best: {best['label']}")
            print(f"  Turnover: {bl_turnover:.1f}x -> {best['median_turnover']:.1f}x "
                  f"({(1 - best['median_turnover']/bl_turnover)*100:.0f}% reduction)")
            print(f"  Survivors: {baseline['survivors']} -> {best['survivors']}")
            print(f"  Sharpe: {baseline['median_sharpe']:.2f} -> {best['median_sharpe']:.2f}")
        else:
            print("  No grid point meets all criteria. Consider relaxing thresholds.")


if __name__ == "__main__":
    main()
