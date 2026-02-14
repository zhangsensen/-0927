#!/usr/bin/env python3
"""
A/B Verification: Pool diversity constraint under F5+Exp4 (production execution).

Takes top-N combos from VEC full-space results, runs VEC with Exp4 hysteresis
twice (constraint OFF vs ON), then compares holdout metrics.

Usage:
  uv run python scripts/analysis/ab_pool_constraint_verify.py --top-n 200
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from batch_vec_backtest import run_vec_backtest
from etf_strategy.core.cost_model import build_cost_array, load_cost_model
from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.etf_pool_mapper import load_pool_mapping, build_pool_array
from etf_strategy.core.execution_model import load_execution_model
from etf_strategy.core.factor_cache import FactorCache
from etf_strategy.core.frozen_params import FrozenETFPool
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import shift_timing_signal
from etf_strategy.regime_gate import compute_regime_gate_arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--extended-k", type=int, default=10,
                        help="Search depth for pool diversify (default: 10)")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs/combo_wfo_config.yaml"))
    parser.add_argument("--vec-results", type=str, default=None,
                        help="VEC full-space results parquet. Auto-detect if omitted.")
    args = parser.parse_args()

    # ── Load config ──
    with open(args.config) as f:
        config = yaml.safe_load(f)
    backtest_config = config.get("backtest", {})

    # ── Find VEC results ──
    if args.vec_results:
        vec_path = Path(args.vec_results)
    else:
        results_dir = ROOT / "results"
        candidates = sorted(results_dir.glob("vec_from_wfo_*/full_space_results.parquet"), reverse=True)
        if not candidates:
            # Also try batch vec results
            candidates = sorted(results_dir.glob("vec_full_backtest_*/vec_all_combos.parquet"), reverse=True)
        if not candidates:
            print("ERROR: No VEC results found")
            sys.exit(1)
        vec_path = candidates[0]

    print(f"Loading VEC results: {vec_path}")
    vec_df = pd.read_parquet(vec_path)

    # Select top-N by training Calmar
    top_df = vec_df.nlargest(args.top_n, "vec_calmar_ratio")
    combos = top_df["combo"].tolist()
    print(f"Selected top {len(combos)} combos by training Calmar")

    # ── Load data ──
    print("\nLoading data...")

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

    factor_index_map = {name: idx for idx, name in enumerate(factor_names)}

    # ── Timing + Regime ──
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

    # ── Cost array ──
    cm = load_cost_model(config)
    cost_arr = build_cost_array(cm, list(etf_codes), qdii_set)

    # ── Pool mapping ──
    pool_mapping = load_pool_mapping(ROOT / "configs" / "etf_pools.yaml")
    pool_ids = build_pool_array(etf_codes, pool_mapping)
    EXTENDED_K = args.extended_k

    n_mapped = int(np.sum(pool_ids >= 0))
    n_pools = len(set(int(p) for p in pool_ids if p >= 0))
    print(f"  Pool mapping: {n_mapped}/{N} ETFs mapped, {n_pools} pools")

    # ── Exec params ──
    FREQ = backtest_config.get("freq", 5)

    hyst = backtest_config.get("hysteresis", {})
    DELTA_RANK = hyst.get("delta_rank", 0.10)
    MIN_HOLD_DAYS = hyst.get("min_hold_days", 9)

    # ── Holdout boundary ──
    training_end_date = config["data"].get("training_end_date", "2025-04-30")
    training_end_ts = pd.Timestamp(training_end_date)
    ho_start_idx = None
    for i, d in enumerate(dates):
        if pd.Timestamp(d) > training_end_ts:
            ho_start_idx = i
            break
    if ho_start_idx is None:
        ho_start_idx = T

    print(f"  Data: {T} days x {N} ETFs x {len(factor_names)} factors")
    print(f"  Execution: FREQ={FREQ}, POS={POS_SIZE}, T1_OPEN={USE_T1_OPEN}")
    print(f"  Exp4: delta_rank={DELTA_RANK}, min_hold_days={MIN_HOLD_DAYS}")
    print(f"  Pool constraint: extended_k={EXTENDED_K}")
    print(f"  Holdout starts at idx {ho_start_idx} ({dates[ho_start_idx] if ho_start_idx < T else 'N/A'})")

    # ── Run A/B: OFF vs ON ──
    def _run_combo(combo_str, pool_ext_k):
        factors = [f.strip() for f in combo_str.split("+")]
        f_indices = [factor_index_map[f] for f in factors]
        eq, ret, _, _, trades, _, risk = run_vec_backtest(
            factors_3d, close_prices, open_prices, high_prices, low_prices,
            timing_arr, f_indices,
            freq=FREQ, pos_size=POS_SIZE, initial_capital=INITIAL_CAPITAL,
            commission_rate=COMMISSION_RATE, lookback=LOOKBACK,
            cost_arr=cost_arr,
            trailing_stop_pct=trailing_stop_pct,
            stop_on_rebalance_only=stop_on_rebalance_only,
            leverage_cap=leverage_cap,
            profit_ladders=profit_ladders,
            use_t1_open=USE_T1_OPEN,
            delta_rank=DELTA_RANK,
            min_hold_days=MIN_HOLD_DAYS,
            pool_ids=pool_ids if pool_ext_k > 0 else None,
            pool_constraint_extended_k=pool_ext_k,
        )
        # Holdout metrics
        if ho_start_idx < T and ho_start_idx < len(eq):
            ho_eq = eq[ho_start_idx:]
            if len(ho_eq) > 1 and ho_eq[0] > 0:
                ho_ret = ho_eq[-1] / ho_eq[0] - 1
                running_max = np.maximum.accumulate(ho_eq)
                dd = (ho_eq - running_max) / np.where(running_max > 0, running_max, 1.0)
                ho_mdd = abs(dd.min())
                ho_calmar = ho_ret / ho_mdd if ho_mdd > 0 else 0
            else:
                ho_ret = ho_mdd = ho_calmar = 0
        else:
            ho_ret = ho_mdd = ho_calmar = 0

        return {
            "combo": combo_str,
            "full_return": ret,
            "full_mdd": risk.get("max_drawdown", 0),
            "full_sharpe": risk.get("sharpe_ratio", 0),
            "trades": trades,
            "ho_return": ho_ret,
            "ho_mdd": ho_mdd,
            "ho_calmar": ho_calmar,
        }

    print(f"\nRunning {len(combos)} combos × 2 (OFF/ON)...")
    results_off = []
    results_on = []

    for i, combo_str in enumerate(combos):
        try:
            r_off = _run_combo(combo_str, 0)
            r_on = _run_combo(combo_str, EXTENDED_K)
            results_off.append(r_off)
            results_on.append(r_on)
        except Exception as e:
            print(f"  ERROR on {combo_str}: {e}")

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(combos)} done")

    df_off = pd.DataFrame(results_off)
    df_on = pd.DataFrame(results_on)

    # ── A/B Comparison ──
    print(f"\n{'='*70}")
    print(f"A/B: POOL DIVERSITY CONSTRAINT (F5+Exp4)")
    print(f"Top {args.top_n} combos, extended_k={EXTENDED_K}")
    print(f"{'='*70}")
    print(f"  A (OFF): {len(df_off)} combos")
    print(f"  B (ON):  {len(df_on)} combos")

    if len(df_off) == 0 or len(df_on) == 0:
        print("  WARNING: Empty results — cannot compare")
        return

    for metric, label in [
        ("full_return", "Full Return"),
        ("full_mdd", "Full MDD"),
        ("full_sharpe", "Full Sharpe"),
        ("ho_return", "Holdout Return"),
        ("ho_mdd", "Holdout MDD"),
        ("ho_calmar", "Holdout Calmar"),
        ("trades", "Trades"),
    ]:
        a_med = df_off[metric].median()
        b_med = df_on[metric].median()
        a_10 = df_off[metric].quantile(0.10)
        b_10 = df_on[metric].quantile(0.10)
        delta_med = b_med - a_med

        print(f"\n  {label}:")
        print(f"    OFF  median={a_med:.4f}  p10={a_10:.4f}")
        print(f"    ON   median={b_med:.4f}  p10={b_10:.4f}")
        print(f"    Δmed={delta_med:+.4f}")

    # ── Judgment ──
    ho_med_off = df_off["ho_return"].median()
    ho_med_on = df_on["ho_return"].median()
    ho_10_off = df_off["ho_return"].quantile(0.10)
    ho_10_on = df_on["ho_return"].quantile(0.10)

    # Percentage of combos where ON improved HO return
    paired = pd.DataFrame({
        "off": df_off["ho_return"].values,
        "on": df_on["ho_return"].values,
    })
    pct_improved = (paired["on"] > paired["off"]).mean() * 100

    print(f"\n{'='*70}")
    print(f"JUDGMENT CRITERIA:")
    print(f"  1. HO median improvement ≥ +1pp: {ho_med_on - ho_med_off:+.4f} → {'PASS' if ho_med_on - ho_med_off >= 0.01 else 'FAIL'}")
    print(f"  2. HO 10th pctl not worse:       {ho_10_on:.4f} vs {ho_10_off:.4f} → {'PASS' if ho_10_on >= ho_10_off - 0.005 else 'FAIL'}")
    print(f"  3. % combos improved:             {pct_improved:.1f}%")
    print(f"{'='*70}")

    # Save
    output_dir = ROOT / "results" / "ab_pool_constraint"
    output_dir.mkdir(parents=True, exist_ok=True)
    df_off.to_parquet(output_dir / "pool_off.parquet", index=False)
    df_on.to_parquet(output_dir / "pool_on.parquet", index=False)
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
