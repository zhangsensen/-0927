#!/usr/bin/env python3
"""
Phase 2 Verification: Cross-bucket A/B under F5+Exp4 (production execution).

Takes top-N combos from VEC full-space results, runs VEC with Exp4 hysteresis,
then compares bucket-pass vs bucket-fail groups on holdout metrics.

Usage:
  uv run python scripts/analysis/ab_bucket_exp4_verify.py --top-n 200
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
from etf_strategy.core.execution_model import load_execution_model
from etf_strategy.core.factor_buckets import check_cross_bucket_constraint
from etf_strategy.core.factor_cache import FactorCache
from etf_strategy.core.frozen_params import FrozenETFPool
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import shift_timing_signal
from etf_strategy.regime_gate import compute_regime_gate_arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--config", type=str, default=str(ROOT / "configs/combo_wfo_config.yaml"))
    parser.add_argument("--vec-results", type=str, default=None,
                        help="VEC full-space results (for selecting top-N). Auto-detect if omitted.")
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
            print("ERROR: No VEC results found")
            sys.exit(1)
        vec_path = candidates[0]

    print(f"Loading VEC results: {vec_path}")
    vec_df = pd.read_parquet(vec_path)

    # Select top-N by training Calmar
    top_df = vec_df.nlargest(args.top_n, "vec_calmar_ratio")
    combos = top_df["combo"].tolist()

    # Tag buckets
    bucket_pass = []
    for c in combos:
        factors = [f.strip() for f in c.split("+")]
        ok, _ = check_cross_bucket_constraint(factors, min_buckets=3, max_per_bucket=2)
        bucket_pass.append(ok)

    n_a = sum(not x for x in bucket_pass)
    n_b = sum(bucket_pass)
    print(f"\nTop {args.top_n} combos: A(fail)={n_a}, B(pass)={n_b}")

    # ── Load data (mirror run_v5_validation.py pattern) ──
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
    print(f"  Holdout starts at idx {ho_start_idx} ({dates[ho_start_idx] if ho_start_idx < T else 'N/A'})")

    # ── Run VEC with Exp4 ──
    print(f"\nRunning {len(combos)} VEC backtests with F5+Exp4...")
    results = []

    for i, (combo_str, bp) in enumerate(zip(combos, bucket_pass)):
        factors = [f.strip() for f in combo_str.split("+")]
        f_indices = [factor_index_map[f] for f in factors]

        try:
            eq, ret, wr, pf, trades, _, risk = run_vec_backtest(
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
            )

            # Compute holdout metrics from equity curve
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

            full_mdd = risk.get("max_drawdown", 0)

            results.append({
                "combo": combo_str,
                "bucket_pass": bp,
                "n_factors": len(factors),
                "full_return": ret,
                "full_mdd": full_mdd,
                "full_sharpe": risk.get("sharpe_ratio", 0),
                "trades": trades,
                "ho_return": ho_ret,
                "ho_mdd": ho_mdd,
                "ho_calmar": ho_calmar,
            })

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(combos)} done")

        except Exception as e:
            print(f"  ERROR on {combo_str}: {e}")

    df = pd.DataFrame(results)

    # ── A/B Comparison ──
    a = df[~df["bucket_pass"]]
    b = df[df["bucket_pass"]]

    print(f"\n{'='*70}")
    print(f"A/B PHASE 2: F5+Exp4 (Production Execution)")
    print(f"Top {args.top_n} combos by training Calmar (no-hysteresis VEC)")
    print(f"{'='*70}")
    print(f"  A (bucket-fail): {len(a)}")
    print(f"  B (bucket-pass): {len(b)}")

    if len(a) == 0 or len(b) == 0:
        print("  WARNING: One group is empty — cannot compare")
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
        a_med = a[metric].median()
        b_med = b[metric].median()
        delta = b_med - a_med
        print(f"  {label:<20} A={a_med:>10.4f}  B={b_med:>10.4f}  delta={delta:>+10.4f}")

    # HO return distribution
    print(f"\n  Holdout Return Distribution:")
    for pct_name, pct in [("10%", 0.10), ("25%", 0.25), ("50%", 0.50), ("75%", 0.75)]:
        a_p = a["ho_return"].quantile(pct)
        b_p = b["ho_return"].quantile(pct)
        print(f"    {pct_name}: A={a_p:>+.4f}  B={b_p:>+.4f}  delta={b_p-a_p:>+.4f}")

    a_pos = (a["ho_return"] > 0).mean()
    b_pos = (b["ho_return"] > 0).mean()
    print(f"    Positive rate: A={a_pos:.1%}  B={b_pos:.1%}  delta={b_pos-a_pos:+.1%}")

    # Judgment
    print(f"\n  Quick Judgment:")
    ho_delta = b["ho_return"].median() - a["ho_return"].median()
    ho_mdd_delta = b["ho_mdd"].median() - a["ho_mdd"].median()
    print(f"    HO median return delta: {ho_delta:+.4f} ({'B better' if ho_delta > 0 else 'A better'})")
    print(f"    HO median MDD delta:    {ho_mdd_delta:+.4f} ({'B better (lower MDD)' if ho_mdd_delta < 0 else 'A better'})")
    print(f"    Conclusion: {'Phase 1 CONFIRMED' if ho_delta > 0 else 'Phase 1 REVERSED — investigate!'}")


if __name__ == "__main__":
    main()
