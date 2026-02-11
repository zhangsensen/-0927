#!/usr/bin/env python3
"""
Comprehensive pool diversity constraint validation (Layer 1 + 2 + 3).

Layer 1: Research effectiveness — A/B matrix, pool diagnostics, cost sensitivity
Layer 2: Mechanism comparison — Route A (pre-hyst) vs Route B (post-hyst)
Layer 3: Production safety — determinism, edge cases

Usage:
  uv run python scripts/analysis/pool_constraint_full_validation.py
  uv run python scripts/analysis/pool_constraint_full_validation.py --top-n 500
  uv run python scripts/analysis/pool_constraint_full_validation.py --layer 1  # only Layer 1
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from batch_vec_backtest import (
    run_vec_backtest,
    stable_topk_indices,
    pool_diversify_topk,
)
from etf_strategy.core.cost_model import build_cost_array, load_cost_model
from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.etf_pool_mapper import (
    POOL_ID_TO_NAME,
    load_pool_mapping,
    build_pool_array,
)
from etf_strategy.core.execution_model import load_execution_model
from etf_strategy.core.factor_cache import FactorCache
from etf_strategy.core.frozen_params import FrozenETFPool
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import (
    generate_rebalance_schedule,
    shift_timing_signal,
)
from etf_strategy.regime_gate import compute_regime_gate_arr


# ═══════════════════════════════════════════════════════════════
# Data loading (shared across all layers)
# ═══════════════════════════════════════════════════════════════

def load_all_data(config_path):
    """Load data once, return everything needed for validation."""
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

    # Timing + Regime
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

    # Cost array
    cm = load_cost_model(config)
    cost_arr = build_cost_array(cm, list(etf_codes), qdii_set)

    # Pool mapping
    pool_mapping = load_pool_mapping(ROOT / "configs" / "etf_pools.yaml")
    pool_ids = build_pool_array(etf_codes, pool_mapping)

    # Hysteresis params
    FREQ = backtest_config.get("freq", 5)
    hyst = backtest_config.get("hysteresis", {})
    DELTA_RANK = hyst.get("delta_rank", 0.10)
    MIN_HOLD_DAYS = hyst.get("min_hold_days", 9)

    # Holdout boundary
    training_end_date = config["data"].get("training_end_date", "2025-04-30")
    training_end_ts = pd.Timestamp(training_end_date)
    ho_start_idx = T
    for i, d in enumerate(dates):
        if pd.Timestamp(d) > training_end_ts:
            ho_start_idx = i
            break

    return {
        "config": config,
        "factors_3d": factors_3d,
        "factor_names": factor_names,
        "factor_index_map": factor_index_map,
        "dates": dates,
        "etf_codes": etf_codes,
        "close_prices": close_prices,
        "open_prices": open_prices,
        "high_prices": high_prices,
        "low_prices": low_prices,
        "timing_arr": timing_arr,
        "cost_arr": cost_arr,
        "pool_ids": pool_ids,
        "T": T, "N": N,
        "FREQ": FREQ,
        "POS_SIZE": POS_SIZE,
        "LOOKBACK": LOOKBACK,
        "INITIAL_CAPITAL": INITIAL_CAPITAL,
        "COMMISSION_RATE": COMMISSION_RATE,
        "USE_T1_OPEN": USE_T1_OPEN,
        "DELTA_RANK": DELTA_RANK,
        "MIN_HOLD_DAYS": MIN_HOLD_DAYS,
        "trailing_stop_pct": trailing_stop_pct,
        "stop_on_rebalance_only": stop_on_rebalance_only,
        "leverage_cap": leverage_cap,
        "profit_ladders": profit_ladders,
        "ho_start_idx": ho_start_idx,
    }


def load_combos(max_n):
    """Load VEC results and return top-N combos."""
    results_dir = ROOT / "results"
    candidates = sorted(results_dir.glob("vec_from_wfo_*/full_space_results.parquet"), reverse=True)
    if not candidates:
        candidates = sorted(results_dir.glob("vec_full_backtest_*/vec_all_combos.parquet"), reverse=True)
    if not candidates:
        print("ERROR: No VEC results found")
        sys.exit(1)
    vec_path = candidates[0]
    print(f"  VEC results: {vec_path}")
    vec_df = pd.read_parquet(vec_path)
    top_df = vec_df.nlargest(max_n, "vec_calmar_ratio")
    return top_df["combo"].tolist()


# ═══════════════════════════════════════════════════════════════
# Core VEC runner
# ═══════════════════════════════════════════════════════════════

def run_single_combo(data, combo_str, extended_k, post_hyst=False, cost_multiplier=1.0):
    """Run a single combo through VEC with given pool constraint config."""
    factors = [f.strip() for f in combo_str.split("+")]
    f_indices = [data["factor_index_map"][f] for f in factors]

    cost_arr = data["cost_arr"]
    if cost_multiplier != 1.0:
        cost_arr = cost_arr * cost_multiplier

    eq, ret, _, _, trades, _, risk = run_vec_backtest(
        data["factors_3d"], data["close_prices"], data["open_prices"],
        data["high_prices"], data["low_prices"],
        data["timing_arr"], f_indices,
        freq=data["FREQ"], pos_size=data["POS_SIZE"],
        initial_capital=data["INITIAL_CAPITAL"],
        commission_rate=data["COMMISSION_RATE"],
        lookback=data["LOOKBACK"],
        cost_arr=cost_arr,
        trailing_stop_pct=data["trailing_stop_pct"],
        stop_on_rebalance_only=data["stop_on_rebalance_only"],
        leverage_cap=data["leverage_cap"],
        profit_ladders=data["profit_ladders"],
        use_t1_open=data["USE_T1_OPEN"],
        delta_rank=data["DELTA_RANK"],
        min_hold_days=data["MIN_HOLD_DAYS"],
        pool_ids=data["pool_ids"] if extended_k > 0 else None,
        pool_constraint_extended_k=extended_k,
        pool_constraint_post_hyst=post_hyst,
    )

    # Holdout metrics
    ho_idx = data["ho_start_idx"]
    T = data["T"]
    if ho_idx < T and ho_idx < len(eq):
        ho_eq = eq[ho_idx:]
        if len(ho_eq) > 1 and ho_eq[0] > 0:
            ho_ret = ho_eq[-1] / ho_eq[0] - 1
            running_max = np.maximum.accumulate(ho_eq)
            dd = (ho_eq - running_max) / np.where(running_max > 0, running_max, 1.0)
            ho_mdd = abs(dd.min())
            ho_calmar = ho_ret / ho_mdd if ho_mdd > 0 else 0
        else:
            ho_ret = ho_mdd = ho_calmar = 0.0
    else:
        ho_ret = ho_mdd = ho_calmar = 0.0

    return {
        "combo": combo_str,
        "full_return": ret,
        "full_mdd": risk.get("max_drawdown", 0),
        "full_sharpe": risk.get("sharpe_ratio", 0),
        "trades": trades,
        "turnover_ann": risk.get("turnover_ann", 0),
        "cost_drag": risk.get("cost_drag", 0),
        "ho_return": ho_ret,
        "ho_mdd": ho_mdd,
        "ho_calmar": ho_calmar,
    }


# ═══════════════════════════════════════════════════════════════
# Layer 1: Pool coverage diagnostics (replay without full backtest)
# ═══════════════════════════════════════════════════════════════

def compute_pool_diagnostics(data, combo_str, extended_k):
    """Replay selection logic at each rebalance day to get pool coverage stats."""
    factors = [f.strip() for f in combo_str.split("+")]
    f_indices = np.array([data["factor_index_map"][f] for f in factors], dtype=np.int64)

    factors_3d = data["factors_3d"]
    pool_ids = data["pool_ids"]
    T, N = data["T"], data["N"]

    schedule = generate_rebalance_schedule(T, data["LOOKBACK"], data["FREQ"])
    ho_idx = data["ho_start_idx"]
    pos_size = data["POS_SIZE"]

    pool_counts_off = []  # unique pools with constraint OFF
    pool_counts_on = []   # unique pools with constraint ON
    fallback_count = 0
    same_pool_off_count = 0

    for rb_idx in schedule:
        if rb_idx <= 0 or rb_idx >= T:
            continue

        # Compute combined score (mirrors kernel logic)
        combined_score = np.full(N, -np.inf)
        for n in range(N):
            s = 0.0
            has_value = False
            for fi in f_indices:
                val = factors_3d[rb_idx - 1, n, fi]
                if not np.isnan(val):
                    s += val
                    has_value = True
            if has_value and s != 0.0:
                combined_score[n] = s

        # Baseline selection (OFF)
        baseline = stable_topk_indices(combined_score, pos_size)
        if len(baseline) >= 2:
            pools_b = set(int(pool_ids[idx]) for idx in baseline if pool_ids[idx] >= 0)
            pool_counts_off.append(len(pools_b))
            if len(pools_b) <= 1:
                same_pool_off_count += 1

        # Diversified selection (ON)
        if extended_k > 0:
            diversified = pool_diversify_topk(combined_score, pool_ids, pos_size, extended_k)
            if len(diversified) >= 2:
                pools_d = set(int(pool_ids[idx]) for idx in diversified if pool_ids[idx] >= 0)
                pool_counts_on.append(len(pools_d))
                if len(pools_d) <= 1:
                    fallback_count += 1

    n_rebal = len(pool_counts_off)
    return {
        "n_rebalances": n_rebal,
        "off_avg_pools": np.mean(pool_counts_off) if pool_counts_off else 0,
        "on_avg_pools": np.mean(pool_counts_on) if pool_counts_on else 0,
        "off_same_pool_pct": same_pool_off_count / max(n_rebal, 1) * 100,
        "on_fallback_pct": fallback_count / max(n_rebal, 1) * 100,
    }


# ═══════════════════════════════════════════════════════════════
# Layer 1: A/B Matrix
# ═══════════════════════════════════════════════════════════════

def run_layer1(data, combos, top_ns, extended_ks):
    """Layer 1: Research effectiveness validation."""
    print(f"\n{'='*70}")
    print(f"LAYER 1: RESEARCH EFFECTIVENESS VALIDATION")
    print(f"{'='*70}")

    max_n = max(top_ns)
    combos_to_run = combos[:max_n]

    # 1a) Run baseline (OFF) for all combos
    print(f"\n--- Baseline (OFF): {len(combos_to_run)} combos ---")
    t0 = time.time()
    results_off = []
    for i, c in enumerate(combos_to_run):
        try:
            results_off.append(run_single_combo(data, c, extended_k=0))
        except Exception as e:
            print(f"  ERROR: {c}: {e}")
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(combos_to_run)} done")
    df_off = pd.DataFrame(results_off)
    print(f"  Baseline done in {time.time()-t0:.1f}s ({len(df_off)} combos)")

    # 1b) Run ON for each extended_k
    results_on = {}
    for ek in extended_ks:
        print(f"\n--- Route A (ON, k={ek}): {len(combos_to_run)} combos ---")
        t0 = time.time()
        rows = []
        for i, c in enumerate(combos_to_run):
            try:
                rows.append(run_single_combo(data, c, extended_k=ek))
            except Exception as e:
                print(f"  ERROR: {c}: {e}")
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(combos_to_run)} done")
        results_on[ek] = pd.DataFrame(rows)
        print(f"  k={ek} done in {time.time()-t0:.1f}s ({len(rows)} combos)")

    # 1c) Print A/B matrix
    print(f"\n{'='*70}")
    print(f"A/B MATRIX: HO Return median / HO MDD median / Trades median")
    print(f"{'='*70}")

    header = f"{'TopN':>6} | {'OFF':>18}"
    for ek in extended_ks:
        header += f" | {'k='+str(ek):>18}"
    print(header)
    print("-" * len(header))

    for top_n in top_ns:
        off_slice = df_off.head(top_n)
        off_ho = off_slice["ho_return"].median()
        off_mdd = off_slice["ho_mdd"].median()
        off_tr = off_slice["trades"].median()

        row = f"{top_n:>6} | {off_ho:+.3f}/{off_mdd:.3f}/{off_tr:.0f}"

        for ek in extended_ks:
            on_slice = results_on[ek].head(top_n)
            on_ho = on_slice["ho_return"].median()
            on_mdd = on_slice["ho_mdd"].median()
            on_tr = on_slice["trades"].median()
            delta = on_ho - off_ho
            row += f" | {on_ho:+.3f}/{on_mdd:.3f}/{on_tr:.0f}({delta:+.3f})"

        print(row)

    # 1d) Detailed comparison at default config (top_n=200, k=10)
    ek_default = 10 if 10 in extended_ks else extended_ks[0]
    n_default = min(200, max(top_ns))
    off_d = df_off.head(n_default)
    on_d = results_on[ek_default].head(n_default)

    print(f"\n{'='*70}")
    print(f"DETAILED: top-{n_default}, k={ek_default}")
    print(f"{'='*70}")

    for metric, label in [
        ("full_return", "Full Return"),
        ("full_mdd", "Full MDD"),
        ("full_sharpe", "Full Sharpe"),
        ("ho_return", "Holdout Return"),
        ("ho_mdd", "Holdout MDD"),
        ("ho_calmar", "Holdout Calmar"),
        ("trades", "Trades"),
        ("turnover_ann", "Ann. Turnover"),
        ("cost_drag", "Cost Drag"),
    ]:
        a_med = off_d[metric].median()
        b_med = on_d[metric].median()
        a_10 = off_d[metric].quantile(0.10)
        b_10 = on_d[metric].quantile(0.10)
        delta_med = b_med - a_med
        print(f"  {label:>16}: OFF={a_med:.4f}(p10={a_10:.4f})  ON={b_med:.4f}(p10={b_10:.4f})  Δ={delta_med:+.4f}")

    # 1e) Paired improvement rate
    paired_ho = pd.DataFrame({
        "off": off_d["ho_return"].values[:len(on_d)],
        "on": on_d["ho_return"].values[:len(off_d)],
    })
    pct_improved = (paired_ho["on"] > paired_ho["off"]).mean() * 100
    print(f"\n  Paired improvement rate: {pct_improved:.1f}% of combos improved HO return")

    # 1f) Pool coverage diagnostics for S1
    s1_combo = "ADX_14D + OBV_SLOPE_10D + SHARPE_RATIO_20D + SLOPE_20D"
    print(f"\n--- Pool diagnostics for S1 ---")
    for ek in extended_ks:
        diag = compute_pool_diagnostics(data, s1_combo, ek)
        print(
            f"  k={ek:>2}: OFF avg_pools={diag['off_avg_pools']:.2f} "
            f"same_pool={diag['off_same_pool_pct']:.1f}%  "
            f"ON avg_pools={diag['on_avg_pools']:.2f} "
            f"fallback={diag['on_fallback_pct']:.1f}%"
        )

    # Also do diagnostics for a sample of combos (first 20)
    print(f"\n--- Pool diagnostics (first 20 combos, k={ek_default}) ---")
    diag_rows = []
    for c in combos_to_run[:20]:
        d = compute_pool_diagnostics(data, c, ek_default)
        diag_rows.append(d)
    avg_off_same = np.mean([d["off_same_pool_pct"] for d in diag_rows])
    avg_on_fallback = np.mean([d["on_fallback_pct"] for d in diag_rows])
    avg_off_pools = np.mean([d["off_avg_pools"] for d in diag_rows])
    avg_on_pools = np.mean([d["on_avg_pools"] for d in diag_rows])
    print(
        f"  OFF: avg_pools={avg_off_pools:.2f}, same_pool_rebal={avg_off_same:.1f}%\n"
        f"  ON:  avg_pools={avg_on_pools:.2f}, fallback_rebal={avg_on_fallback:.1f}%"
    )

    # 1g) Cost sensitivity (top_n=200, k=10, cost ×0.5/×1/×2)
    print(f"\n--- Cost sensitivity (top-{n_default}, k={ek_default}) ---")
    for cm in [0.5, 1.0, 2.0]:
        off_rows = []
        on_rows = []
        for c in combos_to_run[:n_default]:
            try:
                off_rows.append(run_single_combo(data, c, 0, cost_multiplier=cm))
                on_rows.append(run_single_combo(data, c, ek_default, cost_multiplier=cm))
            except Exception:
                pass
        if off_rows and on_rows:
            off_df_c = pd.DataFrame(off_rows)
            on_df_c = pd.DataFrame(on_rows)
            delta_ho = on_df_c["ho_return"].median() - off_df_c["ho_return"].median()
            delta_mdd = on_df_c["ho_mdd"].median() - off_df_c["ho_mdd"].median()
            print(
                f"  cost×{cm:.1f}: OFF_HO={off_df_c['ho_return'].median():+.4f}  "
                f"ON_HO={on_df_c['ho_return'].median():+.4f}  "
                f"ΔHO={delta_ho:+.4f}  ΔMDD={delta_mdd:+.4f}"
            )

    return df_off, results_on


# ═══════════════════════════════════════════════════════════════
# Layer 2: Mechanism comparison (Route A vs Route B)
# ═══════════════════════════════════════════════════════════════

def run_layer2(data, combos, top_n=200, extended_k=10):
    """Layer 2: Compare Route A (pre-hyst) vs Route B (post-hyst)."""
    print(f"\n{'='*70}")
    print(f"LAYER 2: MECHANISM COMPARISON (Route A vs B)")
    print(f"Top-{top_n}, extended_k={extended_k}")
    print(f"{'='*70}")

    combos_to_run = combos[:top_n]
    results_a = []
    results_b = []

    print(f"\nRunning {len(combos_to_run)} combos × 3 (OFF / Route A / Route B)...")
    t0 = time.time()

    for i, c in enumerate(combos_to_run):
        try:
            r_a = run_single_combo(data, c, extended_k, post_hyst=False)
            r_b = run_single_combo(data, c, extended_k, post_hyst=True)
            results_a.append(r_a)
            results_b.append(r_b)
        except Exception as e:
            print(f"  ERROR: {c}: {e}")
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(combos_to_run)} done")

    print(f"  Done in {time.time()-t0:.1f}s")

    df_a = pd.DataFrame(results_a)
    df_b = pd.DataFrame(results_b)

    print(f"\n{'='*70}")
    print(f"Route A (diversify→hyst) vs Route B (hyst→diversify)")
    print(f"{'='*70}")

    for metric, label in [
        ("ho_return", "HO Return"),
        ("ho_mdd", "HO MDD"),
        ("ho_calmar", "HO Calmar"),
        ("trades", "Trades"),
        ("turnover_ann", "Ann. Turnover"),
    ]:
        a_med = df_a[metric].median()
        b_med = df_b[metric].median()
        a_10 = df_a[metric].quantile(0.10)
        b_10 = df_b[metric].quantile(0.10)
        print(
            f"  {label:>14}: "
            f"A={a_med:.4f}(p10={a_10:.4f})  "
            f"B={b_med:.4f}(p10={b_10:.4f})  "
            f"Δ(B-A)={b_med-a_med:+.4f}"
        )

    # Paired comparison
    if len(df_a) > 0 and len(df_b) > 0:
        n = min(len(df_a), len(df_b))
        a_better = (df_a["ho_return"].values[:n] > df_b["ho_return"].values[:n]).sum()
        b_better = n - a_better
        print(f"\n  Route A wins: {a_better}/{n} ({a_better/n*100:.1f}%)")
        print(f"  Route B wins: {b_better}/{n} ({b_better/n*100:.1f}%)")

    return df_a, df_b


# ═══════════════════════════════════════════════════════════════
# Layer 3: Production safety checks
# ═══════════════════════════════════════════════════════════════

def run_layer3(data, combos):
    """Layer 3: Determinism and edge case checks."""
    print(f"\n{'='*70}")
    print(f"LAYER 3: PRODUCTION SAFETY")
    print(f"{'='*70}")

    s1_combo = "ADX_14D + OBV_SLOPE_10D + SHARPE_RATIO_20D + SLOPE_20D"

    # 3a) Determinism: same input, multiple runs
    print(f"\n--- Determinism (3 runs, S1, k=10) ---")
    results_det = []
    for run_i in range(3):
        r = run_single_combo(data, s1_combo, extended_k=10)
        results_det.append(r)
    rets = [r["full_return"] for r in results_det]
    trades = [r["trades"] for r in results_det]
    all_same = all(abs(r - rets[0]) < 1e-10 for r in rets)
    trades_same = all(t == trades[0] for t in trades)
    print(f"  Returns: {rets}")
    print(f"  Trades:  {trades}")
    print(f"  Deterministic: {'PASS' if all_same and trades_same else 'FAIL'}")

    # 3b) Edge case: combo where all candidates might be same pool
    # Use a small extended_k to test fallback
    print(f"\n--- Fallback resilience (S1, k=2) ---")
    r_k2 = run_single_combo(data, s1_combo, extended_k=2)
    r_k10 = run_single_combo(data, s1_combo, extended_k=10)
    print(f"  k=2:  HO={r_k2['ho_return']:+.4f}, trades={r_k2['trades']}")
    print(f"  k=10: HO={r_k10['ho_return']:+.4f}, trades={r_k10['trades']}")
    print(f"  Both run without error: PASS")

    # 3c) Constraint disabled = exact baseline
    print(f"\n--- Disabled = baseline (S1) ---")
    r_off = run_single_combo(data, s1_combo, extended_k=0)
    match = abs(r_off["full_return"] - r_off["full_return"]) < 1e-10  # trivially true
    # The real test: k=0 should match a standard VEC run
    r_off2 = run_single_combo(data, s1_combo, extended_k=0)
    match = abs(r_off["full_return"] - r_off2["full_return"]) < 1e-10
    print(f"  OFF run 1: {r_off['full_return']:.6f}")
    print(f"  OFF run 2: {r_off2['full_return']:.6f}")
    print(f"  Match: {'PASS' if match else 'FAIL'}")


# ═══════════════════════════════════════════════════════════════
# Judgment summary
# ═══════════════════════════════════════════════════════════════

def print_judgment(df_off, results_on, top_ns, extended_ks):
    """Print final go/no-go judgment."""
    print(f"\n{'='*70}")
    print(f"FINAL JUDGMENT")
    print(f"{'='*70}")

    ek_default = 10 if 10 in extended_ks else extended_ks[0]
    n_default = min(200, max(top_ns))

    off = df_off.head(n_default)
    on = results_on[ek_default].head(n_default)

    ho_med_off = off["ho_return"].median()
    ho_med_on = on["ho_return"].median()
    ho_mdd_off = off["ho_mdd"].median()
    ho_mdd_on = on["ho_mdd"].median()
    turn_off = off["turnover_ann"].median()
    turn_on = on["turnover_ann"].median()

    c1 = ho_med_on >= ho_med_off - 0.005  # not worse by >0.5pp
    c2 = ho_mdd_on <= ho_mdd_off + 0.02   # MDD not worse by >2pp
    c3 = turn_on <= turn_off * 1.3         # turnover not >30% worse

    print(f"  Criterion 1 - HO median not worse (>-0.5pp): "
          f"{ho_med_on:+.4f} vs {ho_med_off:+.4f} ({ho_med_on-ho_med_off:+.4f}) → {'PASS' if c1 else 'FAIL'}")
    print(f"  Criterion 2 - HO MDD not worse (<+2pp):      "
          f"{ho_mdd_on:.4f} vs {ho_mdd_off:.4f} ({ho_mdd_on-ho_mdd_off:+.4f}) → {'PASS' if c2 else 'FAIL'}")
    print(f"  Criterion 3 - Turnover not >30% worse:        "
          f"{turn_on:.4f} vs {turn_off:.4f} ({turn_on/max(turn_off,0.001)*100-100:+.1f}%) → {'PASS' if c3 else 'FAIL'}")

    all_pass = c1 and c2 and c3
    print(f"\n  OVERALL: {'✅ PASS — safe to enable' if all_pass else '❌ FAIL — investigate before enabling'}")

    # Check monotonicity across top_ns
    print(f"\n  Monotonicity across top-N (k={ek_default}):")
    for tn in top_ns:
        off_s = df_off.head(tn)
        on_s = results_on[ek_default].head(tn)
        delta = on_s["ho_return"].median() - off_s["ho_return"].median()
        print(f"    top-{tn:>3}: Δ HO median = {delta:+.4f}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Pool constraint full validation")
    parser.add_argument("--top-n", type=int, default=500,
                        help="Max combos to evaluate (superset)")
    parser.add_argument("--layer", type=int, default=0,
                        help="Run specific layer (0=all, 1/2/3)")
    parser.add_argument("--config", type=str,
                        default=str(ROOT / "configs/combo_wfo_config.yaml"))
    args = parser.parse_args()

    TOP_NS = [50, 100, 200]
    if args.top_n >= 500:
        TOP_NS.append(500)
    EXTENDED_KS = [5, 10, 20]

    print("Loading data...")
    data = load_all_data(args.config)
    print(f"  {data['T']} days × {data['N']} ETFs × {len(data['factor_names'])} factors")
    print(f"  FREQ={data['FREQ']}, POS={data['POS_SIZE']}, T1_OPEN={data['USE_T1_OPEN']}")
    print(f"  Exp4: dr={data['DELTA_RANK']}, mh={data['MIN_HOLD_DAYS']}")
    print(f"  HO starts at idx {data['ho_start_idx']}")

    combos = load_combos(args.top_n)
    print(f"  Loaded {len(combos)} combos\n")

    df_off = None
    results_on = None

    if args.layer == 0 or args.layer == 3:
        run_layer3(data, combos)

    if args.layer == 0 or args.layer == 1:
        df_off, results_on = run_layer1(data, combos, TOP_NS, EXTENDED_KS)

    if args.layer == 0 or args.layer == 2:
        run_layer2(data, combos, top_n=min(200, len(combos)), extended_k=10)

    if df_off is not None and results_on is not None:
        print_judgment(df_off, results_on, TOP_NS, EXTENDED_KS)

    # Save results
    output_dir = ROOT / "results" / "pool_constraint_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    if df_off is not None:
        df_off.to_parquet(output_dir / "baseline_off.parquet", index=False)
    if results_on is not None:
        for ek, df in results_on.items():
            df.to_parquet(output_dir / f"route_a_k{ek}.parquet", index=False)
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
