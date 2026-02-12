#!/usr/bin/env python3
"""A/B test: S1 baseline vs S1 + COMMODITY max 1.

Runs VEC for both variants over full period (train + holdout) and compares:
  - Holdout MDD (especially 2025-10 ~ 2025-12 drawdown period)
  - Holdout 10th percentile monthly return
  - Holdout total return (does constraint hurt returns?)

The COMMODITY-only constraint is implemented via pool_ids trick:
  - COMMODITY ETFs (518850, 518880) get pool_id = 5
  - All other ETFs get pool_id = -1 (treated as unique, unrestricted)

Usage:
  uv run python scripts/ab_commodity_max1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from etf_strategy.core.cost_model import build_cost_array, load_cost_model
from etf_strategy.core.execution_model import load_execution_model
from etf_strategy.core.frozen_params import FrozenETFPool, load_frozen_config
from shadow_retrospective_q1q2 import (
    STRATEGIES,
    get_factor_indices,
    load_pipeline_data,
)
from batch_vec_backtest import run_vec_backtest

# COMMODITY ETFs (gold + silver)
COMMODITY_CODES = {"518850", "518880"}
COMMODITY_POOL_ID = 5  # matches POOL_NAME_TO_ID["COMMODITY"]


def run_vec_ab(data, config, factor_indices, pool_ids_override=None, extended_k=0):
    """Run VEC backtest with optional pool constraint."""
    bt_cfg = config.get("backtest", {})
    freq = bt_cfg["freq"]
    pos_size = bt_cfg["pos_size"]
    lookback = bt_cfg["lookback"]
    initial_capital = float(bt_cfg["initial_capital"])

    cost_model = load_cost_model(config)
    qdii_set = set(FrozenETFPool().qdii_codes)
    cost_arr = build_cost_array(cost_model, data["etf_codes"], qdii_set)
    exec_model = load_execution_model(config)

    hysteresis_cfg = bt_cfg.get("hysteresis", {})
    delta_rank = hysteresis_cfg.get("delta_rank", 0.0)
    min_hold_days = hysteresis_cfg.get("min_hold_days", 0)

    result = run_vec_backtest(
        data["factors_3d"],
        data["close"],
        data["open"],
        data["high"],
        data["low"],
        data["timing_arr"],
        factor_indices,
        freq=freq,
        pos_size=pos_size,
        initial_capital=initial_capital,
        commission_rate=float(bt_cfg["commission_rate"]),
        lookback=lookback,
        cost_arr=cost_arr,
        use_t1_open=exec_model.is_t1_open,
        trailing_stop_pct=0.0,
        stop_on_rebalance_only=True,
        delta_rank=delta_rank,
        min_hold_days=min_hold_days,
        pool_ids=pool_ids_override,
        pool_constraint_extended_k=extended_k,
    )

    eq = result[0]
    total_ret = result[1]
    n_trades = result[4]
    risk = result[6]  # risk_metrics dict

    return {
        "equity": eq,
        "total_return": total_ret,
        "n_trades": n_trades,
        "max_dd": risk["max_drawdown"],
        "sharpe": risk["sharpe_ratio"],
    }


def compute_holdout_metrics(eq, dates, holdout_start_idx):
    """Compute holdout-specific metrics from equity curve."""
    eq_ho = eq[holdout_start_idx:]
    dates_ho = dates[holdout_start_idx:]

    # HO return
    ho_ret = eq_ho[-1] / eq_ho[0] - 1

    # HO MDD
    peak = np.maximum.accumulate(eq_ho)
    dd = (eq_ho - peak) / peak
    ho_mdd = dd.min()

    # MDD period
    trough_idx = int(np.argmin(dd))
    peak_idx = int(np.argmax(eq_ho[: trough_idx + 1])) if trough_idx > 0 else 0

    # Monthly returns
    daily_ret = np.diff(eq_ho) / eq_ho[:-1]
    dates_ho_pd = pd.DatetimeIndex(dates_ho)
    df_ret = pd.DataFrame({"ret": np.concatenate([[0.0], daily_ret])}, index=dates_ho_pd)
    monthly = df_ret["ret"].resample("ME").apply(lambda x: (1 + x).prod() - 1)

    worst_month = monthly.min()
    p10_month = monthly.quantile(0.1)

    # Oct-Dec 2025 drawdown specifically
    oct_dec_mask = (dates_ho_pd >= "2025-10-01") & (dates_ho_pd <= "2025-12-31")
    if oct_dec_mask.any():
        eq_oct_dec = eq_ho[oct_dec_mask]
        peak_oct_dec = np.maximum.accumulate(eq_oct_dec)
        dd_oct_dec = (eq_oct_dec - peak_oct_dec) / peak_oct_dec
        oct_dec_mdd = dd_oct_dec.min()
        oct_dec_ret = eq_oct_dec[-1] / eq_oct_dec[0] - 1
    else:
        oct_dec_mdd = 0.0
        oct_dec_ret = 0.0

    return {
        "ho_return": ho_ret,
        "ho_mdd": ho_mdd,
        "ho_mdd_peak_date": str(dates_ho[peak_idx]),
        "ho_mdd_trough_date": str(dates_ho[trough_idx]),
        "worst_month": worst_month,
        "p10_month": p10_month,
        "oct_dec_mdd": oct_dec_mdd,
        "oct_dec_return": oct_dec_ret,
        "monthly_returns": monthly,
    }


def main():
    config_path = ROOT / "configs" / "combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    frozen = load_frozen_config(config, config_path=str(config_path))

    print("=" * 70)
    print("A/B Test: S1 baseline vs S1 + COMMODITY max 1")
    print(f"Config: v{frozen.version}, FREQ={config['backtest']['freq']}, "
          f"POS_SIZE={config['backtest']['pos_size']}")
    print("=" * 70)

    # Load data
    print("\n--- Loading data ---")
    data = load_pipeline_data(config)
    dates = data["dates"]
    etf_codes = data["etf_codes"]
    factor_names = data["factor_names"]

    # Holdout start
    training_end = pd.Timestamp(config["data"]["training_end_date"])
    holdout_start_idx = 0
    for i, d in enumerate(dates):
        if pd.Timestamp(d) > training_end:
            holdout_start_idx = i
            break
    print(f"Holdout: {dates[holdout_start_idx]} ~ {dates[-1]}")

    # Factor indices for S1
    s1_factors = STRATEGIES["S1"]
    factor_idx = get_factor_indices(factor_names, s1_factors)
    print(f"S1 factors: {s1_factors}")

    # Build COMMODITY-only pool_ids
    # All ETFs get -1 (unrestricted), except COMMODITY which gets COMMODITY_POOL_ID
    commodity_pool_ids = np.full(len(etf_codes), -1, dtype=np.int64)
    for i, code in enumerate(etf_codes):
        if code in COMMODITY_CODES:
            commodity_pool_ids[i] = COMMODITY_POOL_ID
    n_commodity = int((commodity_pool_ids == COMMODITY_POOL_ID).sum())
    print(f"COMMODITY ETFs tagged: {n_commodity} ({[etf_codes[i] for i in range(len(etf_codes)) if commodity_pool_ids[i] == COMMODITY_POOL_ID]})")

    # ──────────────────────────────────────────────
    # Run A: Baseline (no pool constraint)
    # ──────────────────────────────────────────────
    print("\n--- Running A: S1 baseline (no constraint) ---")
    result_a = run_vec_ab(data, config, factor_idx)
    metrics_a = compute_holdout_metrics(result_a["equity"], dates, holdout_start_idx)

    # ──────────────────────────────────────────────
    # Run B: COMMODITY max 1
    # ──────────────────────────────────────────────
    print("--- Running B: S1 + COMMODITY max 1 ---")
    result_b = run_vec_ab(
        data, config, factor_idx,
        pool_ids_override=commodity_pool_ids,
        extended_k=10,
    )
    metrics_b = compute_holdout_metrics(result_b["equity"], dates, holdout_start_idx)

    # ──────────────────────────────────────────────
    # Also run C: Full pool diversity (all 7 pools)
    # ──────────────────────────────────────────────
    print("--- Running C: S1 + full 7-pool diversity ---")
    from etf_strategy.core.etf_pool_mapper import load_pool_mapping, build_pool_array
    full_pool_map = load_pool_mapping(ROOT / "configs" / "etf_pools.yaml")
    full_pool_ids = build_pool_array(etf_codes, full_pool_map)

    result_c = run_vec_ab(
        data, config, factor_idx,
        pool_ids_override=full_pool_ids,
        extended_k=10,
    )
    metrics_c = compute_holdout_metrics(result_c["equity"], dates, holdout_start_idx)

    # ──────────────────────────────────────────────
    # Results
    # ──────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")

    header = f"{'Metric':30s} {'A: Baseline':>14s} {'B: Cmd max1':>14s} {'C: 7-pool':>14s} {'B-A':>10s} {'C-A':>10s}"
    print(f"\n{header}")
    print("-" * len(header))

    rows = [
        ("HO Return", f"{metrics_a['ho_return']:+.1%}", f"{metrics_b['ho_return']:+.1%}", f"{metrics_c['ho_return']:+.1%}",
         f"{metrics_b['ho_return'] - metrics_a['ho_return']:+.1%}", f"{metrics_c['ho_return'] - metrics_a['ho_return']:+.1%}"),
        ("HO MDD", f"{metrics_a['ho_mdd']:.1%}", f"{metrics_b['ho_mdd']:.1%}", f"{metrics_c['ho_mdd']:.1%}",
         f"{metrics_b['ho_mdd'] - metrics_a['ho_mdd']:+.1%}", f"{metrics_c['ho_mdd'] - metrics_a['ho_mdd']:+.1%}"),
        ("Worst Month", f"{metrics_a['worst_month']:+.1%}", f"{metrics_b['worst_month']:+.1%}", f"{metrics_c['worst_month']:+.1%}",
         f"{metrics_b['worst_month'] - metrics_a['worst_month']:+.1%}", f"{metrics_c['worst_month'] - metrics_a['worst_month']:+.1%}"),
        ("10th pct Month", f"{metrics_a['p10_month']:+.1%}", f"{metrics_b['p10_month']:+.1%}", f"{metrics_c['p10_month']:+.1%}",
         f"{metrics_b['p10_month'] - metrics_a['p10_month']:+.1%}", f"{metrics_c['p10_month'] - metrics_a['p10_month']:+.1%}"),
        ("Oct-Dec 2025 MDD", f"{metrics_a['oct_dec_mdd']:.1%}", f"{metrics_b['oct_dec_mdd']:.1%}", f"{metrics_c['oct_dec_mdd']:.1%}",
         f"{metrics_b['oct_dec_mdd'] - metrics_a['oct_dec_mdd']:+.1%}", f"{metrics_c['oct_dec_mdd'] - metrics_a['oct_dec_mdd']:+.1%}"),
        ("Oct-Dec 2025 Return", f"{metrics_a['oct_dec_return']:+.1%}", f"{metrics_b['oct_dec_return']:+.1%}", f"{metrics_c['oct_dec_return']:+.1%}",
         f"{metrics_b['oct_dec_return'] - metrics_a['oct_dec_return']:+.1%}", f"{metrics_c['oct_dec_return'] - metrics_a['oct_dec_return']:+.1%}"),
        ("Full Sharpe", f"{result_a['sharpe']:.2f}", f"{result_b['sharpe']:.2f}", f"{result_c['sharpe']:.2f}",
         f"{result_b['sharpe'] - result_a['sharpe']:+.2f}", f"{result_c['sharpe'] - result_a['sharpe']:+.2f}"),
        ("Trades", f"{result_a['n_trades']}", f"{result_b['n_trades']}", f"{result_c['n_trades']}",
         f"{result_b['n_trades'] - result_a['n_trades']:+d}", f"{result_c['n_trades'] - result_a['n_trades']:+d}"),
    ]

    for label, a, b, c, ba, ca in rows:
        print(f"{label:30s} {a:>14s} {b:>14s} {c:>14s} {ba:>10s} {ca:>10s}")

    # MDD period details
    print(f"\nMDD periods:")
    print(f"  A: {metrics_a['ho_mdd_peak_date']} ~ {metrics_a['ho_mdd_trough_date']}")
    print(f"  B: {metrics_b['ho_mdd_peak_date']} ~ {metrics_b['ho_mdd_trough_date']}")
    print(f"  C: {metrics_c['ho_mdd_peak_date']} ~ {metrics_c['ho_mdd_trough_date']}")

    # Monthly return comparison
    print(f"\n{'=' * 70}")
    print("MONTHLY RETURNS (Holdout)")
    print(f"{'=' * 70}")
    monthly_df = pd.DataFrame({
        "A_baseline": metrics_a["monthly_returns"],
        "B_cmd_max1": metrics_b["monthly_returns"],
        "C_7pool": metrics_c["monthly_returns"],
    })
    monthly_df["B-A"] = monthly_df["B_cmd_max1"] - monthly_df["A_baseline"]
    monthly_df["C-A"] = monthly_df["C_7pool"] - monthly_df["A_baseline"]
    for col in monthly_df.columns:
        monthly_df[col] = monthly_df[col].map(lambda x: f"{x:+.2%}")
    print(monthly_df.to_string())

    # ──────────────────────────────────────────────
    # Decision
    # ──────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("DECISION")
    print(f"{'=' * 70}")

    b_mdd_better = metrics_b["ho_mdd"] > metrics_a["ho_mdd"]  # less negative = better
    b_ret_similar = abs(metrics_b["ho_return"] - metrics_a["ho_return"]) < 0.05  # within 5pp
    b_oct_dec_better = metrics_b["oct_dec_mdd"] > metrics_a["oct_dec_mdd"]

    print(f"  B MDD improved:       {'YES' if b_mdd_better else 'NO'} ({metrics_b['ho_mdd'] - metrics_a['ho_mdd']:+.1%})")
    print(f"  B Return preserved:   {'YES' if b_ret_similar else 'NO'} ({metrics_b['ho_return'] - metrics_a['ho_return']:+.1%})")
    print(f"  B Oct-Dec improved:   {'YES' if b_oct_dec_better else 'NO'} ({metrics_b['oct_dec_mdd'] - metrics_a['oct_dec_mdd']:+.1%})")

    if b_mdd_better and b_ret_similar:
        print("\n  >>> COMMODITY max 1 constraint IMPROVES risk without material return loss")
        print("  >>> Recommend enabling for production")
    elif b_mdd_better and not b_ret_similar:
        print("\n  >>> COMMODITY max 1 reduces risk but costs return")
        print("  >>> Trade-off decision needed")
    else:
        print("\n  >>> COMMODITY max 1 does NOT improve risk profile")
        print("  >>> Do not enable")

    # Save equity curves for visual inspection
    out_dir = ROOT / "results" / "ab_commodity_max1"
    out_dir.mkdir(exist_ok=True)
    eq_df = pd.DataFrame({
        "date": dates,
        "A_baseline": result_a["equity"],
        "B_commodity_max1": result_b["equity"],
        "C_7pool": result_c["equity"],
    })
    eq_df.to_csv(out_dir / "equity_curves.csv", index=False)
    print(f"\nEquity curves saved to {out_dir / 'equity_curves.csv'}")


if __name__ == "__main__":
    main()
