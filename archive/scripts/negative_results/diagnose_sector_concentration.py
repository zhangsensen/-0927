#!/usr/bin/env python3
"""S1 板块集中度诊断 (7-pool).

在 holdout 期间检查 S1 的持仓是否经常双持同板块，以及同板块持仓
是否与亏损集中化相关，从而决定是否需要添加板块约束。

输出三项指标:
  1. same_pool_rate: 调仓日 top-2 同粗池比例
  2. same_pool_loss_share: 最差 20% 调仓周期中同池占比
  3. same_pool_dd_contrib: 最大回撤区间内同池状态的天数占比

判据 (比 "30%" 拍脑袋更靠谱):
  - 如果 same_pool_loss_share - same_pool_rate > 15pp → 同池是尾部放大器, 值得做约束
  - 如果差值 < 5pp → 同池不是尾部主因, 约束可能空跑

Usage:
  uv run python scripts/diagnose_sector_concentration.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from etf_strategy.core.etf_pool_mapper import (
    POOL_ID_TO_NAME,
    build_pool_array,
    load_pool_mapping,
)
from etf_strategy.core.frozen_params import load_frozen_config
from shadow_retrospective_q1q2 import (
    STRATEGIES,
    get_factor_indices,
    load_pipeline_data,
    reconstruct_holdings,
    run_vec_for_strategy,
)


def main():
    config_path = ROOT / "configs" / "combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    frozen = load_frozen_config(config, config_path=str(config_path))

    print("=" * 70)
    print("S1 Sector Concentration Diagnostic (7-pool)")
    print(f"Config: v{frozen.version}, FREQ={config['backtest']['freq']}, "
          f"POS_SIZE={config['backtest']['pos_size']}")
    print("=" * 70)

    # Load data
    print("\n--- Loading data & factors ---")
    data = load_pipeline_data(config)
    dates = data["dates"]
    etf_codes = data["etf_codes"]
    factor_names = data["factor_names"]
    T = len(dates)
    N = len(etf_codes)

    # Holdout start
    training_end = pd.Timestamp(config["data"]["training_end_date"])
    holdout_start_idx = 0
    for i, d in enumerate(dates):
        if pd.Timestamp(d) > training_end:
            holdout_start_idx = i
            break
    print(f"Data: {T} days x {N} ETFs x {len(factor_names)} factors")
    print(f"Holdout: {dates[holdout_start_idx]} ~ {dates[-1]}")

    # Pool mapping
    pool_map = load_pool_mapping(ROOT / "configs" / "etf_pools.yaml")
    pool_ids = build_pool_array(etf_codes, pool_map)

    # Reconstruct S1 holdings
    s1_factors = STRATEGIES["S1"]
    factor_idx = get_factor_indices(factor_names, s1_factors)

    print(f"\nReconstructing S1 holdings ({' + '.join(s1_factors)})...")
    h_history, _sc_history, rebal_indices = reconstruct_holdings(data, config, factor_idx)

    # VEC equity curve for period returns
    print("Running VEC for equity curve...")
    eq, _total_ret, _n_trades = run_vec_for_strategy(data, config, factor_idx)

    # Filter to holdout
    ho_mask = np.array(rebal_indices) >= holdout_start_idx
    h_ho = h_history[ho_mask]
    rebal_ho = np.array(rebal_indices)[ho_mask]
    R = len(rebal_ho)

    print(f"Holdout rebalances: {R}")

    # ──────────────────────────────────────────────────────────
    # Metric 1: same_pool_rate
    # ──────────────────────────────────────────────────────────
    same_pool_count = 0
    pool_pairs: list[dict] = []

    for r in range(R):
        held = np.where(h_ho[r])[0]
        if len(held) < 2:
            pools_held = [int(pool_ids[held[0]])] if len(held) == 1 else []
            pool_names = [POOL_ID_TO_NAME.get(p, f"UNK({p})") for p in pools_held]
            pool_pairs.append({
                "date": str(dates[rebal_ho[r]]),
                "etf1": etf_codes[held[0]] if len(held) >= 1 else "",
                "etf2": "",
                "pool1": pool_names[0] if pool_names else "",
                "pool2": "",
                "same_pool": False,
            })
            continue

        p1, p2 = int(pool_ids[held[0]]), int(pool_ids[held[1]])
        n1 = POOL_ID_TO_NAME.get(p1, f"UNK({p1})")
        n2 = POOL_ID_TO_NAME.get(p2, f"UNK({p2})")
        same = p1 == p2

        if same:
            same_pool_count += 1

        pool_pairs.append({
            "date": str(dates[rebal_ho[r]]),
            "etf1": etf_codes[held[0]],
            "etf2": etf_codes[held[1]],
            "pool1": n1,
            "pool2": n2,
            "same_pool": same,
        })

    same_pool_rate = same_pool_count / R if R > 0 else 0.0

    # ──────────────────────────────────────────────────────────
    # Metric 2: same_pool_loss_share
    # ──────────────────────────────────────────────────────────
    # Period returns between consecutive rebalances
    period_returns = []
    for r in range(R):
        t_start = rebal_ho[r]
        t_end = rebal_ho[r + 1] if r + 1 < R else len(eq) - 1
        if t_end > t_start and eq[t_start] > 0:
            period_ret = eq[t_end] / eq[t_start] - 1
        else:
            period_ret = 0.0
        period_returns.append(period_ret)

    period_returns_arr = np.array(period_returns)

    # Worst 20% of periods
    n_worst = max(1, int(R * 0.2))
    worst_indices = np.argsort(period_returns_arr)[:n_worst]

    worst_same_count = sum(1 for i in worst_indices if pool_pairs[i]["same_pool"])
    same_pool_loss_share = worst_same_count / n_worst if n_worst > 0 else 0.0

    # ──────────────────────────────────────────────────────────
    # Metric 3: same_pool_dd_contrib
    # ──────────────────────────────────────────────────────────
    eq_ho = eq[holdout_start_idx:]
    peak = np.maximum.accumulate(eq_ho)
    dd = (eq_ho - peak) / peak

    # Find max DD trough and preceding peak
    trough_idx = int(np.argmin(dd))
    peak_idx = int(np.argmax(eq_ho[: trough_idx + 1])) if trough_idx > 0 else 0

    dd_start = holdout_start_idx + peak_idx
    dd_end = holdout_start_idx + trough_idx

    # Which rebalances fall in DD period?
    dd_rebal_mask = (rebal_ho >= dd_start) & (rebal_ho <= dd_end)
    dd_rebals = np.where(dd_rebal_mask)[0]

    dd_same = 0
    if len(dd_rebals) > 0:
        dd_same = sum(1 for i in dd_rebals if pool_pairs[i]["same_pool"])
        same_pool_dd_contrib = dd_same / len(dd_rebals)
    else:
        same_pool_dd_contrib = 0.0

    # ──────────────────────────────────────────────────────────
    # Results
    # ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")

    print(f"\n1) same_pool_rate: {same_pool_rate:.1%} ({same_pool_count}/{R})")
    print(f"   Top-2 from same 7-pool category on {same_pool_rate:.0%} of rebalance days")

    print(f"\n2) same_pool_loss_share: {same_pool_loss_share:.1%} ({worst_same_count}/{n_worst})")
    print(f"   In worst 20% periods, same-pool fraction = {same_pool_loss_share:.0%}")

    print(f"\n3) same_pool_dd_contrib: {same_pool_dd_contrib:.1%}")
    dd_date_start = dates[dd_start] if dd_start < len(dates) else "N/A"
    dd_date_end = dates[dd_end] if dd_end < len(dates) else "N/A"
    print(f"   Max DD period: {dd_date_start} ~ {dd_date_end}, MDD = {dd.min():.1%}")
    if len(dd_rebals) > 0:
        print(f"   Rebalances in DD period: {len(dd_rebals)}, same-pool: {dd_same}")
    else:
        print("   (No rebalances during max DD period)")

    # ──────────────────────────────────────────────────────────
    # Decision
    # ──────────────────────────────────────────────────────────
    delta = same_pool_loss_share - same_pool_rate
    print(f"\n{'=' * 70}")
    print("DECISION")
    print(f"{'=' * 70}")
    print(f"  same_pool_loss_share - same_pool_rate = {delta:+.1%}")

    if delta > 0.15:
        print(f"  >>> Same-pool IS a loss amplifier (+{delta:.0%} > 15pp)")
        print(f"  >>> Sector constraint WORTH BUILDING")
    elif delta > 0.05:
        print(f"  >>> Marginal effect (+{delta:.0%}), borderline case")
        print(f"  >>> Check detail table before deciding")
    else:
        print(f"  >>> Same-pool is NOT a tail driver ({delta:+.0%} <= 5pp)")
        print(f"  >>> Sector constraint would be EMPTY OVERHEAD")

    # ──────────────────────────────────────────────────────────
    # Detail table
    # ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("DETAIL: All holdout rebalances")
    print(f"{'=' * 70}")

    df_pairs = pd.DataFrame(pool_pairs)
    df_pairs["period_return"] = period_returns_arr
    df_pairs["is_worst20pct"] = False
    for i in worst_indices:
        df_pairs.at[i, "is_worst20pct"] = True

    # Format for display
    df_display = df_pairs.copy()
    df_display["period_return"] = df_display["period_return"].map(lambda x: f"{x:+.2%}")
    print(df_display.to_string(index=False))

    # ──────────────────────────────────────────────────────────
    # Pool frequency distribution
    # ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("POOL FREQUENCY (how often each pool appears in holdings)")
    print(f"{'=' * 70}")

    pool_counts: Counter[str] = Counter()
    for pp in pool_pairs:
        if pp["pool1"]:
            pool_counts[pp["pool1"]] += 1
        if pp["pool2"]:
            pool_counts[pp["pool2"]] += 1

    total_slots = sum(pool_counts.values())
    for pool, cnt in pool_counts.most_common():
        print(f"  {pool:20s}: {cnt:3d} ({cnt / total_slots:.0%})")

    # ──────────────────────────────────────────────────────────
    # Same-pool pair breakdown
    # ──────────────────────────────────────────────────────────
    if same_pool_count > 0:
        print(f"\n{'=' * 70}")
        print(f"SAME-POOL PAIRS ({same_pool_count} occurrences)")
        print(f"{'=' * 70}")
        same_df = df_pairs[df_pairs["same_pool"]]
        print(same_df[["date", "etf1", "etf2", "pool1", "period_return", "is_worst20pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
