#!/usr/bin/env python3
"""
路径 C: 封板 v3.4 双策略在不同成本/执行情景下的真实净表现审计
================================================================================
仅跑 2 个封板策略 × 5 个情景，输出对照表直接决策。

情景矩阵:
  1) COC + flat 2bp          (封板基线复现)
  2) T1_OPEN + flat 2bp      (只看执行价差)
  3) T1_OPEN + low (10/30)   (乐观成本)
  4) T1_OPEN + med (20/50)   (基准成本)
  5) T1_OPEN + high (30/80)  (悲观成本)

用法:
    uv run python scripts/run_sealed_cost_audit.py
"""

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import warnings

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import shift_timing_signal
from etf_strategy.regime_gate import compute_regime_gate_arr, gate_stats
from etf_strategy.core.cost_model import CostModel, build_cost_array
from etf_strategy.core.frozen_params import FrozenETFPool
from batch_vec_backtest import run_vec_backtest


# ─── Sealed v3.4 strategies ───
SEALED_STRATEGIES = {
    "S1 (4F)": "ADX_14D + OBV_SLOPE_10D + SHARPE_RATIO_20D + SLOPE_20D",
    "S2 (5F)": "ADX_14D + OBV_SLOPE_10D + PRICE_POSITION_120D + SHARPE_RATIO_20D + SLOPE_20D",
}

# ─── Cost/execution scenarios ───
SCENARIOS = [
    # (name, use_t1_open, cost_mode, cost_rate_or_tier)
    ("COC@2bp",        False, "uniform", 0.0002),
    ("T1_OPEN@2bp",    True,  "uniform", 0.0002),
    ("T1_OPEN@low",    True,  "split",   "low"),
    ("T1_OPEN@med",    True,  "split",   "med"),
    ("T1_OPEN@high",   True,  "split",   "high"),
]

COST_TIERS = {
    "low":  (0.0010, 0.0030),
    "med":  (0.0020, 0.0050),
    "high": (0.0030, 0.0080),
}


def build_cost_arr_for_scenario(cost_mode, cost_rate_or_tier, etf_codes, qdii_set):
    """Build cost_arr for a given scenario."""
    N = len(etf_codes)
    if cost_mode == "uniform":
        return np.full(N, cost_rate_or_tier, dtype=np.float64)
    else:
        a_share, qdii = COST_TIERS[cost_rate_or_tier]
        arr = np.empty(N, dtype=np.float64)
        for i, code in enumerate(etf_codes):
            arr[i] = qdii if code in qdii_set else a_share
        return arr


def main():
    print("=" * 100)
    print("🔍 路径 C: 封板 v3.4 双策略 × 5 情景 真实净表现审计")
    print("=" * 100)

    # ─── Load config ───
    config_path = ROOT / "configs" / "combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    backtest_config = config.get("backtest", {})

    # ─── Load data (full period, not training-only) ───
    print("\n📊 Loading data (full period)...")
    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )

    # ─── Compute factors ───
    print("📊 Computing factors...")
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)
    factor_names_list = sorted(raw_factors_df.columns.get_level_values(0).unique().tolist())
    raw_factors = {fname: raw_factors_df[fname] for fname in factor_names_list}

    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)

    # ─── Prepare arrays ───
    first_factor = std_factors[factor_names_list[0]]
    dates = first_factor.index
    etf_codes = first_factor.columns.tolist()
    factor_index_map = {name: idx for idx, name in enumerate(factor_names_list)}

    all_factors_stack = np.stack(
        [std_factors[f].values for f in factor_names_list], axis=-1
    )

    close_prices = ohlcv["close"][etf_codes].ffill().bfill().values
    open_prices = ohlcv["open"][etf_codes].ffill().bfill().values
    high_prices = ohlcv["high"][etf_codes].ffill().bfill().values
    low_prices = ohlcv["low"][etf_codes].ffill().bfill().values

    # ─── Timing + regime gate ───
    timing_module = LightTimingModule(
        extreme_threshold=-0.1, extreme_position=0.1,
    )
    timing_series = timing_module.compute_position_ratios(ohlcv["close"])
    timing_arr_raw = timing_series.reindex(dates).fillna(1.0).values
    timing_arr = shift_timing_signal(timing_arr_raw)

    gate_arr = compute_regime_gate_arr(
        ohlcv["close"], dates, backtest_config=backtest_config
    )
    timing_arr = (timing_arr.astype(np.float64) * gate_arr.astype(np.float64)).astype(np.float64)

    qdii_set = set(FrozenETFPool().qdii_codes)
    N = len(etf_codes)

    # ─── Holdout split ───
    training_end = config["data"].get("training_end_date", "2025-04-30")
    holdout_start_idx = 0
    for i, d in enumerate(dates):
        if str(d)[:10] > training_end:
            holdout_start_idx = i
            break
    total_days = len(dates)
    holdout_days = total_days - holdout_start_idx
    holdout_years = holdout_days / 252.0

    print(f"  Total days: {total_days}, Holdout start idx: {holdout_start_idx} ({training_end})")
    print(f"  Holdout days: {holdout_days} (~{holdout_years:.1f} years)")
    print(f"  ETFs: {N}, Factors: {len(factor_names_list)}, QDII: {len(qdii_set)}")

    # ─── Run all strategy × scenario combinations ───
    results = []

    for strat_name, combo_str in SEALED_STRATEGIES.items():
        factors_in_combo = [f.strip() for f in combo_str.split(" + ")]
        try:
            combo_indices = [factor_index_map[f] for f in factors_in_combo]
        except KeyError as e:
            print(f"[ERROR] {strat_name}: unknown factor {e}")
            continue

        current_factors = all_factors_stack[..., combo_indices]
        current_factor_indices = list(range(len(combo_indices)))

        for scenario_name, use_t1_open, cost_mode, cost_param in SCENARIOS:
            cost_arr = build_cost_arr_for_scenario(cost_mode, cost_param, etf_codes, qdii_set)

            try:
                equity_curve, total_return, win_rate, profit_factor, num_trades, _, risk = run_vec_backtest(
                    current_factors,
                    close_prices,
                    open_prices,
                    high_prices,
                    low_prices,
                    timing_arr,
                    current_factor_indices,
                    freq=3,
                    pos_size=2,
                    initial_capital=float(backtest_config["initial_capital"]),
                    commission_rate=float(backtest_config["commission_rate"]),
                    lookback=int(backtest_config["lookback"]),
                    cost_arr=cost_arr,
                    trailing_stop_pct=0.0,
                    stop_on_rebalance_only=True,
                    use_t1_open=use_t1_open,
                )

                # Compute holdout metrics from equity curve
                if holdout_start_idx < len(equity_curve):
                    holdout_equity = equity_curve[holdout_start_idx:]
                    if len(holdout_equity) > 1 and holdout_equity[0] > 0:
                        holdout_ret = (holdout_equity[-1] / holdout_equity[0]) - 1.0
                        peak = holdout_equity[0]
                        max_dd = 0.0
                        for v in holdout_equity:
                            if v > peak:
                                peak = v
                            dd = (peak - v) / peak
                            if dd > max_dd:
                                max_dd = dd
                        holdout_ann_ret = holdout_ret / holdout_years if holdout_years > 0 else 0.0
                    else:
                        holdout_ret = 0.0
                        max_dd = 0.0
                        holdout_ann_ret = 0.0
                else:
                    holdout_ret = 0.0
                    max_dd = 0.0
                    holdout_ann_ret = 0.0

                results.append({
                    "strategy": strat_name,
                    "scenario": scenario_name,
                    "total_return": total_return,
                    "holdout_return": holdout_ret,
                    "holdout_ann_ret": holdout_ann_ret,
                    "sharpe": risk["sharpe_ratio"],
                    "max_drawdown": risk["max_drawdown"],
                    "calmar": risk["calmar_ratio"],
                    "turnover_ann": risk.get("turnover_ann", 0.0),
                    "cost_drag": risk.get("cost_drag", 0.0),
                    "trades": num_trades,
                })

            except Exception as e:
                print(f"[ERROR] {strat_name} × {scenario_name}: {e}")
                results.append({
                    "strategy": strat_name,
                    "scenario": scenario_name,
                    "total_return": float("nan"),
                    "holdout_return": float("nan"),
                    "holdout_ann_ret": float("nan"),
                    "sharpe": float("nan"),
                    "max_drawdown": float("nan"),
                    "calmar": float("nan"),
                    "turnover_ann": float("nan"),
                    "cost_drag": float("nan"),
                    "trades": 0,
                })

    # ─── Print comparison table ───
    print("\n" + "=" * 130)
    print("📊 路径 C: 封板 v3.4 双策略 × 5 情景 对照表")
    print("=" * 130)

    header = (
        f"{'Strategy':<10} | {'Scenario':<16} | {'Total Ret':>10} | {'Holdout':>10} | "
        f"{'HO Ann':>8} | {'Sharpe':>7} | {'MDD':>7} | {'Calmar':>7} | "
        f"{'Turnover':>9} | {'CostDrag':>9} | {'Trades':>6}"
    )
    print(header)
    print("-" * 130)

    for r in results:
        total_ret_str = f"{r['total_return']*100:>9.2f}%" if not np.isnan(r['total_return']) else "     N/A"
        holdout_str = f"{r['holdout_return']*100:>9.2f}%" if not np.isnan(r['holdout_return']) else "     N/A"
        ho_ann_str = f"{r['holdout_ann_ret']*100:>7.1f}%" if not np.isnan(r['holdout_ann_ret']) else "    N/A"
        sharpe_str = f"{r['sharpe']:>7.3f}" if not np.isnan(r['sharpe']) else "    N/A"
        mdd_str = f"{r['max_drawdown']*100:>6.2f}%" if not np.isnan(r['max_drawdown']) else "   N/A"
        calmar_str = f"{r['calmar']:>7.3f}" if not np.isnan(r['calmar']) else "    N/A"
        turnover_str = f"{r['turnover_ann']:>8.1f}x" if not np.isnan(r['turnover_ann']) else "     N/A"
        cost_drag_str = f"{r['cost_drag']*100:>8.2f}%" if not np.isnan(r['cost_drag']) else "     N/A"

        print(
            f"{r['strategy']:<10} | {r['scenario']:<16} | {total_ret_str} | {holdout_str} | "
            f"{ho_ann_str} | {sharpe_str} | {mdd_str} | {calmar_str} | "
            f"{turnover_str} | {cost_drag_str} | {r['trades']:>6}"
        )

        # Separator between scenarios of different strategies
        if r['scenario'] == SCENARIOS[-1][0] and r['strategy'] != list(SEALED_STRATEGIES.keys())[-1]:
            print("-" * 130)

    # ─── Delta analysis ───
    print("\n" + "=" * 130)
    print("📉 退化分析 (相对 COC@2bp 基线)")
    print("=" * 130)

    for strat_name in SEALED_STRATEGIES:
        baseline = [r for r in results if r["strategy"] == strat_name and r["scenario"] == "COC@2bp"]
        if not baseline:
            continue
        base = baseline[0]

        print(f"\n{strat_name}: COC@2bp baseline = {base['total_return']*100:.2f}% total, {base['holdout_return']*100:.2f}% holdout")
        print(f"  {'Scenario':<16} | {'ΔTotal':>10} | {'ΔHoldout':>10} | {'Turnover':>9} | {'CostDrag':>9}")
        print(f"  {'-'*70}")

        for r in results:
            if r["strategy"] != strat_name or r["scenario"] == "COC@2bp":
                continue
            d_total = (r["total_return"] - base["total_return"]) * 100
            d_holdout = (r["holdout_return"] - base["holdout_return"]) * 100
            turnover_str = f"{r['turnover_ann']:.1f}x" if not np.isnan(r['turnover_ann']) else "N/A"
            cost_drag_str = f"{r['cost_drag']*100:.2f}%" if not np.isnan(r['cost_drag']) else "N/A"
            print(
                f"  {r['scenario']:<16} | {d_total:>+9.2f}pp | {d_holdout:>+9.2f}pp | "
                f"{turnover_str:>9} | {cost_drag_str:>9}"
            )

    # ─── Decision helper ───
    print("\n" + "=" * 130)
    print("🎯 决策参考")
    print("=" * 130)

    for strat_name in SEALED_STRATEGIES:
        med_results = [r for r in results if r["strategy"] == strat_name and r["scenario"] == "T1_OPEN@med"]
        if not med_results:
            continue
        r = med_results[0]
        cd = r["cost_drag"]
        ho = r["holdout_return"]

        print(f"\n{strat_name} @ T1_OPEN@med:")
        print(f"  Holdout: {ho*100:.2f}%, Cost Drag: {cd*100:.2f}%, Turnover: {r['turnover_ann']:.1f}x")

        if cd > 0.30:
            print(f"  ⚠️  Cost drag > 30% → Exp4 (降换手) 优先级: 高")
        elif cd > 0.20:
            print(f"  ⚡ Cost drag 20-30% → Exp4 作为稳健性增强 (非救命)")
        else:
            print(f"  ✅ Cost drag < 20% → 策略在 med 成本下健康")

        if ho > 0.03:
            print(f"  ✅ Holdout > 3% → 策略在 med 成本下仍有正 alpha")
        elif ho > 0:
            print(f"  ⚡ Holdout 正但 <3% → 边缘情况，建议 Exp4 巩固")
        else:
            print(f"  ⚠️  Holdout ≤ 0 → 策略在 med 成本下不可行")

    # ─── Save results ───
    df = pd.DataFrame(results)
    out_path = ROOT / "results" / "sealed_v34_cost_audit.csv"
    df.to_csv(out_path, index=False)
    print(f"\n💾 Results saved to: {out_path}")


if __name__ == "__main__":
    main()
