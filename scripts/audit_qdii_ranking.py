#!/usr/bin/env python3
"""
QDII 因子排名审计：回答"为什么实盘期没选到 QDII？"
==========================================================
对封板 S1/S2 策略，在每个调仓日输出全部 43 只 ETF 的综合得分排名，
重点标注 5 只 QDII ETF 的排名、得分及各因子得分明细。

用法:
    uv run python scripts/audit_qdii_ranking.py [--start 2025-11-01] [--end 2026-02-09]
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import warnings

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import (
    shift_timing_signal,
    generate_rebalance_schedule,
)
from etf_strategy.regime_gate import compute_regime_gate_arr
from etf_strategy.core.frozen_params import FrozenETFPool

# ─── Sealed v3.4 strategies ───
SEALED_STRATEGIES = {
    "S1 (4F)": ["ADX_14D", "OBV_SLOPE_10D", "SHARPE_RATIO_20D", "SLOPE_20D"],
    "S2 (5F)": ["ADX_14D", "OBV_SLOPE_10D", "PRICE_POSITION_120D", "SHARPE_RATIO_20D", "SLOPE_20D"],
}

QDII_NAMES = {
    "513100": "纳指100",
    "513500": "标普500",
    "159920": "恒生ETF",
    "513050": "中概互联",
    "513130": "恒生科技",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-11-01", help="审计起始日期")
    parser.add_argument("--end", default="2026-02-09", help="审计结束日期")
    args = parser.parse_args()

    print("=" * 120)
    print("🔍 QDII 因子排名审计：每个调仓日的完整排名")
    print("=" * 120)

    # ─── Load config ───
    config_path = ROOT / "configs" / "combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    backtest_config = config.get("backtest", {})

    # ─── Load data ───
    print(f"\n📊 Loading data ({args.start} ~ {args.end})...")
    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=args.end,
    )

    # ─── Compute factors ───
    print("📊 Computing factors...")
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)
    factor_names_list = sorted(raw_factors_df.columns.get_level_values(0).unique().tolist())
    raw_factors = {fname: raw_factors_df[fname] for fname in factor_names_list}

    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)

    # ─── Prepare ───
    first_factor = std_factors[factor_names_list[0]]
    dates = first_factor.index
    etf_codes = first_factor.columns.tolist()
    qdii_set = set(FrozenETFPool().qdii_codes)

    # ─── Timing + regime gate ───
    timing_module = LightTimingModule(extreme_threshold=-0.1, extreme_position=0.1)
    timing_series = timing_module.compute_position_ratios(ohlcv["close"])
    timing_arr_raw = timing_series.reindex(dates).fillna(1.0).values
    timing_arr = shift_timing_signal(timing_arr_raw)

    gate_arr = compute_regime_gate_arr(
        ohlcv["close"], dates, backtest_config=backtest_config
    )
    timing_arr = (timing_arr.astype(np.float64) * gate_arr.astype(np.float64)).astype(np.float64)

    # ─── Rebalance schedule ───
    lookback = int(backtest_config.get("lookback", 252))
    freq = int(backtest_config.get("freq", 3))
    total_periods = len(dates)
    rebalance_days = generate_rebalance_schedule(total_periods, lookback, freq)

    # Filter to audit window
    start_idx = 0
    for i, d in enumerate(dates):
        if str(d)[:10] >= args.start:
            start_idx = i
            break

    audit_rebalance_days = [d for d in rebalance_days if d >= start_idx]
    print(f"\n  ETFs: {len(etf_codes)}, QDII: {len(qdii_set)}")
    print(f"  Audit window: {args.start} ~ {args.end}")
    print(f"  Rebalance days in window: {len(audit_rebalance_days)}")

    # ─── Audit each rebalance day ───
    all_rows = []

    for strat_name, factor_list in SEALED_STRATEGIES.items():
        print(f"\n{'='*120}")
        print(f"📊 策略: {strat_name} ({' + '.join(factor_list)})")
        print(f"{'='*120}")

        for rb_idx in audit_rebalance_days:
            if rb_idx >= len(dates):
                continue

            date = dates[rb_idx]
            date_str = str(date)[:10]
            timing_val = timing_arr[rb_idx]

            # Compute composite score (same as VEC kernel: sum of standardized factor scores)
            scores = np.zeros(len(etf_codes))
            factor_detail = {}

            for f_name in factor_list:
                if f_name not in std_factors:
                    continue
                f_vals = std_factors[f_name].loc[date].values
                factor_detail[f_name] = f_vals.copy()
                # NaN handling: treat as 0 contribution
                f_vals_clean = np.where(np.isfinite(f_vals), f_vals, 0.0)
                scores += f_vals_clean

            # Rank by score (descending)
            sorted_indices = np.argsort(-scores)

            print(f"\n  📅 {date_str}  (timing={timing_val:.3f})")
            print(f"  {'Rank':<5} {'Code':<10} {'Name':<14} {'Score':>8}", end="")
            for f_name in factor_list:
                print(f" | {f_name[:12]:>12}", end="")
            print()
            print(f"  {'-'*100}")

            # Print top 10 + all QDII
            printed = set()
            qdii_ranks = {}

            # First pass: find QDII ranks
            for rank, idx in enumerate(sorted_indices, 1):
                code = etf_codes[idx]
                if code in qdii_set:
                    qdii_ranks[code] = rank

            # Print top 10
            for rank, idx in enumerate(sorted_indices[:10], 1):
                code = etf_codes[idx]
                is_qdii = code in qdii_set
                marker = "⭐" if is_qdii else "  "
                name = QDII_NAMES.get(code, "")
                print(f"  {marker}{rank:<4} {code:<10} {name:<14} {scores[idx]:>8.3f}", end="")
                for f_name in factor_list:
                    v = factor_detail[f_name][idx]
                    print(f" | {v:>12.3f}" if np.isfinite(v) else f" | {'NaN':>12}", end="")
                print()
                printed.add(code)

            # Print QDII not in top 10
            qdii_not_shown = [c for c in sorted(qdii_set) if c not in printed]
            if qdii_not_shown:
                print(f"  {'...'}")
                for code in qdii_not_shown:
                    idx_in_list = etf_codes.index(code)
                    rank = qdii_ranks.get(code, -1)
                    name = QDII_NAMES.get(code, "")
                    print(f"  ⭐{rank:<4} {code:<10} {name:<14} {scores[idx_in_list]:>8.3f}", end="")
                    for f_name in factor_list:
                        v = factor_detail[f_name][idx_in_list]
                        print(f" | {v:>12.3f}" if np.isfinite(v) else f" | {'NaN':>12}", end="")
                    print()

            # Collect row for summary
            top2_codes = [etf_codes[sorted_indices[i]] for i in range(min(2, len(sorted_indices)))]
            top2_scores = [scores[sorted_indices[i]] for i in range(min(2, len(sorted_indices)))]
            best_qdii_code = min(qdii_ranks, key=qdii_ranks.get) if qdii_ranks else ""
            best_qdii_rank = min(qdii_ranks.values()) if qdii_ranks else -1

            all_rows.append({
                "strategy": strat_name,
                "date": date_str,
                "timing": timing_val,
                "pick1": top2_codes[0] if len(top2_codes) > 0 else "",
                "pick1_score": top2_scores[0] if len(top2_scores) > 0 else np.nan,
                "pick2": top2_codes[1] if len(top2_codes) > 1 else "",
                "pick2_score": top2_scores[1] if len(top2_scores) > 1 else np.nan,
                "best_qdii": best_qdii_code,
                "best_qdii_rank": best_qdii_rank,
                "best_qdii_score": scores[etf_codes.index(best_qdii_code)] if best_qdii_code else np.nan,
                "score_gap": top2_scores[1] - scores[etf_codes.index(best_qdii_code)] if best_qdii_code and len(top2_scores) > 1 else np.nan,
            })

    # ─── Summary table ───
    print("\n" + "=" * 120)
    print("📊 QDII 排名汇总")
    print("=" * 120)
    print(f"{'Strategy':<10} | {'Date':<12} | {'Timing':>7} | {'Pick1':<8} | {'Pick2':<8} | "
          f"{'BestQDII':<10} {'Rank':>4} {'Score':>7} | {'Gap':>7}")
    print("-" * 120)

    for r in all_rows:
        qdii_name = QDII_NAMES.get(r['best_qdii'], r['best_qdii'])
        print(f"{r['strategy']:<10} | {r['date']:<12} | {r['timing']:>7.3f} | "
              f"{r['pick1']:<8} | {r['pick2']:<8} | "
              f"{qdii_name:<10} {r['best_qdii_rank']:>4} {r['best_qdii_score']:>7.3f} | "
              f"{r['score_gap']:>+7.3f}")

    # ─── Save ───
    df = pd.DataFrame(all_rows)
    out_path = ROOT / "results" / "qdii_ranking_audit.csv"
    df.to_csv(out_path, index=False)
    print(f"\n💾 Results saved to: {out_path}")

    # ─── Verdict ───
    avg_best_qdii_rank = df["best_qdii_rank"].mean()
    avg_gap = df["score_gap"].mean()
    pct_top10 = (df["best_qdii_rank"] <= 10).mean() * 100

    print("\n" + "=" * 120)
    print("🎯 诊断结论")
    print("=" * 120)
    print(f"  平均最佳 QDII 排名: {avg_best_qdii_rank:.1f} / {len(etf_codes)}")
    print(f"  QDII 进入 Top10 的比例: {pct_top10:.1f}%")
    print(f"  平均得分差距 (Pick2 vs BestQDII): {avg_gap:+.3f}")

    if avg_best_qdii_rank <= 5:
        print("  ✅ QDII 排名靠前但被选中概率高 → 可能存在 bug")
    elif avg_best_qdii_rank <= 15:
        print("  ⚡ QDII 处于中间位置 → 市场轮动中，可能很快切回")
    else:
        print("  📉 QDII 排名靠后 → 市场环境不利，策略正确地回避了 QDII")

    if avg_gap > 1.0:
        print(f"  📉 得分差距大 ({avg_gap:.2f}) → QDII 在当前因子模型下完全不具竞争力")
    elif avg_gap > 0.3:
        print(f"  ⚡ 得分差距中等 ({avg_gap:.2f}) → QDII 有机会被选中但需市场转向")
    else:
        print(f"  🔥 得分差距小 ({avg_gap:.2f}) → QDII 接近入选门槛，市场微调即可触发")


if __name__ == "__main__":
    main()
