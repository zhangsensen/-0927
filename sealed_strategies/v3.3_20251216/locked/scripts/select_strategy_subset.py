#!/usr/bin/env python3
"""从 Top-N 策略中选出 3~4 个“更分散、更稳健”的子集（研究辅助工具）。

说明：
- 本脚本只做“策略集合的组合优化”，不基于盘中涨跌做判断。
- 输出的是一套可复核的评分规则下的最优子集候选，用于你人工决策。

评分（可调）：
- quality：子集内策略 prod_score_bt 的均值（已综合 BT holdout 收益/回撤等）
- div_penalty_factors：子集内两两因子集合的平均 Jaccard 相似度（越高越重复）
- div_penalty_picks：子集内两两“今日选股集合”的平均 Jaccard 相似度

总分：quality - w_factors*div_penalty_factors - w_picks*div_penalty_picks

用法：
  uv run python scripts/select_strategy_subset.py \
    --candidates results/production_pack_20251215_031920/production_candidates.parquet \
    --signals results/today_signal_20251215/signals_per_strategy.parquet \
    --k 3

  uv run python scripts/select_strategy_subset.py ... --k 4
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--candidates",
        type=str,
        required=True,
        help="production_candidates.parquet (Top6/TopN)",
    )
    p.add_argument(
        "--signals", type=str, required=True, help="signals_per_strategy.parquet"
    )
    p.add_argument("--k", type=int, choices=[3, 4], required=True, help="选择子集大小")
    p.add_argument("--top", type=int, default=5, help="输出前 N 个子集")
    p.add_argument("--w-factors", type=float, default=0.30, help="因子重复惩罚权重")
    p.add_argument("--w-picks", type=float, default=0.20, help="当日选股重复惩罚权重")
    p.add_argument("--out", type=str, default="", help="输出 markdown 路径")
    return p.parse_args()


def _combo_to_factor_set(combo: str) -> frozenset[str]:
    return frozenset([s.strip() for s in str(combo).split("+") if s.strip()])


def _row_to_pick_set(row: pd.Series) -> frozenset[str]:
    picks = []
    for c in ["pick1", "pick2"]:
        v = str(row.get(c, "") or "").strip()
        if v:
            picks.append(v)
    return frozenset(picks)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _avg_pairwise_jaccard(sets: list[frozenset[str]]) -> float:
    if len(sets) <= 1:
        return 0.0
    vals = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            vals.append(_jaccard(sets[i], sets[j]))
    return float(np.mean(vals)) if vals else 0.0


@dataclass(frozen=True)
class StrategyInfo:
    combo: str
    prod_score_bt: float
    bt_holdout_return: float | None
    bt_win_rate: float | None
    bt_max_drawdown: float | None
    factors: frozenset[str]
    picks: frozenset[str]


def main() -> int:
    args = _parse_args()

    df = pd.read_parquet(Path(args.candidates))
    sig = pd.read_parquet(Path(args.signals))

    if "combo" not in df.columns:
        raise ValueError("candidates 缺少 combo 列")
    if "combo" not in sig.columns:
        raise ValueError("signals 缺少 combo 列")

    # join signals -> candidates
    merged = df.merge(sig, on="combo", how="left", suffixes=("", "_sig"))

    need = ["combo", "prod_score_bt"]
    for c in need:
        if c not in merged.columns:
            raise ValueError(f"缺少必需列: {c}")

    strategies: list[StrategyInfo] = []
    for _, r in merged.iterrows():
        strategies.append(
            StrategyInfo(
                combo=str(r["combo"]),
                prod_score_bt=float(r["prod_score_bt"]),
                bt_holdout_return=(
                    float(r["bt_holdout_return"])
                    if "bt_holdout_return" in merged.columns
                    else None
                ),
                bt_win_rate=(
                    float(r["bt_win_rate"]) if "bt_win_rate" in merged.columns else None
                ),
                bt_max_drawdown=(
                    float(r["bt_max_drawdown"])
                    if "bt_max_drawdown" in merged.columns
                    else None
                ),
                factors=_combo_to_factor_set(r["combo"]),
                picks=_row_to_pick_set(r),
            )
        )

    # enumerate subsets
    rows = []
    for subset in itertools.combinations(strategies, args.k):
        quality = float(np.mean([s.prod_score_bt for s in subset]))
        div_f = _avg_pairwise_jaccard([s.factors for s in subset])
        div_p = _avg_pairwise_jaccard([s.picks for s in subset])
        total = quality - args.w_factors * div_f - args.w_picks * div_p

        # aggregation concentration: pick frequency
        pick_counts: dict[str, int] = {}
        for s in subset:
            for t in s.picks:
                pick_counts[t] = pick_counts.get(t, 0) + 1
        top_pick = (
            max(pick_counts.items(), key=lambda kv: kv[1]) if pick_counts else ("", 0)
        )

        rows.append(
            {
                "total_score": total,
                "quality": quality,
                "avg_jaccard_factors": div_f,
                "avg_jaccard_picks": div_p,
                "top_pick": top_pick[0],
                "top_pick_count": top_pick[1],
                "combos": " | ".join([s.combo for s in subset]),
            }
        )

    res = (
        pd.DataFrame(rows)
        .sort_values("total_score", ascending=False)
        .reset_index(drop=True)
    )

    out_path = (
        Path(args.out)
        if args.out
        else (
            ROOT
            / "results"
            / f"today_signal_20251215"
            / f"STRATEGY_SUBSET_K{args.k}.md"
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    head = res.head(args.top)

    md = []
    md.append(f"# 策略子集选择（k={args.k}）\n")
    md.append(
        "说明：基于 BT 生产评分 + 去重惩罚的枚举结果（研究辅助，不构成投资建议）。\n"
    )
    md.append(f"- candidates: {args.candidates}")
    md.append(f"- signals: {args.signals}")
    md.append(
        f"- score = quality - w_factors*avg_jaccard_factors - w_picks*avg_jaccard_picks"
    )
    md.append(f"- w_factors={args.w_factors:.2f}, w_picks={args.w_picks:.2f}\n")

    md.append("## Top 子集（按总分排序）")
    md.append(head.to_markdown(index=False))

    out_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"✅ wrote: {out_path}")
    print(
        head[
            [
                "total_score",
                "quality",
                "avg_jaccard_factors",
                "avg_jaccard_picks",
                "top_pick",
                "top_pick_count",
            ]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
