"""Select extended candidate strategies beyond the final Top-5.

Purpose
-------
The current pipeline produced 5 strict "Gold Standard" candidates.
For deeper analysis, it's often useful to export a larger candidate set (e.g. Top-50/Top-200)
using relaxed but still disciplined gates.

This script reads:
- Rolling OOS summary (train-only gate inputs)
- Holdout validation results (unseen period metrics + trailing window metrics)

and outputs:
- extended_candidates.parquet / .csv
- a short Markdown report summarizing the selection

Example
-------
uv run python scripts/select_extended_candidates.py \
  --rolling results/rolling_oos_consistency_20251215_005920/rolling_oos_summary.parquet \
  --holdout results/holdout_validation_20251215_010103/holdout_validation_results.parquet \
  --outdir results/diagnostics \
  --top-k 200 \
  --min-roll-pos-rate 0.55 \
  --min-roll-worst-return -0.10 \
  --min-roll-calmar 0.6 \
  --min-holdout-return 0.0

Notes
-----
- This does NOT rerun WFO/VEC. It is a post-selection/export helper.
- Keep the holdout truly "unseen": don't iterate thresholds repeatedly on the same holdout
  if you intend to claim it as a clean OOS verification.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Gates:
    min_roll_pos_rate: float
    min_roll_worst_return: float
    min_roll_calmar: float
    min_holdout_return: float


def _pct(x: float) -> str:
    if x != x:
        return "nan"
    return f"{x * 100:.2f}%"


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rolling", type=Path, required=True)
    ap.add_argument("--holdout", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("results/diagnostics"))
    ap.add_argument("--top-k", type=int, default=200)

    # Rolling gates (train-only)
    ap.add_argument("--min-roll-pos-rate", type=float, default=0.55)
    ap.add_argument("--min-roll-worst-return", type=float, default=-0.10)
    ap.add_argument("--min-roll-calmar", type=float, default=0.6)

    # Holdout gate
    ap.add_argument("--min-holdout-return", type=float, default=0.0)

    # Scoring weights (simple, interpretable)
    ap.add_argument("--w-train-calmar", type=float, default=0.30)
    ap.add_argument("--w-roll-worst", type=float, default=0.40)
    ap.add_argument("--w-holdout-calmar", type=float, default=0.30)

    # Extra stress window (optional): use holdout trailing 63d to reflect recent regime
    ap.add_argument("--use-holdout-trail-63d-as-penalty", action="store_true")
    ap.add_argument("--w-penalty-trail-63d-return", type=float, default=0.20)

    return ap


def main() -> None:
    args = build_argparser().parse_args()

    roll = pd.read_parquet(args.rolling)
    hold = pd.read_parquet(args.holdout)

    if "combo" not in roll.columns or "combo" not in hold.columns:
        raise SystemExit("Both rolling and holdout files must contain 'combo' column")

    # Join on combo
    df = roll.merge(
        hold,
        on="combo",
        how="inner",
        suffixes=("__roll", "__hold"),
    )

    required_roll_cols = [
        "all_segment_positive_rate",
        "all_segment_worst_return",
        "full_calmar_ratio",
    ]
    for c in required_roll_cols:
        if c not in df.columns:
            raise SystemExit(f"Missing required rolling column: {c}")

    if "holdout_return" not in df.columns or "holdout_calmar_ratio" not in df.columns:
        raise SystemExit(
            "Missing required holdout columns: holdout_return / holdout_calmar_ratio"
        )

    gates = Gates(
        min_roll_pos_rate=float(args.min_roll_pos_rate),
        min_roll_worst_return=float(args.min_roll_worst_return),
        min_roll_calmar=float(args.min_roll_calmar),
        min_holdout_return=float(args.min_holdout_return),
    )

    m = (
        (df["all_segment_positive_rate"] >= gates.min_roll_pos_rate)
        & (df["all_segment_worst_return"] >= gates.min_roll_worst_return)
        & (df["full_calmar_ratio"] >= gates.min_roll_calmar)
        & (df["holdout_return"] >= gates.min_holdout_return)
    )

    filtered = df.loc[m].copy()

    # Score: similar spirit to triple validation, but generalized
    score = (
        float(args.w_train_calmar) * filtered["vec_calmar_ratio__hold"]
        + float(args.w_roll_worst) * filtered["all_segment_worst_return"]
        + float(args.w_holdout_calmar) * filtered["holdout_calmar_ratio"]
    )

    if args.use_holdout_trail_63d_as_penalty:
        col = "holdout_trail_63d_return"
        if col in filtered.columns:
            # penalty: subtract when recent return is negative
            penalty = (
                float(args.w_penalty_trail_63d_return)
                * filtered[col].clip(upper=0.0).abs()
            )
            score = score - penalty
            filtered["penalty_holdout_trail_63d_return"] = penalty
        else:
            raise SystemExit(
                f"--use-holdout-trail-63d-as-penalty requires column {col}"
            )

    filtered["extended_score"] = score

    # Sort and export
    filtered = filtered.sort_values("extended_score", ascending=False)
    top_k = int(args.top_k)
    out = filtered.head(top_k).copy()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) / f"extended_candidates_{top_k}_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)

    # Keep a focused column set first; keep full join too for advanced analysis.
    focus_cols = [
        "combo",
        "extended_score",
        # train (vec, train-only)
        "vec_return__hold",
        "vec_calmar_ratio__hold",
        "vec_sharpe_ratio__hold",
        "vec_max_drawdown__hold",
        # rolling (train-only summary)
        "all_segment_positive_rate",
        "all_segment_worst_return",
        "full_calmar_ratio",
        "full_sharpe_ratio",
        "full_max_drawdown",
        # holdout
        "holdout_return",
        "holdout_calmar_ratio",
        "holdout_sharpe_ratio",
        "holdout_max_drawdown",
    ]

    # Optional regime stress columns
    for c in [
        "holdout_trail_21d_return",
        "holdout_trail_63d_return",
        "holdout_trail_120d_return",
    ]:
        if c in out.columns:
            focus_cols.append(c)

    for c in ["penalty_holdout_trail_63d_return"]:
        if c in out.columns and c not in focus_cols:
            focus_cols.append(c)

    focus = out[[c for c in focus_cols if c in out.columns]].copy()

    focus.to_parquet(outdir / "extended_candidates.parquet", index=False)
    focus.to_csv(outdir / "extended_candidates.csv", index=False)

    # Full join (for deeper analysis)
    out.to_parquet(outdir / "extended_candidates_full.parquet", index=False)

    # Report
    report = []
    report.append("# 扩展候选策略导出报告\n")
    report.append(f"- rolling: {args.rolling}")
    report.append(f"- holdout: {args.holdout}")
    report.append(f"- join rows: {len(df)}")
    report.append(f"- passed gates: {len(filtered)}")
    report.append(f"- exported Top-K: {len(focus)}\n")

    report.append("## Gates（门槛）\n")
    report.append(f"- Rolling pos_rate >= {gates.min_roll_pos_rate}")
    report.append(f"- Rolling worst_return >= {gates.min_roll_worst_return}")
    report.append(f"- Rolling full_calmar >= {gates.min_roll_calmar}")
    report.append(f"- Holdout return >= {gates.min_holdout_return}\n")

    if len(filtered) > 0:
        report.append("## 分布概览（通过 gates 的集合）\n")
        report.append(
            f"- holdout_return mean/median: {_pct(float(filtered['holdout_return'].mean()))} / {_pct(float(filtered['holdout_return'].median()))}"
        )
        report.append(
            f"- all_segment_worst_return mean/median: {_pct(float(filtered['all_segment_worst_return'].mean()))} / {_pct(float(filtered['all_segment_worst_return'].median()))}\n"
        )

    report.append("## Top 10（按 extended_score）\n")
    top10 = focus.head(10)
    for _, r in top10.iterrows():
        report.append(
            f"- {r['combo']} | score={float(r['extended_score']):.4f} | holdout={_pct(float(r['holdout_return']))} | roll_worst={_pct(float(r['all_segment_worst_return']))} | roll_pos={float(r['all_segment_positive_rate']):.2f}"
        )

    (outdir / "report.md").write_text("\n".join(report), encoding="utf-8")

    print(f"[OK] wrote: {outdir}")


if __name__ == "__main__":
    main()
