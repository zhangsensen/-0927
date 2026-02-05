#!/usr/bin/env python3
"""Generate a clean shortlist report for the latest holdout8m run.

This reads the umbrella run directory created by scripts/run_holdout8m_pipeline.py
(e.g. results/holdout8m_run_YYYYMMDD_HHMMSS/manifest.yaml), merges metrics from:
- VEC (train)
- Rolling OOS (train-only)
- Holdout validation (unseen)
- Final triple candidates
- BT audit (event-driven)

Outputs (inside the umbrella run dir):
- candidates_merged.parquet
- candidates_merged.csv
- STRATEGY_SHORTLIST.md

Run:
  uv run python scripts/report_holdout8m_shortlist.py
  uv run python scripts/report_holdout8m_shortlist.py --run-dir results/holdout8m_run_20251215_024012
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Inputs:
    run_dir: Path
    vec_results: Path
    rolling_summary: Path
    holdout_results: Path
    final_candidates: Path


def _latest_run_dir() -> Path:
    base = ROOT / "results"
    dirs = sorted(
        [p for p in base.glob("holdout8m_run_*") if p.is_dir() and not p.is_symlink()]
    )
    if not dirs:
        raise FileNotFoundError("No results/holdout8m_run_* found")
    return dirs[-1]


def _load_inputs(run_dir: Path) -> Inputs:
    manifest_path = run_dir / "manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    art = manifest.get("artifacts", {})
    return Inputs(
        run_dir=run_dir,
        vec_results=Path(art["vec_results"]),
        rolling_summary=Path(art["rolling_summary"]),
        holdout_results=Path(art["holdout_results"]),
        final_candidates=Path(art["final_candidates"]),
    )


def _latest_bt_dir(after_ts: str | None = None) -> Path:
    base = ROOT / "results"
    dirs = sorted(
        [
            p
            for p in base.glob("bt_backtest_full_*")
            if p.is_dir() and not p.is_symlink()
        ]
    )
    if not dirs:
        raise FileNotFoundError("No results/bt_backtest_full_* found")
    if after_ts:
        dirs = [
            d
            for d in dirs
            if d.name.split("_")[-2] + "_" + d.name.split("_")[-1] >= after_ts
        ]
        if not dirs:
            return sorted(
                [
                    p
                    for p in base.glob("bt_backtest_full_*")
                    if p.is_dir() and not p.is_symlink()
                ]
            )[-1]
    return dirs[-1]


def _rank01(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    # stable rank percentile in [0,1]
    return s.rank(method="average", pct=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate shortlist report for holdout8m run"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Umbrella run dir (results/holdout8m_run_*)",
    )
    parser.add_argument(
        "--bt-dir",
        type=str,
        default=None,
        help="Optional: explicitly set BT output dir (results/bt_backtest_full_*)",
    )
    args = parser.parse_args()

    run_dir = (ROOT / args.run_dir) if args.run_dir else _latest_run_dir()
    inputs = _load_inputs(run_dir)

    # Best-effort pick the latest BT dir; typically it was just produced by the pipeline.
    bt_dir = (ROOT / args.bt_dir) if args.bt_dir else _latest_bt_dir()
    bt_results = bt_dir / "bt_results.parquet"
    if not bt_results.exists():
        raise FileNotFoundError(f"Missing BT results: {bt_results}")

    vec = pd.read_parquet(inputs.vec_results)
    roll = pd.read_parquet(inputs.rolling_summary)
    hold = pd.read_parquet(inputs.holdout_results)
    final = pd.read_parquet(inputs.final_candidates)
    bt = pd.read_parquet(bt_results)

    # Reduce to final candidates only (52 rows in current run)
    final_set = set(final["combo"].astype(str))
    vec = vec[vec["combo"].astype(str).isin(final_set)].copy()
    roll = roll[roll["combo"].astype(str).isin(final_set)].copy()
    hold = hold[hold["combo"].astype(str).isin(final_set)].copy()
    bt = bt[bt["combo"].astype(str).isin(final_set)].copy()

    m = vec.merge(roll, on="combo", how="inner")
    m = m.merge(hold, on="combo", how="inner")
    m = m.merge(bt, on="combo", how="inner")

    # Add factor list columns
    m["factor_count"] = (
        m["combo"].astype(str).apply(lambda x: len([p.strip() for p in x.split("+")]))
    )
    m["factors"] = (
        m["combo"]
        .astype(str)
        .apply(lambda x: ", ".join([p.strip() for p in x.split("+")]))
    )

    # Composite score: emphasize holdout + BT holdout, then robustness
    # (rank-based to be scale-invariant)
    score = 0.0
    # VEC (train)
    if "vec_calmar_ratio" in m.columns:
        score += 0.5 * _rank01(m["vec_calmar_ratio"])
    if "vec_sharpe_ratio" in m.columns:
        score += 0.5 * _rank01(m["vec_sharpe_ratio"])

    # Rolling robustness (train-only)
    if "all_segment_positive_rate" in m.columns:
        score += 1.0 * _rank01(m["all_segment_positive_rate"])
    if "all_segment_worst_return" in m.columns:
        score += 1.0 * _rank01(m["all_segment_worst_return"])

    # Holdout (vectorized)
    if "holdout_return" in m.columns:
        score += 2.0 * _rank01(m["holdout_return"])
    if "holdout_calmar_ratio" in m.columns:
        score += 1.0 * _rank01(m["holdout_calmar_ratio"])
    if "holdout_max_drawdown" in m.columns:
        score += 1.0 * _rank01(-m["holdout_max_drawdown"].astype(float))

    # BT (event-driven)
    if "bt_holdout_return" in m.columns:
        score += 2.0 * _rank01(m["bt_holdout_return"])
    if "bt_calmar_ratio" in m.columns:
        score += 1.0 * _rank01(m["bt_calmar_ratio"])
    if "bt_max_drawdown" in m.columns:
        score += 1.0 * _rank01(-m["bt_max_drawdown"].astype(float))

    m["score"] = score

    # Define a practical “BT OK” gate (can be tuned)
    m["bt_holdout_ok"] = m.get("bt_holdout_return", 0.0) > 0.0
    m["bt_calmar_ok"] = m.get("bt_calmar_ratio", 0.0) > 0.0
    m["bt_ok"] = m["bt_holdout_ok"] & m["bt_calmar_ok"]

    # Sort
    m = m.sort_values("score", ascending=False).reset_index(drop=True)

    out_parquet = run_dir / "candidates_merged.parquet"
    out_csv = run_dir / "candidates_merged.csv"
    m.to_parquet(out_parquet, index=False)
    m.to_csv(out_csv, index=False)

    # Render markdown report
    top_n = min(30, len(m))

    # Pick key columns for display (only those present)
    display_cols = [
        "combo",
        "score",
        "bt_ok",
        "vec_return",
        "vec_calmar_ratio",
        "vec_sharpe_ratio",
        "all_segment_positive_rate",
        "all_segment_worst_return",
        "holdout_return",
        "holdout_calmar_ratio",
        "holdout_max_drawdown",
        "bt_return",
        "bt_train_return",
        "bt_holdout_return",
        "bt_calmar_ratio",
        "bt_max_drawdown",
        "bt_sharpe_ratio",
        "bt_total_trades",
    ]
    display_cols = [c for c in display_cols if c in m.columns]

    def fmt_df(df: pd.DataFrame) -> pd.DataFrame:
        view = df.copy()
        for c in view.columns:
            if (
                c.endswith("_return")
                or c.endswith("_ratio")
                or "drawdown" in c
                or c in {"score"}
            ):
                if view[c].dtype.kind in "fc":
                    view[c] = view[c].astype(float).map(lambda x: f"{x:.4f}")
        return view

    lines: list[str] = []
    lines.append("# Holdout8m 策略短名单 (四重验证 + BT)\n")
    lines.append(f"- 运行目录: {run_dir.relative_to(ROOT)}")
    lines.append(f"- BT 目录: {bt_dir.relative_to(ROOT)}")
    lines.append(f"- Final candidates: {len(m)}\n")

    # Quick counts
    bt_ok_cnt = int(m["bt_ok"].sum())
    lines.append(
        f"- BT OK (bt_holdout_return>0 且 bt_calmar_ratio>0): {bt_ok_cnt} / {len(m)}\n"
    )

    lines.append("## Top 30 (综合得分排序)\n")
    lines.append(fmt_df(m[display_cols].head(top_n)).to_markdown(index=False))

    lines.append("\n## BT OK 子集（更适合逐条复核）\n")
    lines.append(
        fmt_df(m[m["bt_ok"]][display_cols].head(top_n)).to_markdown(index=False)
    )

    lines.append("\n## 逐条过的复核清单（按综合得分）\n")
    for i, combo in enumerate(m["combo"].head(top_n).astype(str), start=1):
        lines.append(f"- [ ] {i:02d}. {combo}")

    (run_dir / "STRATEGY_SHORTLIST.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print("✅ Wrote:")
    print(f"- {out_parquet}")
    print(f"- {out_csv}")
    print(f"- {run_dir/'STRATEGY_SHORTLIST.md'}")


if __name__ == "__main__":
    main()
