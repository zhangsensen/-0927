"""Compare pipeline survivor sets with/without Regime Gate.

This script compares *final* survivors produced by the full validation funnel:
VEC + Rolling (train-only) + Holdout + BT (audit run upstream).

It is intentionally process-level (set comparison + metric distributions), not
per-strategy A/B within the same backtest.

Usage (example):
    uv run python scripts/compare_regime_gate_pipeline_outputs.py \
        --gate-off results/final_triple_validation_20251216_040829/final_candidates.parquet \
        --gate-on  results/final_triple_validation_20251216_041418/final_candidates.parquet

Outputs:
    results/diagnostics/regime_gate_compare_<ts>/
        - report.md
        - overlap.csv
        - only_gate_off.csv
        - only_gate_on.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SummarySpec:
    columns: list[str]


DEFAULT_SUMMARY_COLUMNS = [
    # core selection signals
    "composite_score",
    "score_train",
    "score_roll",
    "score_holdout",
    # VEC (train)
    "vec_return",
    "vec_max_drawdown",
    "vec_calmar_ratio",
    "vec_sharpe_ratio",
    "vec_trades",
    # Rolling (train-only)
    "roll_all_segment_positive_rate",
    "roll_all_segment_worst_return",
    "roll_all_segment_worst_calmar",
    "roll_holdout_segment_positive_rate",
    "roll_holdout_segment_worst_return",
    "roll_holdout_segment_worst_calmar",
    # Holdout
    "holdout_return",
    "holdout_max_drawdown",
    "holdout_calmar_ratio",
    "holdout_sharpe_ratio",
]


def _as_path(p: str) -> Path:
    return Path(p).expanduser().resolve()


def _ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(str(path))


def _safe_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    s = pd.to_numeric(df[col], errors="coerce")
    return s.replace([np.inf, -np.inf], np.nan)


def _describe_metric(df: pd.DataFrame, col: str) -> dict:
    s = _safe_numeric_series(df, col).dropna()
    if s.empty:
        return {
            "metric": col,
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "p10": np.nan,
            "p90": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    q = s.quantile([0.10, 0.25, 0.75, 0.90])
    return {
        "metric": col,
        "n": int(s.shape[0]),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "p25": float(q.loc[0.25]),
        "p75": float(q.loc[0.75]),
        "p10": float(q.loc[0.10]),
        "p90": float(q.loc[0.90]),
        "min": float(s.min()),
        "max": float(s.max()),
    }


def _fmt_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return f"{x * 100:.2f}%"


def _fmt_float(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return f"{x:.4f}"


def _render_table(df: pd.DataFrame, cols: list[str], n: int = 10) -> str:
    show = df.loc[:, [c for c in cols if c in df.columns]].head(n)
    if show.empty:
        return "(empty)"
    return show.to_markdown(index=False)


def _build_overlap_frames(
    df_off: pd.DataFrame, df_on: pd.DataFrame, key: str = "combo"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    off_keys = set(df_off[key].astype(str))
    on_keys = set(df_on[key].astype(str))

    common = sorted(off_keys & on_keys)
    only_off = sorted(off_keys - on_keys)
    only_on = sorted(on_keys - off_keys)

    common_df = pd.DataFrame({key: common})
    only_off_df = pd.DataFrame({key: only_off})
    only_on_df = pd.DataFrame({key: only_on})

    return common_df, only_off_df, only_on_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare final pipeline survivors with/without Regime Gate"
    )
    parser.add_argument(
        "--gate-off", required=True, help="Gate OFF final_candidates.parquet"
    )
    parser.add_argument(
        "--gate-on", required=True, help="Gate ON final_candidates.parquet"
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: results/diagnostics/regime_gate_compare_<ts>)",
    )
    parser.add_argument(
        "--key",
        default="combo",
        help="Primary key column for identifying strategies (default: combo)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Top-N rows to show in tables (default: 15)",
    )

    args = parser.parse_args()

    p_off = _as_path(args.gate_off)
    p_on = _as_path(args.gate_on)
    _ensure_exists(p_off)
    _ensure_exists(p_on)

    df_off = pd.read_parquet(p_off)
    df_on = pd.read_parquet(p_on)

    if args.key not in df_off.columns or args.key not in df_on.columns:
        raise KeyError(f"Key column '{args.key}' must exist in both inputs")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        _as_path(args.out_dir)
        if args.out_dir
        else Path("results/diagnostics") / f"regime_gate_compare_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # overlap sets
    common_df, only_off_df, only_on_df = _build_overlap_frames(
        df_off, df_on, key=args.key
    )
    common_df.to_csv(out_dir / "overlap.csv", index=False)
    only_off_df.to_csv(out_dir / "only_gate_off.csv", index=False)
    only_on_df.to_csv(out_dir / "only_gate_on.csv", index=False)

    n_off = len(df_off)
    n_on = len(df_on)
    n_common = len(common_df)

    jaccard = n_common / max(1, (n_off + n_on - n_common))
    overlap_off = n_common / max(1, n_off)
    overlap_on = n_common / max(1, n_on)

    # metric summaries
    spec = SummarySpec(
        columns=[
            c
            for c in DEFAULT_SUMMARY_COLUMNS
            if (c in df_off.columns or c in df_on.columns)
        ]
    )

    summary_off = pd.DataFrame(
        [_describe_metric(df_off, c) for c in spec.columns]
    ).set_index("metric")
    summary_on = pd.DataFrame(
        [_describe_metric(df_on, c) for c in spec.columns]
    ).set_index("metric")

    compare = (
        summary_on[["mean", "median", "p25", "p75"]]
        .rename(
            columns={
                "mean": "on_mean",
                "median": "on_median",
                "p25": "on_p25",
                "p75": "on_p75",
            }
        )
        .join(
            summary_off[["mean", "median", "p25", "p75"]].rename(
                columns={
                    "mean": "off_mean",
                    "median": "off_median",
                    "p25": "off_p25",
                    "p75": "off_p75",
                }
            ),
            how="outer",
        )
    )

    compare["median_delta(on-off)"] = compare["on_median"] - compare["off_median"]
    compare["mean_delta(on-off)"] = compare["on_mean"] - compare["off_mean"]
    compare.to_csv(out_dir / "metric_distribution_compare.csv")

    # Top lists
    top_cols = [
        args.key,
        "size",
        "composite_score",
        "holdout_calmar_ratio",
        "holdout_return",
        "holdout_max_drawdown",
        "vec_calmar_ratio",
        "vec_return",
        "roll_all_segment_positive_rate",
    ]

    off_top = df_off.sort_values(by=["composite_score"], ascending=False)
    on_top = df_on.sort_values(by=["composite_score"], ascending=False)

    # Common subset with both scores
    df_off_keyed = df_off.set_index(args.key, drop=False)
    df_on_keyed = df_on.set_index(args.key, drop=False)
    common_keys = common_df[args.key].astype(str).tolist()
    common_join = (
        df_on_keyed.loc[common_keys]
        .add_prefix("on_")
        .join(df_off_keyed.loc[common_keys].add_prefix("off_"), how="inner")
        .reset_index(drop=True)
    )

    common_join = common_join.sort_values(by=["on_composite_score"], ascending=False)

    # Only sets enriched with selected columns
    only_off_detail = df_off[
        df_off[args.key].astype(str).isin(only_off_df[args.key])
    ].copy()
    only_on_detail = df_on[
        df_on[args.key].astype(str).isin(only_on_df[args.key])
    ].copy()

    # Report
    report_path = out_dir / "report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Regime Gate：流水线 Survivor 对比（Gate OFF vs Gate ON）\n\n")
        f.write("## 输入\n")
        f.write(f"- Gate OFF: `{p_off}`\n")
        f.write(f"- Gate ON:  `{p_on}`\n\n")

        f.write("## Survivor 规模与重叠\n")
        f.write(f"- Gate OFF final survivors: **{n_off}**\n")
        f.write(f"- Gate ON  final survivors: **{n_on}**\n")
        f.write(f"- Overlap (common): **{n_common}**\n")
        f.write(f"- Jaccard(common / union): **{jaccard:.3f}**\n")
        f.write(f"- Overlap rate vs OFF: **{overlap_off:.3f}**\n")
        f.write(f"- Overlap rate vs ON:  **{overlap_on:.3f}**\n\n")

        f.write("文件输出（可复现）\n")
        f.write(f"- overlap: `{out_dir / 'overlap.csv'}`\n")
        f.write(f"- only_gate_off: `{out_dir / 'only_gate_off.csv'}`\n")
        f.write(f"- only_gate_on: `{out_dir / 'only_gate_on.csv'}`\n")
        f.write(
            f"- metric_distribution_compare: `{out_dir / 'metric_distribution_compare.csv'}`\n\n"
        )

        f.write("## 指标分布对比（中位数/均值差异）\n")
        # Keep a stable ordering that is easy to scan
        ordered = [c for c in DEFAULT_SUMMARY_COLUMNS if c in compare.index]
        view = compare.loc[
            ordered,
            [
                "off_median",
                "on_median",
                "median_delta(on-off)",
                "off_mean",
                "on_mean",
                "mean_delta(on-off)",
            ],
        ].copy()
        f.write(view.to_markdown())
        f.write("\n\n")

        f.write("## Top 策略（按 composite_score）\n")
        f.write("### Gate OFF Top\n")
        f.write(_render_table(off_top, top_cols, n=args.top_n))
        f.write("\n\n### Gate ON Top\n")
        f.write(_render_table(on_top, top_cols, n=args.top_n))
        f.write("\n\n")

        f.write("## 共同策略（common）Top（按 ON composite_score）\n")
        common_cols = [
            "on_combo",
            "on_composite_score",
            "on_holdout_calmar_ratio",
            "on_holdout_return",
            "on_holdout_max_drawdown",
            "off_composite_score",
            "off_holdout_calmar_ratio",
            "off_holdout_return",
            "off_holdout_max_drawdown",
        ]
        f.write(_render_table(common_join, common_cols, n=args.top_n))
        f.write("\n\n")

        f.write("## Gate ON 新增（only_on）Top（按 ON composite_score）\n")
        f.write(
            _render_table(
                only_on_detail.sort_values(by=["composite_score"], ascending=False),
                top_cols,
                n=args.top_n,
            )
        )
        f.write("\n\n")

        f.write("## Gate OFF 独有（only_off）Top（按 OFF composite_score）\n")
        f.write(
            _render_table(
                only_off_detail.sort_values(by=["composite_score"], ascending=False),
                top_cols,
                n=args.top_n,
            )
        )
        f.write("\n\n")

        f.write("---\n")
        f.write(
            "备注：本报告仅比较 **最终 survivors 集合** 的统计分布；它反映了“策略筛选流程”的差异，而不是单一策略内 A/B 切换。\n"
        )

    print(f"✅ Report written: {report_path}")
    print(f"📁 Output dir: {out_dir}")


if __name__ == "__main__":
    main()
