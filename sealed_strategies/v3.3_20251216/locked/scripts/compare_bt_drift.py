#!/usr/bin/env python3
"""BT 漂移分析（可比升级）

用途
- 对比“封版产物” vs “可比升级版（固定 training_end_date，仅推进 end_date）”的 Backtrader 审计结果。
- 输出：
  1) 每个策略的指标差分（delta）与排名变化
  2) 汇总统计 + Top/Bottom 漂移列表（中文 Markdown 报告）

约定
- 两份输入均为 scripts/batch_bt_backtest.py 产出的 bt_results.parquet
- 主键为 combo（因子组合字符串），应完全一致；若不一致会在报告中提示。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


RETURN_LIKE = [
    "bt_return",
    "bt_train_return",
    "bt_holdout_return",
    "bt_annual_return",
    "bt_sharpe_ratio",
    "bt_calmar_ratio",
    "bt_aligned_return",
    "bt_aligned_sharpe",
    "bt_win_rate",
    "bt_profit_factor",
    "bt_avg_pnl",
]

RISK_LIKE = [
    "bt_max_drawdown",  # 越小越好
    "bt_annual_volatility",  # 越小越好
]

COUNT_LIKE = [
    "bt_total_trades",
    "bt_margin_failures",
    "bt_avg_len",
    "bt_max_len",
]


@dataclass(frozen=True)
class MetricSpec:
    name: str
    higher_is_better: bool


RANK_METRICS: list[MetricSpec] = [
    MetricSpec("bt_return", True),
    MetricSpec("bt_annual_return", True),
    MetricSpec("bt_sharpe_ratio", True),
    MetricSpec("bt_calmar_ratio", True),
    MetricSpec("bt_max_drawdown", False),
]


def _as_path(s: str) -> Path:
    p = Path(s)
    return p if p.is_absolute() else (ROOT / p).resolve()


def _ensure_outdir(outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _rank_best_first(series: pd.Series, higher_is_better: bool) -> pd.Series:
    # rank=1 表示最好。method="min" 可确保确定性。
    return series.rank(method="min", ascending=not higher_is_better).astype("int64")


def _safe_quantiles(
    s: pd.Series, qs=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
) -> pd.Series:
    s2 = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s2.empty:
        return pd.Series({q: np.nan for q in qs})
    return s2.quantile(list(qs))


def _fmt_pct(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x * 100:.2f}pp"


def _fmt_float(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.4f}"


def _direction_label(metric: str, delta: float) -> str:
    """给 delta 一个直观方向标签。

    - 对收益类：delta>0 视为改善
    - 对风险类（越小越好）：delta<0 视为改善
    """
    if pd.isna(delta):
        return "NA"

    lower_is_better = metric in {"bt_max_drawdown", "bt_annual_volatility"}
    if lower_is_better:
        if delta < 0:
            return "改善"
        if delta > 0:
            return "恶化"
        return "持平"

    if delta > 0:
        return "改善"
    if delta < 0:
        return "恶化"
    return "持平"


def _write_markdown_report(
    *,
    out_path: Path,
    sealed_path: Path,
    new_path: Path,
    merged: pd.DataFrame,
    stats: dict[str, dict[str, float]],
    top_tables: dict[str, pd.DataFrame],
    meta: dict[str, object],
) -> None:
    lines: list[str] = []
    lines.append("# BT 可比升级漂移分析报告\n")
    lines.append(f"- 封版 bt_results: `{sealed_path}`")
    lines.append(f"- 可比升级 bt_results: `{new_path}`")
    lines.append(
        f"- 策略数（封版/升级/交集）: {meta['sealed_n']} / {meta['new_n']} / {meta['common_n']}"
    )
    if meta.get("missing_in_new"):
        lines.append(f"- ⚠️ 升级版缺失策略（展示最多 20 条）: {meta['missing_in_new']}")
    if meta.get("missing_in_sealed"):
        lines.append(f"- ⚠️ 封版缺失策略（展示最多 20 条）: {meta['missing_in_sealed']}")
    lines.append("")

    lines.append("## 1) 核心结论（先看这个）\n")

    for metric in ["bt_return", "bt_sharpe_ratio", "bt_max_drawdown"]:
        d = stats.get(metric)
        if not d:
            continue
        if metric in {"bt_sharpe_ratio"}:
            fmt = _fmt_float
        else:
            fmt = _fmt_pct
        lines.append(
            f"- `{metric}` Δ均值: {fmt(d['mean'])}（{_direction_label(metric, d['mean'])}），"
            f"Δ中位数: {fmt(d['p50'])}"
        )
    lines.append("")

    lines.append("## 2) 漂移分布（Δ = 升级版 - 封版）\n")
    for metric, d in stats.items():
        if metric in {"bt_sharpe_ratio", "bt_calmar_ratio"}:
            fmt = _fmt_float
        else:
            fmt = _fmt_pct
        lines.append(
            f"- `{metric}`: mean {fmt(d['mean'])} | p10 {fmt(d['p10'])} | p50 {fmt(d['p50'])} | p90 {fmt(d['p90'])}"
        )
    lines.append("")

    lines.append("## 3) 排名变化（rank=1 最好）\n")
    for spec in RANK_METRICS:
        col = f"rank_change__{spec.name}"
        if col not in merged.columns:
            continue
        s = merged[col]
        improved = int((s < 0).sum())
        worsened = int((s > 0).sum())
        unchanged = int((s == 0).sum())
        lines.append(
            f"- `{spec.name}`: 提升 {improved} | 下降 {worsened} | 不变 {unchanged} | 最大提升 {int(s.min())} | 最大下降 {int(s.max())}"
        )
    lines.append("")

    lines.append("## 4) Top/Bottom 漂移样本（用于人工复核）\n")

    def _md_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "（空）"
        return df.to_markdown(index=False)

    for title, df in top_tables.items():
        lines.append(f"### {title}\n")
        lines.append(_md_table(df))
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="封版 vs 可比升级 BT 漂移分析")
    parser.add_argument("--sealed", required=True, help="封版 bt_results.parquet 路径")
    parser.add_argument(
        "--new", required=True, help="可比升级版 bt_results.parquet 路径"
    )
    parser.add_argument(
        "--outdir", default="results/diagnostics/bt_drift", help="输出目录"
    )

    args = parser.parse_args()

    sealed_path = _as_path(args.sealed)
    new_path = _as_path(args.new)
    outdir = _ensure_outdir(_as_path(args.outdir))

    df_sealed = pd.read_parquet(sealed_path)
    df_new = pd.read_parquet(new_path)

    if "combo" not in df_sealed.columns or "combo" not in df_new.columns:
        raise ValueError("输入 bt_results 缺少 combo 列，无法对齐")

    if df_sealed["combo"].duplicated().any() or df_new["combo"].duplicated().any():
        raise ValueError("检测到 combo 重复，无法唯一对齐（请先去重）")

    sealed = df_sealed.set_index("combo").sort_index()
    new = df_new.set_index("combo").sort_index()

    common = sealed.index.intersection(new.index)
    missing_in_new = sealed.index.difference(new.index)
    missing_in_sealed = new.index.difference(sealed.index)

    sealed_c = sealed.loc[common]
    new_c = new.loc[common]

    merged = sealed_c.add_suffix("__sealed").join(
        new_c.add_suffix("__new"), how="inner"
    )
    merged.insert(0, "combo", merged.index)

    all_metrics = list({*(RETURN_LIKE), *(RISK_LIKE), *(COUNT_LIKE)})
    for metric in all_metrics:
        c0 = f"{metric}__sealed"
        c1 = f"{metric}__new"
        if c0 in merged.columns and c1 in merged.columns:
            merged[f"delta__{metric}"] = merged[c1] - merged[c0]

    for spec in RANK_METRICS:
        c0 = f"{spec.name}__sealed"
        c1 = f"{spec.name}__new"
        if c0 not in merged.columns or c1 not in merged.columns:
            continue
        merged[f"rank__{spec.name}__sealed"] = _rank_best_first(
            merged[c0], higher_is_better=spec.higher_is_better
        )
        merged[f"rank__{spec.name}__new"] = _rank_best_first(
            merged[c1], higher_is_better=spec.higher_is_better
        )
        merged[f"rank_change__{spec.name}"] = (
            merged[f"rank__{spec.name}__new"] - merged[f"rank__{spec.name}__sealed"]
        )

    stats: dict[str, dict[str, float]] = {}
    for metric in [
        "bt_return",
        "bt_train_return",
        "bt_holdout_return",
        "bt_annual_return",
        "bt_sharpe_ratio",
        "bt_calmar_ratio",
        "bt_max_drawdown",
    ]:
        dcol = f"delta__{metric}"
        if dcol not in merged.columns:
            continue
        qs = _safe_quantiles(merged[dcol])
        stats[metric] = {
            "mean": float(pd.to_numeric(merged[dcol], errors="coerce").mean()),
            "p10": float(qs.loc[0.1]),
            "p50": float(qs.loc[0.5]),
            "p90": float(qs.loc[0.9]),
        }

    keep_cols = [
        "combo",
        "bt_return__sealed",
        "bt_return__new",
        "delta__bt_return",
        "bt_holdout_return__sealed",
        "bt_holdout_return__new",
        "delta__bt_holdout_return",
        "bt_max_drawdown__sealed",
        "bt_max_drawdown__new",
        "delta__bt_max_drawdown",
        "bt_sharpe_ratio__sealed",
        "bt_sharpe_ratio__new",
        "delta__bt_sharpe_ratio",
        "rank_change__bt_return",
        "rank_change__bt_sharpe_ratio",
        "rank_change__bt_max_drawdown",
    ]
    keep_cols = [c for c in keep_cols if c in merged.columns]

    def _top_df(sort_col: str, n: int, ascending: bool) -> pd.DataFrame:
        df = merged[keep_cols].sort_values(sort_col, ascending=ascending).head(n).copy()
        float_cols = [
            c
            for c in df.columns
            if c.endswith("__sealed") or c.endswith("__new") or c.startswith("delta__")
        ]
        for c in float_cols:
            if c.startswith("delta__bt_") and ("return" in c):
                df[c] = df[c].astype(float).map(lambda x: f"{x:.4f}")
            else:
                df[c] = df[c].astype(float).map(lambda x: f"{x:.4f}")
        for c in [
            "rank_change__bt_return",
            "rank_change__bt_sharpe_ratio",
            "rank_change__bt_max_drawdown",
        ]:
            if c in df.columns:
                df[c] = df[c].astype(int)
        return df

    top_tables: dict[str, pd.DataFrame] = {}
    if "delta__bt_holdout_return" in merged.columns:
        top_tables["Holdout 收益提升 Top 15（Δbt_holdout_return 最大）"] = _top_df(
            "delta__bt_holdout_return", 15, ascending=False
        )
        top_tables["Holdout 收益恶化 Top 15（Δbt_holdout_return 最小）"] = _top_df(
            "delta__bt_holdout_return", 15, ascending=True
        )
    if "rank_change__bt_return" in merged.columns:
        top_tables["总收益排名提升 Top 15（rank_change__bt_return 最小）"] = _top_df(
            "rank_change__bt_return", 15, ascending=True
        )
        top_tables["总收益排名下降 Top 15（rank_change__bt_return 最大）"] = _top_df(
            "rank_change__bt_return", 15, ascending=False
        )

    merged_out = outdir / "bt_drift_detail.parquet"
    merged.to_parquet(merged_out, index=False)
    merged_csv = outdir / "bt_drift_detail.csv"
    merged.to_csv(merged_csv, index=False)

    report_path = outdir / "bt_drift_report.md"
    meta = {
        "sealed_n": int(df_sealed.shape[0]),
        "new_n": int(df_new.shape[0]),
        "common_n": int(len(common)),
        "missing_in_new": list(map(str, missing_in_new[:20])),
        "missing_in_sealed": list(map(str, missing_in_sealed[:20])),
    }

    _write_markdown_report(
        out_path=report_path,
        sealed_path=sealed_path,
        new_path=new_path,
        merged=merged,
        stats=stats,
        top_tables=top_tables,
        meta=meta,
    )

    print("✅ 漂移分析完成")
    print(f"   明细: {merged_out}")
    print(f"   报告: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
