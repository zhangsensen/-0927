#!/usr/bin/env python3
"""scripts/generate_bt_audit_pack_report.py

Generate an audit pack report from `etf_strategy.auditor.runners.parallel_audit` outputs.

Inputs (a run directory):
- summary.csv
- equity/part_*.parquet   (columns: date, return, equity, rank)
- trades/part_*.parquet   (columns include: entry_date, exit_date, pnlcomm, return_pct, rank, combo)

Outputs (written into the same run directory by default):
- equity_with_combo.parquet
- monthly_returns.csv
- quarterly_returns.csv
- drawdown_summary.csv
- TOP_AUDIT_PACK.md

Notes
-----
- This report is deterministic and uses only existing artifacts.
- It does *not* assume any particular split dates; it reports full-period stats.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _load_parquets(folder: Path) -> pd.DataFrame:
    parts = sorted(folder.glob("part_*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No parquet parts found under: {folder}")
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def _compute_drawdown_stats(equity: pd.DataFrame) -> pd.DataFrame:
    # equity: columns date, equity, rank, combo
    equity = equity.sort_values(["rank", "date"]).copy()

    def per_rank(group: pd.DataFrame) -> pd.Series:
        eq = group["equity"].astype(float)
        dates = pd.to_datetime(group["date"])

        roll_max = eq.cummax()
        dd = eq / roll_max - 1.0
        trough_idx = int(dd.idxmin())
        max_dd = float(dd.loc[trough_idx])

        # peak before trough: last time roll_max was achieved before trough
        pre = group.loc[group.index <= trough_idx].copy()
        pre_eq = pre["equity"].astype(float)
        pre_dates = pd.to_datetime(pre["date"])
        pre_roll_max = pre_eq.cummax()
        peak_value = float(pre_roll_max.iloc[-1])

        # peak date = last date where equity == peak_value up to trough
        peak_mask = pre_eq.eq(peak_value)
        peak_date = (
            pre_dates.loc[peak_mask].iloc[-1] if peak_mask.any() else pre_dates.iloc[0]
        )
        trough_date = pd.to_datetime(group.loc[trough_idx, "date"])

        # recovery = first date after trough where equity >= peak_value
        post = group.loc[group.index >= trough_idx].copy()
        post_dates = pd.to_datetime(post["date"])
        post_eq = post["equity"].astype(float)
        rec_mask = post_eq.ge(peak_value)
        recovery_date = post_dates.loc[rec_mask].iloc[0] if rec_mask.any() else pd.NaT

        duration_to_trough = int((trough_date - peak_date).days)
        if pd.isna(recovery_date):
            duration_to_recover = np.nan
        else:
            duration_to_recover = int((recovery_date - peak_date).days)

        return pd.Series(
            {
                "max_drawdown": max_dd,
                "peak_date": peak_date.date().isoformat(),
                "trough_date": trough_date.date().isoformat(),
                "recovery_date": (
                    recovery_date.date().isoformat()
                    if pd.notna(recovery_date)
                    else "NA"
                ),
                "days_peak_to_trough": duration_to_trough,
                "days_peak_to_recover": duration_to_recover,
            }
        )

    out = equity.groupby(["rank", "combo"], as_index=False).apply(per_rank)
    # groupby.apply returns multiindex columns in some pandas versions; normalize
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[-1] for c in out.columns]
    return out.reset_index(drop=True)


def _monthly_quarterly_returns(
    equity: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = equity.copy()
    df["date"] = pd.to_datetime(df["date"])

    # If `return` exists and looks like periodic return, prefer compounding it.
    if "return" in df.columns:
        df["ret"] = df["return"].astype(float)
    else:
        df = df.sort_values(["rank", "date"])
        df["ret"] = df.groupby("rank")["equity"].pct_change().fillna(0.0)

    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["quarter"] = df["date"].dt.to_period("Q").astype(str)

    def agg(group: pd.DataFrame) -> float:
        return float((1.0 + group["ret"]).prod() - 1.0)

    m = (
        df.groupby(["rank", "combo", "month"], as_index=False)
        .apply(lambda g: pd.Series({"ret": agg(g)}))
        .reset_index(drop=True)
    )
    q = (
        df.groupby(["rank", "combo", "quarter"], as_index=False)
        .apply(lambda g: pd.Series({"ret": agg(g)}))
        .reset_index(drop=True)
    )

    monthly = m.pivot_table(
        index=["rank", "combo"], columns="month", values="ret", aggfunc="first"
    ).reset_index()
    quarterly = q.pivot_table(
        index=["rank", "combo"], columns="quarter", values="ret", aggfunc="first"
    ).reset_index()

    # Deterministic column order (chronological)
    month_cols = sorted([c for c in monthly.columns if c not in {"rank", "combo"}])
    quarter_cols = sorted([c for c in quarterly.columns if c not in {"rank", "combo"}])
    monthly = monthly[["rank", "combo"] + month_cols]
    quarterly = quarterly[["rank", "combo"] + quarter_cols]

    return monthly, quarterly


def _to_pct_str(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.2%}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate BT audit pack report from parallel_audit output"
    )
    parser.add_argument(
        "--run-dir", required=True, help="Path to parallel_audit run directory"
    )
    parser.add_argument(
        "--out-dir", default=None, help="Output directory (default: run-dir)"
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "summary.csv"
    equity_dir = run_dir / "equity"
    trades_dir = run_dir / "trades"

    if not summary_path.exists():
        raise FileNotFoundError(str(summary_path))
    if not equity_dir.exists():
        raise FileNotFoundError(str(equity_dir))
    if not trades_dir.exists():
        raise FileNotFoundError(str(trades_dir))

    summary = pd.read_csv(summary_path)
    equity = _load_parquets(equity_dir)
    trades = _load_parquets(trades_dir)

    # Normalize / map rank -> combo
    if "combo" not in equity.columns:
        if "combo" not in summary.columns or (
            "rank" not in summary.columns and "real_rank" not in summary.columns
        ):
            raise ValueError(
                "summary.csv must include columns: (rank or real_rank) and combo"
            )

        rank_col = "rank" if "rank" in summary.columns else "real_rank"
        rank_map = summary[[rank_col, "combo"]].rename(columns={rank_col: "rank"})
        equity = equity.merge(rank_map, on="rank", how="left")

    # Persist a convenient long equity file
    out_equity = out_dir / "equity_with_combo.parquet"
    equity.to_parquet(out_equity, index=False)

    # Returns tables
    monthly, quarterly = _monthly_quarterly_returns(equity)
    out_month = out_dir / "monthly_returns.csv"
    out_quarter = out_dir / "quarterly_returns.csv"
    monthly.to_csv(out_month, index=False)
    quarterly.to_csv(out_quarter, index=False)

    # Drawdown stats
    dd = _compute_drawdown_stats(equity)
    out_dd = out_dir / "drawdown_summary.csv"
    dd.to_csv(out_dd, index=False)

    # Trade summary
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades["exit_date"] = pd.to_datetime(trades["exit_date"])

    trade_sum = (
        trades.groupby(["rank", "combo"], as_index=False)
        .agg(
            total_trades=("pnlcomm", "count"),
            win_rate=("pnlcomm", lambda s: float((s > 0).mean())),
            profit_factor=(
                "pnlcomm",
                lambda s: (
                    float(s[s > 0].sum() / (-s[s < 0].sum()))
                    if (s[s < 0].sum() < 0)
                    else np.inf
                ),
            ),
            avg_pnl=("pnlcomm", "mean"),
            avg_return_pct=("return_pct", "mean"),
        )
        .merge(dd, on=["rank", "combo"], how="left")
        .sort_values(["rank"])
    )

    # Equity-based total return
    equity_tot = (
        equity.sort_values(["rank", "date"])
        .groupby(["rank", "combo"], as_index=False)
        .agg(first_equity=("equity", "first"), last_equity=("equity", "last"))
    )
    equity_tot["total_return_equity"] = (
        equity_tot["last_equity"] / equity_tot["first_equity"] - 1.0
    )

    report_df = trade_sum.merge(
        equity_tot[["rank", "combo", "total_return_equity"]],
        on=["rank", "combo"],
        how="left",
    )

    # Write markdown report
    md_path = out_dir / "TOP_AUDIT_PACK.md"
    display = report_df.copy()
    display["total_return_equity"] = display["total_return_equity"].apply(_to_pct_str)
    display["win_rate"] = display["win_rate"].apply(_to_pct_str)
    display["max_drawdown"] = display["max_drawdown"].apply(_to_pct_str)
    display["avg_return_pct"] = display["avg_return_pct"].apply(_to_pct_str)

    cols = [
        "rank",
        "combo",
        "total_return_equity",
        "max_drawdown",
        "peak_date",
        "trough_date",
        "recovery_date",
        "total_trades",
        "win_rate",
        "profit_factor",
        "avg_pnl",
        "avg_return_pct",
    ]

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Top BT 审计包（trades + equity）\n\n")
        f.write(f"Run Dir: {run_dir.as_posix()}\n\n")
        f.write("## 1) 汇总（按 rank）\n\n")
        f.write(display[cols].to_markdown(index=False))
        f.write("\n\n")
        f.write("## 2) 产物清单\n\n")
        f.write(f"- equity_with_combo.parquet\n")
        f.write(f"- monthly_returns.csv\n")
        f.write(f"- quarterly_returns.csv\n")
        f.write(f"- drawdown_summary.csv\n")

    print(f"Wrote: {out_equity}")
    print(f"Wrote: {out_month}")
    print(f"Wrote: {out_quarter}")
    print(f"Wrote: {out_dd}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
