"""Cross-sectional regime diagnostics for ETF rotation.

This script is intentionally lightweight and *does not* change any strategy logic.
It helps answer questions like:
- Why did many rotation strategies degrade over a specific window?

It computes, for a given window:
- Universe single-ETF start->end returns
- Average pairwise correlation (pre-window vs window)
- A minimal momentum sanity check (trail-20d rank -> forward-3d return)
- Momentum daily IC (Spearman-like rank correlation)

Run (example):
    uv run python scripts/diagnose_cross_section_regime.py \
        --window-start 20251015 --window-end 20251212 \
        --pre-start 20250701 --pre-end 20251014

Data source default: raw/ETF/daily
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Period:
    name: str
    start: int
    end: int


def _to_int_date(v: str) -> int:
    v = v.strip()
    if len(v) != 8 or not v.isdigit():
        raise argparse.ArgumentTypeError(f"Invalid date '{v}', expected YYYYMMDD")
    return int(v)


def _rank_corr(x: pd.Series, y: pd.Series) -> float:
    """Spearman-like correlation via rank + Pearson.

    Avoids a hard scipy dependency.
    """

    df = pd.concat([x, y], axis=1, join="inner").dropna()
    if df.shape[0] < 3:
        return float("nan")
    xr = df.iloc[:, 0].rank(method="average")
    yr = df.iloc[:, 1].rank(method="average")
    return float(xr.corr(yr))


def load_universe_closes(
    universe_dir: Path,
    min_start_date: int,
    required_end_date: int,
    columns: Iterable[str] = ("trade_date", "close"),
) -> pd.DataFrame:
    """Load close prices for all ETFs that cover the required date range."""

    files = sorted(universe_dir.glob("*_daily_*.parquet"))
    series: dict[str, pd.Series] = {}

    for f in files:
        # Expect filename like 510300.SH_daily_20190211_20251212.parquet
        name = f.name
        if "_daily_" not in name:
            continue
        code_part = name.split("_daily_")[0]
        if "." not in code_part:
            continue

        df = pd.read_parquet(f, columns=list(columns))
        df["trade_date"] = pd.to_numeric(df["trade_date"], errors="coerce")
        df = df.dropna(subset=["trade_date", "close"]).copy()
        df["trade_date"] = df["trade_date"].astype("int64")
        df = df.sort_values("trade_date")

        # quick coverage check
        if int(df["trade_date"].iloc[0]) > min_start_date:
            continue
        if int(df["trade_date"].iloc[-1]) < required_end_date:
            continue

        df = df[
            (df["trade_date"] >= min_start_date)
            & (df["trade_date"] <= required_end_date)
        ]
        if df.empty:
            continue

        series[code_part] = df.set_index("trade_date")["close"].astype("float64")

    if not series:
        raise RuntimeError(
            f"No ETF files in {universe_dir} cover [{min_start_date}, {required_end_date}]"
        )

    closes = pd.DataFrame(series).sort_index()
    return closes


def avg_pairwise_corr(returns_df: pd.DataFrame) -> tuple[float, float]:
    corr = returns_df.corr()
    vals = corr.values
    n = vals.shape[0]
    upper = vals[np.triu_indices(n, k=1)]
    return float(np.nanmean(upper)), float(np.nanmedian(upper))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-start", type=_to_int_date, required=True)
    ap.add_argument("--window-end", type=_to_int_date, required=True)
    ap.add_argument("--pre-start", type=_to_int_date, required=True)
    ap.add_argument("--pre-end", type=_to_int_date, required=True)
    ap.add_argument("--universe-dir", type=Path, default=Path("raw/ETF/daily"))
    ap.add_argument("--lookback-start", type=_to_int_date, default="20250101")
    ap.add_argument("--min-universe", type=int, default=30)
    ap.add_argument("--outdir", type=Path, default=Path("results/diagnostics"))
    args = ap.parse_args()

    window = Period("window", args.window_start, args.window_end)
    pre = Period("pre", args.pre_start, args.pre_end)

    if pre.end >= window.start:
        raise SystemExit("pre-end must be < window-start for a clean comparison")

    closes = load_universe_closes(
        args.universe_dir,
        min_start_date=args.lookback_start,
        required_end_date=window.end,
    )

    # Basic stats for the window (single ETF start->end return)
    w = closes[(closes.index >= window.start) & (closes.index <= window.end)]
    window_returns = (w.iloc[-1] / w.iloc[0] - 1.0).sort_values()

    rets = closes.pct_change()

    def slice_returns(p: Period) -> pd.DataFrame:
        x = rets[(rets.index >= p.start) & (rets.index <= p.end)]
        # Drop columns with too many NaNs then require complete rows
        cols = [c for c in x.columns if x[c].notna().mean() > 0.95]
        x = x[cols].dropna(how="any")
        return x

    pre_rets = slice_returns(pre)
    win_rets = slice_returns(window)

    pre_corr_mean, pre_corr_median = avg_pairwise_corr(pre_rets)
    win_corr_mean, win_corr_median = avg_pairwise_corr(win_rets)

    # Momentum sanity check: trailing 20d -> forward 3d
    tr20 = closes / closes.shift(20) - 1.0
    fwd3 = closes.shift(-3) / closes - 1.0

    idx_all = closes.index[
        (closes.index >= window.start) & (closes.index <= window.end - 3)
    ]
    idx_nonoverlap = idx_all[::3]

    def momentum_stats(index: pd.Index) -> dict[str, float]:
        top2, bot2, ew, spread = [], [], [], []
        for d in index:
            sig = tr20.loc[d]
            fw = fwd3.loc[d]
            valid = (~sig.isna()) & (~fw.isna())
            if int(valid.sum()) < int(args.min_universe):
                continue
            sig = sig[valid]
            fw = fw[valid]
            top = sig.nlargest(2).index
            bot = sig.nsmallest(2).index
            top_r = float(fw[top].mean())
            bot_r = float(fw[bot].mean())
            ew_r = float(fw.mean())
            top2.append(top_r)
            bot2.append(bot_r)
            ew.append(ew_r)
            spread.append(top_r - bot_r)

        top2 = np.asarray(top2, dtype=np.float64)
        bot2 = np.asarray(bot2, dtype=np.float64)
        ew = np.asarray(ew, dtype=np.float64)
        spread = np.asarray(spread, dtype=np.float64)

        def _summ(arr: np.ndarray) -> dict[str, float]:
            if arr.size == 0:
                return {
                    "n": 0,
                    "mean": float("nan"),
                    "median": float("nan"),
                    "pos_rate": float("nan"),
                    "p10": float("nan"),
                    "p90": float("nan"),
                    "compounded": float("nan"),
                }
            return {
                "n": int(arr.size),
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "pos_rate": float((arr > 0).mean()),
                "p10": float(np.quantile(arr, 0.1)),
                "p90": float(np.quantile(arr, 0.9)),
                "compounded": float(np.prod(1 + arr) - 1),
            }

        return {
            **{f"top2_{k}": v for k, v in _summ(top2).items()},
            **{f"bot2_{k}": v for k, v in _summ(bot2).items()},
            **{f"ew_{k}": v for k, v in _summ(ew).items()},
            **{f"spread_{k}": v for k, v in _summ(spread).items()},
        }

    mom_nonoverlap = momentum_stats(idx_nonoverlap)

    # IC (daily rank corr): trailing 20d vs forward 3d
    def ic_stats(p: Period) -> dict[str, float]:
        idx = closes.index[(closes.index >= p.start) & (closes.index <= p.end - 3)]
        ics = []
        for d in idx:
            sig = tr20.loc[d]
            fw = fwd3.loc[d]
            valid = (~sig.isna()) & (~fw.isna())
            if int(valid.sum()) < int(args.min_universe):
                continue
            ic = _rank_corr(sig[valid], fw[valid])
            if ic == ic:
                ics.append(ic)
        a = np.asarray(ics, dtype=np.float64)
        if a.size == 0:
            return {
                "n": 0,
                "mean": float("nan"),
                "median": float("nan"),
                "pos_rate": float("nan"),
            }
        return {
            "n": int(a.size),
            "mean": float(a.mean()),
            "median": float(np.median(a)),
            "pos_rate": float((a > 0).mean()),
        }

    ic_pre = ic_stats(pre)
    ic_win = ic_stats(window)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir / f"cross_section_regime_{window.start}_{window.end}_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)

    # Save details
    window_returns.to_csv(outdir / "etf_window_returns.csv", header=True)

    summary = {
        "window_start": window.start,
        "window_end": window.end,
        "pre_start": pre.start,
        "pre_end": pre.end,
        "universe_n": int(closes.shape[1]),
        "pre_avg_pairwise_corr": pre_corr_mean,
        "pre_median_pairwise_corr": pre_corr_median,
        "window_avg_pairwise_corr": win_corr_mean,
        "window_median_pairwise_corr": win_corr_median,
        **{f"ic_pre_{k}": v for k, v in ic_pre.items()},
        **{f"ic_window_{k}": v for k, v in ic_win.items()},
        **{f"mom_nonoverlap_{k}": v for k, v in mom_nonoverlap.items()},
    }
    pd.Series(summary).to_csv(outdir / "summary_metrics.csv", header=False)

    # Write markdown report
    worst = window_returns.head(10)
    best = window_returns.tail(10)

    def fmt_pct(x: float) -> str:
        if x != x:
            return "nan"
        return f"{x * 100:.2f}%"

    report = []
    report.append("# 横截面 Regime 诊断报告\n")
    report.append(f"- Universe: {closes.shape[1]} ETFs (from {args.universe_dir})")
    report.append(f"- Window: {window.start} ~ {window.end}")
    report.append(f"- Pre   : {pre.start} ~ {pre.end}\n")

    report.append("## 1) 单 ETF 区间收益（Window）\n")
    report.append(
        f"- mean: {fmt_pct(float(window_returns.mean()))} | median: {fmt_pct(float(window_returns.median()))}"
    )
    report.append(
        f"- pos%: {float((window_returns > 0).mean()) * 100:.1f}% | neg%: {float((window_returns < 0).mean()) * 100:.1f}%\n"
    )

    report.append("**Worst 10**")
    for k, v in worst.items():
        report.append(f"- {k}: {fmt_pct(float(v))}")
    report.append("\n**Best 10**")
    for k, v in best.items():
        report.append(f"- {k}: {fmt_pct(float(v))}")

    report.append("\n## 2) 相关性变化（两两相关系数）\n")
    report.append(
        f"- Pre avg/median corr   : {pre_corr_mean:.3f} / {pre_corr_median:.3f}"
    )
    report.append(
        f"- Window avg/median corr: {win_corr_mean:.3f} / {win_corr_median:.3f}\n"
    )

    report.append("## 3) 动量是否失效（最小可复核实验）\n")
    report.append(
        "规则：按 trailing 20D return 选 Top2，持有 3 个交易日（FREQ=3 的近似），统计 forward 3D return。\n"
    )

    report.append("**Top2 forward 3D**")
    report.append(
        f"- n={mom_nonoverlap['top2_n']} | mean={fmt_pct(mom_nonoverlap['top2_mean'])} | median={fmt_pct(mom_nonoverlap['top2_median'])} | pos%={mom_nonoverlap['top2_pos_rate']*100:.1f}% | compounded≈{fmt_pct(mom_nonoverlap['top2_compounded'])}"
    )
    report.append("**Bottom2 forward 3D**")
    report.append(
        f"- n={mom_nonoverlap['bot2_n']} | mean={fmt_pct(mom_nonoverlap['bot2_mean'])} | median={fmt_pct(mom_nonoverlap['bot2_median'])} | pos%={mom_nonoverlap['bot2_pos_rate']*100:.1f}% | compounded≈{fmt_pct(mom_nonoverlap['bot2_compounded'])}"
    )
    report.append("**Equal-weight forward 3D (Universe baseline)**")
    report.append(
        f"- n={mom_nonoverlap['ew_n']} | mean={fmt_pct(mom_nonoverlap['ew_mean'])} | median={fmt_pct(mom_nonoverlap['ew_median'])} | pos%={mom_nonoverlap['ew_pos_rate']*100:.1f}% | compounded≈{fmt_pct(mom_nonoverlap['ew_compounded'])}"
    )
    report.append("**Momentum spread (Top2 - Bottom2)**")
    report.append(
        f"- n={mom_nonoverlap['spread_n']} | mean={fmt_pct(mom_nonoverlap['spread_mean'])} | median={fmt_pct(mom_nonoverlap['spread_median'])} | pos%={mom_nonoverlap['spread_pos_rate']*100:.1f}% | compounded≈{fmt_pct(mom_nonoverlap['spread_compounded'])}\n"
    )

    report.append("## 4) 动量 IC（rank corr: trailing 20D vs forward 3D）\n")
    report.append(
        f"- Pre   : n={ic_pre['n']} | mean={ic_pre['mean']:.4f} | median={ic_pre['median']:.4f} | pos%={ic_pre['pos_rate']*100:.1f}%"
    )
    report.append(
        f"- Window: n={ic_win['n']} | mean={ic_win['mean']:.4f} | median={ic_win['median']:.4f} | pos%={ic_win['pos_rate']*100:.1f}%"
    )

    report.append("\n## 5) 结论解读（面向策略）\n")
    report.append(
        "- 若策略组合大量依赖趋势/动量类因子（例如 ADX / SLOPE / PRICE_POSITION / SHARPE_RATIO / MOM 等），当 window 出现 **动量 IC 转负/接近 0** 时，选出来的‘强者’更可能在 3 日后回撤，导致普遍回撤与 Sharpe 下滑。"
    )
    report.append(
        "- 同时 window 的 **横截面相关性上升** 会降低轮动带来的分散收益，使‘相对强弱’更难转化为绝对收益。"
    )

    (outdir / "report.md").write_text("\n".join(report), encoding="utf-8")

    print(f"[OK] wrote: {outdir}/report.md")


if __name__ == "__main__":
    main()
