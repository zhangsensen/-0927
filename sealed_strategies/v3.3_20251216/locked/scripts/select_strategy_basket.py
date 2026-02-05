#!/usr/bin/env python3
"""Select a small basket (7-10) of sealed strategies for the *current* regime.

Design goals:
- Use sealed, audit-grade candidates (v3.2) as the universe (152 strategies).
- Focus on *recent choppy regime* via a trailing window (default: last 120 trading days).
- Use VEC (fast) to recompute recent-window equity curves for correlation/diversification.
- Keep BT metrics as ground-truth *global* quality gates; do NOT re-run large BT.

Outputs a deliverable folder under results/ with:
- selected_strategies.csv
- selected_corr.csv
- basket_report.md

Run:
  uv run python scripts/select_strategy_basket.py \
    --sealed-dir sealed_strategies/v3.2_20251214 \
    --k 8 --window 120
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# Make project imports work when running as a script
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import shift_timing_signal

from batch_vec_backtest import run_vec_backtest, calculate_atr


def _zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return s * 0.0
    return (s - s.mean()) / std


def compute_period_metrics(equity_curve: np.ndarray) -> dict[str, float]:
    if equity_curve is None or len(equity_curve) < 2:
        return {"ret": 0.0, "ann_ret": 0.0, "max_dd": 0.0, "sharpe": 0.0}

    eq = np.asarray(equity_curve, dtype=float)
    eq = np.maximum(eq, 1e-6)

    daily_ret = np.diff(eq) / eq[:-1]
    if len(daily_ret) == 0:
        return {"ret": 0.0, "ann_ret": 0.0, "max_dd": 0.0, "sharpe": 0.0}

    total_ret = float(eq[-1] / eq[0] - 1.0)

    # Annualized return
    try:
        ann_ret = float((1 + total_ret) ** (252 / max(1, len(daily_ret))) - 1)
        if np.isinf(ann_ret) or np.isnan(ann_ret):
            ann_ret = -0.99
        ann_ret = float(np.clip(ann_ret, -0.99, 10.0))
    except Exception:
        ann_ret = -0.99

    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = float(dd.min()) if len(dd) else 0.0

    mu = float(np.mean(daily_ret))
    sigma = float(np.std(daily_ret, ddof=0))
    sharpe = float(mu / sigma * np.sqrt(252)) if sigma > 1e-12 else 0.0
    if np.isinf(sharpe) or np.isnan(sharpe):
        sharpe = 0.0
    sharpe = float(np.clip(sharpe, -20.0, 20.0))

    return {"ret": total_ret, "ann_ret": ann_ret, "max_dd": max_dd, "sharpe": sharpe}


def parse_combo(combo: str) -> list[str]:
    return [p.strip() for p in combo.split("+")]


@dataclass(frozen=True)
class PreparedData:
    dates: pd.DatetimeIndex
    factor_names: list[str]
    all_factors_stack: np.ndarray  # (T, N, F)
    close_prices: np.ndarray  # (T, N)
    open_prices: np.ndarray
    high_prices: np.ndarray
    low_prices: np.ndarray
    timing_arr: np.ndarray  # (T,)
    vol_regime_arr: np.ndarray  # (T,)
    atr_arr: np.ndarray | None
    etf_codes: list[str]


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_data_for_recent_window(
    config: dict, end_date: str | None, min_days: int
) -> PreparedData:
    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )

    cfg_end = end_date or config["data"].get("end_date")
    if not cfg_end:
        raise ValueError("No end_date provided in config/data.")

    # Load a reasonably short slice to speed up factor computation.
    # We keep enough history for factor windows + lookback warmup.
    end_dt = pd.Timestamp(cfg_end)
    start_dt = end_dt - pd.Timedelta(days=int(min_days * 2.2))

    cfg_start = pd.Timestamp(config["data"]["start_date"])
    if start_dt < cfg_start:
        start_dt = cfg_start

    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=str(start_dt.date()),
        end_date=str(end_dt.date()),
    )

    # Factors
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)
    factor_names = sorted(raw_factors_df.columns.get_level_values(0).unique().tolist())
    raw_factors = {fname: raw_factors_df[fname] for fname in factor_names}

    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)

    dates = std_factors[factor_names[0]].index
    etf_codes = std_factors[factor_names[0]].columns.tolist()

    all_factors_stack = np.stack([std_factors[f].values for f in factor_names], axis=-1)

    close_prices = ohlcv["close"][etf_codes].ffill().bfill().values
    open_prices = ohlcv["open"][etf_codes].ffill().bfill().values
    high_prices = ohlcv["high"][etf_codes].ffill().bfill().values
    low_prices = ohlcv["low"][etf_codes].ffill().bfill().values

    # ATR if configured
    stop_method = config["backtest"]["risk_control"].get("stop_method", "fixed")
    atr_arr = None
    if stop_method == "atr":
        atr_window = config["backtest"]["risk_control"].get("atr_window", 14)
        atr_arr = calculate_atr(
            high_prices, low_prices, close_prices, window=atr_window
        )

    # Timing
    timing_module = LightTimingModule(
        extreme_threshold=config["backtest"]["timing"]["extreme_threshold"],
        extreme_position=config["backtest"]["timing"]["extreme_position"],
    )
    timing_series = timing_module.compute_position_ratios(ohlcv["close"])
    timing_arr_raw = timing_series.reindex(dates).fillna(1.0).values
    timing_arr = shift_timing_signal(timing_arr_raw)

    # Vol regime (same logic as other scripts)
    if "510300" in ohlcv["close"].columns:
        hs300 = ohlcv["close"]["510300"]
    else:
        hs300 = ohlcv["close"].iloc[:, 0]

    rets = hs300.pct_change()
    hv = rets.rolling(window=20, min_periods=20).std() * np.sqrt(252) * 100
    hv_5d = hv.shift(5)
    regime_vol = (hv + hv_5d) / 2

    exposure_s = pd.Series(1.0, index=regime_vol.index)
    exposure_s[regime_vol >= 25] = 0.7
    exposure_s[regime_vol >= 30] = 0.4
    exposure_s[regime_vol >= 40] = 0.1

    vol_regime_arr = exposure_s.reindex(dates).fillna(1.0).values

    return PreparedData(
        dates=dates,
        factor_names=factor_names,
        all_factors_stack=all_factors_stack,
        close_prices=close_prices,
        open_prices=open_prices,
        high_prices=high_prices,
        low_prices=low_prices,
        timing_arr=timing_arr,
        vol_regime_arr=vol_regime_arr,
        atr_arr=atr_arr,
        etf_codes=etf_codes,
    )


def run_vec_recent_window(
    combo_factor_indices: list[int],
    prepared: PreparedData,
    config: dict,
    start_idx: int,
    end_idx: int,
) -> tuple[dict[str, float], np.ndarray]:
    lookback = int(config["backtest"]["lookback"])

    slice_start = max(0, start_idx - lookback)
    slice_end = end_idx

    factors_slice = prepared.all_factors_stack[slice_start:slice_end, :, :][
        ..., combo_factor_indices
    ]
    close_slice = prepared.close_prices[slice_start:slice_end, :]
    open_slice = prepared.open_prices[slice_start:slice_end, :]
    high_slice = prepared.high_prices[slice_start:slice_end, :]
    low_slice = prepared.low_prices[slice_start:slice_end, :]
    timing_slice = prepared.timing_arr[slice_start:slice_end]
    vol_slice = prepared.vol_regime_arr[slice_start:slice_end]
    atr_slice = (
        prepared.atr_arr[slice_start:slice_end]
        if prepared.atr_arr is not None
        else None
    )

    eq_curve, *_ = run_vec_backtest(
        factors_slice,
        close_slice,
        open_slice,
        high_slice,
        low_slice,
        timing_slice,
        list(range(len(combo_factor_indices))),
        freq=int(config["backtest"]["freq"]),
        pos_size=int(config["backtest"]["pos_size"]),
        initial_capital=float(config["backtest"]["initial_capital"]),
        commission_rate=float(config["backtest"]["commission_rate"]),
        lookback=lookback,
        target_vol=float(
            config["backtest"]["risk_control"]["dynamic_leverage"]["target_vol"]
        ),
        vol_window=int(
            config["backtest"]["risk_control"]["dynamic_leverage"]["vol_window"]
        ),
        dynamic_leverage_enabled=bool(
            config["backtest"]["risk_control"]["dynamic_leverage"]["enabled"]
        ),
        vol_regime_arr=vol_slice,
        use_atr_stop=config["backtest"]["risk_control"].get("stop_method", "fixed")
        == "atr",
        trailing_stop_pct=float(
            config["backtest"]["risk_control"].get("trailing_stop_pct", 0.0)
        ),
        atr_arr=atr_slice,
        atr_multiplier=float(
            config["backtest"]["risk_control"].get("atr_multiplier", 3.0)
        ),
        stop_on_rebalance_only=bool(
            config["backtest"]["risk_control"].get(
                "stop_check_on_rebalance_only", False
            )
        ),
        individual_trend_arr=None,
        individual_trend_enabled=bool(
            config["backtest"]["timing"]
            .get("individual_timing", {})
            .get("enabled", False)
        ),
        profit_ladders=config["backtest"]["risk_control"].get("profit_ladders", []),
        circuit_breaker_day=float(
            config["backtest"]["risk_control"]
            .get("circuit_breaker", {})
            .get("max_drawdown_day", 0.0)
        ),
        circuit_breaker_total=float(
            config["backtest"]["risk_control"]
            .get("circuit_breaker", {})
            .get("max_drawdown_total", 0.0)
        ),
        circuit_recovery_days=int(
            config["backtest"]["risk_control"]
            .get("circuit_breaker", {})
            .get("recovery_days", 5)
        ),
        cooldown_days=int(config["backtest"]["risk_control"].get("cooldown_days", 0)),
        leverage_cap=float(config["backtest"]["risk_control"].get("leverage_cap", 1.0)),
    )

    rel_start = max(0, start_idx - slice_start)
    eq_recent = np.asarray(eq_curve[rel_start:], dtype=float)

    metrics = compute_period_metrics(eq_recent)

    # Daily returns for correlation
    eq_recent = np.maximum(eq_recent, 1e-6)
    daily_ret = np.diff(eq_recent) / eq_recent[:-1]
    return metrics, daily_ret


def corr_abs(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 5 or len(b) < 5:
        return 0.0
    n = min(len(a), len(b))
    x = a[-n:]
    y = b[-n:]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    c = float(np.corrcoef(x, y)[0, 1])
    if np.isnan(c):
        return 0.0
    return abs(c)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def greedy_select(
    df: pd.DataFrame,
    returns_by_combo: dict[str, np.ndarray],
    factors_by_combo: dict[str, set[str]],
    k: int,
    lambda_corr: float,
    lambda_overlap: float,
    max_jaccard: float,
    max_corr: float | None,
) -> list[str]:
    ordered = df.sort_values("score_recent", ascending=False)["combo"].tolist()
    selected: list[str] = []

    for combo in ordered:
        if len(selected) >= k:
            break
        if combo not in returns_by_combo:
            continue
        if len(selected) == 0:
            selected.append(combo)
            continue

        # Enforce overlap constraint (skip too-similar)
        fset = factors_by_combo.get(combo, set())
        if (
            max(jaccard(fset, factors_by_combo.get(s, set())) for s in selected)
            > max_jaccard
        ):
            continue

        # Score with diversification penalty
        avg_corr = float(
            np.mean(
                [
                    corr_abs(returns_by_combo[combo], returns_by_combo[s])
                    for s in selected
                ]
            )
        )
        max_overlap = float(
            max(jaccard(fset, factors_by_combo.get(s, set())) for s in selected)
        )

        df.loc[df["combo"] == combo, "_avg_corr_tmp"] = avg_corr
        df.loc[df["combo"] == combo, "_max_overlap_tmp"] = max_overlap

    # Select iteratively with full objective
    selected = []
    remaining = set(ordered)

    # Always take best first
    while remaining:
        best = max(
            remaining,
            key=lambda c: float(df.loc[df["combo"] == c, "score_recent"].iloc[0]),
        )
        if best in returns_by_combo:
            selected = [best]
            remaining.remove(best)
            break
        remaining.remove(best)

    while len(selected) < k and remaining:
        best_combo = None
        best_obj = -1e18

        for combo in list(remaining):
            if combo not in returns_by_combo:
                continue

            fset = factors_by_combo.get(combo, set())
            max_ov = max(
                jaccard(fset, factors_by_combo.get(s, set())) for s in selected
            )
            if max_ov > max_jaccard:
                continue

            if max_corr is not None:
                max_c = float(
                    max(
                        corr_abs(returns_by_combo[combo], returns_by_combo[s])
                        for s in selected
                    )
                )
                if max_c > max_corr:
                    continue

            avg_corr = float(
                np.mean(
                    [
                        corr_abs(returns_by_combo[combo], returns_by_combo[s])
                        for s in selected
                    ]
                )
            )
            base = float(df.loc[df["combo"] == combo, "score_recent"].iloc[0])

            obj = base - lambda_corr * avg_corr - lambda_overlap * max_ov
            if obj > best_obj:
                best_obj = obj
                best_combo = combo

        if best_combo is None:
            # If everything is too similar, relax ONLY the overlap constraint first.
            # If max_corr is set, keep it as a hard constraint; if no candidate satisfies it, stop early.
            fallback_combo = None
            fallback_obj = -1e18
            for combo in list(remaining):
                if combo not in returns_by_combo:
                    continue
                if max_corr is not None:
                    max_c = float(
                        max(
                            corr_abs(returns_by_combo[combo], returns_by_combo[s])
                            for s in selected
                        )
                    )
                    if max_c > max_corr:
                        continue
                avg_corr = float(
                    np.mean(
                        [
                            corr_abs(returns_by_combo[combo], returns_by_combo[s])
                            for s in selected
                        ]
                    )
                )
                base = float(df.loc[df["combo"] == combo, "score_recent"].iloc[0])
                obj = base - lambda_corr * avg_corr
                if obj > fallback_obj:
                    fallback_obj = obj
                    fallback_combo = combo

            if fallback_combo is None:
                break

            best_combo = fallback_combo

        selected.append(best_combo)
        remaining.remove(best_combo)

    return selected


def build_report(
    out_dir: Path,
    selected_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    basket_metrics: dict[str, float],
    data_end_date: pd.Timestamp,
    window: int,
) -> None:
    lines: list[str] = []
    lines.append(f"# 策略组合推荐（v3.2 封板）")
    lines.append("")
    lines.append(f"- 数据截至: {data_end_date.date()}（当前 workspace 数据最后一日）")
    lines.append(f"- 近窗口: 最近 {window} 个交易日")
    lines.append(f"- 组合规模: {len(selected_df)} 策略（等权示例）")
    lines.append("")

    lines.append("## 1) 组合近窗口表现（VEC 复算）")
    lines.append("")
    lines.append(f"- 近窗口收益: {basket_metrics['ret']*100:.2f}%")
    lines.append(f"- 近窗口年化收益(估): {basket_metrics['ann_ret']*100:.2f}%")
    lines.append(f"- 近窗口最大回撤: {basket_metrics['max_dd']*100:.2f}%")
    lines.append(f"- 近窗口 Sharpe: {basket_metrics['sharpe']:.2f}")
    lines.append("")

    lines.append("## 2) 推荐策略清单")
    lines.append("")
    lines.append(
        "字段说明：recent_* 为 VEC 最近窗口；bt_* 为封板 BT ground truth；holdout_trail_* 为封板 holdout 尾部窗口指标。"
    )
    lines.append("")

    show_cols = [
        "combo",
        "score_recent",
        "recent_ret",
        "recent_sharpe",
        "recent_max_dd",
        "bt_holdout_return",
        "bt_max_drawdown",
        "bt_profit_factor",
        "bt_win_rate",
    ]
    show_cols = [c for c in show_cols if c in selected_df.columns]

    table = selected_df[show_cols].copy()
    if "recent_ret" in table.columns:
        table["recent_ret"] = (table["recent_ret"] * 100).round(2).astype(str) + "%"
    if "recent_max_dd" in table.columns:
        table["recent_max_dd"] = (table["recent_max_dd"] * 100).round(2).astype(
            str
        ) + "%"
    if "bt_holdout_return" in table.columns:
        table["bt_holdout_return"] = (table["bt_holdout_return"] * 100).round(2).astype(
            str
        ) + "%"
    if "bt_max_drawdown" in table.columns:
        table["bt_max_drawdown"] = (table["bt_max_drawdown"] * 100).round(2).astype(
            str
        ) + "%"
    if "bt_win_rate" in table.columns:
        table["bt_win_rate"] = (table["bt_win_rate"] * 100).round(2).astype(str) + "%"

    lines.append(table.to_markdown(index=False))
    lines.append("")

    lines.append("## 3) 策略间相关性（近窗口日收益，|corr|）")
    lines.append("")
    lines.append("建议：组合内尽量避免 |corr| 长期 > 0.85 的策略对（高度同质）。")
    lines.append("")
    lines.append(corr_df.round(3).to_markdown())
    lines.append("")

    (out_dir / "basket_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sealed-dir", type=str, required=True, help="sealed_strategies/v3.2_YYYYMMDD"
    )
    ap.add_argument("--config", type=str, default="configs/combo_wfo_config.yaml")
    ap.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="override data end date (YYYY-MM-DD). If omitted, uses config data.end_date.",
    )
    ap.add_argument(
        "--window", type=int, default=120, help="recent window (trading days)"
    )
    ap.add_argument("--k", type=int, default=8, help="basket size")
    ap.add_argument("--lambda-corr", type=float, default=0.70)
    ap.add_argument("--lambda-overlap", type=float, default=0.35)
    ap.add_argument("--max-jaccard", type=float, default=0.85)
    ap.add_argument(
        "--max-corr",
        type=float,
        default=None,
        help="hard constraint: max pairwise |corr| with already-selected strategies (e.g. 0.85)",
    )
    ap.add_argument(
        "--min-bt-holdout",
        type=float,
        default=0.0,
        help="gate: bt_holdout_return >= this",
    )
    args = ap.parse_args()

    sealed_dir = Path(args.sealed_dir)
    artifacts = sealed_dir / "artifacts"
    if not artifacts.exists():
        raise FileNotFoundError(f"artifacts not found under {sealed_dir}")

    cfg = load_config(ROOT / args.config)

    prod_all = pd.read_parquet(artifacts / "production_all_candidates.parquet")

    # Basic gates
    df = prod_all.copy()
    if "bt_pass" in df.columns:
        df = df[df["bt_pass"] == True].copy()  # noqa: E712
    if "bt_holdout_return" in df.columns and args.min_bt_holdout is not None:
        df = df[df["bt_holdout_return"] >= float(args.min_bt_holdout)].copy()

    df = df.reset_index(drop=True)

    # Prepare data only once
    lookback = int(cfg["backtest"]["lookback"])
    min_days = int(max(args.window + lookback + 260, 700))
    prepared = prepare_data_for_recent_window(
        cfg, end_date=args.end_date, min_days=min_days
    )

    data_end_date = pd.Timestamp(prepared.dates.max())
    if len(prepared.dates) < args.window + lookback + 5:
        raise ValueError(
            f"Not enough data: T={len(prepared.dates)} window={args.window} lookback={lookback}"
        )

    end_idx = len(prepared.dates)
    start_idx = max(0, end_idx - args.window)

    factor_to_idx = {f: i for i, f in enumerate(prepared.factor_names)}

    returns_by_combo: dict[str, np.ndarray] = {}
    factors_by_combo: dict[str, set[str]] = {}
    vec_rows: list[dict] = []

    for combo in tqdm(df["combo"].tolist(), desc=f"VEC recent-window ({args.window}d)"):
        f_names = parse_combo(combo)
        if any(f not in factor_to_idx for f in f_names):
            continue

        combo_factor_indices = [factor_to_idx[f] for f in f_names]
        metrics, daily_ret = run_vec_recent_window(
            combo_factor_indices=combo_factor_indices,
            prepared=prepared,
            config=cfg,
            start_idx=start_idx,
            end_idx=end_idx,
        )

        returns_by_combo[combo] = daily_ret
        factors_by_combo[combo] = set(f_names)
        vec_rows.append(
            {
                "combo": combo,
                "recent_ret": metrics["ret"],
                "recent_ann_ret": metrics["ann_ret"],
                "recent_max_dd": metrics["max_dd"],
                "recent_sharpe": metrics["sharpe"],
            }
        )

    vec_df = pd.DataFrame(vec_rows)

    merged = df.merge(vec_df, on="combo", how="inner")

    # Recent score focuses on choppy market: reward Sharpe + return, penalize drawdown.
    merged["score_recent"] = (
        1.00 * _zscore(merged["recent_ret"])
        + 0.85 * _zscore(merged["recent_sharpe"])
        - 0.90 * _zscore(merged["recent_max_dd"].abs())
    )

    # Optional: add sealed tail-window signal if present
    if "holdout_trail_120d_sharpe" in merged.columns:
        merged["score_recent"] += 0.30 * _zscore(
            merged["holdout_trail_120d_sharpe"].fillna(0.0)
        )
    if "holdout_trail_120d_return" in merged.columns:
        merged["score_recent"] += 0.20 * _zscore(
            merged["holdout_trail_120d_return"].fillna(0.0)
        )

    # Pick basket with diversification
    selected = greedy_select(
        df=merged,
        returns_by_combo=returns_by_combo,
        factors_by_combo=factors_by_combo,
        k=int(args.k),
        lambda_corr=float(args.lambda_corr),
        lambda_overlap=float(args.lambda_overlap),
        max_jaccard=float(args.max_jaccard),
        max_corr=(float(args.max_corr) if args.max_corr is not None else None),
    )

    selected_df = merged[merged["combo"].isin(selected)].copy()
    selected_df = (
        selected_df.set_index("combo").loc[selected].reset_index()
    )  # keep order

    # Correlation matrix
    rets = [returns_by_combo[c] for c in selected]
    min_len = min(len(r) for r in rets)
    X = np.vstack([r[-min_len:] for r in rets])
    corr = np.corrcoef(X)
    corr_df = pd.DataFrame(
        np.abs(corr),
        index=[f"S{i+1}" for i in range(len(selected))],
        columns=[f"S{i+1}" for i in range(len(selected))],
    )

    # Basket equity (equal weight, daily)
    basket_daily = np.mean(X, axis=0)
    eq = np.ones(len(basket_daily) + 1)
    eq[1:] = np.cumprod(1 + basket_daily)
    basket_metrics = compute_period_metrics(eq)

    # Save outputs
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / f"strategy_basket_selection_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_df.to_csv(out_dir / "selected_strategies.csv", index=False)
    corr_df.to_csv(out_dir / "selected_corr.csv", index=True)

    build_report(
        out_dir=out_dir,
        selected_df=selected_df,
        corr_df=corr_df,
        basket_metrics=basket_metrics,
        data_end_date=data_end_date,
        window=int(args.window),
    )

    print(f"\n✅ Done. Output: {out_dir}")
    print(
        f"Data ends at: {data_end_date.date()} | Window: {args.window} | Basket size: {len(selected_df)}"
    )


if __name__ == "__main__":
    main()
