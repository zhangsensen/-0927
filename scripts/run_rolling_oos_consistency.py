#!/usr/bin/env python3
"""Rolling OOS consistency validation (Parquet-first).

目的
- 用“分段（按月/季度/年）”的方式，把同一套固定交易规则/因子组合在不同市场风格段的表现拆开看。
- 用“一致性指标”替代单段 holdout 的弱检验：例如分段胜率、最差段收益/Calmar、分段中位数等。

注意
- 这里的“滚动 OOS”指的是：把完整回测曲线按时间切片统计一致性。
  本项目的信号生成并不做参数拟合（仅做组合选择），因此这种分段一致性对“跨风格稳健性”非常有用。

输出
- results/rolling_oos_consistency_YYYYMMDD_HHMMSS/rolling_oos_summary.parquet
- （可选）results/.../rolling_oos_segments.parquet  # 体积可能较大
- results/.../rolling_oos_report.md

用法（建议先小规模验证）
  uv run python scripts/run_rolling_oos_consistency.py \
    --input results/vec_from_wfo_xxx/full_space_results.parquet \
    --top-n 2000 \
    --segment Q \
    --n-jobs 32 --prefer threads

全量（62k）会更久，但通常仍是分钟级。
"""

from __future__ import annotations

import sys
import argparse
from dataclasses import dataclass
from pathlib import Path
import os
from typing import Any

import numpy as np
import pandas as pd
import yaml
import warnings
from joblib import Parallel, delayed

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from etf_strategy.core.utils.run_meta import write_step_meta
from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import shift_timing_signal
from etf_strategy.regime_gate import compute_regime_gate_arr

from batch_vec_backtest import run_vec_backtest

warnings.filterwarnings("ignore")


def _resolve_n_jobs(n_jobs: int) -> int:
    """Resolve a safe worker count.

    We avoid using all logical CPUs by default because this workstation may run
    a VM (qemu) and other services, and maxing out all cores can cause thermal
    throttling or instability.
    """
    if int(n_jobs) > 0:
        return int(n_jobs)
    cpu = os.cpu_count() or 1
    # Use ~60% cores, capped at 16, minimum 1.
    return max(1, min(16, int(cpu * 0.6)))


def _read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    try:
        return pd.read_parquet(p)
    except Exception:
        return pd.read_csv(p)


def _max_drawdown(equity: np.ndarray) -> float:
    equity = np.asarray(equity, dtype=np.float64)
    if equity.size <= 1:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    denom = np.where(running_max == 0.0, np.nan, running_max)
    dd = (equity - running_max) / denom
    dd = dd[np.isfinite(dd)]
    return float(abs(dd.min())) if dd.size else 0.0


def _sharpe_from_equity(equity: np.ndarray) -> float:
    equity = np.asarray(equity, dtype=np.float64)
    if equity.size <= 2:
        return 0.0
    rets = np.diff(equity) / equity[:-1]
    rets = rets[np.isfinite(rets)]
    if rets.size <= 1:
        return 0.0
    std = float(np.std(rets, ddof=1))
    if std <= 1e-12:
        return 0.0
    return float(np.mean(rets) / std * np.sqrt(252.0))


def _window_metrics(
    equity_curve: np.ndarray, start_idx: int, end_idx: int
) -> dict[str, float]:
    equity = np.asarray(equity_curve, dtype=np.float64)
    start = max(int(start_idx), 0)
    end = min(int(end_idx), equity.size)
    if end - start <= 1:
        return {"ret": 0.0, "mdd": 0.0, "sharpe": 0.0, "calmar": 0.0}

    window = equity[start:end]
    window = window[np.isfinite(window)]
    if window.size <= 1:
        return {"ret": 0.0, "mdd": 0.0, "sharpe": 0.0, "calmar": 0.0}

    ret = float((window[-1] - window[0]) / window[0]) if window[0] != 0 else 0.0
    mdd = _max_drawdown(window)
    sharpe = _sharpe_from_equity(window)
    calmar = float(ret / mdd) if mdd > 1e-12 else 0.0
    return {"ret": ret, "mdd": mdd, "sharpe": sharpe, "calmar": calmar}


@dataclass(frozen=True)
class Segment:
    label: str
    start_idx: int
    end_idx: int
    is_holdout: bool


def _build_segments(
    all_dates: pd.Index,
    effective_start_idx: int,
    segment: str,
    training_end: str | None,
) -> list[Segment]:
    seg = segment.upper().strip()
    if seg not in {"M", "Q", "Y"}:
        raise ValueError("--segment must be one of: M, Q, Y")

    dates = pd.DatetimeIndex(all_dates)
    if dates.size == 0:
        return []

    # Define period labels
    if seg == "M":
        periods = dates.to_period("M")
    elif seg == "Q":
        periods = dates.to_period("Q")
    else:
        periods = dates.to_period("Y")

    training_end_ts = pd.Timestamp(training_end) if training_end else None

    segments: list[Segment] = []
    start = int(effective_start_idx)
    while start < dates.size - 1:
        p = periods[start]
        # find end index where period changes
        end = start + 1
        while end < dates.size and periods[end] == p:
            end += 1
        if end - start > 1:
            label = str(p)
            # is_holdout: segment starts strictly after training_end
            is_holdout = False
            if training_end_ts is not None:
                is_holdout = bool(dates[start] > training_end_ts)
            segments.append(
                Segment(
                    label=label, start_idx=start, end_idx=end, is_holdout=is_holdout
                )
            )
        start = end

    return segments


def _summarize_segments(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {
            "segment_count": 0,
            "segment_positive_rate": 0.0,
            "segment_worst_return": 0.0,
            "segment_worst_calmar": 0.0,
            "segment_median_return": 0.0,
            "segment_median_calmar": 0.0,
        }

    rets = np.array([m["ret"] for m in metrics], dtype=float)
    calmars = np.array([m["calmar"] for m in metrics], dtype=float)

    return {
        "segment_count": int(len(metrics)),
        "segment_positive_rate": float((rets > 0).mean()),
        "segment_worst_return": float(np.nanmin(rets)),
        "segment_worst_calmar": float(np.nanmin(calmars)),
        "segment_median_return": float(np.nanmedian(rets)),
        "segment_median_calmar": float(np.nanmedian(calmars)),
    }


def process_combo(
    combo_str: str,
    factor_index_map: dict[str, int],
    all_factors_stack: np.ndarray,
    close_prices: np.ndarray,
    open_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    timing_arr: np.ndarray,
    backtest_config: dict[str, Any],
    segments: list[Segment],
    freq: int,
    pos_size: int,
    write_segments: bool,
):
    factors_in_combo = [f.strip() for f in combo_str.split(" + ")]
    try:
        combo_indices = [factor_index_map[f] for f in factors_in_combo]
    except KeyError:
        return None

    current_factors = all_factors_stack[..., combo_indices]
    current_factor_indices = list(range(len(combo_indices)))

    try:
        (
            equity_curve,
            total_return,
            win_rate,
            profit_factor,
            num_trades,
            rounding_diag,
            risk_metrics,
        ) = run_vec_backtest(
            current_factors,
            close_prices,
            open_prices,
            high_prices,
            low_prices,
            timing_arr,
            current_factor_indices,
            freq=freq,
            pos_size=pos_size,
            initial_capital=float(backtest_config["initial_capital"]),
            commission_rate=float(backtest_config["commission_rate"]),
            lookback=int(backtest_config["lookback"]),
            trailing_stop_pct=0.0,
            stop_on_rebalance_only=True,
        )

        seg_rows = []
        seg_metrics_all: list[dict[str, float]] = []
        seg_metrics_holdout: list[dict[str, float]] = []

        for s in segments:
            m = _window_metrics(equity_curve, start_idx=s.start_idx, end_idx=s.end_idx)
            seg_metrics_all.append(m)
            if s.is_holdout:
                seg_metrics_holdout.append(m)
            if write_segments:
                seg_rows.append(
                    {
                        "combo": combo_str,
                        "segment": s.label,
                        "is_holdout": bool(s.is_holdout),
                        "segment_return": float(m["ret"]),
                        "segment_max_drawdown": float(m["mdd"]),
                        "segment_sharpe": float(m["sharpe"]),
                        "segment_calmar": float(m["calmar"]),
                    }
                )

        summary = {
            "combo": combo_str,
            "size": int(len(combo_indices)),
            # keep full-period metrics (already computed by engine)
            "full_total_return": float(total_return),
            "full_num_trades": int(num_trades),
            "full_max_drawdown": float(risk_metrics.get("max_drawdown", 0.0)),
            "full_calmar_ratio": float(risk_metrics.get("calmar_ratio", 0.0)),
            "full_sharpe_ratio": float(risk_metrics.get("sharpe_ratio", 0.0)),
        }
        summary.update(
            {f"all_{k}": v for k, v in _summarize_segments(seg_metrics_all).items()}
        )
        summary.update(
            {
                f"holdout_{k}": v
                for k, v in _summarize_segments(seg_metrics_holdout).items()
            }
        )

        return summary, seg_rows

    except Exception:
        return None


def _write_report(out_dir: Path, summary_df: pd.DataFrame, top: int = 30) -> None:
    cols = [
        "combo",
        "size",
        "full_total_return",
        "full_max_drawdown",
        "full_calmar_ratio",
        "all_segment_positive_rate",
        "all_segment_worst_return",
        "all_segment_worst_calmar",
        "holdout_segment_positive_rate",
        "holdout_segment_worst_return",
        "holdout_segment_worst_calmar",
    ]
    cols = [c for c in cols if c in summary_df.columns]

    lines: list[str] = []
    lines.append("# Rolling OOS Consistency Report\n")
    lines.append(f"- Input rows: {len(summary_df)}\n")

    def table(df: pd.DataFrame, title: str) -> None:
        lines.append(f"\n## {title}\n")
        view = df[cols].head(top).copy()
        lines.append(view.to_markdown(index=False))

    # Rank by worst-segment calmar (robustness)
    if "all_segment_worst_calmar" in summary_df.columns:
        table(
            summary_df.sort_values("all_segment_worst_calmar", ascending=False),
            "Top by worst-segment Calmar (all segments)",
        )

    if "holdout_segment_worst_calmar" in summary_df.columns:
        table(
            summary_df.sort_values("holdout_segment_worst_calmar", ascending=False),
            "Top by worst-segment Calmar (holdout segments only)",
        )

    # Rank by positive rate
    if "all_segment_positive_rate" in summary_df.columns:
        table(
            summary_df.sort_values(
                ["all_segment_positive_rate", "all_segment_worst_calmar"],
                ascending=False,
            ),
            "Top by segment positive rate",
        )

    (out_dir / "rolling_oos_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rolling OOS consistency validation (Parquet-first)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径 (yaml)。默认使用环境变量 WFO_CONFIG_PATH 或 configs/combo_wfo_config.yaml",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="training results table (.parquet/.csv) containing at least 'combo'",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Only evaluate Top-N by vec_calmar_ratio (descending). Default: all rows.",
    )
    parser.add_argument(
        "--segment",
        type=str,
        default="Q",
        choices=["M", "Q", "Y"],
        help="Calendar segmentation: M/Q/Y",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel workers. -1 means safe default (not all cores).",
    )
    parser.add_argument(
        "--prefer",
        type=str,
        default="threads",
        choices=["threads", "processes"],
        help="joblib backend",
    )
    parser.add_argument(
        "--write-segments",
        action="store_true",
        help="Write per-segment table (can be large)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Override data end date for evaluation (YYYY-MM-DD). Use training_end_date to avoid holdout leakage.",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("📦 ROLLING OOS CONSISTENCY")
    print("=" * 80)
    print(f"Input: {args.input}")

    # Load config
    config_path = (
        Path(args.config)
        if args.config
        else Path(
            os.environ.get(
                "WFO_CONFIG_PATH", str(ROOT / "configs/combo_wfo_config.yaml")
            )
        )
    )
    with open(config_path) as f:
        config = yaml.safe_load(f)

    backtest_config = config.get("backtest", {})

    # Fixed trading rule
    FREQ = 3
    POS_SIZE = 2
    EXTREME_THRESHOLD = -0.1
    EXTREME_POSITION = 0.1

    training_end = config.get("data", {}).get("training_end_date", None)
    full_end_cfg = config["data"]["end_date"]
    full_end = args.end_date or full_end_cfg

    # Load results table
    df_in = _read_table(args.input)
    if "combo" not in df_in.columns:
        raise ValueError("Input table must contain 'combo' column")

    if args.top_n is not None:
        if "vec_calmar_ratio" not in df_in.columns:
            raise ValueError("--top-n requires 'vec_calmar_ratio' column in input")
        df_in = (
            df_in.sort_values("vec_calmar_ratio", ascending=False)
            .head(int(args.top_n))
            .copy()
        )

    combos = df_in["combo"].astype(str).tolist()
    print(f"Combos to evaluate: {len(combos)}")

    # Load full OHLCV (for regime segmentation across years)
    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=full_end,
    )

    # Factors
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)
    factor_names_list = sorted(
        raw_factors_df.columns.get_level_values(0).unique().tolist()
    )
    raw_factors = {fname: raw_factors_df[fname] for fname in factor_names_list}

    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)

    first_factor = std_factors[factor_names_list[0]]
    all_dates = first_factor.index
    etf_codes = first_factor.columns.tolist()

    all_factors_stack = np.stack(
        [std_factors[f].values for f in factor_names_list], axis=-1
    )

    close_prices = ohlcv["close"][etf_codes].ffill().bfill().values
    open_prices = ohlcv["open"][etf_codes].ffill().bfill().values
    high_prices = ohlcv["high"][etf_codes].ffill().bfill().values
    low_prices = ohlcv["low"][etf_codes].ffill().bfill().values

    # Timing signal
    timing_module = LightTimingModule(
        extreme_threshold=EXTREME_THRESHOLD, extreme_position=EXTREME_POSITION
    )
    timing_series = timing_module.compute_position_ratios(ohlcv["close"])
    timing_arr_raw = (
        timing_series.reindex(all_dates).fillna(1.0).values.astype(np.float64)
    )
    timing_arr = shift_timing_signal(timing_arr_raw)

    # Apply Regime Gate if enabled
    regime_gate_cfg = backtest_config.get("regime_gate", {})
    if regime_gate_cfg.get("enabled", False):
        print(
            f"🛡️ Regime Gate ENABLED (mode={regime_gate_cfg.get('mode', 'volatility')})"
        )
        gate_arr = compute_regime_gate_arr(
            close_df=ohlcv["close"], dates=all_dates, backtest_config=backtest_config
        )
        # Apply gate to timing signal
        timing_arr = timing_arr * gate_arr
    else:
        print("🛡️ Regime Gate DISABLED")

    effective_start_idx = int(backtest_config["lookback"])
    segments = _build_segments(
        all_dates, effective_start_idx, args.segment, training_end
    )
    print(
        f"Segmentation: {args.segment}, segments={len(segments)} (effective_start_idx={effective_start_idx})"
    )

    factor_index_map = {name: idx for idx, name in enumerate(factor_names_list)}

    # Parallel evaluation
    n_jobs = _resolve_n_jobs(int(args.n_jobs))
    print(f"Workers: {n_jobs} (requested: {args.n_jobs})")

    parallel_results = Parallel(
        n_jobs=n_jobs, prefer=args.prefer, verbose=5, batch_size=64
    )(
        delayed(process_combo)(
            combo_str,
            factor_index_map,
            all_factors_stack,
            close_prices,
            open_prices,
            high_prices,
            low_prices,
            timing_arr,
            backtest_config,
            segments,
            FREQ,
            POS_SIZE,
            bool(args.write_segments),
        )
        for combo_str in combos
    )

    summaries = []
    seg_rows_all = []
    for r in parallel_results:
        if r is None:
            continue
        s, seg_rows = r
        summaries.append(s)
        if args.write_segments and seg_rows:
            seg_rows_all.extend(seg_rows)

    summary_df = pd.DataFrame(summaries)

    # Merge original columns (train metrics etc.) for convenience
    merge_cols = [c for c in df_in.columns if c not in summary_df.columns]
    merged = df_in.merge(summary_df, on="combo", how="inner")

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / f"rolling_oos_consistency_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "rolling_oos_summary.parquet"
    merged.to_parquet(summary_path, index=False)
    print(f"Saved summary: {summary_path}")

    if args.write_segments:
        seg_df = pd.DataFrame(seg_rows_all)
        seg_path = out_dir / "rolling_oos_segments.parquet"
        seg_df.to_parquet(seg_path, index=False)
        print(f"Saved segments: {seg_path}")

    _write_report(out_dir, merged)
    print(f"Saved report: {out_dir / 'rolling_oos_report.md'}")

    write_step_meta(out_dir, step="rolling", inputs={"vec_results": str(args.input)}, config=str(args.config or "default"), extras={"combo_count": len(merged), "segment": str(args.segment)})


if __name__ == "__main__":
    main()
