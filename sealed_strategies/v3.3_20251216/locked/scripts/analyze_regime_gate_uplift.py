#!/usr/bin/env python3
"""Analyze Regime Gate uplift (A/B: gate on vs off).

This script evaluates a list of factor combos under:
- Gate OFF: exposure = 1.0
- Gate ON : exposure = timing * regime_gate (shifted, no lookahead)

Outputs a markdown report under results/diagnostics/.

Usage
  uv run python scripts/analyze_regime_gate_uplift.py --top-n 10
  uv run python scripts/analyze_regime_gate_uplift.py --combos "A + B + C" "X + Y"
  uv run python scripts/analyze_regime_gate_uplift.py --combos-path results/final_triple_validation_xxx/final_candidates.parquet --top-n 20
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent

# Project imports
import sys

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import shift_timing_signal
from etf_strategy.regime_gate import compute_regime_gate_arr, gate_stats

from batch_vec_backtest import run_vec_backtest


def _read_config() -> dict[str, Any]:
    config_path = ROOT / "configs/combo_wfo_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def _latest_dir(glob_pat: str) -> Path:
    base = ROOT / "results"
    dirs = sorted([d for d in base.glob(glob_pat) if d.is_dir() and not d.is_symlink()])
    if not dirs:
        raise FileNotFoundError(f"No dir matched: results/{glob_pat}")
    return dirs[-1]


def _max_drawdown(equity: np.ndarray) -> float:
    equity = np.asarray(equity, dtype=np.float64)
    if equity.size <= 1:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    denom = np.where(running_max == 0.0, np.nan, running_max)
    dd = (equity - running_max) / denom
    dd = dd[np.isfinite(dd)]
    return float(abs(dd.min())) if dd.size else 0.0


def _window_metrics(
    equity_curve: np.ndarray, start_idx: int, end_idx: int | None = None
) -> dict[str, float]:
    equity = np.asarray(equity_curve, dtype=np.float64)
    if equity.size == 0:
        return {"ret": 0.0, "mdd": 0.0, "calmar": 0.0}

    start = max(int(start_idx), 0)
    end = equity.size if end_idx is None else min(int(end_idx), equity.size)
    if end - start <= 1:
        return {"ret": 0.0, "mdd": 0.0, "calmar": 0.0}

    window = equity[start:end]
    window = window[np.isfinite(window)]
    if window.size <= 1 or window[0] == 0:
        return {"ret": 0.0, "mdd": 0.0, "calmar": 0.0}

    ret = float((window[-1] - window[0]) / window[0])
    mdd = _max_drawdown(window)
    calmar = float(ret / mdd) if mdd > 1e-12 else 0.0
    return {"ret": ret, "mdd": mdd, "calmar": calmar}


@dataclass(frozen=True)
class EvalResult:
    combo: str
    period: str
    gate: str
    total_return: float
    max_drawdown: float
    calmar: float
    num_trades: int


def _eval_combo(
    combo: str,
    *,
    factor_index_map: dict[str, int],
    all_factors_stack: np.ndarray,
    close_prices: np.ndarray,
    open_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    timing_arr: np.ndarray,
    backtest_config: dict[str, Any],
    freq: int,
    pos_size: int,
    holdout_start_idx: int,
) -> tuple[list[EvalResult], np.ndarray]:
    factors = [f.strip() for f in combo.split(" + ")]
    combo_indices = [factor_index_map[f] for f in factors]
    current_factors = all_factors_stack[..., combo_indices]
    current_factor_indices = list(range(len(combo_indices)))

    equity_curve, total_return, _wr, _pf, num_trades, _diag, risk = run_vec_backtest(
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

    full_ret = float(total_return)
    full_mdd = float(risk.get("max_drawdown", 0.0))
    full_calmar = float(risk.get("calmar_ratio", 0.0))

    hold = _window_metrics(equity_curve, start_idx=holdout_start_idx)

    results = [
        EvalResult(
            combo=combo,
            period="full",
            gate="",
            total_return=full_ret,
            max_drawdown=full_mdd,
            calmar=full_calmar,
            num_trades=int(num_trades),
        ),
        EvalResult(
            combo=combo,
            period="holdout",
            gate="",
            total_return=float(hold["ret"]),
            max_drawdown=float(hold["mdd"]),
            calmar=float(hold["calmar"]),
            num_trades=int(num_trades),
        ),
    ]

    return results, np.asarray(equity_curve, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A/B test regime gate uplift for selected combos"
    )
    parser.add_argument(
        "--combos",
        nargs="*",
        default=None,
        help="Combos as strings (each must use ' + ' separator)",
    )
    parser.add_argument(
        "--combos-path",
        type=str,
        default=None,
        help="Path to parquet/csv containing 'combo' column",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Top-N combos to evaluate from combos-path (default: 10)",
    )
    args = parser.parse_args()

    cfg = _read_config()
    backtest_config = cfg.get("backtest", {})

    # Core params (locked)
    FREQ = 3
    POS_SIZE = 2
    EXTREME_THRESHOLD = -0.1
    EXTREME_POSITION = 0.1

    start_date = cfg["data"]["start_date"]
    end_date = cfg["data"]["end_date"]
    training_end = cfg["data"].get("training_end_date")

    # Determine combos
    combos: list[str]
    if args.combos:
        combos = [str(c).strip() for c in args.combos if str(c).strip()]
    else:
        combos_path = (
            Path(args.combos_path)
            if args.combos_path
            else (_latest_dir("final_triple_validation_*") / "final_candidates.parquet")
        )
        df = (
            pd.read_parquet(combos_path)
            if combos_path.suffix.lower() != ".csv"
            else pd.read_csv(combos_path)
        )
        if "combo" not in df.columns:
            raise ValueError(f"Missing 'combo' in {combos_path}")
        combos = df["combo"].astype(str).head(int(args.top_n)).tolist()

    # Load OHLCV (full)
    loader = DataLoader(
        data_dir=cfg["data"].get("data_dir"), cache_dir=cfg["data"].get("cache_dir")
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=cfg["data"]["symbols"], start_date=start_date, end_date=end_date
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

    factor_index_map = {name: idx for idx, name in enumerate(factor_names_list)}

    # Timing
    timing_module = LightTimingModule(
        extreme_threshold=EXTREME_THRESHOLD, extreme_position=EXTREME_POSITION
    )
    timing_series = timing_module.compute_position_ratios(ohlcv["close"])
    timing_arr_raw = (
        timing_series.reindex(all_dates).fillna(1.0).values.astype(np.float64)
    )
    timing_arr_base = shift_timing_signal(timing_arr_raw)

    # Holdout start idx (strictly > training_end)
    if training_end:
        te = pd.Timestamp(training_end)
        holdout_start_candidates = np.where(all_dates > te)[0]
        holdout_start_idx = (
            int(holdout_start_candidates[0])
            if holdout_start_candidates.size
            else len(all_dates)
        )
    else:
        holdout_start_idx = len(all_dates)

    # Gate arrays
    backtest_off = dict(backtest_config)
    backtest_off["regime_gate"] = dict((backtest_config.get("regime_gate") or {}))
    backtest_off["regime_gate"]["enabled"] = False

    gate_off = np.ones(len(all_dates), dtype=np.float64)
    gate_on = compute_regime_gate_arr(
        close_df=ohlcv["close"], dates=all_dates, backtest_config=backtest_config
    )

    # Evaluate
    rows: list[dict[str, Any]] = []

    for combo in combos:
        for gate_name, gate_arr, bt_cfg in (
            ("off", gate_off, backtest_off),
            ("on", gate_on, backtest_config),
        ):
            timing_arr = (
                timing_arr_base.astype(np.float64) * gate_arr.astype(np.float64)
            ).astype(np.float64)
            try:
                results, _eq = _eval_combo(
                    combo,
                    factor_index_map=factor_index_map,
                    all_factors_stack=all_factors_stack,
                    close_prices=close_prices,
                    open_prices=open_prices,
                    high_prices=high_prices,
                    low_prices=low_prices,
                    timing_arr=timing_arr,
                    backtest_config=bt_cfg,
                    freq=FREQ,
                    pos_size=POS_SIZE,
                    holdout_start_idx=holdout_start_idx,
                )
                for r in results:
                    rows.append(
                        {
                            "combo": r.combo,
                            "period": r.period,
                            "gate": gate_name,
                            "ret": r.total_return,
                            "mdd": r.max_drawdown,
                            "calmar": r.calmar,
                            "trades": r.num_trades,
                        }
                    )
            except Exception as e:
                rows.append(
                    {
                        "combo": combo,
                        "period": "error",
                        "gate": gate_name,
                        "error": str(e),
                    }
                )

    out_df = pd.DataFrame(rows)

    # Pivot for deltas
    piv = out_df[out_df["period"].isin(["full", "holdout"])].pivot_table(
        index=["combo", "period"],
        columns="gate",
        values=["ret", "mdd", "calmar"],
        aggfunc="first",
    )
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.reset_index()
    piv["ret_uplift"] = piv["ret_on"] - piv["ret_off"]
    piv["mdd_delta"] = piv["mdd_on"] - piv["mdd_off"]
    piv["calmar_uplift"] = piv["calmar_on"] - piv["calmar_off"]

    # Report
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / "diagnostics" / f"gate_uplift_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "gate_uplift_report.md"

    s_on = gate_stats(gate_on)
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Regime Gate Uplift (A/B)\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Config\n")
        f.write(f"- Date range: {start_date} -> {end_date}\n")
        f.write(f"- training_end_date: {training_end}\n")
        f.write(
            f"- Gate stats (ON): mean={s_on['mean']:.3f}, min={s_on['min']:.3f}, max={s_on['max']:.3f}\n\n"
        )

        f.write("## Results (gate ON vs OFF)\n")
        view = piv.sort_values(
            ["period", "calmar_uplift"], ascending=[True, False]
        ).copy()
        # Pretty
        for c in [
            "ret_off",
            "ret_on",
            "ret_uplift",
            "mdd_off",
            "mdd_on",
            "mdd_delta",
            "calmar_off",
            "calmar_on",
            "calmar_uplift",
        ]:
            view[c] = view[c].astype(float)
        f.write(
            view[
                [
                    "combo",
                    "period",
                    "ret_off",
                    "ret_on",
                    "ret_uplift",
                    "mdd_off",
                    "mdd_on",
                    "mdd_delta",
                    "calmar_off",
                    "calmar_on",
                    "calmar_uplift",
                ]
            ].to_markdown(index=False)
        )
        f.write("\n")

    out_df.to_parquet(out_dir / "gate_uplift_raw.parquet", index=False)
    piv.to_parquet(out_dir / "gate_uplift_pivot.parquet", index=False)

    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
