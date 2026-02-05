#!/usr/bin/env python3
"""
Extend Validation for Sealed Strategies (v3.2)
----------------------------------------------
Re-evaluates the 152 sealed strategies on extended data (up to 2025-12-12).
Performs VEC + BT validation to ensure consistency and performance on the new "Holdout" period.

Metrics calculated:
- Train (2020-01-01 -> 2025-04-30): Should match sealed values.
- Holdout_Original (2025-05-01 -> 2025-10-14): Should match sealed values.
- Holdout_Extended (2025-05-01 -> 2025-12-12): The new "Cold Data" test.
- Recent_120d (2025-06-16 -> 2025-12-12): For recent regime fit.

Output:
- results/extended_validation_20251212/extended_candidates.parquet
- results/extended_validation_20251212/extended_report.md
"""

import sys
import argparse
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import (
    shift_timing_signal,
    generate_rebalance_schedule,
)
from etf_strategy.auditor.core.engine import GenericStrategy, PandasData
from scripts.batch_bt_backtest import run_bt_backtest
from scripts.batch_vec_backtest import run_vec_backtest, calculate_atr

# Constants
TRAIN_END_DATE = "2025-04-30"
HOLDOUT_ORIG_END_DATE = "2025-10-14"
EXTENDED_END_DATE = "2025-12-12"


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_metrics_from_daily(daily_ret: pd.Series) -> dict:
    if daily_ret.empty:
        return {
            "ret": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
        }

    total_ret = (1 + daily_ret).prod() - 1

    # Sharpe
    mu = daily_ret.mean()
    sigma = daily_ret.std(ddof=0)
    sharpe = (mu / sigma * np.sqrt(252)) if sigma > 1e-9 else 0.0

    # Max DD
    cum = (1 + daily_ret).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    # Win Rate
    wins = (daily_ret > 0).sum()
    total = len(daily_ret)
    win_rate = wins / total if total > 0 else 0.0

    # Profit Factor (approximate from daily returns, not trade-based)
    gains = daily_ret[daily_ret > 0].sum()
    losses = abs(daily_ret[daily_ret < 0].sum())
    pf = gains / losses if losses > 1e-9 else 0.0

    return {
        "ret": total_ret,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "win_rate": win_rate,
        "profit_factor": pf,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sealed-dir", type=str, required=True, help="Path to sealed strategies dir"
    )
    parser.add_argument("--config", type=str, default="configs/combo_wfo_config.yaml")
    args = parser.parse_args()

    sealed_dir = Path(args.sealed_dir)
    artifacts_dir = sealed_dir / "artifacts"
    candidates_path = artifacts_dir / "production_all_candidates.parquet"

    if not candidates_path.exists():
        print(f"Error: Candidates file not found at {candidates_path}")
        sys.exit(1)

    print(f"Loading candidates from {candidates_path}...")
    candidates_df = pd.read_parquet(candidates_path)
    combos = candidates_df["combo"].unique().tolist()
    print(f"Found {len(combos)} unique strategies.")

    # Load Config & Override End Date
    cfg = load_config(ROOT / args.config)
    cfg["data"]["end_date"] = EXTENDED_END_DATE

    # Prepare Data
    print("Preparing data up to 2025-12-12...")
    loader = DataLoader(
        data_dir=cfg["data"].get("data_dir"),
        cache_dir=cfg["data"].get("cache_dir"),
    )

    # Load OHLCV
    ohlcv = loader.load_ohlcv(
        etf_codes=cfg["data"]["symbols"],
        start_date=cfg["data"]["start_date"],
        end_date=EXTENDED_END_DATE,
    )

    # Compute Factors
    print("Computing factors...")
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)

    # Process Cross-Section (Standardization)
    print("Processing cross-section...")
    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(
        {
            f: raw_factors_df[f]
            for f in raw_factors_df.columns.get_level_values(0).unique()
        }
    )

    # Prepare Timing & Vol Regime
    print("Computing timing and regime...")
    timing_module = LightTimingModule(
        extreme_threshold=cfg["backtest"]["timing"]["extreme_threshold"],
        extreme_position=cfg["backtest"]["timing"]["extreme_position"],
    )
    timing_series = timing_module.compute_position_ratios(ohlcv["close"])
    timing_series = timing_series.reindex(ohlcv["close"].index).fillna(1.0)
    # Shift timing signal by 1 day to avoid lookahead
    timing_arr = shift_timing_signal(timing_series.values)
    timing_series_shifted = pd.Series(timing_arr, index=timing_series.index)

    # Vol Regime
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
    vol_regime_series = exposure_s.reindex(ohlcv["close"].index).fillna(1.0)

    # Prepare Data Feeds for BT
    data_feeds = {}
    for code in ohlcv["close"].columns:
        df = pd.DataFrame(
            {
                "open": ohlcv["open"][code],
                "high": ohlcv["high"][code],
                "low": ohlcv["low"][code],
                "close": ohlcv["close"][code],
                "volume": ohlcv["volume"][code],
            }
        )
        data_feeds[code] = df

    # Rebalance Schedule
    rebalance_schedule = generate_rebalance_schedule(
        ohlcv["close"].index, freq=int(cfg["backtest"]["freq"])
    )

    # Prepare VEC inputs
    factor_names = sorted(std_factors.keys())
    factor_to_idx = {f: i for i, f in enumerate(factor_names)}
    all_factors_stack = np.stack([std_factors[f].values for f in factor_names], axis=-1)

    close_prices = ohlcv["close"].ffill().bfill().values
    open_prices = ohlcv["open"].ffill().bfill().values
    high_prices = ohlcv["high"].ffill().bfill().values
    low_prices = ohlcv["low"].ffill().bfill().values

    # ATR for VEC
    stop_method = cfg["backtest"]["risk_control"].get("stop_method", "fixed")
    atr_arr = None
    if stop_method == "atr":
        atr_window = cfg["backtest"]["risk_control"].get("atr_window", 14)
        atr_arr = calculate_atr(
            high_prices, low_prices, close_prices, window=atr_window
        )

    # Results container
    results = []

    print(f"Starting validation loop for {len(combos)} strategies...")

    for combo in tqdm(combos):
        factor_list = [p.strip() for p in combo.split("+")]

        # --- 1. VEC Backtest (Full Period) ---
        combo_indices = [factor_to_idx[f] for f in factor_list if f in factor_to_idx]
        if len(combo_indices) != len(factor_list):
            print(f"Warning: Missing factors for {combo}")
            continue

        vec_metrics, vec_curve, _, _, _, _, _ = run_vec_backtest(
            factors_3d=all_factors_stack,
            close_prices=close_prices,
            open_prices=open_prices,
            high_prices=high_prices,
            low_prices=low_prices,
            timing_arr=timing_arr,
            factor_indices=combo_indices,
            freq=int(cfg["backtest"]["freq"]),
            pos_size=int(cfg["backtest"]["pos_size"]),
            initial_capital=float(cfg["backtest"]["initial_capital"]),
            commission_rate=float(cfg["backtest"]["commission_rate"]),
            lookback=int(cfg["backtest"]["lookback"]),
            vol_regime_arr=vol_regime_series.values,
            use_atr_stop=(stop_method == "atr"),
            atr_arr=atr_arr,
            # Other params default
        )

        # --- 2. BT Backtest (Full Period) ---
        # Construct combined score df
        score_df = pd.DataFrame(
            0.0, index=ohlcv["close"].index, columns=ohlcv["close"].columns
        )
        for f in factor_list:
            score_df += std_factors[f]

        bt_ret, bt_max_dd, bt_sharpe, bt_daily_rets = run_bt_backtest(
            combined_score_df=score_df,
            timing_series=timing_series_shifted,
            vol_regime_series=vol_regime_series,
            etf_codes=ohlcv["close"].columns.tolist(),
            data_feeds=data_feeds,
            rebalance_schedule=rebalance_schedule,
            freq=int(cfg["backtest"]["freq"]),
            pos_size=int(cfg["backtest"]["pos_size"]),
            initial_capital=float(cfg["backtest"]["initial_capital"]),
            commission_rate=float(cfg["backtest"]["commission_rate"]),
            collect_daily_returns=True,
        )

        # --- 3. Compute Segment Metrics ---
        # Use BT daily returns for authoritative metrics
        if bt_daily_rets is None:
            print(f"Error: No daily returns for {combo}")
            continue

        # Train
        train_rets = bt_daily_rets.loc[:TRAIN_END_DATE]
        m_train = compute_metrics_from_daily(train_rets)

        # Holdout Original
        ho_orig_rets = bt_daily_rets.loc["2025-05-01":HOLDOUT_ORIG_END_DATE]
        m_ho_orig = compute_metrics_from_daily(ho_orig_rets)

        # Holdout Extended
        ho_ext_rets = bt_daily_rets.loc["2025-05-01":EXTENDED_END_DATE]
        m_ho_ext = compute_metrics_from_daily(ho_ext_rets)

        # Recent 120d (approx 6 months)
        # Calculate start date for recent 120 trading days
        if len(bt_daily_rets) > 120:
            recent_rets = bt_daily_rets.iloc[-120:]
        else:
            recent_rets = bt_daily_rets
        m_recent = compute_metrics_from_daily(recent_rets)

        # VEC vs BT Alignment Check (Full Period)
        vec_total_ret = vec_metrics["ret"]
        bt_total_ret = (1 + bt_daily_rets).prod() - 1
        diff = abs(vec_total_ret - bt_total_ret)

        results.append(
            {
                "combo": combo,
                # Train
                "train_ret": m_train["ret"],
                "train_sharpe": m_train["sharpe"],
                "train_max_dd": m_train["max_dd"],
                # Holdout Original
                "ho_orig_ret": m_ho_orig["ret"],
                "ho_orig_sharpe": m_ho_orig["sharpe"],
                "ho_orig_max_dd": m_ho_orig["max_dd"],
                # Holdout Extended
                "ho_ext_ret": m_ho_ext["ret"],
                "ho_ext_sharpe": m_ho_ext["sharpe"],
                "ho_ext_max_dd": m_ho_ext["max_dd"],
                "ho_ext_win_rate": m_ho_ext["win_rate"],
                "ho_ext_pf": m_ho_ext["profit_factor"],
                # Recent
                "recent_ret": m_recent["ret"],
                "recent_sharpe": m_recent["sharpe"],
                "recent_max_dd": m_recent["max_dd"],
                # Alignment
                "vec_bt_diff": diff,
            }
        )

    # Save Results
    res_df = pd.DataFrame(results)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / f"extended_validation_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    res_path = out_dir / "extended_candidates.parquet"
    res_df.to_parquet(res_path)
    res_df.to_csv(out_dir / "extended_candidates.csv", index=False)

    # Generate Report
    report_lines = []
    report_lines.append(f"# Extended Validation Report (v3.2 Sealed)")
    report_lines.append(f"- Date: {datetime.now()}")
    report_lines.append(f"- Data End Date: {EXTENDED_END_DATE}")
    report_lines.append(f"- Strategies Evaluated: {len(res_df)}")
    report_lines.append("")

    report_lines.append("## Top 20 by Extended Holdout Return")
    top20 = res_df.sort_values("ho_ext_ret", ascending=False).head(20)
    report_lines.append(
        top20[
            [
                "combo",
                "ho_ext_ret",
                "ho_ext_sharpe",
                "ho_ext_max_dd",
                "recent_ret",
                "vec_bt_diff",
            ]
        ].to_markdown(index=False, floatfmt=".4f")
    )

    report_lines.append("")
    report_lines.append("## Top 20 by Recent 120d Sharpe")
    top20_recent = res_df.sort_values("recent_sharpe", ascending=False).head(20)
    report_lines.append(
        top20_recent[
            [
                "combo",
                "recent_sharpe",
                "recent_ret",
                "recent_max_dd",
                "ho_ext_ret",
                "vec_bt_diff",
            ]
        ].to_markdown(index=False, floatfmt=".4f")
    )

    (out_dir / "extended_report.md").write_text("\n".join(report_lines))

    print(f"\nDone. Results saved to {out_dir}")


if __name__ == "__main__":
    main()
