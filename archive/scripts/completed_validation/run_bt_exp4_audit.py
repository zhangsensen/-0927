#!/usr/bin/env python3
"""BT Audit for F5_ON production candidate (and F20_OFF defensive fallback).

Runs Backtrader event-driven simulation for sealed strategies S1/S2 under:
  - F5_ON:  freq=5, delta_rank=0.10, min_hold_days=9  (production primary)
  - F20_OFF: freq=20, delta_rank=0, min_hold_days=0    (defensive fallback)

Compares BT results against VEC reference from freq_x_exp4_scan.
Acceptance criteria:
  1. Trade count within 20% of VEC
  2. Return direction consistency (no sign flip on holdout)
  3. VEC-BT gap explainable (systemic ~4-8pp from float vs int shares)

Usage:
    uv run python scripts/run_bt_exp4_audit.py
"""

from __future__ import annotations

import gc
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import yaml
import backtrader as bt

from etf_strategy.core.cost_model import load_cost_model
from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.factor_cache import FactorCache
from etf_strategy.core.frozen_params import FrozenETFPool
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import (
    shift_timing_signal,
    generate_rebalance_schedule,
)
from etf_strategy.regime_gate import compute_regime_gate_arr
from etf_strategy.auditor.core.engine import GenericStrategy, PandasData
from aligned_metrics import compute_aligned_metrics

# ── Audit configurations ──────────────────────────────────────────────────
CONFIGS = {
    "F5_ON": {"freq": 5, "delta_rank": 0.10, "min_hold_days": 9},
    "F20_OFF": {"freq": 20, "delta_rank": 0.0, "min_hold_days": 0},
}

SEALED = {
    "S1": "ADX_14D + OBV_SLOPE_10D + SHARPE_RATIO_20D + SLOPE_20D",
    "S2": "ADX_14D + OBV_SLOPE_10D + PRICE_POSITION_120D + SHARPE_RATIO_20D + SLOPE_20D",
}

# VEC reference values from freq_x_exp4_scan (for comparison)
VEC_REF = {
    ("S1", "F5_ON"):  {"full_return": 0.4253, "holdout_return": 0.3020, "trades": 71},
    ("S2", "F5_ON"):  {"full_return": 0.2523, "holdout_return": 0.2376, "trades": 70},
    ("S1", "F20_OFF"): {"full_return": 0.4280, "holdout_return": 0.1951, "trades": 85},
    ("S2", "F20_OFF"): {"full_return": 0.5464, "holdout_return": 0.1951, "trades": 86},
}


def run_single_bt(
    combined_score_df,
    timing_series,
    vol_regime_series,
    etf_codes,
    data_feeds,
    rebalance_schedule,
    freq,
    pos_size,
    initial_capital,
    commission_rate,
    use_t1_open,
    cost_model,
    qdii_codes,
    delta_rank,
    min_hold_days,
):
    """Run a single BT and return metrics + trade list."""
    cerebro = bt.Cerebro(cheat_on_open=use_t1_open)
    cerebro.broker.setcash(initial_capital)

    if cost_model is not None and cost_model.is_split_market and qdii_codes is not None:
        for ticker in data_feeds:
            rate = cost_model.get_cost(ticker, qdii_codes)
            cerebro.broker.setcommission(commission=rate, name=ticker, leverage=1.0)
    else:
        cerebro.broker.setcommission(commission=commission_rate, leverage=1.0)

    if use_t1_open:
        cerebro.broker.set_coc(False)
        cerebro.broker.set_coo(True)
    else:
        cerebro.broker.set_coc(True)
    cerebro.broker.set_checksubmit(False)

    for ticker, df in data_feeds.items():
        data = PandasData(dataname=df, name=ticker)
        cerebro.adddata(data)

    # Compute max commission rate for conservative position sizing
    max_comm = commission_rate
    if cost_model is not None and cost_model.is_split_market:
        tier = cost_model.active_tier
        max_comm = max(tier.a_share, tier.qdii)

    cerebro.addstrategy(
        GenericStrategy,
        scores=combined_score_df,
        timing=timing_series,
        vol_regime=vol_regime_series,
        etf_codes=etf_codes,
        freq=freq,
        pos_size=pos_size,
        rebalance_schedule=rebalance_schedule,
        target_vol=0.20,
        vol_window=20,
        dynamic_leverage_enabled=True,
        use_t1_open=use_t1_open,
        delta_rank=delta_rank,
        min_hold_days=min_hold_days,
        sizing_commission_rate=max_comm,
    )

    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio, _name="sharpe",
        timeframe=bt.TimeFrame.Days, compression=1,
        riskfreerate=0.0, annualize=True,
    )
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(
        bt.analyzers.TimeReturn, _name="timereturn",
        timeframe=bt.TimeFrame.Days,
    )

    start_val = cerebro.broker.getvalue()
    results = cerebro.run()
    end_val = cerebro.broker.getvalue()
    strat = results[0]

    bt_return = (end_val / start_val) - 1

    # Trade details
    trade_analysis = strat.analyzers.trades.get_analysis()
    total_trades = trade_analysis.get("total", {}).get("total", 0)
    win_trades = trade_analysis.get("won", {}).get("total", 0)
    win_rate = win_trades / total_trades if total_trades > 0 else 0.0
    avg_len = trade_analysis.get("len", {}).get("average", 0.0)
    won_pnl = trade_analysis.get("won", {}).get("pnl", {}).get("total", 0.0)
    lost_pnl = abs(trade_analysis.get("lost", {}).get("pnl", {}).get("total", 0.0))
    profit_factor = won_pnl / lost_pnl if lost_pnl > 0 else float("inf")

    # Drawdown
    dd_analysis = strat.analyzers.drawdown.get_analysis()
    max_drawdown = dd_analysis.get("max", {}).get("drawdown", 0.0) / 100.0

    # Sharpe
    sharpe_analysis = strat.analyzers.sharpe.get_analysis()
    sharpe_ratio = sharpe_analysis.get("sharperatio", 0.0) or 0.0

    # Daily returns for train/holdout split
    tr_analysis = strat.analyzers.timereturn.get_analysis()
    daily_returns = pd.Series(tr_analysis).sort_index()
    daily_returns.index = pd.to_datetime(daily_returns.index)

    # Order log for trade sequence audit
    order_log = pd.DataFrame(strat.orders)

    return {
        "bt_return": bt_return,
        "bt_trades": total_trades,
        "bt_win_rate": win_rate,
        "bt_avg_hold": avg_len,
        "bt_profit_factor": profit_factor,
        "bt_mdd": max_drawdown,
        "bt_sharpe": sharpe_ratio,
        "daily_returns": daily_returns,
        "order_log": order_log,
    }


def main():
    print("=" * 80)
    print("BT Exp4 Audit: F5_ON (production) + F20_OFF (defensive)")
    print("=" * 80)

    # ── Load config & data ─────────────────────────────────────────────────
    config_path = ROOT / "configs" / "combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Override frozen params for experimentation
    import os
    os.environ["FROZEN_PARAMS_MODE"] = "warn"

    from etf_strategy.core.execution_model import load_execution_model
    exec_model = load_execution_model(config)
    USE_T1_OPEN = exec_model.is_t1_open

    training_end_date = config["data"].get("training_end_date", "2025-04-30")
    training_end_ts = pd.to_datetime(training_end_date)

    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )

    # Factors (cached)
    factor_cache = FactorCache(cache_dir=Path(config["data"].get("cache_dir") or ".cache"))
    cached = factor_cache.get_or_compute(ohlcv=ohlcv, config=config, data_dir=loader.data_dir)
    std_factors = cached["std_factors"]

    factor_names = sorted(std_factors.keys())
    first_factor = std_factors[factor_names[0]]
    dates = first_factor.index
    etf_codes = first_factor.columns.tolist()

    backtest_config = config.get("backtest", {})
    pos_size = backtest_config.get("pos_size", 2)
    initial_capital = float(backtest_config.get("initial_capital", 1_000_000.0))
    commission_rate = float(backtest_config.get("commission_rate", 0.0002))
    lookback = backtest_config.get("lookback", 252)

    cost_model = load_cost_model(config)
    qdii_codes = set(FrozenETFPool().qdii_codes)

    # Timing + regime gate
    timing_config = backtest_config.get("timing", {})
    timing_module = LightTimingModule(
        extreme_threshold=timing_config.get("extreme_threshold", -0.4),
        extreme_position=timing_config.get("extreme_position", 0.3),
    )
    timing_series_raw = timing_module.compute_position_ratios(ohlcv["close"])
    timing_arr_shifted = shift_timing_signal(
        timing_series_raw.reindex(dates).fillna(1.0).values
    )
    timing_series = pd.Series(timing_arr_shifted, index=dates)

    gate_arr = compute_regime_gate_arr(ohlcv["close"], dates, backtest_config=backtest_config)
    timing_series = timing_series * pd.Series(gate_arr, index=dates)

    # Vol regime (same as batch_bt_backtest.py)
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
    vol_regime_series = exposure_s.reindex(dates).fillna(1.0)

    # Data feeds
    data_feeds = {}
    for ticker in etf_codes:
        df = pd.DataFrame({
            "open": ohlcv["open"][ticker],
            "high": ohlcv["high"][ticker],
            "low": ohlcv["low"][ticker],
            "close": ohlcv["close"][ticker],
            "volume": ohlcv["volume"][ticker],
        }).reindex(dates).ffill().fillna(0.01)
        data_feeds[ticker] = df

    T = len(dates)
    print(f"Data: {T} days x {len(etf_codes)} ETFs, training_end={training_end_date}")
    print(f"Execution: {exec_model.mode}, Cost: {cost_model.tier}")
    print()

    # ── Run BT for each config x strategy ──────────────────────────────────
    all_results = []

    for cfg_name, cfg_params in CONFIGS.items():
        freq = cfg_params["freq"]
        dr = cfg_params["delta_rank"]
        mh = cfg_params["min_hold_days"]

        rebalance_schedule = generate_rebalance_schedule(
            total_periods=T, lookback_window=lookback, freq=freq,
        )

        for strat_name, combo_str in SEALED.items():
            label = f"{strat_name}_{cfg_name}"
            print(f"--- {label}: freq={freq}, dr={dr}, mh={mh} ---")

            # Build combined score
            factors = [f.strip() for f in combo_str.split(" + ")]
            combined_score_df = pd.DataFrame(0.0, index=dates, columns=etf_codes)
            for f_name in factors:
                combined_score_df = combined_score_df.add(std_factors[f_name], fill_value=0)

            result = run_single_bt(
                combined_score_df=combined_score_df,
                timing_series=timing_series,
                vol_regime_series=vol_regime_series,
                etf_codes=etf_codes,
                data_feeds=data_feeds,
                rebalance_schedule=rebalance_schedule,
                freq=freq,
                pos_size=pos_size,
                initial_capital=initial_capital,
                commission_rate=commission_rate,
                use_t1_open=USE_T1_OPEN,
                cost_model=cost_model,
                qdii_codes=qdii_codes,
                delta_rank=dr,
                min_hold_days=mh,
            )

            # Compute train/holdout split
            dr_s = result["daily_returns"]
            eq = (1.0 + dr_s.fillna(0.0)).cumprod() * initial_capital

            train_eq = eq.loc[eq.index <= training_end_ts]
            hold_eq = eq.loc[eq.index > training_end_ts]

            train_return = (train_eq.iloc[-1] / train_eq.iloc[0] - 1) if len(train_eq) >= 2 else np.nan
            holdout_return = (hold_eq.iloc[-1] / hold_eq.iloc[0] - 1) if len(hold_eq) >= 2 else np.nan

            # Get VEC reference
            vec_ref = VEC_REF.get((strat_name, cfg_name), {})
            vec_full = vec_ref.get("full_return", np.nan)
            vec_holdout = vec_ref.get("holdout_return", np.nan)
            vec_trades = vec_ref.get("trades", np.nan)

            # Compute gaps
            full_gap = result["bt_return"] - vec_full if not np.isnan(vec_full) else np.nan
            holdout_gap = holdout_return - vec_holdout if not np.isnan(vec_holdout) else np.nan
            trade_gap_pct = (
                (result["bt_trades"] - vec_trades) / vec_trades * 100
                if vec_trades and vec_trades > 0 else np.nan
            )

            # Direction consistency
            if not np.isnan(vec_holdout) and not np.isnan(holdout_return):
                direction_ok = (vec_holdout > 0) == (holdout_return > 0)
            else:
                direction_ok = None

            row = {
                "label": label,
                "strategy": strat_name,
                "config": cfg_name,
                "freq": freq,
                "delta_rank": dr,
                "min_hold_days": mh,
                "bt_full_return": result["bt_return"],
                "bt_train_return": train_return,
                "bt_holdout_return": holdout_return,
                "bt_trades": result["bt_trades"],
                "bt_win_rate": result["bt_win_rate"],
                "bt_avg_hold": result["bt_avg_hold"],
                "bt_profit_factor": result["bt_profit_factor"],
                "bt_mdd": result["bt_mdd"],
                "bt_sharpe": result["bt_sharpe"],
                "vec_full_return": vec_full,
                "vec_holdout_return": vec_holdout,
                "vec_trades": vec_trades,
                "full_gap_pp": full_gap * 100 if not np.isnan(full_gap) else np.nan,
                "holdout_gap_pp": holdout_gap * 100 if not np.isnan(holdout_gap) else np.nan,
                "trade_gap_pct": trade_gap_pct,
                "direction_ok": direction_ok,
            }
            all_results.append(row)

            print(f"  BT full={result['bt_return']:.4f}  VEC full={vec_full:.4f}  gap={full_gap*100:+.1f}pp")
            print(f"  BT holdout={holdout_return:.4f}  VEC holdout={vec_holdout:.4f}  gap={holdout_gap*100:+.1f}pp")
            print(f"  BT trades={result['bt_trades']}  VEC trades={vec_trades}  gap={trade_gap_pct:+.1f}%")
            print(f"  Direction OK: {direction_ok}")
            print(f"  Win rate={result['bt_win_rate']:.1%}, Avg hold={result['bt_avg_hold']:.1f}d, PF={result['bt_profit_factor']:.2f}")
            print()

            # Save order log
            if len(result["order_log"]) > 0:
                order_dir = ROOT / "results" / "bt_exp4_audit"
                order_dir.mkdir(parents=True, exist_ok=True)
                result["order_log"].to_csv(
                    order_dir / f"orders_{label}.csv", index=False
                )

            gc.collect()

    # ── Summary ────────────────────────────────────────────────────────────
    df_results = pd.DataFrame(all_results)
    output_dir = ROOT / "results" / "bt_exp4_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_dir / "audit_summary.csv", index=False)

    print("=" * 80)
    print("AUDIT SUMMARY")
    print("=" * 80)

    for _, row in df_results.iterrows():
        label = row["label"]
        checks = []

        # Check 1: Trade count within 20%
        tg = abs(row["trade_gap_pct"]) if not np.isnan(row["trade_gap_pct"]) else 999
        trade_ok = tg <= 20
        checks.append(f"Trades {'PASS' if trade_ok else 'FAIL'} ({row['trade_gap_pct']:+.1f}%)")

        # Check 2: Direction consistency
        dir_ok = row["direction_ok"]
        checks.append(f"Direction {'PASS' if dir_ok else 'FAIL'}")

        # Check 3: Return gap explainable (< 15pp for full, < 10pp for holdout)
        fg = abs(row["full_gap_pp"]) if not np.isnan(row["full_gap_pp"]) else 999
        hg = abs(row["holdout_gap_pp"]) if not np.isnan(row["holdout_gap_pp"]) else 999
        gap_ok = fg < 15 and hg < 10
        checks.append(f"Gap {'PASS' if gap_ok else 'WARN'} (full={row['full_gap_pp']:+.1f}pp, ho={row['holdout_gap_pp']:+.1f}pp)")

        all_pass = trade_ok and dir_ok and gap_ok
        status = "PASS" if all_pass else "REVIEW"

        print(f"\n{label}: [{status}]")
        for c in checks:
            print(f"  {c}")
        print(f"  BT: full={row['bt_full_return']:.4f}, holdout={row['bt_holdout_return']:.4f}, trades={int(row['bt_trades'])}")
        print(f"  VEC: full={row['vec_full_return']:.4f}, holdout={row['vec_holdout_return']:.4f}, trades={int(row['vec_trades'])}")

    print(f"\nResults saved to: {output_dir}/audit_summary.csv")
    print(f"Order logs saved to: {output_dir}/orders_*.csv")


if __name__ == "__main__":
    main()
