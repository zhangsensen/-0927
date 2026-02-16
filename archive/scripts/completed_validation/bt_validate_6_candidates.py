#!/usr/bin/env python3
"""
BT ground truth validation for 6 non-OHLCV factor candidates.
Loads standard OHLCV factors via FactorCache + non-OHLCV factors from parquet files,
then runs full BT with production params (F5, Exp4 hysteresis, SPLIT_MARKET cost model).

Output: holdout-specific metrics (return, MDD, Sharpe, worst month, trades, margin failures).
"""
import gc
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import yaml
import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.factor_cache import FactorCache
from etf_strategy.core.cost_model import load_cost_model, CostModel
from etf_strategy.core.frozen_params import load_frozen_config, FrozenETFPool
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import (
    shift_timing_signal,
    generate_rebalance_schedule,
)
from etf_strategy.regime_gate import compute_regime_gate_arr, gate_stats
from etf_strategy.auditor.core.engine import GenericStrategy, PandasData
from aligned_metrics import compute_aligned_metrics


# --- 5 priority combos (updated per team lead) ---
TARGET_COMBOS = [
    "AMIHUD_ILLIQUIDITY + CALMAR_RATIO_60D + CORRELATION_TO_MARKET_20D",  # C2 baseline
    "ADX_14D + AMIHUD_ILLIQUIDITY + CALMAR_RATIO_60D + SHARE_ACCEL",     # best new-factor
    "ADX_14D + AMIHUD_ILLIQUIDITY + CALMAR_RATIO_60D",                   # minimal OHLCV
    "ADX_14D + CALMAR_RATIO_60D + SHARE_ACCEL",                          # minimal new-factor
    "ADX_14D + OBV_SLOPE_10D + SHARPE_RATIO_20D + SLOPE_20D",            # S1 broken baseline
]

NON_OHLCV_DIR = ROOT / "results" / "non_ohlcv_factors"


def worst_month(equity: np.ndarray, window: int = 21) -> float:
    """Compute worst monthly return from equity curve."""
    if len(equity) < window + 1:
        return 0.0
    monthly_rets = []
    for i in range(window, len(equity), window):
        start = max(0, i - window)
        r = equity[i] / equity[start] - 1.0
        monthly_rets.append(r)
    return min(monthly_rets) if monthly_rets else 0.0


def compute_mdd(equity: np.ndarray) -> float:
    """Compute maximum drawdown from equity curve."""
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(abs(dd.min()))


def run_single_bt(
    combo_str,
    std_factors,
    timing_series,
    vol_regime_series,
    etf_codes,
    data_feeds,
    rebalance_schedule,
    freq,
    pos_size,
    initial_capital,
    commission_rate,
    target_vol,
    vol_window,
    dynamic_leverage_enabled,
    use_t1_open,
    cost_model,
    qdii_codes,
    delta_rank,
    min_hold_days,
    training_end_ts,
):
    """Run BT for a single combo, return detailed metrics."""
    factors = [f.strip() for f in combo_str.split(" + ")]
    dates = timing_series.index

    # Build combined score
    combined_score_df = pd.DataFrame(0.0, index=dates, columns=etf_codes)
    for f in factors:
        if f not in std_factors:
            return {"combo": combo_str, "error": f"Missing factor: {f}"}
        combined_score_df = combined_score_df.add(std_factors[f], fill_value=0)

    # BT engine
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

    if cost_model is not None and cost_model.is_split_market:
        sizing_comm = max(cost_model.active_tier.a_share, cost_model.active_tier.qdii)
    else:
        sizing_comm = commission_rate

    cerebro.addstrategy(
        GenericStrategy,
        scores=combined_score_df,
        timing=timing_series,
        vol_regime=vol_regime_series,
        etf_codes=etf_codes,
        freq=freq,
        pos_size=pos_size,
        rebalance_schedule=rebalance_schedule,
        target_vol=target_vol,
        vol_window=vol_window,
        dynamic_leverage_enabled=dynamic_leverage_enabled,
        use_t1_open=use_t1_open,
        sizing_commission_rate=sizing_comm,
        delta_rank=delta_rank,
        min_hold_days=min_hold_days,
    )

    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(
        bt.analyzers.TimeReturn, _name="timereturn", timeframe=bt.TimeFrame.Days
    )

    start_val = cerebro.broker.getvalue()
    results = cerebro.run()
    end_val = cerebro.broker.getvalue()
    strat = results[0]

    # Extract daily returns
    tr_analysis = strat.analyzers.timereturn.get_analysis()
    daily_returns = pd.Series(tr_analysis).sort_index()
    daily_returns.index = pd.to_datetime(daily_returns.index)

    # Trade analysis
    trade_analysis = strat.analyzers.trades.get_analysis()
    total_trades = trade_analysis.get("total", {}).get("total", 0)
    win_trades = trade_analysis.get("won", {}).get("total", 0)
    win_rate = win_trades / total_trades if total_trades > 0 else 0.0
    won_pnl = trade_analysis.get("won", {}).get("pnl", {}).get("total", 0.0)
    lost_pnl = abs(trade_analysis.get("lost", {}).get("pnl", {}).get("total", 0.0))
    profit_factor = won_pnl / lost_pnl if lost_pnl > 0 else float("inf")

    margin_failures = strat.margin_failures

    # Build equity curve
    eq = (1.0 + daily_returns.fillna(0.0)).cumprod() * initial_capital
    eq = eq.sort_index()

    # Full period
    full_return = (end_val / start_val) - 1.0
    full_mdd = compute_mdd(eq.values)

    # Split train/holdout
    train_end = pd.to_datetime(training_end_ts) if training_end_ts else None
    result = {
        "combo": combo_str,
        "full_return": full_return,
        "full_mdd": full_mdd,
        "total_trades": total_trades,
        "margin_failures": margin_failures,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
    }

    if train_end is not None:
        train_eq = eq.loc[eq.index <= train_end]
        hold_eq = eq.loc[eq.index > train_end]

        if len(train_eq) >= 2:
            result["train_return"] = (train_eq.iloc[-1] / train_eq.iloc[0]) - 1.0
            result["train_mdd"] = compute_mdd(train_eq.values)
        else:
            result["train_return"] = 0.0
            result["train_mdd"] = 0.0

        if len(hold_eq) >= 2:
            # Prepend last train point for continuous equity
            if len(train_eq) >= 1:
                ho_eq = pd.concat([train_eq.iloc[-1:], hold_eq])
            else:
                ho_eq = hold_eq
            ho_vals = ho_eq.values

            result["ho_return"] = (ho_vals[-1] / ho_vals[0]) - 1.0
            result["ho_mdd"] = compute_mdd(ho_vals)
            result["ho_worst_month"] = worst_month(ho_vals)

            # Holdout Sharpe
            ho_daily = hold_eq.pct_change().dropna()
            if len(ho_daily) > 5:
                result["ho_sharpe"] = (
                    ho_daily.mean() / ho_daily.std() * np.sqrt(252)
                    if ho_daily.std() > 1e-10
                    else 0.0
                )
            else:
                result["ho_sharpe"] = 0.0

            # Holdout trades: count trades after training_end
            # We can approximate from total - train proportionally, but BT doesn't split
            # Just report total trades
            result["ho_trades"] = total_trades  # BT doesn't separate easily
        else:
            result["ho_return"] = 0.0
            result["ho_mdd"] = 0.0
            result["ho_worst_month"] = 0.0
            result["ho_sharpe"] = 0.0
            result["ho_trades"] = 0

    return result


def main():
    print("=" * 80)
    print("BT Ground Truth Validation: 5 Priority Candidates (SHARE_ACCEL focus)")
    print("=" * 80)

    # Load config
    config_path = ROOT / "configs" / "combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    frozen = load_frozen_config(config, config_path=str(config_path))
    print(f"Frozen params: version={frozen.version}")

    # Execution model
    from etf_strategy.core.execution_model import load_execution_model
    exec_model = load_execution_model(config)
    USE_T1_OPEN = exec_model.is_t1_open
    print(f"Execution model: {exec_model.mode}")

    training_end_date = config.get("data", {}).get("training_end_date")

    # Load data
    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )

    # Compute standard OHLCV factors via cache
    factor_cache = FactorCache(cache_dir=Path(config["data"].get("cache_dir") or ".cache"))
    cached = factor_cache.get_or_compute(ohlcv=ohlcv, config=config, data_dir=loader.data_dir)
    std_factors = cached["std_factors"]

    factor_names = sorted(std_factors.keys())
    first_factor = std_factors[factor_names[0]]
    dates = first_factor.index
    etf_codes = first_factor.columns.tolist()

    print(f"Standard OHLCV factors: {len(factor_names)}")

    # Load non-OHLCV factors from parquet
    non_ohlcv_loaded = 0
    if NON_OHLCV_DIR.exists():
        for pq_file in sorted(NON_OHLCV_DIR.glob("*.parquet")):
            fname = pq_file.stem
            if fname not in std_factors:
                df = pd.read_parquet(pq_file)
                df.index = pd.to_datetime(df.index)
                # Align to base dates and symbols
                df = df.reindex(dates)
                for col in etf_codes:
                    if col not in df.columns:
                        df[col] = np.nan
                df = df[etf_codes]
                std_factors[fname] = df
                non_ohlcv_loaded += 1
    print(f"Non-OHLCV factors loaded: {non_ohlcv_loaded}")
    print(f"Total factors: {len(std_factors)}")

    # Backtest params from config
    backtest_config = config.get("backtest", {})
    freq = backtest_config.get("freq", 5)
    pos_size = backtest_config.get("pos_size", 2)
    initial_capital = float(backtest_config.get("initial_capital", 1_000_000.0))
    commission_rate = float(backtest_config.get("commission_rate", 0.0002))
    lookback = backtest_config.get("lookback", 252)

    # Hysteresis from config
    hyst_config = backtest_config.get("hysteresis", {})
    delta_rank = float(hyst_config.get("delta_rank", 0.0))
    min_hold_days = int(hyst_config.get("min_hold_days", 0))

    # Cost model
    cost_model = load_cost_model(config)
    qdii_codes = set(FrozenETFPool().qdii_codes)
    tier = cost_model.active_tier

    print(f"\nProduction params:")
    print(f"  FREQ={freq}, POS_SIZE={pos_size}, CAPITAL={initial_capital:.0f}")
    print(f"  COMMISSION={commission_rate}, LOOKBACK={lookback}")
    print(f"  Hysteresis: delta_rank={delta_rank}, min_hold_days={min_hold_days}")
    print(f"  Cost model: mode={cost_model.mode}, A={tier.a_share*10000:.0f}bp, QDII={tier.qdii*10000:.0f}bp")
    print(f"  Execution: T1_OPEN={USE_T1_OPEN}")

    # Timing + Regime gate
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

    gate_arr = compute_regime_gate_arr(
        ohlcv["close"], dates, backtest_config=backtest_config
    )
    timing_series = timing_series * pd.Series(gate_arr, index=dates)
    if bool(backtest_config.get("regime_gate", {}).get("enabled", False)):
        s = gate_stats(gate_arr)
        print(f"  Regime gate: mean={s['mean']:.2f}, min={s['min']:.2f}, max={s['max']:.2f}")

    vol_regime_series = pd.Series(1.0, index=dates)

    # Dynamic leverage
    dl_config = backtest_config.get("risk_control", {}).get("dynamic_leverage", {})
    dynamic_leverage_enabled = dl_config.get("enabled", False)
    target_vol = dl_config.get("target_vol", 0.20)
    vol_window = dl_config.get("vol_window", 20)

    # Data feeds
    data_feeds = {}
    for ticker in etf_codes:
        df = pd.DataFrame({
            "open": ohlcv["open"][ticker],
            "high": ohlcv["high"][ticker],
            "low": ohlcv["low"][ticker],
            "close": ohlcv["close"][ticker],
            "volume": ohlcv["volume"][ticker],
        })
        df = df.reindex(dates)
        df = df.ffill().fillna(0.01)
        data_feeds[ticker] = df

    # Rebalance schedule
    rebalance_schedule = generate_rebalance_schedule(
        total_periods=len(dates),
        lookback_window=lookback,
        freq=freq,
    )

    print(f"\nData: {len(dates)} days x {len(etf_codes)} ETFs")
    print(f"Rebalance days: {len(rebalance_schedule)}")

    # Run BT for each combo
    print(f"\n{'='*80}")
    print(f"Running BT for {len(TARGET_COMBOS)} combos...")
    print(f"{'='*80}\n")

    all_results = []
    for i, combo in enumerate(TARGET_COMBOS):
        t0 = time.time()
        print(f"[{i+1}/{len(TARGET_COMBOS)}] {combo}")
        result = run_single_bt(
            combo_str=combo,
            std_factors=std_factors,
            timing_series=timing_series,
            vol_regime_series=vol_regime_series,
            etf_codes=etf_codes,
            data_feeds=data_feeds,
            rebalance_schedule=rebalance_schedule,
            freq=freq,
            pos_size=pos_size,
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            target_vol=target_vol,
            vol_window=vol_window,
            dynamic_leverage_enabled=dynamic_leverage_enabled,
            use_t1_open=USE_T1_OPEN,
            cost_model=cost_model,
            qdii_codes=qdii_codes,
            delta_rank=delta_rank,
            min_hold_days=min_hold_days,
            training_end_ts=training_end_date,
        )
        elapsed = time.time() - t0
        all_results.append(result)

        if "error" in result:
            print(f"  ERROR: {result['error']} ({elapsed:.1f}s)")
        else:
            print(
                f"  Full: {result['full_return']:+.1%} | "
                f"HO: {result.get('ho_return', 0):+.1%} | "
                f"HO MDD: {result.get('ho_mdd', 0):.1%} | "
                f"HO Sharpe: {result.get('ho_sharpe', 0):+.2f} | "
                f"HO WM: {result.get('ho_worst_month', 0):+.1%} | "
                f"Trades: {result['total_trades']} | "
                f"Margin: {result['margin_failures']} | "
                f"({elapsed:.1f}s)"
            )
        gc.collect()

    # Summary table
    print(f"\n{'='*80}")
    print("BT GROUND TRUTH RESULTS")
    print(f"{'='*80}")

    # Load VEC results for comparison
    vec_path = ROOT / "results" / "vec_full_backtest_20260212_180443" / "vec_holdout_analysis.csv"
    vec_df = None
    if vec_path.exists():
        vec_df = pd.read_csv(vec_path)

    print(f"\n{'Combo':<65} {'HO Ret':>8} {'HO MDD':>8} {'HO Shrp':>8} {'HO WM':>8} {'Trades':>7} {'Margin':>7}")
    print("-" * 112)

    for r in all_results:
        if "error" in r:
            print(f"{r['combo']:<65} ERROR: {r['error']}")
            continue
        print(
            f"{r['combo']:<65} "
            f"{r.get('ho_return', 0):>+7.1%} "
            f"{r.get('ho_mdd', 0):>7.1%} "
            f"{r.get('ho_sharpe', 0):>+7.2f} "
            f"{r.get('ho_worst_month', 0):>+7.1%} "
            f"{r.get('total_trades', 0):>7d} "
            f"{r.get('margin_failures', 0):>7d}"
        )

    # VEC-BT gap analysis
    if vec_df is not None:
        print(f"\n{'='*80}")
        print("VEC-BT GAP ANALYSIS")
        print(f"{'='*80}")
        print(f"\n{'Combo':<55} {'VEC HO':>8} {'BT HO':>8} {'Gap':>8} {'VEC MDD':>8} {'BT MDD':>8} {'MDD Gap':>8}")
        print("-" * 104)

        for r in all_results:
            if "error" in r:
                continue
            combo = r["combo"]
            # Find in VEC
            vec_row = vec_df[vec_df["combo"] == combo]
            if len(vec_row) == 0:
                # Try normalized match
                combo_norm = " + ".join(sorted([f.strip() for f in combo.split("+")]))
                vec_row = vec_df[vec_df["combo"].apply(
                    lambda x: " + ".join(sorted([f.strip() for f in x.split("+")])) == combo_norm
                )]
            if len(vec_row) > 0:
                vec_ho_ret = vec_row.iloc[0].get("ho_return", vec_row.iloc[0].get("ho_ret", 0))
                vec_ho_mdd = vec_row.iloc[0]["ho_mdd"]
                bt_ho_ret = r.get("ho_return", 0)
                bt_ho_mdd = r.get("ho_mdd", 0)
                gap_ret = bt_ho_ret - vec_ho_ret
                gap_mdd = bt_ho_mdd - vec_ho_mdd
                print(
                    f"{combo:<55} "
                    f"{vec_ho_ret:>+7.1%} "
                    f"{bt_ho_ret:>+7.1%} "
                    f"{gap_ret:>+7.1%} "
                    f"{vec_ho_mdd:>7.1%} "
                    f"{bt_ho_mdd:>7.1%} "
                    f"{gap_mdd:>+7.1%}"
                )
            else:
                print(f"{combo:<55} (not found in VEC)")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / "results" / f"bt_6candidates_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(output_dir / "bt_results.csv", index=False)
    df_results.to_parquet(output_dir / "bt_results.parquet", index=False)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
