#!/usr/bin/env python3
"""
Detailed BT Analysis for Top Strategies.
Generates detailed reports and equity curves for the top N strategies by BT return.
"""
import sys
import argparse
from pathlib import Path
import json

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "etf_rotation_optimized"))
sys.path.insert(0, str(ROOT))

import yaml
import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime
import matplotlib.pyplot as plt

from core.data_loader import DataLoader
from core.precise_factor_library_v2 import PreciseFactorLibrary
from core.cross_section_processor import CrossSectionProcessor
from core.market_timing import LightTimingModule
from core.utils.rebalance import shift_timing_signal, generate_rebalance_schedule
from strategy_auditor.core.engine import GenericStrategy, PandasData

FREQ = 8
POS_SIZE = 3
INITIAL_CAPITAL = 1_000_000.0
COMMISSION_RATE = 0.0002
LOOKBACK = 252

def run_detailed_backtest(combo_name, combined_score_df, timing_series, etf_codes, data_feeds, rebalance_schedule, output_dir):
    """Run a single detailed backtest"""
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(INITIAL_CAPITAL)
    cerebro.broker.setcommission(commission=COMMISSION_RATE, leverage=1.0)
    cerebro.broker.set_coc(True)
    cerebro.broker.set_checksubmit(False)

    for ticker, df in data_feeds.items():
        data = PandasData(dataname=df, name=ticker)
        cerebro.adddata(data)

    cerebro.addstrategy(
        GenericStrategy, 
        scores=combined_score_df, 
        timing=timing_series, 
        etf_codes=etf_codes, 
        freq=FREQ, 
        pos_size=POS_SIZE,
        rebalance_schedule=rebalance_schedule
    )

    # Add Analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, compression=1, riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='annual')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')

    print(f"   Running backtest for: {combo_name[:50]}...")
    results = cerebro.run()
    strat = results[0]

    # Extract Metrics
    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0.0)
    if sharpe is None: sharpe = 0.0
    
    dd = strat.analyzers.drawdown.get_analysis()
    # Backtrader returns drawdown as percentage (e.g. 20.0 for 20%)
    max_dd = dd.get('max', {}).get('drawdown', 0.0)
    max_dd_len = dd.get('max', {}).get('len', 0)
    
    annual = strat.analyzers.annual.get_analysis()
    
    trades = strat.analyzers.trades.get_analysis()
    total_trades = trades.get('total', {}).get('total', 0)
    win_trades = trades.get('won', {}).get('total', 0)
    win_rate = win_trades / total_trades if total_trades > 0 else 0
    
    # Extract Equity Curve
    tret = strat.analyzers.timereturn.get_analysis()
    dates = list(tret.keys())
    returns = list(tret.values())
    
    equity_df = pd.DataFrame({'date': dates, 'daily_return': returns})
    equity_df['date'] = pd.to_datetime(equity_df['date'])
    equity_df = equity_df.sort_values('date')
    equity_df['equity'] = INITIAL_CAPITAL * (1 + equity_df['daily_return']).cumprod()
    equity_df['drawdown'] = 1 - equity_df['equity'] / equity_df['equity'].cummax()
    
    # Calculate metrics manually for verification
    manual_max_dd = equity_df['drawdown'].max()
    daily_std = equity_df['daily_return'].std()
    daily_mean = equity_df['daily_return'].mean()
    annual_sharpe = (daily_mean / daily_std) * np.sqrt(252) if daily_std > 0 else 0.0
    
    # Save Equity Curve
    safe_name = "".join([c if c.isalnum() else "_" for c in combo_name])[:50]
    equity_path = output_dir / f"equity_{safe_name}.csv"
    equity_df.to_csv(equity_path, index=False)
    
    # Generate Plot
    plt.figure(figsize=(12, 6))
    plt.plot(equity_df['date'], equity_df['equity'], label='Equity')
    plt.title(f"Equity Curve: {combo_name[:50]}...")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_dir / f"plot_{safe_name}.png")
    plt.close()
    
    return {
        "combo": combo_name,
        "total_return": (equity_df['equity'].iloc[-1] / INITIAL_CAPITAL) - 1,
        "sharpe": annual_sharpe, # Use manual annualized sharpe
        "max_drawdown": manual_max_dd, # Use manual max dd (decimal)
        "max_drawdown_len": max_dd_len,
        "win_rate": win_rate,
        "total_trades": total_trades,
        "annual_returns": annual,
        "equity_path": str(equity_path)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to BT results CSV")
    parser.add_argument("--topk", type=int, default=10, help="Number of top strategies to analyze")
    args = parser.parse_args()

    # 1. Load Top Strategies
    df = pd.read_csv(args.input)
    if 'bt_return' not in df.columns:
        print("❌ 'bt_return' column missing.")
        return
        
    top_strategies = df.nlargest(args.topk, 'bt_return')
    print(f"✅ Selected Top {len(top_strategies)} strategies by BT Return")

    # 2. Load Data (Once)
    config_path = ROOT / "configs/combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )

    # 3. Compute Factors
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)
    factor_names_list = raw_factors_df.columns.get_level_values(0).unique().tolist()
    raw_factors = {fname: raw_factors_df[fname] for fname in factor_names_list}

    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)
    
    factor_names = sorted(std_factors.keys())
    first_factor = std_factors[factor_names[0]]
    dates = first_factor.index
    etf_codes = first_factor.columns.tolist()

    timing_module = LightTimingModule()
    timing_series_raw = timing_module.compute_position_ratios(ohlcv["close"])
    timing_arr_shifted = shift_timing_signal(timing_series_raw.reindex(dates).fillna(1.0).values)
    timing_series = pd.Series(timing_arr_shifted, index=dates)
    
    rebalance_schedule = generate_rebalance_schedule(
        total_periods=len(dates),
        lookback_window=LOOKBACK,
        freq=FREQ,
    )

    # Prepare data feeds
    data_feeds = {}
    for ticker in etf_codes:
        df_feed = pd.DataFrame({
            "open": ohlcv["open"][ticker],
            "high": ohlcv["high"][ticker],
            "low": ohlcv["low"][ticker],
            "close": ohlcv["close"][ticker],
            "volume": ohlcv["volume"][ticker],
        }).reindex(dates).ffill().fillna(0.01)
        data_feeds[ticker] = df_feed

    # 4. Run Analysis
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / "results" / f"detailed_bt_analysis_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    detailed_results = []
    
    print("=" * 80)
    print("🚀 Running Detailed Analysis")
    print("=" * 80)
    
    for idx, row in top_strategies.iterrows():
        combo = row['combo']
        factors = [f.strip() for f in combo.split(" + ")]
        
        # Construct Score
        combined_score_df = pd.DataFrame(0.0, index=dates, columns=etf_codes)
        for f in factors:
            combined_score_df = combined_score_df.add(std_factors[f], fill_value=0)
            
        res = run_detailed_backtest(
            combo, combined_score_df, timing_series, etf_codes, 
            data_feeds, rebalance_schedule, output_dir
        )
        detailed_results.append(res)

    # 5. Save & Report
    results_df = pd.DataFrame(detailed_results)
    results_df.to_csv(output_dir / "detailed_metrics.csv", index=False)
    
    print("\n" + "=" * 100)
    print("📊 Detailed BT Analysis Report (Top 10 by Return)")
    print("=" * 100)
    print(f"{'Rank':<4} | {'Return':<8} | {'Sharpe':<6} | {'MaxDD':<7} | {'WinRate':<7} | {'Trades':<6} | Combo")
    print("-" * 100)
    
    for i, row in enumerate(results_df.itertuples(), 1):
        combo_short = row.combo[:40] + "..." if len(row.combo) > 40 else row.combo
        print(f"{i:<4} | {row.total_return*100:7.1f}% | {row.sharpe:6.2f} | {row.max_drawdown*100:6.1f}% | {row.win_rate*100:6.1f}% | {row.total_trades:<6} | {combo_short}")
        
    print("\n✅ Analysis Complete. Equity curves saved to:", output_dir)

if __name__ == "__main__":
    main()
