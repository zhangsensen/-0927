#!/usr/bin/env python3
"""
Run BT backtest on top scored strategies from VEC results.
"""
import gc
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "etf_rotation_optimized"))
sys.path.insert(0, str(ROOT))

import yaml
import pandas as pd
import numpy as np
import backtrader as bt
from tqdm import tqdm
from datetime import datetime
import multiprocessing as mp

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

def run_bt_backtest(combined_score_df, timing_series, etf_codes, data_feeds, rebalance_schedule):
    """单组合 BT 回测引擎"""
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

    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    start_val = cerebro.broker.getvalue()
    results = cerebro.run()
    end_val = cerebro.broker.getvalue()
    strat = results[0]

    bt_return = (end_val / start_val) - 1
    
    # Extract Trade Metrics
    trade_analysis = strat.analyzers.trades.get_analysis()
    total_trades = trade_analysis.total.closed if 'total' in trade_analysis else 0
    won_trades = trade_analysis.won.total if 'won' in trade_analysis else 0
    win_rate = won_trades / total_trades if total_trades > 0 else 0.0

    return bt_return, strat.margin_failures, total_trades, win_rate

# 全局变量，用于子进程共享数据 (Copy-on-Write)
_shared_data = {}

def init_worker(data_feeds, std_factors, timing_series, etf_codes):
    """子进程初始化：保存共享数据"""
    global _shared_data
    _shared_data['data_feeds'] = data_feeds
    _shared_data['std_factors'] = std_factors
    _shared_data['timing_series'] = timing_series
    _shared_data['etf_codes'] = etf_codes
    
    # ✅ 预计算调仓日程 (所有组合共享)
    T = len(timing_series)
    _shared_data['rebalance_schedule'] = generate_rebalance_schedule(
        total_periods=T,
        lookback_window=LOOKBACK,
        freq=FREQ,
    )

def process_combo(row_data):
    """单个组合的处理函数"""
    # 禁用 GC 以提升性能（子进程生命周期短，无需 GC）
    gc.disable()
    
    combo_str = row_data['combo']
    
    # 从全局变量获取数据
    data_feeds = _shared_data['data_feeds']
    std_factors = _shared_data['std_factors']
    timing_series = _shared_data['timing_series']
    etf_codes = _shared_data['etf_codes']
    rebalance_schedule = _shared_data['rebalance_schedule']
    
    factors = [f.strip() for f in combo_str.split(" + ")]
    dates = timing_series.index

    # 构造得分矩阵 (使用 DataFrame.add 保持 NaN 处理一致性)
    # ✅ 与 full_vec_bt_comparison.py 保持一致：fill_value=0 避免 NaN 传播
    combined_score_df = pd.DataFrame(0.0, index=dates, columns=etf_codes)
    for f in factors:
        combined_score_df = combined_score_df.add(std_factors[f], fill_value=0)

    # 运行回测
    bt_return, margin_failures, total_trades, win_rate = run_bt_backtest(
        combined_score_df, 
        timing_series, 
        etf_codes, 
        data_feeds,
        rebalance_schedule
    )
    
    return {
        "combo": combo_str,
        "bt_return": bt_return,
        "bt_margin_failures": margin_failures,
        "bt_trades": total_trades,
        "bt_win_rate": win_rate,
        "score_balanced": row_data.get('score_balanced', 0),
        "vec_return": row_data.get('vec_return', 0),
        "vec_trades": row_data.get('vec_trades', 0),
        "vec_win_rate": row_data.get('vec_win_rate', 0),
        "icir": row_data.get('icir', 0)
    }

def main():
    parser = argparse.ArgumentParser(description="Run BT on scored strategies")
    parser.add_argument("--input", type=str, required=True, help="Path to scored CSV file")
    parser.add_argument("--topk", type=int, default=100, help="Number of top strategies to run")
    args = parser.parse_args()

    print("=" * 80)
    print(f"BT Backtest on Top {args.topk} Scored Strategies")
    print("=" * 80)

    # 1. Load Scored Results
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return

    df_scored = pd.read_csv(input_path)
    print(f"✅ Loaded scored results: {len(df_scored)} combos")

    # Sort by score_balanced and take top K
    if 'score_balanced' not in df_scored.columns:
        print("❌ Column 'score_balanced' not found in input file")
        return
    
    df_top = df_scored.nlargest(args.topk, 'score_balanced')
    print(f"✅ Selected top {len(df_top)} strategies (Min Score: {df_top['score_balanced'].min():.4f})")

    # 2. Load Data
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

    # Prepare data feeds
    data_feeds = {}
    for ticker in etf_codes:
        df = pd.DataFrame(
            {
                "open": ohlcv["open"][ticker],
                "high": ohlcv["high"][ticker],
                "low": ohlcv["low"][ticker],
                "close": ohlcv["close"][ticker],
                "volume": ohlcv["volume"][ticker],
            }
        )
        df = df.reindex(dates)
        df = df.ffill().fillna(0.01)
        data_feeds[ticker] = df

    print(f"✅ Data loaded: {len(dates)} days x {len(etf_codes)} ETFs")

    # 4. Run Backtest
    num_workers = 30
    print(f"🚀 Starting multiprocessing backtest (Workers: {num_workers})...")

    tasks = [row.to_dict() for _, row in df_top.iterrows()]

    results = []
    with mp.Pool(processes=num_workers, initializer=init_worker, initargs=(data_feeds, std_factors, timing_series, etf_codes)) as pool:
        for res in tqdm(pool.imap(process_combo, tasks), total=len(tasks), desc="BT Backtest"):
            results.append(res)

    # 5. Save Results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / "results" / f"bt_scored_top{args.topk}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    df_results = pd.DataFrame(results)
    
    # Calculate alignment
    df_results['diff_pp'] = (df_results['bt_return'] - df_results['vec_return']) * 100
    
    df_results.to_csv(output_dir / "bt_results.csv", index=False)

    print(f"\n✅ BT Backtest Complete")
    print(f"   Output: {output_dir}")
    print(f"   Mean Diff: {df_results['diff_pp'].abs().mean():.4f} pp")
    print(f"   Max Diff: {df_results['diff_pp'].abs().max():.4f} pp")

if __name__ == "__main__":
    main()
