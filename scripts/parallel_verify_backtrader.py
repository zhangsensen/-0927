
import backtrader as bt
import pandas as pd
import numpy as np
import sys
import os
import yaml
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import multiprocessing
from functools import partial

# Add project root to path
sys.path.insert(0, os.getcwd())

from etf_rotation_optimized.core.data_loader import DataLoader
from etf_rotation_optimized.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_rotation_optimized.core.cross_section_processor import CrossSectionProcessor
from etf_rotation_optimized.core.market_timing import LightTimingModule

# Constants
FREQ = 8
POS_SIZE = 3
INITIAL_CAPITAL = 1_000_000.0
COMMISSION_RATE = 0.0002
LOOKBACK = 252

# Global variables for workers (Copy-on-Write optimization)
GLOBAL_DATA_FEEDS = {}
GLOBAL_STD_FACTORS = {}
GLOBAL_TIMING = None
GLOBAL_ETF_CODES = []
GLOBAL_INDEX = None

class PandasData(bt.feeds.PandasData):
    params = (
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', None),
    )

class GenericStrategy(bt.Strategy):
    params = (
        ('scores', None),      # DataFrame of scores (index=date, columns=tickers)
        ('timing', None),      # Series of timing (index=date)
        ('etf_codes', None),   # List of ETF codes
        ('freq', FREQ),
        ('pos_size', POS_SIZE),
    )

    def __init__(self):
        self.inds = {}
        self.etf_map = {d._name: d for d in self.datas}
        
    def next(self):
        if len(self) < LOOKBACK:
            return

        # Rebalance every FREQ days
        if len(self) % self.params.freq == 0:
            dt = self.datas[0].datetime.date(0)
            dt_ts = pd.Timestamp(dt)
            self.rebalance(dt_ts)

    def rebalance(self, current_date):
        try:
            prev_date = self.datas[0].datetime.date(-1)
            prev_ts = pd.Timestamp(prev_date)
            
            if prev_ts not in self.params.scores.index:
                return

            row = self.params.scores.loc[prev_ts]
            valid = row[row.notna() & (row != 0)]
            
            target_weights = {}
            if len(valid) >= self.params.pos_size:
                top_k = valid.sort_values().tail(self.params.pos_size).index.tolist()
                
                timing_ratio = 1.0
                if self.params.timing is not None and current_date in self.params.timing.index:
                    timing_ratio = self.params.timing.loc[current_date]
                
                weight = timing_ratio / len(top_k)
                for ticker in top_k:
                    target_weights[ticker] = weight
            
            sells = []
            buys = []
            
            for ticker in self.params.etf_codes:
                target = target_weights.get(ticker, 0.0)
                data = self.etf_map[ticker]
                
                value = self.broker.get_value([data])
                total_value = self.broker.getvalue()
                current_pct = value / total_value if total_value > 0 else 0
                
                if target < current_pct - 0.001: # Sell
                    sells.append((data, target))
                elif target > current_pct + 0.001: # Buy
                    buys.append((data, target))
                else:
                    sells.append((data, target))
            
            for data, target in sells:
                self.order_target_percent(data, target=target)
            for data, target in buys:
                self.order_target_percent(data, target=target)
                
        except Exception as e:
            pass

def worker_task(strategy_row):
    """
    Worker function to run a single backtest.
    Uses global variables to access data (efficient on Linux via fork).
    """
    try:
        combo_name = strategy_row['combo']
        orig_return = strategy_row['total_return']
        rank = strategy_row['real_rank']
        
        factors = [f.strip() for f in combo_name.split(" + ")]
        
        # Combine Scores
        combined_score = pd.DataFrame(0.0, index=GLOBAL_INDEX, columns=GLOBAL_ETF_CODES)
        valid_factors = True
        for f in factors:
            if f not in GLOBAL_STD_FACTORS:
                valid_factors = False
                break
            combined_score = combined_score.add(GLOBAL_STD_FACTORS[f], fill_value=0)
            
        if not valid_factors:
            return None
            
        # Run Backtrader
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(INITIAL_CAPITAL)
        cerebro.broker.setcommission(commission=COMMISSION_RATE, leverage=2.0)
        cerebro.broker.set_coc(True)
        
        # Add data feeds (referencing global dict)
        for ticker, df in GLOBAL_DATA_FEEDS.items():
            data = PandasData(dataname=df, name=ticker)
            cerebro.adddata(data)
            
        cerebro.addstrategy(GenericStrategy, 
                            scores=combined_score, 
                            timing=GLOBAL_TIMING,
                            etf_codes=GLOBAL_ETF_CODES)
        
        # Run without output
        start_val = cerebro.broker.getvalue()
        cerebro.run()
        end_val = cerebro.broker.getvalue()
        
        bt_return = (end_val / start_val) - 1
        diff = bt_return - orig_return
        
        return {
            'real_rank': rank,
            'combo': combo_name,
            'orig_return': orig_return,
            'bt_return': bt_return,
            'diff': diff,
            'bt_final_equity': end_val
        }
    except Exception as e:
        # print(f"Error in strategy {strategy_row.get('real_rank')}: {e}")
        return None

def run_parallel_verification():
    print("🚀 启动 Backtrader 并行全量验证 (Top 5000)...")
    
    # 1. Load Top 5000 Strategies
    summary_path = Path("results/top5000_summary.csv")
    if not summary_path.exists():
        print("❌ Summary file not found!")
        return
        
    df_summary = pd.read_csv(summary_path)
    strategies_to_run = df_summary # Run all 5000
    
    # 2. Load Config & Data
    config_path = Path("configs/combo_wfo_config.yaml")
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
    print("🔧 计算因子...")
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)
    
    print("📐 标准化因子...")
    factor_names = raw_factors_df.columns.get_level_values(0).unique().tolist()
    raw_factors = {fname: raw_factors_df[fname] for fname in factor_names}
    
    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)
    
    # 4. Timing
    timing_module = LightTimingModule()
    timing_series = timing_module.compute_position_ratios(ohlcv["close"])
    
    # 5. Prepare Globals
    global GLOBAL_DATA_FEEDS, GLOBAL_STD_FACTORS, GLOBAL_TIMING, GLOBAL_ETF_CODES, GLOBAL_INDEX
    
    GLOBAL_STD_FACTORS = std_factors
    GLOBAL_TIMING = timing_series
    GLOBAL_ETF_CODES = ohlcv['close'].columns.tolist()
    GLOBAL_INDEX = std_factors[factor_names[0]].index
    
    print("📥 准备数据 Feeds...")
    for ticker in GLOBAL_ETF_CODES:
        df = pd.DataFrame({
            'open': ohlcv['open'][ticker],
            'high': ohlcv['high'][ticker],
            'low': ohlcv['low'][ticker],
            'close': ohlcv['close'][ticker],
            'volume': ohlcv['volume'][ticker]
        })
        df = df.reindex(GLOBAL_INDEX)
        df = df.ffill().fillna(0.01)
        GLOBAL_DATA_FEEDS[ticker] = df

    # 6. Parallel Execution
    num_cores = min(multiprocessing.cpu_count(), 16) # Use up to 16 cores
    print(f"\n🔥 使用 {num_cores} 个核心并行回测 {len(strategies_to_run)} 个策略...")
    
    results = []
    strategies_list = [row for _, row in strategies_to_run.iterrows()]
    
    # Use chunksize to reduce IPC overhead
    chunksize = max(1, len(strategies_list) // (num_cores * 4))
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        # Use tqdm to show progress
        for res in tqdm(pool.imap(worker_task, strategies_list, chunksize=chunksize), 
                       total=len(strategies_list), 
                       desc="Parallel Backtesting"):
            if res:
                results.append(res)
                
                # Incremental save every 500
                if len(results) % 500 == 0:
                     pd.DataFrame(results).to_csv("results/top5000_backtrader_verification_parallel.csv", index=False)

    # Final Save
    output_path = Path("results/top5000_backtrader_verification_parallel.csv")
    df_res = pd.DataFrame(results)
    df_res.to_csv(output_path, index=False)
    
    print("-" * 100)
    print(f"💾 完整结果已保存至: {output_path}")
    
    avg_diff = df_res['diff'].mean()
    print(f"\n📊 平均差异: {avg_diff*100:.2f}%")
    
    print("\n🏆 Backtrader 验证后 Top 20:")
    df_res_sorted = df_res.sort_values('bt_return', ascending=False).head(20)
    print(df_res_sorted[['real_rank', 'bt_return', 'orig_return', 'diff', 'combo']].to_string(index=False))
    
    print(f"✅ 验证完成")

if __name__ == "__main__":
    run_parallel_verification()
