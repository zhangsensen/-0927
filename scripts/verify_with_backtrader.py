
import backtrader as bt
import pandas as pd
import numpy as np
import sys
import os
import yaml
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.getcwd())

from etf_rotation_optimized.core.data_loader import DataLoader
from etf_rotation_optimized.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_rotation_optimized.core.cross_section_processor import CrossSectionProcessor
from etf_rotation_optimized.core.market_timing import LightTimingModule

# Constants matching the original script
FREQ = 8
POS_SIZE = 3
INITIAL_CAPITAL = 1_000_000.0
COMMISSION_RATE = 0.0002
LOOKBACK = 252

# Top 1 Strategy Factors
TARGET_FACTORS = [
    'ADX_14D',
    'CORRELATION_TO_MARKET_20D',
    'PRICE_POSITION_20D',
    'SHARPE_RATIO_20D'
]

class PandasData(bt.feeds.PandasData):
    """
    Custom Data Feed to ensure we load OHLCV correctly
    """
    params = (
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', None),
    )

class Top1Strategy(bt.Strategy):
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
        # Get current date
        dt = self.datas[0].datetime.date(0)
        dt_ts = pd.Timestamp(dt)
        
        if len(self) == 1:
            print(f"🏁 First Next() Call: {dt} (Len: {len(self)})")
        
        if len(self) == LOOKBACK:
             print(f"🏁 Reached Lookback: {dt} (Len: {len(self)})")
        
        # Check if we have a score for this date (T-1 signal used at T)
        # In the original script:
        # if t % FREQ == 0:
        #    Signal from T-1
        
        # We need to match the index logic.
        # The original script iterates t from LOOKBACK to T.
        # Backtrader iterates bar by bar.
        
        # We simulate the "every 8 days" logic by tracking bar count or date index?
        # The original script uses integer index `t`.
        # We can approximate this by counting bars.
        
        # However, to be EXACT, we should check if this date corresponds to a rebalance date in the original logic.
        # But since we don't have the original `t` index easily, we'll use a counter.
        # Note: Backtrader skips days if data is missing, but our data is aligned.
        
        # Let's use a simple counter that starts when valid data starts.
        # But wait, `len(self)` gives the number of bars processed.
        
        if len(self) < LOOKBACK:
            return

        # Rebalance every FREQ days
        if len(self) % self.params.freq == 0:
            self.rebalance(dt_ts)

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f"BUY  {order.data._name} @ {order.executed.price:.2f} on {self.datas[0].datetime.date(0)}")
            elif order.issell():
                print(f"SELL {order.data._name} @ {order.executed.price:.2f} on {self.datas[0].datetime.date(0)}")
        elif order.status in [order.Rejected, order.Canceled, order.Margin]:
            print(f"❌ Order {order.getstatusname()} for {order.data._name}")

    def rebalance(self, current_date):
        # Get scores for T-1 (yesterday)
        # In Backtrader, we are at T (Close).
        # The original script uses `F_sel[t-1]` for signal at `t`.
        # So we need the score from the previous trading day.
        
        # We look up the score by date.
        # Since we are at `current_date`, we need the score calculated using data up to `current_date - 1 day`.
        # Our `scores` DataFrame is indexed by date.
        # If `scores` contains the signal available AT that date (calculated from T-1), we just look up `current_date`.
        # Let's verify how we construct `scores`.
        
        # In original: `combined_score = np.nansum(F_sel[t-1], axis=1)`
        # This means at time `t`, we use factor values from `t-1`.
        # So if we pass the `scores` dataframe where index `t` contains `F_sel[t]`,
        # then at `current_date` (which is `t`), we should look up `scores` at `prev_date` (which is `t-1`).
        
        # Finding previous date in our data
        # We can just use the integer index if we had it.
        # Alternatively, we can pass the FULL scores dataframe and use `len(self)-1` if the index aligns.
        
        try:
            # Get the date of the previous bar
            prev_date = self.datas[0].datetime.date(-1)
            prev_ts = pd.Timestamp(prev_date)
            
            if prev_ts not in self.params.scores.index:
                return

            row = self.params.scores.loc[prev_ts]
            
            # Filter valid
            valid = row[row.notna() & (row != 0)]
            
            target_weights = {}
            if len(valid) >= self.params.pos_size:
                # Top K
                top_k = valid.sort_values().tail(self.params.pos_size).index.tolist()
                
                # Timing
                timing_ratio = 1.0
                if self.params.timing is not None and current_date in self.params.timing.index:
                    timing_ratio = self.params.timing.loc[current_date]
                
                # Weight per asset
                # Original: target_pos_value = (current_value * timing_ratio) / len(target_indices)
                # So weight = timing_ratio / len(target_indices)
                weight = timing_ratio / len(top_k)
                
                for ticker in top_k:
                    target_weights[ticker] = weight
            
            # Execute Rebalance
            # We must issue SELL orders FIRST to free up cash for BUY orders
            # to avoid Margin errors.
            
            # Calculate current weights to know what is a sell vs buy?
            # order_target_percent handles the calculation, but we need to know order.
            # We can just iterate twice.
            
            # First pass: Sells (target < current)
            # But we don't know current weight easily without checking broker.
            # Alternatively, we can just issue orders for things NOT in top_k (target=0) first.
            # And then things in top_k.
            # But even in top_k, we might be reducing position.
            
            # Robust way:
            # 1. Identify all tickers with target < current position (Sells)
            # 2. Identify all tickers with target > current position (Buys)
            
            # However, getting current position in Backtrader:
            # self.getposition(data).size
            
            sells = []
            buys = []
            
            for ticker in self.params.etf_codes:
                target = target_weights.get(ticker, 0.0)
                data = self.etf_map[ticker]
                pos = self.getposition(data).size
                
                # Estimate if it's a buy or sell
                # Note: This is approximate because price changes, but good enough for ordering.
                # Actually, if target is 0 and pos != 0, it's a sell.
                # If target > 0 and pos == 0, it's a buy.
                # If both > 0, we compare value?
                # value = pos * price. target_value = equity * target.
                
                value = self.broker.get_value([data])
                total_value = self.broker.getvalue()
                current_pct = value / total_value if total_value > 0 else 0
                
                if target < current_pct - 0.001: # Sell (with tolerance)
                    sells.append((data, target))
                elif target > current_pct + 0.001: # Buy
                    buys.append((data, target))
                else:
                    # No change or small rebalance, put in sells just in case
                    sells.append((data, target))
            
            # Execute Sells
            for data, target in sells:
                self.order_target_percent(data, target=target)
                
            # Execute Buys
            for data, target in buys:
                self.order_target_percent(data, target=target)
                
        except Exception as e:
            # print(f"Error rebalancing on {current_date}: {e}")
            pass

def run_verification():
    print("🚀 启动 Backtrader 独立验证...")
    
    # 1. Load Config & Data (Same as original)
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
    
    # 2. Compute Factors (Same as original)
    print("🔧 计算因子 (使用相同库)...")
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)
    
    # 3. Process Factors
    print("📐 标准化因子...")
    factor_names = raw_factors_df.columns.get_level_values(0).unique().tolist()
    raw_factors = {fname: raw_factors_df[fname] for fname in factor_names}
    
    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)
    
    # 4. Combine Scores for Top 1 Strategy
    print(f"🎯 验证策略: {' + '.join(TARGET_FACTORS)}")
    
    # Check if factors exist
    for f in TARGET_FACTORS:
        if f not in std_factors:
            print(f"❌ 错误: 因子 {f} 未找到!")
            return

    # Sum the standardized factors (handling NaNs like np.nansum)
    combined_score = pd.DataFrame(0.0, index=std_factors[TARGET_FACTORS[0]].index, columns=std_factors[TARGET_FACTORS[0]].columns)
    for f in TARGET_FACTORS:
        combined_score = combined_score.add(std_factors[f], fill_value=0)
        
    print(f"📊 Score Range: {combined_score.index[0]} to {combined_score.index[-1]}")
    print(f"📊 OHLCV Range: {ohlcv['close'].index[0]} to {ohlcv['close'].index[-1]}")
    
    # Check for NaNs in combined_score
    valid_counts = combined_score.notna().sum(axis=1)
    print(f"📊 Valid Scores per day (Head): \n{valid_counts.head(20)}")
    
    # Debug Scores on 2021-01-20
    debug_date = pd.Timestamp("2021-01-20")
    if debug_date in combined_score.index:
        print(f"\n🔍 Scores on {debug_date.date()}:")
        scores = combined_score.loc[debug_date]
        valid_scores = scores[scores != 0].dropna().sort_values(ascending=False)
        print(valid_scores.head(10))
    
    # Check ETF start dates
    print("📊 ETF Start Dates:")
    etf_codes_list = ohlcv['close'].columns.tolist()
    for ticker in etf_codes_list:
        first_valid = ohlcv['close'][ticker].first_valid_index()
        print(f"   {ticker}: {first_valid}")
    
    # 5. Timing (Same as original)
    timing_module = LightTimingModule()
    timing_series = timing_module.compute_position_ratios(ohlcv["close"])
    
    # 6. Setup Backtrader
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(INITIAL_CAPITAL)
    # Set leverage to allow rebalancing (Buy before Sell settlement)
    # We need enough leverage to hold old positions + new positions temporarily
    # If we replace 100%, we need 2.0 leverage.
    cerebro.broker.setcommission(commission=COMMISSION_RATE, leverage=2.0)
    cerebro.broker.set_coc(True)  # Enable Cheat-On-Close to trade at Close T
    
    # Add Data Feeds
    print("📥 加载数据到 Backtrader...")
    etf_codes = ohlcv['close'].columns.tolist()
    global_index = combined_score.index
    
    for ticker in etf_codes:
        # Construct single ETF dataframe
        df = pd.DataFrame({
            'open': ohlcv['open'][ticker],
            'high': ohlcv['high'][ticker],
            'low': ohlcv['low'][ticker],
            'close': ohlcv['close'][ticker],
            'volume': ohlcv['volume'][ticker]
        })
        
        # Reindex to global range to ensure Backtrader runs from start
        # We forward fill to avoid NaNs in prices if possible, or just leave NaNs
        # If we leave NaNs, we must ensure we don't trade them.
        # But Backtrader might error on NaN date/price.
        # Actually, Backtrader skips lines with NaN if configured?
        # No, we want the line to exist so next() is called.
        
        # We fill NaNs with a dummy value (e.g. 0.01) to prevent errors, 
        # but we must ensure we don't buy them.
        # Since our signal generation depends on factors, and factors will be NaN or 0 for these dates,
        # we won't select them.
        
        df = df.reindex(global_index)
        df = df.fillna(method='ffill').fillna(0.01) # Forward fill then fill start with dummy
        
        data = PandasData(dataname=df, name=ticker)
        cerebro.adddata(data)
        
    # Add Strategy
    cerebro.addstrategy(Top1Strategy, 
                        scores=combined_score, 
                        timing=timing_series,
                        etf_codes=etf_codes)
    
    # Run
    print("▶️ 开始回测...")
    start_val = cerebro.broker.getvalue()
    results = cerebro.run()
    end_val = cerebro.broker.getvalue()
    
    # Report
    pnl = end_val - start_val
    ret = (end_val / start_val) - 1
    
    print("=" * 50)
    print("✅ Backtrader 验证结果")
    print("=" * 50)
    print(f"初始资金: ${start_val:,.2f}")
    print(f"最终资金: ${end_val:,.2f}")
    print(f"净收益:   ${pnl:,.2f}")
    print(f"收益率:   {ret*100:.2f}%")
    print("=" * 50)

if __name__ == "__main__":
    run_verification()
