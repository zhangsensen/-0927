
import backtrader as bt
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import sys
from tqdm import tqdm
import multiprocessing
from functools import partial

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import shift_timing_signal, generate_rebalance_schedule

# --- Strategy Class (Must be at top level for pickling) ---
class V3Strategy(bt.Strategy):
    params = (
        ('scores', None),      # DataFrame of scores
        ('vol_regime', None),  # Series of exposure
        ('timing', None),      # Series of timing
        ('rebalance_schedule', None),
        ('pos_size', 2),
    )

    def __init__(self):
        self.rebalance_set = set(self.params.rebalance_schedule)
        # Cache data feeds for faster access
        self.dnames = {d._name: d for d in self.datas}

    def next(self):
        # Optimization: Use date(0) directly
        dt = self.datas[0].datetime.date(0)
        
        if dt not in self.rebalance_set:
            return
            
        # Get Exposure & Timing
        # Optimization: Use get() with default instead of try/except
        ts = pd.Timestamp(dt)
        
        # Note: accessing .loc on Series is reasonably fast, but could be optimized
        # by converting to dict or aligned array if needed. 
        # For now, let's keep it simple as logic overhead is small compared to engine.
        try:
            exposure = self.params.vol_regime.at[ts]
            timing = self.params.timing.at[ts]
        except:
            exposure = 1.0
            timing = 1.0
            
        target_exposure = exposure * timing
        
        # Get Scores
        try:
            current_scores = self.params.scores.loc[ts]
        except:
            return
            
        # Select Top N
        # Optimization: nlargest is faster than sort_values().head()
        valid_scores = current_scores.dropna()
        if valid_scores.empty:
            return
            
        targets = valid_scores.nlargest(self.params.pos_size).index.tolist()
        
        # Execute
        # 1. Sell non-targets
        for name, d in self.dnames.items():
            pos = self.getposition(d).size
            if pos > 0 and name not in targets:
                self.order_target_percent(d, target=0.0)
                
        # 2. Buy targets
        if not targets:
            return
            
        weight = target_exposure / len(targets)
        for name in targets:
            d = self.dnames[name]
            self.order_target_percent(d, target=weight)

def run_single_bt(strategy_cls, data_feeds, scores, vol_regime, timing, schedule, pos_size, cheat_on_close):
    cerebro = bt.Cerebro()
    
    # Optimization: Pre-load feeds? No, must add to instance.
    for name, df in data_feeds.items():
        data = bt.feeds.PandasData(dataname=df, name=name)
        cerebro.adddata(data)
        
    cerebro.addstrategy(
        strategy_cls,
        scores=scores,
        vol_regime=vol_regime,
        timing=timing,
        rebalance_schedule=schedule,
        pos_size=pos_size
    )
    
    cerebro.broker.setcash(1_000_000.0)
    cerebro.broker.setcommission(commission=0.0002)
    
    if cheat_on_close:
        cerebro.broker.set_checksubmit(False)
        cerebro.broker.set_coc(True)
    else:
        cerebro.broker.set_checksubmit(True)
        cerebro.broker.set_coc(False)
        
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.0, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    
    results = cerebro.run()
    strat = results[0]
    
    final_val = cerebro.broker.getvalue()
    total_ret = final_val / 1_000_000.0 - 1
    
    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0.0)
    if sharpe is None: sharpe = 0.0
    
    max_dd = strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0.0) / 100.0
    
    years = 5.95 # Approx
    ann_ret = (1 + total_ret) ** (1/years) - 1
    
    return {
        'ann_ret': ann_ret,
        'max_dd': -max_dd,
        'sharpe': sharpe
    }

def worker_task(args):
    """
    Worker function for multiprocessing.
    args: (row_dict, std_factors, exposure_s, timing_s, sched_dates, data_feeds)
    """
    row, std_factors, exposure_s, timing_s, sched_dates, data_feeds = args
    
    combo_str = row['combo']
    factors = [f.strip() for f in combo_str.split('+')]
    
    # Reconstruct scores (fast)
    # Assuming std_factors is a dict of Series/DFs
    # We need the index and columns from one of them to init
    first_fac = next(iter(std_factors.values()))
    scores = pd.DataFrame(0.0, index=first_fac.index, columns=first_fac.columns)
    
    for f in factors:
        scores += std_factors[f]
        
    # Run BT-A
    res_a = run_single_bt(V3Strategy, data_feeds, scores, exposure_s, timing_s, sched_dates, 2, cheat_on_close=True)
    
    # Run BT-B
    res_b = run_single_bt(V3Strategy, data_feeds, scores, exposure_s, timing_s, sched_dates, 2, cheat_on_close=False)
    
    return {
        'combo': combo_str,
        'vec_ann_ret': row['ann_return'],
        'vec_max_dd': row['max_dd'],
        'vec_sharpe': row['sharpe'],
        
        'bt_a_ann_ret': res_a['ann_ret'],
        'bt_a_max_dd': res_a['max_dd'],
        'bt_a_sharpe': res_a['sharpe'],
        
        'bt_b_ann_ret': res_b['ann_ret'],
        'bt_b_max_dd': res_b['max_dd'],
        'bt_b_sharpe': res_b['sharpe'],
        
        'engine_return_gap': abs(row['ann_return'] - res_a['ann_ret']),
        'engine_dd_gap': abs(row['max_dd'] - res_a['max_dd']),
        
        'exec_return_ratio': res_b['ann_ret'] / res_a['ann_ret'] if res_a['ann_ret'] != 0 else 0,
        'exec_sharpe_ratio': res_b['sharpe'] / res_a['sharpe'] if res_a['sharpe'] != 0 else 0
    }

def main():
    print("🚀 Starting Optimized Batch BT Cross-Check...")
    
    # 1. Load Data
    config_path = ROOT / 'configs/combo_wfo_config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    loader = DataLoader(data_dir=config['data'].get('data_dir'), cache_dir=config['data'].get('cache_dir'))
    ohlcv = loader.load_ohlcv(etf_codes=config['data']['symbols'], start_date=config['data']['start_date'], end_date=config['data']['end_date'])
    
    # Prepare Data Feeds (Dict of DFs)
    data_feeds = {}
    close_df = ohlcv['close'].ffill().bfill()
    open_df = ohlcv['open'].ffill().bfill()
    high_df = ohlcv['high'].ffill().bfill()
    low_df = ohlcv['low'].ffill().bfill()
    vol_df = ohlcv['volume'].fillna(0)
    
    for col in ohlcv['close'].columns:
        data_feeds[col] = pd.DataFrame({
            'open': open_df[col],
            'high': high_df[col],
            'low': low_df[col],
            'close': close_df[col],
            'volume': vol_df[col]
        })
        
    # 2. Compute Factors
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)
    factor_names_list = sorted(raw_factors_df.columns.get_level_values(0).unique().tolist())
    
    processor = CrossSectionProcessor(verbose=False)
    # Convert to dict of DataFrames for easier pickling/passing
    std_factors = {fname: raw_factors_df[fname] for fname in factor_names_list}
    std_factors = processor.process_all_factors(std_factors)
    
    # 3. Common Inputs
    if '510300' in ohlcv['close'].columns:
        hs300 = ohlcv['close']['510300']
    else:
        hs300 = ohlcv['close'].iloc[:, 0]
    rets = hs300.pct_change()
    hv = rets.rolling(window=20, min_periods=20).std() * np.sqrt(252) * 100
    hv_5d = hv.shift(5)
    regime_vol = (hv + hv_5d) / 2
    
    exposure_s = pd.Series(1.0, index=regime_vol.index)
    exposure_s[regime_vol >= 25] = 0.7
    exposure_s[regime_vol >= 30] = 0.4
    exposure_s[regime_vol >= 40] = 0.1
    
    timing_module = LightTimingModule()
    timing_signals = timing_module.compute_position_ratios(ohlcv['close'])
    timing_s = shift_timing_signal(timing_signals)
    timing_s = pd.Series(timing_s, index=ohlcv['close'].index)
    
    dates = ohlcv['close'].index
    T = len(dates)
    sched_indices = generate_rebalance_schedule(T, 252, 3)
    sched_dates = [dates[i].date() for i in sched_indices]
    
    # 4. Load Candidates
    candidates = pd.read_csv(ROOT / 'results/v3_top200_candidates.csv')
    
    # Prepare Args
    tasks = []
    for idx, row in candidates.iterrows():
        tasks.append((row, std_factors, exposure_s, timing_s, sched_dates, data_feeds))
        
    # 5. Run Parallel
    # Use fewer than max cores to avoid memory pressure if data is large
    # But here data is shared via COW (Copy On Write) on Linux, so it should be fine.
    n_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"🔥 Running on {n_workers} cores...")
    
    results = []
    with multiprocessing.Pool(processes=n_workers) as pool:
        for res in tqdm(pool.imap(worker_task, tasks), total=len(tasks)):
            results.append(res)
            
    # Save
    res_df = pd.DataFrame(results)
    res_df.to_csv(ROOT / 'results/v3_top200_bt_crosscheck.csv', index=False)
    print("✅ Done.")

if __name__ == "__main__":
    main()
