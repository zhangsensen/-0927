import os
import sys
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import logging
import random
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Add project root
sys.path.append(os.getcwd())

try:
    from etf_strategy.core.data_loader import DataLoader
    from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
    from etf_strategy.core.cross_section_processor import CrossSectionProcessor
    from etf_strategy.core.market_timing import LightTimingModule
except ImportError:
    sys.path.append(str(Path(os.getcwd()).parent))
    from etf_strategy.core.data_loader import DataLoader
    from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
    from etf_strategy.core.cross_section_processor import CrossSectionProcessor
    from etf_strategy.core.market_timing import LightTimingModule

COMMISSION_RATE = 0.0002 # 2 bps

def analyze_random_trades(sample_size=20):
    print("="*80)
    print(f"🎲 随机抽样交易特征分析 (Random Sample Trade Analysis)")
    print(f"🎯 样本数量: {sample_size}")
    print("="*80)
    
    # 1. Load Results
    result_file = 'results_combo_wfo/20251126_190236_20251127_125624/top12597_backtest_by_ic_20251126_190236_20251127_125624_full.csv'
    if not os.path.exists(result_file):
        print(f"❌ 结果文件不存在: {result_file}")
        return

    df_results = pd.read_csv(result_file)
    # Filter for positive sharpe to avoid complete garbage, or just sample all?
    # User wants to verify "results", implying the good ones. Let's sample from Top 2000 to be relevant.
    # Or just sample from the whole set to see the general behavior.
    # Let's sample from the whole set but maybe bias slightly towards "usable" strategies (Sharpe > 0).
    df_pool = df_results[df_results['sharpe'] > 0]
    sample_df = df_pool.sample(n=sample_size, random_state=int(time.time()))
    
    print(f"📚 从 {len(df_pool)} 个正收益组合中随机抽取 {sample_size} 个")

    # 2. Load Data (Once)
    print("\n📥 加载基础数据...")
    with open("configs/combo_wfo_config.yaml") as f:
        config = yaml.safe_load(f)
        
    loader = DataLoader(
        data_dir=config["data"]["data_dir"],
        cache_dir=config["data"]["cache_dir"]
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
        use_cache=True
    )
    
    # 3. Compute Factors (Once)
    print("🧮 计算全量因子库...")
    lib = PreciseFactorLibrary()
    factors_df = lib.compute_all_factors(ohlcv)
    
    factors_dict = {}
    available_factors = factors_df.columns.get_level_values(0).unique()
    for f in available_factors:
        factors_dict[f] = factors_df[f]
        
    proc = CrossSectionProcessor(
        lower_percentile=config["cross_section"]["winsorize_lower"] * 100,
        upper_percentile=config["cross_section"]["winsorize_upper"] * 100,
        verbose=False
    )
    std_factors = proc.process_all_factors(factors_dict)
    
    factor_names = sorted(std_factors.keys())
    F_data_full = np.stack([std_factors[n].values for n in factor_names], axis=-1)
    
    close_prices = ohlcv["close"].values
    dates = ohlcv["close"].index
    etf_codes = ohlcv["close"].columns.tolist()
    
    # Timing
    timing_module = LightTimingModule(extreme_threshold=-0.3, extreme_position=0.3)
    timing_series = timing_module.compute_position_ratios(ohlcv['close'])
    timing_series = timing_series.shift(1).fillna(1.0)
    timing_arr = timing_series.values
    
    # 4. Analysis Loop
    stats_list = []
    
    print(f"\n🚀 开始分析...")
    
    for idx, (i, row) in enumerate(sample_df.iterrows()):
        combo_str = row['combo']
        freq = int(row.get('test_freq', 8))
        pos_size = int(row.get('test_position_size', 3))
        
        # Parse combo
        combo_factors = [f.strip() for f in combo_str.split(" + ")]
        try:
            factor_indices = [factor_names.index(f) for f in combo_factors]
        except ValueError:
            continue
            
        F_sel = F_data_full[:, :, factor_indices]
        T, N, F = F_sel.shape
        lookback = 252
        
        # Simulation for Trade Generation
        # We use the "Signal Log" approach which is faster and sufficient for trade stats
        
        signal_log = []
        
        for t in range(lookback + 1, T):
            if (t - (lookback + 1)) % freq == 0:
                date = dates[t]
                f_day = F_sel[t] # Data known at T
                
                # Equal Weight Signal
                score = np.nanmean(f_day, axis=1)
                valid_mask = ~np.isnan(score)
                
                codes = []
                if valid_mask.sum() >= pos_size:
                    score_filled = score.copy()
                    score_filled[~valid_mask] = -np.inf
                    top_idx = np.argsort(score_filled)[-pos_size:]
                    codes = [etf_codes[x] for x in top_idx]
                
                # Timing (for T+1 execution, we use timing known at T+1? No, known at T for T+1 trade?)
                # Let's stick to the logic: Signal generated at T, Executed at T+1 Close.
                # We need timing signal for T+1.
                # timing_arr is aligned such that timing_arr[k] is the position ratio for day k.
                # So for T+1, we check timing_arr[t+1].
                
                next_t = t + 1
                timing = 1.0
                if next_t < len(timing_arr):
                    timing = timing_arr[next_t]
                
                signal_log.append({
                    "date": date,
                    "holdings": set(codes),
                    "timing": timing
                })
        
        # Reconstruct Trades
        trades = []
        open_positions = {} # code -> {entry_date, entry_price}
        
        def get_price(d, c):
            try:
                ix = dates.get_loc(d)
                cix = etf_codes.index(c)
                return close_prices[ix, cix]
            except:
                return None

        for k in range(len(signal_log)):
            sig = signal_log[k]
            sig_date = sig["date"]
            
            # Execution Date (T+1)
            try:
                sig_idx = dates.get_loc(sig_date)
                exec_idx = sig_idx + 1
                if exec_idx >= len(dates): break
                exec_date = dates[exec_idx]
            except:
                continue
            
            # Determine Target Holdings based on Timing
            # If timing < 0.1 (e.g. 0), we hold nothing.
            # If timing > 0, we hold the target codes.
            # (Simplified: We don't track partial position reductions due to timing, just In/Out)
            # Actually, LightTimingModule usually outputs 0.3 or 1.0.
            # If 0.3, we still hold the ETFs, just less. So the "Trade" (Entry/Exit) logic is based on the Set of Codes.
            # Unless timing goes to 0? (LightTimingModule min is usually > 0 or 0. If 0, we sell all).
            
            target_codes = sig["holdings"]
            if sig["timing"] < 0.01:
                target_codes = set()
            
            # Check Sells
            current_codes = list(open_positions.keys())
            for code in current_codes:
                if code not in target_codes:
                    entry = open_positions.pop(code)
                    exit_price = get_price(exec_date, code)
                    if exit_price:
                        ret = (exit_price - entry["entry_price"]) / entry["entry_price"] - COMMISSION_RATE * 2
                        trades.append(ret)
            
            # Check Buys
            for code in target_codes:
                if code not in open_positions:
                    entry_price = get_price(exec_date, code)
                    if entry_price:
                        open_positions[code] = {
                            "entry_date": exec_date,
                            "entry_price": entry_price
                        }
        
        # Stats
        if len(trades) > 0:
            trades_arr = np.array(trades)
            win_rate = (trades_arr > 0).mean()
            avg_win = trades_arr[trades_arr > 0].mean() if (trades_arr > 0).any() else 0
            avg_loss = trades_arr[trades_arr < 0].mean() if (trades_arr < 0).any() else 0
            profit_factor = abs(trades_arr[trades_arr > 0].sum() / trades_arr[trades_arr < 0].sum()) if (trades_arr < 0).any() else 999.0
            
            stats_list.append({
                "combo": combo_str,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "n_trades": len(trades),
                "sharpe": row['sharpe']
            })
            
        if (idx + 1) % 5 == 0:
            print(f"  ...已分析 {idx + 1}/{sample_size}")

    # Summary
    df_stats = pd.DataFrame(stats_list)
    
    print("\n📊 随机样本交易特征统计:")
    print(f"  平均胜率 (Win Rate): {df_stats['win_rate'].mean():.2%}")
    print(f"  胜率中位数: {df_stats['win_rate'].median():.2%}")
    print(f"  平均盈亏比 (Profit Factor): {df_stats['profit_factor'].mean():.2f}")
    print(f"  平均单笔盈利: {df_stats['avg_win'].mean():.2%}")
    print(f"  平均单笔亏损: {df_stats['avg_loss'].mean():.2%}")
    print(f"  平均交易次数: {df_stats['n_trades'].mean():.1f}")
    
    print("\n📉 胜率分布:")
    print(f"  < 40%: {(df_stats['win_rate'] < 0.4).mean():.1%}")
    print(f"  40% - 50%: {((df_stats['win_rate'] >= 0.4) & (df_stats['win_rate'] < 0.5)).mean():.1%}")
    print(f"  50% - 60%: {((df_stats['win_rate'] >= 0.5) & (df_stats['win_rate'] < 0.6)).mean():.1%}")
    print(f"  > 60%: {(df_stats['win_rate'] >= 0.6).mean():.1%}")
    
    print("\n💡 为什么胜率低？")
    print("  典型的趋势策略通常胜率在 40%-50% 之间。")
    print("  关键在于盈亏比 (Profit Factor)。")
    print("  只要 盈亏比 > 1.5，即使胜率只有 40% 也能盈利。")
    print(f"  当前样本平均盈亏比: {df_stats['profit_factor'].mean():.2f} (健康水平通常 > 1.2)")

    # Correlation
    corr = df_stats['win_rate'].corr(df_stats['sharpe'])
    print(f"\n🎯 胜率与 Sharpe 的相关性: {corr:.3f}")
    
    # Top 5 by Profit Factor
    print("\n🏆 盈亏比最高的 5 个策略:")
    print(df_stats.nlargest(5, 'profit_factor')[['combo', 'win_rate', 'profit_factor', 'sharpe']].to_string(index=False))

if __name__ == "__main__":
    analyze_random_trades(20)
