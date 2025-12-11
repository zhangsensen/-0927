
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import logging
from datetime import datetime

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from top1_production_validation import Top1Validator
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class PolicyDefenseValidator(Top1Validator):
    def __init__(self):
        super().__init__()
        self.target_factors = ['ADX_14D', 'MAX_DD_60D', 'PRICE_POSITION_120D', 'PV_CORR_20D', 'SHARPE_RATIO_20D']
        
    def prepare_factors_v0(self):
        print("Computing V0 Factors (Original)...")
        lib = PreciseFactorLibrary()
        data_dict = {
            'open': self.ohlcv['open'],
            'high': self.ohlcv['high'],
            'low': self.ohlcv['low'],
            'close': self.ohlcv['close'],
            'volume': self.ohlcv['volume']
        }
        raw_factors_df = lib.compute_all_factors(data_dict)
        
        processor = CrossSectionProcessor(verbose=False)
        raw_factors_dict = {f: raw_factors_df[f] for f in self.target_factors}
        std_factors = processor.process_all_factors(raw_factors_dict)
        
        factors_3d = np.zeros((self.T, self.N, 5))
        for i, f in enumerate(self.target_factors):
            factors_3d[:, :, i] = std_factors[f].values
            
        return factors_3d

    def apply_defense_layer(self, factors_v0):
        print("Applying Defense Layer (V1)...")
        factors_v1 = factors_v0.copy()
        
        open_ = self.ohlcv['open']
        high = self.ohlcv['high']
        low = self.ohlcv['low']
        close = self.ohlcv['close']
        volume = self.ohlcv['volume']
        
        # 1. Hard Filter: Gap Up Fade
        prev_close = close.shift(1)
        gap_up_pct = (open_ - prev_close) / prev_close * 100
        is_fade = close < open_
        mask_gap = (gap_up_pct > 5.0) & is_fade
        
        # 2. Hard Filter: High Volume Stall
        # volume_5d_avg: mean of last 5 days (including today)
        vol_5d_avg = volume.rolling(window=5, min_periods=1).mean()
        vol_ratio = volume / (vol_5d_avg + 1e-10) # Avoid div by zero
        body_pct = (close - open_).abs() / open_ * 100
        mask_stall = (vol_ratio > 3.0) & (body_pct < 2.0)
        
        # Combine Hard Filters
        mask_hard = mask_gap | mask_stall
        
        # 3. Soft Adjustment: CLV
        rng = high - low
        clv = (close - low) / (rng + 1e-10)
        # Fix one-bar (high==low) to 0.5
        clv[rng == 0] = 0.5
        
        multiplier = 0.6 + 0.4 * clv
        
        # Apply to factors
        # Broadcast masks and multiplier to (T, N, 5)
        # mask_hard is DataFrame (T, N), factors_3d is (T, N, 5)
        
        mask_hard_val = mask_hard.values
        multiplier_val = multiplier.values
        
        # Stats
        stats = {
            'gap_fade_count': mask_gap.sum().sum(),
            'stall_count': mask_stall.sum().sum(),
            'total_filtered': mask_hard.sum().sum(),
            'avg_clv': clv.mean().mean()
        }
        
        # Apply
        for i in range(5):
            # Apply Hard Filter (Set to 0)
            factors_v1[:, :, i][mask_hard_val] = -999.0 # Set to very low score instead of 0 to ensure not picked
            # Or just 0? If standardized scores are around 0, 0 might still be picked.
            # The user said "final_score = 0". 
            # If other scores are negative, 0 is good. If others are positive, 0 is bad.
            # Standardized scores usually mean=0, std=1. So 0 is average.
            # To strictly "not buy", we should set it to -infinity.
            # Let's use -10.0 (very bad score).
            
            # Apply Soft Adjustment
            # Only apply to those NOT filtered (though filtered ones get overwritten anyway)
            factors_v1[:, :, i] *= multiplier_val
            
            # Re-apply hard filter to be sure
            factors_v1[:, :, i][mask_hard_val] = -100.0
            
        return factors_v1, stats, mask_hard, clv

    def analyze_specific_dates(self, equity_v0, equity_v1, mask_hard):
        print("\n【关键场景分析】")
        
        # Debug Equity
        print(f"DEBUG: Equity V0 Head: {equity_v0[:5]}")
        print(f"DEBUG: Equity V0 Tail: {equity_v0[-5:]}")
        print(f"DEBUG: Equity V1 Head: {equity_v1[:5]}")
        print(f"DEBUG: Equity V1 Tail: {equity_v1[-5:]}")
        
        # Helper to get return for a specific date
        def get_ret(equity, date_str):
            try:
                idx = self.dates.get_loc(date_str)
                if idx == 0: return 0.0
                return equity[idx] / equity[idx-1] - 1
            except KeyError:
                return np.nan

        # 1. 2024-10-08
        date_str = '2024-10-08'
        if date_str in self.dates:
            idx = self.dates.get_loc(date_str)
            # Check what was filtered on this day
            filtered_tickers = mask_hard.loc[date_str][mask_hard.loc[date_str]].index.tolist()
            print(f"📅 {date_str} (史诗级高开低走):")
            print(f"   V0 当日收益: {get_ret(equity_v0, date_str):.2%}")
            print(f"   V1 当日收益: {get_ret(equity_v1, date_str):.2%}")
            print(f"   触发过滤的ETF ({len(filtered_tickers)}只): {', '.join(filtered_tickers[:5])}...")
            
            # Check specific tickers
            for t in ['512480', '512690']:
                if t in self.tickers:
                    is_filtered = mask_hard.loc[date_str, t]
                    print(f"   {t} 过滤状态: {is_filtered}")
                    # Debug why not filtered
                    if t == '512480':
                        idx_t = self.dates.get_loc(date_str)
                        op = self.ohlcv['open'][t].iloc[idx_t]
                        cl = self.ohlcv['close'][t].iloc[idx_t]
                        prev_cl = self.ohlcv['close'][t].iloc[idx_t-1]
                        gap = (op - prev_cl) / prev_cl * 100
                        print(f"   DEBUG 512480: Open={op}, Close={cl}, PrevClose={prev_cl}, Gap={gap:.2f}%, IsFade={cl<op}")
            
            # Check next few days drawdown
            # Find max drawdown in next 5 days
            try:
                e0_slice = equity_v0[idx:idx+5]
                e1_slice = equity_v1[idx:idx+5]
                dd0 = (e0_slice.min() / e0_slice[0]) - 1
                dd1 = (e1_slice.min() / e1_slice[0]) - 1
                print(f"   随后5日最大回撤: V0 {dd0:.2%} vs V1 {dd1:.2%}")
            except:
                pass
        else:
            print(f"📅 {date_str}: 数据未覆盖")

    def run_comparison(self):
        # 1. Prepare Factors
        factors_v0 = self.prepare_factors_v0()
        factors_v1, stats, mask_hard, clv = self.apply_defense_layer(factors_v0)
        
        # 2. Run Backtests
        print("Running V0 Backtest...")
        factor_indices = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        equity_v0 = self.run_vec_backtest(factors_v0, factor_indices)
        
        print("Running V1 Backtest...")
        equity_v1 = self.run_vec_backtest(factors_v1, factor_indices)
        
        # 3. Metrics
        def calc_metrics(equity):
            # Filter zeros (warmup period)
            valid_equity = equity[equity > 0]
            if len(valid_equity) == 0: return 0, 0, 0
            
            start_val = valid_equity[0]
            end_val = valid_equity[-1]
            
            # Annual Return (CAGR)
            # We use the time span of valid equity
            years = len(valid_equity) / 252
            ret = (end_val / 1_000_000) ** (1/years) - 1
            
            # Max Drawdown
            s = pd.Series(valid_equity)
            dd = (s / s.cummax() - 1).min()
            
            # Sharpe
            daily_ret = s.pct_change().dropna()
            sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252)
            return ret, dd, sharpe
            
        ret0, dd0, sh0 = calc_metrics(equity_v0)
        ret1, dd1, sh1 = calc_metrics(equity_v1)
        
        # 4. Report
        print("\n【回测对比报告】")
        print(f"| 指标 | V0 (原始) | V1 (防御) | 变化 |")
        print(f"|---|---|---|---|")
        print(f"| 年化收益 | {ret0:.2%} | {ret1:.2%} | {ret1-ret0:.2%} |")
        print(f"| 最大回撤 | {dd0:.2%} | {dd1:.2%} | {dd1-dd0:.2%} |")
        print(f"| Sharpe | {sh0:.2f} | {sh1:.2f} | {sh1-sh0:.2f} |")
        
        print("\n【过滤统计】")
        print(f"高开崩塌触发次数: {stats['gap_fade_count']}")
        print(f"天量滞涨触发次数: {stats['stall_count']}")
        print(f"总过滤人次: {stats['total_filtered']}")
        print(f"平均CLV: {stats['avg_clv']:.2f}")
        
        self.analyze_specific_dates(equity_v0, equity_v1, mask_hard)
        
        # 5. Year by Year
        print("\n【分年度收益对比】")
        eq0_s = pd.Series(equity_v0, index=self.dates)
        eq1_s = pd.Series(equity_v1, index=self.dates)
        
        # Replace 0 with NaN for resampling
        eq0_s = eq0_s.replace(0, np.nan).ffill().fillna(1_000_000)
        eq1_s = eq1_s.replace(0, np.nan).ffill().fillna(1_000_000)
        
        y_ret0 = eq0_s.resample('YE').last().pct_change().fillna(0)
        y_ret1 = eq1_s.resample('YE').last().pct_change().fillna(0)
        
        # Fix first year return (relative to 1M)
        first_idx = y_ret0.index[0]
        y_ret0.loc[first_idx] = eq0_s.resample('YE').last().iloc[0] / 1_000_000 - 1
        y_ret1.loc[first_idx] = eq1_s.resample('YE').last().iloc[0] / 1_000_000 - 1
        
        df_y = pd.DataFrame({'V0': y_ret0, 'V1': y_ret1})
        df_y['Diff'] = df_y['V1'] - df_y['V0']
        print(df_y.map(lambda x: f"{x:.2%}").to_markdown())

if __name__ == "__main__":
    validator = PolicyDefenseValidator()
    validator.run_comparison()
