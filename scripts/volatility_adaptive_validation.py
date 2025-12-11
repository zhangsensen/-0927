
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

class VolatilityAdaptiveValidator(Top1Validator):
    def __init__(self):
        super().__init__()
        self.target_factors = ['ADX_14D', 'MAX_DD_60D', 'PRICE_POSITION_120D', 'PV_CORR_20D', 'SHARPE_RATIO_20D']
        
    def prepare_factors(self):
        print("Computing Factors...")
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

    def calculate_volatility_regime(self):
        print("Calculating Volatility Regime...")
        # 1. Get HS300 (510300)
        if '510300' in self.tickers:
            hs300_close = self.ohlcv['close']['510300']
        else:
            # Fallback to first column if 510300 not found (unlikely)
            print("Warning: 510300 not found, using first ticker as proxy.")
            hs300_close = self.ohlcv['close'].iloc[:, 0]
            
        # 2. Calculate HV
        # HV = std(daily_returns[-20:]) * sqrt(252) * 100
        rets = hs300_close.pct_change()
        hv = rets.rolling(window=20, min_periods=20).std() * np.sqrt(252) * 100
        
        # 3. Smooth: Regime_Vol = (HV_current + HV_5d_ago) / 2
        hv_5d = hv.shift(5)
        regime_vol = (hv + hv_5d) / 2
        
        # 4. Map to Exposure
        exposure = pd.Series(1.0, index=regime_vol.index)
        
        # < 15%: 1.0 (Default)
        # 15-25%: 1.0 (Default)
        
        # 25-30%: 0.7
        mask_yellow = (regime_vol >= 25) & (regime_vol < 30)
        exposure[mask_yellow] = 0.7
        
        # 30-40%: 0.4
        mask_orange = (regime_vol >= 30) & (regime_vol < 40)
        exposure[mask_orange] = 0.4
        
        # > 40%: 0.1
        mask_red = (regime_vol >= 40)
        exposure[mask_red] = 0.1
        
        # Fill NaNs with 1.0 (early period)
        exposure = exposure.fillna(1.0)
        regime_vol = regime_vol.fillna(0.0)
        
        # Create Log DataFrame
        log_df = pd.DataFrame({
            'HS300_Close': hs300_close,
            'HV_Raw': hv,
            'Regime_Vol': regime_vol,
            'Max_Exposure': exposure
        })
        
        return exposure.values, log_df

    def run_adaptive_comparison(self):
        # 1. Prepare Factors
        factors_3d = self.prepare_factors()
        factor_indices = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        
        # 2. Calculate Regime
        vol_exposure, log_df = self.calculate_volatility_regime()
        
        # Save Log
        log_path = ROOT / "results/regime_log.csv"
        log_df.to_csv(log_path)
        print(f"Regime log saved to {log_path}")
        
        # 3. Run V0 (Original)
        print("Running V0 (Original)...")
        # Original timing is usually all 1s or from LightTimingModule
        # self.timing_arr is already computed in __init__
        equity_v0 = self.run_vec_backtest(factors_3d, factor_indices)
        
        # 4. Run V3 (Adaptive)
        print("Running V3 (Adaptive)...")
        # Combine original timing with vol exposure
        # timing_arr is (T,)
        timing_v3 = self.timing_arr * vol_exposure
        
        # We need to inject this timing into the backtest
        # Temporarily swap self.timing_arr
        original_timing = self.timing_arr
        self.timing_arr = timing_v3
        equity_v3 = self.run_vec_backtest(factors_3d, factor_indices)
        self.timing_arr = original_timing # Restore
        
        # 5. Add Cash Return for V3
        # The VEC kernel assumes uninvested cash earns 0.
        # We can approximate the cash return component.
        # Cash weight = 1 - timing_v3
        # Cash return = 2% / 252 daily
        # This is a post-processing adjustment to equity curve
        
        cash_rate_daily = 0.02 / 252
        cash_weights = 1.0 - timing_v3
        
        # Reconstruct daily returns from equity
        # Note: equity_v3 from kernel is the portfolio value assuming 0 return on cash
        # We need to add the cash return.
        # Daily PnL = Equity_t - Equity_{t-1}
        # Adjusted PnL = Daily PnL + (Equity_{t-1} * Cash_Weight_{t-1} * Cash_Rate)
        
        equity_v3_adj = np.zeros_like(equity_v3)
        equity_v3_adj[0] = equity_v3[0] # Initial capital
        
        for t in range(1, len(equity_v3)):
            daily_pnl_strat = equity_v3[t] - equity_v3[t-1]
            cash_interest = equity_v3[t-1] * cash_weights[t-1] * cash_rate_daily
            equity_v3_adj[t] = equity_v3_adj[t-1] + daily_pnl_strat + cash_interest
            
        # 6. Metrics & Report
        self.generate_report(equity_v0, equity_v3_adj, log_df)

    def generate_report(self, eq0, eq3, log_df):
        def calc_metrics(equity):
            valid_equity = equity[equity > 0]
            if len(valid_equity) == 0: return 0, 0, 0, 0
            
            years = len(valid_equity) / 252
            ret = (valid_equity[-1] / 1_000_000) ** (1/years) - 1
            
            s = pd.Series(valid_equity)
            dd = (s / s.cummax() - 1).min()
            
            daily_ret = s.pct_change().dropna()
            sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252)
            
            calmar = ret / abs(dd) if dd != 0 else 0
            return ret, dd, sharpe, calmar

        m0 = calc_metrics(eq0)
        m3 = calc_metrics(eq3)
        
        print("\n【V0 vs V3 对比报告】")
        print(f"| 指标 | V0 (原始) | V3 (自适应) | 变化 |")
        print(f"|---|---|---|---|")
        print(f"| 年化收益 | {m0[0]:.2%} | {m3[0]:.2%} | {m3[0]-m0[0]:.2%} |")
        print(f"| 最大回撤 | {m0[1]:.2%} | {m3[1]:.2%} | {m3[1]-m0[1]:.2%} |")
        print(f"| Sharpe | {m0[2]:.2f} | {m3[2]:.2f} | {m3[2]-m0[2]:.2f} |")
        print(f"| Calmar | {m0[3]:.2f} | {m3[3]:.2f} | {m3[3]-m0[3]:.2f} |")
        
        # Regime Stats
        print("\n【市场状态分布】")
        total_days = len(log_df)
        n_calm = (log_df['Regime_Vol'] < 15).sum()
        n_normal = ((log_df['Regime_Vol'] >= 15) & (log_df['Regime_Vol'] < 25)).sum()
        n_alert = ((log_df['Regime_Vol'] >= 25) & (log_df['Regime_Vol'] < 30)).sum()
        n_high = ((log_df['Regime_Vol'] >= 30) & (log_df['Regime_Vol'] < 40)).sum()
        n_extreme = (log_df['Regime_Vol'] >= 40).sum()
        
        print(f"🟢 平静期 (<15%): {n_calm}天 ({n_calm/total_days:.1%})")
        print(f"🔵 正常期 (15-25%): {n_normal}天 ({n_normal/total_days:.1%})")
        print(f"🟡 轻度警戒 (25-30%): {n_alert}天 ({n_alert/total_days:.1%})")
        print(f"🟠 高度警戒 (30-40%): {n_high}天 ({n_high/total_days:.1%})")
        print(f"🔴 极端危机 (>40%): {n_extreme}天 ({n_extreme/total_days:.1%})")
        
        # Specific Periods Analysis
        print("\n【关键时期回撤对比】")
        dates = self.dates
        eq0_s = pd.Series(eq0, index=dates)
        eq3_s = pd.Series(eq3, index=dates)
        
        def get_period_stats(start, end, name):
            try:
                sub0 = eq0_s[start:end]
                sub3 = eq3_s[start:end]
                if len(sub0) == 0: return
                
                ret0 = sub0.iloc[-1]/sub0.iloc[0] - 1
                ret3 = sub3.iloc[-1]/sub3.iloc[0] - 1
                
                dd0 = (sub0 / sub0.cummax() - 1).min()
                dd3 = (sub3 / sub3.cummax() - 1).min()
                
                print(f"{name} ({start} ~ {end}):")
                print(f"   收益: V0 {ret0:.2%} -> V3 {ret3:.2%}")
                print(f"   回撤: V0 {dd0:.2%} -> V3 {dd3:.2%}")
                
                # Check Vol
                sub_vol = log_df.loc[start:end, 'Regime_Vol']
                max_vol = sub_vol.max()
                min_exp = log_df.loc[start:end, 'Max_Exposure'].min()
                print(f"   期间最高波动率: {max_vol:.2f}%, 最低仓位: {min_exp:.0%}")
            except Exception as e:
                print(f"Error analyzing {name}: {e}")

        get_period_stats('2024-10-08', '2024-10-31', '2024年10月震荡')
        get_period_stats('2022-01-01', '2022-12-31', '2022年熊市')
        get_period_stats('2023-01-01', '2023-12-31', '2023年震荡市')

if __name__ == "__main__":
    validator = VolatilityAdaptiveValidator()
    validator.run_adaptive_comparison()
