#!/usr/bin/env python3
"""
完整对比脚本：逐步模拟 VEC 和 BT 的回测，记录每个调仓日的差异
"""

import sys
from pathlib import Path
import argparse

ROOT = Path(__file__).parent.parent

import yaml
import pandas as pd
import numpy as np
import backtrader as bt

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import shift_timing_signal, generate_rebalance_schedule, ensure_price_views

# 默认参数（可被命令行覆盖）
INITIAL_CAPITAL = 1_000_000.0
COMMISSION_RATE = 0.0002
LOOKBACK = 252

parser = argparse.ArgumentParser(description="Compare vectorized and BT engines for a given combo")
parser.add_argument(
    "--combo",
    type=str,
    default="CORRELATION_TO_MARKET_20D + MAX_DD_60D + RET_VOL_20D",
    help="Factor combo string (e.g. 'ADX_14D + CMF_20D + MAX_DD_60D + RET_VOL_20D')",
)
parser.add_argument("--freq", type=int, default=8, help="Rebalance frequency (default: 8)")
parser.add_argument("--pos", type=int, default=3, help="Position size (default: 3)")
args = parser.parse_args()

FREQ = args.freq
POS_SIZE = args.pos

# 加载配置
config_path = ROOT / "configs/combo_wfo_config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

loader = DataLoader(
    data_dir=config['data'].get('data_dir'),
    cache_dir=config['data'].get('cache_dir'),
)
ohlcv = loader.load_ohlcv(
    etf_codes=config['data']['symbols'],
    start_date=config['data']['start_date'],
    end_date=config['data']['end_date'],
)

factor_lib = PreciseFactorLibrary()
raw_factors_df = factor_lib.compute_all_factors(ohlcv)

factor_names_list = raw_factors_df.columns.get_level_values(0).unique().tolist()
raw_factors = {fname: raw_factors_df[fname] for fname in factor_names_list}

processor = CrossSectionProcessor(verbose=False)
std_factors = processor.process_all_factors(raw_factors)

factor_names = sorted(std_factors.keys())
dates = std_factors[factor_names[0]].index
etf_codes = std_factors[factor_names[0]].columns.tolist()
N = len(etf_codes)
T = len(dates)

timing_module = LightTimingModule()
# 使用共享 helper shift_timing_signal 避免未来函数
timing_series_raw = timing_module.compute_position_ratios(ohlcv['close'])
timing_arr = shift_timing_signal(timing_series_raw.reindex(dates).fillna(1.0).values)
# ✅ 为 BT 策略创建 shifted timing Series
timing_series = pd.Series(timing_arr, index=dates)

# 测试策略
test_factors = [f.strip() for f in args.combo.split('+')]
test_factors = [f for f in test_factors if f]

combined_score_df = pd.DataFrame(0.0, index=dates, columns=etf_codes)
for f in test_factors:
    combined_score_df = combined_score_df.add(std_factors[f], fill_value=0)

factors_3d = np.stack([std_factors[f].values for f in factor_names], axis=-1)
factor_indices = [factor_names.index(f) for f in test_factors]
close_prices = ohlcv['close'][etf_codes].ffill().bfill().values
open_prices = ohlcv['open'][etf_codes].ffill().bfill().values

# ✅ 使用 ensure_price_views 校验价格数据
_, open_validated, close_validated = ensure_price_views(
    close_prices, open_prices,
    validate=True, min_valid_index=LOOKBACK
)

print("="*80)
print(f"策略: {' + '.join(test_factors)}")
print("="*80)
print()

# =============================================================================
# VEC 回测 - 使用共享 helper 生成调仓日程
# =============================================================================
# ✅ 使用共享 helper 生成调仓日程（与 BT 引擎一致）
rebalance_schedule = generate_rebalance_schedule(
    total_periods=T,
    lookback_window=LOOKBACK,
    freq=FREQ,
)
rebalance_set = set(rebalance_schedule.tolist())

vec_cash = INITIAL_CAPITAL
vec_holdings = np.zeros(N)
vec_equity_curve = []
vec_rebalances = []

for t in range(T):
    # Mark to Market
    vec_value = vec_cash
    for n in range(N):
        if vec_holdings[n] > 0:
            vec_value += vec_holdings[n] * close_prices[t, n]
    vec_equity_curve.append(vec_value)
    
    if t < LOOKBACK:
        continue
    
    # ✅ 使用 rebalance_set 判断调仓日（与 batch_vec_backtest.py / run_combo_wfo.py 一致）
    if t in rebalance_set:
        # 计算综合得分 (使用 T-1)
        combined_vec = np.zeros(N)
        for n in range(N):
            score = 0.0
            for f_idx in factor_indices:
                val = factors_3d[t-1, n, f_idx]
                if not np.isnan(val):
                    score += val
            combined_vec[n] = score
        
        valid_mask = (combined_vec != 0) & (~np.isnan(combined_vec))
        valid_count = valid_mask.sum()
        
        if valid_count < POS_SIZE:
            continue
            
        combined_vec[~valid_mask] = -np.inf
        
        sorted_indices = np.argsort(combined_vec)
        target_set = set()
        buy_order = []
        for k in range(POS_SIZE):
            idx = sorted_indices[-(k+1)]
            if combined_vec[idx] > -np.inf:
                target_set.add(idx)
                if vec_holdings[idx] == 0:
                    buy_order.append(idx)
        
        timing_ratio = timing_arr[t]
        
        # 卖出
        for n in range(N):
            if vec_holdings[n] > 0 and n not in target_set:
                price = close_prices[t, n]
                proceeds = vec_holdings[n] * price * (1 - COMMISSION_RATE)
                vec_cash += proceeds
                vec_holdings[n] = 0.0
        
        # 计算当前价值
        current_value = vec_cash
        for n in range(N):
            if vec_holdings[n] > 0:
                current_value += vec_holdings[n] * close_prices[t, n]
        
        # 买入 (Net-New Logic - 与 run_combo_wfo.py 一致)
        if len(buy_order) > 0:
            # 计算保留持仓的市值
            kept_value = 0.0
            for n in range(N):
                if vec_holdings[n] > 0:
                    kept_value += vec_holdings[n] * close_prices[t, n]
            
            # Net-New 逻辑
            target_exposure = current_value * timing_ratio
            available_for_new = target_exposure - kept_value
            available_for_new = max(0.0, available_for_new)
            target_pos_value = available_for_new / len(buy_order) / (1 + COMMISSION_RATE)
            
            for n in buy_order:
                price = close_prices[t, n]
                if price > 0:
                    shares = target_pos_value / price
                    cost = shares * price * (1 + COMMISSION_RATE)
                    if cost <= vec_cash + 1e-5:  # Add tolerance
                        vec_holdings[n] = shares
                        vec_cash -= cost
        
        vec_rebalances.append({
            't': t,
            'date': dates[t],
            'equity': current_value,
            'top_k': [etf_codes[i] for i in target_set],
        })

vec_final_value = vec_cash
for n in range(N):
    if vec_holdings[n] > 0:
        vec_final_value += vec_holdings[n] * close_prices[T-1, n]

vec_return = (vec_final_value / INITIAL_CAPITAL) - 1

print(f"VEC 最终收益: {vec_return:.4%}")
print(f"VEC 调仓次数: {len(vec_rebalances)}")
print()

# =============================================================================
# BT 回测 (完整版)
# =============================================================================
# 准备数据 feeds
data_feeds = {}
for ticker in etf_codes:
    df = pd.DataFrame({
        'open': ohlcv['open'][ticker],
        'high': ohlcv['high'][ticker],
        'low': ohlcv['low'][ticker],
        'close': ohlcv['close'][ticker],
        'volume': ohlcv['volume'][ticker]
    })
    df = df.reindex(dates)
    df = df.ffill().fillna(0.01)
    data_feeds[ticker] = df

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

class FullDebugStrategy(bt.Strategy):
    params = (
        ('scores', None),
        ('timing', None),
        ('etf_codes', None),
        ('freq', FREQ),
        ('pos_size', POS_SIZE),
        ('rebalance_schedule', None),  # ✅ 添加调仓日程参数
    )

    def __init__(self):
        self.etf_map = {d._name: d for d in self.datas}
        self.rebalances = []
        self.order_failures = []
        self.equity_curve = []
        # ✅ 预计算调仓日集合
        if self.params.rebalance_schedule is not None:
            self.rebalance_set = set(self.params.rebalance_schedule.tolist())
        else:
            self.rebalance_set = None

    def notify_order(self, order):
        if order.status in [order.Margin, order.Rejected]:
            self.order_failures.append({
                'date': self.datas[0].datetime.date(0),
                'ticker': order.data._name,
                'status': 'Margin' if order.status == order.Margin else 'Rejected',
            })
        
    def next(self):
        self.equity_curve.append({
            'date': pd.Timestamp(self.datas[0].datetime.date(0)),
            'equity': self.broker.getvalue()
        })

        if len(self) < LOOKBACK:
            return

        bar_index = len(self) - 1
        
        # ✅ 使用 rebalance_set 判断调仓日（与 VEC 完全一致）
        should_rebalance = False
        if self.rebalance_set is not None:
            should_rebalance = bar_index in self.rebalance_set
        else:
            # fallback: 旧逻辑
            should_rebalance = (bar_index % self.params.freq == 0)
        
        if should_rebalance:
            dt = self.datas[0].datetime.date(0)
            dt_ts = pd.Timestamp(dt)
            self._rebalance(dt_ts)

    def _rebalance(self, current_date):
        try:
            prev_date = self.datas[0].datetime.date(-1)
            prev_ts = pd.Timestamp(prev_date)
            
            if prev_ts not in self.params.scores.index:
                return

            row = self.params.scores.loc[prev_ts]
            valid = row[row.notna() & (row != 0)]
            
            if len(valid) < self.params.pos_size:
                return
                
            top_k = valid.sort_values(ascending=False).head(self.params.pos_size).index.tolist()
            target_set = set(top_k)
            
            timing_ratio = 1.0
            if self.params.timing is not None and current_date in self.params.timing.index:
                timing_ratio = self.params.timing.loc[current_date]
            
            # 获取当前持仓
            current_holdings = {}
            for d in self.datas:
                pos = self.getposition(d)
                if pos.size > 0:
                    current_holdings[d._name] = pos.size
            
            # 第一步：卖出非目标持仓（在 COC 模式下立即释放现金）
            kept_holdings_value = 0.0
            cash_after_sells = self.broker.getcash()
            for ticker, shares in current_holdings.items():
                data = self.etf_map[ticker]
                price = data.close[0]
                if ticker not in target_set:
                    # 卖出会立即释放现金（COC 模式）
                    self.close(data)
                    cash_after_sells += shares * price * (1 - COMMISSION_RATE)
                else:
                    # 保留的持仓价值
                    kept_holdings_value += shares * price
            
            # 第三步：计算目标敞口（与 VEC 完全一致）
            current_value = cash_after_sells + kept_holdings_value
            target_exposure = current_value * timing_ratio
            available_for_new = target_exposure - kept_holdings_value
            available_for_new = max(0.0, available_for_new)
            
            # 买入新仓位
            new_tickers = [t for t in top_k if t not in current_holdings]
            new_count = len(new_tickers)
            
            if new_count > 0:
                # 计算每个新仓位的目标金额
                target_pos_value = available_for_new / new_count / (1 + COMMISSION_RATE)
                
                # 批量提交买入订单（避免逐个提交导致的现金不足）
                buy_orders = []
                total_cost = 0.0
                
                for ticker in new_tickers:
                    data = self.etf_map[ticker]
                    price = data.close[0]
                    if np.isnan(price) or price <= 0:
                        continue
                    
                    # 计算目标股数和成本
                    shares = target_pos_value / price
                    cost = shares * price * (1 + COMMISSION_RATE)
                    buy_orders.append((ticker, data, shares, cost))
                    total_cost += cost
                
                # 检查总成本是否超过可用资金（使用极小安全边际，避免浮点误差）
                safety_margin = 1 - 1e-6
                if total_cost <= available_for_new * safety_margin:
                    # 资金充足，提交所有订单
                    for ticker, data, shares, cost in buy_orders:
                        self.buy(data, size=shares)
                else:
                    # 资金不足，按比例缩减
                    scale_factor = (available_for_new * safety_margin) / total_cost
                    for ticker, data, shares, cost in buy_orders:
                        adjusted_shares = shares * scale_factor
                        self.buy(data, size=adjusted_shares)
            
            # 记录调仓信息
            current_equity = self.broker.getvalue()
            self.rebalances.append({
                'date': current_date,
                'equity': current_equity,
                'top_k': top_k,
            })
                
        except Exception as e:
            import traceback
            traceback.print_exc()

cerebro = bt.Cerebro()
cerebro.broker.setcash(INITIAL_CAPITAL)
# 不使用杠杆，确保与 VEC 对齐（leverage=1.0 表示无杠杆）
cerebro.broker.setcommission(commission=COMMISSION_RATE, leverage=1.0)
cerebro.broker.set_coc(True)
# 关键：禁用订单提交时的现金检查，允许基于 COC 的资金即时到账
cerebro.broker.set_checksubmit(False)

for ticker, df in data_feeds.items():
    data = PandasData(dataname=df, name=ticker)
    cerebro.adddata(data)

cerebro.addstrategy(FullDebugStrategy, 
                    scores=combined_score_df, 
                    timing=timing_series,
                    etf_codes=etf_codes,
                    rebalance_schedule=rebalance_schedule)  # ✅ 传入调仓日程

start_val = cerebro.broker.getvalue()
results = cerebro.run()
end_val = cerebro.broker.getvalue()
strat = results[0]

bt_return = (end_val / start_val) - 1

print(f"BT 最终收益: {bt_return:.4%}")
print(f"BT 调仓次数: {len(strat.rebalances)}")
print(f"BT 订单失败次数: {len(strat.order_failures)}")
print()

# =============================================================================
# 对比分析
# =============================================================================
print("="*80)
print("差异分析")
print("="*80)
print(f"VEC 收益: {vec_return:.4%}")
print(f"BT 收益:  {bt_return:.4%}")
print(f"差异:     {(bt_return - vec_return)*100:+.2f} pp")
print()

# 找到差异开始的位置
print("【净值曲线对比】")
if len(strat.rebalances) > 0 and len(vec_rebalances) > 0:
    print(f"{'日期':<12} {'VEC净值':>15} {'BT净值':>15} {'差异':>10}")
    print("-" * 55)
    
    for i in range(min(20, len(strat.rebalances))):
        vec_r = vec_rebalances[i]
        bt_r = strat.rebalances[i]
        diff = bt_r['equity'] - vec_r['equity']
        diff_pct = diff / vec_r['equity'] * 100
        
        print(f"{str(vec_r['date'].date()):<12} {vec_r['equity']:>15,.0f} {bt_r['equity']:>15,.0f} {diff_pct:>+9.2f}%")
        
        # 检查选股差异
        if set(vec_r['top_k']) != set(bt_r['top_k']):
            print(f"  ⚠️ 选股不同: VEC={vec_r['top_k']}, BT={bt_r['top_k']}")

print()
print("【订单失败记录】")
if len(strat.order_failures) > 0:
    print(f"共 {len(strat.order_failures)} 次订单失败")
    for f in strat.order_failures[:10]:
        print(f"  {f['date']}: {f['ticker']} - {f['status']}")
else:
    print("无订单失败")

# Save comparison data
vec_df = pd.DataFrame({'equity': vec_equity_curve}, index=dates)
bt_df = pd.DataFrame(strat.equity_curve).set_index('date')
bt_df = bt_df[~bt_df.index.duplicated(keep='last')]

comparison = pd.DataFrame(index=vec_df.index)
comparison['vec_equity'] = vec_df['equity']
comparison['bt_equity'] = bt_df['equity']
comparison['diff'] = comparison['bt_equity'] - comparison['vec_equity']
comparison['diff_pct'] = comparison['diff'] / comparison['vec_equity']

comparison.to_csv('comparison_equity.csv')
print("\nSaved comparison_equity.csv")

# Find first divergence > 0.1%
divergence = comparison[comparison['diff_pct'].abs() > 0.001]
if not divergence.empty:
    print("\nFirst Divergence (>0.1%):")
    print(divergence.head())
