import backtrader as bt
import pandas as pd
import numpy as np

# Constants - 必须与向量化回测完全一致
FREQ = 8
POS_SIZE = 3
INITIAL_CAPITAL = 1_000_000.0
COMMISSION_RATE = 0.0002
LOOKBACK = 252

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
    """
    与向量化回测 (run_unified_wfo.py) 完全对齐的 Backtrader 策略
    
    核心要点：调仓分为两个阶段，先卖出不在目标集合中的持仓，再用真实到账现金按
    Net-New 逻辑买入新标的。保持已有目标持仓的仓位规模不动，确保与向量化引擎一致。
    """
    params = (
        ('scores', None),      # DataFrame of scores (index=date, columns=tickers)
        ('timing', None),      # Series of timing (index=date)
        ('etf_codes', None),   # List of ETF codes
        ('freq', FREQ),
        ('pos_size', POS_SIZE),
        ('rebalance_schedule', None),  # ✅ 添加调仓日程参数
    )

    def __init__(self):
        self.etf_map = {d._name: d for d in self.datas}
        self.orders = []
        self.trades = []
        self.margin_failures = 0
        self.safety_margin = 1 - 1e-6  # 预留极小浮点空间，避免触发 Margin
        
        # ✅ 预计算调仓日集合
        if self.params.rebalance_schedule is not None:
            self.rebalance_set = set(self.params.rebalance_schedule.tolist())
        else:
            self.rebalance_set = None
        
    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.orders.append({
                'date': self.datas[0].datetime.date(0),
                'ticker': order.data._name,
                'type': 'BUY' if order.isbuy() else 'SELL',
                'price': order.executed.price,
                'size': order.executed.size,
                'value': order.executed.value,
                'comm': order.executed.comm
            })
        elif order.status in [order.Margin, order.Rejected]:
            self.margin_failures += 1

    def notify_trade(self, trade):
        if trade.isclosed:
            self.trades.append({
                'ticker': trade.data._name,
                'entry_date': bt.num2date(trade.dtopen).date(),
                'exit_date': bt.num2date(trade.dtclose).date(),
                'pnl': trade.pnl,
                'pnlcomm': trade.pnlcomm,
                'return_pct': trade.pnl / trade.price_open / abs(trade.size) if trade.size != 0 else 0
            })

    def prenext(self):
        """关键修复：让 prenext 阶段也执行调仓逻辑
        
        Backtrader 在有数据源还没准备好时调用 prenext() 而不是 next()
        如果不实现 prenext()，早期日期会被跳过，导致与 VEC 的数据起点不一致
        """
        self.next()

    def next(self):
        if len(self) < LOOKBACK:
            return

        bar_index = len(self) - 1
        
        # 调仓日
        should_rebalance = False
        if self.rebalance_set is not None:
            should_rebalance = bar_index in self.rebalance_set
        else:
            should_rebalance = (bar_index % self.params.freq == 0)

        if should_rebalance:
            dt = self.datas[0].datetime.date(0)
            dt_ts = pd.Timestamp(dt)
            self.rebalance(dt_ts)

    def _get_current_holdings(self):
        """获取当前实际持仓"""
        holdings = {}
        for d in self.datas:
            pos = self.getposition(d)
            if pos.size > 0:
                holdings[d._name] = pos.size
        return holdings
    
    def rebalance(self, current_date):
        try:
            prev_date = self.datas[0].datetime.date(-1)
            prev_ts = pd.Timestamp(prev_date)
            
            if prev_ts not in self.params.scores.index:
                return

            row = self.params.scores.loc[prev_ts]
            valid = row[row.notna() & (row != 0)]
            
            if len(valid) < self.params.pos_size:
                return
                
            # 按得分从高到低排序，选取 Top K
            top_k = valid.sort_values(ascending=False).head(self.params.pos_size).index.tolist()
            target_set = set(top_k)
            
            timing_ratio = 1.0
            if self.params.timing is not None and current_date in self.params.timing.index:
                timing_ratio = self.params.timing.loc[current_date]
            
            current_holdings = self._get_current_holdings()

            # === 关键修复: 手动估算卖出后的可用资金 ===
            # 先获取当前现金
            cash_after_sells = self.broker.getcash()
            
            # 卖出不在目标集合中的仓位，并预估卖出收入
            kept_holdings_value = 0.0
            for ticker, shares in current_holdings.items():
                data = self.etf_map[ticker]
                price = data.close[0]
                if ticker not in target_set:
                    # 卖出订单（COC 模式下现金即时到账）
                    self.close(data)
                    # 手动预估卖出收入（扣除佣金）
                    cash_after_sells += shares * price * (1 - COMMISSION_RATE)
                else:
                    # 保留的持仓价值
                    kept_holdings_value += shares * price

            # === 计算当前总价值和目标敞口 ===
            current_value = cash_after_sells + kept_holdings_value
            target_exposure = current_value * timing_ratio
            available_for_new = max(0.0, target_exposure - kept_holdings_value)

            new_tickers = [t for t in top_k if t not in current_holdings]
            new_count = len(new_tickers)
            if new_count == 0 or available_for_new <= 0:
                return

            # 佣金提前扣除，确保真实买入成本不会超额
            target_pos_value = available_for_new / new_count / (1 + COMMISSION_RATE)

            buy_orders = []
            total_cost = 0.0
            for ticker in new_tickers:
                data = self.etf_map.get(ticker)
                if data is None:
                    continue
                price = data.close[0]
                if np.isnan(price) or price <= 0:
                    continue

                shares = target_pos_value / price
                if shares <= 0:
                    continue

                cost = shares * price * (1 + COMMISSION_RATE)
                buy_orders.append((data, shares, cost))
                total_cost += cost

            if not buy_orders:
                return

            # 若预算不足则按比例缩放，避免 Margin
            budget = available_for_new * self.safety_margin
            scale = 1.0
            if total_cost > budget and budget > 0:
                scale = budget / total_cost

            for data, shares, _ in buy_orders:
                adj_shares = shares * scale
                if adj_shares <= 0:
                    continue
                self.buy(data, size=adj_shares)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
