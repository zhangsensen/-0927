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
    与向量化回测 (run_combo_wfo.py) 完全对齐的 Backtrader 策略
    
    核心要点：调仓分为两个阶段，先卖出不在目标集合中的持仓，再用真实到账现金按
    Net-New 逻辑买入新标的。保持已有目标持仓的仓位规模不动，确保与向量化引擎一致。
    """
    params = (
        ('scores', None),      # DataFrame of scores (index=date, columns=tickers)
        ('timing', None),      # Series of timing (index=date)
        ('etf_codes', None),   # List of ETF codes
        ('freq', FREQ),
        ('pos_size', POS_SIZE),
        ('rebalance_schedule', None),  # ✅ 调仓日程参数
        # ✅ P2: 动态降权参数
        ('target_vol', 0.20),
        ('vol_window', 20),
        ('dynamic_leverage_enabled', True),
    )

    def __init__(self):
        self.etf_map = {d._name: d for d in self.datas}
        self.orders = []
        self.trades = []
        self.margin_failures = 0
        self.safety_margin = 1 - 1e-6  # 与 FullDebugStrategy 对齐，极小安全边际
        
        # ✅ 追踪每个标的的持仓成本 (用于计算 return_pct)
        self.position_costs = {}  # {ticker: total_cost}
        
        # ✅ 预计算调仓日集合
        self.use_date_schedule = False
        if self.params.rebalance_schedule is not None:
            schedule = self.params.rebalance_schedule
            if len(schedule) > 0 and (isinstance(schedule[0], str) or hasattr(schedule[0], 'date')):
                # 如果是日期字符串或对象
                self.rebalance_set = set(pd.to_datetime(d).date() for d in schedule)
                self.use_date_schedule = True
            else:
                # 假设是整数索引
                self.rebalance_set = set(schedule.tolist() if hasattr(schedule, 'tolist') else schedule)

        
        # ✅ P2: 动态降权 - 环形缓冲区存储日收益率
        self.returns_buffer = []
        self.prev_portfolio_value = None
        self.current_leverage = 1.0
        
    def notify_order(self, order):
        if order.status in [order.Completed]:
            ticker = order.data._name
            executed_value = abs(order.executed.value)
            
            # 追踪持仓成本
            if order.isbuy():
                self.position_costs[ticker] = self.position_costs.get(ticker, 0) + executed_value
            
            print(f"BT TRADE: {self.datas[0].datetime.date(0)} {ticker} {'BUY' if order.isbuy() else 'SELL'} Price: {order.executed.price:.3f}")

            self.orders.append({
                'date': self.datas[0].datetime.date(0),
                'ticker': ticker,
                'type': 'BUY' if order.isbuy() else 'SELL',
                'price': order.executed.price,
                'size': order.executed.size,
                'value': executed_value,
                'comm': order.executed.comm
            })
        elif order.status in [order.Margin, order.Rejected]:
            self.margin_failures += 1

    def notify_trade(self, trade):
        if trade.isclosed:
            ticker = trade.data._name
            
            # 从追踪的持仓成本中获取
            cost = self.position_costs.pop(ticker, 0)
            if cost == 0:
                cost = 1  # fallback
            
            return_pct = trade.pnlcomm / cost if cost > 0 else 0
            
            self.trades.append({
                'ticker': ticker,
                'entry_date': bt.num2date(trade.dtopen).date(),
                'exit_date': bt.num2date(trade.dtclose).date(),
                'pnl': trade.pnl,
                'pnlcomm': trade.pnlcomm,
                'return_pct': return_pct,
                'cost': cost,  # 投入成本，便于审计
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
        
        # ✅ P2: 计算当日组合净值并更新环形缓冲区
        current_value = self.broker.getvalue()
        if self.prev_portfolio_value is not None and self.prev_portfolio_value > 0:
            daily_return = (current_value - self.prev_portfolio_value) / self.prev_portfolio_value
            self.returns_buffer.append(daily_return)
            # 保持缓冲区长度不超过 vol_window
            if len(self.returns_buffer) > self.params.vol_window:
                self.returns_buffer.pop(0)
        self.prev_portfolio_value = current_value

        bar_index = len(self) - 1
        
        # 调仓日
        should_rebalance = False
        if self.rebalance_set is not None:
            if self.use_date_schedule:
                current_date = self.datas[0].datetime.date(0)
                should_rebalance = current_date in self.rebalance_set
            else:
                should_rebalance = bar_index in self.rebalance_set
        else:
            should_rebalance = (bar_index % self.params.freq == 0)

        if should_rebalance:
            # ✅ P2: 在调仓日计算动态 leverage
            if self.params.dynamic_leverage_enabled and len(self.returns_buffer) >= self.params.vol_window // 2:
                import statistics
                if len(self.returns_buffer) >= 2:
                    daily_std = statistics.stdev(self.returns_buffer)
                    realized_vol = daily_std * (252 ** 0.5)  # 年化
                    if realized_vol > 0.0001:
                        self.current_leverage = min(1.0, self.params.target_vol / realized_vol)
                    else:
                        self.current_leverage = 1.0
                else:
                    self.current_leverage = 1.0
            else:
                self.current_leverage = 1.0
            
            dt = self.datas[0].datetime.date(0)
            dt_ts = pd.Timestamp(dt)
            self.rebalance(dt_ts)
    
    def rebalance(self, current_date):
        """
        与 FullDebugStrategy._rebalance 完全对齐的调仓逻辑
        移除 Risk-Off 资产处理，保持与 VEC 一致
        """
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
            
            # ✅ P2: 应用动态 leverage
            timing_ratio = timing_ratio * self.current_leverage
            
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
            
            # 计算目标敞口（与 VEC 完全一致）
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
                    data = self.etf_map.get(ticker)
                    if data is None:
                        continue
                    price = data.close[0]
                    if np.isnan(price) or price <= 0:
                        continue
                    
                    # 计算目标股数和成本
                    shares = target_pos_value / price
                    cost = shares * price * (1 + COMMISSION_RATE)
                    buy_orders.append((ticker, data, shares, cost))
                    total_cost += cost
                
                # 检查总成本是否超过可用资金（使用极小安全边际，避免浮点误差）
                if total_cost <= available_for_new * self.safety_margin:
                    # 资金充足，提交所有订单
                    for ticker, data, shares, cost in buy_orders:
                        self.buy(data, size=shares)
                else:
                    # 资金不足，按比例缩减
                    scale_factor = (available_for_new * self.safety_margin) / total_cost
                    for ticker, data, shares, cost in buy_orders:
                        adjusted_shares = shares * scale_factor
                        self.buy(data, size=adjusted_shares)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
