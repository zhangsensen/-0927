
import backtrader as bt
import pandas as pd
import numpy as np

# Constants
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
    params = (
        ('scores', None),      # DataFrame of scores (index=date, columns=tickers)
        ('timing', None),      # Series of timing (index=date)
        ('etf_codes', None),   # List of ETF codes
        ('freq', FREQ),
        ('pos_size', POS_SIZE),
        ('debug', False),      # Enable debug logging
    )

    def __init__(self):
        self.inds = {}
        self.etf_map = {d._name: d for d in self.datas}
        self.orders = []
        self.trades = []
        
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
                if self.params.debug:
                    print(f"DEBUG: {current_date} - Prev date {prev_date} not in scores index")
                return

            row = self.params.scores.loc[prev_ts]
            valid = row[row.notna() & (row != 0)]
            
            target_weights = {}
            if len(valid) >= self.params.pos_size:
                # 按得分从高到低排序，选取 Top K
                top_k = valid.sort_values(ascending=False).head(self.params.pos_size).index.tolist()
                
                if self.params.debug:
                    print(f"DEBUG: {current_date} (Signal: {prev_date}) - Top {self.params.pos_size}: {top_k}")
                    for t in top_k:
                        print(f"   {t}: {valid[t]:.4f}")
                
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
            if self.params.debug:
                print(f"ERROR in rebalance: {e}")
            pass
