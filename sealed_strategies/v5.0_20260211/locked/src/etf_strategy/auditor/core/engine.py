import backtrader as bt
import pandas as pd
import numpy as np

from etf_strategy.core.utils.position_sizing import resolve_pos_size_for_day
from etf_strategy.core.hysteresis import apply_hysteresis

# Constants - 必须与向量化回测完全一致
FREQ = 3
POS_SIZE = 2
INITIAL_CAPITAL = 1_000_000.0
COMMISSION_RATE = 0.0002
LOOKBACK = 252


class PandasData(bt.feeds.PandasData):
    params = (
        ("datetime", None),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", None),
    )


class GenericStrategy(bt.Strategy):
    """
    与向量化回测 (run_combo_wfo.py) 完全对齐的 Backtrader 策略

    核心要点：调仓分为两个阶段，先卖出不在目标集合中的持仓，再用真实到账现金按
    Net-New 逻辑买入新标的。保持已有目标持仓的仓位规模不动，确保与向量化引擎一致。
    """

    params = (
        ("scores", None),  # DataFrame of scores (index=date, columns=tickers)
        ("timing", None),  # Series of timing (index=date)
        ("vol_regime", None),  # Series of vol regime multipliers (index=date)
        ("etf_codes", None),  # List of ETF codes
        ("freq", FREQ),
        ("pos_size", POS_SIZE),
        ("lookback", LOOKBACK),  # ✅ P1: Parameterized lookback (default from module constant)
        ("rebalance_schedule", None),  # ✅ 调仓日程参数
        # ✅ P2: 动态降权参数
        ("target_vol", 0.20),
        ("vol_window", 20),
        ("dynamic_leverage_enabled", True),
        # ✅ v4.0: 动态持仓
        ("dynamic_pos_config", None),  # dict from parse_dynamic_pos_config()
        ("gate_series", None),  # Series of gate exposure values (for dynamic pos sizing)
        # ✅ Exp1: T+1 Open 执行模式
        ("use_t1_open", False),
        # ✅ Exp4: 换仓迟滞
        ("delta_rank", 0.0),       # rank01 gap threshold for swap (0 = disabled)
        ("min_hold_days", 0),      # minimum hold days before sell (0 = disabled)
        # ✅ Exp2: per-ticker commission — use max rate for conservative sizing
        ("sizing_commission_rate", COMMISSION_RATE),
    )

    def __init__(self):
        self.etf_map = {d._name: d for d in self.datas}
        self.orders = []
        self.trades = []
        self.margin_failures = 0
        self.safety_margin = 1 - 1e-6  # 与 FullDebugStrategy 对齐，极小安全边际

        # ✅ 追踪每个标的的持仓成本 (用于计算 return_pct)
        self.position_costs = {}  # {ticker: total_cost}
        # ✅ 阴影持仓，避免持仓状态滞后导致卖出缺失
        self.shadow_holdings = {d._name: 0.0 for d in self.datas}

        # ✅ 预计算调仓日集合
        self.use_date_schedule = False
        if self.params.rebalance_schedule is not None:
            schedule = self.params.rebalance_schedule
            if len(schedule) > 0 and (
                isinstance(schedule[0], str) or hasattr(schedule[0], "date")
            ):
                # 如果是日期字符串或对象
                self.rebalance_set = set(pd.to_datetime(d).date() for d in schedule)
                self.use_date_schedule = True
            else:
                # 假设是整数索引
                self.rebalance_set = set(
                    schedule.tolist() if hasattr(schedule, "tolist") else schedule
                )

        # ✅ Exp1: T+1 Open 挂单状态
        self._pending_targets = None  # (top_k, timing_ratio) or None

        # ✅ Exp4: per-ticker hold duration tracking
        self._hold_days = {d._name: 0 for d in self.datas}

        # ✅ Exp4.1: signal-side state for round-robust hysteresis
        # Decouples hysteresis decisions from broker execution state (shadow_holdings)
        self._signal_portfolio = set()
        self._signal_hold_days = {d._name: 0 for d in self.datas}

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
                self.position_costs[ticker] = (
                    self.position_costs.get(ticker, 0) + executed_value
                )

            print(
                f"BT TRADE: {self.datas[0].datetime.date(0)} {ticker} {'BUY' if order.isbuy() else 'SELL'} Price: {order.executed.price:.3f}"
            )

            self.orders.append(
                {
                    "date": self.datas[0].datetime.date(0),
                    "ticker": ticker,
                    "type": "BUY" if order.isbuy() else "SELL",
                    "price": order.executed.price,
                    "size": order.executed.size,
                    "value": executed_value,
                    "comm": order.executed.comm,
                }
            )
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

            self.trades.append(
                {
                    "ticker": ticker,
                    "entry_date": bt.num2date(trade.dtopen).date(),
                    "exit_date": bt.num2date(trade.dtclose).date(),
                    "pnl": trade.pnl,
                    "pnlcomm": trade.pnlcomm,
                    "return_pct": return_pct,
                    "cost": cost,  # 投入成本，便于审计
                }
            )

    def prenext(self):
        """关键修复：让 prenext 阶段也执行调仓逻辑

        Backtrader 在有数据源还没准备好时调用 prenext() 而不是 next()
        如果不实现 prenext()，早期日期会被跳过，导致与 VEC 的数据起点不一致
        """
        self.next()

    def next_open(self):
        """✅ Exp1: T+1 Open — 在 bar t+1 的 open 时刻执行昨日的调仓决策。

        With set_coo(True), orders submitted here fill at current bar's open price.
        This gives correct T+1 Open execution: signal at close(t) → fill at open(t+1).
        """
        if not self.params.use_t1_open:
            return
        if len(self) < self.params.lookback:
            return
        if self._pending_targets is None:
            return

        pend_top_k, pend_timing = self._pending_targets
        self._pending_targets = None
        dt = self.datas[0].datetime.date(0)
        dt_ts = pd.Timestamp(dt)
        self.rebalance(dt_ts, precomputed=(pend_top_k, pend_timing), use_open=True)

    def next(self):
        if len(self) < self.params.lookback:
            return

        # ✅ P2: 计算当日组合净值并更新环形缓冲区
        current_value = self.broker.getvalue()
        if self.prev_portfolio_value is not None and self.prev_portfolio_value > 0:
            daily_return = (
                current_value - self.prev_portfolio_value
            ) / self.prev_portfolio_value
            self.returns_buffer.append(daily_return)
            # 保持缓冲区长度不超过 vol_window
            if len(self.returns_buffer) > self.params.vol_window:
                self.returns_buffer.pop(0)
        self.prev_portfolio_value = current_value

        # ✅ Exp4: increment hold_days for all held positions (every bar)
        for ticker in self._hold_days:
            if self.shadow_holdings.get(ticker, 0.0) > 0:
                self._hold_days[ticker] += 1

        # ✅ Exp4.1: increment signal hold_days for signal portfolio
        for ticker in self._signal_portfolio:
            self._signal_hold_days[ticker] += 1

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
            should_rebalance = bar_index % self.params.freq == 0

        if should_rebalance:
            # ✅ P2: 在调仓日计算动态 leverage
            if (
                self.params.dynamic_leverage_enabled
                and len(self.returns_buffer) >= self.params.vol_window // 2
            ):
                import statistics

                if len(self.returns_buffer) >= 2:
                    daily_std = statistics.stdev(self.returns_buffer)
                    realized_vol = daily_std * (252**0.5)  # 年化
                    if realized_vol > 0.0001:
                        self.current_leverage = min(
                            1.0, self.params.target_vol / realized_vol
                        )
                    else:
                        self.current_leverage = 1.0
                else:
                    self.current_leverage = 1.0
            else:
                self.current_leverage = 1.0

            dt = self.datas[0].datetime.date(0)
            dt_ts = pd.Timestamp(dt)

            if self.params.use_t1_open:
                # T+1 Open: 计算目标并存储，明天执行
                targets = self._compute_rebalance_targets(dt_ts)
                if targets is not None:
                    self._pending_targets = targets
            else:
                self.rebalance(dt_ts)

    def _compute_rebalance_targets(self, current_date):
        """计算调仓目标 (top_k + timing_ratio)，不执行交易。"""
        try:
            prev_date = self.datas[0].datetime.date(-1)
            prev_ts = pd.Timestamp(prev_date)

            if prev_ts not in self.params.scores.index:
                return None

            row = self.params.scores.loc[prev_ts]
            valid = row[row.notna() & (row != 0)]

            effective_pos_size = self.params.pos_size
            dps_cfg = self.params.dynamic_pos_config
            gate_s = self.params.gate_series
            if dps_cfg is not None and dps_cfg.get("enabled", False) and gate_s is not None:
                current_date_ts = pd.Timestamp(self.datas[0].datetime.date(0))
                if current_date_ts in gate_s.index:
                    gate_val = float(gate_s.loc[current_date_ts])
                    effective_pos_size = resolve_pos_size_for_day(dps_cfg, gate_val)

            if len(valid) < effective_pos_size:
                return None

            # ✅ Exp4: apply hysteresis when enabled
            if self.params.delta_rank > 0 or self.params.min_hold_days > 0:
                etf_list = list(self.params.etf_codes)
                N = len(etf_list)

                # Build score array (N,) — invalid = -inf
                score_arr = np.full(N, -np.inf, dtype=np.float64)
                for i, t in enumerate(etf_list):
                    if t in row.index:
                        v = row[t]
                        if pd.notna(v) and v != 0:
                            score_arr[i] = float(v)

                # Exp4.1: Build holdings mask from signal portfolio (not shadow_holdings)
                hmask = np.zeros(N, dtype=np.bool_)
                for t in self._signal_portfolio:
                    idx = etf_list.index(t) if t in etf_list else -1
                    if idx >= 0:
                        hmask[idx] = True

                # Exp4.1: Build hold_days from signal hold_days (not _hold_days)
                hdays = np.zeros(N, dtype=np.int64)
                for t, d in self._signal_hold_days.items():
                    if t in etf_list:
                        hdays[etf_list.index(t)] = d

                # Compute top_indices (descending by score, valid only)
                valid_mask = score_arr > -np.inf
                desc_order = np.argsort(-score_arr)
                top_list = []
                for idx in desc_order:
                    if valid_mask[idx] and len(top_list) < effective_pos_size:
                        top_list.append(idx)
                top_indices = np.array(top_list, dtype=np.int64)

                if len(top_indices) < effective_pos_size:
                    return None

                target_mask = apply_hysteresis(
                    score_arr, hmask, hdays, top_indices,
                    effective_pos_size,
                    float(self.params.delta_rank),
                    int(self.params.min_hold_days),
                )
                top_k = [etf_list[i] for i in range(N) if target_mask[i]]

                # Exp4.1: update signal portfolio from hysteresis decision
                new_signal = set(top_k)
                # Hold_days init: COC=1 (same-bar buy), T+1_OPEN=0 (next-bar buy)
                init_days = 0 if self.params.use_t1_open else 1
                for ticker in new_signal - self._signal_portfolio:
                    self._signal_hold_days[ticker] = init_days
                for ticker in self._signal_portfolio - new_signal:
                    self._signal_hold_days[ticker] = 0
                self._signal_portfolio = new_signal
            else:
                top_k = (
                    valid.sort_values(ascending=False)
                    .head(effective_pos_size)
                    .index.tolist()
                )

            timing_ratio = 1.0
            if (
                self.params.timing is not None
                and current_date in self.params.timing.index
            ):
                timing_ratio = self.params.timing.loc[current_date]
            if (
                self.params.vol_regime is not None
                and current_date in self.params.vol_regime.index
            ):
                timing_ratio = timing_ratio * self.params.vol_regime.loc[current_date]
            timing_ratio = timing_ratio * self.current_leverage

            return (top_k, timing_ratio)
        except Exception:
            import traceback
            traceback.print_exc()
            return None

    def rebalance(self, current_date, precomputed=None, use_open=False):
        """
        与 FullDebugStrategy._rebalance 完全对齐的调仓逻辑。

        Args:
            precomputed: (top_k, timing_ratio) 预计算目标，用于 T+1 Open 执行
            use_open: True 时使用 data.open[0] 定价，False 使用 data.close[0]
        """
        try:
            if precomputed is not None:
                top_k, timing_ratio = precomputed
            else:
                result = self._compute_rebalance_targets(current_date)
                if result is None:
                    return
                top_k, timing_ratio = result

            target_set = set(top_k)

            # 获取当前持仓（使用阴影持仓，避免状态滞后）
            current_holdings = dict(self.shadow_holdings)

            # 第一步：卖出非目标持仓
            kept_holdings_value = 0.0
            cash_after_sells = self.broker.getcash()
            for ticker, shares in list(current_holdings.items()):
                data = self.etf_map[ticker]
                price = data.open[0] if use_open else data.close[0]
                if np.isnan(price) or price <= 0:
                    price = data.close[0]
                if ticker not in target_set and shares > 0:
                    self.close(data)
                    cash_after_sells += shares * price * (1 - self.params.sizing_commission_rate)
                    self.shadow_holdings[ticker] = 0.0
                    self._hold_days[ticker] = 0  # Exp4: reset on sell
                else:
                    kept_holdings_value += shares * price

            # 计算目标敞口（与 VEC 完全一致）
            current_value = cash_after_sells + kept_holdings_value
            target_exposure = current_value * timing_ratio
            available_for_new = target_exposure - kept_holdings_value
            available_for_new = max(0.0, available_for_new)

            # 买入新仓位
            new_tickers = [t for t in top_k if current_holdings.get(t, 0.0) <= 0]
            # Exp4: reset hold_days for new buys (will start counting next bar)
            for ticker in new_tickers:
                self._hold_days[ticker] = 0
            new_count = len(new_tickers)

            if new_count > 0:
                target_pos_value = available_for_new / new_count / (1 + self.params.sizing_commission_rate)

                buy_orders = []
                total_cost = 0.0

                for ticker in new_tickers:
                    data = self.etf_map.get(ticker)
                    if data is None:
                        continue
                    price = data.open[0] if use_open else data.close[0]
                    if np.isnan(price) or price <= 0:
                        price = data.close[0]
                    if np.isnan(price) or price <= 0:
                        continue

                    shares = target_pos_value / price
                    cost = shares * price * (1 + self.params.sizing_commission_rate)
                    buy_orders.append((ticker, data, shares, cost))
                    total_cost += cost

                if total_cost <= available_for_new * self.safety_margin:
                    for ticker, data, shares, cost in buy_orders:
                        self.buy(data, size=shares)
                        self.shadow_holdings[ticker] = (
                            self.shadow_holdings.get(ticker, 0.0) + shares
                        )
                else:
                    scale_factor = (available_for_new * self.safety_margin) / total_cost
                    for ticker, data, shares, cost in buy_orders:
                        adjusted_shares = shares * scale_factor
                        self.buy(data, size=adjusted_shares)
                        self.shadow_holdings[ticker] = (
                            self.shadow_holdings.get(ticker, 0.0) + adjusted_shares
                        )

        except Exception as e:
            import traceback

            traceback.print_exc()
