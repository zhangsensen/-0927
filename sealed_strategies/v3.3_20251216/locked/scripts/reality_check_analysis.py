import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import random
from datetime import datetime, timedelta

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from top1_production_validation import Top1Validator
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def get_etf_name(ticker):
    # Simple mapping or placeholder
    names = {
        "510300": "沪深300ETF",
        "510500": "中证500ETF",
        "510050": "上证50ETF",
        "513100": "纳指ETF",
        "513500": "标普500ETF",
        "512880": "证券ETF",
        "512000": "券商ETF",
        "512660": "军工ETF",
        "512010": "医药ETF",
        "512800": "银行ETF",
        "512690": "酒ETF",
        "512480": "半导体ETF",
        "512100": "1000ETF",
        "512070": "非银ETF",
        "515000": "科技ETF",
        "588000": "科创50ETF",
        "159915": "创业板ETF",
        "159949": "创业板50",
        "518880": "黄金ETF",
        "513050": "中概互联",
        "513330": "恒生科技",
        "515030": "新能源车ETF",
        "515790": "光伏ETF",
        "510330": "华夏300",
    }
    return names.get(ticker, ticker)


class RealityCheck(Top1Validator):
    def __init__(self):
        super().__init__()
        self.factors_df = None
        self.total_scores = None
        self.holdings = None  # DataFrame of holdings (1 or 0)
        self.strategy_returns = None

    def prepare_data(self):
        print(
            f"DEBUG: self.T={self.T}, len(rebalance_schedule)={len(self.rebalance_schedule)}"
        )
        print(f"DEBUG: len(dates)={len(self.dates)}")

        print("正在准备数据和因子...")
        # 1. Compute Factors
        lib = PreciseFactorLibrary()
        data_dict = {
            "open": self.ohlcv["open"],
            "high": self.ohlcv["high"],
            "low": self.ohlcv["low"],
            "close": self.ohlcv["close"],
            "volume": self.ohlcv["volume"],
        }
        raw_factors_df = lib.compute_all_factors(data_dict)

        # Target Factors (Must match Top1Validator)
        target_factors = [
            "ADX_14D",
            "MAX_DD_60D",
            "PRICE_POSITION_120D",
            "PV_CORR_20D",
            "SHARPE_RATIO_20D",
        ]

        # Process Factors
        processor = CrossSectionProcessor(verbose=False)
        raw_factors_dict = {f: raw_factors_df[f] for f in target_factors}
        std_factors = processor.process_all_factors(raw_factors_dict)

        # Combine Scores (Equal Weight as per default)
        # Note: In Top1Validator, it creates factors_3d but doesn't explicitly show weights.
        # Usually it's equal weight sum of standardized scores.
        self.factors_df = pd.DataFrame(0.0, index=self.dates, columns=self.tickers)
        for f in target_factors:
            self.factors_df += std_factors[f]

        # Store individual factors for display
        self.raw_factors_dict = raw_factors_dict

        # 2. Simulate Strategy to get Holdings
        # Logic: Every FREQ days, rank by score, pick Top POS_SIZE
        self.holdings = pd.DataFrame(0, index=self.dates, columns=self.tickers)
        self.daily_pnl = pd.Series(0.0, index=self.dates)

        current_holdings = []

        # Market Context Data
        self.hs300 = (
            self.ohlcv["close"]["510300"]
            if "510300" in self.tickers
            else self.ohlcv["close"].iloc[:, 0]
        )
        self.market_vol = self.hs300.pct_change().rolling(20).std() * np.sqrt(252)

        # Convert schedule to set for fast lookup
        rebalance_indices_set = set(self.rebalance_schedule)

        print("正在回放交易记录...")
        for i in range(self.T):
            date = self.dates[i]

            # Calculate PnL from previous day's holdings
            if i > 0 and current_holdings:
                day_ret = 0
                for ticker in current_holdings:
                    ret = (
                        self.ohlcv["close"][ticker].iloc[i]
                        / self.ohlcv["close"][ticker].iloc[i - 1]
                        - 1
                    )
                    day_ret += ret
                self.daily_pnl.iloc[i] = day_ret / len(current_holdings)

                # Mark holdings
                self.holdings.loc[date, current_holdings] = 1

            # Rebalance
            if i in rebalance_indices_set:
                # Get scores for today
                scores = self.factors_df.iloc[i]
                # Filter for valid data
                valid_scores = scores.dropna()
                if not valid_scores.empty:
                    # Sort descending
                    top_picks = (
                        valid_scores.sort_values(ascending=False)
                        .head(self.POS_SIZE)
                        .index.tolist()
                    )
                    current_holdings = top_picks

    def task1_random_replay(self):
        print("\n【任务1：随机抽取5个历史调仓日复盘】")
        rebalance_indices = self.rebalance_schedule
        # Filter indices to ensure we have 3 days of future data
        valid_indices = [i for i in rebalance_indices if i < self.T - 3]

        random.seed(42)
        selected_indices = random.sample(valid_indices, 5)
        selected_indices.sort()

        for idx in selected_indices:
            date = self.dates[idx]
            next_date = self.dates[idx + 1]
            future_date = self.dates[idx + 3]

            print(f"\n📅 调仓日: {date.strftime('%Y-%m-%d')}")

            # 1. Market Context
            hs300_ret = (
                self.hs300.iloc[idx] / self.hs300.iloc[idx - 1] - 1 if idx > 0 else 0
            )
            mkt_vol = self.market_vol.iloc[idx]
            print(f"   市场环境: 沪深300涨跌 {hs300_ret:.2%}, 波动率 {mkt_vol:.2%}")

            # 2. Top 5 Picks
            scores = self.factors_df.iloc[idx].dropna().sort_values(ascending=False)
            top5 = scores.head(5).index.tolist()

            print(f"   策略Top5选股:")
            print(
                f"   {'代码':<8} {'名称':<10} {'总分':<8} {'ADX':<8} {'MaxDD':<8} {'Pos120':<8} {'PVCorr':<8} {'Sharpe':<8}"
            )
            for ticker in top5:
                name = get_etf_name(ticker)
                score = scores[ticker]
                f_vals = [
                    self.raw_factors_dict[f][ticker].iloc[idx]
                    for f in [
                        "ADX_14D",
                        "MAX_DD_60D",
                        "PRICE_POSITION_120D",
                        "PV_CORR_20D",
                        "SHARPE_RATIO_20D",
                    ]
                ]
                # Note: Raw factors might be NaN or different scales. Just showing them.
                print(
                    f"   {ticker:<8} {name:<10} {score:>6.2f} {f_vals[0]:>8.2f} {f_vals[1]:>8.2f} {f_vals[2]:>8.2f} {f_vals[3]:>8.2f} {f_vals[4]:>8.2f}"
                )

            # 3. Future Performance (Next 3 days)
            print(f"   未来3天表现 (持有至 {future_date.strftime('%Y-%m-%d')}):")
            for ticker in top5[: self.POS_SIZE]:  # Only show for actual holdings
                p0 = self.ohlcv["close"][ticker].iloc[idx]
                p3 = self.ohlcv["close"][ticker].iloc[idx + 3]
                ret = p3 / p0 - 1
                print(f"   -> 持仓 {ticker} ({get_etf_name(ticker)}): {ret:.2%}")

            # 4. Execution Check (T+1)
            print(f"   执行检查 (T+1 {next_date.strftime('%Y-%m-%d')}):")
            for ticker in top5[: self.POS_SIZE]:
                vol = self.ohlcv["volume"][ticker].iloc[idx + 1]
                high = self.ohlcv["high"][ticker].iloc[idx + 1]
                low = self.ohlcv["low"][ticker].iloc[idx + 1]
                close = self.ohlcv["close"][ticker].iloc[idx + 1]
                prev_close = self.ohlcv["close"][ticker].iloc[idx]

                issues = []
                if vol == 0:
                    issues.append("停牌")
                if high == low and close > prev_close:
                    issues.append("一字涨停")
                if high == low and close < prev_close:
                    issues.append("一字跌停")
                amount = vol * close
                if amount < 50_000_000:
                    issues.append(f"流动性低({amount/1e6:.1f}M)")

                if issues:
                    print(f"   ⚠️ {ticker}: {', '.join(issues)}")
                else:
                    print(f"   ✅ {ticker}: 执行正常")

    def task2_attribution(self):
        print("\n【任务2：收益归因分析】")
        # 1. By Ticker
        ticker_pnl = {}
        for ticker in self.tickers:
            # Mask returns by holdings
            held_days = self.holdings[ticker] == 1
            if held_days.sum() > 0:
                # Simple sum of daily returns (approx)
                rets = self.ohlcv["close"][ticker].pct_change().fillna(0)
                total_ret = rets[held_days].sum()
                ticker_pnl[ticker] = total_ret

        sorted_pnl = sorted(ticker_pnl.items(), key=lambda x: x[1], reverse=True)
        print("1. 贡献最大的5只ETF:")
        for t, r in sorted_pnl[:5]:
            print(f"   {t} ({get_etf_name(t)}): {r:.2%}")

        # 2. By Year
        print("\n2. 分年度收益:")
        yearly_ret = self.daily_pnl.resample("Y").sum()
        for date, ret in yearly_ret.items():
            print(f"   {date.year}: {ret:.2%}")

        # 3. Trend vs Chop
        # Define Trend: HS300 > MA20. Chop: HS300 < MA20 (Simple proxy)
        ma20 = self.hs300.rolling(20).mean()
        is_trend = self.hs300 > ma20

        trend_ret = self.daily_pnl[is_trend].mean() * 252
        chop_ret = self.daily_pnl[~is_trend].mean() * 252
        print(f"\n3. 市场风格表现 (年化):")
        print(f"   趋势市 (HS300 > MA20): {trend_ret:.2%}")
        print(f"   震荡/熊市 (HS300 < MA20): {chop_ret:.2%}")

        # 4. Crisis Analysis
        print("\n4. 关键时期表现:")
        # 2022 Bear
        ret_2022 = self.daily_pnl["2022"].sum()
        print(f"   2022年熊市: {ret_2022:.2%}")
        # 2025 Chop (Assuming 2025 data exists)
        if "2025" in self.daily_pnl.index.year.astype(str):
            ret_2025 = self.daily_pnl["2025"].sum()
            print(f"   2025年震荡: {ret_2025:.2%}")

    def task3_failures(self):
        print("\n【任务3：策略失效场景分析】")
        # Calculate 3-day rolling return of the strategy
        rolling_3d = self.daily_pnl.rolling(3).sum()

        # Find 5 worst days (end of 3-day period)
        worst_days = rolling_3d.sort_values().head(5).index

        for date in worst_days:
            # Find the rebalance date prior to this
            # This is an approximation, just looking at the context of the loss
            print(
                f"\n📉 失效时刻: {date.strftime('%Y-%m-%d')} (3日亏损 {rolling_3d[date]:.2%})"
            )

            # Market Context
            idx = self.dates.get_loc(date)
            hs300_ret = self.hs300.iloc[idx - 3 : idx].sum()  # Approx 3-day ret
            print(f"   同期沪深300表现: {hs300_ret:.2%}")

            # Holdings
            held = self.holdings.loc[date]
            tickers = held[held == 1].index.tolist()
            print(f"   持仓: {', '.join([f'{t}({get_etf_name(t)})' for t in tickers])}")

            # Why loss?
            for t in tickers:
                r = (
                    self.ohlcv["close"][t].iloc[idx]
                    / self.ohlcv["close"][t].iloc[idx - 3]
                    - 1
                )
                print(f"   -> {t}: {r:.2%}")

    def task4_risks(self):
        print("\n【任务4：实盘执行风险扫描】")

        low_liq_count = 0
        limit_count = 0
        total_trades = 0

        rebalance_indices = self.rebalance_schedule

        for idx in rebalance_indices:
            if idx >= self.T - 1:
                continue

            # Get target holdings
            scores = self.factors_df.iloc[idx].dropna().sort_values(ascending=False)
            targets = scores.head(self.POS_SIZE).index.tolist()

            # Check T+1 execution
            next_idx = idx + 1
            for ticker in targets:
                total_trades += 1
                vol = self.ohlcv["volume"][ticker].iloc[next_idx]
                close = self.ohlcv["close"][ticker].iloc[next_idx]
                high = self.ohlcv["high"][ticker].iloc[next_idx]
                low = self.ohlcv["low"][ticker].iloc[next_idx]
                prev_close = self.ohlcv["close"][ticker].iloc[idx]

                amount = vol * close
                if amount < 50_000_000:
                    low_liq_count += 1

                if vol == 0 or (high == low and abs(close / prev_close - 1) > 0.09):
                    limit_count += 1

        print(f"总计划交易次数: {total_trades}")
        print(
            f"流动性不足 (<5000万) 次数: {low_liq_count} ({low_liq_count/total_trades:.1%})"
        )
        print(
            f"无法执行 (涨跌停/停牌) 次数: {limit_count} ({limit_count/total_trades:.1%})"
        )

    def run(self):
        self.prepare_data()
        self.task1_random_replay()
        self.task2_attribution()
        self.task3_failures()
        self.task4_risks()


if __name__ == "__main__":
    checker = RealityCheck()
    checker.run()
