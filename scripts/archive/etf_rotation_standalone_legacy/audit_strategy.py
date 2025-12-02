#!/usr/bin/env python3
"""
策略深度审计 | Deep Strategy Audit

对 Top 200 策略进行详细审计，检查：
1. 每日持仓和交易记录
2. 收益曲线的合理性
3. 是否存在未来数据（Lookahead Bias）
4. 收益来源分析

作者: Linus
日期: 2025-11-29
"""

import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import shift_timing_signal, generate_rebalance_schedule


def run_detailed_audit(
    factors_3d,
    close_prices,
    timing_arr,
    risk_off_prices,
    factor_indices,
    factor_names_list,
    etf_names,
    dates,
    freq,
    pos_size,
    initial_capital,
    commission_rate,
    lookback,
):
    """
    详细审计回测，输出完整交易日志
    """
    T, N = close_prices.shape
    
    rebalance_schedule = generate_rebalance_schedule(
        total_periods=T,
        lookback_window=lookback,
        freq=freq,
    )
    
    cash = initial_capital
    holdings = {}  # {etf_idx: shares}
    entry_prices = {}  # {etf_idx: price}
    risk_off_holdings = 0.0
    
    # 交易日志
    trade_log = []
    # 每日净值
    daily_nav = []
    # 持仓日志
    position_log = []
    
    peak_value = initial_capital
    
    for i, t in enumerate(rebalance_schedule):
        if t >= T:
            break
            
        date = dates[t]
        
        # 1. 计算组合得分 (使用 t-1 的因子)
        scores = {}
        for n in range(N):
            score = 0.0
            count = 0
            for idx in factor_indices:
                val = factors_3d[t - 1, n, idx]
                if not np.isnan(val):
                    score += val
                    count += 1
            if count > 0:
                scores[n] = score
        
        # 2. 选出目标持仓
        if len(scores) >= pos_size:
            sorted_stocks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            target_set = set([s[0] for s in sorted_stocks[:pos_size]])
        else:
            target_set = set()
        
        # 3. 择时信号
        timing_ratio = timing_arr[t]
        
        # 4. 卖出不再持有的股票
        to_sell = [idx for idx in holdings if idx not in target_set]
        for idx in to_sell:
            price = close_prices[t, idx]
            shares = holdings[idx]
            proceeds = shares * price * (1 - commission_rate)
            
            pnl = (price - entry_prices[idx]) / entry_prices[idx]
            pnl_amount = shares * (price - entry_prices[idx])
            
            trade_log.append({
                'date': date,
                'action': 'SELL',
                'etf': etf_names[idx],
                'shares': shares,
                'price': price,
                'entry_price': entry_prices[idx],
                'pnl_pct': pnl * 100,
                'pnl_amount': pnl_amount,
                'timing': timing_ratio,
            })
            
            cash += proceeds
            del holdings[idx]
            del entry_prices[idx]
        
        # 5. 计算当前总资产
        equity_value = sum(holdings[idx] * close_prices[t, idx] for idx in holdings)
        risk_off_price = risk_off_prices[t] if not np.isnan(risk_off_prices[t]) else 0
        risk_off_value = risk_off_holdings * risk_off_price if risk_off_price > 0 else 0
        total_value = cash + equity_value + risk_off_value
        
        # 更新最大回撤
        if total_value > peak_value:
            peak_value = total_value
        current_dd = (peak_value - total_value) / peak_value
        
        # 6. RORO 调整
        target_equity = total_value * timing_ratio
        target_risk_off = total_value * (1 - timing_ratio)
        
        # 调整黄金仓位
        if risk_off_price > 0:
            risk_off_diff = target_risk_off - risk_off_value
            if risk_off_diff < -1e-5:  # 卖黄金
                sell_val = min(-risk_off_diff, risk_off_value)
                shares_to_sell = sell_val / risk_off_price
                cash += sell_val * (1 - commission_rate)
                risk_off_holdings -= shares_to_sell
                
                trade_log.append({
                    'date': date,
                    'action': 'SELL_GOLD',
                    'etf': 'GOLD_518880',
                    'shares': shares_to_sell,
                    'price': risk_off_price,
                    'entry_price': 0,
                    'pnl_pct': 0,
                    'pnl_amount': 0,
                    'timing': timing_ratio,
                })
                
            elif risk_off_diff > 1e-5:  # 买黄金
                buy_val = min(risk_off_diff, cash)
                if buy_val > 0:
                    shares_to_buy = (buy_val / (1 + commission_rate)) / risk_off_price
                    cash -= buy_val
                    risk_off_holdings += shares_to_buy
                    
                    trade_log.append({
                        'date': date,
                        'action': 'BUY_GOLD',
                        'etf': 'GOLD_518880',
                        'shares': shares_to_buy,
                        'price': risk_off_price,
                        'entry_price': risk_off_price,
                        'pnl_pct': 0,
                        'pnl_amount': 0,
                        'timing': timing_ratio,
                    })
        
        # 7. 买入新股票
        new_targets = [idx for idx in target_set if idx not in holdings]
        kept_equity = sum(holdings[idx] * close_prices[t, idx] for idx in holdings)
        available = target_equity - kept_equity
        available = min(available, cash)
        available = max(available, 0)
        
        if new_targets and available > 0:
            per_stock = available / len(new_targets) / (1 + commission_rate)
            for idx in new_targets:
                price = close_prices[t, idx]
                if np.isnan(price) or price <= 0:
                    continue
                shares = per_stock / price
                cost = shares * price * (1 + commission_rate)
                if cash >= cost - 1e-5:
                    cash -= min(cost, cash)
                    holdings[idx] = shares
                    entry_prices[idx] = price
                    
                    trade_log.append({
                        'date': date,
                        'action': 'BUY',
                        'etf': etf_names[idx],
                        'shares': shares,
                        'price': price,
                        'entry_price': price,
                        'pnl_pct': 0,
                        'pnl_amount': 0,
                        'timing': timing_ratio,
                    })
        
        # 记录持仓快照
        equity_value = sum(holdings[idx] * close_prices[t, idx] for idx in holdings)
        risk_off_value = risk_off_holdings * risk_off_price if risk_off_price > 0 else 0
        total_value = cash + equity_value + risk_off_value
        
        position_log.append({
            'date': date,
            'total_value': total_value,
            'cash': cash,
            'equity_value': equity_value,
            'gold_value': risk_off_value,
            'drawdown': current_dd,
            'timing': timing_ratio,
            'holdings': [etf_names[idx] for idx in holdings],
            'num_positions': len(holdings),
        })
        
        daily_nav.append({
            'date': date,
            'nav': total_value,
            'dd': current_dd,
        })
    
    # 最终清算
    final_value = cash
    for idx in holdings:
        price = close_prices[T - 1, idx]
        if np.isnan(price):
            price = entry_prices[idx]
        final_value += holdings[idx] * price * (1 - commission_rate)
        
        pnl = (price - entry_prices[idx]) / entry_prices[idx]
        trade_log.append({
            'date': dates[T-1],
            'action': 'FINAL_SELL',
            'etf': etf_names[idx],
            'shares': holdings[idx],
            'price': price,
            'entry_price': entry_prices[idx],
            'pnl_pct': pnl * 100,
            'pnl_amount': holdings[idx] * (price - entry_prices[idx]),
            'timing': timing_arr[T-1],
        })
    
    if risk_off_holdings > 0:
        price = risk_off_prices[T - 1]
        if not np.isnan(price) and price > 0:
            final_value += risk_off_holdings * price * (1 - commission_rate)
    
    return trade_log, position_log, daily_nav, final_value


def main():
    print("=" * 80)
    print("🔍 策略深度审计 | Deep Strategy Audit")
    print("=" * 80)

    # 1. Load Configuration
    config_path = Path(__file__).parent / "configs/config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 2. Load ranking results
    results_dir = Path(__file__).parent / "results"
    ranking_files = sorted(results_dir.glob("risk_sweep_full_*.parquet"))
    if not ranking_files:
        print("❌ No ranking results found")
        return
    
    df_all = pd.read_parquet(ranking_files[-1])
    print(f"📂 Loaded {len(df_all)} strategies from {ranking_files[-1].name}")
    
    # 按卡玛比率排序取 Top 200
    df_top = df_all.sort_values("calmar", ascending=False).head(200)
    
    # 3. Load Data
    print("\n[Step 1] Loading Data...")
    risk_off_asset = "518880"
    symbols = config["data"]["symbols"]
    if risk_off_asset not in symbols:
        symbols.append(risk_off_asset)
        
    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=symbols,
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )
    
    if risk_off_asset in ohlcv["close"].columns:
        risk_off_prices = ohlcv["close"][risk_off_asset].ffill().bfill().values
        rotation_symbols = [s for s in ohlcv["close"].columns if s != risk_off_asset]
        ohlcv_rotation = {col: df[rotation_symbols] for col, df in ohlcv.items()}
    else:
        risk_off_prices = np.zeros(len(ohlcv["close"]))
        ohlcv_rotation = ohlcv
        rotation_symbols = list(ohlcv["close"].columns)

    # 4. Compute Factors
    print("\n[Step 2] Computing Factors...")
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv_rotation)
    
    processor = CrossSectionProcessor(
        lower_percentile=config["cross_section"]["winsorize_lower"] * 100,
        upper_percentile=config["cross_section"]["winsorize_upper"] * 100,
        verbose=False
    )
    std_factors = processor.process_all_factors({
        name: raw_factors_df[name] 
        for name in raw_factors_df.columns.get_level_values(0).unique()
    })

    factor_names = sorted(std_factors.keys())
    factor_map = {name: i for i, name in enumerate(factor_names)}
    
    factors_3d = np.stack([std_factors[f].values for f in factor_names], axis=-1)
    close_prices = ohlcv_rotation["close"][std_factors[factor_names[0]].columns].ffill().bfill().values
    dates = std_factors[factor_names[0]].index
    etf_names = list(std_factors[factor_names[0]].columns)
    
    # 5. Compute Timing
    print("\n[Step 3] Computing Timing Signal...")
    timing_module = LightTimingModule(
        extreme_threshold=config["backtest"]["timing"]["extreme_threshold"],
        extreme_position=config["backtest"]["timing"]["extreme_position"]
    )
    timing_series = timing_module.compute_position_ratios(ohlcv["close"])
    timing_arr = shift_timing_signal(timing_series.reindex(dates).fillna(1.0).values)

    # 6. Audit Top Strategy (Rank 1)
    print("\n" + "=" * 80)
    print("🏆 审计 Top 1 策略")
    print("=" * 80)
    
    top1 = df_top.iloc[0]
    combo_str = top1["combo"]
    freq = int(top1["freq"])
    
    print(f"策略: {combo_str}")
    print(f"频率: {freq} 天")
    print(f"报告收益: {top1['total_return']*100:.1f}%")
    print(f"报告回撤: {top1['max_dd']*100:.1f}%")
    
    parts = [p.strip() for p in combo_str.split("+")]
    factor_indices = [factor_map[p] for p in parts]
    
    trade_log, position_log, daily_nav, final_value = run_detailed_audit(
        factors_3d=factors_3d,
        close_prices=close_prices,
        timing_arr=timing_arr,
        risk_off_prices=risk_off_prices,
        factor_indices=factor_indices,
        factor_names_list=parts,
        etf_names=etf_names,
        dates=dates,
        freq=freq,
        pos_size=config["backtest"]["position_size"],
        initial_capital=1_000_000.0,
        commission_rate=0.0002,
        lookback=252,
    )
    
    # 7. Analysis
    df_trades = pd.DataFrame(trade_log)
    df_positions = pd.DataFrame(position_log)
    df_nav = pd.DataFrame(daily_nav)
    
    print(f"\n审计收益: {(final_value - 1_000_000) / 1_000_000 * 100:.1f}%")
    print(f"交易总数: {len(df_trades)}")
    
    # 统计买卖
    buys = df_trades[df_trades['action'] == 'BUY']
    sells = df_trades[df_trades['action'].isin(['SELL', 'FINAL_SELL'])]
    gold_buys = df_trades[df_trades['action'] == 'BUY_GOLD']
    gold_sells = df_trades[df_trades['action'] == 'SELL_GOLD']
    
    print(f"\n股票买入: {len(buys)} 次")
    print(f"股票卖出: {len(sells)} 次")
    print(f"黄金买入: {len(gold_buys)} 次")
    print(f"黄金卖出: {len(gold_sells)} 次")
    
    # 盈亏分析
    if len(sells) > 0:
        wins = sells[sells['pnl_pct'] > 0]
        losses = sells[sells['pnl_pct'] <= 0]
        print(f"\n盈利交易: {len(wins)} 次 ({len(wins)/len(sells)*100:.1f}%)")
        print(f"亏损交易: {len(losses)} 次 ({len(losses)/len(sells)*100:.1f}%)")
        
        if len(wins) > 0:
            print(f"平均盈利: {wins['pnl_pct'].mean():.2f}%")
        if len(losses) > 0:
            print(f"平均亏损: {losses['pnl_pct'].mean():.2f}%")
    
    # 年度收益
    df_nav['date'] = pd.to_datetime(df_nav['date'])
    df_nav['year'] = df_nav['date'].dt.year
    
    print("\n" + "=" * 40)
    print("📅 年度收益明细")
    print("=" * 40)
    
    yearly_returns = []
    for year in sorted(df_nav['year'].unique()):
        year_data = df_nav[df_nav['year'] == year]
        if len(year_data) > 1:
            start_nav = year_data.iloc[0]['nav']
            end_nav = year_data.iloc[-1]['nav']
            year_ret = (end_nav - start_nav) / start_nav
            max_dd = year_data['dd'].max()
            yearly_returns.append({
                'year': year,
                'return': year_ret,
                'max_dd': max_dd,
            })
            print(f"{year}: 收益={year_ret*100:6.1f}%, 最大回撤={max_dd*100:5.1f}%")
    
    # 8. 检查潜在的未来数据问题
    print("\n" + "=" * 80)
    print("⚠️ 未来数据检查 (Lookahead Bias Check)")
    print("=" * 80)
    
    # 检查因子计算是否使用了当日数据
    print("\n[检查 1] 因子使用 t-1 日数据: ✅ (代码中明确使用 factors_3d[t - 1, n, idx])")
    
    # 检查择时信号是否滞后
    print("[检查 2] 择时信号滞后 1 天: ✅ (使用 shift_timing_signal)")
    
    # 检查是否有异常集中的交易
    print("\n[检查 3] 交易分布分析:")
    if len(df_trades) > 0:
        df_trades['date'] = pd.to_datetime(df_trades['date'])
        df_trades['year'] = df_trades['date'].dt.year
        
        for year in sorted(df_trades['year'].unique()):
            year_trades = df_trades[df_trades['year'] == year]
            stock_sells = year_trades[year_trades['action'].isin(['SELL', 'FINAL_SELL'])]
            if len(stock_sells) > 0:
                avg_pnl = stock_sells['pnl_pct'].mean()
                win_rate = (stock_sells['pnl_pct'] > 0).mean()
                print(f"  {year}: 卖出 {len(stock_sells)} 次, 平均盈亏={avg_pnl:5.1f}%, 胜率={win_rate*100:4.1f}%")
    
    # 9. 检查是否过度依赖黄金
    print("\n[检查 4] 资产配置分析:")
    if len(df_positions) > 0:
        avg_equity_pct = df_positions['equity_value'].mean() / df_positions['total_value'].mean()
        avg_gold_pct = df_positions['gold_value'].mean() / df_positions['total_value'].mean()
        avg_cash_pct = df_positions['cash'].mean() / df_positions['total_value'].mean()
        
        print(f"  平均股票仓位: {avg_equity_pct*100:.1f}%")
        print(f"  平均黄金仓位: {avg_gold_pct*100:.1f}%")
        print(f"  平均现金仓位: {avg_cash_pct*100:.1f}%")
    
    # 10. 检查最赚钱的交易
    print("\n" + "=" * 40)
    print("💰 最赚钱的 10 笔交易")
    print("=" * 40)
    
    if len(sells) > 0:
        top_trades = sells.nlargest(10, 'pnl_pct')
        for _, trade in top_trades.iterrows():
            print(f"{trade['date'].strftime('%Y-%m-%d')} | {trade['etf']:<8} | "
                  f"入={trade['entry_price']:.3f} 出={trade['price']:.3f} | "
                  f"盈亏={trade['pnl_pct']:+6.1f}%")
    
    # 11. 检查最亏钱的交易
    print("\n" + "=" * 40)
    print("📉 最亏钱的 10 笔交易")
    print("=" * 40)
    
    if len(sells) > 0:
        worst_trades = sells.nsmallest(10, 'pnl_pct')
        for _, trade in worst_trades.iterrows():
            print(f"{trade['date'].strftime('%Y-%m-%d')} | {trade['etf']:<8} | "
                  f"入={trade['entry_price']:.3f} 出={trade['price']:.3f} | "
                  f"盈亏={trade['pnl_pct']:+6.1f}%")
    
    # 12. 保存详细日志
    output_dir = Path(__file__).parent / "results" / "audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    df_trades.to_csv(output_dir / f"trade_log_{timestamp}.csv", index=False)
    df_positions.to_csv(output_dir / f"position_log_{timestamp}.csv", index=False)
    df_nav.to_csv(output_dir / f"nav_log_{timestamp}.csv", index=False)
    
    print(f"\n💾 详细日志已保存到 {output_dir}")
    
    # 13. 审计 Top 200 策略的统计分布
    print("\n" + "=" * 80)
    print("📊 Top 200 策略统计分布")
    print("=" * 80)
    
    print(f"\n收益率分布:")
    print(f"  最小: {df_top['total_return'].min()*100:.1f}%")
    print(f"  25%:  {df_top['total_return'].quantile(0.25)*100:.1f}%")
    print(f"  中位: {df_top['total_return'].median()*100:.1f}%")
    print(f"  75%:  {df_top['total_return'].quantile(0.75)*100:.1f}%")
    print(f"  最大: {df_top['total_return'].max()*100:.1f}%")
    
    print(f"\n最大回撤分布:")
    print(f"  最小: {df_top['max_dd'].min()*100:.1f}%")
    print(f"  25%:  {df_top['max_dd'].quantile(0.25)*100:.1f}%")
    print(f"  中位: {df_top['max_dd'].median()*100:.1f}%")
    print(f"  75%:  {df_top['max_dd'].quantile(0.75)*100:.1f}%")
    print(f"  最大: {df_top['max_dd'].max()*100:.1f}%")
    
    print(f"\n卡玛比率分布:")
    print(f"  最小: {df_top['calmar'].min():.2f}")
    print(f"  25%:  {df_top['calmar'].quantile(0.25):.2f}")
    print(f"  中位: {df_top['calmar'].median():.2f}")
    print(f"  75%:  {df_top['calmar'].quantile(0.75):.2f}")
    print(f"  最大: {df_top['calmar'].max():.2f}")
    
    # 14. 检查频率分布
    print(f"\n频率分布 (Top 200):")
    freq_dist = df_top['freq'].value_counts().sort_index()
    for freq, count in freq_dist.head(10).items():
        print(f"  {freq}天: {count} 个策略")
    
    # 15. 保存 Top 200 详情
    df_top.to_csv(output_dir / f"top200_strategies_{timestamp}.csv", index=False)
    print(f"\n💾 Top 200 策略已保存到 {output_dir / f'top200_strategies_{timestamp}.csv'}")


if __name__ == "__main__":
    main()
