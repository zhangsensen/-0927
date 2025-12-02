#!/usr/bin/env python3
"""
单组合 BT 审计脚本：导出详细的逐笔交易记录

用法:
    # 默认跑历史最佳组合
    uv run python scripts/debug_bt_single_combo.py
    
    # 指定组合
    uv run python scripts/debug_bt_single_combo.py --combo "ADX_14D + PRICE_POSITION_20D"

输出:
    - bt_trades_{combo_hash}.csv: 每笔交易明细
    - bt_orders_{combo_hash}.csv: 每笔订单明细
    - bt_summary_{combo_hash}.json: 汇总指标
"""
import sys
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import yaml
import pandas as pd
import numpy as np
import backtrader as bt

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import shift_timing_signal, generate_rebalance_schedule
from etf_strategy.auditor.core.engine import GenericStrategy, PandasData, LOOKBACK, COMMISSION_RATE

# 自定义策略：额外记录逐日 equity
class EquityTrackingStrategy(GenericStrategy):
    """扩展 GenericStrategy，添加逐日 equity 记录"""
    
    def __init__(self):
        super().__init__()
        self.equity_curve = []  # [(date, equity), ...]
    
    def next(self):
        # 调用父类的 next()
        super().next()
        # 记录当日 equity
        dt = self.datas[0].datetime.date(0)
        equity = self.broker.getvalue()
        self.equity_curve.append({'date': dt, 'equity': equity})


# 历史最佳组合（默认）
DEFAULT_COMBO = "ADX_14D + MAX_DD_60D + PRICE_POSITION_120D + PRICE_POSITION_20D + SHARPE_RATIO_20D"


def load_config():
    """加载配置"""
    config_path = ROOT / "configs" / "combo_wfo_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_data(config):
    """准备数据和因子"""
    print("\n📊 加载数据...")
    
    # 解析配置
    data_config = config.get('data', config)
    etf_codes = data_config.get('symbols', data_config.get('etf_codes', []))
    start_date = data_config.get('start_date', '2020-01-01')
    end_date = data_config.get('end_date', '2025-10-14')
    data_dir = data_config.get('data_dir', str(ROOT / "raw" / "ETF" / "daily"))
    
    loader = DataLoader(data_dir=data_dir)
    ohlcv = loader.load_ohlcv(etf_codes=etf_codes, start_date=start_date, end_date=end_date)
    close_df = ohlcv['close']
    
    print(f"   日期范围: {close_df.index[0]} ~ {close_df.index[-1]}")
    print(f"   ETF 数量: {len(etf_codes)}")
    
    print("\n🔧 计算因子...")
    factor_lib = PreciseFactorLibrary()
    factors_df = factor_lib.compute_all_factors(ohlcv)
    raw_factors = {name: factors_df[name] for name in factor_lib.list_factors()}
    
    print("\n📐 横截面标准化...")
    processor = CrossSectionProcessor(lower_percentile=2.5, upper_percentile=97.5, verbose=False)
    std_factors = processor.process_all_factors(raw_factors)
    
    print("\n⏱️ 计算择时信号...")
    timing_module = LightTimingModule(extreme_threshold=-0.1, extreme_position=0.1)
    timing_series_raw = timing_module.compute_position_ratios(ohlcv['close'])
    
    # 转为 numpy array 进行 shift，然后再转回 Series
    timing_array = shift_timing_signal(timing_series_raw.values)
    timing_series = pd.Series(timing_array, index=timing_series_raw.index)
    
    # 准备 data feeds
    data_feeds = {}
    for ticker in etf_codes:
        df = pd.DataFrame({
            'open': ohlcv['open'][ticker],
            'high': ohlcv['high'][ticker],
            'low': ohlcv['low'][ticker],
            'close': ohlcv['close'][ticker],
            'volume': ohlcv['volume'][ticker],
        })
        df.index = pd.to_datetime(df.index)
        data_feeds[ticker] = df
    
    return std_factors, timing_series, data_feeds, etf_codes


def run_single_combo_backtest(combo_str, std_factors, timing_series, data_feeds, etf_codes, 
                               freq, pos_size, initial_capital):
    """运行单组合回测"""
    print(f"\n🚀 运行 BT 回测: {combo_str}")
    print(f"   FREQ={freq}, POS_SIZE={pos_size}, CAPITAL={initial_capital:,.0f}")
    
    # 解析因子组合
    factor_names = [f.strip() for f in combo_str.split(' + ')]
    print(f"   因子: {factor_names}")
    
    # 合成分数
    factor_df = None
    for fn in factor_names:
        if fn not in std_factors:
            print(f"   ⚠️ 因子 {fn} 不存在，跳过")
            continue
        if factor_df is None:
            factor_df = std_factors[fn].copy()
        else:
            factor_df = factor_df + std_factors[fn]
    
    if factor_df is None:
        raise ValueError("无有效因子")
    
    combined_score_df = factor_df
    
    # 生成调仓日程
    T = len(timing_series)
    rebalance_schedule = generate_rebalance_schedule(
        total_periods=T,
        lookback_window=LOOKBACK,
        freq=freq,
    )
    
    # 构建 Cerebro
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_capital)
    cerebro.broker.setcommission(commission=COMMISSION_RATE, leverage=1.0)
    cerebro.broker.set_coc(True)
    cerebro.broker.set_checksubmit(False)

    for ticker, df in data_feeds.items():
        data = PandasData(dataname=df, name=ticker)
        cerebro.adddata(data)

    cerebro.addstrategy(
        EquityTrackingStrategy, 
        scores=combined_score_df, 
        timing=timing_series, 
        etf_codes=etf_codes, 
        freq=freq, 
        pos_size=pos_size,
        rebalance_schedule=rebalance_schedule,
        dynamic_leverage_enabled=False,  # 调试时关闭动态降权
    )
    
    # Analyzers
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', 
                       timeframe=bt.TimeFrame.Days, compression=1,
                       riskfreerate=0.0, annualize=True)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    start_val = cerebro.broker.getvalue()
    print(f"\n   初始资金: {start_val:,.2f}")
    
    results = cerebro.run()
    
    end_val = cerebro.broker.getvalue()
    print(f"   最终资金: {end_val:,.2f}")
    
    strat = results[0]
    
    return strat, start_val, end_val


def export_results(strat, start_val, end_val, combo_str, output_dir):
    """导出结果"""
    # 生成组合哈希
    combo_hash = hashlib.md5(combo_str.encode()).hexdigest()[:8]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 0. 导出 equity 曲线
    if hasattr(strat, 'equity_curve') and strat.equity_curve:
        equity_df = pd.DataFrame(strat.equity_curve)
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        equity_df = equity_df.set_index('date').sort_index()
        
        # 计算回撤
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
        equity_df['drawdown_pct'] = equity_df['drawdown'] * 100
        
        # 计算日收益率
        equity_df['daily_return'] = equity_df['equity'].pct_change(fill_method=None)
        
        equity_path = output_dir / f"bt_equity_{combo_hash}.csv"
        equity_df.to_csv(equity_path)
        print(f"\n📈 Equity 曲线: {equity_path}")
        print(f"   数据点: {len(equity_df)}")
        print(f"   最大回撤: {equity_df['drawdown'].min()*100:.2f}%")
        print(f"   日收益率标准差: {equity_df['daily_return'].std()*100:.2f}%")
    
    # 1. 导出交易明细（来自 GenericStrategy.trades）
    trades_df = pd.DataFrame(strat.trades)
    if len(trades_df) > 0:
        # 计算持仓天数
        trades_df['holding_days'] = trades_df.apply(
            lambda r: (r['exit_date'] - r['entry_date']).days if pd.notna(r['exit_date']) else 0, 
            axis=1
        )
        trades_df['is_win'] = trades_df['pnlcomm'] > 0
        
        trades_path = output_dir / f"bt_trades_{combo_hash}.csv"
        trades_df.to_csv(trades_path, index=False)
        print(f"\n📄 交易明细: {trades_path}")
        print(f"   总交易笔数: {len(trades_df)}")
        print(f"   盈利笔数: {trades_df['is_win'].sum()}")
        print(f"   亏损笔数: {(~trades_df['is_win']).sum()}")
        print(f"   胜率: {trades_df['is_win'].mean()*100:.1f}%")
        print(f"   平均持仓天数: {trades_df['holding_days'].mean():.1f}")
        print(f"   平均收益率: {trades_df['return_pct'].mean()*100:.2f}%")
    else:
        print("\n⚠️ 无交易记录")
    
    # 2. 导出订单明细（来自 GenericStrategy.orders）
    orders_df = pd.DataFrame(strat.orders)
    if len(orders_df) > 0:
        orders_path = output_dir / f"bt_orders_{combo_hash}.csv"
        orders_df.to_csv(orders_path, index=False)
        print(f"\n📄 订单明细: {orders_path}")
        print(f"   总订单数: {len(orders_df)}")
        print(f"   买入订单: {(orders_df['type'] == 'BUY').sum()}")
        print(f"   卖出订单: {(orders_df['type'] == 'SELL').sum()}")
    
    # 3. 汇总指标
    bt_return = (end_val / start_val) - 1
    
    # 从 analyzer 获取指标
    dd_analysis = strat.analyzers.drawdown.get_analysis()
    max_drawdown = dd_analysis.get('max', {}).get('drawdown', 0.0) / 100.0
    
    sharpe_analysis = strat.analyzers.sharpe.get_analysis()
    sharpe_ratio = sharpe_analysis.get('sharperatio', 0.0) or 0.0
    
    trade_analysis = strat.analyzers.trades.get_analysis()
    total_trades = trade_analysis.get('total', {}).get('total', 0)
    win_trades = trade_analysis.get('won', {}).get('total', 0)
    win_rate = win_trades / total_trades if total_trades > 0 else 0.0
    
    # 平均持仓周期
    len_stats = trade_analysis.get('len', {})
    avg_len = len_stats.get('average', 0.0)
    
    summary = {
        'combo': combo_str,
        'start_value': start_val,
        'end_value': end_val,
        'total_return': bt_return,
        'total_return_pct': f"{bt_return*100:.2f}%",
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': f"{max_drawdown*100:.2f}%",
        'sharpe_ratio': sharpe_ratio,
        'total_trades': total_trades,
        'win_trades': win_trades,
        'loss_trades': total_trades - win_trades,
        'win_rate': win_rate,
        'avg_holding_days': avg_len,
        'margin_failures': strat.margin_failures,
        'timestamp': datetime.now().isoformat(),
    }
    
    summary_path = output_dir / f"bt_summary_{combo_hash}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 汇总指标: {summary_path}")
    
    # 打印汇总
    print("\n" + "="*60)
    print("📊 BT 审计结果汇总")
    print("="*60)
    print(f"组合: {combo_str}")
    print(f"收益率: {bt_return*100:.2f}%")
    print(f"最大回撤: {max_drawdown*100:.2f}%")
    print(f"Sharpe: {sharpe_ratio:.3f}")
    print(f"总交易: {total_trades} 笔")
    print(f"胜率: {win_rate*100:.1f}%")
    print(f"平均持仓: {avg_len:.1f} 天")
    print(f"Margin Failures: {strat.margin_failures}")
    print("="*60)
    
    return summary


def print_sample_trades(strat, n=15):
    """打印样本交易"""
    if not strat.trades:
        return
    
    trades_df = pd.DataFrame(strat.trades)
    
    # 计算持仓天数
    trades_df['holding_days'] = trades_df.apply(
        lambda r: (r['exit_date'] - r['entry_date']).days if pd.notna(r['exit_date']) else 0, 
        axis=1
    )
    trades_df['is_win'] = trades_df['pnlcomm'] > 0
    
    print(f"\n📋 前 {min(n, len(trades_df))} 笔交易:")
    print("-"*100)
    print(f"{'Ticker':<10} {'Entry':<12} {'Exit':<12} {'Days':<6} {'PnL%':>8} {'PnL':>12} {'Win':>5}")
    print("-"*100)
    
    for _, row in trades_df.head(n).iterrows():
        pnl_pct = row['return_pct'] * 100 if 'return_pct' in row else 0
        print(f"{row['ticker']:<10} {str(row['entry_date']):<12} {str(row['exit_date']):<12} "
              f"{row['holding_days']:<6} {pnl_pct:>7.2f}% {row['pnlcomm']:>11.2f} "
              f"{'✅' if row['is_win'] else '❌':>5}")
    
    print("-"*100)
    
    # 按标的统计
    if len(trades_df) > 0:
        print(f"\n📋 按标的统计 (Top 10 收益):")
        ticker_stats = trades_df.groupby('ticker').agg({
            'pnlcomm': ['count', 'sum', 'mean'],
            'is_win': 'mean',
            'return_pct': 'mean',
        }).round(4)
        ticker_stats.columns = ['trades', 'total_pnl', 'avg_pnl', 'win_rate', 'avg_return']
        ticker_stats = ticker_stats.sort_values('total_pnl', ascending=False).head(10)
        print(ticker_stats.to_string())
        
        print(f"\n📋 按标的统计 (Bottom 10 收益):")
        ticker_stats_bottom = trades_df.groupby('ticker').agg({
            'pnlcomm': ['count', 'sum', 'mean'],
            'is_win': 'mean',
            'return_pct': 'mean',
        }).round(4)
        ticker_stats_bottom.columns = ['trades', 'total_pnl', 'avg_pnl', 'win_rate', 'avg_return']
        ticker_stats_bottom = ticker_stats_bottom.sort_values('total_pnl', ascending=True).head(10)
        print(ticker_stats_bottom.to_string())


def main():
    parser = argparse.ArgumentParser(description="单组合 BT 审计")
    parser.add_argument('--combo', type=str, default=DEFAULT_COMBO, help='因子组合字符串')
    parser.add_argument('--output', type=str, default=None, help='输出目录')
    parser.add_argument('--freq', type=int, default=3, help='调仓频率')
    parser.add_argument('--pos', type=int, default=2, help='持仓数量')
    parser.add_argument('--capital', type=float, default=1_000_000.0, help='初始资金')
    args = parser.parse_args()
    
    print("="*60)
    print("🔍 单组合 BT 审计")
    print("="*60)
    
    # 加载配置和数据
    config = load_config()
    std_factors, timing_series, data_feeds, etf_codes = prepare_data(config)
    
    # 运行回测
    strat, start_val, end_val = run_single_combo_backtest(
        combo_str=args.combo,
        std_factors=std_factors,
        timing_series=timing_series,
        data_feeds=data_feeds,
        etf_codes=etf_codes,
        freq=args.freq,
        pos_size=args.pos,
        initial_capital=args.capital,
    )
    
    # 确定输出目录
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ROOT / "results" / f"debug_bt_{timestamp}"
    
    # 导出结果
    summary = export_results(strat, start_val, end_val, args.combo, output_dir)
    
    # 打印样本交易
    print_sample_trades(strat, n=15)
    
    print(f"\n✅ 审计完成！结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
