#!/usr/bin/env python3
"""
🔬 深度审计脚本：Top 100 策略的权益曲线、回撤分析、交易日志
用于验证策略是否为"圣杯"，置信度拉到最高

参考 batch_bt_backtest.py 实现
"""

import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import yaml
import pandas as pd
import numpy as np
import backtrader as bt
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import shift_timing_signal, generate_rebalance_schedule
from etf_strategy.auditor.core.engine import GenericStrategy, PandasData


class DeepAuditStrategy(GenericStrategy):
    """扩展的审计策略，记录详细交易日志"""
    
    def __init__(self):
        super().__init__()
        self.daily_equity = []
        self.daily_dates = []
        self.all_orders = []
        self.all_trades = []
        
    def next(self):
        # 记录每日权益
        dt = self.datas[0].datetime.date(0)
        equity = self.broker.getvalue()
        self.daily_dates.append(dt)
        self.daily_equity.append(equity)
        
        # 调用父类逻辑
        super().next()
    
    def notify_order(self, order):
        super().notify_order(order)
        if order.status in [order.Completed]:
            self.all_orders.append({
                'date': self.datas[0].datetime.date(0),
                'ticker': order.data._name,
                'type': 'BUY' if order.isbuy() else 'SELL',
                'price': order.executed.price,
                'size': order.executed.size,
                'value': abs(order.executed.value),
                'comm': order.executed.comm,
            })
    
    def notify_trade(self, trade):
        super().notify_trade(trade)
        if trade.isclosed:
            entry_date = bt.num2date(trade.dtopen).date()
            exit_date = bt.num2date(trade.dtclose).date()
            holding_days = (exit_date - entry_date).days
            
            self.all_trades.append({
                'ticker': trade.data._name,
                'entry_date': entry_date,
                'exit_date': exit_date,
                'holding_days': holding_days,
                'entry_price': trade.price,
                'size': abs(trade.size),
                'pnl': trade.pnl,
                'pnl_comm': trade.pnlcomm,
                'return_pct': trade.pnl / (trade.price * abs(trade.size)) * 100 if trade.size != 0 else 0,
            })


def run_deep_audit(combo: str, combined_score_df, timing_series, etf_codes, data_feeds, 
                   rebalance_schedule, config) -> dict:
    """运行单个策略的深度审计"""
    
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(config['initial_capital'])
    cerebro.broker.setcommission(commission=config['commission_rate'], leverage=1.0)
    cerebro.broker.set_coc(True)
    cerebro.broker.set_checksubmit(False)

    for ticker, df in data_feeds.items():
        data = PandasData(dataname=df, name=ticker)
        cerebro.adddata(data)

    cerebro.addstrategy(
        DeepAuditStrategy, 
        scores=combined_score_df, 
        timing=timing_series, 
        etf_codes=etf_codes, 
        freq=config['freq'], 
        pos_size=config['pos_size'],
        rebalance_schedule=rebalance_schedule,
        dynamic_leverage_enabled=False,
    )
    
    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', 
                       timeframe=bt.TimeFrame.Days, compression=1,
                       riskfreerate=0.0, annualize=True)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    start_val = cerebro.broker.getvalue()
    results = cerebro.run()
    end_val = cerebro.broker.getvalue()
    strat = results[0]

    # 提取权益曲线
    equity_df = pd.DataFrame({
        'date': strat.daily_dates,
        'equity': strat.daily_equity
    })
    
    # 提取交易日志
    trades_df = pd.DataFrame(strat.all_trades) if strat.all_trades else pd.DataFrame()
    orders_df = pd.DataFrame(strat.all_orders) if strat.all_orders else pd.DataFrame()
    
    # 提取分析器结果
    dd_analysis = strat.analyzers.drawdown.get_analysis()
    max_drawdown = dd_analysis.get('max', {}).get('drawdown', 0.0) / 100.0
    
    trade_analysis = strat.analyzers.trades.get_analysis()
    total_trades = trade_analysis.get('total', {}).get('total', 0)
    win_trades = trade_analysis.get('won', {}).get('total', 0)
    win_rate = win_trades / total_trades if total_trades > 0 else 0.0
    
    won_pnl = trade_analysis.get('won', {}).get('pnl', {}).get('total', 0.0)
    lost_pnl = abs(trade_analysis.get('lost', {}).get('pnl', {}).get('total', 0.0))
    profit_factor = won_pnl / lost_pnl if lost_pnl > 0 else float('inf')
    
    avg_len = trade_analysis.get('len', {}).get('average', 0.0)
    max_len = trade_analysis.get('len', {}).get('max', 0)
    
    bt_return = (end_val / start_val) - 1
    years = len(equity_df) / 252.0
    annual_return = (1.0 + bt_return) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    calmar_ratio = annual_return / max_drawdown if max_drawdown > 0.0001 else 0.0
    
    return {
        'combo': combo,
        'equity_df': equity_df,
        'trades_df': trades_df,
        'orders_df': orders_df,
        'metrics': {
            'return': bt_return,
            'max_drawdown': max_drawdown,
            'annual_return': annual_return,
            'calmar_ratio': calmar_ratio,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_len': avg_len,
            'max_len': max_len,
        },
        'margin_failures': strat.margin_failures,
    }


def analyze_drawdown_periods(equity_series: pd.Series) -> pd.DataFrame:
    """分析所有回撤期间"""
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max
    
    # 找到所有回撤超过 3% 的期间
    periods = []
    in_drawdown = False
    start_idx = None
    peak_value = None
    
    for i, (date, dd) in enumerate(drawdown.items()):
        if dd < -0.03 and not in_drawdown:  # 回撤超过 3% 开始记录
            in_drawdown = True
            start_idx = i
            peak_value = rolling_max.iloc[i]
        elif dd >= -0.01 and in_drawdown:  # 恢复到 1% 以内结束
            in_drawdown = False
            period_dd = drawdown.iloc[start_idx:i].min()
            trough_idx = drawdown.iloc[start_idx:i].idxmin()
            periods.append({
                'start_date': equity_series.index[start_idx],
                'trough_date': trough_idx,
                'end_date': date,
                'duration_days': (date - equity_series.index[start_idx]).days,
                'max_drawdown': period_dd * 100,
                'peak_value': peak_value,
                'trough_value': equity_series.loc[trough_idx],
            })
    
    # 如果还在回撤中
    if in_drawdown:
        period_dd = drawdown.iloc[start_idx:].min()
        trough_idx = drawdown.iloc[start_idx:].idxmin()
        periods.append({
            'start_date': equity_series.index[start_idx],
            'trough_date': trough_idx,
            'end_date': equity_series.index[-1],
            'duration_days': (equity_series.index[-1] - equity_series.index[start_idx]).days,
            'max_drawdown': period_dd * 100,
            'peak_value': peak_value,
            'trough_value': equity_series.loc[trough_idx],
            'still_in_drawdown': True,
        })
    
    return pd.DataFrame(periods).sort_values('max_drawdown') if periods else pd.DataFrame()


def main():
    print("=" * 100)
    print("🔬 深度审计：Top 100 策略 - 权益曲线、回撤分析、交易日志")
    print("   目标：验证是否为圣杯策略，置信度拉到最高")
    print("=" * 100)
    
    # 加载配置（完全参考 batch_bt_backtest.py）
    config_path = ROOT / "configs/combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 加载数据
    print("\n⏳ 加载数据...")
    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )
    
    # 计算因子
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)
    
    factor_names_list = raw_factors_df.columns.get_level_values(0).unique().tolist()
    raw_factors = {fname: raw_factors_df[fname] for fname in factor_names_list}
    
    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)
    
    factor_names = sorted(std_factors.keys())
    first_factor = std_factors[factor_names[0]]
    dates = first_factor.index
    etf_codes = first_factor.columns.tolist()
    
    # 提取参数
    backtest_config = config.get("backtest", {})
    freq = backtest_config.get("freq", 3)
    pos_size = backtest_config.get("pos_size", 2)
    initial_capital = float(backtest_config.get("initial_capital", 1_000_000.0))
    commission_rate = float(backtest_config.get("commission_rate", 0.0002))
    lookback = backtest_config.get("lookback", 252)
    
    timing_config = backtest_config.get("timing", {})
    extreme_threshold = timing_config.get("extreme_threshold", -0.1)
    extreme_position = timing_config.get("extreme_position", 0.1)
    
    audit_config = {
        'freq': freq,
        'pos_size': pos_size,
        'initial_capital': initial_capital,
        'commission_rate': commission_rate,
        'lookback': lookback,
    }
    
    print(f"📊 审计参数: FREQ={freq}, POS={pos_size}, Capital={initial_capital:,.0f}")
    print(f"📊 择时参数: threshold={extreme_threshold}, position={extreme_position}")
    
    # 择时
    timing_module = LightTimingModule(
        extreme_threshold=extreme_threshold,
        extreme_position=extreme_position,
    )
    timing_series_raw = timing_module.compute_position_ratios(ohlcv["close"])
    timing_arr_shifted = shift_timing_signal(timing_series_raw.reindex(dates).fillna(1.0).values)
    timing_series = pd.Series(timing_arr_shifted, index=dates)
    
    # 生成调仓日程
    total_periods = len(dates)
    rebalance_schedule = generate_rebalance_schedule(total_periods, lookback, freq)
    
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
        df = df.reindex(dates)
        df = df.ffill().fillna(0.01)
        data_feeds[ticker] = df
    
    print(f"✅ 数据加载完成：{len(dates)} 天 × {len(etf_codes)} 只 ETF")
    
    # 读取 Top 100 策略
    vec_df = pd.read_csv(ROOT / 'results' / 'vec_full_space_20251130_235418' / 'full_space_results.csv')
    top100 = vec_df.sort_values('vec_calmar_ratio', ascending=False).head(100)
    
    print(f"📋 待审计策略数: {len(top100)}")
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / f"results/deep_audit_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 存储结果
    all_results = []
    all_equity_curves = {}
    all_drawdown_periods = []
    all_trades = []
    
    print(f"\n🚀 开始深度审计...")
    
    for idx, (_, row) in enumerate(tqdm(top100.iterrows(), total=len(top100), desc="深度审计")):
        combo = row['combo']
        factors = [f.strip() for f in combo.split(' + ')]
        
        try:
            # 计算组合得分
            factor_dfs = [std_factors[f] for f in factors]
            combined_score_df = sum(factor_dfs) / len(factor_dfs)
            
            # 运行审计
            result = run_deep_audit(
                combo, combined_score_df, timing_series, etf_codes, 
                data_feeds, rebalance_schedule, audit_config
            )
            
            all_results.append({
                'rank': idx + 1,
                'combo': combo,
                **result['metrics'],
                'margin_failures': result['margin_failures'],
            })
            
            # 保存权益曲线
            equity_df = result['equity_df']
            all_equity_curves[f"Strategy_{idx+1}"] = equity_df.set_index('date')['equity']
            
            # 分析回撤期间
            equity_series = pd.Series(equity_df['equity'].values, index=pd.to_datetime(equity_df['date']))
            dd_periods = analyze_drawdown_periods(equity_series)
            if len(dd_periods) > 0:
                dd_periods['rank'] = idx + 1
                dd_periods['combo'] = combo
                all_drawdown_periods.append(dd_periods)
            
            # 保存交易
            if len(result['trades_df']) > 0:
                trades = result['trades_df'].copy()
                trades['rank'] = idx + 1
                trades['combo'] = combo
                all_trades.append(trades)
            
            # Top 10 保存详细文件
            if idx < 10:
                equity_df.to_csv(output_dir / f"equity_{idx+1}.csv", index=False)
                if len(result['trades_df']) > 0:
                    result['trades_df'].to_csv(output_dir / f"trades_{idx+1}.csv", index=False)
                if len(result['orders_df']) > 0:
                    result['orders_df'].to_csv(output_dir / f"orders_{idx+1}.csv", index=False)
                    
        except Exception as e:
            print(f"\n❌ 策略 {idx+1} 审计失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存汇总结果
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "audit_summary.csv", index=False)
    
    # 合并权益曲线
    if all_equity_curves:
        equity_combined = pd.DataFrame(all_equity_curves)
        equity_combined.to_csv(output_dir / "all_equity_curves.csv")
    
    # 合并回撤分析
    if all_drawdown_periods:
        dd_combined = pd.concat(all_drawdown_periods, ignore_index=True)
        dd_combined.to_csv(output_dir / "all_drawdown_periods.csv", index=False)
    
    # 合并交易
    if all_trades:
        trades_combined = pd.concat(all_trades, ignore_index=True)
        trades_combined.to_csv(output_dir / "all_trades.csv", index=False)
    
    print(f"\n✅ 深度审计完成！输出目录: {output_dir}")
    
    # 打印统计
    print("\n" + "=" * 100)
    print("📊 审计统计汇总")
    print("=" * 100)
    
    print(f"""
策略数量: {len(results_df)}
收益率:   {results_df['return'].min()*100:.1f}% - {results_df['return'].max()*100:.1f}% (中位数: {results_df['return'].median()*100:.1f}%)
回撤:     {results_df['max_drawdown'].min()*100:.1f}% - {results_df['max_drawdown'].max()*100:.1f}% (中位数: {results_df['max_drawdown'].median()*100:.1f}%)
Calmar:   {results_df['calmar_ratio'].min():.3f} - {results_df['calmar_ratio'].max():.3f} (中位数: {results_df['calmar_ratio'].median():.3f})
胜率:     {results_df['win_rate'].min()*100:.1f}% - {results_df['win_rate'].max()*100:.1f}% (中位数: {results_df['win_rate'].median()*100:.1f}%)
盈亏比:   {results_df['profit_factor'].min():.2f} - {results_df['profit_factor'].max():.2f} (中位数: {results_df['profit_factor'].median():.2f})
""")
    
    return output_dir


if __name__ == "__main__":
    main()
