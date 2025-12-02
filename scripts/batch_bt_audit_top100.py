#!/usr/bin/env python3
"""
批量 BT 审计脚本：对 Top100 策略全部跑 BT 回测

目的：检验排名靠前的策略是否存在过拟合

用法:
    uv run python scripts/batch_bt_audit_top100.py
    uv run python scripts/batch_bt_audit_top100.py --top 20  # 只跑前 20 个
    uv run python scripts/batch_bt_audit_top100.py --parallel 4  # 并行度

输出:
    results/bt_audit_top100_{timestamp}/
    ├── summary.csv           # 所有策略汇总
    ├── equity_curves.parquet # 所有策略的 equity 曲线
    ├── {rank}_trades.csv     # 每个策略的交易明细
    └── analysis.png          # 分析图表
"""
import sys
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

warnings.filterwarnings('ignore')

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


class EquityTrackingStrategy(GenericStrategy):
    """扩展 GenericStrategy，添加逐日 equity 记录"""
    
    def __init__(self):
        super().__init__()
        self.equity_curve = []
    
    def next(self):
        super().next()
        dt = self.datas[0].datetime.date(0)
        equity = self.broker.getvalue()
        self.equity_curve.append({'date': dt, 'equity': equity})


def load_config():
    """加载配置"""
    config_path = ROOT / "configs" / "combo_wfo_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_shared_data():
    """准备共享数据（只加载一次）"""
    print("📊 加载数据和计算因子...")
    
    config = load_config()
    data_config = config.get('data', config)
    etf_codes = data_config.get('symbols', data_config.get('etf_codes', []))
    start_date = data_config.get('start_date', '2020-01-01')
    end_date = data_config.get('end_date', '2025-10-14')
    data_dir = data_config.get('data_dir', str(ROOT / "raw" / "ETF" / "daily"))
    
    loader = DataLoader(data_dir=data_dir)
    ohlcv = loader.load_ohlcv(etf_codes=etf_codes, start_date=start_date, end_date=end_date)
    
    print(f"   日期范围: {ohlcv['close'].index[0]} ~ {ohlcv['close'].index[-1]}")
    print(f"   ETF 数量: {len(etf_codes)}")
    
    # 计算因子
    print("🔧 计算因子...")
    factor_lib = PreciseFactorLibrary()
    factors_df = factor_lib.compute_all_factors(ohlcv)
    raw_factors = {name: factors_df[name] for name in factor_lib.list_factors()}
    
    # 横截面标准化
    print("📐 横截面标准化...")
    processor = CrossSectionProcessor(lower_percentile=2.5, upper_percentile=97.5, verbose=False)
    std_factors = processor.process_all_factors(raw_factors)
    
    # 择时信号
    print("⏱️ 计算择时信号...")
    timing_module = LightTimingModule(extreme_threshold=-0.1, extreme_position=0.1)
    timing_series_raw = timing_module.compute_position_ratios(ohlcv['close'])
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


def run_single_backtest(combo_str, std_factors, timing_series, data_feeds, etf_codes,
                        freq=3, pos_size=2, initial_capital=1_000_000):
    """运行单个组合的回测"""
    # 解析因子组合
    factor_names = [f.strip() for f in combo_str.split(' + ')]
    
    # 合成分数
    factor_df = None
    for fn in factor_names:
        if fn not in std_factors:
            continue
        if factor_df is None:
            factor_df = std_factors[fn].copy()
        else:
            factor_df = factor_df + std_factors[fn]
    
    if factor_df is None:
        return None
    
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
        dynamic_leverage_enabled=False,
    )
    
    # Analyzers
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', 
                       timeframe=bt.TimeFrame.Days, compression=1,
                       riskfreerate=0.0, annualize=True)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    start_val = cerebro.broker.getvalue()
    results = cerebro.run()
    end_val = cerebro.broker.getvalue()
    
    strat = results[0]
    
    # 提取结果
    bt_return = (end_val / start_val) - 1
    
    dd_analysis = strat.analyzers.drawdown.get_analysis()
    max_drawdown = dd_analysis.get('max', {}).get('drawdown', 0.0) / 100.0
    
    sharpe_analysis = strat.analyzers.sharpe.get_analysis()
    sharpe_ratio = sharpe_analysis.get('sharperatio', 0.0) or 0.0
    
    trade_analysis = strat.analyzers.trades.get_analysis()
    total_trades = trade_analysis.get('total', {}).get('total', 0)
    win_trades = trade_analysis.get('won', {}).get('total', 0)
    win_rate = win_trades / total_trades if total_trades > 0 else 0.0
    
    # Equity curve
    equity_df = pd.DataFrame(strat.equity_curve)
    if len(equity_df) > 0:
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        equity_df = equity_df.set_index('date').sort_index()
    
    # Trades
    trades_df = pd.DataFrame(strat.trades)
    if len(trades_df) > 0:
        trades_df['holding_days'] = trades_df.apply(
            lambda r: (r['exit_date'] - r['entry_date']).days if pd.notna(r['exit_date']) else 0, 
            axis=1
        )
    
    return {
        'combo': combo_str,
        'bt_return': bt_return,
        'bt_max_drawdown': max_drawdown,
        'bt_sharpe': sharpe_ratio,
        'total_trades': total_trades,
        'win_trades': win_trades,
        'win_rate': win_rate,
        'margin_failures': strat.margin_failures,
        'equity_df': equity_df,
        'trades_df': trades_df,
    }


def load_top100_combos(top_n: int = 100):
    """加载 Top N 策略
    
    Args:
        top_n: 加载前 N 个策略，如果 > 100，则从 all_combos_scored.parquet 加载
    """
    # 查找最新的 selection 结果
    selection_dirs = sorted(ROOT.glob('results/selection_v2_*'))
    if not selection_dirs:
        raise FileNotFoundError("未找到 selection_v2_* 结果目录")
    
    latest_dir = selection_dirs[-1]
    
    # 如果需要超过 100 个，从 all_combos_scored 加载
    if top_n > 100:
        parquet_path = latest_dir / 'all_combos_scored.parquet'
        if not parquet_path.exists():
            raise FileNotFoundError(f"未找到 {parquet_path}")
        df = pd.read_parquet(parquet_path)
        # 按综合得分排序
        df = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
        print(f"📋 加载全量策略: {parquet_path.name} ({len(df)} 个)")
    else:
        parquet_path = latest_dir / 'top100_by_composite.parquet'
        if not parquet_path.exists():
            raise FileNotFoundError(f"未找到 {parquet_path}")
        df = pd.read_parquet(parquet_path)
        print(f"📋 加载 Top100 策略: {parquet_path.name}")
    
    print(f"   来源目录: {latest_dir.name}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="批量 BT 审计 Top100 策略")
    parser.add_argument('--top', type=int, default=100, help='审计前 N 个策略')
    parser.add_argument('--freq', type=int, default=3, help='调仓频率')
    parser.add_argument('--pos', type=int, default=2, help='持仓数量')
    parser.add_argument('--capital', type=float, default=1_000_000.0, help='初始资金')
    args = parser.parse_args()
    
    print("="*70)
    print("🔍 批量 BT 审计 Top100 策略")
    print("="*70)
    
    # 加载策略（根据 --top 参数决定加载源）
    top100_df = load_top100_combos(top_n=args.top)
    combos_to_audit = top100_df.head(args.top)
    print(f"\n将审计 {len(combos_to_audit)} 个策略")
    
    # 准备共享数据
    std_factors, timing_series, data_feeds, etf_codes = prepare_shared_data()
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / "results" / f"bt_audit_top100_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 输出目录: {output_dir}")
    
    # 逐个运行回测
    results = []
    all_equity_curves = {}
    
    print(f"\n🚀 开始批量回测...")
    print("-"*70)
    
    for idx, row in combos_to_audit.iterrows():
        rank = idx + 1
        combo = row['combo']
        vec_return = row.get('vec_return', 0)
        
        print(f"[{rank:3d}/{len(combos_to_audit)}] {combo[:60]}...", end=" ", flush=True)
        
        try:
            result = run_single_backtest(
                combo_str=combo,
                std_factors=std_factors,
                timing_series=timing_series,
                data_feeds=data_feeds,
                etf_codes=etf_codes,
                freq=args.freq,
                pos_size=args.pos,
                initial_capital=args.capital,
            )
            
            if result is None:
                print("❌ 无效因子")
                continue
            
            # 计算 VEC/BT 差异
            diff_pp = abs(result['bt_return'] - vec_return) * 100
            
            print(f"BT={result['bt_return']*100:+.2f}% VEC={vec_return*100:+.2f}% "
                  f"diff={diff_pp:.2f}pp {'✅' if diff_pp < 1 else '⚠️'}")
            
            # 保存结果
            result['rank'] = rank
            result['vec_return'] = vec_return
            result['diff_pp'] = diff_pp
            
            # 保存交易明细
            if len(result['trades_df']) > 0:
                trades_path = output_dir / f"{rank:03d}_trades.csv"
                result['trades_df'].to_csv(trades_path, index=False)
            
            # 保存 equity curve
            if len(result['equity_df']) > 0:
                all_equity_curves[f"rank_{rank:03d}"] = result['equity_df']['equity']
            
            # 移除大对象以节省内存
            result_summary = {k: v for k, v in result.items() 
                            if k not in ['equity_df', 'trades_df']}
            results.append(result_summary)
            
        except Exception as e:
            print(f"❌ 错误: {e}")
            continue
    
    print("-"*70)
    print(f"✅ 完成 {len(results)}/{len(combos_to_audit)} 个策略")
    
    # 保存汇总结果
    summary_df = pd.DataFrame(results)
    summary_df = summary_df.sort_values('rank')
    summary_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n📄 汇总结果: {summary_path}")
    
    # 保存所有 equity curves
    if all_equity_curves:
        equity_all_df = pd.DataFrame(all_equity_curves)
        equity_path = output_dir / "equity_curves.parquet"
        equity_all_df.to_parquet(equity_path)
        print(f"📈 Equity 曲线: {equity_path}")
    
    # 打印分析
    print("\n" + "="*70)
    print("📊 审计结果分析")
    print("="*70)
    
    print(f"\n【VEC/BT 对齐情况】")
    print(f"   平均差异: {summary_df['diff_pp'].mean():.4f} pp")
    print(f"   最大差异: {summary_df['diff_pp'].max():.4f} pp")
    print(f"   差异 < 0.5pp: {(summary_df['diff_pp'] < 0.5).sum()}/{len(summary_df)}")
    print(f"   差异 < 1.0pp: {(summary_df['diff_pp'] < 1.0).sum()}/{len(summary_df)}")
    
    print(f"\n【BT 收益分布】")
    print(f"   最高收益: {summary_df['bt_return'].max()*100:.2f}%")
    print(f"   最低收益: {summary_df['bt_return'].min()*100:.2f}%")
    print(f"   平均收益: {summary_df['bt_return'].mean()*100:.2f}%")
    print(f"   中位数收益: {summary_df['bt_return'].median()*100:.2f}%")
    
    print(f"\n【收益 > 100% 的策略数】: {(summary_df['bt_return'] > 1.0).sum()}")
    print(f"【收益 > 150% 的策略数】: {(summary_df['bt_return'] > 1.5).sum()}")
    print(f"【收益 > 200% 的策略数】: {(summary_df['bt_return'] > 2.0).sum()}")
    
    print(f"\n【回撤分布】")
    print(f"   最大回撤最小: {summary_df['bt_max_drawdown'].min()*100:.2f}%")
    print(f"   最大回撤最大: {summary_df['bt_max_drawdown'].max()*100:.2f}%")
    print(f"   平均最大回撤: {summary_df['bt_max_drawdown'].mean()*100:.2f}%")
    
    print(f"\n【胜率分布】")
    print(f"   最高胜率: {summary_df['win_rate'].max()*100:.1f}%")
    print(f"   最低胜率: {summary_df['win_rate'].min()*100:.1f}%")
    print(f"   平均胜率: {summary_df['win_rate'].mean()*100:.1f}%")
    
    # Top 10 by BT return
    print(f"\n【BT 收益 Top 10】")
    top10_bt = summary_df.nlargest(10, 'bt_return')
    for _, row in top10_bt.iterrows():
        print(f"   Rank {row['rank']:3d}: BT={row['bt_return']*100:+.2f}% "
              f"VEC={row['vec_return']*100:+.2f}% MDD={row['bt_max_drawdown']*100:.1f}%")
    
    # 检查是否有明显过拟合迹象
    print(f"\n【过拟合检查】")
    # 计算排名靠前的策略 vs 靠后策略的平均收益
    top25 = summary_df[summary_df['rank'] <= 25]['bt_return'].mean()
    bottom25 = summary_df[summary_df['rank'] > 75]['bt_return'].mean() if len(summary_df) > 75 else summary_df[summary_df['rank'] > len(summary_df)//2]['bt_return'].mean()
    
    print(f"   Top 25 平均 BT 收益: {top25*100:.2f}%")
    print(f"   Bottom 25 平均 BT 收益: {bottom25*100:.2f}%")
    print(f"   差距: {(top25 - bottom25)*100:.2f}pp")
    
    if top25 - bottom25 > 0.5:  # 差距超过 50%
        print(f"   ⚠️ 排名靠前的策略明显优于靠后策略，可能存在选择偏差")
    else:
        print(f"   ✅ 策略收益分布相对均匀，过拟合风险较低")
    
    # 生成可视化
    try:
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. BT vs VEC 散点图
        ax1 = axes[0, 0]
        ax1.scatter(summary_df['vec_return']*100, summary_df['bt_return']*100, 
                   alpha=0.6, c=summary_df['rank'], cmap='viridis')
        ax1.plot([0, 250], [0, 250], 'r--', label='y=x')
        ax1.set_xlabel('VEC Return (%)')
        ax1.set_ylabel('BT Return (%)')
        ax1.set_title('VEC vs BT Return (color=rank)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 收益分布直方图
        ax2 = axes[0, 1]
        ax2.hist(summary_df['bt_return']*100, bins=20, edgecolor='black', alpha=0.7)
        ax2.axvline(x=summary_df['bt_return'].mean()*100, color='red', 
                   linestyle='--', label=f'Mean: {summary_df["bt_return"].mean()*100:.1f}%')
        ax2.set_xlabel('BT Return (%)')
        ax2.set_ylabel('Count')
        ax2.set_title('BT Return Distribution')
        ax2.legend()
        
        # 3. 排名 vs 收益
        ax3 = axes[1, 0]
        ax3.scatter(summary_df['rank'], summary_df['bt_return']*100, alpha=0.6)
        ax3.set_xlabel('Composite Rank')
        ax3.set_ylabel('BT Return (%)')
        ax3.set_title('Rank vs BT Return')
        ax3.grid(True, alpha=0.3)
        
        # 4. 回撤 vs 收益
        ax4 = axes[1, 1]
        sc = ax4.scatter(summary_df['bt_max_drawdown']*100, summary_df['bt_return']*100,
                        c=summary_df['win_rate']*100, cmap='RdYlGn', alpha=0.7)
        ax4.set_xlabel('Max Drawdown (%)')
        ax4.set_ylabel('BT Return (%)')
        ax4.set_title('Risk-Return (color=win_rate)')
        plt.colorbar(sc, ax=ax4, label='Win Rate %')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig_path = output_dir / "analysis.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 分析图表: {fig_path}")
        plt.close()
        
    except Exception as e:
        print(f"\n⚠️ 生成图表失败: {e}")
    
    print("\n" + "="*70)
    print(f"✅ 审计完成！结果保存在: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
