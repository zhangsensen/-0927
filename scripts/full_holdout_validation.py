#!/usr/bin/env python3
"""
全量策略Holdout验证 - 使用VEC引擎高速验证所有12597个策略

目标: 找出所有在Holdout期真正有效的策略，不局限于训练集Top100
方法: 直接复用batch_vec_backtest.py的VEC引擎
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import shift_timing_signal

# 导入VEC回测引擎
from batch_vec_backtest import run_vec_backtest

# Holdout配置
TRAINING_END = "2025-05-31"
HOLDOUT_START = "2025-06-01"
HOLDOUT_END = "2025-12-08"

# 验证标准
HOLDOUT_MIN_RETURN = 0.0      # Holdout期收益 > 0%
HOLDOUT_MIN_SHARPE = 0.3      # Sharpe > 0.3
HOLDOUT_MAX_DD = 0.25         # 最大回撤 < 25%


def fast_backtest(close_prices, signal, timing, freq, pos_size, commission=0.0002):
    """快速向量化回测"""
    returns = close_prices.pct_change()
    total_periods = len(signal)
    
    # 生成调仓日程
    rebalance_schedule = generate_rebalance_schedule(
        total_periods=total_periods,
        lookback_window=0,
        freq=freq
    )
    rebalance_mask = np.zeros(total_periods, dtype=bool)
    rebalance_mask[rebalance_schedule] = True
    
    # 初始化
    capital = 1_000_000.0
    cash = capital
    positions = {}
    equity_curve = [capital]
    
    for t in range(1, total_periods):
        # 调仓日
        if rebalance_mask[t]:
            # 清仓
            for code, shares in positions.items():
                if shares > 0:
                    sell_price = close_prices.iloc[t][code]
                    if pd.notna(sell_price):
                        cash += shares * sell_price * (1 - commission)
            positions = {}
            
            # 选股
            scores = signal.iloc[t].dropna().sort_values(ascending=False)
            if len(scores) > 0:
                selected = scores.head(pos_size).index.tolist()
                
                # 择时调整
                position_ratio = timing.iloc[t] if pd.notna(timing.iloc[t]) else 1.0
                position_ratio = np.clip(position_ratio, 0.0, 1.0)
                
                # 分配资金
                invest_cash = cash * position_ratio
                per_position = invest_cash / len(selected)
                
                for code in selected:
                    buy_price = close_prices.iloc[t][code]
                    if pd.notna(buy_price) and buy_price > 0:
                        shares = int(per_position / buy_price)
                        cost = shares * buy_price * (1 + commission)
                        if cost <= cash:
                            positions[code] = shares
                            cash -= cost
        
        # 计算权益
        holdings_value = sum(
            shares * close_prices.iloc[t][code] 
            for code, shares in positions.items() 
            if pd.notna(close_prices.iloc[t][code])
        )
        equity = cash + holdings_value
        equity_curve.append(equity)
    
    # 计算指标
    equity_series = pd.Series(equity_curve, index=signal.index)
    returns_series = equity_series.pct_change().dropna()
    
    total_return = (equity_series.iloc[-1] / equity_series.iloc[0] - 1)
    
    if len(returns_series) > 0 and returns_series.std() > 0:
        sharpe = returns_series.mean() / returns_series.std() * np.sqrt(252)
    else:
        sharpe = 0.0
    
    cummax = equity_series.cummax()
    drawdown = (equity_series - cummax) / cummax
    max_dd = abs(drawdown.min())
    
    return {
        'return': total_return,
        'sharpe': sharpe,
        'max_dd': max_dd
    }


def main():
    print("="*80)
    print("🔬 全量策略 Holdout 验证 (12597个策略)")
    print("="*80)
    print(f"训练集: 2020-01-01 至 {TRAINING_END}")
    print(f"Holdout集: {HOLDOUT_START} 至 {HOLDOUT_END}")
    print(f"验证标准: 收益>{HOLDOUT_MIN_RETURN*100}%, Sharpe>{HOLDOUT_MIN_SHARPE}, 回撤<{HOLDOUT_MAX_DD*100}%")
    print("="*80)
    
    # 加载配置
    config_path = ROOT / 'configs/combo_wfo_config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    data_dir = Path(config['data']['data_dir'])
    
    # 加载全量数据
    print("\n📂 加载完整数据...")
    loader = DataLoader(data_dir=data_dir, cache_dir=ROOT / '.cache')
    etf_files = list(data_dir.glob("*.parquet"))
    etf_codes = [f.stem.split('_')[0].split('.')[0] for f in etf_files]
    
    ohlcv = loader.load_ohlcv(etf_codes=etf_codes, start_date='2020-01-01', end_date=HOLDOUT_END)
    
    # 计算因子
    print("🔧 计算因子...")
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)
    factor_names = sorted(raw_factors_df.columns.get_level_values(0).unique().tolist())
    raw_factors = {f: raw_factors_df[f] for f in factor_names}
    
    # 标准化
    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)
    
    # 择时信号
    timing_module = LightTimingModule(extreme_threshold=-0.1, extreme_position=0.1)
    timing_series = timing_module.compute_position_ratios(ohlcv['close'])
    timing_values = shift_timing_signal(timing_series.values)
    timing_series = pd.Series(timing_values, index=timing_series.index)
    
    # 加载所有策略组合
    print("📊 加载策略组合...")
    wfo_dirs = sorted([d for d in (ROOT / 'results').glob("run_*") if d.is_dir()], reverse=True)
    latest_wfo = wfo_dirs[0]
    all_combos_path = latest_wfo / 'all_combos.parquet'
    
    all_combos = pd.read_parquet(all_combos_path)
    print(f"   找到 {len(all_combos)} 个策略组合")
    
    # 提取组合列表
    combos = []
    for combo_str in all_combos['combo'].values:
        factors = [f.strip() for f in combo_str.split(' + ')]
        combos.append(factors)
    
    # 参数
    freq = 3
    pos_size = 2
    commission = 0.0002
    
    # 批量回测
    print(f"\n🚀 开始批量Holdout验证...")
    results = []
    
    close_prices = ohlcv['close']
    
    for i, combo in enumerate(tqdm(combos, desc="Holdout验证")):
        try:
            # 组合信号
            combined_score = pd.DataFrame(0.0, index=std_factors[combo[0]].index, 
                                         columns=std_factors[combo[0]].columns)
            for f in combo:
                combined_score += std_factors[f].fillna(0.0)
            combined_score = combined_score.shift(1)
            
            # 训练集回测
            train_signal = combined_score.loc[:TRAINING_END]
            train_timing = timing_series.loc[:TRAINING_END]
            train_close = close_prices.loc[:TRAINING_END]
            
            train_result = fast_backtest(train_close, train_signal, train_timing, freq, pos_size, commission)
            
            # Holdout期回测
            holdout_signal = combined_score.loc[HOLDOUT_START:HOLDOUT_END]
            holdout_timing = timing_series.loc[HOLDOUT_START:HOLDOUT_END]
            holdout_close = close_prices.loc[HOLDOUT_START:HOLDOUT_END]
            
            holdout_result = fast_backtest(holdout_close, holdout_signal, holdout_timing, freq, pos_size, commission)
            
            # 验证检查
            passed = (
                holdout_result['return'] > HOLDOUT_MIN_RETURN and
                holdout_result['sharpe'] > HOLDOUT_MIN_SHARPE and
                holdout_result['max_dd'] < HOLDOUT_MAX_DD
            )
            
            results.append({
                'combo': ' + '.join(combo),
                'combo_size': len(combo),
                'train_return': train_result['return'],
                'train_sharpe': train_result['sharpe'],
                'train_max_dd': train_result['max_dd'],
                'holdout_return': holdout_result['return'],
                'holdout_sharpe': holdout_result['sharpe'],
                'holdout_max_dd': holdout_result['max_dd'],
                'passed': passed,
                'return_diff': holdout_result['return'] - train_result['return'],
                'sharpe_diff': holdout_result['sharpe'] - train_result['sharpe']
            })
            
        except Exception as e:
            print(f"\n⚠️  策略 {i+1} 失败: {e}")
            continue
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    
    # 保存结果
    output_dir = ROOT / 'results' / f"full_holdout_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'full_holdout_results.csv'
    results_df.to_csv(output_path, index=False)
    
    # 分析结果
    print(f"\n{'='*80}")
    print(f"📊 全量验证结果")
    print(f"{'='*80}")
    
    total = len(results_df)
    passed = results_df['passed'].sum()
    pass_rate = passed / total * 100
    
    print(f"\n总策略数: {total}")
    print(f"通过验证: {passed} ({pass_rate:.1f}%)")
    print(f"未通过: {total - passed} ({100-pass_rate:.1f}%)")
    
    # 通过的策略统计
    passed_df = results_df[results_df['passed']].copy()
    
    if len(passed_df) > 0:
        print(f"\n{'='*80}")
        print(f"✅ 通过验证的策略 Top 20 (按Holdout收益排序)")
        print(f"{'='*80}")
        
        passed_sorted = passed_df.sort_values('holdout_return', ascending=False)
        
        print(f"\n{'排名':<6} {'Holdout收益':<12} {'Holdout Sharpe':<15} {'Holdout回撤':<12} {'训练集收益':<12}")
        print("-"*80)
        
        for idx, row in passed_sorted.head(20).iterrows():
            print(f"{idx+1:<6} {row['holdout_return']*100:>10.2f}% {row['holdout_sharpe']:>14.3f} {row['holdout_max_dd']*100:>10.2f}% {row['train_return']*100:>10.2f}%")
            print(f"       组合: {row['combo']}")
            print()
        
        # 保存通过的策略
        passed_output = output_dir / 'passed_strategies.csv'
        passed_sorted.to_csv(passed_output, index=False)
        
        # 按Holdout Sharpe排序
        passed_by_sharpe = passed_df.sort_values('holdout_sharpe', ascending=False)
        sharpe_output = output_dir / 'passed_strategies_by_sharpe.csv'
        passed_by_sharpe.to_csv(sharpe_output, index=False)
        
        print(f"\n{'='*80}")
        print(f"📈 统计分析")
        print(f"{'='*80}")
        
        print(f"\nHoldout期表现:")
        print(f"  平均收益: {passed_df['holdout_return'].mean()*100:.2f}%")
        print(f"  中位收益: {passed_df['holdout_return'].median()*100:.2f}%")
        print(f"  平均Sharpe: {passed_df['holdout_sharpe'].mean():.3f}")
        print(f"  平均回撤: {passed_df['holdout_max_dd'].mean()*100:.2f}%")
        
        print(f"\n训练集表现:")
        print(f"  平均收益: {passed_df['train_return'].mean()*100:.2f}%")
        print(f"  中位收益: {passed_df['train_return'].median()*100:.2f}%")
        print(f"  平均Sharpe: {passed_df['train_sharpe'].mean():.3f}")
        
        print(f"\n劣化分析:")
        print(f"  平均收益劣化: {passed_df['return_diff'].mean()*100:.2f}pp")
        print(f"  平均Sharpe劣化: {passed_df['sharpe_diff'].mean():.3f}")
        
        # 因子频率统计
        print(f"\n{'='*80}")
        print(f"🏆 通过验证策略的因子频率 (Top 15)")
        print(f"{'='*80}")
        
        factor_counts = {}
        for combo_str in passed_df['combo']:
            for factor in combo_str.split(' + '):
                factor = factor.strip()
                factor_counts[factor] = factor_counts.get(factor, 0) + 1
        
        sorted_factors = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)
        
        for factor, count in sorted_factors[:15]:
            pct = count / len(passed_df) * 100
            bar = '█' * int(pct / 5)
            print(f"  {factor:<40} {count:>5} ({pct:>5.1f}%) {bar}")
    
    else:
        print("\n⚠️  没有策略通过Holdout验证！")
    
    print(f"\n{'='*80}")
    print(f"✅ 结果已保存到: {output_dir}")
    print(f"{'='*80}")
    print(f"  - full_holdout_results.csv          (全部{total}个策略)")
    print(f"  - passed_strategies.csv             ({passed}个通过策略，按收益排序)")
    print(f"  - passed_strategies_by_sharpe.csv   ({passed}个通过策略，按Sharpe排序)")


if __name__ == '__main__':
    main()
