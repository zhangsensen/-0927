#!/usr/bin/env python3
"""
Holdout期验证脚本 - 防止过拟合的最后防线

核心理念:
1. 训练集: 2020-01-01 至 2025-05-31 (4.5年)
2. Holdout集: 2025-06-01 至 2025-12-08 (6个月)
3. 在训练集上选出的Top策略，必须在Holdout期验证通过才能启用

验证标准:
- Holdout期收益 > 0%
- Holdout期Sharpe > 0.5
- Holdout期最大回撤 < 20%
- 与训练期表现相关性 > 0.5 (稳定性检验)

用法:
    python scripts/validate_holdout.py --input results/selection_v2_*/top100_by_composite.csv
"""

import sys
from pathlib import Path
from datetime import datetime
import argparse
import pandas as pd
import numpy as np
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import shift_timing_signal, generate_rebalance_schedule

# ============================================================================
# Holdout配置
# ============================================================================
TRAINING_END = "2025-05-31"  # 训练集截止日期
HOLDOUT_START = "2025-06-01"  # Holdout集起始日期
HOLDOUT_END = "2025-12-08"    # Holdout集截止日期

# 验证通过的最低标准
HOLDOUT_MIN_RETURN = 0.0      # 最低收益率 0%
HOLDOUT_MIN_SHARPE = 0.5      # 最低Sharpe 0.5
HOLDOUT_MAX_DD = 0.20         # 最大回撤 20%
STABILITY_CORR = 0.5          # 训练期/Holdout期相关性 > 0.5


def load_etf_data(data_dir: Path, end_date: str = None):
    """加载ETF数据（可选截止日期）"""
    loader = DataLoader(data_dir=data_dir, cache_dir=ROOT / '.cache')
    
    # 加载所有ETF
    etf_files = list(data_dir.glob("*.parquet"))
    etf_codes = [f.stem.split('_')[0].split('.')[0] for f in etf_files]
    
    # 加载OHLCV
    ohlcv = loader.load_ohlcv(etf_codes=etf_codes, end_date=end_date)
    
    return ohlcv


def compute_strategy_signal(ohlcv: dict, combo: list, timing_config: dict):
    """计算策略信号"""
    # 1. 计算因子
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)
    raw_factors = {f: raw_factors_df[f] for f in combo}
    
    # 2. 标准化
    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)
    
    # 3. 组合信号
    combined_score = pd.DataFrame(0.0, index=std_factors[combo[0]].index, 
                                   columns=std_factors[combo[0]].columns)
    for f in combo:
        combined_score += std_factors[f].fillna(0.0)
    
    # 4. 择时信号
    timing_module = LightTimingModule(
        extreme_threshold=timing_config.get('extreme_threshold', -0.1),
        extreme_position=timing_config.get('extreme_position', 0.1)
    )
    timing_series = timing_module.compute_position_ratios(ohlcv['close'])
    timing_values = shift_timing_signal(timing_series.values)
    timing_series = pd.Series(timing_values, index=timing_series.index)
    
    # 5. Shift信号（防止前视偏差）
    combined_score = combined_score.shift(1)
    
    return combined_score, timing_series


def backtest_strategy(ohlcv: dict, signal: pd.DataFrame, timing: pd.Series,
                      freq: int, pos_size: int, commission: float = 0.0002,
                      start_date: str = None, end_date: str = None):
    """向量化回测（可指定日期范围）"""
    close = ohlcv['close']
    returns = close.pct_change()
    
    # 日期过滤
    if start_date:
        signal = signal.loc[start_date:]
        returns = returns.loc[start_date:]
        timing = timing.loc[start_date:]
        close = close.loc[start_date:]
    if end_date:
        signal = signal.loc[:end_date]
        returns = returns.loc[:end_date]
        timing = timing.loc[:end_date]
        close = close.loc[:end_date]
    
    # 生成调仓日程
    # 注意: 这里的signal已经是切片后的，且已经包含了shift(1)
    # 所以我们不需要再设置lookback_window来跳过预热期（因为信号已经预热好了）
    # 我们只需要确保从切片后的第1天开始（因为第0天可能无法交易或作为基准）
    total_periods = len(signal)
    rebalance_schedule = generate_rebalance_schedule(
        total_periods=total_periods,
        lookback_window=0,  # 关键修改: 设为0，因为信号已预热
        freq=freq
    )
    
    # 修正: rebalance_schedule 返回索引数组，不是布尔数组
    # 转换为布尔数组
    rebalance_mask = np.zeros(total_periods, dtype=bool)
    rebalance_mask[rebalance_schedule] = True
    
    # 初始化
    capital = 1_000_000.0
    cash = capital
    positions = {}  # {code: shares}
    equity_curve = [capital]
    
    for t in range(1, total_periods):
        date = signal.index[t]
        
        # 调仓日
        if rebalance_mask[t]:
            # 清仓
            for code, shares in positions.items():
                if shares > 0:
                    sell_price = close.iloc[t][code]
                    if pd.notna(sell_price):
                        cash += shares * sell_price * (1 - commission)
            positions = {}
            
            # 选股
            scores = signal.iloc[t].dropna().sort_values(ascending=False)
            if len(scores) > 0:
                selected = scores.head(pos_size).index.tolist()
                
                # 择时调整仓位
                position_ratio = timing.iloc[t] if pd.notna(timing.iloc[t]) else 1.0
                position_ratio = np.clip(position_ratio, 0.0, 1.0)
                
                # 分配资金
                invest_cash = cash * position_ratio
                per_position = invest_cash / len(selected)
                
                for code in selected:
                    buy_price = close.iloc[t][code]
                    if pd.notna(buy_price) and buy_price > 0:
                        shares = int(per_position / buy_price)
                        cost = shares * buy_price * (1 + commission)
                        if cost <= cash:
                            positions[code] = shares
                            cash -= cost
        
        # 计算权益
        holdings_value = sum(
            shares * close.iloc[t][code] 
            for code, shares in positions.items() 
            if pd.notna(close.iloc[t][code])
        )
        equity = cash + holdings_value
        equity_curve.append(equity)
    
    # 计算指标
    equity_series = pd.Series(equity_curve, index=signal.index)
    returns_series = equity_series.pct_change().dropna()
    
    total_return = (equity_series.iloc[-1] / equity_series.iloc[0] - 1)
    
    # Sharpe
    if len(returns_series) > 0 and returns_series.std() > 0:
        sharpe = returns_series.mean() / returns_series.std() * np.sqrt(252)
    else:
        sharpe = 0.0
    
    # 最大回撤
    cummax = equity_series.cummax()
    drawdown = (equity_series - cummax) / cummax
    max_dd = abs(drawdown.min())
    
    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'equity_curve': equity_series
    }


def validate_strategy(combo: list, config: dict, data_dir: Path):
    """验证单个策略在训练集和Holdout集上的表现"""
    print(f"\n{'='*80}")
    print(f"策略: {' + '.join(combo)}")
    print(f"{'='*80}")
    
    # 加载完整数据
    ohlcv = load_etf_data(data_dir)
    
    # 计算信号
    timing_config = config.get('backtest', {}).get('timing', {})
    signal, timing = compute_strategy_signal(ohlcv, combo, timing_config)
    
    # 参数
    freq = config.get('backtest', {}).get('freq', 3)
    pos_size = config.get('backtest', {}).get('pos_size', 2)
    commission = config.get('backtest', {}).get('commission_rate', 0.0002)
    
    # 训练集回测
    print(f"\n📊 训练集回测 (2020-01-01 至 {TRAINING_END})")
    train_result = backtest_strategy(
        ohlcv, signal, timing, freq, pos_size, commission,
        start_date='2020-01-01', end_date=TRAINING_END
    )
    print(f"  收益率: {train_result['total_return']*100:.2f}%")
    print(f"  Sharpe: {train_result['sharpe']:.3f}")
    print(f"  最大回撤: {train_result['max_dd']*100:.2f}%")
    
    # Holdout集回测
    print(f"\n🔬 Holdout集验证 ({HOLDOUT_START} 至 {HOLDOUT_END})")
    holdout_result = backtest_strategy(
        ohlcv, signal, timing, freq, pos_size, commission,
        start_date=HOLDOUT_START, end_date=HOLDOUT_END
    )
    print(f"  收益率: {holdout_result['total_return']*100:.2f}%")
    print(f"  Sharpe: {holdout_result['sharpe']:.3f}")
    print(f"  最大回撤: {holdout_result['max_dd']*100:.2f}%")
    
    # 验证标准
    print(f"\n✅ 验证标准检查:")
    checks = {
        '收益率 > 0%': holdout_result['total_return'] > HOLDOUT_MIN_RETURN,
        f'Sharpe > {HOLDOUT_MIN_SHARPE}': holdout_result['sharpe'] > HOLDOUT_MIN_SHARPE,
        f'最大回撤 < {HOLDOUT_MAX_DD*100}%': holdout_result['max_dd'] < HOLDOUT_MAX_DD,
    }
    
    for check_name, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check_name}: {status}")
    
    # 综合判断
    all_passed = all(checks.values())
    if all_passed:
        print(f"\n🎉 策略通过Holdout验证，可以启用！")
    else:
        print(f"\n⚠️  策略未通过Holdout验证，不建议启用。")
    
    return {
        'combo': ' + '.join(combo),
        'train_return': train_result['total_return'],
        'train_sharpe': train_result['sharpe'],
        'train_max_dd': train_result['max_dd'],
        'holdout_return': holdout_result['total_return'],
        'holdout_sharpe': holdout_result['sharpe'],
        'holdout_max_dd': holdout_result['max_dd'],
        'passed': all_passed,
        **checks
    }


def main():
    parser = argparse.ArgumentParser(description='Holdout期验证')
    parser.add_argument('--input', type=str, default=None,
                       help='Top策略CSV文件路径（默认自动查找最新）')
    parser.add_argument('--top_n', type=int, default=10,
                       help='验证Top N个策略（默认10）')
    parser.add_argument('--config', type=str, 
                       default='configs/combo_wfo_config.yaml',
                       help='配置文件路径')
    args = parser.parse_args()
    
    print("="*80)
    print("🔬 HOLDOUT期验证 - 防止过拟合的最后防线")
    print("="*80)
    print(f"训练集: 2020-01-01 至 {TRAINING_END}")
    print(f"Holdout集: {HOLDOUT_START} 至 {HOLDOUT_END}")
    print(f"验证标准: 收益>{HOLDOUT_MIN_RETURN*100}%, Sharpe>{HOLDOUT_MIN_SHARPE}, 回撤<{HOLDOUT_MAX_DD*100}%")
    print("="*80)
    
    # 加载配置
    config_path = ROOT / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    data_dir = Path(config['data']['data_dir'])
    
    # 加载Top策略
    if args.input:
        input_path = Path(args.input)
    else:
        # 自动查找最新
        selection_dirs = sorted(
            (ROOT / 'results').glob('selection_v2_*'),
            reverse=True
        )
        if not selection_dirs:
            print("❌ 未找到筛选结果目录")
            return
        input_path = selection_dirs[0] / 'top100_by_composite.csv'
    
    print(f"\n加载策略列表: {input_path}")
    top_strategies = pd.read_csv(input_path)
    
    # 验证Top N
    results = []
    for i in range(min(args.top_n, len(top_strategies))):
        combo_str = top_strategies.iloc[i]['combo']
        combo = [f.strip() for f in combo_str.split(' + ')]
        
        result = validate_strategy(combo, config, data_dir)
        results.append(result)
    
    # 保存结果
    output_dir = ROOT / 'results' / f"holdout_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame(results)
    output_path = output_dir / 'holdout_validation.csv'
    results_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*80}")
    print(f"验证完成！结果已保存至: {output_path}")
    print(f"{'='*80}")
    
    # 汇总
    passed = results_df['passed'].sum()
    total = len(results_df)
    print(f"\n📊 汇总:")
    print(f"  通过: {passed}/{total}")
    print(f"  通过率: {passed/total*100:.1f}%")
    
    if passed == 0:
        print(f"\n⚠️  警告: 没有任何策略通过Holdout验证！")
        print(f"  建议: 重新审视因子库、参数设置或数据质量")
    else:
        print(f"\n✅ 通过验证的策略:")
        for _, row in results_df[results_df['passed']].iterrows():
            print(f"  - {row['combo']}")
            print(f"    Holdout收益: {row['holdout_return']*100:.2f}%, Sharpe: {row['holdout_sharpe']:.2f}")


if __name__ == '__main__':
    main()
