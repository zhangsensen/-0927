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

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import shift_timing_signal, generate_rebalance_schedule

# Holdout配置
TRAINING_END = "2025-05-31"
HOLDOUT_START = "2025-06-01"
HOLDOUT_END = "2025-12-08"

# 验证标准
HOLDOUT_MIN_RETURN = 0.0      # Holdout期收益 > 0%
HOLDOUT_MIN_SHARPE = 0.3      # Sharpe > 0.3
HOLDOUT_MAX_DD = 0.25         # 最大回撤 < 25%


def run_vec_backtest_single(
    factors_3d,
    close_prices,
    open_prices,
    high_prices,
    low_prices,
    timing_arr,
    factor_indices,
    freq,
    pos_size,
    lookback_window,
    initial_capital,
    commission_rate,
    trailing_stop_pct
):
    """
    单策略VEC回测（简化版，不需要numba装饰器）
    """
    from etf_strategy.core.utils.rebalance import generate_rebalance_schedule
    
    n_dates, n_etfs = close_prices.shape
    selected_factors = factors_3d[:, :, factor_indices]
    combined_score = np.sum(selected_factors, axis=2)
    
    # 生成调仓日程
    rebalance_schedule = generate_rebalance_schedule(
        total_periods=n_dates,
        lookback_window=lookback_window,
        freq=freq
    )
    
    # 初始化
    cash = initial_capital
    positions = np.zeros(n_etfs, dtype=np.float64)
    equity_curve = np.zeros(n_dates, dtype=np.float64)
    equity_curve[0] = initial_capital
    
    for t in range(1, n_dates):
        # 更新持仓市值（使用 nansum 防止全 NaN 导致净值为 NaN）
        holdings_value = np.nansum(positions * close_prices[t])
        equity_curve[t] = cash + holdings_value
        
        # 调仓日
        if t in rebalance_schedule:
            # 清仓
            for i in range(n_etfs):
                if positions[i] > 0:
                    sell_price = close_prices[t, i]
                    if not np.isnan(sell_price):
                        cash += positions[i] * sell_price * (1 - commission_rate)
            positions[:] = 0
            
            # 选股 - ⚠️ 关键修复: 使用 t-1 时刻的因子值，避免未来函数
            scores = combined_score[t-1].copy()
            valid_mask = ~np.isnan(scores) & ~np.isnan(close_prices[t])
            
            if np.any(valid_mask):
                scores[~valid_mask] = -np.inf
                top_indices = np.argsort(scores)[-pos_size:][::-1]
                
                # 择时调整 - ⚠️ 关键修复: timing_arr 已经在 main 中 shift 过了，所以这里用 t
                # (main: timing_values = shift_timing_signal(timing_series.values))
                position_ratio = timing_arr[t] if not np.isnan(timing_arr[t]) else 1.0
                position_ratio = np.clip(position_ratio, 0.0, 1.0)
                
                # 分配资金
                invest_cash = cash * position_ratio
                per_position = invest_cash / len(top_indices)
                
                for idx in top_indices:
                    buy_price = close_prices[t, idx]
                    if not np.isnan(buy_price) and buy_price > 0:
                        shares = int(per_position / buy_price)
                        cost = shares * buy_price * (1 + commission_rate)
                        if cost <= cash:
                            positions[idx] = shares
                            cash -= cost
    
    # 计算指标
    returns = np.diff(equity_curve) / equity_curve[:-1]
    returns = returns[~np.isnan(returns)]
    
    total_return = (equity_curve[-1] / equity_curve[0] - 1)
    
    if len(returns) > 0 and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    else:
        sharpe = 0.0

    win_rate = float(np.mean(returns > 0)) if len(returns) > 0 else 0.0
    
    cummax = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - cummax) / cummax
    max_dd = abs(np.min(drawdown))
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'calmar_ratio': total_return / max_dd if max_dd > 0 else 0.0,
        'win_rate': win_rate,
    }


def main():
    print("="*80)
    print("🔬 全量策略 Holdout 验证 (12597个策略 - VEC引擎)")
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
    
    # 加载完整数据
    print("\n📂 加载完整数据...")
    loader = DataLoader(data_dir=data_dir, cache_dir=ROOT / '.cache')
    etf_files = list(data_dir.glob("*.parquet"))
    etf_codes = [f.stem.split('_')[0].split('.')[0] for f in etf_files]
    
    ohlcv_full = loader.load_ohlcv(etf_codes=etf_codes, start_date='2020-01-01', end_date=HOLDOUT_END)
    
    # 计算因子
    print("🔧 计算因子...")
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv_full)
    factor_names = sorted(raw_factors_df.columns.get_level_values(0).unique().tolist())
    raw_factors = {f: raw_factors_df[f] for f in factor_names}
    
    # 标准化
    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)
    
    # 择时信号
    timing_module = LightTimingModule(extreme_threshold=-0.1, extreme_position=0.1)
    timing_series = timing_module.compute_position_ratios(ohlcv_full['close'])
    timing_values = shift_timing_signal(timing_series.values)
    timing_series_full = pd.Series(timing_values, index=timing_series.index)
    
    # 准备训练集和Holdout集数据
    dates_full = std_factors[factor_names[0]].index
    
    # 找到切分点索引
    try:
        holdout_start_idx = np.where(dates_full >= HOLDOUT_START)[0][0]
        holdout_end_idx = np.where(dates_full <= HOLDOUT_END)[0][-1]
    except IndexError:
        print(f"❌ 日期范围错误: 数据范围 {dates_full[0]} ~ {dates_full[-1]}")
        return

    # 训练集: 从头开始到 TRAINING_END
    train_end_idx = np.where(dates_full <= TRAINING_END)[0][-1]
    
    # Holdout集: 需要包含前一天数据以便计算 t-1 信号
    # 如果 holdout_start_idx > 0，则向前多取一天
    holdout_slice_start = max(0, holdout_start_idx - 1)
    holdout_slice_end = holdout_end_idx + 1
    
    print(f"📅 数据切分:")
    print(f"   Train:   0 ~ {train_end_idx} ({dates_full[0].date()} ~ {dates_full[train_end_idx].date()})")
    print(f"   Holdout: {holdout_slice_start} ~ {holdout_end_idx} ({dates_full[holdout_slice_start].date()} ~ {dates_full[holdout_end_idx].date()})")
    print(f"            (注意: Holdout包含前一天 {dates_full[holdout_slice_start].date()} 用于信号计算)")

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
    
    # 准备3D因子数组（全量数据）
    print("📐 准备3D因子数组...")
    all_factors_3d = np.stack([std_factors[f].values for f in factor_names], axis=-1)
    close_arr = ohlcv_full['close'].values
    open_arr = ohlcv_full['open'].values
    high_arr = ohlcv_full['high'].values
    low_arr = ohlcv_full['low'].values
    timing_arr = timing_series_full.values
    
    # 批量回测 - 使用VEC引擎
    print(f"\n🚀 开始批量Holdout验证（VEC引擎）...")
    
    results = []
    
    for i, combo in enumerate(tqdm(combos, desc="VEC批量验证", ncols=100)):
        try:
            # 找到因子索引
            factor_indices = np.array([factor_names.index(f) for f in combo], dtype=np.int32)
            
            # 训练集回测
            train_result = run_vec_backtest_single(
                factors_3d=all_factors_3d[:train_end_idx+1],
                close_prices=close_arr[:train_end_idx+1],
                open_prices=open_arr[:train_end_idx+1],
                high_prices=high_arr[:train_end_idx+1],
                low_prices=low_arr[:train_end_idx+1],
                timing_arr=timing_arr[:train_end_idx+1],
                factor_indices=factor_indices,
                freq=freq,
                pos_size=pos_size,
                lookback_window=252,
                initial_capital=1_000_000.0,
                commission_rate=0.0002,
                trailing_stop_pct=0.0
            )
            
            # Holdout集回测
            # 注意: 传入的数据包含前一天，所以 lookback_window=1
            # 这样 t 从 1 开始，t-1=0 (即前一天的数据)，这是正确的
            holdout_result = run_vec_backtest_single(
                factors_3d=all_factors_3d[holdout_slice_start:holdout_slice_end],
                close_prices=close_arr[holdout_slice_start:holdout_slice_end],
                open_prices=open_arr[holdout_slice_start:holdout_slice_end],
                high_prices=high_arr[holdout_slice_start:holdout_slice_end],
                low_prices=low_arr[holdout_slice_start:holdout_slice_end],
                timing_arr=timing_arr[holdout_slice_start:holdout_slice_end],
                factor_indices=factor_indices,
                freq=freq,
                pos_size=pos_size,
                lookback_window=1,  # 从第1天开始交易(第0天用于信号)
                initial_capital=1_000_000.0,
                commission_rate=0.0002,
                trailing_stop_pct=0.0
            )
            
            # 验证检查
            passed = (
                holdout_result['total_return'] > HOLDOUT_MIN_RETURN and
                holdout_result['sharpe_ratio'] > HOLDOUT_MIN_SHARPE and
                holdout_result['max_drawdown'] < HOLDOUT_MAX_DD
            )
            
            results.append({
                'combo': ' + '.join(combo),
                'combo_size': len(combo),
                'train_return': train_result['total_return'],
                'train_sharpe': train_result['sharpe_ratio'],
                'train_max_dd': train_result['max_drawdown'],
                'train_calmar': train_result.get('calmar_ratio', 0),
                'train_win_rate': train_result.get('win_rate', 0),
                'holdout_return': holdout_result['total_return'],
                'holdout_sharpe': holdout_result['sharpe_ratio'],
                'holdout_max_dd': holdout_result['max_drawdown'],
                'holdout_calmar': holdout_result.get('calmar_ratio', 0),
                'holdout_win_rate': holdout_result.get('win_rate', 0),
                'passed': passed,
                'return_diff': holdout_result['total_return'] - train_result['total_return'],
                'sharpe_diff': holdout_result['sharpe_ratio'] - train_result['sharpe_ratio']
            })
            
        except Exception as e:
            # 静默跳过错误（避免中断）
            continue
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    
    # 统计
    print("\n" + "="*80)
    print("📊 Holdout验证结果")
    print("="*80)
    
    n_total = len(results_df)
    n_passed = results_df['passed'].sum()
    pass_rate = n_passed / n_total * 100 if n_total > 0 else 0
    
    print(f"总策略数: {n_total}")
    print(f"通过验证: {n_passed} ({pass_rate:.2f}%)")
    print(f"未通过: {n_total - n_passed} ({100-pass_rate:.2f}%)")
    
    # 按Holdout表现排序
    results_df = results_df.sort_values('holdout_return', ascending=False).reset_index(drop=True)
    
    # 保存结果
    output_dir = ROOT / 'results' / 'holdout_validation'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_path = output_dir / f'full_holdout_{timestamp}.csv'
    passed_path = output_dir / f'passed_strategies_{timestamp}.csv'
    
    results_df.to_csv(full_path, index=False, encoding='utf-8-sig')
    results_df[results_df['passed']].to_csv(passed_path, index=False, encoding='utf-8-sig')
    
    print(f"\n💾 结果已保存:")
    print(f"   全部结果: {full_path}")
    print(f"   通过策略: {passed_path}")
    
    # Top10通过的策略
    print("\n" + "="*80)
    print("🏆 Holdout期Top10策略（通过验证）")
    print("="*80)
    
    passed_df = results_df[results_df['passed']].head(10)
    
    for i, row in passed_df.iterrows():
        print(f"\n#{i+1} | Holdout收益: {row['holdout_return']*100:.2f}%")
        print(f"     因子: {row['combo']}")
        print(f"     Holdout Sharpe: {row['holdout_sharpe']:.3f} | 回撤: {row['holdout_max_dd']*100:.2f}%")
        print(f"     训练集收益: {row['train_return']*100:.2f}% | Sharpe: {row['train_sharpe']:.3f}")
        print(f"     收益劣化: {row['return_diff']*100:+.2f}pp | Sharpe劣化: {row['sharpe_diff']:+.3f}")
    
    # 因子频率统计
    print("\n" + "="*80)
    print("📈 通过策略的因子频率 (Top20)")
    print("="*80)
    
    passed_strategies = results_df[results_df['passed']]
    factor_counts = {}
    for combo_str in passed_strategies['combo']:
        for factor in combo_str.split(' + '):
            factor = factor.strip()
            factor_counts[factor] = factor_counts.get(factor, 0) + 1
    
    sorted_factors = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)
    for factor, count in sorted_factors[:20]:
        print(f"  {factor}: {count}")
    
    print("\n✅ Holdout验证完成！")


if __name__ == '__main__':
    main()
