#!/usr/bin/env python
"""
计算策略的 Sortino Ratio 和增强指标

Sortino Ratio = (收益 - 无风险利率) / 下行标准差
只惩罚负收益的波动，比 Sharpe 更适合评估策略质量

用法:
    uv run python scripts/calculate_sortino.py

输出:
    results/golden_analysis_*/enhanced_metrics.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from etf_strategy.core.data_loader import load_etf_data
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor


def calculate_sortino_ratio(daily_returns: np.ndarray, target_return: float = 0.0) -> float:
    """
    计算 Sortino Ratio
    
    Args:
        daily_returns: 日收益率序列
        target_return: 目标收益率（默认 0）
    
    Returns:
        Sortino Ratio (年化)
    """
    excess_returns = daily_returns - target_return
    downside_returns = np.minimum(excess_returns, 0)
    downside_std = np.std(downside_returns)
    
    if downside_std == 0:
        return np.inf if np.mean(excess_returns) > 0 else 0
    
    # 年化
    annual_return = np.mean(daily_returns) * 252
    annual_downside_std = downside_std * np.sqrt(252)
    
    return annual_return / annual_downside_std


def calculate_max_drawdown_duration(equity_curve: np.ndarray) -> int:
    """
    计算最大回撤持续天数
    
    Args:
        equity_curve: 权益曲线
    
    Returns:
        最大回撤持续天数
    """
    cummax = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - cummax) / cummax
    
    # 找到最大回撤的起点和终点
    max_dd_idx = np.argmin(drawdown)
    peak_idx = np.argmax(equity_curve[:max_dd_idx + 1])
    
    # 找到恢复点（如果存在）
    recovery_idx = None
    for i in range(max_dd_idx, len(equity_curve)):
        if equity_curve[i] >= cummax[peak_idx]:
            recovery_idx = i
            break
    
    if recovery_idx is None:
        recovery_idx = len(equity_curve) - 1
    
    return recovery_idx - peak_idx


def calculate_monthly_win_rate(daily_returns: np.ndarray, dates: pd.DatetimeIndex) -> float:
    """
    计算月度胜率
    
    Args:
        daily_returns: 日收益率序列
        dates: 日期索引
    
    Returns:
        月度胜率 (0-1)
    """
    df = pd.DataFrame({'return': daily_returns}, index=dates)
    monthly_returns = df.resample('M')['return'].sum()
    
    if len(monthly_returns) == 0:
        return 0.5
    
    return (monthly_returns > 0).mean()


def run_enhanced_backtest(combo: str, config: dict) -> dict:
    """
    运行增强回测，计算 Sortino 等指标
    
    Args:
        combo: 因子组合字符串
        config: 配置字典
    
    Returns:
        包含增强指标的字典
    """
    # 加载数据
    ohlcv = load_etf_data(
        config['symbols'],
        config['start_date'],
        config['end_date'],
        config['data_path']
    )
    
    # 计算因子
    factors = combo.split(' + ')
    factor_lib = PreciseFactorLibrary()
    factor_values = factor_lib.compute_factors(ohlcv, factors)
    
    # 横截面处理
    processor = CrossSectionProcessor(
        freq=config['freq'],
        pos_size=config['pos_size']
    )
    
    # 获取信号和日收益
    signals, daily_returns, equity_curve = processor.generate_signals_with_returns(
        factor_values, ohlcv
    )
    
    # 计算增强指标
    dates = ohlcv.index
    
    return {
        'combo': combo,
        'sortino_ratio': calculate_sortino_ratio(daily_returns),
        'max_dd_duration': calculate_max_drawdown_duration(equity_curve),
        'monthly_win_rate': calculate_monthly_win_rate(daily_returns, dates),
        'daily_returns_std': np.std(daily_returns) * np.sqrt(252),  # 年化波动率
        'skewness': pd.Series(daily_returns).skew(),
        'kurtosis': pd.Series(daily_returns).kurtosis(),
    }


def main():
    """主函数"""
    import yaml
    
    # 加载配置
    config_path = Path('configs/combo_wfo_config.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 加载策略列表
    strategies_path = Path('results/golden_analysis_20251202_001109/optimized_30_strategies.csv')
    if not strategies_path.exists():
        strategies_path = Path('results/golden_analysis_20251202_001109/final_30_strategies.csv')
    
    strategies = pd.read_csv(strategies_path)
    print(f"加载 {len(strategies)} 个策略")
    
    # 配置参数
    backtest_config = {
        'symbols': config['symbols'],
        'start_date': config['backtest']['start_date'],
        'end_date': config['backtest']['end_date'],
        'data_path': config['data_path'],
        'freq': config['strategy']['rebalance_freq'],
        'pos_size': config['strategy']['position_size'],
    }
    
    # 运行增强回测
    results = []
    for i, row in strategies.iterrows():
        combo = row['combo']
        print(f"[{i+1}/{len(strategies)}] {combo[:60]}...")
        
        try:
            metrics = run_enhanced_backtest(combo, backtest_config)
            results.append(metrics)
        except Exception as e:
            print(f"  错误: {e}")
            results.append({
                'combo': combo,
                'sortino_ratio': np.nan,
                'max_dd_duration': np.nan,
                'monthly_win_rate': np.nan,
            })
    
    # 保存结果
    df_results = pd.DataFrame(results)
    
    # 合并原有指标
    df_merged = strategies.merge(df_results, on='combo', how='left')
    
    output_path = Path('results/golden_analysis_20251202_001109/enhanced_metrics.csv')
    df_merged.to_csv(output_path, index=False)
    print(f"\n✅ 增强指标已保存到 {output_path}")
    
    # 打印统计
    print("\n=== Sortino Ratio 统计 ===")
    print(f"平均: {df_results['sortino_ratio'].mean():.2f}")
    print(f"最大: {df_results['sortino_ratio'].max():.2f}")
    print(f"最小: {df_results['sortino_ratio'].min():.2f}")


if __name__ == '__main__':
    main()
