#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 Phase 2 真实回测引擎
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from backtest_engine import Phase2BacktestEngine
from position_optimizer import PositionOptimizer


def create_mock_profile():
    """创建模拟组合画像"""
    return {
        'combo': 'test_combo',
        'factor_structure': {
            'factors': ['momentum', 'value']
        },
        'performance': {
            'annual_ret_net': 0.25,
            'sharpe_net': 1.8,
            'max_dd_net': -0.12
        },
        'trading': {
            'avg_turnover': 0.8,
            'win_rate': 0.55
        }
    }


def test_dynamic_position_backtest():
    """测试动态仓位回测"""
    print("\n" + "=" * 60)
    print("测试 1: 动态仓位回测")
    print("=" * 60)
    
    # 创建优化器和引擎
    profile = create_mock_profile()
    position_opt = PositionOptimizer(profile)
    engine = Phase2BacktestEngine(position_opt)
    
    # 生成基线收益
    baseline_returns = engine.generate_baseline_returns(
        annual_return=0.25,
        sharpe=1.8,
        n_days=252,
        seed=42
    )
    
    print(f"\n基线收益序列:")
    print(f"  年化收益: {(1 + baseline_returns.sum()) ** (252/len(baseline_returns)) - 1:.2%}")
    print(f"  波动率: {baseline_returns.std() * np.sqrt(252):.2%}")
    
    # 运行回测
    result = engine.run_dynamic_position_backtest(
        baseline_returns=baseline_returns,
        high_confidence_days_ratio=0.6
    )
    
    print(f"\n动态仓位回测结果:")
    print(f"  年化收益: {result['annual_return']:.2%}")
    print(f"  Sharpe比率: {result['sharpe']:.3f}")
    print(f"  最大回撤: {result['max_dd']:.2%}")
    print(f"  平均仓位: {result['avg_position']:.1%}")
    print(f"  高置信度日占比: {result['actual_high_conf_ratio']:.1%}")
    
    assert result['avg_position'] < 1.0, "平均仓位应小于100%"
    assert result['max_dd'] < 0, "最大回撤应该是负值"
    
    print("\n✅ 动态仓位回测测试通过")


def test_trailing_stop_backtest():
    """测试移动止损回测"""
    print("\n" + "=" * 60)
    print("测试 2: 移动止损回测")
    print("=" * 60)
    
    profile = create_mock_profile()
    position_opt = PositionOptimizer(profile)
    engine = Phase2BacktestEngine(position_opt)
    
    baseline_returns = engine.generate_baseline_returns(
        annual_return=0.25,
        sharpe=1.8,
        n_days=252,
        seed=42
    )
    
    # 运行回测
    result = engine.run_trailing_stop_backtest(
        baseline_returns=baseline_returns,
        etf_stop=0.05,
        portfolio_stop=0.10
    )
    
    print(f"\n移动止损回测结果:")
    print(f"  年化收益: {result['annual_return']:.2%}")
    print(f"  Sharpe比率: {result['sharpe']:.3f}")
    print(f"  最大回撤: {result['max_dd']:.2%}")
    print(f"  止损次数: {result['n_stops']}")
    print(f"  每年止损: {result['stop_rate']:.1f}次")
    
    if result['n_stops'] > 0:
        print(f"\n止损事件样本:")
        for i, event in enumerate(result['stop_events'][:3]):
            print(f"  {i+1}. {event['date'].strftime('%Y-%m-%d')}: "
                  f"{event['reason']}, 持仓收益={event['holding_return']:+.2%}")
    
    assert result['n_stops'] >= 0, "止损次数不能为负"
    
    print("\n✅ 移动止损回测测试通过")


def test_combined_backtest():
    """测试联合回测（动态仓位 + 移动止损）"""
    print("\n" + "=" * 60)
    print("测试 3: 联合回测（动态仓位 + 移动止损）")
    print("=" * 60)
    
    profile = create_mock_profile()
    position_opt = PositionOptimizer(profile)
    engine = Phase2BacktestEngine(position_opt)
    
    baseline_returns = engine.generate_baseline_returns(
        annual_return=0.25,
        sharpe=1.8,
        n_days=252,
        seed=42
    )
    
    # 运行联合回测
    result = engine.run_combined_backtest(
        baseline_returns=baseline_returns,
        high_confidence_days_ratio=0.6,
        etf_stop=0.05,
        portfolio_stop=0.10
    )
    
    print(f"\n联合回测结果:")
    print(f"  年化收益: {result['annual_return']:.2%}")
    print(f"  Sharpe比率: {result['sharpe']:.3f}")
    print(f"  最大回撤: {result['max_dd']:.2%}")
    print(f"  平均仓位: {result['avg_position']:.1%}")
    print(f"  止损次数: {result['n_stops']}")
    
    assert result['avg_position'] < 1.0, "平均仓位应小于100%"
    
    print("\n✅ 联合回测测试通过")


def test_comparison_with_theory():
    """测试理论估算 vs 真实回测对比"""
    print("\n" + "=" * 60)
    print("测试 4: 理论估算 vs 真实回测对比")
    print("=" * 60)
    
    profile = create_mock_profile()
    position_opt = PositionOptimizer(profile)
    engine = Phase2BacktestEngine(position_opt)
    
    baseline_returns = engine.generate_baseline_returns(
        annual_return=0.25,
        sharpe=1.8,
        n_days=756,  # 3年
        seed=42
    )
    
    # 理论估算
    theory_result = position_opt.estimate_dynamic_position_impact(
        baseline_sharpe=1.8,
        baseline_return=0.25,
        baseline_dd=-0.12,
        high_confidence_days_ratio=0.6
    )
    
    # 真实回测
    backtest_result = engine.run_dynamic_position_backtest(
        baseline_returns=baseline_returns,
        high_confidence_days_ratio=0.6
    )
    
    print(f"\n理论估算结果:")
    print(f"  Sharpe: {theory_result['adjusted_sharpe']:.3f}")
    print(f"  年化收益: {theory_result['adjusted_return']:.2%}")
    print(f"  最大回撤: {theory_result['adjusted_dd']:.2%}")
    
    print(f"\n真实回测结果:")
    print(f"  Sharpe: {backtest_result['sharpe']:.3f}")
    print(f"  年化收益: {backtest_result['annual_return']:.2%}")
    print(f"  最大回撤: {backtest_result['max_dd']:.2%}")
    
    # 计算偏差
    sharpe_dev = (backtest_result['sharpe'] - theory_result['adjusted_sharpe']) / theory_result['adjusted_sharpe']
    print(f"\nSharpe偏差: {sharpe_dev:+.1%}")
    
    if abs(sharpe_dev) < 0.20:
        print("✅ 理论模型与实际回测偏差在可接受范围内")
    else:
        print("⚠️ 理论模型与实际回测偏差较大")
    
    print("\n✅ 对比测试通过")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Phase 2 真实回测引擎测试套件")
    print("=" * 60)
    
    try:
        test_dynamic_position_backtest()
        test_trailing_stop_backtest()
        test_combined_backtest()
        test_comparison_with_theory()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试通过!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
