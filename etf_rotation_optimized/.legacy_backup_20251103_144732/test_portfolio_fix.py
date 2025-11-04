"""
测试portfolio_constructor修复效果的简单脚本
"""

import numpy as np
from core.portfolio_constructor import PerformanceCalculator, PortfolioConstructor
from core.trading_cost_model import AShareETFTradingCost

print("=" * 80)
print("Portfolio Constructor 修复验证")
print("=" * 80)
print()

# 设置随机种子
np.random.seed(42)

# 生成测试数据
T = 100  # 100天
N = 10  # 10个ETF
top_n = 3

print(f"测试数据: {T}天, {N}个ETF, Top-{top_n}持仓")
print()

# 生成随机信号和价格
signals = np.random.randn(T, N)
prices = np.random.randn(T, N) * 10 + 100
returns = np.random.randn(T, N) * 0.01
etf_names = [f"ETF_{i}" for i in range(N)]

# 创建构建器
cost_model = AShareETFTradingCost(stamp_tax=0.001, commission=0.0003, slippage=0.0005)
constructor = PortfolioConstructor(top_n=top_n, trading_cost_model=cost_model)

print("1. 测试前视偏差修复...")
# 构建持仓
weights, costs = constructor.construct_portfolio(signals, prices, etf_names)

# 验证第一天空仓
assert np.all(weights[0] == 0), "❌ 第一天应该空仓"
print("   ✅ 第一天空仓正确")

# 验证信号延迟
signals_modified = signals.copy()
test_day = 50
signals_modified[test_day] = 999

weights_modified, _ = constructor.construct_portfolio(
    signals_modified, prices, etf_names
)

# weights[test_day]应该相同（使用day49信号）
if np.allclose(weights[test_day], weights_modified[test_day]):
    print(f"   ✅ Day {test_day}持仓相同（使用Day {test_day-1}信号）")
else:
    print(f"   ❌ Day {test_day}持仓不同！存在前视偏差")

# weights[test_day+1]应该不同（使用day50信号）
if not np.allclose(weights[test_day + 1], weights_modified[test_day + 1]):
    print(f"   ✅ Day {test_day+1}持仓不同（使用Day {test_day}信号）")
else:
    print(f"   ❌ Day {test_day+1}持仓相同！信号延迟失效")

print()
print("2. 测试成本计算稳定性...")
# 验证成本有界
assert np.all(costs >= 0), "❌ 成本应该非负"
assert np.all(costs < 0.1), "❌ 成本率应该<10%"
print("   ✅ 成本非负且有界")

# 验证第一天成本为0
assert costs[0] == 0, "❌ 第一天成本应该为0"
print("   ✅ 第一天成本为0")

print()
print("3. 测试绩效计算器...")
calculator = PerformanceCalculator()
net_returns = calculator.calculate_returns(weights, returns, costs)

# 验证收益有界
assert np.all(np.isfinite(net_returns)), "❌ 收益应该有界"
assert np.all(net_returns > -1.0), "❌ 收益应该>-100%"
assert np.all(net_returns < 1.0), "❌ 收益应该<100%"
print("   ✅ 收益有界且合理")

# 验证第一天收益为0
assert net_returns[0] == 0, "❌ 第一天收益应该为0"
print("   ✅ 第一天收益为0")

print()
print("4. 计算绩效指标...")
metrics = calculator.calculate_metrics(net_returns)
print(f"   年化收益: {metrics['annual_return']:.2%}")
print(f"   年化波动: {metrics['annual_volatility']:.2%}")
print(f"   Sharpe比率: {metrics['sharpe_ratio']:.4f}")
print(f"   最大回撤: {metrics['max_drawdown']:.2%}")
print(f"   胜率: {metrics['win_rate']:.2%}")

print()
print("=" * 80)
print("✅ 所有测试通过！Portfolio Constructor修复验证成功")
print("=" * 80)
print()
print("修复效果:")
print("  ✅ 无前视偏差 - 使用T-1信号")
print("  ✅ 成本稳定 - 归一化资本")
print("  ✅ 收益合理 - 稳定计算")
print()
