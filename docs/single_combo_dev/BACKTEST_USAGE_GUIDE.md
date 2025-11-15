<!-- ALLOW-MD -->
# Phase 2 真实回测系统使用指南

## 概述

本系统在理论估算基础上，增加了基于逐日路径的真实回测实现，支持**双轨验证**：

1. **理论估算**：基于参数敏感性的快速估算（无需逐日数据）
2. **真实回测**：基于逐日价格路径的完整回测（模拟信号分布）

通过对比两种方法，可以评估理论模型的准确性和假设的合理性。

---

## 新增模块

### 1. `backtest_engine.py` - 回测引擎

**核心类**: `Phase2BacktestEngine`

**主要方法**:

- `run_dynamic_position_backtest()`: 动态仓位真实回测
- `run_trailing_stop_backtest()`: 移动止损真实回测
- `run_combined_backtest()`: 联合回测（动态仓位 + 止损）
- `generate_baseline_returns()`: 生成基线日收益序列（用于模拟）

**设计特点**:

- 逐日循环，记录完整路径
- 模拟信号分布（根据高置信度日占比）
- 输出与理论估算一致的指标格式

---

### 2. `experiment_runner.py` - 扩展的实验执行器

**新增方法**:

- `run_experiment_2_1_with_backtest()`: 实验2.1（理论 + 回测）
- `run_experiment_2_2_with_backtest()`: 实验2.2（理论 + 回测）
- `run_all_phase2_experiments_with_backtest()`: 完整Phase 2双轨实验
- `_generate_phase2_comparison_report()`: 生成对比报告

**报告输出**:

- `phase2_comparison_report.md`: 理论 vs 实际对比报告
- `exp_2_1_dynamic_position_theory.csv`: 理论估算结果
- `exp_2_1_dynamic_position_backtest.csv`: 真实回测结果
- `exp_2_2_trailing_stop_theory.csv`: 理论估算结果
- `exp_2_2_trailing_stop_backtest.csv`: 真实回测结果

---

## 使用方法

### 方式1: 命令行运行（推荐）

```bash
# 仅运行理论估算（Phase 2）
python single_combo_dev/experiment_runner.py --phase 2

# 运行理论估算 + 真实回测（双轨验证）
python single_combo_dev/experiment_runner.py --phase 2 --backtest
```

### 方式2: 代码调用

```python
from experiment_runner import SingleComboDeveloper
from selection import analyze_single_combo
import pandas as pd

# 加载组合画像
df = pd.read_csv('selection/top200_selected_test.csv')
profile = analyze_single_combo(df, rank=1)

# 创建开发器
developer = SingleComboDeveloper(
    combo_profile=profile,
    output_dir='single_combo_dev/experiments/rank1'
)

# 运行双轨实验
results = developer.run_all_phase2_experiments_with_backtest()

# 查看结果
print(results['exp_2_1_theory'])    # 动态仓位理论估算
print(results['exp_2_1_backtest'])  # 动态仓位真实回测
print(results['exp_2_2_theory'])    # 移动止损理论估算
print(results['exp_2_2_backtest'])  # 移动止损真实回测
```

### 方式3: 单独调用回测引擎

```python
from backtest_engine import Phase2BacktestEngine
from position_optimizer import PositionOptimizer
import pandas as pd

# 创建优化器和引擎
profile = {...}  # 组合画像
position_opt = PositionOptimizer(profile)
engine = Phase2BacktestEngine(position_opt)

# 生成基线收益（或使用真实历史收益）
baseline_returns = engine.generate_baseline_returns(
    annual_return=0.25,
    sharpe=1.8,
    n_days=756,  # 3年
    seed=42
)

# 运行动态仓位回测
result = engine.run_dynamic_position_backtest(
    baseline_returns=baseline_returns,
    high_confidence_days_ratio=0.6
)

print(f"年化收益: {result['annual_return']:.2%}")
print(f"Sharpe比率: {result['sharpe']:.3f}")
print(f"最大回撤: {result['max_dd']:.2%}")
print(f"平均仓位: {result['avg_position']:.1%}")
```

---

## 测试验证

运行测试套件验证功能：

```bash
python single_combo_dev/test_backtest_engine.py
```

**测试内容**:
1. 动态仓位回测功能
2. 移动止损回测功能
3. 联合回测功能
4. 理论估算 vs 真实回测对比

**预期输出**:
```
✅ 动态仓位回测测试通过
✅ 移动止损回测测试通过
✅ 联合回测测试通过
✅ 对比测试通过
✅ 所有测试通过!
```

---

## 输出报告解读

### `phase2_comparison_report.md` 结构

1. **报告说明**：双轨验证方法介绍
2. **基线性能**：组合原始指标
3. **实验 2.1: 动态仓位映射**
   - 理论估算结果
   - 真实回测结果
   - 偏差分析（Sharpe偏差、回撤偏差）
4. **实验 2.2: 移动止损**
   - 理论估算结果
   - 真实回测结果
   - 偏差分析
5. **综合评估**
   - 模型准确性总结
   - 实施建议
   - 下一步工作

### 偏差评估标准

- **Sharpe偏差 < 10%**: ✅ 理论模型与实际回测吻合良好
- **Sharpe偏差 10-20%**: ⚠️ 存在一定偏差
- **Sharpe偏差 > 20%**: ❌ 偏差较大，需要修正假设

---

## 当前限制与改进方向

### 1. 信号分布模拟

**当前方法**:
- 根据`high_confidence_days_ratio`随机生成高/低置信度信号
- 高置信度：signal_strength 和 consistency_ratio ∈ [0.7, 1.0]
- 低置信度：signal_strength 和 consistency_ratio ∈ [0.0, 0.5]

**局限性**:
- 无法反映真实因子的时序特征
- 无法模拟信号与未来收益的真实关系

**改进方向**:
- 使用真实历史因子数据
- 根据因子IC统计特征生成更真实的信号分布

### 2. 基线收益生成

**当前方法**:
- 根据年化收益和Sharpe比率生成正态分布的日收益序列

**局限性**:
- 无法反映真实收益的非正态特征（厚尾、偏度）
- 无法模拟市场状态切换（牛市/熊市）

**改进方向**:
- 使用真实历史ETF价格数据
- 使用GARCH等模型生成更真实的收益序列

### 3. 回测框架扩展

**待增强功能**:
- [ ] 多ETF组合回测（当前仅支持单ETF简化版）
- [ ] 交易成本建模（滑点、手续费）
- [ ] 调仓日逻辑（当前每日都可调仓）
- [ ] 仓位约束（最小交易单位、整手要求）

---

## FAQ

### Q1: 为什么理论估算和真实回测结果有偏差？

**A**: 主要原因：
1. **信号分布假设**：理论模型假设高置信度日和低置信度日的收益差异固定，实际回测中信号与收益的关系更复杂
2. **路径依赖**：理论模型基于统计平均，真实回测存在路径依赖（如止损触发的时机）
3. **非线性效应**：动态仓位和止损的交互效应在理论模型中被简化

### Q2: 如何使用真实历史数据进行回测？

**A**: 步骤：
1. 准备逐日ETF价格数据（DataFrame, index=日期）
2. 准备逐日因子信号数据（signal_strength, consistency_ratio）
3. 调用回测引擎时传入真实数据：
   ```python
   result = engine.run_dynamic_position_backtest(
       baseline_returns=real_etf_returns,  # 真实收益
       high_confidence_days_ratio=0.6      # 可选，用于计算置信度
   )
   ```
4. 如果有真实信号数据，可以直接修改回测引擎的信号生成部分

### Q3: 如何调整回测参数？

**A**: 主要参数：
- `high_confidence_days_ratio`: 高置信度日占比（0.3-0.8）
- `etf_stop / portfolio_stop`: 止损阈值（3%-12%）
- `position_levels`: 仓位映射规则（默认[(0.5,0.5), (0.7,0.7), (0.9,1.0)]）
- `cooldown_days`: 止损后冷却期（默认5天）

### Q4: 如何解读对比报告？

**A**: 关键指标：
- **Sharpe偏差**: 理论模型预测准确性的核心指标
- **回撤偏差**: 风险控制效果的验证
- **准确性评估**: 根据偏差大小给出的综合判断
- **实施建议**: 根据真实回测结果给出的参数推荐

---

## 技术细节

### 动态仓位回测逻辑

```python
# 1. 生成信号分布（高置信度 vs 低置信度）
# 2. 逐日循环：
for each_day:
    # 计算置信度
    confidence = min(signal_strength, consistency_ratio)
    
    # 映射到仓位
    if confidence >= 0.9:
        position = high_position
    elif confidence >= 0.7:
        position = mid_position
    else:
        position = low_position
    
    # 计算实际收益
    actual_return = baseline_return * position
    
# 3. 计算回测指标（年化收益、Sharpe、最大回撤）
```

### 移动止损回测逻辑

```python
# 初始化状态
is_holding = True
buy_price = 1.0
cooldown_days = 0

# 逐日循环
for each_day:
    if cooldown_days > 0:
        # 冷却期，不持仓
        actual_return = 0
        cooldown_days -= 1
    
    elif is_holding:
        # 更新持仓收益
        holding_return = current_price / buy_price - 1
        
        # 检查止损
        if holding_return <= -etf_stop or holding_return <= -portfolio_stop:
            # 触发止损，平仓
            is_holding = False
            cooldown_days = 5
        
        actual_return = baseline_return
```

---

## 版本历史

**v1.0** (当前版本)
- ✅ 动态仓位真实回测
- ✅ 移动止损真实回测
- ✅ 联合回测
- ✅ 理论 vs 实际对比报告
- ✅ 基线收益生成器
- ✅ 信号分布模拟

**待开发功能**:
- 多ETF组合回测
- 真实因子数据接入
- 交易成本建模
- 调仓日逻辑

---

## 联系与反馈

如有问题或建议，请查看：
- 技术文档：`PHASE2_ENHANCEMENT_REPORT.md`
- 代码注释：`position_optimizer.py`
- 测试用例：`test_backtest_engine.py`
