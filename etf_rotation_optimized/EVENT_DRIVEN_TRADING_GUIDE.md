# 事件驱动交易系统 - A股ETF专用

**创建时间**: 2025-11-03  
**模式**: 事件驱动（每日评估，有信号就交易）  
**约束**: A股T+1交易规则

---

## 核心特性

### 1. 事件驱动模式 ✅

```
传统模式（固定周期）:
  - 每20天调仓一次
  - 错过中间的交易机会
  - 无法及时止损

事件驱动模式（本系统）:
  - 每天评估信号
  - 有强信号就交易
  - 像交易股票一样灵活
```

### 2. A股T+1约束 ✅

```python
# T+1规则
今天买入 → 明天才能卖出

# 实现
if last_buy_day[i] == current_day:
    # 强制保持持仓（不能卖）
    new_weights[i] = max(new_weights[i], current_weights[i])
```

### 3. 最小持有期 ✅

```python
# 避免频繁交易
min_holding_days = 3  # 至少持有3天

# 实现
if holding_days[i] < min_holding_days:
    # 强制保持持仓
    new_weights[i] = max(new_weights[i], current_weights[i])
```

### 4. 每日换手限制 ✅

```python
# 控制交易频率
max_daily_turnover = 0.5  # 每天最多换手50%

# 优先级
1. 强信号买入（优先执行）
2. 弱信号卖出（次优先）
```

### 5. 信号质量过滤 ✅

```python
# 只在信号强时交易
signal_strength_threshold = 0.0  # Z-score阈值

# 计算
z_score = (signal - mean) / std
if z_score > threshold:
    # 信号足够强，可以交易
```

---

## 配置参数

### configs/default.yaml

```yaml
backtest:
  # 持仓配置
  top_n: 6  # 持仓6只ETF
  
  # 事件驱动参数
  min_holding_days: 3  # 最小持有3天
  max_daily_turnover: 0.5  # 每日最多换手50%
  signal_strength_threshold: 0.0  # 信号Z-score阈值
  
  # 交易成本
  commission_rate: 0.0003  # 万3佣金
  stamp_tax_rate: 0.0  # ETF免印花税
  slippage_bps: 5.0  # 5bp滑点
```

---

## 交易逻辑

### 每日流程

```
Day T:
  1. 评估信号（使用T-1数据）
  2. 信号质量过滤（Z-score > threshold）
  3. 选择Top-N ETF
  4. 应用T+1约束（今天买的不能今天卖）
  5. 应用最小持有期（持有<3天不能卖）
  6. 应用换手限制（每日最多50%）
  7. 计算交易成本
  8. 更新持仓
```

### 约束优先级

```
1. T+1约束（最高优先级）
   - 今天买的必须明天才能卖
   
2. 最小持有期
   - 持有不足N天的不能卖
   
3. 换手限制
   - 按信号强度优先级分配换手额度
```

---

## 与固定周期对比

### 固定周期（20天调仓）

```
优点:
  - 交易成本低
  - 实现简单

缺点:
  - 错过中间机会
  - 无法及时止损
  - 灵活性差
```

### 事件驱动（每日评估）

```
优点:
  - 及时捕捉机会
  - 灵活止损
  - 像交易股票

缺点:
  - 交易成本可能更高
  - 需要约束控制
```

### 本系统（事件驱动+约束）

```
平衡:
  - 每日评估（灵活）
  - T+1约束（合规）
  - 最小持有期（控制成本）
  - 换手限制（控制频率）
  - 信号过滤（只交易强信号）
```

---

## 使用方法

### 1. 运行WFO（使用事件驱动）

```bash
cd /Users/zhangshenshen/深度量化0927/etf_rotation_optimized

# 清除缓存
rm -rf cache/ .cache/

# 运行完整流程
./run_full_wfo_real_data.sh
```

### 2. 查看结果

```bash
# WFO结果
cat results/wfo/*/wfo_summary.csv

# 查看交易统计
# - trade_count: 交易次数
# - signal_triggered_days: 信号触发天数
# - avg_turnover: 平均换手率
# - trade_frequency: 交易频率
```

### 3. 调整参数

```yaml
# 更激进（更多交易）
min_holding_days: 1
max_daily_turnover: 0.8
signal_strength_threshold: -0.5

# 更保守（更少交易）
min_holding_days: 5
max_daily_turnover: 0.3
signal_strength_threshold: 0.5
```

---

## 关键代码

### 事件驱动构建器

```python
from core.event_driven_portfolio_constructor import EventDrivenPortfolioConstructor

constructor = EventDrivenPortfolioConstructor(
    top_n=6,
    min_holding_days=3,
    max_daily_turnover=0.5,
    signal_strength_threshold=0.0,
    trading_cost_model=cost_model
)

weights, costs, stats = constructor.construct_portfolio(
    factor_signals=signals,
    etf_prices=prices,
    etf_names=names
)
```

### 统计信息

```python
stats = {
    'trade_count': 交易次数,
    'signal_triggered_days': 信号触发天数,
    'avg_turnover': 平均换手率,
    'trade_frequency': 交易频率
}
```

---

## 验证要点

### 1. T+1约束验证

```python
# 检查：今天买入的ETF，今天不能卖出
assert weights[t][i] >= weights[t-1][i] if last_buy_day[i] == t
```

### 2. 最小持有期验证

```python
# 检查：持有不足N天的不能卖出
assert weights[t][i] >= weights[t-1][i] if holding_days[i] < min_holding_days
```

### 3. 换手限制验证

```python
# 检查：每日换手不超过限制
turnover = sum(abs(weights[t] - weights[t-1]))
assert turnover <= max_daily_turnover
```

---

## 性能预期

### 交易频率

```
参数: min_holding_days=3, max_daily_turnover=0.5

预期:
  - 交易频率: 20-40%（每5-10天交易一次）
  - 平均换手: 30-50%
  - 交易成本: 0.02-0.04%/天
```

### 收益提升

```
相比固定周期:
  - 及时捕捉机会 → 收益提升
  - 灵活止损 → 回撤降低
  - 约束控制 → 成本可控
```

---

## 注意事项

### 1. 成本控制

```
- 设置合理的min_holding_days（建议3-5天）
- 设置合理的max_daily_turnover（建议30-50%）
- 提高signal_strength_threshold（只交易强信号）
```

### 2. 回测验证

```
- 对比固定周期和事件驱动的收益
- 检查交易频率是否合理
- 验证T+1约束是否生效
```

### 3. 实盘注意

```
- A股ETF确实是T+1
- 注意流动性（避免大单冲击）
- 监控滑点（实盘可能更大）
```

---

## 总结

### 核心优势

```
✅ 每日评估信号（事件驱动）
✅ A股T+1约束（合规）
✅ 最小持有期（控制成本）
✅ 换手限制（控制频率）
✅ 信号过滤（只交易强信号）
✅ 像交易股票一样灵活
```

### 适用场景

```
✅ 高频信号（每日更新）
✅ 需要及时止损
✅ 追求灵活性
✅ A股ETF交易
```

---

**文件位置**:
- 构建器: `core/event_driven_portfolio_constructor.py`
- Pipeline集成: `core/pipeline.py`
- 配置文件: `configs/default.yaml`
