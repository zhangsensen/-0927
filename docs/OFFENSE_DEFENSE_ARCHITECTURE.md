# 🏗️ 进攻 vs 防御架构 (Offense vs Defense Architecture)

> **版本**: v1.0
> **日期**: 2025-11-30
> **作者**: AI Quant Architect

---

## 📋 概述

本文档描述了 ETF 轮动策略的核心架构重构方案，将 **"预测能力 (Alpha)"** 与 **"风控能力 (Risk)"** 彻底分离。

### 核心哲学

| 模块 | 定位 | 职责 | 核心指标 |
|------|------|------|----------|
| **WFO** | 进攻端 (Offense) | 寻找"预测最准"的因子组合 | Rank IC |
| **VEC** | 防御端 (Defense) | 择时、止盈止损、资金管理 | Calmar Ratio |

### 零杠杆原则

严格执行 **1.0 或 0.0** 仓位，剔除所有动态波动率调整（Target Volatility），回归最朴素的实盘逻辑。

---

## 🔧 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        WFO (纯预测引擎)                          │
│  ├── 输入: 18 个基础因子                                         │
│  ├── 优化目标: Maximize(Rank_IC)                                 │
│  ├── 输出: top100_by_ic.parquet                                 │
│  └── 🚫 不含: 止损、资金曲线、Sharpe 优化                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    VEC (实战指挥官)                              │
│  ├── A. 信号层: 读取 WFO 因子权重，计算 ETF 得分                 │
│  ├── B. 择时层: 大盘择时 (MA / Risk-Off)                        │
│  └── C. 动态风控层: 止损、止盈、熔断                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    BT (审计层)                                   │
│  └── 用 Backtrader 逐笔核对 VEC 结果                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛡️ 动态风控层 (VEC Defense Layer)

### 1. 动态移动止损 (Trailing Stop Loss)

基于 **高水位 (High Water Mark)**，当价格跌破高水位的 (1 - stop_pct) 时触发止损。

```yaml
trailing_stop_pct: 0.08   # 8% 移动止损
cooldown_days: 3          # 止损后 3 天内不买入该标的
```

**逻辑** (v1.1 修复后):
- ⚠️ **HWM 滞后更新**: 使用昨日 HWM 计算今日止损线，避免"先跌后涨"的前视偏差
- 止损价格: `Stop_Price = prev_HWM * (1 - stop_pct)`
- 触发条件: `Low_Price < Stop_Price`
- **精确执行价格**:
  - 跳空低开 (`Open < Stop`): `exec_price = Open`
  - 盘中触发 (`Open >= Stop, Low < Stop`): `exec_price = Stop_Price`
  - 最终: `exec_price = max(exec_price, Low)` (防止数据异常)
- 止损后更新 HWM 供明日使用

### 2. 阶梯式止盈 (Dynamic Profit Taking)

当持仓收益达到阈值后，自动收紧止损线以保护利润。

```yaml
profit_ladders:
  - threshold: 0.15   # 收益达到 15%
    new_stop: 0.05    # 止损线收紧至 5%
  - threshold: 0.30   # 收益达到 30%
    new_stop: 0.03    # 止损线收紧至 3%
```

**效果**:
- 初始止损: -8%
- 收益 > 15%: 止损线上移至 -5% (保护本金)
- 收益 > 30%: 止损线上移至 -3% (锁定利润)

### 3. 熔断机制 (Circuit Breaker)

账户级别风控，防止单日/总体大幅回撤。

```yaml
circuit_breaker:
  max_drawdown_day: 0.05    # 单日跌幅超过 5% 触发
  max_drawdown_total: 0.20  # 总回撤超过 20% 触发
  recovery_days: 5          # 熔断后等待 5 天才能重新开仓
```

**状态机**:
```
正常交易 ──> [触发条件] ──> 熔断激活 ──> [倒计时结束] ──> 正常交易
              ↑                              │
              └──────── recovery_days ───────┘
```

### 4. 零杠杆原则 (Zero Leverage)

```yaml
leverage_cap: 1.0         # 最大仓位上限
dynamic_leverage:
  enabled: false          # 禁用动态杠杆
```

---

## 📊 配置文件结构

```yaml
# configs/combo_wfo_config.yaml

backtest:
  risk_control:
    enabled: true
    
    # 零杠杆原则
    leverage_cap: 1.0
    allow_fractional: false
    
    # 1. 动态止损
    trailing_stop_pct: 0.08
    cooldown_days: 3
    
    # 2. 阶梯止盈
    profit_ladders:
      - threshold: 0.15
        new_stop: 0.05
      - threshold: 0.30
        new_stop: 0.03
    
    # 3. 熔断机制
    circuit_breaker:
      max_drawdown_day: 0.05
      max_drawdown_total: 0.20
      recovery_days: 5
    
    # 4. 动态杠杆 (已禁用)
    dynamic_leverage:
      enabled: false
```

---

## 🧪 回测结果对比

| 配置 | 平均收益率 | 平均最大回撤 | 平均 Calmar | 说明 |
|------|-----------|-------------|-------------|------|
| 基础止损 (8%) | 0.57% | 39.54% | 0.023 | 止损频繁触发 |
| +阶梯止盈+冷却期 (v1.0) | 1.91% | 33.28% | 0.037 | 保护利润效果明显 |
| +精确执行价修复 (v1.1) | **6.80%** | 33.92% | **0.067** | 更诚实的回测 |

**Top 组合 (v1.1 配置)**:
- `ADX_14D + PRICE_POSITION_20D`: 63.7% 收益, 19.3% 回撤, Calmar 0.59
- `ADX_14D + PRICE_POSITION_120D + PRICE_POSITION_20D`: 59.3% 收益, 21.3% 回撤, Calmar 0.51

---

## 🔄 工作流

```bash
# 1. 挖掘: WFO 找到预测力最强的因子组合
uv run python src/etf_strategy/run_combo_wfo.py

# 2. 回测: VEC 应用风控规则进行实战模拟
uv run python scripts/batch_vec_backtest.py

# 3. 审计: BT 逐笔核对 (可选)
uv run python scripts/batch_bt_backtest.py
```

---

## 📝 实现细节

### Numba 兼容性

所有风控逻辑在 `vec_backtest_kernel` 函数中实现，使用 Numba JIT 编译以保持高性能。

关键参数通过 `np.array` 传递以满足 Numba 要求:
```python
profit_ladder_thresholds = np.array([0.15, 0.30, np.inf], dtype=np.float64)
profit_ladder_stops = np.array([0.05, 0.03, 0.08], dtype=np.float64)
```

### 状态追踪

```python
# 每个标的的当前止损率
current_stop_pcts = np.full(N, trailing_stop_pct)

# 冷却期剩余天数
cooldown_remaining = np.zeros(N, dtype=np.int64)

# 熔断状态
circuit_breaker_active = False
circuit_breaker_countdown = 0
```

---

## ⚠️ 注意事项

1. **阶梯止盈会增加交易次数**: 因为止损线收紧后更容易触发
2. **熔断可能过于保守**: 总回撤 20% 在长期回测中很容易触发
3. **冷却期防止追高**: 但也可能错过反弹机会
4. **参数需要根据市场环境调整**: 没有放之四海皆准的最优参数

---

## 🔮 后续优化方向

1. **自适应止损**: 根据波动率动态调整止损幅度
2. **多周期信号**: 结合日/周/月级别信号
3. **风险平价**: 按波动率分配仓位权重
4. **机器学习择时**: 用 ML 模型预测市场状态

---

**🔒 v1.1 修复完成 | HWM 滞后更新 + 精确执行价 | 进攻与防御分离 | 零杠杆原则**
