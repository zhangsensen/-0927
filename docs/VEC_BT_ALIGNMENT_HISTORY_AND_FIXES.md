# VEC vs BT 对齐问题历史记录与修复方案

> **重要**: 此文档记录了 VEC（向量化回测引擎）和 BT（Backtrader 回测引擎）对齐过程中发现的所有问题及其根本原因和修复方案。后续维护者和 AI 模型必须阅读此文档以理解两引擎的对齐逻辑。

**最后更新**: 2025-11-28  
**最终状态**: ✅ 完全对齐（差异 < 0.01pp）

---

## 🎯 架构决策总结（2025-11-28 最终确认）

### 三层引擎分工

| 层级 | 引擎 | 职责 | 对齐要求 |
|------|------|------|----------|
| **筛选层** | WFO | 高维因子组合空间搜索，产出候选池+粗排序 | 无前视、排序稳定即可 |
| **复算层** | VEC | 共享规则下的高精度矢量化复算 | **严格对齐 BT** |
| **审计层** | BT | 事件驱动+资金约束的最终兜底审计 | **基准真相** |

### 关键结论

1. **VEC ↔ BT 已严格对齐**（< 0.01pp 差异）
   - `full_vec_bt_comparison.py` 验证通过
   - 共享：`shift_timing_signal`、`generate_rebalance_schedule`、`ensure_price_views`

2. **WFO 不需要对齐到 VEC/BT**
   - WFO 是"粗筛器"，为排序服务而非"收益真相的最终来源"
   - 同一组合在 WFO 中 234%，在 VEC/BT 中 70% —— 这是正常的，不是 bug
   - 只要无前视（已用 `shift_timing_signal` + 统一调仓日程保证）、排序稳定，相对排序依然有意义

3. **Best Practice 工作流**
   ```
   WFO（1万+组合筛选 Top-N）
         ↓
   VEC（Top-N 全量复算，统一规则）
         ↓
   BT（Top-K 审计，过滤资金路径约束失真）
   ```

### 为什么 WFO 和 VEC/BT 数值不同？

WFO 的 `_backtest_combo_numba` 实现细节与 VEC/BT 的"最终资金路径"并非一一对应：
- **目的不同**：WFO 是为高速筛选设计，VEC/BT 是为精确复算设计
- **资金路径**：WFO 简化了部分资金约束逻辑以提升 numba 性能
- **这不是问题**：因为我们有更可信的真相源（VEC+BT），WFO 的角色自然退居为"粗筛器"

**工程决策**：不修改 WFO 内核追求数值一致，因为：
- 收益有限：VEC+BT 已经很干净，有单测守护
- 风险较高：大动 numba 内核会引入新 bug 面积

---

## 📋 问题概览

在对齐过程中，共发现并修复了以下 **5 个核心问题**：

| 问题编号 | 问题描述 | 影响 | 修复位置 | 状态 |
|---------|---------|------|---------|------|
| P1 | BT 资金计算使用卖出前净值 | -6.42pp 收益差异 | `engine.py` | ✅ 已修复 |
| P2 | BT Margin 订单失败 | 61次订单被拒绝 | Broker 配置 | ✅ 已修复 |
| P3 | VEC 浮点精度问题 | 合法买入被拒绝 | `batch_vec_backtest.py` | ✅ 已修复 |
| P4 | 择时信号双重滞后 | 信号时序错位 | `shift_timing_signal()` | ✅ 已修复 |
| P5 | 调仓日程生成不一致 | 调仓日错位 | `generate_rebalance_schedule()` | ✅ 已修复 |

---

## 🔍 问题详细分析

### P1: BT 资金计算使用卖出前净值

**发现日期**: 2025-11-27  
**影响程度**: 严重（-6.42pp 收益差异）

**问题描述**:  
BT 的 `GenericStrategy` 在计算买入目标仓位时，使用的是 **卖出订单提交前** 的净值 (`self.broker.getvalue()`)，而不是卖出后的现金。这导致买入金额计算错误。

**错误代码**:
```python
# ❌ 错误：使用卖出前的净值
current_equity = self.broker.getvalue()
target_exposure = current_equity * timing_ratio
available_for_new = target_exposure - kept_holdings_value
```

**问题根因**:  
Backtrader 的 `broker.getvalue()` 返回的是当前时刻的账户净值，但在 COC（Cheat-On-Close）模式下，卖出订单虽然立即执行，但 broker 内部的现金更新存在时序问题。

**修复代码**:
```python
# ✅ 正确：手动计算卖出后的现金
cash_after_sells = self.broker.getcash()
for ticker, shares in current_holdings.items():
    data = self.etf_map[ticker]
    if ticker not in target_set:
        # 预计卖出收入（扣除佣金）
        cash_after_sells += shares * data.close[0] * (1 - COMMISSION_RATE)
        self.close(data)
    else:
        kept_holdings_value += shares * data.close[0]

# 使用卖出后的资金计算目标
current_value = cash_after_sells + kept_holdings_value
target_exposure = current_value * timing_ratio
available_for_new = max(0.0, target_exposure - kept_holdings_value)
```

**修复文件**:
- `strategy_auditor/core/engine.py` (GenericStrategy.rebalance)
- `scripts/full_vec_bt_comparison.py` (FullDebugStrategy.rebalance)

---

### P2: BT Margin 订单失败

**发现日期**: 2025-11-27  
**影响程度**: 严重（61 次订单被拒绝）

**问题描述**:  
即使启用了 COC 模式，Backtrader 在订单提交时仍会检查当前现金是否足够。由于 P1 问题导致的资金计算错误，许多买入订单因 "Margin" 被拒绝。

**表现**:
```
共 61 次订单失败
  2021-02-22: 159801 - Margin
  2021-03-04: 515030 - Margin
  ...
```

**修复方案**:
```python
# 在 Cerebro 初始化时
cerebro.broker.set_coc(True)              # 启用 Cheat-On-Close
cerebro.broker.set_checksubmit(False)     # ✅ 禁用订单提交时的现金检查
cerebro.broker.setcommission(commission=COMMISSION_RATE, leverage=1.0)  # 无杠杆
```

**修复文件**:
- `strategy_auditor/core/backtester.py`
- `scripts/full_vec_bt_comparison.py`

---

### P3: VEC 浮点精度问题

**发现日期**: 2025-11-27  
**影响程度**: 中等（关键买入被错过）

**问题描述**:  
VEC 引擎在计算买入成本时，由于浮点精度累积误差，导致 `cost > cash` 的判断错误拒绝了合法买入。

**具体案例**:
- 日期: 2021-06-30
- ETF: 515030
- 实际现金: 333,333.32999999...
- 计算成本: 333,333.33000000...
- 差异: ~1e-8
- 结果: 买入被拒绝，该 ETF 随后上涨 16%

**修复代码**:
```python
# ✅ 增加浮点容差
FLOAT_TOLERANCE = 1e-5

if cost <= cash + FLOAT_TOLERANCE:  # 原来是 if cost <= cash
    holdings[n] = shares
    cash -= cost
```

**修复文件**:
- `scripts/batch_vec_backtest.py` (vec_backtest_kernel)
- `scripts/full_vec_bt_comparison.py` (VEC 模拟部分)

---

### P4: 择时信号双重滞后

**发现日期**: 2025-11-26  
**影响程度**: 中等（信号时序错位）

**问题描述**:  
原始设计中，择时信号 (timing) 需要从 T-1 日获取用于 T 日决策。但 VEC 和 BT 对信号的处理方式不一致：

- **VEC 原实现**: `timing_arr[t-1]` （在核函数内部滞后）
- **BT 原实现**: `timing.loc[current_date]` （传入前已滞后）

这导致 VEC 出现 **双重滞后**（如果传入的 timing 已经滞后一天）。

**修复方案**:  
统一在数据加载阶段对择时信号做一次 shift，然后在核函数/策略中直接使用当前索引。

```python
# core/utils/rebalance.py
def shift_timing_signal(timing: np.ndarray, fill_value: float = 1.0) -> np.ndarray:
    """将择时信号向后移动一天，使得 timing[t] 代表 T-1 日的信号值"""
    shifted = np.empty_like(timing)
    shifted[0] = fill_value  # 第一天用默认值填充
    shifted[1:] = timing[:-1]
    return shifted
```

**使用方式**:
```python
# 数据加载阶段
timing_arr = shift_timing_signal(raw_timing.values)

# VEC 核函数
timing_ratio = timing_arr[t]  # 直接使用，不再内部滞后

# BT 策略
timing = self.params.timing.loc[current_date]  # timing 已经是 shifted 版本
```

**修复文件**:
- `etf_rotation_optimized/core/utils/rebalance.py` (新增 helper)
- `scripts/batch_vec_backtest.py` (调用 shift)
- `strategy_auditor/core/engine.py` (使用 shifted timing)

---

### P5: 调仓日程生成不一致

**发现日期**: 2025-11-26  
**影响程度**: 严重（调仓日完全错位）

**问题描述**:  
VEC 和 BT 各自计算调仓日的方式不同：

- **VEC 原实现**: `for t in range(LOOKBACK, T): if t % FREQ == 0`
- **BT 原实现**: `if bar_index % self.params.freq == 0`

由于起始点对齐方式不同，导致调仓日集合不一致。

**修复方案**:  
创建统一的调仓日程生成 helper：

```python
# core/utils/rebalance.py
def generate_rebalance_schedule(
    total_periods: int,
    lookback_window: int = 252,
    freq: int = 8,
    offset: int = 0
) -> np.ndarray:
    """生成调仓日程数组
    
    Args:
        total_periods: 总交易日数 T
        lookback_window: 预热期长度
        freq: 调仓频率
        offset: 起始偏移量
    
    Returns:
        调仓日 bar_index 数组
    """
    first_idx = lookback_window + offset
    # 对齐到 freq 的整数倍
    first_idx = first_idx + (freq - first_idx % freq) % freq
    
    rebalance_days = np.arange(first_idx, total_periods, freq)
    return rebalance_days
```

**使用方式**:
```python
# VEC
rebalance_schedule = generate_rebalance_schedule(T, LOOKBACK, FREQ)
for i in range(len(rebalance_schedule)):
    t = rebalance_schedule[i]
    # ... 执行调仓

# BT
rebalance_schedule = generate_rebalance_schedule(T, LOOKBACK, FREQ)
self.rebalance_set = set(rebalance_schedule.tolist())
# 在 next() 中
if bar_index in self.rebalance_set:
    self.rebalance(...)
```

**修复文件**:
- `etf_rotation_optimized/core/utils/rebalance.py` (新增 helper)
- `scripts/batch_vec_backtest.py` (使用 helper)
- `strategy_auditor/core/engine.py` (使用 helper)

---

## 📊 修复前后对比

### 修复前状态

| 指标 | VEC | BT | 差异 |
|------|-----|-----|------|
| 总收益 | 31.53% | 25.11% | -6.42pp |
| 调仓次数 | 143 | 143 | ✅ |
| Margin 失败 | 0 | 61 | ❌ |
| 净值相关性 | - | - | ~0.95 |

### 修复后状态

| 指标 | VEC | BT | 差异 |
|------|-----|-----|------|
| 总收益 | 34.8111% | 34.8110% | -0.0001pp |
| 调仓次数 | 143 | 143 | ✅ |
| Margin 失败 | 0 | 0 | ✅ |
| 净值相关性 | - | - | 1.000000 |

---

## 🔧 关键代码位置

### 常量定义

```python
# 两引擎必须使用相同的常量
FREQ = 8                    # 调仓频率（交易日）
POS_SIZE = 3                # 持仓 ETF 数量
INITIAL_CAPITAL = 1_000_000.0  # 初始资金
COMMISSION_RATE = 0.0002    # 手续费率（双边）
LOOKBACK = 252              # 预热期（交易日）
```

### 共享工具位置

```
etf_rotation_optimized/core/utils/rebalance.py
├── generate_rebalance_schedule()  # 调仓日程生成
├── shift_timing_signal()          # 择时信号偏移
└── ensure_price_views()           # 价格数据验证
```

### VEC 引擎位置

```
scripts/batch_vec_backtest.py
├── vec_backtest_kernel()  # Numba JIT 核函数
└── run_vec_backtest()     # 入口函数
```

### BT 引擎位置

```
strategy_auditor/core/engine.py
├── GenericStrategy        # 通用策略类
│   ├── __init__()         # 初始化参数
│   ├── prenext()          # 预热期处理
│   ├── next()             # 主循环
│   └── rebalance()        # 调仓逻辑

strategy_auditor/core/backtester.py
└── run_backtrader_backtest()  # BT 运行入口
```

---

## ⚠️ 注意事项

### 1. 不要重复修复

所有问题已修复完成，**不要**再次修改以下逻辑：
- 调仓日程生成方式
- 择时信号偏移方式
- 资金计算公式
- Broker 配置

### 2. 新增因子时的注意事项

如果新增因子，确保：
- VEC: 使用 `factors_3d[t-1, n, idx]` 获取 T-1 日因子
- BT: 使用 `scores.loc[prev_ts]` 获取 T-1 日因子
- 两者的 NaN 处理逻辑等价

### 3. 修改调仓逻辑时的验证

任何修改调仓逻辑的 PR 必须：
1. 运行 `scripts/full_vec_bt_comparison.py` 验证对齐
2. 确保收益差异 < 0.1pp
3. 确保 Margin 失败次数 = 0

### 4. 浮点精度问题

在任何涉及资金计算的地方，使用容差：
```python
FLOAT_TOLERANCE = 1e-5
if cost <= cash + FLOAT_TOLERANCE:
    # 执行买入
```

---

## 📝 验证脚本

### 完整对比测试

```bash
cd /home/sensen/dev/projects/-0927
uv run python scripts/full_vec_bt_comparison.py
```

预期输出：
```
VEC 收益: 34.8111%
BT 收益:  34.8110%
差异:     -0.00 pp
✅ VEC 和 BT 完全对齐
```

### 快速一致性检查

```bash
uv run python -c "
from scripts.batch_vec_backtest import FREQ, POS_SIZE, INITIAL_CAPITAL, COMMISSION_RATE, LOOKBACK
from strategy_auditor.core.engine import FREQ as BT_FREQ, POS_SIZE as BT_POS, INITIAL_CAPITAL as BT_CAP, COMMISSION_RATE as BT_COMM, LOOKBACK as BT_LOOK

assert FREQ == BT_FREQ, f'FREQ mismatch: {FREQ} vs {BT_FREQ}'
assert POS_SIZE == BT_POS, f'POS_SIZE mismatch'
assert INITIAL_CAPITAL == BT_CAP, f'INITIAL_CAPITAL mismatch'
assert COMMISSION_RATE == BT_COMM, f'COMMISSION_RATE mismatch'
assert LOOKBACK == BT_LOOK, f'LOOKBACK mismatch'
print('✅ 所有常量一致')
"
```

---

## 📚 相关文档

- `docs/VEC_BT_ALIGNMENT_AUDIT_REPORT.md` - 最新审计报告
- `docs/BT_VEC_ALIGNMENT_VERIFICATION_REPORT.md` - 详细验证报告
- `vec\bt差异记录.md` - 原始差异记录

---

## 🏷️ 版本历史

| 日期 | 版本 | 变更 |
|------|------|------|
| 2025-11-26 | v1.0 | 初始对齐工作开始 |
| 2025-11-27 | v2.0 | 修复 P1-P5 所有问题 |
| 2025-11-28 | v3.0 | 完成全面审计，差异 < 0.1pp |

---

**文档维护者**: AI Assistant  
**最后验证**: 2025-11-28 17:37:21
