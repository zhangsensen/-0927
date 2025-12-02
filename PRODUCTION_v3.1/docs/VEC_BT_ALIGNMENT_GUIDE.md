# VEC/BT 对齐完整指南

> **版本**: v1.1  
> **更新日期**: 2025-12-01  
> **状态**: ✅ 已验证 (100/100 组合对齐, 最大差异 < 0.05pp)
>
> ⚠️ **注意**: 本文档中的代码示例使用 FREQ=8 等参数仅作为演示，
> **生产环境请使用 v3.0 参数 (FREQ=3, POS_SIZE=2)**。
> 详见 `BEST_STRATEGY_43ETF_UNIFIED.md`

---

## 🎯 核心目标

确保 **VEC (向量化回测)** 与 **BT (Backtrader 事件驱动回测)** 的结果差异 < 0.1pp (0.001)。

| 指标 | 目标 | 当前状态 |
|------|------|----------|
| 平均差异 | < 0.1pp | ✅ 0.0254pp |
| 最大差异 | < 0.5pp | ✅ 0.0441pp |
| 对齐组合比例 | 100% | ✅ 100/100 |
| Margin 失败 | 0 | ✅ 0 |

---

## 📐 对齐的三个关键维度

### 1. 调仓日程一致 (Rebalance Schedule)

**问题**: VEC 和 BT 可能使用不同的日期逻辑计算调仓日。

**解决方案**: 使用统一的 `generate_rebalance_schedule()` 函数。

```python
from core.utils.rebalance import generate_rebalance_schedule

# 统一参数
LOOKBACK = 252  # 预热期（交易日）
FREQ = 8        # 调仓频率（每 8 个交易日）

# 生成调仓日程
rebalance_schedule = generate_rebalance_schedule(
    total_periods=T,
    lookback_window=LOOKBACK,
    freq=FREQ,
)
# 返回: [256, 264, 272, 280, ...] (第一个调仓日 = LOOKBACK + FREQ - 1)
```

**验证方法**:
```python
# 在 VEC 和 BT 中打印前 5 个调仓日，确保一致
print(f"Rebalance days: {rebalance_schedule[:5]}")
```

---

### 2. 择时信号滞后 (Timing Signal Shift)

**问题**: 择时信号必须滞后 1 天，避免前视偏差。

**错误做法**:
```python
# ❌ 错误：使用当日信号当日执行
timing_ratio = timing_arr[t]  # 当日计算，当日使用 -> 前视偏差
```

**正确做法**:
```python
from core.utils.rebalance import shift_timing_signal

# ✅ 正确：预先 shift 整个数组
timing_arr = shift_timing_signal(timing_series_raw.values)
# 在 kernel 中直接使用 timing_arr[t] 即为 t-1 日信号
```

**原理**:
```
日期    | t-1    | t (调仓日) | t+1
--------|--------|------------|--------
原始信号 | 0.8    | 0.6        | 1.0
shift后  | NaN    | 0.8        | 0.6     <- t 日使用的是 t-1 日的信号
```

---

### 3. 价格执行模式 (Price Execution Mode)

**问题**: VEC 和 BT 的成交价必须一致。

**推荐模式**: **Cheat-On-Close** (使用当日收盘价成交)

| 操作 | 价格 | 说明 |
|------|------|------|
| 买入 | `close[t]` | 当日收盘价 |
| 卖出 | `close[t]` | 当日收盘价 |
| 期末平仓 | `close[T-1]` | 最后一日收盘价 |

**BT 配置**:
```python
cerebro = bt.Cerebro(cheat_on_close=True)
cerebro.broker.set_coc(True)
```

**VEC Kernel**:
```python
# 卖出
price = close_prices[t, n]
proceeds = holdings[n] * price * (1.0 - commission_rate)

# 买入
price = close_prices[t, idx]
shares = target_pos_value / price
cost = shares * price * (1.0 + commission_rate)
```

---

## 🐛 常见陷阱与解决方案

### 陷阱 1: Numba argsort 不稳定排序

**现象**: 同一代码在 Numba JIT 和 Pure Python 中返回不同结果。

**根因**: `np.argsort` 对相等元素的排序顺序在 Numba 和 Python 中不一致。

**影响**: 当多个 ETF 得分相同时，选择的 ETF 不同，导致收益差异。

**示例**:
```python
import numpy as np
from numba import njit

@njit
def numba_argsort(arr):
    return np.argsort(arr)

arr = np.array([1.0, 1.0, 0.5])  # 两个相等的 1.0
print(numba_argsort(arr))  # 可能输出 [2, 0, 1]
print(np.argsort(arr))     # 可能输出 [2, 1, 0]  <- 不同!
```

**解决方案**: 使用稳定的 top-k 选择函数:

```python
@njit(cache=True)
def stable_topk_indices(scores, k):
    """稳定排序：按 score 降序，score 相同时按索引升序（tie-breaker）"""
    N = len(scores)
    result = np.empty(k, dtype=np.int64)
    used = np.zeros(N, dtype=np.bool_)
    
    for i in range(k):
        best_idx = -1
        best_score = -np.inf
        for n in range(N):
            if used[n]:
                continue
            # 关键：score 相同时选择索引更小的
            if scores[n] > best_score or (scores[n] == best_score and (best_idx < 0 or n < best_idx)):
                best_score = scores[n]
                best_idx = n
        if best_idx < 0 or best_score == -np.inf:
            return result[:i]
        result[i] = best_idx
        used[best_idx] = True
    return result
```

---

### 陷阱 2: Risk-Off 资产逻辑不一致

**现象**: BT 有 Risk-Off 资产（如货币基金），VEC 没有。

**影响**: 择时信号降低仓位时，资金去向不同。

**解决方案**: **移除 Risk-Off 资产逻辑**，保持简单。

```python
# ❌ 错误: BT 中有 Risk-Off 资产
if timing_ratio < 1.0:
    buy_risk_off_asset(...)  # VEC 没有这个逻辑

# ✅ 正确: 简单地减少暴露
target_exposure = current_value * timing_ratio
available_for_new = target_exposure - kept_value
```

---

### 陷阱 3: 资金计算顺序错误

**现象**: 买入前未更新 cash，导致资金不足。

**正确顺序**:
```python
# Step 1: 先卖出
for n in range(N):
    if should_sell(n):
        proceeds = sell(n)
        cash += proceeds  # ✅ 立即更新 cash

# Step 2: 计算当前总值（包含刚卖出的现金）
current_value = cash + sum(held_positions)

# Step 3: 再买入
for n in new_targets:
    cost = buy(n)
    cash -= cost
```

---

### 陷阱 4: Margin 失败 (资金不足)

**现象**: BT 报告 margin_failures > 0。

**原因**:
1. 买入金额计算未考虑手续费
2. 浮点精度问题导致微小超额

**解决方案**:
```python
# 目标仓位值（已预留手续费）
target_pos_value = available_for_new / new_count / (1.0 + commission_rate)

# 买入时检查资金
cost = shares * price * (1.0 + commission_rate)
if cash >= cost - 1e-5:  # 允许 1e-5 容差
    actual_cost = min(cost, cash)  # 不超过现有现金
    cash -= actual_cost
    holdings[idx] = shares
```

---

### 陷阱 5: Set 遍历顺序不确定

**现象**: 同一代码多次运行结果不同。

**根因**: Python `set` 遍历顺序不确定。

**解决方案**: 始终使用 `sorted()`:
```python
# ❌ 错误
for etf in selected_etfs:  # set 遍历顺序不确定
    ...

# ✅ 正确
for etf in sorted(selected_etfs):  # 确定性遍历
    ...
```

---

## 🔍 调试方法

### 方法 1: 逐日对比

```python
# 在 VEC 和 BT 中分别记录每日状态
debug_log = []
for t in rebalance_schedule:
    debug_log.append({
        'date': dates[t],
        'cash': cash,
        'holdings': holdings.copy(),
        'total_value': current_value,
        'selected_etfs': sorted(target_set),
    })

# 导出 CSV 逐日对比
pd.DataFrame(debug_log).to_csv('vec_debug.csv')
pd.DataFrame(bt_debug_log).to_csv('bt_debug.csv')
```

### 方法 2: 禁用 Numba JIT

```bash
# 用 Pure Python 运行，排除 Numba 问题
NUMBA_DISABLE_JIT=1 uv run python scripts/batch_vec_backtest.py
```

### 方法 3: 最小化测试

```python
# 只测试 1 个组合，打印详细日志
combo = ['CORRELATION_TO_MARKET_20D', 'MAX_DD_60D']
# ... 运行并对比
```

---

## 📂 关键文件清单

| 文件 | 用途 | 状态 |
|------|------|------|
| `core/utils/rebalance.py` | 共享工具 (shift_timing_signal, generate_rebalance_schedule) | ✅ 稳定 |
| `scripts/batch_vec_backtest.py` | VEC 批量回测 (含 stable_topk_indices) | ✅ 已修复 |
| `strategy_auditor/core/engine.py` | BT GenericStrategy | ✅ 已重写 |
| `scripts/full_vec_bt_comparison.py` | 参考实现 (FullDebugStrategy) | ✅ Ground Truth |

---

## ✅ 验证检查清单

在完成任何修改后，运行以下验证:

```bash
# 1. 运行 VEC 回测
uv run python scripts/batch_vec_backtest.py

# 2. 运行 BT 回测
uv run python scripts/batch_bt_backtest.py

# 3. 对比结果
uv run python -c "
import pandas as pd
from pathlib import Path

vec_df = pd.read_csv(sorted(Path('results').glob('vec_full_backtest_*/vec_all_combos.csv'))[-1])
bt_df = pd.read_csv(sorted(Path('results').glob('bt_backtest_full_*/bt_results.csv'))[-1])

merged = pd.merge(vec_df[['combo', 'vec_return']], bt_df[['combo', 'bt_return']], on='combo')
merged['diff_pp'] = abs(merged['vec_return'] - merged['bt_return']) * 100

print(f'平均差异: {merged[\"diff_pp\"].mean():.4f}pp')
print(f'最大差异: {merged[\"diff_pp\"].max():.4f}pp')
print(f'对齐率: {(merged[\"diff_pp\"] < 0.1).sum()}/{len(merged)}')
"
```

**通过标准**:
- [ ] 平均差异 < 0.1pp
- [ ] 最大差异 < 0.5pp
- [ ] 对齐率 = 100%
- [ ] Margin 失败 = 0

---

## 📚 历史修复记录

| 日期 | 问题 | 解决方案 | 影响 |
|------|------|----------|------|
| 2025-11-29 | Numba argsort 不稳定 | 新增 stable_topk_indices() | 差异从 8.98pp 降至 0.04pp |
| 2025-11-29 | GenericStrategy Risk-Off 逻辑 | 重写 rebalance() 对齐 FullDebugStrategy | Margin 失败从 2141 降至 0 |
| 2025-11-28 | 调仓日不一致 | 统一使用 generate_rebalance_schedule() | 消除日期漂移 |
| 2025-11-27 | Set 遍历不确定性 | 全局使用 sorted() | 确保可复现性 |

---

## 🏆 最佳策略验证

```
因子组合: CORRELATION_TO_MARKET_20D + MAX_DD_60D + PRICE_POSITION_120D + PRICE_POSITION_20D
VEC: 121.0160%
BT:  121.0601%
差异: 0.0441pp ✅
```

---

**文档维护者**: AI Quant Architect  
**最后验证**: 2025-11-29 20:05
