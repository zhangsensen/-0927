# 架构设计与引擎对齐

> **最后更新**: 2025-11-28

---

## 三层引擎设计理念

### 为什么需要三层？

| 层级 | 目的 | 特点 |
|------|------|------|
| **WFO** | 快速筛选 | Numba 加速，2.5s 完成 12,597 组合 |
| **VEC** | 精确复算 | 向量化，与 BT 对齐 |
| **BT** | 兜底审计 | 事件驱动，资金约束 |

### 对齐要求

```
WFO ─────────→ 无前视、排序稳定即可
                    ↓
VEC ←────────→ BT   严格对齐 (< 0.01pp)
```

**核心结论**：
- VEC ↔ BT：必须严格对齐，这是"基准一致性"
- WFO ↔ VEC/BT：数值可能不同，但排序稳定

---

## 无前视偏差保证

### 1. 择时信号滞后

```python
# core/utils/rebalance.py
def shift_timing_signal(timing_values, dates, rebalance_dates):
    """
    确保 T 日调仓使用的是 T-1 日的择时信号
    """
    shifted = np.ones(len(dates))
    for i, d in enumerate(dates):
        if d in rebalance_dates:
            prev_idx = i - 1
            if prev_idx >= 0:
                shifted[i] = timing_values[prev_idx]
    return shifted
```

### 2. 统一调仓日程

```python
# core/utils/rebalance.py
def generate_rebalance_schedule(T, lookback, freq):
    """
    生成确定性的调仓日程
    第一个调仓日 = lookback + freq - (lookback % freq)
    """
    first_rebalance = lookback + freq - (lookback % freq)
    return set(range(first_rebalance, T, freq))
```

### 3. 价格校验

```python
# core/utils/rebalance.py
def ensure_price_views(open_prices, close_prices):
    """
    确保 open/close 价格视图正确
    如果 open 全为 NaN，fallback 到 close
    """
```

---

## 引擎实现对比

### 信号计算

| 步骤 | VEC | BT |
|------|-----|-----|
| 因子加载 | `factors_3d[t-1, n, f_idx]` | 同左 |
| 得分计算 | `nansum` | 同左 |
| 有效性判断 | `score != 0 and not nan` | 同左 |

### 调仓逻辑

| 步骤 | VEC | BT |
|------|-----|-----|
| 卖出 | 按持仓市值计算 | `self.close(data)` |
| 买入 | `available / new_targets / (1+comm)` | 同左公式 |
| 执行价格 | `close[t]` (COC 模式) | Cheat-On-Close |

---

## WFO 与 VEC/BT 差异说明

### 为什么数值不同？

WFO 的 `_backtest_combo_numba` 是为高速筛选设计的，部分实现细节与 VEC/BT 不同：

1. **资金路径简化** - WFO 简化了部分资金约束以提升 numba 性能
2. **目的不同** - WFO 是"粗筛器"，真相在 VEC/BT

### 为什么不修改 WFO？

- **收益有限**：VEC+BT 已经很干净，有单测守护
- **风险较高**：大动 numba 内核会引入新 bug
- **排序有效**：WFO 的相对排序依然有意义

---

## 共享模块

### `core/utils/rebalance.py`

```python
# VEC 和 BT 共用的核心函数
from core.utils.rebalance import (
    shift_timing_signal,        # 择时信号滞后
    generate_rebalance_schedule, # 调仓日程
    ensure_price_views,         # 价格校验
)
```

### `core/market_timing.py`

```python
# 择时模块
from core.market_timing import LightTimingModule

timing = LightTimingModule(config)
timing_values = timing.compute(close_prices)  # 返回 [0, 1] 信号
```

---

## 验证方法

### 快速验证

```bash
python scripts/full_vec_bt_comparison.py --combo "ADX_14D + CMF_20D + MAX_DD_60D + RET_VOL_20D + SHARPE_RATIO_20D"
```

### 预期输出

```
VEC 收益: 70.7992%
BT 收益:  70.7991%
差异:     -0.00 pp  ✅
```

### 批量验证

```bash
# VEC 批量
python scripts/batch_vec_backtest.py

# BT 批量
python scripts/batch_bt_backtest.py

# 对比结果
# 预期: 所有组合差异 < 0.1pp
```

---

## 常见问题

### Q: 为什么 WFO 显示 234%，VEC/BT 只有 70%？

A: 这是正常的。WFO 的实现细节与 VEC/BT 不同，但 WFO 的排序是有效的。真实收益以 VEC/BT 为准。

### Q: 如何确保无前视偏差？

A: 三重保证：
1. `shift_timing_signal` - 择时信号 T-1
2. `generate_rebalance_schedule` - 确定性调仓日
3. `ensure_price_views` - 价格校验

### Q: 什么时候需要清理缓存？

A: 修改以下内容后：
- 因子计算逻辑
- 数据加载逻辑
- 调仓规则
