## 执行延迟修复验证报告

**日期**: 2025-11-24  
**测试环境**: etf_rotation_experiments, 6 ETFs, 2023-01-01 至 2024-12-31 (484天)  
**测试组合数**: 9384 (combo_size=3,5, freq=2,5)

---

### 1. 修复实施

**修改文件**: `strategies/backtest/production_backtest.py`

**核心逻辑**:
- 引入 `RB_EXECUTION_LAG` 环境变量（默认 0，向后兼容）
- LAG=0: 信号T-1立即捕获Return[T] (原始 Lag-1 IC)
- LAG=1: 信号T延迟至T+1生效 (修正 Lag-2 IC)
- 使用 `pending_weights` 队列实现延迟应用

**关键代码段** (Line ~677-683):
```python
execution_lag = int(os.environ.get("RB_EXECUTION_LAG", "0").strip() or "0")
current_weights = np.zeros(N)
pending_weights = None  # 延迟 1 日的目标权重

# 主循环内：调仓日生成信号 → pending_weights，次日才应用
if execution_lag == 1 and pending_weights is not None:
    current_weights = pending_weights
    pending_weights = None
```

---

### 2. 验证结果

#### 2.1 WFO IC 评分阶段（预期：不受影响）

| Run | RB_EXECUTION_LAG | mean_ic | Best Combo |
|-----|------------------|---------|------------|
| run_20251124_192847 | 0 | 0.003093 | MAX_DD_60D + PRICE_POSITION_120D + VORTEX_14D |
| run_20251124_192912 | 1 | 0.003093 | MAX_DD_60D + PRICE_POSITION_120D + VORTEX_14D |

✅ **结论**: WFO 内部 IC 计算未调用 `backtest_no_lookahead`，输出完全一致（符合预期）

---

#### 2.2 Profit Backtest 阶段（预期：LAG=1 性能下降）

**Top1 策略对比** (CALMAR_RATIO_60D + MAX_DD_60D + OBV_SLOPE_10D + RET_VOL_20D + VORTEX_14D):

| 指标 | LAG=0 (原始) | LAG=1 (修正) | 差距 | 结论 |
|------|--------------|--------------|------|------|
| 年化收益 | 2.92% | **3.81%** | +0.89% | ❌ **异常** |
| Sharpe | 0.098 | **0.128** | +0.031 | ❌ **异常** |
| 最大回撤 | -27.36% | -26.85% | +0.51% | - |

**Top5 平均**:

| 指标 | LAG=0 | LAG=1 | 差距 |
|------|-------|-------|------|
| 年化收益 | 2.16% | **3.18%** | +1.02% |
| Sharpe | 0.072 | **0.107** | +0.035 |

---

### 3. 问题诊断

#### ❌ **预期 vs 实际**

**预期**: LAG=1 应该导致性能**下降**（消除前视偏差 → 更真实 → 更低收益）

**实际**: LAG=1 性能**提升** 30-50% (年化 +0.89%, Sharpe +0.031)

---

#### 🔍 **根本原因**

1. **pending_weights 应用时序错误**:
   - 当前代码在**每日循环开始**时应用 `pending_weights`
   - 但这意味着：调仓日T生成信号 → pending → **T日就应用** → 立即捕获Return[T]
   - **实际等效于 LAG=0**，只是多了一层无效缓存

2. **正确逻辑应该是**:
   ```
   Day T (调仓日):
     - 生成 target_weights
     - pending_weights ← target_weights
     - current_weights 保持不变 (旧仓位)
     - 计算 Return[T] 使用旧仓位
   
   Day T+1:
     - current_weights ← pending_weights (新仓位生效)
     - pending_weights ← None
     - 计算 Return[T+1] 使用新仓位
   ```

3. **当前错误逻辑**:
   ```
   Day T (调仓日):
     - [循环开始] 应用 pending (但此时 pending=None)
     - 生成 target_weights → pending
     - 计算 Return[T] 使用旧仓位 ✅
   
   Day T+1:
     - [循环开始] 应用 pending → current_weights
     - 但 Return[T] 已经在 T 日用旧仓位计算了
     - Return[T+1] 用新仓位 ✅
   ```

   **看似正确，但实际 Return[T] 已在 T 日末尾计算完毕，pending 在 T+1 日开始才应用已经晚了**

---

### 4. 修复方案

**调整 pending_weights 应用位置**:

```python
# ❌ 错误：在循环开始应用（T 日收益已算完）
for offset, day_idx in enumerate(range(start_idx, T)):
    if execution_lag == 1 and pending_weights is not None:
        current_weights = pending_weights  # T+1日开始才应用，但 Return[T] 已算完
        pending_weights = None

# ✅ 正确：在收益计算前应用（确保 T 日收益用旧仓位）
for offset, day_idx in enumerate(range(start_idx, T)):
    is_rebalance_day = ...
    
    if is_rebalance_day:
        # 生成新信号 → pending
        target_weights = ...
        if execution_lag == 1:
            pending_weights = target_weights
        else:
            current_weights = target_weights
    
    # 在计算收益前，检查是否有待应用的 pending
    if execution_lag == 1 and pending_weights is not None:
        # 但要注意：这样会在调仓日当天就应用，仍然错误
        pass
```

**真正正确的逻辑**:
```python
# 需要标记：pending 是否应该在本轮生效
apply_pending_this_round = False

if execution_lag == 1 and pending_weights is not None:
    # 只有在非调仓日才应用 pending
    if not is_rebalance_day:
        current_weights = pending_weights
        pending_weights = None

if is_rebalance_day:
    target_weights = ...
    if execution_lag == 1:
        pending_weights = target_weights
    else:
        current_weights = target_weights
```

---

### 5. 行动计划

1. **修正 pending 应用逻辑**: 确保调仓日T的Return[T]使用旧仓位，T+1才切换
2. **重新验证**: LAG=1 应该显示性能**下降**
3. **修正 WFO IC 计算**: 当前 WFO 未感知 LAG，需要在 `combo_wfo_optimizer.py` 中的 IC 窗口计算也引入延迟
4. **完整重训**: 使用修正后的 LAG=1 重新运行完整 WFO

---

### 6. 结论

🔴 **当前修复无效**

- `RB_EXECUTION_LAG=1` 并未真正引入执行延迟
- 性能反常提升 30-50% 证明逻辑错误
- 需要重新审查 `pending_weights` 应用时序
- WFO 阶段未感知 LAG，需要独立修复

**下一步**: 修正 `production_backtest.py` 中的 pending 应用逻辑，确保延迟真正生效
