# VEC vs BT 对齐问题历史记录

> **重要**: 此文档记录了 VEC/BT 对齐过程中的所有问题和修复方案。必读！

## 最终状态: ✅ 完全对齐 (差异 < 0.1pp)

## 6 个核心问题及修复 (2025-11-29 更新)

### P1: BT 资金计算错误 (-6.42pp)
- **问题**: BT 使用 `broker.getvalue()` (卖出前净值) 计算买入目标
- **修复**: 改用 `cash_after_sells = broker.getcash() + 预计卖出收入`
- **位置**: `src/etf_strategy/auditor/core/engine.py` → `rebalance()`

### P2: BT Margin 订单失败 (61次)
- **问题**: COC 模式下 BT 仍检查现金导致订单被拒
- **修复**: `cerebro.broker.set_checksubmit(False)`
- **位置**: `src/etf_strategy/auditor/core/backtester.py`

### P3: VEC 浮点精度 (关键买入被拒)
- **问题**: `cost > cash` 因浮点误差拒绝合法买入
- **修复**: `if cost <= cash + 1e-5`
- **位置**: `scripts/batch_vec_backtest.py`

### P4: 择时信号双重滞后
- **问题**: VEC 内部再次滞后已滞后的 timing
- **修复**: 统一使用 `shift_timing_signal()` 预处理
- **位置**: `src/etf_strategy/core/utils/rebalance.py`

### P5: 调仓日程不一致
- **问题**: VEC/BT 各自计算调仓日导致错位
- **修复**: 统一使用 `generate_rebalance_schedule()`
- **位置**: `src/etf_strategy/core/utils/rebalance.py`

### P6: Numba argsort 不稳定排序 (8.98pp 差异) ⭐ 新发现
- **问题**: `np.argsort` 在 Numba JIT 和 Pure Python 中对相等元素排序顺序不一致
- **现象**: PRICE_POSITION_120D + PRICE_POSITION_20D 组合中，每天约 7-8 个 ETF 得分相同（1.0），选择不同 ETF 导致 8.98pp 收益差异
- **修复**: 新增 `stable_topk_indices()` 函数，使用 ETF 索引作为 tie-breaker
- **位置**: `scripts/batch_vec_backtest.py`
- **关键代码**:
```python
@njit(cache=True)
def stable_topk_indices(scores, k):
    """稳定排序：按 score 降序，score 相同时按索引升序"""
    N = len(scores)
    result = np.empty(k, dtype=np.int64)
    used = np.zeros(N, dtype=np.bool_)
    for i in range(k):
        best_idx = -1
        best_score = -np.inf
        for n in range(N):
            if used[n]:
                continue
            if scores[n] > best_score or (scores[n] == best_score and (best_idx < 0 or n < best_idx)):
                best_score = scores[n]
                best_idx = n
        if best_idx < 0 or best_score == -np.inf:
            return result[:i]
        result[i] = best_idx
        used[best_idx] = True
    return result
```

## 共享常量 (两引擎必须一致)
```python
FREQ = 8
POS_SIZE = 3
INITIAL_CAPITAL = 1_000_000.0
COMMISSION_RATE = 0.0002
LOOKBACK = 252
```

## 关键代码逻辑

### 分数计算 (T-1 日)
- VEC: `factors_3d[t-1, n, idx]`
- BT: `scores.loc[prev_ts]`

### 执行价格 (T 日收盘)
- VEC: `close_prices[t, n]`
- BT: COC 模式 → `data.close[0]`

### 资金分配 (Net-New)
```python
current_value = cash_after_sells + kept_holdings_value
target_exposure = current_value * timing_ratio
available_for_new = max(0, target_exposure - kept_holdings_value)
target_pos_value = available_for_new / new_count / (1 + COMMISSION_RATE)
```

## 验证命令
```bash
# 完整验证流程
uv run python scripts/batch_vec_backtest.py   # VEC 批量
uv run python scripts/batch_bt_backtest.py    # BT 批量

# 预期: 100/100 组合对齐，平均差异 < 0.03pp，最大差异 < 0.05pp
# Margin 失败: 0
```

## 最新验证结果 (2025-11-29)
```
平均差异: 0.0254pp ✅
最大差异: 0.0441pp ✅
对齐率: 100/100 ✅
Margin 失败: 0 ✅

核心策略: CORRELATION_TO_MARKET_20D + MAX_DD_60D + PRICE_POSITION_120D + PRICE_POSITION_20D
VEC: 121.0160%
BT:  121.0601%
差异: 0.0441pp
```

## 详细文档
- `docs/VEC_BT_ALIGNMENT_GUIDE.md` - **完整对齐指南** (推荐阅读)
- `docs/VEC_BT_ALIGNMENT_HISTORY_AND_FIXES.md` - 完整历史
- `docs/archive/VEC_BT_ALIGNMENT_AUDIT_REPORT.md` - 审计报告

## 快速调试技巧

### 排查 Numba 问题
```bash
# 禁用 JIT 运行，对比结果
NUMBA_DISABLE_JIT=1 uv run python scripts/batch_vec_backtest.py
```

### 清除 Numba 缓存
```bash
rm -rf ~/.cache/numba
find . -name "__pycache__" -type d -exec rm -rf {} +
```

### 检查数据是否 contiguous
```python
print(f'contiguous: {arr.flags["C_CONTIGUOUS"]}')
arr = np.ascontiguousarray(arr)  # 转换为连续内存
```
