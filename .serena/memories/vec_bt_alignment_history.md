# VEC vs BT 对齐问题历史记录

> **重要**: 此文档记录了 VEC/BT 对齐过程中的所有问题和修复方案。必读！

## 最终状态: ✅ 完全对齐 (差异 < 0.1pp)

## 5 个核心问题及修复

### P1: BT 资金计算错误 (-6.42pp)
- **问题**: BT 使用 `broker.getvalue()` (卖出前净值) 计算买入目标
- **修复**: 改用 `cash_after_sells = broker.getcash() + 预计卖出收入`
- **位置**: `strategy_auditor/core/engine.py` → `rebalance()`

### P2: BT Margin 订单失败 (61次)
- **问题**: COC 模式下 BT 仍检查现金导致订单被拒
- **修复**: `cerebro.broker.set_checksubmit(False)`
- **位置**: `strategy_auditor/core/backtester.py`

### P3: VEC 浮点精度 (关键买入被拒)
- **问题**: `cost > cash` 因浮点误差拒绝合法买入
- **修复**: `if cost <= cash + 1e-5`
- **位置**: `scripts/batch_vec_backtest.py`

### P4: 择时信号双重滞后
- **问题**: VEC 内部再次滞后已滞后的 timing
- **修复**: 统一使用 `shift_timing_signal()` 预处理
- **位置**: `etf_rotation_optimized/core/utils/rebalance.py`

### P5: 调仓日程不一致
- **问题**: VEC/BT 各自计算调仓日导致错位
- **修复**: 统一使用 `generate_rebalance_schedule()`
- **位置**: `etf_rotation_optimized/core/utils/rebalance.py`

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
uv run python scripts/full_vec_bt_comparison.py
# 预期: VEC 34.8111% vs BT 34.8110%, 差异 < 0.01pp
```

## 详细文档
- `docs/VEC_BT_ALIGNMENT_HISTORY_AND_FIXES.md` - 完整历史
- `docs/VEC_BT_ALIGNMENT_AUDIT_REPORT.md` - 审计报告
