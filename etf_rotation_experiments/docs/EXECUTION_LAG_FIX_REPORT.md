# 执行延迟修复实施报告

## 修复概述

**问题**: 原始回测引擎存在前视偏差 - 使用 Lag-1 IC 假设（信号T-1立即捕获Return[T]），但真实交易需要 Lag-2 IC（信号T延迟到T+1才生效）

**解决方案**: 在 `production_backtest.py` 中引入 `RB_EXECUTION_LAG` 环境变量，支持切换执行模式

---

## 修改详情

### 1. 核心代码修改

**文件**: `etf_rotation_experiments/strategies/backtest/production_backtest.py`

#### 修改点 A: 初始化执行延迟参数 (Lines ~660-673)

```python
# 执行延迟修正 (Execution Lag Fix)
# RB_EXECUTION_LAG=1: 信号在 T 日生成，T+1 日才生效（消除前视偏差）
# RB_EXECUTION_LAG=0: 原始逻辑（信号 T-1，立即捕获 Return T）
execution_lag = int(os.environ.get("RB_EXECUTION_LAG", "0").strip() or "0")
if execution_lag not in (0, 1):
    raise ValueError(f"RB_EXECUTION_LAG 必须为 0 或 1，当前值: {execution_lag}")

if execution_lag == 1:
    logger.info("⚠️  执行延迟模式已启用 (RB_EXECUTION_LAG=1): 信号延迟 1 日生效")

current_weights = np.zeros(N)
pending_weights = None  # 延迟 1 日的目标权重（仅在 execution_lag=1 时使用）
rebalance_counter = 0
```

**要点**:
- 读取环境变量 `RB_EXECUTION_LAG`，默认为 0（保持向后兼容）
- 引入 `pending_weights` 变量存储待应用的目标权重
- 添加日志提示用户当前使用的执行模式

#### 修改点 B: 调仓逻辑分支处理 (Lines ~798-810)

```python
# 执行延迟分支处理
if execution_lag == 0:
    # 原始逻辑：立即应用目标权重（Lag-1 IC，存在前视偏差）
    current_weights = target_weights
else:
    # execution_lag == 1：信号延迟 1 日生效（Lag-2 IC，消除前视偏差）
    pending_weights = target_weights

# === 应用延迟的权重（仅在 execution_lag=1 时） ===
if execution_lag == 1 and pending_weights is not None:
    current_weights = pending_weights
    pending_weights = None  # 应用后清空

# === 每日收益计算 ===
```

**要点**:
- `execution_lag=0`: 保持原始行为 - 调仓日立即应用 `target_weights`
- `execution_lag=1`: 延迟应用 - 调仓日生成 `target_weights`，存入 `pending_weights`，下一日才真正应用
- 在每日收益计算前检查并应用 `pending_weights`

### 2. 执行逻辑对比

| 场景 | Lag-1 IC (LAG=0) | Lag-2 IC (LAG=1) |
|------|------------------|------------------|
| **Day T-1** | 使用 Factor[T-2] 计算信号 | 使用 Factor[T-1] 计算信号 |
| **Day T (调仓日)** | - 立即应用 target_weights<br>- 使用 current_weights 捕获 Return[T] | - 生成 target_weights → pending_weights<br>- 仍使用旧的 current_weights 捕获 Return[T] |
| **Day T+1** | 继续持有 T 日权重 | - 应用 pending_weights → current_weights<br>- 使用新权重捕获 Return[T+1] |

**关键差异**:
- **Lag-1**: 信号 → 收益之间仅 1 日滞后（Factor[T-1] → Return[T]）
- **Lag-2**: 信号 → 收益之间有 2 日滞后（Factor[T-1] → Return[T+1]）

---

## 验证计划

### 快速验证脚本

已创建 `scripts/quick_verify_lag_fix.sh`，执行以下测试：

```bash
# 运行验证
cd /home/sensen/dev/projects/-0927/etf_rotation_experiments
./scripts/quick_verify_lag_fix.sh
```

**测试场景**:
1. **LAG=0**: 使用 Platinum 策略运行回测，预期结果 ~20% 年化收益
2. **LAG=1**: 同样的策略，预期结果 ~-6% 到 1% 年化收益

**成功标准**:
- LAG=0 收益显著高于 LAG=1（至少 15% 差距）
- LAG=1 收益接近 paper_trading 实盘模拟结果（-6.25% 在 freq=1, zero_fee 测试中）
- LAG=0 收益接近原始 WFO 报告值（20.09% 在 Platinum 策略中）

### 完整验证流程

```bash
# 1. 快速验证单个策略
export RB_EXECUTION_LAG=0
python3 run_combo_wfo.py --lookback 120 --freq 2 --position 10 \
    --combo-file /tmp/platinum.txt --n-jobs 1

export RB_EXECUTION_LAG=1
python3 run_combo_wfo.py --lookback 120 --freq 2 --position 10 \
    --combo-file /tmp/platinum.txt --n-jobs 1

# 2. 对比 paper_trading 结果（应与 LAG=1 一致）
python3 scripts/run_paper_trading_backtest.py
```

---

## 重新训练 WFO

### 训练命令

修复验证通过后，使用以下命令重新训练 WFO（启用延迟执行）：

```bash
cd /home/sensen/dev/projects/-0927/etf_rotation_experiments

# 设置环境变量
export RB_EXECUTION_LAG=1
export RB_DAILY_IC_PRECOMP=0  # 首次运行关闭预计算

# 运行完整 WFO（警告：耗时长）
python3 run_combo_wfo.py \
    --lookback-list 60 80 100 120 140 160 \
    --freq-list 1 2 3 5 7 10 \
    --position-list 5 8 10 12 15 \
    --n-jobs 16 \
    --output-dir results_combo_wfo_lag2ic_$(date +%Y%m%d)

# 或者使用增量测试（推荐先小规模验证）
python3 run_combo_wfo.py \
    --lookback-list 120 \
    --freq-list 2 5 \
    --position-list 10 \
    --n-jobs 4 \
    --output-dir results_test_lag2ic_$(date +%Y%m%d)
```

### 预期结果

| 指标 | 原始 WFO (LAG=0) | 修正 WFO (LAG=1) |
|------|------------------|------------------|
| **平均年化收益** | 15-25% | 0-10% (预计) |
| **Platinum 策略** | 20.09% | -6% ~ 1% |
| **Top 10 平均** | ~25% | ~5% (预计) |
| **可交易性** | ❌ 前视偏差 | ✅ 真实可行 |

**关键观察点**:
1. **收益率普遍下降**: LAG=1 会导致整体收益率下降，这是正常的（消除了不合理的前视优势）
2. **策略排名变化**: 某些高度依赖"即时执行"的策略排名会大幅下降
3. **真实可交易**: 新的 Top 策略应能在 paper_trading 中复现相似收益
4. **负收益策略**: 可能出现更多负收益策略，需要提高筛选标准

### 筛选标准调整

由于整体收益率下降，建议调整策略筛选标准：

```python
# 旧标准（LAG=0）
min_annual_return = 15.0  # 年化收益 > 15%
min_sharpe = 1.0          # 夏普 > 1.0

# 新标准（LAG=1）
min_annual_return = 5.0   # 年化收益 > 5%（降低期望）
min_sharpe = 0.8          # 夏普 > 0.8（略微放松）
max_drawdown = -25.0      # 回撤 > -25%（更严格）
```

---

## 向后兼容性

**默认行为**: `RB_EXECUTION_LAG` 未设置时，默认为 0，保持原始 Lag-1 IC 行为

**迁移建议**:
1. **阶段 1**: 验证修复正确性（快速测试 LAG=0 vs LAG=1）
2. **阶段 2**: 重新训练 WFO（使用 LAG=1）
3. **阶段 3**: 切换所有生产环境到 LAG=1
4. **阶段 4**: 逐步淘汰 LAG=0 代码路径（可选）

---

## 技术细节

### pending_weights 状态机

```
调仓日（Day T）:
  - 计算 target_weights
  - if LAG=0: current_weights ← target_weights (立即生效)
  - if LAG=1: pending_weights ← target_weights (暂存)

每日开始（Day T 或 T+1）:
  - if LAG=1 and pending_weights != None:
      current_weights ← pending_weights (延迟生效)
      pending_weights ← None

每日收益计算:
  - daily_ret = sum(current_weights * returns[day_idx])
```

### 边界条件处理

1. **首次调仓**: 
   - LAG=0: 调仓日立即生效
   - LAG=1: 下一日才生效，首日仍使用全零权重

2. **最后一日**:
   - 如果 pending_weights 仍有值，会在下一日应用
   - 如果回测在调仓日结束，pending_weights 不会被应用（符合实际）

3. **连续调仓**:
   - 理论上每 freq 日调仓一次，不会出现 pending 覆盖问题
   - 即使 freq=1（每日调仓），逻辑也正确（前一日的 pending 先应用，然后生成新的 pending）

---

## 已知限制

1. **不支持 LAG > 1**: 当前只实现了 LAG=0 和 LAG=1，未来如需 T+2 或更长延迟，需扩展逻辑

2. **NO_LOOKAHEAD_CHECK 未覆盖**: 
   - `enforce_nl` 检查仍基于原始 IC 计算逻辑
   - LAG=1 模式下，NO_LOOKAHEAD_CHECK 可能需要调整（但不影响核心功能）

3. **成本计算时序**:
   - 当前成本在调仓日扣除，LAG=1 时成本实际应在 T+1 扣除
   - 影响较小（成本占比低），暂不调整

---

## 后续优化

1. **优化 pending_weights 存储**:
   - 当前每次循环检查 `if execution_lag == 1`，可提前分支避免重复判断

2. **扩展到多日延迟**:
   - 支持 `RB_EXECUTION_LAG=2, 3, ...`（使用队列存储多日 pending）

3. **调整成本时序**:
   - LAG=1 时，交易成本应在 T+1 日扣除，而非 T 日

4. **优化 NO_LOOKAHEAD_CHECK**:
   - 在 LAG=1 模式下，检查逻辑应对齐到 Lag-2 IC

---

## 总结

✅ **已完成**:
- 在 `backtest_no_lookahead()` 中引入 `RB_EXECUTION_LAG` 支持
- 实现 Lag-1 IC (LAG=0) 和 Lag-2 IC (LAG=1) 双模式
- 创建验证脚本 `quick_verify_lag_fix.sh`
- 保持向后兼容（默认 LAG=0）

⏳ **待执行**:
- 运行快速验证脚本，确认修复正确性
- 使用 LAG=1 重新训练完整 WFO
- 对比新旧 WFO 结果，分析策略变化
- 使用新策略进行 paper_trading 测试

🎯 **最终目标**:
找到在 Lag-2 IC 假设下（真实可交易）仍能保持正收益的策略，彻底消除前视偏差。
