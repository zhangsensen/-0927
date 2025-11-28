<!-- ALLOW-MD -->
# 项目审计报告：ETF轮动策略优化 (2025-11-26)

**审计对象**: `/home/sensen/dev/projects/-0927/etf_rotation_optimized`
**参考标准**: `/home/sensen/dev/projects/-0927/etf_rotation_experiments/core/1126.md`
**审计时间**: 2025-11-26

---

## 1. 审计概览

本次审计旨在验证当前生产环境代码是否符合 `1126.md` 中确定的优化结论。

| 检查项 | 目标值 (1126.md) | 当前值 (Codebase) | 状态 |
|:---|:---|:---|:---|
| **选股因子** | MOM_20D, RET_VOL_20D, RSI_14 | WFO Top100 (动态) | ⚠️ **需确认** |
| **调仓频率** | 8天 | 8天 (Config) | ✅ **一致** |
| **持仓数量** | 3只 | 4只 (Default) | ❌ **不一致** |
| **择时模块** | Light Timing (Enabled) | **未实现** | ❌ **缺失** |
| **止损机制** | Disabled | Disabled | ✅ **一致** |
| **回测引擎** | 生产级 (No Lookahead) | 生产级 (No Lookahead) | ✅ **一致** |

---

## 2. 详细发现

### 2.1 择时模块缺失 (Critical)
- **描述**: `1126.md` 明确要求启用 `Light Timing` 模块 (Threshold=-0.4, Position=0.3)。
- **现状**: 
  - `etf_rotation_optimized/core/market_timing.py` 文件不存在。
  - `run_production_backtest.py` 中 `backtest_no_lookahead` 函数虽然文档注释提到了 `timing_arr`，但实际代码签名和逻辑中完全缺失该功能。
- **影响**: 策略无法在极端市场下进行防御，回撤控制无法达到 -15% 的目标（当前预计为 -20%）。

### 2.2 持仓数量配置偏差
- **描述**: 优化结论推荐持仓 **3只** ETF。
- **现状**: 
  - `run_production_backtest.py` 默认参数为 `position_size=4`。
  - `combo_wfo_config.yaml` 中 `test_all_position_sizes` 为 `false`，且注释称“已验证最优值为4”。
- **影响**: 持仓过于分散可能导致收益率略微下降。

### 2.3 因子组合约束
- **描述**: 结论推荐固定使用 `MOM_20D + RET_VOL_20D + RSI_14`。
- **现状**: 当前生产脚本加载 WFO 跑出的 Top 100 组合。虽然 Top 组合中可能包含该黄金组合，但系统并未强制锁定该组合。
- **建议**: 如果目标是固化策略，应在配置中指定单一组合回测，或在 WFO 结果加载后进行筛选。

---

## 3. 修复计划

### 步骤 1: 实现择时模块
创建 `etf_rotation_optimized/core/market_timing.py`，实现 `LightTimingModule`：
- **输入**: 市场指数 (HS300) 和 黄金 ETF (518880) 的价格数据。
- **逻辑**: 
  - MA Signal: Price > MA200 (HS300)
  - Mom Signal: MOM_20D > 0 (HS300)
  - Gold Signal: Price > MA200 (Gold)
  - Composite = 0.4*MA + 0.4*Mom + 0.2*Gold
  - Signal = 0.3 if Composite < -0.4 else 1.0

### 步骤 2: 升级回测引擎
修改 `etf_rotation_optimized/real_backtest/run_production_backtest.py`：
- 更新 `backtest_no_lookahead` 签名，增加 `timing_signal` 参数。
- 在每日循环中，将 `timing_signal[day_idx]` 应用于持仓权重（或总仓位）。

### 步骤 3: 修正配置
修改 `etf_rotation_optimized/configs/combo_wfo_config.yaml`：
- 将默认持仓数说明改为 3。
- (可选) 增加择时相关配置参数。

---

**结论**: 当前生产代码尚未包含 1126 优化的核心成果（择时模块），需立即进行代码补全和集成。
