# WFO / VEC / BT 三层引擎状态

> **最后更新**: 2026-02-12
> **版本**: v5.0 (sealed 2026-02-11, P0 fixes 2026-02-12)

---

## 当前结论

- WFO / VEC / BT 三层参数统一：`FREQ=5`、`POS_SIZE=2`、`LOOKBACK=252`
- Hysteresis 全链路贯通：`delta_rank=0.10`、`min_hold_days=9`（WFO/VEC/BT 三层均已启用）
- 信号链路无前视：因子、择时、regime gate 均在 t-1 计算，调仓在 t+1 open 执行
- Bounded factors 统一为 7 个：ADX_14D, CMF_20D, CORR_MKT_20D, PP_20D, PP_120D, PV_CORR_20D, RSI_14
- 调仓日程：`generate_rebalance_schedule()` 统一 VEC/BT 调仓日
- Regime gate：仅通过 `timing_arr` 应用一次（`vol_regime_series = 1.0`）

---

## P0 修复记录 (2026-02-12)

### P0-1: WFO Hysteresis 对齐 (CRITICAL)

**症状**: WFO 筛选 20,270 组合时未启用 hysteresis，但 VEC/BT 执行时启用 → 排名完全错误

**根因**: `combo_wfo_optimizer.py` 调用 `_compute_rebalanced_return()` 使用默认 `delta_rank=0, min_hold_days=0`

**修复**:
- `ComboWFOConfig` 新增 `delta_rank`/`min_hold_days` 字段
- `_test_combo_single_window()` 传递 hysteresis 参数至 OOS 回测
- 验证: log 确认 `✅ Hysteresis enabled in WFO OOS: delta_rank=0.1, min_hold_days=9`

**影响**: WFO 重跑后 mean OOS return 从 -1.93% 提升至 -0.64% (+1.29pp)，Top 10 完全重排为 ADX+AMIHUD 2-4因子组合

### P0-2: Bounded Factors 不一致

3处定义不一致（4个 vs 7个）→ 统一为 7 个

### P0-3: Auditor FREQ 硬编码

`engine.py` 模块级 `FREQ=3` → `FREQ=5`

### P0-5: Circuit Breaker 误配

`max_dd_pct: 25` 误触发 → 禁用（regime gate 提供仓位控制）

### P0-6: bfill 前视风险

4个脚本 `.ffill().bfill()` → `.ffill().fillna(1.0)`

---

## VEC/BT 对齐

| 指标 | 基线 (无 hysteresis) | 当前 (F5+Exp4) |
|------|---------------------|---------------|
| 中位差 | ~4.8pp | ~5-7pp (holdout) |
| 红线 | > 20pp | > 20pp |
| 原因 | float vs 整手 | + 链式偏差放大 |

**注意**: F5+Exp4 下 VEC-BT gap 可能达 12-22pp（全期），因 hysteresis 的整手取整导致链式分歧。Holdout 口径 ~5-7pp 可接受。

---

## 近期产出

- **WFO 重跑结果**: `results/run_20260212_141151/` (P0-1 修复后，含 hysteresis)
- **VEC 代数因子验证**: `results/vec_full_backtest_20260212_113426/` (27 候选, F5+Exp4)
- **深度审阅报告**: `reports/deep_review_final_report.md` (23 issues, 6P0 + 8P1 + 9P2)
- **测试**: 157/157 通过 (6 test files)

---

## 关键文件

| 组件 | 文件 | 说明 |
|------|------|------|
| WFO | `src/etf_strategy/run_combo_wfo.py` | 入口，读 config 传 hysteresis |
| WFO 优化器 | `src/etf_strategy/core/combo_wfo_optimizer.py` | OOS 评分含 hysteresis |
| VEC | `scripts/batch_vec_backtest.py` | 从 config 读 delta_rank/min_hold_days |
| BT | `scripts/batch_bt_backtest.py` | Backtrader engine, vol_regime=1.0 |
| Hysteresis | `src/etf_strategy/core/hysteresis.py` | @njit 共享内核 |
| Frozen Params | `src/etf_strategy/core/frozen_params.py` | 版本锁 v3.4~v5.0 |
| Config | `configs/combo_wfo_config.yaml` | 单一配置源 |

---

## 后续建议

- **BT ground truth**: 6 个代数因子候选待 BT 验证
- **Shadow C2**: AMIHUD+CALMAR+CORR_MKT 已 BT 通过，8-12 周 shadow 中
- **新数据源**: IOPV/NAV (Exp7)、FX rates (Exp6) 尚未接入
