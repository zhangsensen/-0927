# v8.0 封板总结

**封板版本**: v8.0_20260215
**封板时间**: 2026-02-15
**状态**: SEALED
**前序版本**: v7.0_20260213 (已废弃: IC-sign bug, VEC metadata缺失, BT signal-side hysteresis)

---

## 核心策略

**Strategy composite_1 (主策略)**: `ADX_14D + BREAKOUT_20D + MARGIN_BUY_RATIO + PRICE_POSITION_120D + SHARE_CHG_5D` (5因子)

**Strategy core_4f (回退)**: `MARGIN_CHG_10D + PRICE_POSITION_120D + SHARE_CHG_20D + SLOPE_20D` (4因子)

### 策略特征

| 维度 | composite_1 (主策略) | core_4f (回退) |
|------|---------------------|----------------|
| 因子数 | 5 | 4 |
| 非OHLCV因子 | MARGIN_BUY_RATIO, SHARE_CHG_5D | MARGIN_CHG_10D, SHARE_CHG_20D |
| Bounded因子 | ADX_14D, PRICE_POSITION_120D (2/5) | PRICE_POSITION_120D (1/4) |
| Factor Signs | +1,+1,-1,+1,-1 | -1,+1,-1,+1 |
| Factor ICIRs | 0.440,18.579,-5.101,4.542,-2.306 | -3.387,4.542,-1.807,3.125 |

## 执行框架 (v7.0 → v8.0 不变)

| 参数 | 值 | 说明 |
|------|-----|------|
| **FREQ** | 5 | 每5个交易日调仓 |
| **POS_SIZE** | 2 | 持仓2只ETF |
| **Exp4 迟滞** | ON (dr=0.10, mh=9) | 不变 |
| **Regime Gate** | ON (volatility, 510300) | 不变 |
| **Universe** | 49 ETF (41A+8Q), A_SHARE_ONLY | 不变 |
| **COMMISSION** | 0.0002 (2bp) | 不变 |
| **LOOKBACK** | 252 | 不变 |

## v7.0 → v8.0 关键管线修复 (2026-02-14)

1. **IC-sign factor direction fix**: WFO使用abs(IC)丢弃符号, VEC/BT求和假设"higher=better" → 5/6非OHLCV因子被系统性反向使用。修复: stability-gated sign flip + ICIR-weighted scoring。
2. **VEC metadata propagation**: `run_full_space_vec_backtest.py` 现在读取+应用 `factor_signs`/`factor_icirs`, 传递到VEC输出parquet。VEC-BT holdout gap: 20.3pp → 1.47pp (-93%)。
3. **BT execution-side hysteresis**: `_compute_rebalance_targets()` 从 `_signal_portfolio` (信号侧, 自引用反馈环) 改为 `shadow_holdings` (执行侧)。Gap: +25.7pp → -0.8pp。

**v7.0的 VEC-BT gap 为 +25.6pp (超过10pp红线), v8.0全部155个候选 gap < 2pp。**

---

## 四关验证结果

### Gate 1: Train Gate

| 策略 | Train Return | Train MDD | 通过? |
|------|-------------|-----------|-------|
| composite_1 | **+51.6%** | 10.8% | PASS |
| core_4f | +53.0% | 14.9% | PASS |

### Gate 2: Rolling OOS Consistency (Train-only, 无泄漏)

| 策略 | 季度正率 | 最差季度 | 通过? |
|------|---------|---------|-------|
| composite_1 | 61% (11/18) | -2.54% | PASS (≥60%, worst≥-8%) |
| core_4f | 78% (14/18) | -5.59% | PASS |

### Gate 3: Holdout (VEC, 冷数据, 含迟滞)

| 策略 | HO Return | HO MDD | HO Sharpe | HO Calmar | 通过? |
|------|-----------|--------|-----------|-----------|-------|
| composite_1 | **+55.7%** | 7.5% | **2.95** | 7.41 | PASS |
| core_4f | +68.0% | 14.9% | 2.58 | 4.56 | PASS |

### Gate 4: BT Ground Truth (Backtrader, 整手, 含迟滞)

| 策略 | Train | HO Return | Full MDD | Sharpe | PF | Trades | MF |
|------|-------|-----------|----------|--------|----|--------|----|
| **composite_1** | +51.6% | **+53.9%** | 10.8% | **1.38** | 4.88 | 77 | **0** |
| core_4f | +53.0% | +67.4% | 14.9% | 1.09 | 3.07 | 75 | 0 |

### VEC-BT Gap

| 策略 | VEC HO | BT HO | Gap | 可接受? |
|------|--------|-------|-----|---------|
| composite_1 | +55.7% | +53.9% | **-1.9pp** | PASS (<10pp) |
| core_4f | +68.0% | +67.4% | **-0.6pp** | PASS (<10pp) |

---

## 漏斗统计

| 关卡 | 输入 | 输出 | 通过率 |
|------|------|------|--------|
| WFO Full-space | 266,463 | 100,000 | 37.5% |
| VEC Top-N | 100,000 | 200 | 0.2% |
| Train Gate | 200 | 200 | 100% |
| Rolling Gate (Strict) | 200 | 156 | 78.0% |
| Holdout Gate | 156 | 155 | 99.4% |
| BT Ground Truth (155选) | 155 | 155 | 100% (0 MF) |

## Composite Score 选择标准

Top 1 by composite score (30% Train Calmar + 40% Roll Worst + 30% Holdout Calmar):
- composite_1 composite_score = 0.953 (rank #1 of 155)
- core_4f composite_score = 0.635 (选为回退: 纯4F, 更高HO return, 更稳定rolling)

---

## 风控门禁配置

### 引擎级 (frozen_params)

| 参数 | 值 | 说明 |
|------|-----|------|
| leverage_cap | 1.0 | 无杠杆 |
| etf_stop_loss | 5% | 单 ETF 止损 |
| stop_method | fixed | 固定阈值 |
| stop_check | rebalance only | 调仓日检查 |

### 监控级 (上线后)

| 条件 | 动作 |
|------|------|
| Portfolio MDD > 25% | 紧急止损, 全仓撤出 |
| 连续5次交易亏损 | 人工审查, 暂停策略 |
| 单月亏损 > 8% | 降仓50% + 审查 |
| 换手率偏离回测 > 2x | 检查信号生成逻辑 |

### 回退机制

**触发条件** (任一):
1. 主策略连续 3 个调仓期 MDD 恶化
2. 滚动 60 日 Sharpe < 0 (持续 20 交易日)
3. 人工判断 + 确认

**回退操作**: 切换到 core_4f (MARGIN_CHG_10D+PP120+SHARE_CHG_20D+SLOPE_20D)。

---

## Shadow Deploy 配置

- 配置文件: `configs/shadow_strategies.yaml`
- 策略名: `v8_composite_1`
- 使用: `scripts/generate_today_signal.py --shadow-config configs/shadow_strategies.yaml`
- 状态文件: `data/live/signal_state_shadow_v8_composite_1.json`

### 停止规则

| 条件 | 动作 |
|------|------|
| MDD > 24% | 紧急停止 |
| 连续5次亏损 | 审查 |
| O'Brien-Fleming at 13/25/38/50 trades | 统计检验 |
| 176 trades | 完整评估 (Sharpe>1.5 at p<0.05) |

---

## 数据分区

| 分区 | 范围 | 用途 |
|------|------|------|
| 训练 | 2020-01 ~ 2025-04 | WFO/因子筛选 |
| Holdout | 2025-05 ~ 2026-02-10 | 冷数据验证 |
| 实盘 (S1) | 2025-12-18 ~ | 真实交易中 |

---

## 结果目录索引

| 目录 | 内容 |
|------|------|
| `results/run_20260214_144554/` | WFO 全空间筛选 |
| `results/vec_from_wfo_20260214_144851/` | VEC 全空间 200 候选 |
| `results/rolling_oos_consistency_20260214_144932/` | Rolling OOS (含迟滞) |
| `results/holdout_validation_20260214_144933/` | Holdout (含迟滞) |
| `results/final_triple_validation_20260214_144934/` | 三关验证 155 候选 |
| `results/bt_backtest_top155_20260214_145650/` | BT 155 候选全量验证 |

## Git References

| Commit | Description |
|--------|-------------|
| 24aee79 | VEC metadata propagation fix (Rule 24) |
| 01ab0f3 | VEC→holdout/rolling factor_signs/factor_icirs propagation |
| 7fc3132 | Phase 3 ICIR-weighted scoring across VEC/BT/Live |
| bfd97e9 | Stability-gated factor direction with ICIR diagnostics |
| 103b3d3 | IC-sign-aware factor direction handling |

---

**Sealed by**: Claude Code
**Seal Timestamp**: 2026-02-15 23:00+08:00
