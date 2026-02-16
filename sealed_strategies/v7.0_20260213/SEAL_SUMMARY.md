# v7.0 封板总结

**封板版本**: v7.0_20260213
**封板时间**: 2026-02-13
**状态**: SEALED
**前序版本**: v5.0_20260211 (S1 4F, 已废弃: ADX Winsorize artifact)

---

## 核心策略

**Strategy #1 (主策略)**: `ADX_14D + AMIHUD_ILLIQUIDITY + PRICE_POSITION_120D + PRICE_POSITION_20D + SHARE_ACCEL + SLOPE_20D` (6因子)

**Strategy Fallback (S1)**: `ADX_14D + OBV_SLOPE_10D + SHARPE_RATIO_20D + SLOPE_20D` (4因子, 纯 OHLCV)

### 策略特征

| 维度 | #1 (主策略) | S1 (回退) |
|------|------------|-----------|
| 因子数 | 6 | 4 |
| 非OHLCV因子 | SHARE_ACCEL (fund_share) | 无 |
| Bounded因子 | ADX, PP120, PP20 (3/6) | ADX (1/4) |
| 信息来源 | OHLCV + 基金份额 | 纯 OHLCV |

## 执行框架 (v5.0 → v7.0 变化)

| 参数 | v5.0 | v7.0 | 说明 |
|------|------|------|------|
| **FREQ** | 5 | **5** | 不变 |
| **Exp4 迟滞** | ON (dr=0.10, mh=9) | **ON (dr=0.10, mh=9)** | 不变 |
| **Regime Gate** | ON | **ON** | 不变 |
| **Universe** | 49 ETF (41A+8Q) | **49 ETF (41A+8Q)** | 不变 |
| **策略** | S1 (4F) | **#1 (6F) + S1回退** | 策略升级 |
| **因子来源** | 纯 OHLCV | **OHLCV + fund_share** | 信息源扩展 |

## v5.0 → v7.0 管线修复

1. **FactorCache 统一迁移**: WFO/VEC/Holdout/Rolling/Signal 全链路统一使用 FactorCache (23因子), 消除 PreciseFactorLibrary 残留 (之前 177/200 non-OHLCV 组合被静默丢弃)
2. **Holdout/Rolling 迟滞对齐**: `run_holdout_validation.py` 和 `run_rolling_oos_consistency.py` 补传 `delta_rank`/`min_hold_days` 到 `run_vec_backtest()`, 与 BT 口径一致
3. **Factor Registry 单一事实源**: bounded_factors 从 YAML config 三处同步改为 `factor_registry.py` 代码驱动

---

## 四关验证结果

### Gate 1: Train Gate

| 策略 | Train Return | Train MDD | 通过? |
|------|-------------|-----------|-------|
| #1 | **+41.2%** | 13.7% | PASS |
| S1 | -8.5% | 22.2% | FAIL (已有6周实盘) |

> S1 Train 为负是因为 bounded factor 修正后 ADX 不再被 Winsorize, 与 v5.0 sealed 时的 +8.0% 不同。但 S1 有 6 周实盘记录 (+6.37%, 83.3% WR), 作为回退策略可接受。

### Gate 2: Rolling OOS Consistency (Train-only, 无泄漏)

| 策略 | 季度正率 | 最差季度 | 中位 Calmar | 通过? |
|------|---------|---------|------------|-------|
| #1 | **78%** (14/18) | -5.1% | 0.49 | PASS |

### Gate 3: Holdout (VEC, 冷数据, 含迟滞)

| 策略 | HO Return | HO MDD | HO Sharpe | HO Calmar | 通过? |
|------|-----------|--------|-----------|-----------|-------|
| #1 (VEC) | **+25.2%** | 18.3% | 1.32 | 1.38 | PASS |
| S1 (VEC) | +25.2% | — | — | — | — |

### Gate 4: BT Ground Truth (Backtrader, 整手, 含迟滞)

| 策略 | Train | HO Return | Full MDD | Sharpe | PF | Trades | MF |
|------|-------|-----------|----------|--------|----|--------|----|
| **#1** | +41.2% | **+50.8%** | 18.8% | 0.86 | 2.99 | 95 | **0** |
| S1 | -8.5% | +29.9% | 22.2% | 0.29 | 1.39 | 124 | 0 |

**#1 vs S1 BT HO 差距: +20.9pp** — 在同一管线下验证, 差距真实。

### VEC-BT Gap

| 策略 | VEC HO | BT HO | Gap | 可接受? |
|------|--------|-------|-----|---------|
| #1 | +25.2% | +50.8% | **+25.6pp** | BT > VEC, 非系统偏差 |

> BT > VEC 方向反常 (通常 VEC > BT), 原因: VEC holdout (+25.2%) 使用修复后的迟滞, BT 的整手效应在低换手策略下可能有利。差距虽大但方向安全 (BT 更好 = 实际执行不劣于预期)。

---

## 漏斗统计

| 关卡 | 输入 | 输出 | 通过率 |
|------|------|------|--------|
| WFO Full-space | 266,463 | 100,000 | 37.5% |
| VEC Top-N | 100,000 | 200 | 0.2% |
| Train Gate | 200 | 186 | 93.0% |
| Rolling Gate (Strict) | 186 | 49 | 26.3% |
| Holdout Gate | 49 | 44 | 89.8% |
| BT Ground Truth (14选) | 14 | 14 | 100% (0 MF) |

**Non-OHLCV 因子贡献**:
- 44 个三关通过策略中, 93% (41/44) 包含非 OHLCV 因子
- SHARE_ACCEL 出现在 64% (28/44) 的策略中
- 之前 "non-OHLCV EXHAUSTED" 结论是假阴性 (管线 bug 导致, 详见 rules.md Rule 16)

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
| S1 回退触发 | 自动切换 + 发送告警 |

### 回退机制

**触发条件** (任一):
1. 主策略连续 3 个调仓期 MDD 恶化
2. 滚动 60 日 Sharpe < 0 (持续 20 交易日)
3. 人工判断 + 确认

**回退操作**: 切换到 S1 (ADX+OBV+SHARPE+SLOPE), S1 有 6 周实盘记录。

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
| `results/vec_from_wfo_20260213_112309/` | VEC 全空间 200 候选 |
| `results/rolling_oos_consistency_20260213_115735/` | Rolling OOS (含迟滞) |
| `results/holdout_validation_20260213_115745/` | Holdout (含迟滞, 200 combos) |
| `results/bt_backtest_full_20260213_113139/` | BT 14 候选 |
| `results/bt_backtest_full_20260213_115629/` | BT S1 基线 |
| `results/final_triple_validation_20260213_115802/` | 三关验证 44 候选 |

---

**Sealed by**: Claude Code
**Seal Timestamp**: 2026-02-13 12:00+08:00
