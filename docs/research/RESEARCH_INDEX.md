# 研究方向索引

> **最后更新**: 2026-02-16
> **当前阶段**: Phase 1 完成 → Phase 2 待启动

---

## 状态定义

| 状态 | 含义 |
|------|------|
| **EXHAUSTED** | 方向已穷尽，无进一步研究价值 |
| **COMPLETED** | 研究完成，有明确结论 |
| **IN_PROGRESS** | 正在进行 |
| **PENDING** | 待启动 |
| **BLOCKED** | 被阻塞（数据/技术/依赖） |

---

## 研究方向总览

### 🔴 EXHAUSTED（穷尽方向）

这些方向已经充分研究，结论明确，不应再投入资源。

| 方向 | 结论 | 关键发现 | 文档 |
|------|------|---------|------|
| **Phase 1: Non-OHLCV 优化** | v8.0 最优 | 23 因子空间已饱和，200 combos 无一超越 composite_1 | [phase1_non_ohlcv_optimization_20260216.md](phase1_non_ohlcv_optimization_20260216.md) |
| **条件因子切换** | NEGATIVE | 5 个假设全推翻，+15pp 是路径依赖 artifact | [conditional_factor_negative_results.md](conditional_factor_negative_results.md) |
| **行业约束** | NEGATIVE | 同行业双持已是最优，MDD 反而恶化 | [sector_constraint_negative_results.md](sector_constraint_negative_results.md) |
| **代数因子挖掘** | MARGINAL | GP 挖掘 78 个代数因子，仅 6 个 BT 候选，边际递减 | [algebraic_factor_vec_validation.md](algebraic_factor_vec_validation.md) |
| **C2 Shadow** | SUPERSEDED | 被 v8.0 core_4f 取代，不再独立追踪 | [c2_alpha_reality_check_20260212.md](c2_alpha_reality_check_20260212.md) |

**核心认知**: OHLCV 衍生因子的信息空间已近饱和（Kaiser 有效维度 5/17），突破需要新数据源。

---

### 🟢 COMPLETED（已完成）

| 方向 | 结论 | 收益 | 文档 |
|------|------|------|------|
| **跨桶约束** | POSITIVE | HO +4.9pp | [bucket_constraints_ablation.md](bucket_constraints_ablation.md) |
| **v8.0 管线修复** | COMPLETED | VEC-BT gap 25pp→2pp | 见 `memory/rules.md` Rule 22/24/26 |
| **v8.0 封板** | SEALED | composite_1 + core_4f | `sealed_strategies/v8.0_20260215/` |

---

### 🟡 PENDING（待启动）

#### Phase 2: 新信息源因子开发

| 实验 | 优先级 | 数据源 | 因子候选 | 可行性 |
|------|--------|--------|---------|--------|
| **B4 汇率** | ⭐⭐⭐ | AkShare BOC FX | USD_CNY_MOM_5D, FX_CARRY | **数据已有**，1 天可验证 |
| **B2 北向资金** | ⭐⭐ | Tushare moneyflow_hsgt | NORTHBOUND_NET_5D, NB_ACCEL | 需确认个股→ETF 映射 |
| **B1 IOPV 折溢价** | ⭐ | QMT/Wind 实时 | IOPV_PREMIUM_5D | 数据管道待建，2-3 天 |
| **B3 期权 IV** | ⭐ | Tushare opt_daily | IV_RANK_20D, IV_SKEW | 仅 50/300ETF，覆盖面窄 |

**新增建议**: 探索 Family A + Family B 组合（纯算法，无需新数据）
- Family A (composite_1): BREAKOUT + MARGIN_BUY + SHARE_CHG_5D → 高 Sharpe, 低 MDD
- Family B (core_4f): MARGIN_CHG + PP120 + SLOPE → 高绝对收益, 高稳定性

---

## 因子研究矩阵

### 因子家族

| 家族 | 代表因子 | IC 方向 | Exp4 兼容性 | v8.0 使用 |
|------|---------|---------|------------|----------|
| **趋势动量** | SLOPE_20D, BREAKOUT_20D | + | ✅ 稳定 | composite_1, core_4f |
| **价格位置** | PP_20D, PP_120D | + | ⚠️ 不稳定 | PP_120D in both |
| **资金流** | MARGIN_BUY, SHARE_CHG | - | ✅ 稳定 | 两者都有 |
| **流动性** | AMIHUD, CMF | - | ⚠️ 部分 | 仅 AMIHUD 研究 |
| **波动率** | ADX_14D, RSI_14 | + | ✅ 稳定 | composite_1 |

### 因子空间饱和度

```
活跃因子: 23 (17 OHLCV + 6 non-OHLCV)
Kaiser 有效维度: 5/17
PC1 解释度: 59.8%
结论: 空间饱和，新组合边际递减
```

---

## 研究方法论

### 四关验证（必选）

| 关卡 | 检查 | 门槛 |
|------|------|------|
| Train Gate | train_return > 0 | 硬性 |
| Rolling Gate | pos_rate ≥ 60% | 一致性 |
| Holdout Gate | holdout_return > 0 | 冷数据 |
| BT Gate | margin_failures = 0 | 执行可行 |

### VEC-BT Gap 监控

| Gap 范围 | 含义 | 行动 |
|---------|------|------|
| < 2pp | 正常 | 无需行动 |
| 2-5pp | 轻微偏差 | 检查参数一致性 |
| 5-10pp | 可能有遗漏 | Rule 19 检查清单 |
| > 10pp | 结构性 bug | 消融法定位 |

---

## 快速导航

- **项目状态**: `../PROJECT_STATUS.md`
- **开发指南**: `../CLAUDE.md`
- **经验教训**: `../../memory/rules.md`
- **封板策略**: `../../sealed_strategies/v8.0_20260215/SEAL_SUMMARY.md`
