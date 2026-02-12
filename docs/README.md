# ETF 轮动策略研究平台 — 文档索引

> **版本**: v5.0 (sealed 2026-02-11)
> **最后更新**: 2026-02-12
> **状态**: S1 生产运行中 | C2 Shadow 候选中

---

## 核心文档

| 文档 | 说明 |
|------|------|
| [../CLAUDE.md](../CLAUDE.md) | LLM 开发指南（Claude Code 必读） |
| [../README.md](../README.md) | 项目总览与快速开始 |
| [../PROJECT_STATUS.md](../PROJECT_STATUS.md) | 当前项目状态、策略表现与研究汇总 |

---

## 架构与引擎

| 文档 | 说明 |
|------|------|
| [WFO_VEC_STATUS.md](WFO_VEC_STATUS.md) | 三层引擎对齐状态 + P0 修复记录 |
| [ETF_POOL_ARCHITECTURE.md](ETF_POOL_ARCHITECTURE.md) | 49 ETF 池设计（41 A股 + 8 QDII） |
| [DATA_FORMAT_SPECIFICATION.md](DATA_FORMAT_SPECIFICATION.md) | Parquet 数据格式规范 |
| [KNOWN_DIFFERENCES.md](KNOWN_DIFFERENCES.md) | 已知引擎差异清单 |

---

## 策略与生产

| 文档 | 说明 |
|------|------|
| [v5_prod1_go_live_summary.md](v5_prod1_go_live_summary.md) | v5.0 上线总结 |
| [STRATEGY_DEVELOPMENT_WORKFLOW.md](STRATEGY_DEVELOPMENT_WORKFLOW.md) | 策略开发流程 |

---

## 研究文档

| 文档 | 结论 | 说明 |
|------|------|------|
| [research/algebraic_factor_vec_validation.md](research/algebraic_factor_vec_validation.md) | 边际 | 代数因子 VEC 验证（6 个 BT 候选） |
| [research/bucket_constraints_ablation.md](research/bucket_constraints_ablation.md) | POSITIVE | 跨桶约束 +4.9pp |
| [research/conditional_factor_negative_results.md](research/conditional_factor_negative_results.md) | NEGATIVE | 条件因子切换 — 5 假设全推翻 |
| [research/sector_constraint_negative_results.md](research/sector_constraint_negative_results.md) | NEGATIVE | 行业约束 — MDD 恶化 |

---

## 历史与审计

| 文档 | 说明 |
|------|------|
| [PROJECT_DEEP_DIVE.md](PROJECT_DEEP_DIVE.md) | 项目技术深潜 (v3.4 基线) |
| [OVERFITTING_DIAGNOSIS_REPORT.md](OVERFITTING_DIAGNOSIS_REPORT.md) | 过拟合诊断报告 |
| [FACTOR_EXPANSION_V42.md](FACTOR_EXPANSION_V42.md) | v4.2 因子扩展 |
| [../reports/deep_review_final_report.md](../reports/deep_review_final_report.md) | 深度审阅报告 (6P0 + 8P1 + 9P2) |
| [archive/](archive/) | v3.0~v3.2 历史文档（已归档，仅供考古） |

---

## GPU 加速（实验性）

| 文档 | 说明 |
|------|------|
| [GPU_OPTIMIZATION_GUIDE.md](GPU_OPTIMIZATION_GUIDE.md) | GPU IC 计算加速指南 |
| [GPU_IMPLEMENTATION_SUMMARY.md](GPU_IMPLEMENTATION_SUMMARY.md) | GPU 实施总结 |

---

## 封存策略

```
sealed_strategies/
├── v3.1_20251216/     # FREQ=3 基线
├── v3.4_20251220/     # 稳定版封板
├── v4.0_20260131/     # 16因子正交集
├── v4.1_20260203/     # cost model引入
├── v4.2_20260205/     # 因子扩展
├── v5.0_20260211/     # 当前生产版 (FREQ=5 + Exp4)
└── c2_shadow_20260211/ # C2影子策略
```

---

## 三层引擎快速导航

```
WFO (筛选)  →  VEC (精算)  →  BT (审计)
FREQ=5         FREQ=5         FREQ=5
Hysteresis ON  Hysteresis ON  Hysteresis ON
~2 min         ~5 min         ~30-60 min
```

**关键原则**: 任何新因子/信号必须在生产执行框架 (FREQ=5 + Exp4 + Regime Gate) 下评估。
