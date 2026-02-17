# Memory — 项目经验教训知识库

> 本目录存储项目中的重要复盘和教训，供所有 AI 助手和开发者参考。
> 创建日期: 2026-02-12 | 最后更新: 2026-02-17

---

## 当前状态

**v8.0 sealed (2026-02-15)** — 三维度天花板已确认，纯运维模式。

```
WHAT (选什么)  → 已穷尽: Kaiser 5/17, Phase1无组合超越v8.0
WHEN (何时信任) → 已关闭: 离散度⊂波动率(rho=0.538) + 零预测力
HOW  (怎么组合) → 已关闭: PS降级崩塌(Sharpe -75%)
未来方向      → 策略级条件切换 (需新正交数据源, 当前不可执行, 无活跃研究)
```

下一评估点: **~2026-04-15** (shadow 8-12周后评估 S1→v8.0 切换)

---

## 文档索引

| 文件 | 日期 | 类型 | 简介 |
|------|------|------|------|
| [rules.md](./rules.md) | 2026-02 | 规则 | 策略验证诊断规则（33条），每次评估策略前必读 |
| [2026-02_alpha_theory_and_data_value.md](./2026-02_alpha_theory_and_data_value.md) | 2026-02-17 | **战略** | Alpha理论、数据价值框架、三维度天花板确认、未来指引 |
| [2026-02_when_dimension_research_brief.md](./2026-02_when_dimension_research_brief.md) | 2026-02-17 | **完结** | WHEN/HOW维度研究完结报告：三方向全KILL |
| [2026-02_moneyflow_validation.md](./2026-02_moneyflow_validation.md) | 2026-02-17 | 决策 | Moneyflow因子验证：与fund_share/margin同维度，放弃 |
| [factor_reference.md](./factor_reference.md) | 2026-02-16 | 参考 | 44因子完整手册：属性/IC/方向/v8用途/分桶/架构 |
| [2026-02_ic_sign_infrastructure_lesson.md](./2026-02_ic_sign_infrastructure_lesson.md) | 2026-02-16 | 教训 | IC-sign方向修复：水管通了水一直都在 |
| [2026-02_v8_sealing_lessons.md](./2026-02_v8_sealing_lessons.md) | 2026-02-15 | 教训 | v5→v8四版封板复盘，3个bug累积效应 |
| [2026-02_vec_bt_alignment_fix.md](./2026-02_vec_bt_alignment_fix.md) | 2026-02-14 | 教训 | VEC-BT +25.7pp gap根因：信号态vs执行态反馈环 |
| [2026-02_pipeline_metadata_loss.md](./2026-02_pipeline_metadata_loss.md) | 2026-02-14 | 元教训 | 管线信息瓶颈：同一bug的三次变体 |
| [2026-02_performance_optimization_failure.md](./2026-02_performance_optimization_failure.md) | 2026-02-12 | 教训 | 10x加速目标实际仅提升2.4%的复盘 |
| [METHOD_TEMPLATE.md](./METHOD_TEMPLATE.md) | 2026-02-16 | 参考 | 方法论文档模板 |

---

## 新 AI 助手快速上手

### 必读顺序

1. **CLAUDE.md** — 项目规则和架构
2. **rules.md** — 33条验证规则 (最关键)
3. **2026-02_alpha_theory_and_data_value.md** — 战略全景
4. **2026-02_when_dimension_research_brief.md** — 最新研究完结报告

### 关键认知

- v8.0 是**三维度天花板** (WHAT+WHEN+HOW 全部量化确认)
- **不要**在现有数据上做因子研究 — 已系统性穷尽
- **不要**尝试 Capital Split ensemble — Rule 32 已证明 PS 降级崩塌
- **不要**用截面离散度做 WHEN 信号 — Rule 33 已证明与波动率同维度
- **未来提升路径** (当前不可执行): 新正交数据源 (北向资金优先) → 策略级条件切换框架
- 当前正确操作: shadow 监控 + 日常信号生成 + 数据更新

### 日常运维命令

```bash
make update-data    # 更新市场数据
make signal         # 生成今日交易信号 (含shadow)
```

---

## 添加新文档

1. 命名: `YYYY-MM_topic.md`
2. 更新本索引
3. 更新 `rules.md` (如有新规则)
4. 更新 alpha theory 文档 (如有战略性发现)
