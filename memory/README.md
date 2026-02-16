# Memory — 项目经验教训知识库

> 本目录存储项目中的重要复盘和教训，供所有 AI 助手和开发者参考。
> 创建日期: 2026-02-12

---

## 📚 文档索引

| 文件 | 日期 | 类型 | 简介 |
|------|------|------|------|
| [rules.md](./rules.md) | 2026-02 | 规则 | 策略验证诊断规则（26条），每次评估策略前必读 |
| [2026-02_performance_optimization_failure.md](./2026-02_performance_optimization_failure.md) | 2026-02-12 | 教训 | 10x加速目标实际仅提升2.4%的复盘 |
| [2026-02_vec_bt_alignment_fix.md](./2026-02_vec_bt_alignment_fix.md) | 2026-02-14 | 教训 | VEC-BT +25.7pp gap 根因：信号态 vs 执行态反馈环 |
| [2026-02_pipeline_metadata_loss.md](./2026-02_pipeline_metadata_loss.md) | 2026-02-14 | 元教训 | 管线信息瓶颈：同一 bug 的三次变体 (Rule 14/19/24) |
| [2026-02_v8_sealing_lessons.md](./2026-02_v8_sealing_lessons.md) | 2026-02-15 | 教训 | v5→v8 四版封板复盘，3个bug累积效应，封板流程检查清单 |
| [2026-02_ic_sign_infrastructure_lesson.md](./2026-02_ic_sign_infrastructure_lesson.md) | 2026-02-16 | 教训 | IC-sign 方向修复：水管通了水一直都在，统计声称修正 |
| [factor_reference.md](./factor_reference.md) | 2026-02-16 | 参考 | 44因子完整手册：属性/IC/方向/v8用途/分桶/架构 |
| [2026-02_alpha_theory_and_data_value.md](./2026-02_alpha_theory_and_data_value.md) | 2026-02-16 | 战略 | Alpha理论、数据价值框架、Phase 1复盘、Phase 2指引 |
| [2026-02_moneyflow_validation.md](./2026-02_moneyflow_validation.md) | 2026-02-17 | 决策 | Moneyflow因子价值验证：与fund_share/margin同维度，放弃 |
| [2026-02_when_dimension_research_brief.md](./2026-02_when_dimension_research_brief.md) | 2026-02-17 | 研究 | WHEN维度研究简报：策略激活时机+策略组合，待验证 |

---

## 🔍 如何使用

### 新 AI 助手加入项目时

1. **必读**：阅读本目录下所有文档
2. **理解**：了解历史教训，避免重复犯错
3. **遵守**：按文档中的检查清单执行

### 添加新教训时

1. 在本目录创建新文件，命名格式：`YYYY-MM_topic.md`
2. 更新本索引文件
3. 在文档中包含：
   - 事件背景
   - 核心错误/问题
   - 检查清单/教训
   - 相关文件引用

---

## 📋 文档模板

```markdown
# [标题]

> 日期: YYYY-MM-DD
> 类型: 教训 / 最佳实践 / 决策记录

## 事件

[描述发生了什么]

## 问题/错误

1. ...
2. ...

## 教训

- [ ] 检查项1
- [ ] 检查项2

## 相关文件

- `path/to/file1`
- `path/to/file2`
```

---

*维护者*: 项目团队 | *用途*: AI 助手 + 人类开发者参考