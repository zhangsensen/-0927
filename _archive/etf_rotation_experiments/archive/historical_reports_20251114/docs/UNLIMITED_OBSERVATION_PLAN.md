# Unlimited 排序观测期决策准则

- **观测周期**：自 2025-11-11 起两周内至少产生 4 个新的 `run_ts`。
- **数据采集**：每日将最新 run 的 TopK 指标写入 `monitoring/ranking_unlimited_daily.csv`。
- **判定门槛**：
  - 任一 TopK (50/100/200/500/1000) 年化劣化超过 Baseline 0.5%，当日 Gate 标记为失败。
  - 连续 2 日 Gate 失败立即触发回滚。
  - 观测期结束时，若 Gate 失败次数 ≥2 或平均提升 <0.5%，则暂停 Unlimited，回滚到 Baseline。
- **周度复盘**：每周更新 `docs/UNLIMITED_WEEKLY_REPORT.md`，汇总当周 Gate 结果、收益曲线与异常说明。
- **最终决策**：
  - 满足 Gate 且 Top500/Top1000 平均年化提升 ≥0.5%：进入 P1/P2 深化阶段。
  - 未达标：输出问题分析与改进计划，维护现有 Baseline 排序。

