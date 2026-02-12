# v3.2 生产策略交付说明（可审计、不可质疑）

**日期**：2025-12-14  
**交付目标**：给出一批可直接进入实盘候选池的稳定策略（100+），并明确“最终口径=BT Ground Truth”。

## 1. v3.2 交付结论（你只需要看这一段）

- 本次交付的生产候选：Top 120（从 152 条无泄漏候选中筛出）。
- 所有收益/回撤/胜率等关键指标以 Backtrader（BT）审计为准，并按 Train / Holdout 分段输出。
- Rolling 稳定性 gate 使用 train-only summary，已规避 holdout 泄漏风险。

## 2. 固定交易规则（封板）

- 调仓频率：FREQ=3（交易日）
- 持仓数量：POS=2
- 手续费：0.0002
- 其他：不止损、不 cash（按现有引擎规则）

## 3. 验证与审计分层（为什么 v3.2 “不可质疑”）

- Screening 层（快，负责筛出候选）：VEC + Rolling + Holdout
- Audit 层（慢，负责给最终可交付口径）：BT（事件驱动回测）

**关键声明**：
- VEC 与 BT 可能因执行假设不同而产生差异；v3.2 生产收益一律采用 BT。
- 因此，任何“对齐质疑”都可以通过查看 BT 的分段收益字段直接回答。

## 4. 数据区间与切分（严格一致）

- Train：2020-01-01 → 2025-04-30
- Holdout：2025-05-01 → 2025-10-14

> 分割点来自 `configs/combo_wfo_config.yaml` 的 `training_end_date`。

## 5. 交付产物（唯一可信来源）

- 生产候选（Top 120）：
  - `results/production_pack_20251214_014022/production_candidates.parquet`
- 全量候选（All 152）：
  - `results/production_pack_20251214_014022/production_all_candidates.parquet`
- 生产报告（Top 120，含 BT 分段收益）：
  - `results/production_pack_20251214_014022/PRODUCTION_REPORT.md`
- 策略列表（人类可读版）：
  - `docs/PRODUCTION_STRATEGY_LIST_V3_2.md`

## 6. 如何挑策略（建议做法）

- 默认从 Top 120 里挑 3–8 条做篮子，避免单策略风险。
- 组合构建优先看：
  - `bt_holdout_return`（OOS-first）
  - `bt_max_drawdown`（全周期最大回撤）
  - `bt_calmar_ratio`（全周期收益/回撤比）
  - `bt_profit_factor` 与 `bt_total_trades`（交易质量与样本量）

## 7. 可复现（两条命令）

```bash
# BT 审计（会输出 bt_train_return / bt_holdout_return）
uv run python scripts/batch_bt_backtest.py \
  --combos results/final_triple_validation_20251214_011753/final_candidates.parquet

# 生产包（Top 120）
uv run python scripts/generate_production_pack.py \
  --candidates results/final_triple_validation_20251214_011753/final_candidates.parquet \
  --bt-results results/bt_backtest_full_20251214_013635/bt_results.parquet \
  --top-n 120
```

## 8. 风险提示（必须读）

- Holdout 窗口较短，且市场风格会变化；实盘必须做持续监控。
- QDII 暴露是主要 Alpha 来源之一（禁止移除 QDII ETF），但也意味着海外市场 regime shift 风险。
- BT 假设理想执行；真实滑点与冲击成本会降低收益。
