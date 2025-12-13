# Release Notes — v3.2（封板）

**日期**：2025-12-14  
**目标**：交付一批“可审计、不可质疑”的稳定策略清单（100+），并把 BT 审计口径与训练/冷数据区间严格对齐。

## 核心变化

### 1) 审计口径升级：BT 分段收益（Train / Holdout）
- BT 审计输出新增：`bt_train_return`、`bt_holdout_return`。
- Train / Holdout 分割点来自配置：`training_end_date=2025-04-30`。
- 目的：消除“VEC 与 BT 跨区间比较”导致的可质疑点。

### 2) Production Pack 升级：以 BT 为 Ground Truth
- 生产清单排序使用 BT 指标（OOS-first）：优先 `bt_holdout_return`，辅以 `bt_calmar_ratio`、`bt_max_drawdown`、`bt_train_return`。
- VEC 仅作为筛选层（Screening），最终可交付收益指标以 BT 为准。

### 3) 泄漏控制：Rolling 仅使用训练期 summary
- Rolling OOS consistency summary 使用 train-only 产物（holdout segment count = 0）。
- 目的：避免把 holdout 段信息混入 gate。

## 交付产物（v3.2）

- Triple Validation（无泄漏候选）：
  - `results/final_triple_validation_20251214_011753/final_candidates.parquet`（152）
- BT 审计（含分段收益）：
  - `results/bt_backtest_full_20251214_013635/bt_results.parquet`（152）
- Production Pack（交付 Top 120 + 全量 152）：
  - `results/production_pack_20251214_014022/production_candidates.parquet`（Top120）
  - `results/production_pack_20251214_014022/production_all_candidates.parquet`（All152）
  - `results/production_pack_20251214_014022/PRODUCTION_REPORT.md`
- 说明文档：
  - `docs/PRODUCTION_STRATEGIES_V3_2.md`
  - `docs/PRODUCTION_STRATEGY_LIST_V3_2.md`

## 可复现命令

```bash
# 1) BT 审计（基于 final candidates）
uv run python scripts/batch_bt_backtest.py \
  --combos results/final_triple_validation_20251214_011753/final_candidates.parquet

# 2) 生成 production pack（BT-ground-truth）
uv run python scripts/generate_production_pack.py \
  --candidates results/final_triple_validation_20251214_011753/final_candidates.parquet \
  --bt-results results/bt_backtest_full_20251214_013635/bt_results.parquet \
  --top-n 120
```

## 封板范围（不改交易逻辑）
- 交易规则锁死：FREQ=3、POS=2、无止损、不持有现金（按现有配置与引擎执行）。
- 不改核心引擎与因子库；仅允许 bugfix / 数据更新 / 文档完善 / 审计输出增强。
