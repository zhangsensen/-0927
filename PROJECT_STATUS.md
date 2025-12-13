# 项目现状速览（v3.2 已交付 | 2025-12-14）

## ✅ 交付结论（你只需要看这一段）

- 已交付一批“可审计、不可质疑”的稳定策略清单：**Top 120**（来自 152 条无泄漏候选）。
- 生产口径统一为 **BT（Backtrader）Ground Truth**，并按 Train / Holdout 分段输出收益。
- Rolling 稳定性 gate 使用 **train-only summary**，已规避 holdout 泄漏。

## 🔒 封板范围（v3.2）

- 交易规则锁死：FREQ=3、POS=2、commission=0.0002；不止损、不 cash（按现有引擎规则）。
- 允许：数据更新、bugfix（不改逻辑）、性能优化（不改结果）、文档与审计增强。
- 禁止：修改核心回测引擎逻辑、修改 ETF 池定义（尤其禁止移除任何 QDII）。

## 📦 v3.2 关键产物（可追溯、可复现）

### 1) 无泄漏候选（Triple Validation）
- `results/final_triple_validation_20251214_011753/final_candidates.parquet`（152）

### 2) BT 审计（含分段收益）
- `results/bt_backtest_full_20251214_013635/bt_results.parquet`（152，含 `bt_train_return` / `bt_holdout_return`）

### 3) Production Pack（交付）
- `results/production_pack_20251214_014022/production_candidates.parquet`（Top 120）
- `results/production_pack_20251214_014022/production_all_candidates.parquet`（All 152）
- `results/production_pack_20251214_014022/PRODUCTION_REPORT.md`

## 📚 v3.2 文档

- `docs/PRODUCTION_STRATEGIES_V3_2.md`
- `docs/PRODUCTION_STRATEGY_LIST_V3_2.md`
- `docs/RELEASE_NOTES_V3_2.md`

## 🔁 可复现命令

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
