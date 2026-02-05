# 🚀 ETF 轮动策略快速参考卡

> **版本**: v3.2 | **更新**: 2025-12-14 | **状态**: 🔒 封板（BT 审计口径已固化）

---

## ⚡ 30 秒速览

```
策略: 43 ETF 高频轮动
收益: 237.45% (5.7年)
参数: FREQ=3, POS=2, LOOKBACK=252
因子: ADX_14D + MAX_DD_60D + PP_120D + PP_20D + SHARPE_20D

⚠️ 关键: 5只QDII贡献90%+收益，禁止移除！
```

---

## 📊 ETF 池结构

| 类别 | 数量 | 收益贡献 | 状态 |
|------|------|---------|------|
| A 股 ETF | 38 | ~147% | ✅ |
| **QDII** | **5** | **+90%** | ⚠️ **禁止移除** |
| 合计 | 43 | 237% | 🔒 |

### 5 只 QDII 明细

| 代码 | 名称 | 贡献 | 胜率 |
|------|------|------|------|
| 513500 | 标普500 | +25% | 69% |
| 513130 | 恒生科技(港元) | +24% | 53% |
| 513100 | 纳指100 | +22% | 61% |
| 159920 | 恒生指数 | +17% | 70% |
| 513050 | 中概互联 | +2% | 44% |

---

## ❌ 禁止操作

1. **移除任何 QDII** → 收益损失 60pp
2. **新增 ETF** → 横截面污染风险
3. **修改 FREQ/POS** → 策略失效

---

## ✅ 允许操作

1. 数据更新（新日期）
2. Bug 修复（不改逻辑）
3. 文档完善
4. 性能优化（不改结果）

---

## 📁 关键文件

| 文件 | 用途 |
|------|------|
| `configs/combo_wfo_config.yaml` | 🔧 主配置 (43 ETF 列表) |
| `docs/ETF_POOL_ARCHITECTURE.md` | 📖 ETF 池深度分析 |
| `docs/BEST_STRATEGY_43ETF_UNIFIED.md` | 📖 最佳策略说明 |
| `AGENTS.md` | 🤖 AI Agent 指南 |

---

## 🔧 常用命令

```bash
# 推荐交付流水线（v3.2：四重验证 + 封板）

# 1) WFO：探索入口（粗筛）
uv run python src/etf_strategy/run_combo_wfo.py

# 2) VEC：向量化精算（Screening）
uv run python scripts/run_full_space_vec_backtest.py

# 3) Rolling + Holdout：无泄漏与一致性验证（产出 final candidates）
uv run python scripts/final_triple_validation.py

# 4) BT：事件驱动审计（Ground Truth，含 Train/Holdout 分段收益）
uv run python scripts/batch_bt_backtest.py

# v3.2 交付：BT Ground Truth Production Pack
uv run python scripts/generate_production_pack.py \
	--candidates results/final_triple_validation_20251214_011753/final_candidates.parquet \
	--bt-results results/bt_backtest_full_20251214_013635/bt_results.parquet \
	--top-n 120

# 5) 封板：冻结产物 + 脚本 + 配置 + 源码快照 + 依赖锁定
uv run python scripts/seal_release.py \
	--version v3.2 --date 20251214 \
	--final-candidates results/final_triple_validation_20251214_011753/final_candidates.parquet \
	--bt-results results/bt_backtest_full_20251214_013635/bt_results.parquet \
	--production-dir results/production_pack_20251214_014022 \
	--force
```

---

## 📦 v3.2 交付产物（直接用于上线）

- 生产候选（Top 120）：`results/production_pack_20251214_014022/production_candidates.parquet`
- 全量候选（All 152）：`results/production_pack_20251214_014022/production_all_candidates.parquet`
- 生产报告：`results/production_pack_20251214_014022/PRODUCTION_REPORT.md`
- 说明文档：`docs/PRODUCTION_STRATEGIES_V3_2.md` / `docs/PRODUCTION_STRATEGY_LIST_V3_2.md`

---

**🔒 v3.2 | BT Ground Truth | QDII=90%+ | 禁止修改 ETF 池**
