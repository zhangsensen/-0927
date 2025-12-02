# 🚀 ETF 轮动策略快速参考卡

> **版本**: v3.1 | **更新**: 2025-12-01 | **状态**: 🔒 生产锁定

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
# 运行回测
uv run python src/etf_strategy/run_combo_wfo.py  # WFO
uv run python scripts/batch_vec_backtest.py       # VEC
uv run python scripts/batch_bt_backtest.py        # BT 审计
```

---

**🔒 v3.1 | 237% | QDII=90%+ | 禁止修改 ETF 池**
