# 项目现状速览（v3.4 生产 | 2026-02-05 更新）

## 当前版本

- **生产版本**: v3.4 (sealed: 2025-12-16)
- **策略 #1**: ADX_14D + OBV_SLOPE_10D + SHARPE_RATIO_20D + SLOPE_20D → 136.52%
- **策略 #2**: + PRICE_POSITION_120D → 129.85%, MaxDD 13.93%

## 封板范围

- 交易规则锁死：FREQ=3、POS=2、commission=0.0002
- 允许：数据更新、bugfix（不改逻辑）、性能优化（不改结果）
- 禁止：修改核心回测引擎逻辑、修改 ETF 池定义（尤其禁止移除任何 QDII）

## 封存产物

所有可追溯产物已归档至 sealed strategies：

| 版本 | 路径 | 内容 |
|------|------|------|
| v3.4 | `sealed_strategies/v3.4_20251216/` | 生产候选 + 源码快照 + 校验和 |
| v3.3 | `sealed_strategies/v3.3_20251216/` | 含 Regime Gate 版本 |
| v3.1 | `sealed_strategies/v3.1_20251215/` | 初版封板 |

## 可复现命令

```bash
# 完整流水线（WFO → VEC → BT → 验证）
uv run python scripts/run_full_pipeline.py

# 单步执行
uv run python src/etf_strategy/run_combo_wfo.py         # WFO 筛选
uv run python scripts/batch_vec_backtest.py              # VEC 回测
uv run python scripts/batch_bt_backtest.py               # BT 审计
uv run python scripts/final_triple_validation.py         # 三门验证
uv run python scripts/generate_production_pack.py        # 生产包

# 每日信号
uv run python scripts/generate_today_signal.py

# 数据更新
uv run python scripts/update_daily_from_qmt_bridge.py --all
```

## 关键文档

- 完整项目说明：`docs/PROJECT_DEEP_DIVE.md`
- ETF 池架构：`docs/ETF_POOL_ARCHITECTURE.md`
- 快速参考：`docs/QUICK_REFERENCE.md`
