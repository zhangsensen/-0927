# 🧪 分池策略实验项目

> **状态**: 实验性项目  
> **结论**: 分池策略效果不如整体策略，仅作为实验保留

---

## ⚠️ 重要说明

**本目录包含的分池策略实验已证明效果不如整体策略**：

| 策略类型 | 最高收益 | 说明 |
|----------|----------|------|
| **整体策略 (43 ETF)** | **121.0%** | ✅ 生产推荐 |
| 分池策略 (最佳池) | 73.7% | ❌ 实验保留 |
| ATR 动态风控 | 111.6% | ❌ 实验保留 |

### 为什么分池效果更差？

1. **跨资产轮动受限** - 分池后无法在股票/债券/商品间切换
2. **因子效力被削弱** - 同池 ETF 相关性高，CORRELATION 因子失效
3. **样本量不足** - 最小池只有 2-3 只 ETF，选择空间有限
4. **风险规避能力弱** - EQUITY 池下跌时无处可逃

---

## 📁 目录结构

```
experiments/pool_strategies/
├── README.md                 # 本文件
├── scripts/                  # 分池相关脚本
│   ├── run_pool_wfo.py       # 分池 WFO
│   ├── run_pool_vec.py       # 分池 VEC
│   ├── run_all_pools_full_pipeline.py
│   ├── run_all_pools_atr_optimization.py
│   ├── run_all_pools_joint_optimization.py
│   ├── run_all_pools_with_risk.py
│   ├── run_single_pool_full_pipeline.py
│   └── run_allweather_wfo.py
├── results/                  # 分池实验结果
└── docs/                     # 分池相关文档
```

---

## 📊 实验结果汇总

### 分池 WFO + VEC 结果

| 池名 | 收益 | 夏普 | 回撤 | ETF 数量 |
|------|------|------|------|----------|
| EQUITY_CYCLICAL | 73.7% | 0.82 | 19.6% | 6 |
| QDII | 67.2% | 0.76 | 21.2% | 5 |
| EQUITY_DEFENSIVE | 38.6% | 0.43 | 25.7% | 3 |
| A_SHARE_LIVE | 34.9% | 0.40 | 29.5% | 7 |
| EQUITY_GROWTH | 30.3% | 0.36 | 36.2% | 17 |
| EQUITY_BROAD | 27.1% | 0.36 | 26.8% | 7 |
| BOND | 26.9% | 1.28 | 3.7% | 3 |

### ATR 动态风控结果

| 池名 | 收益 | 夏普 | 回撤 | ATR 配置 |
|------|------|------|------|----------|
| EQUITY_CYCLICAL | 111.6% | 1.16 | 17.4% | ATR(10) 1.5×SL 5.0×TP |
| EQUITY_DEFENSIVE | 82.3% | 0.64 | 30.2% | ATR(14) 3.0×SL 3.0×TP |
| A_SHARE_LIVE | 74.0% | 0.64 | 28.6% | ATR(10) 2.0×SL 4.0×TP |

---

## 🔧 如何运行（仅供实验）

```bash
# 切换到实验目录
cd experiments/pool_strategies

# 运行分池 WFO（需要从项目根目录运行）
cd /path/to/project
uv run python experiments/pool_strategies/scripts/run_pool_wfo.py

# 运行分池 VEC
uv run python experiments/pool_strategies/scripts/run_pool_vec.py
```

---

## 📝 变更记录

| 日期 | 操作 |
|------|------|
| 2025-11-28 | 从主项目迁移分池相关代码到实验目录 |

---

**⚠️ 注意：生产环境请使用主项目的整体策略（`run_unified_wfo.py` + `batch_vec_backtest.py`）**
