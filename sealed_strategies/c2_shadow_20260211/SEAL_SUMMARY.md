# C2 Shadow 候选封板

**封板版本**: c2_shadow_20260211
**封板时间**: 2026-02-11
**状态**: SHADOW CANDIDATE (非生产, 待 8-12 周 Shadow 验证)
**代码基线**: v5.0_20260211 (同一代码库, 仅因子组合不同)

---

## 策略定义

**C2 因子**: `AMIHUD_ILLIQUIDITY + CALMAR_RATIO_60D + CORRELATION_TO_MARKET_20D` (3因子)

| 因子 | 维度桶 | 含义 |
|------|--------|------|
| AMIHUD_ILLIQUIDITY | B (SUSTAINED_POSITION) | 流动性溢价 — 低流动性ETF倾向于有更高风险补偿 |
| CALMAR_RATIO_60D | D (MICROSTRUCTURE) | 60日Calmar比率 — 收益/回撤效率 |
| CORRELATION_TO_MARKET_20D | E (TREND_STRENGTH_RISK) | 市场相关性 — 低相关ETF提供分散化 |

**跨桶覆盖**: 3桶 (B+D+E), 通过 cross-bucket constraint (min_buckets=3)

**对比 S1**: `ADX_14D + OBV_SLOPE_10D + SHARPE_RATIO_20D + SLOPE_20D` (桶 A+C+E)
- S1 和 C2 共享桶 E, 但因子不同 (SLOPE_20D vs CORRELATION_TO_MARKET_20D)
- 信号空间完全分离 (持仓 0 重合)

## 执行框架 (与 S1 v5.0 完全一致)

| 参数 | 值 |
|------|-----|
| FREQ | 5 |
| POS_SIZE | 2 |
| LOOKBACK | 252 |
| COMMISSION | 0.0002 (2bp) |
| 迟滞 | ON (dr=0.10, mh=9, max 1 swap) |
| Regime Gate | ON (vol mode, 510300, 25/30/40%) |
| 执行模型 | T1_OPEN |
| 成本模型 | SPLIT_MARKET (per-ETF tier) |
| Universe | A_SHARE_ONLY (38 traded + 5 QDII monitored) |

## 绩效指标

### VEC 回测 (med 成本, T1_OPEN)

| 策略 | 全期收益 | HO收益 | HO MDD | HO最差月 | HO Calmar | HO Sharpe | 交易数 |
|------|---------|--------|--------|---------|-----------|-----------|--------|
| **C2** | +21.7% | **+63.0%** | **-10.4%** | -9.4% | 6.16 | 2.63 | 28 |
| S1 | +53.7% | +42.7% | -11.8% | -2.7% | 3.61 | 2.15 | 75 |

### BT Ground Truth (med 成本, T1_OPEN, 0 margin failures)

| 策略 | HO收益 | HO MDD | HO最差月 | HO Calmar | HO Sharpe | 交易数 |
|------|--------|--------|---------|-----------|-----------|--------|
| **C2** | **+45.9%** | -12.2% | -7.1% | 3.77 | 2.20 | 34 |
| S1 | +32.5% | -11.0% | -10.1% | 2.95 | 1.77 | 120 |

**BT-C2 vs BT-S1**: +13.4pp HO return advantage, 0 margin failures

### VEC-BT Gap

| 策略 | VEC HO | BT HO | Gap | 说明 |
|------|--------|-------|-----|------|
| C2 | +63.0% | +45.9% | -17.1pp | C2交易少(28 vs 34), 每次整数手偏差放大 |
| S1 | +42.7% | +32.5% | -10.2pp | 正常范围 |

C2 的 VEC-BT gap 较大但可解释: 仅 28 笔 VEC 交易, 整数手 rounding 效应被放大。

## 验证门控

### Gate 1: 滚动 OOS

| 指标 | 结果 |
|------|------|
| 收益为正的窗口 | 3/4 (75%) |
| 最差月达标 | 4/4 (100%) |
| 判定 | **PASS** |

### Gate 2: PnL 集中度

| 指标 | C2 | S1 | 判定 |
|------|-----|-----|------|
| Gini 系数 | 0.448 | 0.468 | C2 更均匀 |
| Top-20% 贡献 | 92% | 117% | C2 更分散 |
| 判定 | **PASS** | — | — |

### Gate 3: BT Ground Truth

- 0 margin failures
- HO +45.9% > 0
- **PASS**

## 正交性与尾部分析 (Q1 + Q2)

### Q1: 持仓正交性 — PASS

| 指标 | 值 | 判定依据 |
|------|-----|---------|
| Top-2 重合率 | **0/2 (100% 调仓日)** | Jaccard < 0.25 |
| Jaccard 相似度 | **0.000** | — |
| Spearman 秩相关 | 0.350 ± 0.296 | 参考 (中段排序受市场因子影响, 尖端选择完全不同) |

### Q2: 尾部共崩风险 — PASS

**主判据 (market-conditional):**

| 指标 | 值 | 门槛 | 判定 |
|------|-----|------|------|
| P(both<0 \| mkt crash) | **0.300** (6/20) | <= 0.50 | PASS |
| Wilson 95% CI upper | **0.519** | <= 0.65 | PASS |
| Worst week overlap | **1/5** | <= 2 | PASS |

→ 市场崩溃日, 70% 的时间至少一条腿非负

**Beta-neutral 残差 (参考):**

| 指标 | 值 | 解读 |
|------|-----|------|
| 残差 P(both<0) | 0.382 | 高于独立基线 0.25, 存在 shared sector/style 暴露 |
| 残差相关 | 0.497 | 去掉 510300 beta 后仍有中等正相关 |

→ Idiosyncratic 层面非完全独立, Shadow 阶段需持续监控

**监控项 (不用于 verdict):**

| 指标 | 值 | 说明 |
|------|-----|------|
| Co-crash ratio | 4.30 | 被共享 A 股 beta 膨胀 |
| Complementarity corr | -0.548 | OR mask 偏负, 非共崩指标 |
| 日收益相关 | 0.494 | A 股 beta 物理约束 |
| 周收益相关 | 0.423 | — |

## Verdict 判定规则

```yaml
Q1_PASS:
  condition: "Jaccard < 0.25"
  spearman: "reference only"

Q2_PASS:
  condition: >
    worst_week_overlap <= 2
    AND P(both<0 | mkt left 10%) <= 0.50
    AND Wilson_CI95_upper <= 0.65
  co_crash_ratio: "monitoring only (shared beta inflates)"

COMBINED:
  proceed_to_shadow: "Q1_PASS AND Q2_PASS"
```

## Shadow 阶段目标

### 回答 3 个问题

| 问题 | 指标 | 历史答案 | Shadow 要做的 |
|------|------|---------|-------------|
| Q1 正交性 | Jaccard, top-2 overlap | 0.000 | 确认新行情下仍成立 |
| Q2 尾部错开 | P(jl\|mkt), worst week | 0.30, 1/5 | 滚动监控 |
| Q3 执行漂移 | 现金拖尾, 整数手偏差 | — | BT 口径, Forward 阶段新增 |

### Shadow 形态

1. **Phase A (4-6 周)**: 纯信号 Shadow — 扩展 `generate_today_signal.py` 输出 C2 信号, 每日记录 signal snapshot
2. **Phase B (6-8 周)**: 带执行 Shadow — 模拟整数手下单, 回答 Q3

### Go/No-Go 阈值

| 指标 | 继续 | 暂停 | 终止 |
|------|------|------|------|
| P(both<0 \| mkt crash) 滚动 | <= 0.50 | 0.50-0.65 | > 0.65 |
| Worst week overlap (滚动 5周) | <= 2 | 3 | >= 4 |
| Top-2 Jaccard (滚动) | < 0.15 | 0.15-0.30 | >= 0.30 |
| 残差 P(both<0) | < 0.40 | 0.40-0.50 | >= 0.50 |

## 复现步骤

```bash
# 使用 v5.0 代码库, C2 因子组合
cd /home/sensen/dev/projects/-0927

# 1. VEC 回测
uv run python scripts/batch_vec_backtest.py  # 修改 factor combo 为 C2

# 2. BT Ground Truth
uv run python scripts/validate_c2_bt_ground_truth.py

# 3. Shadow 回顾性分析
uv run python scripts/shadow_retrospective_q1q2.py
```

## 文件清单

```
c2_shadow_20260211/
├── SEAL_SUMMARY.md                    # 本文档
├── CHECKSUMS.sha256                   # 完整性校验
├── artifacts/
│   ├── strategy_params.yaml           # C2 冻结参数
│   ├── bt_vs_vec_comparison.csv       # BT/VEC 对比 (S1+C2)
│   ├── equity_curves.csv              # S1/C2 权益曲线 (holdout)
│   ├── holding_overlap_ho.csv         # 持仓重合度 (holdout)
│   └── retrospective_q1q2.json        # Q1+Q2 完整指标
└── scripts/
    └── shadow_retrospective_q1q2.py   # 分析脚本快照
```

---

**Sealed by**: Claude Code
**Seal Timestamp**: 2026-02-11
**Next Action**: 搭建 Forward Shadow 信号生成 (扩展 generate_today_signal.py)
