# C2 Alpha实测：+71.9%里只有16pp是真alpha（2026-02-12）

## 一句话结论

C2的+71.9% HO回报 = 33pp市场beta + 22pp集中度溢价 + 16pp gate×选股交互。风险调整后边际alpha约Sharpe +0.23，值得shadow但不值得盲信。

---

## 核心数据

### Alpha分解（VEC, F5+Exp4, 49-ETF, med cost, HO 2025-05~2026-02）

| 策略 | HO收益 | HO MDD | HO Sharpe | 交易数 |
|------|--------|--------|-----------|--------|
| C2 gate ON（生产配置） | +71.9% | 11.6% | **2.75** | 30 |
| C2 gate OFF | +54.9% | 12.5% | 2.47 | 45 |
| 等权买入持有 | +33.0% | 7.2% | 2.52 | 0 |
| 随机选2只 gate ON（20种子中位数） | +22.8% | 14.6% | 1.43 | - |
| 随机选2只 gate OFF | +22.5% | 13.7% | 1.24 | - |
| S1 修正后 | +30.4% | 10.1% | 1.84 | 71 |

### 回报拆解

```
C2 gate ON +71.9% =
  +33.0pp  市场beta（等权都有）
  +22.5pp  集中度溢价（随机选2只+Exp4）
  +16.4pp  gate × 选股交互（C2独有alpha）
```

### 风险调整后的真相

- C2 gate OFF Sharpe 2.47 **< 等权 Sharpe 2.52** → 纯选股无风险调整优势
- C2 gate ON Sharpe 2.75 **> 等权 Sharpe 2.52** → +0.23来自gate×selection
- Gate单独效果 ≈ 0（等权开/关gate差-0.2pp）
- **Gate不是择时工具，是C2集中持仓的风险管理放大器**

---

## S1 崩塌根因（2×2+1隔离实验）

| 配置 | HO收益 | HO Sharpe | 说明 |
|------|--------|-----------|------|
| S1@43 old bounded(4) | +42.7% | 2.16 | v5.0封存值（bug产物） |
| S1@43 new bounded(7) | +30.8% | 1.83 | bounded修正，-11.9pp |
| S1@49 new bounded(7) | +30.4% | 1.84 | 池子扩展仅-0.4pp |
| S1@49 old bounded(4) | +20.9% | 1.16 | old bounded+大池=最差 |
| C2@43 new bounded(7) | +59.4% | 2.55 | C2任何配置都碾压 |

**根因**：ADX_14D从z-score+Winsorize → rank标准化，占退化的97%。池子扩展仅3%。

**致命发现**：S1修正后+30.4% < 等权+33.0%。S1连被动策略都跑不赢。

---

## 三大不确定性

| 风险 | 严重程度 | 数据支撑 |
|------|----------|----------|
| 样本不足 | **高** | 30笔HO交易，需176笔才能Sharpe>1.5 at p<0.05（~3.5年） |
| Pool敏感性 | **高** | 加3个不交易QDII → 回报波动42pp，标准化偶然性大 |
| 单一市场环境 | **中** | HO期间等权+33%=牛市，未经熊市检验 |

---

## 统计显著性框架

### Bootstrap置信区间（1000次重采样）

| 指标 | 5th分位 | 中位数 | 95th分位 |
|------|---------|--------|----------|
| HO收益 | +21.7% | +71.9% | +132.8% |
| HO Sharpe | 1.30 | 2.75 | 8.96 |
| HO MDD | -18.9% | -10.4% | -4.2% |

- 负收益概率：0.3%
- Sharpe<1.0概率：3.1%

### 样本量需求

| 目标 | p<0.05 | p<0.01 | 时间（FREQ=5） |
|------|--------|--------|----------------|
| Sharpe > 1.0 | 396笔 | 589笔 | ~7.8年 |
| Sharpe > 1.5 | 176笔 | 262笔 | ~3.5年 |

### 停止规则（O'Brien-Fleming序贯检验）

| 检查点 | 交易数 | z阈值 | 动作 |
|--------|--------|-------|------|
| Look 1 | 13笔 | ±3.92 | 仅极端证据触发 |
| Look 2 | 25笔 | ±2.77 | 强证据 |
| Look 3 | 38笔 | ±2.26 | 中等证据 |
| Look 4 | 50笔 | ±1.96 | 标准显著性 |

**紧急停止**：MDD > 24% / 连续5笔亏损 / 50笔后Sharpe < 0

---

## IOPV/NAV（Exp7）：数据可用，信号为零

- 49只ETF的NAV数据齐全（2020-2026），A股T+1、QDII美股T+2
- Premium因子IC全部 < 0.065（不达标），根因：38只A股premium≈0，仅5只QDII有信号
- **结论：Exp7关闭。跨截面约束下不可能作为排序因子**

---

## Non-OHLCV因子：5种方案全部失败（P0-P4）

| 方案 | 结果 | 根因 |
|------|------|------|
| P0 代数交叉 | 612个候选→0通过正交性 | 继承OHLCV父因子rank结构 |
| P0b 残差正交化 | IC下降28-50% | 去除OHLCV成分=去除信号 |
| P1 Rank EMA平滑 | 最佳+0.6pp（噪声） | 因子权重25%，Δrank无法触发Exp4 |
| P3 二值否决 | 全部变差-17~-42pp | C2 83%胜率，否决的都是好交易 |
| P4 聚合择时 | 最佳+2.0pp（不稳健） | 换参数就翻号 |

**fund_share/margin数据源天花板已到**。突破需要全新数据：北向资金、期权IV。

---

## Shadow部署：零代码改动

```bash
# 已有基础设施，直接运行
uv run python scripts/generate_today_signal.py --shadow-config configs/shadow_strategies.yaml
```

- C2已配置在 `configs/shadow_strategies.yaml`
- 独立状态文件 `data/live/shadow_*.json`
- JSONL日志 + markdown报告

---

## 决策矩阵

| 问题 | 回答 | 信心 |
|------|------|------|
| C2有alpha吗？ | 有，约Sharpe +0.23（gate×选股） | 中 |
| +71.9%可信吗？ | 33pp是beta，16pp是alpha，其余是集中度溢价 | 高 |
| 能替换S1吗？ | S1已证伪（跑输等权），C2至少不差 | 高 |
| 能all-in吗？ | 不能，30笔交易+42pp pool敏感性=证据不足 | 高 |
| Shadow值得吗？ | 值得，成本为零，唯一获取数据的方式 | 高 |
| 下一步研究？ | 新数据源（北向/期权IV），OHLCV已饱和 | 高 |

---

## 文件索引

| 内容 | 路径 |
|------|------|
| 终局完整报告 | `reports/project_endgame_report.md` |
| Alpha分解结果 | `results/alpha_decomposition_20260212_192734/` |
| S1隔离结果 | `results/s1_isolation/isolation_results_20260212_193039.csv` |
| 显著性分析 | `results/significance_analysis/` |
| Pool归因脚本 | `scripts/research/c2_pool_attribution.py` |
| Non-OHLCV研究 | `scripts/research/p3_smart_money_filter.py`, `p4_aggregate_timing.py` |
| Shadow配置 | `configs/shadow_strategies.yaml` |
| 深度评审报告 | `reports/deep_review_final_report.md` |
