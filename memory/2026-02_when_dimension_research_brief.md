# WHEN 维度研究简报 — 策略激活时机 & 策略组合

> 日期: 2026-02-17
> 状态: 待验证
> 前置知识: 读 CLAUDE.md + memory/MEMORY.md

---

## 背景

v8.0 封板后，系统性验证了所有"选股维度(WHAT)"的优化方向，全部关闭：
- 因子重组合(Phase1): Kaiser 5/17 饱和
- 新数据管道(moneyflow): rho=0.58 同维度
- 得分离散度: rho<0.08 无信号

**但 composite_1 Rolling win rate 只有 61%**（39%的时间窗口亏损），稳定性问题未解决。

关键洞察：我们穷尽了"选哪个ETF"的空间，但完全没探索"什么时候该信任选股"和"两个策略怎么配合"。

---

## 三个独立的 Alpha 优化轴

| 轴 | 问题 | 状态 | 对应稳定性改善 |
|----|------|------|---------------|
| **WHAT** | 选哪个 ETF | 已饱和 (Kaiser 5/17) | — |
| **WHEN** | 什么时候信任选股 | **未探索** | 识别39%坏窗口→降仓→提升Rolling |
| **HOW** | 两个策略怎么配合 | **未探索** | 取长补短→平滑收益曲线 |

---

## 方向 1: 截面收益离散度 (WHEN)

### 核心逻辑

选股策略的前提是"不同ETF有不同回报"。当所有ETF同涨同跌时(PC1主导)，选股毫无意义。

```
高离散: 49个ETF收益率分化大 → 选对赚很多，选错亏很多 → 选股alpha最大
低离散: 49个ETF同方向运动 → 选谁都差不多 → 选股alpha≈0

PCA验证: PC1 median=59.8%, 2021年47.7%(低→分化), 2024年71.2%(高→同向)
```

### 信号定义

```python
# 过去20天49个ETF的截面收益率标准差
ret_20d = price_df.pct_change(20)  # 每ETF的20日收益
dispersion = ret_20d.std(axis=1)    # 截面标准差 (一个时间序列)
```

### 验证方案

1. 在每个rebalance日计算 dispersion
2. 和 composite_1 的 period-level 收益做相关性
3. 做 tercile: 高离散期 vs 低离散期的策略收益对比
4. 分 train/holdout 检查方向一致性 (Rule 4)

### 使用方式（如果验证通过）

不改因子选股逻辑，只在仓位管理层：
- 高离散 → 正常仓位 (regime gate 输出 × 1.0)
- 低离散 → 降低仓位 (regime gate 输出 × 0.3~0.5)
- 和 regime gate 互补: regime gate 管"市场危不危险"，dispersion 管"选股有没有用"

### 与之前失败的"得分离散度"的区别

- **得分离散度** (已验证, 无信号): 因子排名清不清楚 → 信号质量维度
- **收益离散度** (待验证): ETF回报是否分化 → 市场环境维度
- 两者概念完全不同，不能因为前者失败就跳过后者

---

## 方向 2: 策略一致性 / Ensemble (HOW)

### 核心逻辑

composite_1 和 core_4f 是两个不同的 alpha 家族：

```
Family A (composite_1): 趋势突破 + 散户/杠杆流出 → 高Sharpe(1.38), 低MDD(10.8%), Rolling 61%
Family B (core_4f):     持续上升 + 双向流出 → 高绝对收益(HO+67%), Rolling 78%

关键问题: 它们的 39% 和 22% 失败期是否重叠？
- 如果不重叠 → ensemble 可以大幅提升稳定性
- 如果重叠 → 没用（都在同一种市场环境下失败）
```

### 验证方案

1. 在每个 rolling window 中标记 composite_1 是否盈利、core_4f 是否盈利
2. 计算重叠率: P(both_fail) / P(either_fail)
3. 如果 P(both_fail) << P(A_fail) * P(B_fail) → 负相关失败 → ensemble 有价值
4. 设计 ensemble 规则:
   - 两者一致时(选同一个ETF) → 满仓
   - 分歧时 → 各半仓或跟 Rolling 更高的那个

### 数据来源

需要在相同的 rebalance dates 上同时跑 composite_1 和 core_4f 的因子得分。
两个策略的 WFO 结果都在 `results/run_20260214_115216/` 中。

---

## 方向 3: 因子近期有效性 / Factor Momentum (WHEN)

### 核心逻辑

如果 composite_1 的 5 个因子在最近 20 天的截面 IC 都是正的，说明因子正在"工作"。
如果 IC 转负，说明市场环境变了，策略可能要失效。

### 风险

- 49 ETF × 20 天 = ~980 个数据点估一个 IC，噪声极大
- 因子动量在大样本(3000+股票)中有强学术支持，但在 49 ETF 中信噪比可能不够
- 过拟合风险高（需要 IC 的阈值参数）

### 优先级: 低于方向1和2

---

## 验证优先级

| 优先级 | 方向 | 耗时 | 预期价值 | 风险 |
|--------|------|------|---------|------|
| **P0** | 截面收益离散度 | 30min | 中-高 | 概念清晰, 过拟合风险低(单参数) |
| **P1** | 策略 ensemble (失败重叠率) | 30min | 中-高 | 需同时跑两策略 |
| P2 | 因子动量 | 1hr | 中 | 样本量小, 过拟合风险高 |

---

## 过拟合防护

- 每个信号最多 1 个阈值参数
- Train/Holdout 必须方向一致 (Rule 4)
- 如果 train 有信号但 holdout 反转 → 立即关闭（如得分离散度案例）
- 总体改善如果 < 5pp → 不值得增加复杂度

---

## 关键数据位置

- 价格: `raw/ETF/daily/{code}.{SH|SZ}_daily_*.parquet` (adj_close列)
- 基金份额: `raw/ETF/fund_share/fund_share_{code}.parquet` (fd_share列)
- 融资融券: `raw/ETF/margin/pool43_2020_now.parquet` (ts_code+trade_date+rzmre列)
- WFO结果: `results/run_20260214_115216/` (top_combos.parquet)
- v8.0 策略定义:
  - composite_1: ADX_14D + BREAKOUT_20D + MARGIN_BUY_RATIO + PRICE_POSITION_120D + SHARE_CHG_5D
  - core_4f: MARGIN_CHG_10D + PRICE_POSITION_120D + SHARE_CHG_20D + SLOPE_20D
- 已有 Rolling 结果: `results/rolling_oos_consistency_20260216_094332/`
- Production config: `configs/combo_wfo_config.yaml`

## 当前状态

- 分支: master (clean, pushed)
- v8.0 sealed, shadow 配置就绪
- 所有 WHAT 维度研究已关闭
- WHEN/HOW 维度待验证
