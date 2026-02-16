# 因子完整参考手册

> 日期: 2026-02-16 (v8.0 封板后)
> 单一事实源: `src/etf_strategy/core/factor_registry.py` → `FACTOR_SPECS`

## 概览

| 维度 | 数量 |
|------|------|
| 注册因子总数 | 44 (38 OHLCV + 6 non-OHLCV) |
| 活跃因子 (WFO搜索空间) | 23 (17 OHLCV + 6 non-OHLCV) |
| 有界因子 (rank标准化) | 7 |
| 非有界因子 (Z-score + Winsorize) | 37 |

---

## 活跃因子全表 (23个)

### 方向说明

- **Registry direction**: `factor_registry.py` 中的设计意图 (high_is_good / low_is_good / neutral)
- **WFO sign**: 实际生产中 WFO 根据滚动窗口 IC 符号 + stability gate(≥0.8) 决定的方向
- **两者关系**: Registry direction 是文档性质的，WFO sign 是实际生产决策。v8.0 中两者一致。

### OHLCV 因子 (17个, 数据源: `raw/ETF/daily/`)

| 因子 | 类别 | 有界 | 值域 | Registry方向 | Train IC | HO IC | HO ICIR | v8用途 |
|------|------|------|------|-------------|----------|-------|---------|--------|
| **ADX_14D** | E:趋势强度 | **是** | [0,100] | high_is_good | +0.024 | +0.045 | 0.153 | C1(+1), S1 |
| **AMIHUD_ILLIQUIDITY** | D:微观结构 | 否 | — | low_is_good | +0.007 | +0.035 | 0.215 | — |
| **BREAKOUT_20D** | A:趋势动量 | 否 | — | high_is_good | +0.028 | +0.019 | 0.066 | C1(+1) |
| **CALMAR_RATIO_60D** | B:持续位置 | 否 | — | high_is_good | +0.026 | +0.033 | 0.103 | — |
| **CORRELATION_TO_MARKET_20D** | E:趋势强度 | **是** | [-1,1] | low_is_good | +0.003 | +0.104 | 0.267 | — |
| **GK_VOL_RATIO_20D** | D:微观结构 | 否 | — | neutral | -0.031 | -0.011 | 0.047 | — |
| **MAX_DD_60D** | E:趋势强度 | 否 | — | low_is_good | -0.012 | +0.045 | 0.132 | — |
| **MOM_20D** | A:趋势动量 | 否 | — | high_is_good | +0.015 | +0.064 | 0.176 | — |
| **OBV_SLOPE_10D** | C:量价确认 | 否 | — | high_is_good | NaN | NaN | NaN | S1 |
| **PRICE_POSITION_20D** | A:趋势动量 | **是** | [0,1] | neutral | +0.028 | +0.071 | 0.251 | — |
| **PRICE_POSITION_120D** | B:持续位置 | **是** | [0,1] | neutral | +0.024 | +0.056 | 0.198 | C1(+1), 4F(+1) |
| **PV_CORR_20D** | D:微观结构 | **是** | [-1,1] | high_is_good | -0.011 | -0.012 | 0.054 | — |
| **SHARPE_RATIO_20D** | A:趋势动量 | 否 | — | high_is_good | +0.025 | +0.080 | 0.258 | S1 |
| **SLOPE_20D** | A:趋势动量 | 否 | — | high_is_good | +0.010 | +0.064 | 0.204 | S1, 4F(+1) |
| **UP_DOWN_VOL_RATIO_20D** | C:量价确认 | 否 | — | high_is_good | +0.015 | +0.064 | 0.226 | — |
| **VOL_RATIO_20D** | D:微观结构 | 否 | — | high_is_good | -0.001 | +0.032 | 0.139 | — |
| **VORTEX_14D** | A:趋势动量 | 否 | — | neutral | +0.008 | +0.065 | 0.218 | — |

### 非OHLCV 因子 (6个)

| 因子 | 数据源 | 类别 | Registry方向 | Train IC | HO IC | HO ICIR | Stability | v8用途 |
|------|--------|------|-------------|----------|-------|---------|-----------|--------|
| **SHARE_CHG_5D** | fund_share | F:份额流动 | low_is_good | -0.042 | **-0.101** | **0.450** | **1.00** | C1(**-1**) |
| **SHARE_CHG_10D** | fund_share | F:份额流动 | low_is_good | -0.050 | **-0.096** | **0.399** | **1.00** | — |
| **SHARE_CHG_20D** | fund_share | F:份额流动 | low_is_good | -0.044 | -0.067 | 0.273 | **1.00** | 4F(**-1**) |
| **SHARE_ACCEL** | fund_share | F:份额流动 | high_is_good | +0.035 | +0.032 | 0.130 | 0.89 | — |
| **MARGIN_CHG_10D** | margin | G:杠杆行为 | low_is_good | -0.028 | -0.057 | **0.350** | **1.00** | 4F(**-1**) |
| **MARGIN_BUY_RATIO** | margin | G:杠杆行为 | low_is_good | -0.014 | -0.057 | **0.363** | **1.00** | C1(**-1**) |

**非OHLCV 因子特征**:
- 5/6 是负向因子 (low_is_good)，只有 SHARE_ACCEL 是正向
- IC-sign fix 前被系统性反向使用 → "Non-OHLCV all failed" 的错误结论
- HO ICIR 普遍高于 OHLCV 因子 (0.27~0.45 vs 0.05~0.27)
- Stability 全部 ≥ 0.89，方向稳定

---

## 非活跃因子 (21个, 注册但未参与WFO)

| 因子 | 来源 | 有界 | 方向 | 停用原因 |
|------|------|------|------|---------|
| ABNORMAL_VOLUME_20D | ohlcv | 否 | neutral | 弱 IC, 与 UP_DOWN_VOL 冗余 |
| CMF_20D | ohlcv | **是** [-1,1] | high_is_good | **注意: 虽然注册但不在 active_factors 中** |
| DD_DURATION_60D | ohlcv | 否 | low_is_good | 弱 IC |
| DOWNSIDE_DEV_20D | ohlcv | 否 | low_is_good | 弱 IC |
| HURST_60D | ohlcv | 否 | neutral | IC ≈ 0 |
| IBS | ohlcv | 否 | high_is_good | 弱 IC, 高噪声 |
| INFO_DISCRETE_20D | ohlcv | 否 | low_is_good | 弱 IC |
| KURT_20D | ohlcv | 否 | low_is_good | 弱 IC |
| MEAN_REV_RATIO_20D | ohlcv | 否 | low_is_good | 弱 IC |
| PERM_ENTROPY_20D | ohlcv | 否 | low_is_good | 弱 IC |
| REALIZED_VOL_20D | ohlcv | 否 | low_is_good | 与 RET_VOL_20D 完全相同 |
| RELATIVE_STRENGTH_VS_MARKET_20D | ohlcv | 否 | high_is_good | 与 MOM_20D 高度冗余 |
| RET_VOL_20D | ohlcv | 否 | low_is_good | 弱 IC |
| RSI_14 | ohlcv | **是** [0,100] | neutral | IC ≈ 0 |
| SKEW_20D | ohlcv | 否 | low_is_good | 弱 IC |
| SPREAD_PROXY | ohlcv | 否 | low_is_good | 方向不稳定 (train/HO 反转) |
| TSMOM_60D | ohlcv | 否 | high_is_good | IC ≈ 0 |
| TSMOM_120D | ohlcv | 否 | high_is_good | 弱 IC |
| TURNOVER_ACCEL_5_20 | ohlcv | 否 | high_is_good | IC ≈ 0 |
| ULCER_INDEX_20D | ohlcv | 否 | low_is_good | 方向不稳定 (train/HO 反转) |
| VOL_RATIO_60D | ohlcv | 否 | high_is_good | IC ≈ 0 |

---

## 因子分桶 (Cross-bucket constraint)

| 桶 | 名称 | 活跃因子 | 说明 |
|----|------|---------|------|
| A | 趋势动量 | MOM, SLOPE, SHARPE, BREAKOUT, VORTEX, PP20 | 6个, 最大桶 |
| B | 持续位置 | PP120, CALMAR | 2个 |
| C | 量价确认 | OBV_SLOPE, UP_DOWN_VOL | 2个 (CMF不在active中) |
| D | 微观结构 | PV_CORR, AMIHUD, GK_VOL | 3个 |
| E | 趋势强度/风险 | ADX, CORR_MKT, MAX_DD, VOL_RATIO | 4个 |
| F | 份额流动 | SHARE_CHG_5D/10D/20D, SHARE_ACCEL | 4个 (非OHLCV) |
| G | 杠杆行为 | MARGIN_CHG, MARGIN_BUY | 2个 (非OHLCV) |

**约束**: min_buckets=3, max_per_bucket=2 → 搜索空间减半, HO median +4.84pp

---

## v8.0 封板策略因子详情

### composite_1 (主策略, 5F)

```
score = (+1)*w1*5*ADX_14D + (+1)*w2*5*BREAKOUT_20D + (-1)*w3*5*MARGIN_BUY_RATIO
      + (+1)*w4*5*PRICE_POSITION_120D + (-1)*w5*5*SHARE_CHG_5D
```

| 因子 | Sign | ICIR | Weight | 桶 | 解读 |
|------|------|------|--------|-----|------|
| ADX_14D | +1 | 0.440 | 0.014 | E | 趋势强度高 → 选入 |
| BREAKOUT_20D | +1 | 18.579 | 0.608 | A | 突破信号强 → 选入 (权重最大) |
| MARGIN_BUY_RATIO | -1 | -5.101 | 0.167 | G | 融资买入比低 → 选入 (逆向) |
| PRICE_POSITION_120D | +1 | 4.542 | 0.149 | B | 半年价格位置高 → 选入 |
| SHARE_CHG_5D | -1 | -2.306 | 0.075 | F | 份额减少 → 选入 (逆向) |

**解读**: 选择趋势突破 + 半年高位 + 融资冷淡 + 份额流出的 ETF。逻辑: 价格强势但散户/杠杆资金未跟进 = 机构主导的趋势。

### core_4f (回退策略, 4F)

| 因子 | Sign | ICIR | Weight | 桶 | 解读 |
|------|------|------|--------|-----|------|
| MARGIN_CHG_10D | -1 | -3.387 | 0.263 | G | 融资余额下降 → 选入 (逆向) |
| PRICE_POSITION_120D | +1 | 4.542 | 0.353 | B | 半年价格位置高 → 选入 (权重最大) |
| SHARE_CHG_20D | -1 | -1.807 | 0.140 | F | 份额减少 → 选入 (逆向) |
| SLOPE_20D | +1 | 3.125 | 0.243 | A | 上升斜率高 → 选入 |

**解读**: 选择上升趋势 + 半年高位 + 杠杆/份额双流出的 ETF。比 composite_1 更依赖流出信号。

---

## 方向处理架构

```
factor_registry.py          combo_wfo_optimizer.py        batch_vec/bt_backtest.py
  direction (文档)    →    sign_stability gate (≥0.8)  →   sign * weight * N * factor
  (不参与计算)              factor_signs (实际决策)          (生产执行)
                           factor_icirs (权重来源)
```

**关键设计**:
1. Registry `direction` 是文档性质，**不参与计算**
2. WFO 根据滚动窗口 IC 符号一致率 (sign_stability) 决定是否翻转
3. stability < 0.8 的因子保持 +1 (不翻转, 避免不稳定方向)
4. VEC/BT/Live 消费 WFO 输出的 factor_signs + factor_icirs

---

## IC 数据来源与局限

- **数据文件**: `results/factor_direction_stability_20260214_013357/direction_analysis.csv`
- **Train**: 2020-01 ~ 2025-04 (1270 交易日), **HO**: 2025-05 ~ 2026-02 (189 交易日)
- **IC 计算**: 截面 Spearman rank correlation, 每日计算后取均值
- **ICIR**: mean(IC) / std(IC), 衡量 IC 的信噪比
- **OBV_SLOPE_10D**: IC 为 NaN (成交量数据缺失导致分析脚本跳过)
- **注意**: direction_analysis.csv 包含 56 行 (含 EMA 变体等非注册因子), 活跃因子仅 23 个

---

## 相关文件

| 文件 | 角色 |
|------|------|
| `src/etf_strategy/core/factor_registry.py` | 因子元数据单一事实源 (44因子) |
| `src/etf_strategy/core/precise_factor_library_v2.py` | OHLCV 因子计算库 |
| `src/etf_strategy/core/non_ohlcv_factors.py` | 非OHLCV 因子计算 |
| `src/etf_strategy/core/cross_section_processor.py` | 截面标准化 (Z-score/rank) |
| `src/etf_strategy/core/factor_cache.py` | 因子缓存 |
| `src/etf_strategy/core/factor_buckets.py` | 分桶约束 |
| `configs/combo_wfo_config.yaml` | 活跃因子列表 (23个) |
| `results/factor_direction_stability_20260214_013357/` | IC/ICIR 方向分析 |
| `docs/FACTOR_EXPANSION_V42.md` | v4.2 因子扩展研究 |
| `docs/ETF_DATA_GUIDE.md` | 数据源指南 |
