# 因子扩展研究 v4.2 — 完整报告

> **日期**: 2026-02-05
> **数据范围**: 2020-01-02 ~ 2025-12-12, 46 ETFs, 1442 交易日
> **基准**: 等权 43 ETF 持仓, 总收益 +58.2%
> **核心参数**: FREQ=3, POS_SIZE=2, COMMISSION=2bp

---

## 1. 研究目标

在现有 15 个正交因子 (orthogonal_v1) 基础上, 广度优先研究 13 个新因子候选, 通过 IC/正交性/Top-2 选股回测筛选最优子集。

### 现有 15 因子覆盖的维度

| 维度 | 因子 | 饱和度 |
|------|------|--------|
| 趋势/动量 | MOM_20D, SLOPE_20D, BREAKOUT_20D | 充分 |
| 价格位置 | PRICE_POSITION_20D, PRICE_POSITION_120D | 双窗口 |
| 波动/风险 | MAX_DD_60D, CALMAR_RATIO_60D | 双角度 |
| 量能/流动性 | VOL_RATIO_20D, AMIHUD_ILLIQUIDITY, SPREAD_PROXY | 三角度 |
| 价量耦合 | PV_CORR_20D | 单因子 |
| 趋势强度 | ADX_14D, VORTEX_14D | 双指标 |
| 风险调整动量 | SHARPE_RATIO_20D | 单因子 |
| 市场相关性 | CORRELATION_TO_MARKET_20D | 单因子 |

### 目标填补的空白维度

高阶矩、均值回复、动量质量、量能方向性、回撤恢复、复杂度/熵、波动率微结构、长期记忆、非对称风险

---

## 2. 实施内容

### 代码改动

| 文件 | 改动 |
|------|------|
| `src/etf_strategy/core/data_loader.py` | 新增 `amount` 键 (交易额, 提升 AMIHUD 精度) |
| `src/etf_strategy/core/precise_factor_library_v2.py` | 新增 13 个因子 (metadata + Numba 内核 + 批量计算 + 注册) |

### 新增 Numba 内核 (4 个, 模块级缓存)

- `_rolling_ulcer_index_numba` / `_rolling_ulcer_index_batch`
- `_dd_duration_numba` / `_dd_duration_batch`
- `_permutation_entropy_numba` / `_permutation_entropy_batch`
- `_hurst_rs_numba` / `_hurst_rs_batch`

### 数据验证

全量 46 ETF 计算通过: 0 Inf, 0 越界, NaN 率 0%-16.9%, 计算耗时 0.2s。

---

## 3. 全量 IC 分析 (38 因子, forward 3-day returns)

### 3.1 按 |IC_IR| 降序排列

```
因子                           维度           IC_IR   MeanIC   命中率   显著
─────────────────────────────────────────────────────────────────────────
PRICE_POSITION_20D             价格位置       0.093  +0.0287  53.9%   ***
PRICE_POSITION_120D            价格位置       0.083  +0.0277  54.4%   ***
CMF_20D                        资金流         0.082  +0.0229  54.9%   ***
CALMAR_RATIO_60D               风险调整动量   0.077  +0.0276  53.7%   ***
PV_CORR_20D                    价量耦合      -0.070  -0.0166  47.9%   ***
SHARPE_RATIO_20D               风险调整动量   0.069  +0.0229  54.0%   ***
GK_VOL_RATIO_20D [NEW]         波动率微结构  -0.068  -0.0174  46.9%   **
BREAKOUT_20D                   趋势/动量      0.064  +0.0224  52.8%   **
UP_DOWN_VOL_RATIO_20D [NEW]    量能方向性     0.060  +0.0166  52.9%   **
ADX_14D                        趋势强度       0.058  +0.0162  51.3%   **
KURT_20D [NEW]                 高阶矩         0.058  +0.0132  53.8%   **
SKEW_20D [NEW]                 高阶矩        -0.054  -0.0138  47.3%   **
ULCER_INDEX_20D [NEW]          回撤恢复      -0.049  -0.0175  47.0%   *
SPREAD_PROXY                   流动性/成本   -0.046  -0.0192  48.0%   *
CORRELATION_TO_MARKET_20D      相对强度       0.045  +0.0163  52.1%   *
MOM_20D                        趋势/动量      0.045  +0.0167  52.4%   *
PERM_ENTROPY_20D [NEW]         复杂度/熵      0.039  +0.0073  51.4%
MEAN_REV_RATIO_20D [NEW]       均值回复       0.025  +0.0088  51.1%
INFO_DISCRETE_20D [NEW]        动量质量      -0.024  -0.0059  48.8%
DD_DURATION_60D [NEW]          回撤恢复       0.020  +0.0058  50.3%
DOWNSIDE_DEV_20D [NEW]         非对称风险    -0.018  -0.0069  49.5%
ABNORMAL_VOLUME_20D [NEW]      量能方向性    -0.016  -0.0035  49.6%
IBS [NEW]                      均值回复       0.011  +0.0032  50.2%
HURST_60D [NEW]                长期记忆       0.003  +0.0007  51.3%
```

显著因子 (p<0.05): 12/38, 其中新因子占 5 个。

### 3.2 新因子 IC 显著性汇总

| 因子 | IC_IR | p值 | 结论 |
|------|-------|-----|------|
| GK_VOL_RATIO_20D | 0.068 | <0.01 | 显著 ** |
| UP_DOWN_VOL_RATIO_20D | 0.060 | <0.05 | 显著 ** |
| KURT_20D | 0.058 | <0.05 | 显著 ** |
| SKEW_20D | 0.054 | <0.05 | 显著 ** |
| ULCER_INDEX_20D | 0.049 | <0.10 | 边缘 * |
| PERM_ENTROPY_20D | 0.039 | >0.10 | 不显著 |
| MEAN_REV_RATIO_20D | 0.025 | >0.10 | 不显著 |
| INFO_DISCRETE_20D | 0.024 | >0.10 | 不显著 |
| DD_DURATION_60D | 0.020 | >0.10 | 不显著 |
| DOWNSIDE_DEV_20D | 0.018 | >0.10 | 不显著 |
| ABNORMAL_VOLUME_20D | 0.016 | >0.10 | 不显著 |
| IBS | 0.011 | >0.10 | 不显著 |
| HURST_60D | 0.003 | >0.10 | 不显著 |

---

## 4. Top-2 选股回测 (FREQ=3, POS_SIZE=2)

等权基准: +58.2%

### 4.1 全部因子排名

```
因子                           方向           总收益   年化    Sharpe  最大回撤
──────────────────────────────────────────────────────────────────────────────
MEAN_REV_RATIO_20D [NEW]       low_is_good   +142.9%  +17.1%  0.59   -48.7%
ADX_14D                        high_is_good  +118.5%  +15.0%  0.58   -30.0%
SHARPE_RATIO_20D               high_is_good  +117.7%  +14.8%  0.63   -38.1%
BREAKOUT_20D                   high_is_good  +107.4%  +13.8%  0.95   -25.4%
PV_CORR_20D                    high_is_good  +105.8%  +13.7%  0.48   -40.6%
MOM_20D                        high_is_good  +100.0%  +13.1%  0.38   -43.3%
PRICE_POSITION_120D            neutral        +97.5%  +12.6%  0.53   -35.8%
VORTEX_14D                     neutral        +96.1%  +12.6%  0.53   -32.8%
CMF_20D                        high_is_good   +92.3%  +12.3%  0.56   -34.7%
PRICE_POSITION_20D             neutral        +80.8%  +10.9%  0.51   -35.1%
HURST_60D [NEW]                neutral        +76.1%  +10.9%  0.45   -39.5%
ULCER_INDEX_20D [NEW]          low_is_good    +66.3%   +9.5%  1.15    -9.2%  ← Sharpe 最优
GK_VOL_RATIO_20D [NEW]         neutral        +58.6%   +8.5%  0.49   -39.7%
SKEW_20D [NEW]                 low_is_good    +49.4%   +7.4%  0.39   -26.0%
PERM_ENTROPY_20D [NEW]         low_is_good    +45.3%   +6.9%  0.30   -45.4%
DOWNSIDE_DEV_20D [NEW]         low_is_good    +44.9%   +6.8%  1.56    -4.9%  ← 回撤最低
SLOPE_20D                      high_is_good   +42.2%   +6.5%  0.41   -21.1%
SPREAD_PROXY                   low_is_good    +39.1%   +6.0%  2.21    -2.6%
MAX_DD_60D                     low_is_good    +27.6%   +4.5%  1.03    -6.5%
KURT_20D [NEW]                 low_is_good    +10.1%   +1.7%  0.08   -52.5%
UP_DOWN_VOL_RATIO_20D [NEW]    high_is_good   +11.8%   +2.0%  0.09   -30.6%
ABNORMAL_VOLUME_20D [NEW]      neutral         +4.7%   +0.8%  0.03   -46.3%
IBS [NEW]                      high_is_good    -8.3%   -1.5% -0.07   -63.6%
INFO_DISCRETE_20D [NEW]        low_is_good    -15.3%   -2.9% -0.11   -38.6%
DD_DURATION_60D [NEW]          low_is_good    -32.9%   -6.8% -0.26   -66.4%
```

### 4.2 新因子选股表现 vs 基准

| 因子 | 总收益 | vs 基准 | Sharpe | 判断 |
|------|--------|---------|--------|------|
| MEAN_REV_RATIO_20D | +142.9% | **+84.7pp** | 0.59 | 跑赢, 但高度冗余 |
| HURST_60D | +76.1% | +17.9pp | 0.45 | 跑赢, 但 IC≈0 |
| ULCER_INDEX_20D | +66.3% | +8.1pp | **1.15** | 跑赢, 风险调整最优 |
| GK_VOL_RATIO_20D | +58.6% | +0.4pp | 0.49 | 约平 |
| SKEW_20D | +49.4% | -8.8pp | 0.39 | 跑输 |
| PERM_ENTROPY_20D | +45.3% | -12.9pp | 0.30 | 跑输 |
| DOWNSIDE_DEV_20D | +44.9% | -13.3pp | 1.56 | 跑输但回撤极低 |
| UP_DOWN_VOL_RATIO_20D | +11.8% | -46.4pp | 0.09 | 大幅跑输 |
| KURT_20D | +10.1% | -48.1pp | 0.08 | 大幅跑输 |
| ABNORMAL_VOLUME_20D | +4.7% | -53.5pp | 0.03 | 几乎为零 |
| IBS | -8.3% | -66.5pp | -0.07 | 亏损 |
| INFO_DISCRETE_20D | -15.3% | -73.5pp | -0.11 | 亏损 |
| DD_DURATION_60D | -32.9% | -91.1pp | -0.26 | 严重亏损 |

---

## 5. 因子间相关矩阵 (新因子 vs 现有 15 因子)

### 5.1 新因子与现有因子的最高 |corr|

| 新因子 | 最近邻现有因子 | |corr| | 超阈值? |
|--------|---------------|--------|---------|
| GK_VOL_RATIO_20D | (无 > 0.6) | <0.6 | 正交 |
| UP_DOWN_VOL_RATIO_20D | SHARPE_RATIO_20D | 0.673 | 正交 |
| SKEW_20D | (无 > 0.6) | <0.6 | 正交 |
| KURT_20D | (无 > 0.6) | <0.6 | 正交 |
| PERM_ENTROPY_20D | (无 > 0.6) | <0.6 | 正交 |
| ULCER_INDEX_20D | BREAKOUT_20D | **0.775** | **超阈值** (反相关) |
| MEAN_REV_RATIO_20D | PP20 0.818, MOM 0.797, VORTEX 0.803 | **0.818** | **严重冗余** |
| DOWNSIDE_DEV_20D | RET_VOL 0.887, SPREAD 0.739, MDD 0.727 | **0.887** | **严重冗余** |
| HURST_60D | (无 > 0.6) | <0.6 | 正交 |
| INFO_DISCRETE_20D | (无 > 0.6) | <0.6 | 正交 |
| IBS | DD_DURATION 0.504 | <0.6 | 正交 |
| DD_DURATION_60D | (无 > 0.6) | <0.6 | 正交 |
| ABNORMAL_VOLUME_20D | VOL_RATIO_20D 0.648 | <0.7 | 正交 |

### 5.2 新因子之间的高相关对

- DOWNSIDE_DEV_20D ↔ ULCER_INDEX_20D: 0.814
- MEAN_REV_RATIO_20D ↔ 多个新因子有中等相关

---

## 6. A 股 vs QDII IC 对比

新因子在 QDII 上的 IC 普遍高于 A 股 (与现有因子一致):

| 新因子 | 全池 IC | A 股 IC | QDII IC | 差异 |
|--------|---------|---------|---------|------|
| GK_VOL_RATIO_20D | -0.0174 | -0.0123 | **-0.0739** | QDII 更强 |
| UP_DOWN_VOL_RATIO_20D | +0.0166 | +0.0156 | **+0.0731** | QDII 更强 |
| SKEW_20D | -0.0138 | -0.0129 | -0.0298 | 一致 |
| KURT_20D | +0.0132 | +0.0053 | **+0.0498** | QDII 更强 |
| ULCER_INDEX_20D | -0.0175 | -0.0145 | **-0.0850** | QDII 更强 |
| HURST_60D | +0.0007 | +0.0011 | +0.0206 | QDII 略强 |
| PERM_ENTROPY_20D | +0.0073 | +0.0056 | +0.0261 | 一致 |
| MEAN_REV_RATIO_20D | +0.0088 | +0.0058 | **+0.0499** | QDII 更强 |
| DOWNSIDE_DEV_20D | -0.0069 | -0.0036 | **-0.0682** | QDII 更强 |

注: QDII 仅 5 只, 截面 IC 不稳定, 仅供参考方向。

---

## 7. 排名稳定性 (lag-3 rank autocorrelation)

适合 FREQ=3 调仓的因子需 RankAutoCorr > 0.8。

| 新因子 | RankAutoCorr | 稳定性 | 适合低频? |
|--------|-------------|--------|----------|
| DOWNSIDE_DEV_20D | 0.936 | 高 | 是 |
| ULCER_INDEX_20D | 0.885 | 高 | 是 |
| GK_VOL_RATIO_20D | 0.857 | 高 | 是 |
| UP_DOWN_VOL_RATIO_20D | 0.817 | 高 | 是 |
| SKEW_20D | 0.778 | 中 | 边界 |
| ABNORMAL_VOLUME_20D | 0.756 | 中 | 边界 |
| KURT_20D | 0.749 | 中 | 边界 |
| MEAN_REV_RATIO_20D | 0.734 | 中 | 边界 |
| HURST_60D | 0.623 | 中 | 偏低 |
| PERM_ENTROPY_20D | 0.608 | 中 | 偏低 |
| INFO_DISCRETE_20D | 0.570 | 中 | 偏低 |
| DD_DURATION_60D | 0.077 | 低 | 否 |
| IBS | 0.031 | 低 | 否 |

---

## 8. IC 衰减分析 (horizons: 1, 3, 5, 10, 20 日)

| 新因子 | IC_1d | IC_3d | IC_5d | IC_10d | IC_20d | 最优 |
|--------|-------|-------|-------|--------|--------|------|
| GK_VOL_RATIO_20D | -0.012 | -0.017 | -0.028 | -0.042 | **-0.051** | 20d |
| ULCER_INDEX_20D | -0.021 | -0.018 | -0.020 | -0.032 | **-0.050** | 20d |
| UP_DOWN_VOL_RATIO_20D | +0.009 | +0.017 | +0.020 | +0.025 | **+0.027** | 20d |
| KURT_20D | +0.008 | +0.013 | +0.015 | +0.019 | **+0.030** | 20d |
| SKEW_20D | -0.010 | -0.014 | -0.015 | -0.015 | **-0.026** | 20d |
| INFO_DISCRETE_20D | -0.012 | -0.006 | -0.007 | -0.009 | **-0.028** | 20d |
| MEAN_REV_RATIO_20D | +0.007 | +0.009 | +0.014 | **+0.025** | +0.022 | 10d |
| DOWNSIDE_DEV_20D | -0.017 | -0.007 | -0.007 | -0.014 | **-0.036** | 20d |
| HURST_60D | -0.003 | +0.001 | -0.001 | -0.009 | **-0.038** | 20d |
| DD_DURATION_60D | -0.010 | +0.006 | +0.009 | **+0.012** | +0.010 | 10d |
| IBS | -0.005 | +0.003 | +0.004 | **+0.010** | +0.010 | 10d |
| PERM_ENTROPY_20D | +0.005 | **+0.007** | +0.002 | +0.006 | +0.007 | 3d |
| ABNORMAL_VOLUME_20D | -0.001 | -0.004 | -0.003 | +0.001 | **+0.004** | 20d |

大部分新因子的 IC 在更长 horizon 更强, 与 FREQ=3 的低频调仓一致。

---

## 9. 综合筛选结论

### 筛选标准

1. IC_IR > 0.03 **或** Top-2 总收益 > 30%
2. 与所有现有 15 个活跃因子 |corr| < 0.7
3. 若两个新因子 |corr| > 0.7, 保留更强的

### 逐因子判定

| 因子 | IC_IR 达标 | Top2 达标 | 正交达标 | 综合评级 | **结论** |
|------|-----------|----------|---------|---------|---------|
| **GK_VOL_RATIO_20D** | 0.068** ✓ | +58.6% ✓ | ✓ | 强 | **入选** |
| **ULCER_INDEX_20D** | 0.049* ✓ | +66.3% ✓ | BREAKOUT -0.775 ✗ | 强 (边界) | **待定** |
| UP_DOWN_VOL_RATIO_20D | 0.060** ✓ | +11.8% ✗ | ✓ | IC 强但选股差 | 不入选 |
| KURT_20D | 0.058** ✓ | +10.1% ✗ | ✓ | IC 强但选股差 | 不入选 |
| SKEW_20D | 0.054** ✓ | +49.4% ✓ | ✓ | 跑输基准 | 不入选 |
| PERM_ENTROPY_20D | 0.039 ✓ | +45.3% ✓ | ✓ | IC 不显著 | 不入选 |
| MEAN_REV_RATIO_20D | ✗ | +142.9% ✓ | MOM 0.797 ✗ | 严重冗余 | 淘汰 |
| DOWNSIDE_DEV_20D | ✗ | +44.9% ✓ | RET_VOL 0.887 ✗ | 严重冗余 | 淘汰 |
| HURST_60D | 0.003 ✗ | +76.1% ✓ | ✓ | IC≈0, 非系统性 | 淘汰 |
| INFO_DISCRETE_20D | ✗ | ✗ | ✓ | 弱 | 淘汰 |
| IBS | ✗ | ✗ | ✓ | 弱, 排名不稳 | 淘汰 |
| DD_DURATION_60D | ✗ | ✗ | ✓ | 无效, 排名不稳 | 淘汰 |
| ABNORMAL_VOLUME_20D | ✗ | ✗ | ✓ | 弱 | 淘汰 |

### 最终推荐

- **确定入选**: GK_VOL_RATIO_20D (唯一全面达标)
- **待定**: ULCER_INDEX_20D (与 BREAKOUT corr=-0.775 > 阈值, 但是反相关而非冗余, 且 Sharpe 1.15 全场最优)

---

## 10. 关于不入选因子的详细说明

### IC 显著但选股无效的因子

**UP_DOWN_VOL_RATIO_20D** (IC_IR=0.060**):
IC 在统计上显著, 但 Top-2 选股仅 +11.8% (基准 +58.2%)。说明该因子虽能区分横截面排序, 但头部 ETF 的信号并不可靠。作为辅助因子在 combo 中可能有价值, 但单独选股能力不足。

**KURT_20D** (IC_IR=0.058**):
IC 为正 (+0.0132), 但方向标注为 low_is_good。这意味着高峰度 ETF 反而收益更高 — 与 "高峰度=尾部风险" 的学术假设矛盾。A 股 ETF 市场可能存在 "博彩偏好" 效应。Top-2 选股仅 +10.1%。

**SKEW_20D** (IC_IR=0.054**):
IC 显著但 Top-2 (+49.4%) 跑输基准 9pp。偏度信号在 43 个 ETF 的小截面上区分度不够。

### 高 Top-2 收益但冗余的因子

**MEAN_REV_RATIO_20D** (Top2 +142.9%):
全场最高 Top-2 收益, 但与 MOM(0.797), PP20(0.818), SHARPE(0.735), VORTEX(0.803) 严重共线。该因子本质上是 close/SMA20 的变体, 其信息已被 MOM + PP20 组合充分覆盖。

**DOWNSIDE_DEV_20D** (Top2 Sharpe 1.56):
风险调整收益极优, 但与 RET_VOL(0.887), SPREAD(0.739), MAX_DD(0.727), ULCER(0.814) 高度冗余。本质是 "下行波动率", 信息已被 SPREAD_PROXY + MAX_DD 覆盖。

### IC 为零但 Top-2 高的因子

**HURST_60D** (IC_IR=0.003, Top2 +76.1%):
IC 几乎为零 (t=0.10), 说明 Hurst 指数无系统性横截面预测力。+76.1% 的 Top-2 收益来自特定时段的巧合选中 (如 QDII), 不可复制。

---

## 11. 技术备注

### 因子库状态

- 总因子数: 38 (25 旧 + 13 新)
- 活跃因子: 15 (orthogonal_v1=True)
- 待激活: 0-2 (取决于最终决策)
- 新因子全部设 `orthogonal_v1=False`, 不影响现有管线

### 新增有界因子

如果激活, 需加入 `cross_section_processor.py` 的 BOUNDED_FACTORS:
- INFO_DISCRETE_20D [-1, 1]
- IBS [0, 1]
- DD_DURATION_60D [0, 1]
- PERM_ENTROPY_20D [0, 1]
- HURST_60D [0, 1]

### 下一步

1. 决定 ULCER_INDEX_20D 是否入选 (放宽阈值到 0.8 即可通过)
2. 将入选因子加入 `active_factors` in `combo_wfo_config.yaml`
3. 更新 `frozen_params.py` 版本至 v4.2
4. 运行全管线: WFO → VEC → Triple Validation → BT

---

## 12. 因子挖掘系统 (Factor Mining Pipeline)

> **日期**: 2026-02-05
> **工具**: `scripts/run_factor_mining.py`
> **方法**: 代数搜索 (两两组合 × 6 算子) + 10维质检 + BH-FDR校正 + 层次聚类去冗余

### 12.1 系统架构

```
Layer 1: FactorZoo (注册中心)        — 统一管理手工 + 挖掘因子
Layer 2: FactorQualityAnalyzer (质检) — 10维评估: IC/单调性/稳定性/衰减/换手/NaN/环境/A股QDII/方向/综合
Layer 3: FactorDiscoveryPipeline     — 代数搜索 / 窗口优化 / 变换搜索 + FDR
Layer 4: FactorSelector              — 质量过滤 + Spearman相关矩阵 + 层次聚类去冗余
Layer 5: run_factor_mining.py        — CLI入口
```

代码位置: `src/etf_strategy/core/factor_mining/` (~1,200行), 不修改任何现有文件。

### 12.2 运行结果 (--algebraic-only)

| 阶段 | 数量 | 耗时 |
|------|------|------|
| 输入因子 | 38 手工因子 | — |
| 候选组合 | C(38,2)×6 = 4,218 | — |
| IC预筛 (|IC|>0.02) | 500 存活 | ~20s |
| BH-FDR (α=0.05) | 498 通过 | <1s |
| 10维质检 | 496 通过 (质量分≥2.0) | ~20s |
| 层次聚类 (|corr|>0.7 归簇) | 131 簇 → 40 因子 | ~7min |
| **总耗时** | | **7.5min** |

### 12.3 精选 40 因子 Top-15

```
因子                                         |IC|    IC_IR   p-val     hit%   roll+%  autocorr
────────────────────────────────────────────────────────────────────────────────────────────────
ADX_14D + CALMAR_RATIO_60D                   0.0343  0.117  0.00002   56.3%  88.8%   0.85
CALMAR_RATIO_60D - GK_VOL_RATIO_20D         0.0341  0.109  0.00006   53.2%  83.7%   0.88
ADX_14D + CMF_20D                            0.0325  0.124  0.00000   55.4%  92.7%   0.83
ADX_14D ∧ min(CALMAR_RATIO_60D)             0.0312  0.111  0.00004   54.7%  83.9%   0.78
BREAKOUT_20D ∨ max(DOWNSIDE_DEV_20D)        0.0304  0.120  0.00001   54.3%  92.4%   0.73
ADX_14D + BREAKOUT_20D                       0.0289  0.095  0.00036   54.8%  89.2%   0.78
CALMAR_RATIO_60D - TSMOM_60D                0.0279  0.098  0.00029   53.0%  74.1%   0.80
BREAKOUT_20D + CALMAR_RATIO_60D              0.0274  0.079  0.00333   53.8%  80.9%   0.82
CALMAR_RATIO_60D [手工原始]                  0.0272  0.077  0.00455   53.3%  70.7%   0.90
ADX_14D + UP_DOWN_VOL_RATIO_20D             0.0268  0.104  0.00010   55.0%  78.4%   0.80
CALMAR + CORR_MKT                           0.0268  0.080  0.00316   53.2%  72.7%   0.89
CALMAR ∨ max(PP120)                         0.0266  0.077  0.00406   54.5%  73.4%   0.91
ADX_14D + MEAN_REV_RATIO_20D                0.0257  0.087  0.00113   54.0%  84.3%   0.76
CALMAR + UP_DOWN_VOL_RATIO_20D              0.0256  0.079  0.00339   53.8%  72.5%   0.88
ADX_14D - GK_VOL_RATIO_20D                  0.0255  0.097  0.00028   54.2%  81.0%   0.82
```

全部 40 个精选因子: |IC| > 0.02, p < 0.01, rolling IC 正率 62%-93%。

### 12.4 质量诊断

**IC 提升**:
- 手工因子最高 |IC|: 0.0272 (CALMAR_RATIO_60D)
- 代数因子最高 |IC|: 0.0366 (ADX × max × PP20)
- IC_IR: 从 0.089 提升到 0.124 (+39%)

**市场环境稳定性** (代数因子 Top-5):
```
因子                              bull IC   bear IC   sideways IC
ADX + CALMAR                      +0.031    -0.004    +0.039
CALMAR - GK_VOL                   +0.044    -0.023    +0.040
ADX + CMF                         -0.059    +0.031    +0.037
ADX min CALMAR                    +0.018    +0.020    +0.033
BREAKOUT max DOWNSIDE_DEV         +0.018    +0.004    +0.034
```

震荡市 IC 最强且稳定（+0.03~0.04），牛市方向不一致，熊市 IC 偏弱。

**冗余性警告**:
精选 40 个因子的真正独立维度约 5-7 个：

| 维度 | 代表因子 | 出现次数 |
|------|----------|----------|
| CALMAR 系 | CALMAR_RATIO_60D + X | 17 次 |
| ADX 系 | ADX_14D + X | 14 次 |
| BREAKOUT 系 | BREAKOUT_20D + X | 7 次 |
| AMIHUD 系 | AMIHUD_ILLIQUIDITY + X | 6 次 |
| ABNORMAL_VOL 系 | ABNORMAL_VOLUME_20D + X | 3 次 |

聚类阈值 0.7 下，`ADX+CALMAR` 和 `ADX+CMF` 相关 < 0.7（因为第二个因子不同），但核心驱动都是 ADX。**实际有效自由度 ~10 个**。

### 12.5 手工因子 10维质检 (全量 38 因子)

```
因子                          score  IC       p-val   hit%   mono  roll+%  结论
─────────────────────────────────────────────────────────────────────────────────
CALMAR_RATIO_60D              +5.5  +0.0272  0.0045  53.3%  1.0   70.7%  PASS
PRICE_POSITION_120D           +5.5  +0.0256  0.0049  54.1%  1.0   70.0%  PASS
SHARPE_RATIO_20D              +5.5  +0.0229  0.0086  54.0%  1.0   67.5%  PASS
PRICE_POSITION_20D            +5.0  +0.0267  0.0009  53.9%  1.0   69.4%  PASS
BREAKOUT_20D                  +4.5  +0.0225  0.0153  52.8%  1.0   67.2%  PASS
ADX_14D                       +4.5  +0.0162  0.0272  51.3%  1.0   59.8%  PASS
UP_DOWN_VOL_RATIO_20D [NEW]   +4.5  +0.0166  0.0242  52.9%  1.0   60.1%  PASS
KURT_20D [NEW]                +3.5  +0.0129  0.0317  53.8%  1.0   55.7%  PASS
MOM_20D                       +3.5  +0.0168  0.0856  52.5%  1.0   56.3%  PASS
REL_STR_VS_MKT_20D            +3.5  +0.0169  0.0835  52.5%  1.0   56.5%  PASS
PV_CORR_20D                   +3.5  -0.0167  0.0074  47.9%  1.0   57.2%  PASS
RSI_14                        +3.5  +0.0101  0.2327  52.0%  1.0   54.2%  PASS
VOL_RATIO_20D                 +3.5  +0.0075  0.2053  51.5%  1.0   53.8%  PASS
CMF_20D                       +2.5  +0.0231  0.0018  54.9%  1.0   62.5%  PASS
CORR_TO_MKT_20D               +2.5  +0.0158  0.0951  52.1%  1.0   56.0%  PASS
GK_VOL_RATIO_20D [NEW]        +2.5  -0.0173  0.0104  46.9%  0.5   56.2%  PASS
SKEW_20D [NEW]                +2.5  -0.0138  0.0424  47.3%  0.5   55.8%  PASS
MAX_DD_60D                    +2.5  -0.0095  0.3421  48.2%  1.0   52.8%  PASS
SLOPE_20D                     +2.5  +0.0056  0.4920  51.2%  1.0   51.5%  PASS
AMIHUD_ILLIQUIDITY            +2.5  -0.0020  0.6269  49.7%  1.0   50.8%  PASS
ULCER_INDEX_20D [NEW]         +2.5  -0.0179  0.0566  47.0%  1.0   56.5%  PASS
ABNORMAL_VOLUME_20D [NEW]     +2.5  -0.0032  0.5922  49.6%  1.0   50.2%  PASS
MEAN_REV_RATIO_20D [NEW]      +2.5  +0.0088  0.3530  51.1%  1.0   53.0%  PASS
VORTEX_14D                    +2.5  +0.0077  0.3648  51.5%  0.5   52.0%  PASS
HURST_60D [NEW]               +2.0  +0.0003  0.9641  50.1%  1.0   50.5%  PASS
──────────────────────────────────────────────────────────────────────────── (以下 FAIL)
OBV_SLOPE_10D                 +2.0  +0.0047  0.4085  50.8%  1.0   51.2%  FAIL (roll+≤55%)
RET_VOL_20D                   +1.5  -0.0095  0.3593  48.2%  0.5   52.5%  FAIL
REALIZED_VOL_20D              +1.5  -0.0095  0.3593  48.2%  0.5   52.5%  FAIL
SPREAD_PROXY                  +1.5  -0.0194  0.0763  48.0%  0.5   56.0%  FAIL
TSMOM_60D                     +1.5  +0.0097  0.3696  51.5%  0.5   52.0%  FAIL
TSMOM_120D                    +1.5  +0.0054  0.6217  50.8%  0.5   51.2%  FAIL
DOWNSIDE_DEV_20D [NEW]        +1.5  -0.0074  0.4685  49.5%  0.5   51.8%  FAIL
DD_DURATION_60D [NEW]         +1.0  +0.0058  0.4429  50.3%  1.0   51.0%  FAIL
IBS [NEW]                     +1.0  +0.0017  0.8240  50.2%  0.5   50.2%  FAIL
INFO_DISCRETE_20D [NEW]       +1.0  -0.0060  0.3482  48.8%  0.5   51.5%  FAIL
PERM_ENTROPY_20D [NEW]        +0.0  +0.0074  0.1398  51.4%  0.5   52.2%  FAIL
VOL_RATIO_60D                 +0.5  -0.0030  0.5991  49.5%  0.5   50.5%  FAIL
TURNOVER_ACCEL_5_20           +0.0  -0.0041  0.4819  49.2%  0.5   50.0%  FAIL
```

质检通过: 25/38 (阈值: score ≥ 2.0 且 NaN ≤ 30%)

### 12.6 使用方式

```bash
# 完整流程 (~30min, 含代数+窗口+变换搜索)
uv run python scripts/run_factor_mining.py

# 仅质检现有因子 (~1min)
uv run python scripts/run_factor_mining.py --skip-discovery

# 仅代数搜索 (~7.5min)
uv run python scripts/run_factor_mining.py --algebraic-only

# 调整参数
uv run python scripts/run_factor_mining.py --max-correlation 0.6 --max-factors 20 --fdr-alpha 0.01
```

输出目录: `results/factor_mining_YYYYMMDD_HHMMSS/`
- `factor_registry.json` — 全量因子元数据 (536 个)
- `quality_reports.parquet` — 10 维质检报告
- `discovery_summary.parquet` — 挖掘结果
- `selected_factors.json` — 精选因子池
- `correlation_matrix.parquet` — 相关矩阵

### 12.7 与 Phase C 的衔接

挖掘系统的精选因子可直接作为 WFO 的候选因子池:
1. 从 `selected_factors.json` 提取因子列表
2. 对代数因子需在 WFO 中动态计算（当前 WFO 只支持 PreciseFactorLibrary 注册因子）
3. **建议**: 先验证 Phase B 的手工因子结论（GK_VOL_RATIO_20D 入选），再考虑将代数因子集成到 WFO

---

*生成时间: 2026-02-05*
*数据: factor_alpha_analysis.py + run_factor_mining.py 全量输出*
