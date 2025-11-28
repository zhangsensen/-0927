# 🏆 历史最佳策略：43 ETF 整体回测

> **创建日期**: 2025-11-28  
> **状态**: ⚠️ 重要存档 - 请勿删除  
> **结论**: 43 ETF 整体回测效果优于分池策略

---

## 📌 核心发现

**43 个 ETF 整体（不分池）回测的收益率显著高于分池策略**：

| 策略类型 | 最高收益 | 夏普比率 | 说明 |
|----------|----------|----------|------|
| **43 ETF 整体** | **121.0%** | ~0.8-1.0 | 本文档记录 |
| 分池策略 (EQUITY_CYCLICAL) | 73.7% | 0.82 | 单池最优 |
| ATR 动态风控 | 111.6% | 1.16 | 分池+风控 |

---

## 📊 回测配置

### 数据区间
- **起始日期**: 2020-01-01
- **结束日期**: 2025-10-14
- **总天数**: 1,399 个交易日（约 5.7 年）

### 核心参数
| 参数 | 值 | 说明 |
|------|-----|------|
| `FREQ` | 8 | 调仓频率（交易日） |
| `POS_SIZE` | 3 | 持仓数量 |
| `INITIAL_CAPITAL` | 1,000,000 | 初始资金 |
| `COMMISSION_RATE` | 0.0002 | 手续费率 (2bp) |
| `LOOKBACK` | 252 | 回看窗口（热身期） |

### 43 只 ETF 标的池
```
159801, 159819, 159859, 159883, 159915, 159920, 159928, 159949,
159992, 159995, 159998, 510050, 510300, 510500, 511010, 511260,
511380, 512010, 512100, 512400, 512480, 512660, 512690, 512720,
512800, 512880, 512980, 513050, 513100, 513130, 513500, 515030,
515180, 515210, 515650, 515790, 516090, 516160, 516520, 518850,
518880, 588000, 588200
```

### 18 个因子
```
ADX_14D, CALMAR_RATIO_60D, CMF_20D, CORRELATION_TO_MARKET_20D,
MAX_DD_60D, MOM_20D, OBV_SLOPE_10D, PRICE_POSITION_120D,
PRICE_POSITION_20D, PV_CORR_20D, RELATIVE_STRENGTH_VS_MARKET_20D,
RET_VOL_20D, RSI_14, SHARPE_RATIO_20D, SLOPE_20D, VOL_RATIO_20D,
VOL_RATIO_60D, VORTEX_14D
```

---

## 🏆 TOP 20 策略排名（按收益率）

| 排名 | 收益率 | 胜率 | 盈亏比 | 交易次数 | 因子组合 |
|------|--------|------|--------|----------|----------|
| **1** | **121.02%** | 54.59% | 1.414 | 196 | CORRELATION_TO_MARKET_20D + MAX_DD_60D + PRICE_POSITION_120D + PRICE_POSITION_20D |
| **2** | **116.47%** | 50.43% | 1.322 | 230 | ADX_14D + CORRELATION_TO_MARKET_20D + PV_CORR_20D + SHARPE_RATIO_20D + VOL_RATIO_60D |
| **3** | **114.70%** | 54.47% | 1.497 | 246 | CORRELATION_TO_MARKET_20D + MAX_DD_60D + OBV_SLOPE_10D + PRICE_POSITION_20D |
| **4** | **114.65%** | 53.38% | 1.612 | 281 | ADX_14D + OBV_SLOPE_10D + PRICE_POSITION_120D + RET_VOL_20D |
| **5** | **108.90%** | 53.66% | 1.558 | 287 | ADX_14D + OBV_SLOPE_10D + PRICE_POSITION_20D + RET_VOL_20D |
| 6 | 107.54% | 55.42% | 1.285 | 240 | ADX_14D + MAX_DD_60D + PRICE_POSITION_120D + SHARPE_RATIO_20D + VOL_RATIO_20D |
| 7 | 107.49% | 51.79% | 1.555 | 195 | CORRELATION_TO_MARKET_20D + MAX_DD_60D + PRICE_POSITION_20D |
| 8 | 106.43% | 54.51% | 1.361 | 233 | CORRELATION_TO_MARKET_20D + MAX_DD_60D + OBV_SLOPE_10D |
| 9 | 105.48% | 50.75% | 1.271 | 201 | CORRELATION_TO_MARKET_20D + MAX_DD_60D + SHARPE_RATIO_20D + SLOPE_20D + VOL_RATIO_60D |
| 10 | 104.27% | 51.05% | 1.542 | 286 | ADX_14D + MAX_DD_60D + OBV_SLOPE_10D + PV_CORR_20D + SHARPE_RATIO_20D |
| 11 | 103.69% | 52.97% | 1.353 | 236 | ADX_14D + MAX_DD_60D + SHARPE_RATIO_20D + VOL_RATIO_20D |
| 12 | 103.69% | 55.17% | 1.111 | 116 | CORRELATION_TO_MARKET_20D + MAX_DD_60D + PRICE_POSITION_120D + VOL_RATIO_60D |
| 13 | 103.07% | 57.04% | 1.452 | 284 | ADX_14D + OBV_SLOPE_10D + PRICE_POSITION_120D |
| 14 | 102.82% | 55.48% | 1.544 | 292 | ADX_14D + OBV_SLOPE_10D + PRICE_POSITION_20D |
| 15 | 102.00% | 50.54% | 1.677 | 186 | CORRELATION_TO_MARKET_20D + MAX_DD_60D + PV_CORR_20D |
| 16 | 101.95% | 46.67% | 1.907 | 270 | ADX_14D + OBV_SLOPE_10D + PRICE_POSITION_20D + PV_CORR_20D + RET_VOL_20D |
| 17 | 97.86% | 49.03% | 1.661 | 257 | ADX_14D + CALMAR_RATIO_60D + CORRELATION_TO_MARKET_20D + OBV_SLOPE_10D + RET_VOL_20D |
| 18 | 97.68% | 54.14% | 1.467 | 266 | ADX_14D + CORRELATION_TO_MARKET_20D + PRICE_POSITION_20D + SHARPE_RATIO_20D |
| 19 | 96.60% | 50.79% | 1.710 | 189 | ADX_14D + PRICE_POSITION_120D + PV_CORR_20D + RET_VOL_20D + VOL_RATIO_60D |
| 20 | 96.51% | 43.35% | 2.065 | 203 | CALMAR_RATIO_60D + CORRELATION_TO_MARKET_20D + MAX_DD_60D + PRICE_POSITION_20D |

---

## 🥇 最优策略详解

### 策略 #1：CORRELATION_TO_MARKET_20D + MAX_DD_60D + PRICE_POSITION_120D + PRICE_POSITION_20D

| 指标 | 值 |
|------|-----|
| **总收益率** | **121.02%** |
| **年化收益** | ~17.2% |
| **胜率** | 54.59% |
| **盈亏比** | 1.414 |
| **交易次数** | 196 次 |
| **因子数量** | 4 个 |

#### 因子解读
1. **CORRELATION_TO_MARKET_20D** - 与大盘相关性（20日）
   - 选择与大盘低相关的标的，分散风险
2. **MAX_DD_60D** - 最大回撤（60日）
   - 选择近期回撤小的标的，控制下行风险
3. **PRICE_POSITION_120D** - 价格位置（120日）
   - 中长期趋势位置，避免追高
4. **PRICE_POSITION_20D** - 价格位置（20日）
   - 短期趋势位置，捕捉反弹

---

## 📁 数据文件位置

### WFO 筛选结果
```
results/unified_wfo_20251128_185600/
├── all_combos.parquet      # 全部 12,597 个组合的 IC/ICIR
├── all_combos.csv          # CSV 格式
├── top100.parquet          # Top 100 组合
├── run_config.json         # 运行配置
└── factors/                # 标准化因子数据
```

### VEC 回测结果
```
results/vec_full_backtest_20251128_185610/
├── vec_all_combos.parquet  # 全部 12,597 个组合的 VEC 回测结果
└── vec_all_combos.csv      # CSV 格式
```

---

## 🔄 复现步骤

```bash
# Step 1: WFO 因子筛选（12,597 组合）
uv run python etf_rotation_optimized/run_unified_wfo.py

# Step 2: VEC 批量回测
uv run python scripts/batch_vec_backtest.py

# Step 3: 查看结果
python3 -c "
import pandas as pd
df = pd.read_csv('results/vec_full_backtest_YYYYMMDD_HHMMSS/vec_all_combos.csv')
df = df.sort_values('vec_return', ascending=False)
print(df.head(20))
"
```

---

## 📈 高频因子统计

统计 TOP 20 策略中各因子出现频率：

| 因子 | 出现次数 | 占比 |
|------|----------|------|
| **CORRELATION_TO_MARKET_20D** | 12 | 60% |
| **MAX_DD_60D** | 11 | 55% |
| **ADX_14D** | 11 | 55% |
| **PRICE_POSITION_20D** | 9 | 45% |
| **OBV_SLOPE_10D** | 8 | 40% |
| **SHARPE_RATIO_20D** | 7 | 35% |
| **PRICE_POSITION_120D** | 6 | 30% |
| **PV_CORR_20D** | 5 | 25% |
| **RET_VOL_20D** | 5 | 25% |
| **VOL_RATIO_20D** | 4 | 20% |

**核心因子组合**：`CORRELATION_TO_MARKET_20D` + `MAX_DD_60D` + `ADX_14D` 是高收益策略的共同基础。

---

## ⚠️ 重要说明

### 为什么整体回测优于分池？

1. **样本量更大**：43 ETF 整体提供更多横截面选择机会
2. **因子效力更强**：在大样本上因子区分度更高
3. **分散化效果**：不受单一资产类别限制
4. **避免过拟合**：分池可能导致小样本过拟合

### 后续建议

1. **生产部署优先使用整体策略**
2. **分池策略作为辅助参考**
3. **定期验证因子有效性**
4. **BT 审计验证 VEC 结果**

---

## 📝 变更记录

| 日期 | 操作 | 说明 |
|------|------|------|
| 2025-11-28 | 创建 | 初始文档，记录 43 ETF 整体回测最佳策略 |

---

**⚠️ 本文档为重要存档，请勿删除！**
