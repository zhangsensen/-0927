# 🚀 ETF横截面因子系统 - 生产环境真实状态

**状态**: �� 部分就绪（43个有效因子）  
**最后更新**: 2025-10-16  
**成功率**: 27.2% (43/158)  

---

## 📊 真实系统概述

**可用因子**: 43个（100%有效）  
**失败因子**: 115个（VBT不支持）  
**ETF覆盖**: 43只  
**数据完整度**: 100%（有效因子）  

---

## 🎯 核心脚本

### `production_full_cross_section.py` - 唯一生产脚本

**功能**: 计算所有ETF的完整因子横截面数据

**使用方法**:
```bash
cd /Users/zhangshenshen/深度量化0927
python scripts/production_full_cross_section.py
```

**输出文件**:
- `output/cross_sections/cross_section_YYYYMMDD.parquet` - 横截面数据
- `output/cross_sections/factor_effectiveness_stats.csv` - 真实统计

---

## 📈 真实可用因子 (43个)

### 1. 传统因子 (3个)
- `BB_20_2.0_Width`: 布林带宽度
- `BOLB_20`: 布林带位置
- `OBV`: 能量潮指标

### 2. 自定义因子 (16个)

**流动性因子** (5个):
- `avg_volume_21d`, `avg_amount_21d`, `volume_stability`
- `turnover_rate`, `liquidity_score`

**技术指标** (7个):
- `rsi_14`, `macd`, `macd_signal`, `macd_histogram`
- `bb_position`, `williams_r`, `cci_14`

**综合评分** (4个):
- `vpt`, `technical_score`, `technical_score_normalized`, `composite_score`

### 3. VBT动态因子 (20个)

**MA系列** (12个):
- `VBT_MA3_window5/10/20/50/120`
- `VBT_MA5_window5/10/20/50/120`
- `VBT_MA10_window10/20/50/120`

**EMA系列** (4个):
- `VBT_EMA_window10/20/50/120`

**其他** (4个):
- `VBT_ATR_window7/14/21`
- `VBT_BB__*` (布林带变体)

### 4. TA-Lib因子 (4个)
- `TA_SAR_acceleration0.02_maximum0.2`
- `TA_SAR_acceleration0.02_maximum0.4`
- `TA_SAR_acceleration0.04_maximum0.2`
- `TA_SAR_acceleration0.04_maximum0.4`

---

## ⚠️ 已知限制

### 不可用因子类别

**VBT不支持** (60+个):
- 成交量指标: AD, ADOSC
- 趋势指标: ADX, AROON
- 动量指标: APO, PPO, CCI, MFI, ROC, MOM, WILLR
- 其他: STOCH, STOCHRSI, ULTOSC

**TA-Lib K线形态** (50+个):
- 所有CDL_*系列形态识别
- HT_SINE, HT_TRENDLINE等

**传统因子bug** (8个):
- MOMENTUM_21D/63D/126D/252D
- VOLATILITY_20D/60D/120D/252D

---

## 🔧 配置说明

### 传统因子配置
编辑 `configs/legacy_factors.yaml`:
```yaml
enabled_categories:
  - technical  # BB_20_2.0_Width, BOLB_20
  - volume     # OBV
  # momentum和volatility暂时禁用（有bug）
```

### 时间窗口配置
```python
# scripts/production_full_cross_section.py
start_date = target_date - timedelta(days=365)  # 1年数据窗口
```

---

## 📊 输出数据格式

### 横截面数据 (Parquet)
```
Index: symbol (43只ETF)
Columns: 43个有效因子
```

### 统计数据 (CSV)
```
factor_id, valid_count, valid_rate, non_zero_count, is_all_zero, mean, std
```

**关键字段**:
- `non_zero_count`: 非零数据点数量
- `is_all_zero`: 是否计算失败（全零/全空）

---

## 🎯 真实验证结果

**最新验证**: 2025-10-16

```
横截面数据: 43只ETF × 43个有效因子
真正有效: 43个 (27.2%)
计算失败: 115个 (72.8%)
数据完整度: 100% (有效因子)
```

**有效因子示例** (前10个):
1. avg_volume_21d: 43/43 非零
2. avg_amount_21d: 43/43 非零
3. volume_stability: 43/43 非零
4. turnover_rate: 43/43 非零
5. liquidity_score: 43/43 非零
6. rsi_14: 43/43 非零
7. macd: 43/43 非零
8. macd_signal: 43/43 非零
9. macd_histogram: 43/43 非零
10. bb_position: 43/43 非零

---

## 🚨 故障排查

### 问题1: 大量因子失败
**症状**: ERROR日志中出现"module 'vectorbt' has no attribute 'XXX'"  
**原因**: VectorBT 0.28不支持该指标  
**解决**: 这是正常的，系统会自动跳过。未来版本会移除这些因子注册

### 问题2: 统计显示失败因子
**症状**: factor_effectiveness_stats.csv中`is_all_zero=True`  
**原因**: 因子计算失败，返回NaN  
**解决**: 使用`non_zero_count > 0`的因子，忽略失败因子

### 问题3: 传统因子缺失
**症状**: MOMENTUM_*和VOLATILITY_*未出现在输出中  
**原因**: 传统计算器有bug，已在配置中禁用  
**解决**: 使用VBT动态因子替代（如VBT_MOM, VBT_ROC, VBT_ATR）

---

## 📝 维护指南

### 查看真实有效因子
```python
import pandas as pd

stats = pd.read_csv('output/cross_sections/factor_effectiveness_stats.csv')
effective = stats[stats['non_zero_count'] >= 20]  # 至少20个ETF有数据
print(effective['factor_id'].tolist())
```

### 过滤失败因子
```python
cross = pd.read_parquet('output/cross_sections/cross_section_20251014.parquet')
stats = pd.read_csv('output/cross_sections/factor_effectiveness_stats.csv')

# 只保留有效因子
valid_factors = stats[~stats['is_all_zero']]['factor_id'].tolist()
cross_clean = cross[valid_factors]
```

---

## 🎊 系统特性

✅ **真实统计** - 准确反映失败情况  
✅ **数据完整** - 有效因子100%完整  
✅ **代码质量** - Linus式清理完成  
⚠️  **因子有限** - 43个可用（需扩展）  
⚠️  **需要优化** - 移除不支持的因子注册  

---

## 📞 技术支持

**详细报告**: `PRODUCTION_REALITY_REPORT.md`  
**日志文件**: `production_full_cross_section.log`  
**配置文件**: `configs/legacy_factors.yaml`  

---

**最后更新**: 2025-10-16  
**版本**: v1.0.1-reality  
**状态**: 🟡 部分就绪，需要优化  
