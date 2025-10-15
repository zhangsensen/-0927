# ETF因子引擎 - 生产版本

## 🎯 项目概述

ETF日线因子计算引擎，基于VectorBT + TA-Lib实现，支持370+个技术指标的批量计算。

### 核心特性
- ✅ **370+因子**：VBT(152) + TA-Lib(193) + 自定义(25)
- ✅ **高覆盖率**：平均97.6%
- ✅ **时序安全**：T+1保证，无未来信息泄露
- ✅ **One Pass**：一次性计算全量因子，后续筛选
- ✅ **研究/生产分离**：灵活的筛选阈值

---

## 📁 项目结构

```
etf_factor_engine_production/
├── factor_system/
│   └── factor_engine/
│       └── adapters/
│           ├── __init__.py
│           └── vbt_adapter.py          # VBT适配器（核心）
├── scripts/
│   ├── produce_full_etf_panel.py       # 全量面板生产
│   ├── filter_factors_from_panel.py    # 因子筛选
│   └── test_one_pass_panel.py          # 测试验证
├── configs/
│   └── etf_config.yaml                 # 配置文件
├── factor_output/
│   └── etf_rotation/                   # 输出目录
│       ├── panel_FULL_*.parquet        # 全量面板
│       ├── factor_summary_*.csv        # 因子概要
│       └── panel_meta.json             # 元数据
└── README.md                           # 本文件
```

---

## 🚀 快速开始

### 1. 生产全量因子面板

```bash
python3 scripts/produce_full_etf_panel.py \
    --start-date 20200102 \
    --end-date 20251014 \
    --data-dir raw/ETF/daily \
    --output-dir factor_output/etf_rotation
```

**输出**：
- `panel_FULL_20200102_20251014.parquet`：370个因子 × 56,575样本
- `factor_summary_20200102_20251014.csv`：因子概要统计
- `panel_meta.json`：元数据

### 2. 筛选高质量因子

```bash
# 生产模式（严格）
python3 scripts/filter_factors_from_panel.py \
    --panel-file factor_output/etf_rotation/panel_FULL_20200102_20251014.parquet \
    --summary-file factor_output/etf_rotation/factor_summary_20200102_20251014.csv \
    --mode production

# 研究模式（宽松）
python3 scripts/filter_factors_from_panel.py \
    --panel-file factor_output/etf_rotation/panel_FULL_20200102_20251014.parquet \
    --summary-file factor_output/etf_rotation/factor_summary_20200102_20251014.csv \
    --mode research
```

### 3. 验证结果

```bash
python3 scripts/test_one_pass_panel.py
```

---

## 📊 因子列表

### VBT内置指标（152个）
- **MA系列**：13个窗口 (5-252)
- **EMA系列**：12个窗口 (5-120)
- **MACD**：4组参数 × 3指标 = 12个
- **RSI**：8个窗口 (6-60)
- **BBANDS**：7窗口 × 3标准差 × 5指标 = 105个
- **STOCH**：4窗口 × 2平滑 × 2指标 = 16个
- **ATR**：6个窗口 (7-60)
- **OBV**：1个

### TA-Lib完整指标（193个）
- **Overlap Studies**：SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, MAMA等
- **Momentum**：MACD, RSI, MOM, ROC, ROCP, ROCR等
- **Volatility**：BBANDS, ATR, NATR, TRANGE, STDDEV, VAR等
- **Volume**：OBV, AD, ADOSC等
- **Cycle**：HT_DCPERIOD, HT_DCPHASE, HT_TRENDMODE等
- **Price Transform**：AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE等
- **Pattern Recognition**：DOJI, HAMMER, ENGULFING, MORNINGSTAR等

### 自定义统计指标（25个）
- **收益率系列**：8个周期 (1-60天)
- **波动率系列**：5个窗口 (5-60天)
- **价格位置**：4个窗口 (10-60天)
- **成交量比率**：4个窗口 (5-30天)
- **动量指标**：4个窗口 (5-30天)

---

## 🎯 使用场景

### 场景1：ETF轮动策略
```python
import pandas as pd

# 加载筛选后的因子
panel = pd.read_parquet('factor_output/etf_rotation/panel_filtered_production.parquet')

# 计算因子得分
scores = panel.rank(pct=True).mean(axis=1)

# 每月选择Top 5
monthly_top5 = scores.groupby(level='date').apply(lambda x: x.nlargest(5))
```

### 场景2：因子研究
```python
# 加载全量面板
panel = pd.read_parquet('factor_output/etf_rotation/panel_FULL_20200102_20251014.parquet')

# 计算IC
returns = panel.groupby(level='symbol')['close'].pct_change(20)
ic = panel.corrwith(returns, axis=0)

# 筛选高IC因子
high_ic_factors = ic[ic.abs() > 0.05].index.tolist()
```

---

## ⚙️ 配置说明

### 数据要求
- **格式**：Parquet
- **列名**：ts_code, trade_date, open, high, low, close, vol
- **频率**：日线
- **市场**：ETF

### 筛选参数
- **生产模式**：coverage≥80%, zero_variance=False, 去重
- **研究模式**：coverage≥30%, 允许零方差, 去重

---

## 📈 性能指标

### 计算性能
- **单ETF**：~50ms（370个指标）
- **43个ETF**：~2秒（5年数据）
- **内存峰值**：~2GB

### 存储空间
- **全量面板**：~150MB（Parquet压缩）
- **筛选面板**：~80MB

---

## 🔧 技术栈

- **Python**: 3.11+
- **VectorBT**: 0.28+
- **TA-Lib**: 0.6.7+
- **Pandas**: 2.3+
- **NumPy**: 2.3+

---

## 📝 更新日志

### v1.0.0 (2025-10-15)
- ✅ 打通VBT引擎
- ✅ 实现370+因子计算
- ✅ One Pass全量面板方案
- ✅ 研究/生产分离筛选
- ✅ 完整的诊断和验证工具

---

## 📞 快速命令

```bash
# 生产全量面板
python3 scripts/produce_full_etf_panel.py --start-date 20200102 --end-date 20251014

# 生产模式筛选
python3 scripts/filter_factors_from_panel.py --mode production

# 研究模式筛选
python3 scripts/filter_factors_from_panel.py --mode research

# 验证结果
python3 scripts/test_one_pass_panel.py
```

---

**版本**：v1.0.0  
**日期**：2025-10-15  
**状态**：✅ 生产就绪
