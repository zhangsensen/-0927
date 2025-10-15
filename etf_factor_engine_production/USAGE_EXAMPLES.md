# 使用示例

## 📚 目录
1. [基础使用](#基础使用)
2. [ETF轮动策略](#etf轮动策略)
3. [因子研究](#因子研究)
4. [自定义筛选](#自定义筛选)

---

## 基础使用

### 1. 生产全量因子面板

```bash
cd /Users/zhangshenshen/深度量化0927/etf_factor_engine_production

python3 scripts/produce_full_etf_panel.py \
    --start-date 20200102 \
    --end-date 20251014 \
    --data-dir ../raw/ETF/daily \
    --output-dir ../factor_output/etf_rotation
```

### 2. 筛选高质量因子

```bash
# 生产模式（coverage≥80%）
python3 scripts/filter_factors_from_panel.py \
    --panel-file ../factor_output/etf_rotation/panel_FULL_20200102_20251014.parquet \
    --summary-file ../factor_output/etf_rotation/factor_summary_20200102_20251014.csv \
    --mode production \
    --output-dir ../factor_output/etf_rotation

# 研究模式（coverage≥30%）
python3 scripts/filter_factors_from_panel.py \
    --panel-file ../factor_output/etf_rotation/panel_FULL_20200102_20251014.parquet \
    --summary-file ../factor_output/etf_rotation/factor_summary_20200102_20251014.csv \
    --mode research \
    --output-dir ../factor_output/etf_rotation
```

### 3. 验证结果

```bash
python3 scripts/test_one_pass_panel.py
```

---

## ETF轮动策略

### 月度轮动（Top 5）

```python
import pandas as pd
import numpy as np

# 加载筛选后的因子
panel = pd.read_parquet('../factor_output/etf_rotation/panel_filtered_production.parquet')

# 计算综合得分（等权）
scores = panel.rank(pct=True, axis=0).mean(axis=1)

# 按月分组，选择Top 5
def select_top5(group):
    return group.nlargest(5)

monthly_holdings = scores.groupby(pd.Grouper(level='date', freq='M')).apply(select_top5)

print(f"选中的ETF数量: {monthly_holdings.groupby(level='date').size().mean():.1f} 个/月")
print(f"月度换手率: {monthly_holdings.groupby(level='date').apply(lambda x: x.index.get_level_values('symbol').nunique()).mean():.1f}")
```

### 动态权重分配

```python
# 基于因子得分的动态权重
def calculate_weights(group):
    """计算每个ETF的权重"""
    scores = group.values
    # 归一化到[0, 1]
    normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
    # 等权或按得分加权
    weights = normalized / normalized.sum()
    return pd.Series(weights, index=group.index)

monthly_weights = scores.groupby(pd.Grouper(level='date', freq='M')).apply(
    lambda x: calculate_weights(x.nlargest(5))
)

print("权重分布:")
print(monthly_weights.describe())
```

---

## 因子研究

### IC分析

```python
import pandas as pd

# 加载全量面板
panel = pd.read_parquet('../factor_output/etf_rotation/panel_FULL_20200102_20251014.parquet')

# 计算未来收益
def calculate_forward_returns(group, period=20):
    """计算未来N日收益"""
    close = group['close'] if 'close' in group.columns else group.iloc[:, 0]
    return close.pct_change(period).shift(-period)

# 按symbol分组计算收益
returns = panel.groupby(level='symbol').apply(
    lambda x: calculate_forward_returns(x.reset_index(level=0, drop=True), period=20)
)

# 计算IC（信息系数）
ic_values = {}
for col in panel.columns:
    if col != 'close':
        ic = panel[col].corr(returns)
        ic_values[col] = ic

ic_df = pd.DataFrame.from_dict(ic_values, orient='index', columns=['IC'])
ic_df = ic_df.sort_values('IC', key=abs, ascending=False)

print("Top 20 因子（按IC绝对值）:")
print(ic_df.head(20))

# 筛选高IC因子
high_ic_factors = ic_df[ic_df['IC'].abs() > 0.05].index.tolist()
print(f"\n高IC因子数量（|IC|>0.05）: {len(high_ic_factors)}")
```

### 因子相关性分析

```python
# 计算因子相关性矩阵
corr_matrix = panel.corr()

# 找出高相关因子对
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr = corr_matrix.iloc[i, j]
        if abs(corr) > 0.9:
            high_corr_pairs.append((
                corr_matrix.columns[i],
                corr_matrix.columns[j],
                corr
            ))

print(f"高相关因子对（|ρ|>0.9）: {len(high_corr_pairs)}")
for f1, f2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:10]:
    print(f"{f1} <-> {f2}: {corr:.4f}")
```

### 因子稳定性分析

```python
# 滚动窗口IC
def rolling_ic(panel, returns, window=60):
    """计算滚动IC"""
    rolling_ics = {}
    for col in panel.columns:
        if col != 'close':
            ic_series = panel[col].rolling(window).corr(returns)
            rolling_ics[col] = ic_series
    return pd.DataFrame(rolling_ics)

rolling_ic_df = rolling_ic(panel, returns, window=60)

# 计算IC稳定性（IC标准差）
ic_stability = rolling_ic_df.std()
ic_stability = ic_stability.sort_values()

print("最稳定的20个因子（IC标准差最小）:")
print(ic_stability.head(20))
```

---

## 自定义筛选

### 按因子类别筛选

```python
import pandas as pd

# 加载因子概要
summary = pd.read_csv('../factor_output/etf_rotation/factor_summary_20200102_20251014.csv')

# 筛选VBT因子
vbt_factors = summary[summary['factor_id'].str.startswith('VBT_')]
print(f"VBT因子: {len(vbt_factors)} 个")

# 筛选TA-Lib因子
talib_factors = summary[summary['factor_id'].str.startswith('TA_')]
print(f"TA-Lib因子: {len(talib_factors)} 个")

# 筛选自定义因子
custom_factors = summary[~summary['factor_id'].str.startswith(('VBT_', 'TA_'))]
print(f"自定义因子: {len(custom_factors)} 个")

# 按覆盖率筛选
high_coverage = summary[summary['coverage'] >= 0.95]
print(f"\n高覆盖率因子（≥95%）: {len(high_coverage)} 个")

# 保存筛选结果
selected_factors = high_coverage['factor_id'].tolist()

import yaml
with open('../factor_output/etf_rotation/factors_custom.yaml', 'w') as f:
    yaml.dump({'factors': selected_factors}, f)

print(f"✅ 已保存 {len(selected_factors)} 个因子到 factors_custom.yaml")
```

### 按指标类型筛选

```python
# 动量类因子
momentum_factors = summary[
    summary['factor_id'].str.contains('RSI|MOM|ROC|MACD', case=False)
]
print(f"动量类因子: {len(momentum_factors)} 个")

# 趋势类因子
trend_factors = summary[
    summary['factor_id'].str.contains('MA|EMA|KAMA', case=False)
]
print(f"趋势类因子: {len(trend_factors)} 个")

# 波动率类因子
volatility_factors = summary[
    summary['factor_id'].str.contains('ATR|BB|STDDEV|VAR|VOLATILITY', case=False)
]
print(f"波动率类因子: {len(volatility_factors)} 个")

# 成交量类因子
volume_factors = summary[
    summary['factor_id'].str.contains('OBV|AD|VOLUME', case=False)
]
print(f"成交量类因子: {len(volume_factors)} 个")
```

### 组合筛选

```python
# 组合条件筛选
selected = summary[
    (summary['coverage'] >= 0.90) &  # 覆盖率≥90%
    (~summary['zero_variance']) &     # 非零方差
    (summary['factor_id'].str.contains('RSI|MACD|BB', case=False))  # 特定类型
]

# 去重（如果有identical_group_id）
if 'identical_group_id' in selected.columns:
    selected = selected.drop_duplicates(subset=['identical_group_id'], keep='first')

print(f"组合筛选结果: {len(selected)} 个因子")
print("\n因子列表:")
for factor_id in selected['factor_id'].tolist():
    print(f"  - {factor_id}")
```

---

## 📊 结果可视化

### 因子覆盖率分布

```python
import matplotlib.pyplot as plt

summary = pd.read_csv('../factor_output/etf_rotation/factor_summary_20200102_20251014.csv')

plt.figure(figsize=(10, 6))
plt.hist(summary['coverage'], bins=50, edgecolor='black')
plt.xlabel('Coverage')
plt.ylabel('Count')
plt.title('Factor Coverage Distribution')
plt.axvline(0.80, color='r', linestyle='--', label='Production Threshold (80%)')
plt.axvline(0.30, color='g', linestyle='--', label='Research Threshold (30%)')
plt.legend()
plt.savefig('../factor_output/etf_rotation/coverage_distribution.png', dpi=300)
plt.close()

print("✅ 覆盖率分布图已保存")
```

### IC热图

```python
import seaborn as sns

# 计算因子IC
ic_df = pd.DataFrame(ic_values, index=['IC']).T
ic_df = ic_df.sort_values('IC', ascending=False)

# 绘制热图
plt.figure(figsize=(12, 8))
sns.heatmap(ic_df.head(50).T, cmap='RdYlGn', center=0, annot=False)
plt.title('Top 50 Factors IC Heatmap')
plt.tight_layout()
plt.savefig('../factor_output/etf_rotation/ic_heatmap.png', dpi=300)
plt.close()

print("✅ IC热图已保存")
```

---

**最后更新**: 2025-10-15  
**版本**: v1.0.0
