# One Pass 全量面板方案 - 使用指南

## 🎯 方案概述

**核心理念**：一次性计算所有因子，只做4条最小安全约束，不做前置筛选，把筛选权交给后续分析。

### 4条最小安全约束

1. **T+1安全**：每个因子内部必须先shift(1)再pct_change/rolling
2. **min_history**：窗口不足一律NaN（不前填/不插值）
3. **口径一致**：统一用同一价格列，price_field写入元数据
4. **容错记账**：计算报错不终止，列全NaN；记录missing_fields/错误原因

### 告警不阻塞

- **覆盖率告警**：coverage<10% 打WARN（仍保留列）
- **零方差告警**：整列常数值 打WARN（仍保留列）
- **重复列告警**：与他列完全一致分配identical_group_id（仍保留列）
- **时序哨兵**：随机抽样数点，断言仅使用≤T数据（失败标注leak_suspect=true，但仍写出）

---

## 🚀 快速开始

### Step 1: 生产全量面板

```bash
python3 scripts/produce_full_etf_panel.py \
    --start-date 20240101 \
    --end-date 20251014 \
    --data-dir raw/ETF/daily \
    --output-dir factor_output/etf_rotation
```

**输出**：
- `factor_output/etf_rotation/panel_FULL_20240101_20251014.parquet`（全量面板）
- `factor_output/etf_rotation/factor_summary_20240101_20251014.csv`（因子概要）
- `factor_output/etf_rotation/panel_meta.json`（元数据）

### Step 2: 筛选高质量因子

#### 生产模式（严格）

```bash
python3 scripts/filter_factors_from_panel.py \
    --panel-file factor_output/etf_rotation/panel_FULL_20240101_20251014.parquet \
    --summary-file factor_output/etf_rotation/factor_summary_20240101_20251014.csv \
    --mode production
```

**筛选规则**：
- coverage ≥ 80%
- zero_variance = False
- leak_suspect = False
- 去重（identical_group_id）

**输出**：
- `panel_filtered_production.parquet`（筛选后的面板）
- `factors_selected_production.yaml`（因子清单）
- `correlation_matrix.csv`（相关性矩阵）

#### 研究模式（宽松）

```bash
python3 scripts/filter_factors_from_panel.py \
    --panel-file factor_output/etf_rotation/panel_FULL_20240101_20251014.parquet \
    --summary-file factor_output/etf_rotation/factor_summary_20240101_20251014.csv \
    --mode research
```

**筛选规则**：
- coverage ≥ 30%（宽松）
- 允许零方差
- 允许泄露嫌疑
- 去重（identical_group_id）

**用途**：快速做IC/IR/相关性分析，自选"高价值"因子

---

## 📊 输出物说明

### 1. 全量面板（panel_FULL_*.parquet）

**结构**：
- **索引**：MultiIndex (symbol, date)
- **列**：所有注册因子（包含失败的，值为NaN）
- **特点**：尽可能全，包含资金流/分钟依赖的因子（自然为NaN）

**示例**：
```python
import pandas as pd

panel = pd.read_parquet("factor_output/etf_rotation/panel_FULL_20240101_20251014.parquet")
print(panel.shape)  # (56575, 154)  # 假设有154个因子
print(panel.head())
```

### 2. 因子概要（factor_summary_*.csv）

**字段**：
- `factor_id`：因子ID
- `coverage`：覆盖率（非NaN占比）
- `zero_variance`：是否零方差
- `min_history`：最小历史数据要求
- `required_fields`：所需字段（如adj_close）
- `reason`：计算结果（success或错误原因）
- `identical_group_id`：重复组ID（如有）

**示例**：
```python
summary = pd.read_csv("factor_output/etf_rotation/factor_summary_20240101_20251014.csv")

# 查看覆盖率分布
print(summary['coverage'].describe())

# 查看失败因子
failed = summary[summary['reason'] != 'success']
print(failed[['factor_id', 'reason']])

# 查看重复组
duplicates = summary[summary['identical_group_id'].notna()]
print(duplicates.groupby('identical_group_id')['factor_id'].apply(list))
```

### 3. 元数据（panel_meta.json）

**内容**：
```json
{
  "engine_version": "1.0.0",
  "price_field": "adj_close",
  "run_params": {
    "start_date": "20240101",
    "end_date": "20251014",
    "data_dir": "raw/ETF/daily"
  },
  "timestamp": "2025-10-15T12:00:00"
}
```

---

## 🎯 后续筛选策略

### 一行规则（示例）

```python
import pandas as pd

# 加载概要
summary = pd.read_csv("factor_output/etf_rotation/factor_summary_20240101_20251014.csv")

# 筛选规则
selected = summary[
    (summary['coverage'] >= 0.8) &
    (~summary['zero_variance']) &
    (summary['reason'] == 'success')
]

# 去重（每组保留第一个）
if 'identical_group_id' in selected.columns:
    selected = selected.drop_duplicates(subset=['identical_group_id'], keep='first')

# 提取因子列表
factor_list = selected['factor_id'].tolist()
print(f"筛选出 {len(factor_list)} 个因子")

# 保存清单
import yaml
with open('factors_selected.yaml', 'w') as f:
    yaml.dump({'factors': factor_list}, f)
```

### 研究/生产随选

#### 研究模式
- **目标**：快速探索，找出高价值因子
- **阈值**：coverage≥30%，允许零方差
- **用途**：IC/IR分析，相关性分析，因子挖掘

#### 生产模式
- **目标**：高质量因子，稳定可靠
- **阈值**：coverage≥80%，zero_variance=False，leak_suspect=False
- **用途**：策略评分，轮动决策，实盘交易

---

## 🔧 工程约定

### 1. 索引统一
- **MultiIndex**: (symbol, date)
- **date**: normalize()，tz-naive
- **排序**: sort_index()

### 2. 价格口径
- **优先**: adj_close
- **回退**: close
- **记录**: meta['price_field']

### 3. 失败不阻塞
- **try/except**: 因子计算失败不终止
- **全NaN列**: 失败因子写入全NaN
- **记录原因**: summary['reason']

### 4. 参数入缓存键
- **格式**: factor_id + params + price_field + engine_version
- **目的**: 避免不同窗口算成同一列

### 5. 性能优化
- **向量化**: 使用groupby.apply，避免显式循环
- **分批写列**: chunk处理（如需要）
- **中间态**: 落tmp/防止长任务中断（如需要）

---

## 📈 使用场景

### 场景1：快速因子探索

```bash
# 1. 生产全量面板
python3 scripts/produce_full_etf_panel.py --start-date 20240101 --end-date 20251014

# 2. 研究模式筛选（宽松）
python3 scripts/filter_factors_from_panel.py \
    --panel-file factor_output/etf_rotation/panel_FULL_20240101_20251014.parquet \
    --summary-file factor_output/etf_rotation/factor_summary_20240101_20251014.csv \
    --mode research

# 3. 分析因子IC/IR
python3 scripts/analyze_factor_ic.py \
    --panel-file factor_output/etf_rotation/panel_filtered_research.parquet
```

### 场景2：生产策略部署

```bash
# 1. 生产全量面板
python3 scripts/produce_full_etf_panel.py --start-date 20240101 --end-date 20251014

# 2. 生产模式筛选（严格）
python3 scripts/filter_factors_from_panel.py \
    --panel-file factor_output/etf_rotation/panel_FULL_20240101_20251014.parquet \
    --summary-file factor_output/etf_rotation/factor_summary_20240101_20251014.csv \
    --mode production

# 3. 使用筛选后的因子进行轮动
python3 scripts/etf_monthly_rotation.py \
    --trade-date 20241031 \
    --panel-file factor_output/etf_rotation/panel_filtered_production.parquet \
    --factor-list factor_output/etf_rotation/factors_selected_production.yaml
```

### 场景3：自定义筛选

```python
import pandas as pd
import yaml

# 加载全量面板和概要
panel = pd.read_parquet("factor_output/etf_rotation/panel_FULL_20240101_20251014.parquet")
summary = pd.read_csv("factor_output/etf_rotation/factor_summary_20240101_20251014.csv")

# 自定义筛选规则
selected = summary[
    (summary['coverage'] >= 0.6) &  # 覆盖率≥60%
    (~summary['zero_variance']) &
    (summary['factor_id'].str.contains('Momentum|VOLATILITY'))  # 只要动量和波动类
]

# 提取因子
factor_list = selected['factor_id'].tolist()

# 提取筛选后的面板
selected_panel = panel[factor_list]

# 保存
selected_panel.to_parquet("factor_output/etf_rotation/panel_custom.parquet")
with open('factors_custom.yaml', 'w') as f:
    yaml.dump({'factors': factor_list}, f)
```

---

## ⚠️ 注意事项

### 1. 数据质量
- 确保raw/ETF/daily下的数据完整
- 检查adj_close列是否存在
- 验证日期格式正确

### 2. 计算时间
- 全量计算可能需要较长时间（取决于因子数量和数据量）
- 建议先用小范围日期测试
- 可考虑并行计算优化（如需要）

### 3. 存储空间
- 全量面板可能较大（取决于因子数量）
- 建议定期清理旧面板
- 使用parquet格式压缩存储

### 4. 因子注册
- 确保所有因子已在FactorRegistry中注册
- 检查因子的min_history属性
- 验证因子的calculate方法正确

---

## 🎉 优势

### 1. 单通道方案
- 一次性计算所有因子，避免重复计算
- 统一的数据口径和时序逻辑
- 后续筛选灵活，不需要重新计算

### 2. 告警不阻塞
- 覆盖率低、零方差、重复列只告警，仍保留
- 计算失败不终止，记录原因
- 最大化保留信息，后续自主筛选

### 3. 研究/生产分离
- 研究模式：宽松筛选，快速探索
- 生产模式：严格筛选，高质量因子
- 一个面板，两种用途

### 4. 可追溯性
- 元数据记录价格口径、运行参数
- 因子概要记录覆盖率、零方差、重复组
- 完整的诊断信息，便于问题排查

---

## 📞 快速命令

```bash
# 生产全量面板
python3 scripts/produce_full_etf_panel.py \
    --start-date 20240101 \
    --end-date 20251014

# 生产模式筛选
python3 scripts/filter_factors_from_panel.py \
    --panel-file factor_output/etf_rotation/panel_FULL_20240101_20251014.parquet \
    --summary-file factor_output/etf_rotation/factor_summary_20240101_20251014.csv \
    --mode production

# 研究模式筛选
python3 scripts/filter_factors_from_panel.py \
    --panel-file factor_output/etf_rotation/panel_FULL_20240101_20251014.parquet \
    --summary-file factor_output/etf_rotation/factor_summary_20240101_20251014.csv \
    --mode research

# 自定义覆盖率阈值
python3 scripts/filter_factors_from_panel.py \
    --panel-file factor_output/etf_rotation/panel_FULL_20240101_20251014.parquet \
    --summary-file factor_output/etf_rotation/factor_summary_20240101_20251014.csv \
    --mode production \
    --min-coverage 0.7
```

---

**最后更新**：2025-10-15  
**版本**：v1.0.0  
**状态**：✅ 生产就绪
