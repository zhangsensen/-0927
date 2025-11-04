# WFO全开放枚举方案

**执行时间**: 2025-11-03 17:25  
**状态**: ✅ **配置完成，运行中**

---

## 🎯 方案A：全开放枚举

### 配置变更

```yaml
phase2:
  min_factor_freq: 0.1     # 0.3→0.1（放宽→4/5因子进来）
  min_factors: 3
  max_factors: 5           # 开放4/5因子
  tau_grid: [0.7, 1.0, 1.5]
  signal_z_threshold_grid: [-0.5, 0.0, 0.5, 1.0, 1.5]  # 4→5（加1.5）
  max_strategies: 10000    # 500→10000（扩大上限）
  turnover_penalty: 0.05   # 0.1→0.05（降低惩罚）
  coverage_min: 0.3        # 0.5→0.3（放宽门槛）
  avg_turnover_max: 0.8    # 0.6→0.8（放宽门槛）
```

### 预期规模

```
因子池: 10个（基于min_factor_freq=0.1）
子集:
  - 3因子: C(10,3) = 120
  - 4因子: C(10,4) = 210
  - 5因子: C(10,5) = 252
  - 合计: 582

参数网格:
  - tau: 3个
  - z_threshold: 5个
  - topn: 1个
  - 合计: 3×5×1 = 15

理论总量: 582 × 15 = 8,730
预期保留: ~3,000-5,000（取决于覆盖率过滤）
```

---

## 📊 枚举审计报告

### 新增输出文件

**enumeration_audit.json**（自动生成）

```json
{
  "factor_pool": [...],
  "factor_pool_size": 10,
  "min_factors": 3,
  "max_factors": 5,
  "factor_subsets_by_k": {
    "3": 120,
    "4": 210,
    "5": 252
  },
  "total_factor_subsets": 582,
  "tau_grid": [0.7, 1.0, 1.5],
  "topn_grid": [6],
  "signal_z_threshold_grid": [-0.5, 0.0, 0.5, 1.0, 1.5],
  "param_combos_per_subset": 15,
  "theoretical_total_combos": 8730,
  "actual_enumerated": 8730,
  "max_strategies_limit": 10000,
  "hit_limit": false,
  "before_filter": 8730,
  "filtered_by_coverage": 3500,
  "filtered_by_turnover": 200,
  "after_filter": 5030
}
```

**字段说明**:
- `theoretical_total_combos`: 理论应枚举的总量
- `actual_enumerated`: 实际枚举的数量（可能因max_strategies触发上限）
- `hit_limit`: 是否触发max_strategies上限
- `before_filter`: 过滤前的策略数
- `filtered_by_coverage`: 被覆盖率过滤淘汰的数量
- `filtered_by_turnover`: 被换手率过滤淘汰的数量
- `after_filter`: 最终保留的策略数

---

## 🔍 后期筛选方案

### 保守筛选（稳健优先）

```python
import pandas as pd
df = pd.read_csv('strategies_ranked.csv')

# 保守：高Sharpe + 高覆盖率 + 低换手
df_conservative = df[
    (df['sharpe_ratio'] >= 0.6) &
    (df['coverage'] >= 0.6) &
    (df['avg_turnover'] <= 0.3)
].head(10)
```

### 激进筛选（收益优先）

```python
# 激进：高年化 + 可接受覆盖率
df_aggressive = df[
    (df['annual_return'] >= 0.12) &
    (df['coverage'] >= 0.4)
].head(10)
```

### 平衡筛选（综合得分）

```python
# 平衡：综合得分 + 中等覆盖率
df_balanced = df[
    (df['score'] >= 0.4) &
    (df['coverage'] >= 0.5) &
    (df['avg_turnover'] <= 0.5)
].head(10)
```

### 按Z阈值分组对比

```python
# 对比不同Z阈值的Top-1策略
for z in [-0.5, 0.0, 0.5, 1.0, 1.5]:
    subset = df[df['z_threshold'] == z]
    if not subset.empty:
        top1 = subset.iloc[0]
        print(f"z={z}: 年化={top1['annual_return']:.2%}, Sharpe={top1['sharpe_ratio']:.3f}, 覆盖率={top1['coverage']:.1%}")
```

### 按因子数分组对比

```python
# 对比3/4/5因子的最优策略
for k in [3, 4, 5]:
    subset = df[df['n_factors'] == k]
    if not subset.empty:
        top1 = subset.iloc[0]
        print(f"{k}因子: 年化={top1['annual_return']:.2%}, Sharpe={top1['sharpe_ratio']:.3f}")
```

---

## 🔪 Linus式验证清单

### 运行完成后验证

1. **枚举完整性**
   ```bash
   cat enumeration_audit.json | jq '.theoretical_total_combos, .actual_enumerated, .hit_limit'
   ```
   - 确认 `actual_enumerated == theoretical_total_combos`
   - 确认 `hit_limit == false`

2. **过滤合理性**
   ```bash
   cat enumeration_audit.json | jq '.before_filter, .filtered_by_coverage, .filtered_by_turnover, .after_filter'
   ```
   - 计算过滤率：`(before_filter - after_filter) / before_filter`
   - 预期：30%-50%被过滤（合理）

3. **Top-5质量**
   ```bash
   head -6 top5_strategies.csv
   ```
   - 确认Sharpe > 0.6
   - 确认覆盖率 > 30%
   - 确认因子组合多样性

4. **4/5因子是否入选**
   ```bash
   cat strategies_ranked.csv | cut -d',' -f2 | sort | uniq -c
   ```
   - 确认有4因子和5因子策略
   - 对比3/4/5因子的最优表现

---

## 📋 预期结果

### 可能的发现

1. **Z阈值最优区间**
   - 预期：z=1.0-1.5表现最优
   - 验证：对比不同z的Top-1策略

2. **因子数最优配置**
   - 预期：3因子可能仍最优（简洁有效）
   - 验证：对比3/4/5因子的Sharpe分布

3. **覆盖率vs收益权衡**
   - 预期：高z阈值牺牲覆盖率但提升Sharpe
   - 验证：绘制覆盖率-Sharpe散点图

4. **换手率分布**
   - 预期：高z阈值略增换手率
   - 验证：对比不同z的平均换手率

---

## 🚀 后续优化方向

### 如果4/5因子表现更优

```yaml
# 聚焦4/5因子，加密z网格
phase2:
  min_factors: 4
  max_factors: 5
  signal_z_threshold_grid: [0.5, 0.75, 1.0, 1.25, 1.5]
```

### 如果z=1.0-1.5最优

```yaml
# 在最优区间加密
phase2:
  signal_z_threshold_grid: [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
  min_factors: 3
  max_factors: 3
```

### 如果需要更高覆盖率

```yaml
# 降低z阈值，增加TopN
phase2:
  signal_z_threshold_grid: [0.0, 0.5, 1.0]
  topn_grid: [6, 8]
  coverage_min: 0.6
```

---

**执行时间**: 2025-11-03 17:25  
**预期完成**: ~15-20分钟  
**状态**: ✅ **配置完成，等待结果**
