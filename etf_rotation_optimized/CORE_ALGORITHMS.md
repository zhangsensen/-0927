# ETF轮动系统 - 核心算法详解

**版本**: v2.0-optimized  
**最后更新**: 2025-10-27  
**目标**: 深度理解系统的核心算法实现

---

## 📋 算法目录

1. [IC计算算法](#ic计算算法)
2. [标准化算法](#标准化算法)
3. [极值截断算法](#极值截断算法)
4. [FDR校正算法](#fdr校正算法)
5. [WFO窗口划分](#wfo窗口划分)
6. [因子选择算法](#因子选择算法)

---

## IC计算算法

### 🧮 算法原理

IC (Information Coefficient) 衡量因子与未来收益的相关性。

**公式**:
```
IC_t = Corr(Factor_t, Return_{t+1})
```

其中:
- `Factor_t`: 第t天的因子值 (标准化后)
- `Return_{t+1}`: 第t+1天的收益率
- `Corr`: 相关系数 (Pearson/Spearman/Kendall)

### 🔧 实现细节

```python
def compute_ic(factors, returns, method='pearson'):
    """
    计算IC时间序列
    
    Args:
        factors: 标准化因子 (N, M) - N个交易日，M个标的
        returns: 收益率 (N, M)
        method: 相关系数方法
    
    Returns:
        ic_series: IC时间序列 (N,)
    
    处理规则:
    1. 逐日计算相关系数
    2. 最少需要20个有效观察
    3. 忽略NaN值
    4. 返回NaN表示计算失败
    """
    ic_values = []
    
    for t in range(len(factors)):
        factor_t = factors[t]      # (M,)
        return_t = returns[t]      # (M,)
        
        # 获取有效数据
        valid_idx = ~(np.isnan(factor_t) | np.isnan(return_t))
        n_valid = valid_idx.sum()
        
        if n_valid >= IC_MIN_OBSERVATIONS:
            # 计算相关系数
            if method == 'pearson':
                ic = np.corrcoef(
                    factor_t[valid_idx], 
                    return_t[valid_idx]
                )[0, 1]
            elif method == 'spearman':
                ic = stats.spearmanr(
                    factor_t[valid_idx], 
                    return_t[valid_idx]
                )[0]
            elif method == 'kendall':
                ic = stats.kendalltau(
                    factor_t[valid_idx], 
                    return_t[valid_idx]
                )[0]
            ic_values.append(ic)
        else:
            ic_values.append(np.nan)
    
    return np.array(ic_values)
```

### 📊 IC统计量计算

```python
def compute_ic_stats(ic_series):
    """
    计算IC统计量
    
    Returns:
        ICStats: 包含以下指标
        - mean: 平均IC
        - std: IC标准差
        - ir: IC比 (mean / std)
        - t_stat: t统计量
        - p_value: 显著性p值
        - sharpe: 年化Sharpe比
    """
    valid_ic = ic_series[~np.isnan(ic_series)]
    n = len(valid_ic)
    
    mean = np.mean(valid_ic)
    std = np.std(valid_ic)
    ir = mean / (std + EPSILON)
    
    # t-test
    t_stat = mean / (std / np.sqrt(n) + EPSILON)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))
    
    # 年化Sharpe
    sharpe = mean * np.sqrt(TRADING_DAYS_PER_YEAR) / (std + EPSILON)
    
    return ICStats(
        mean=mean,
        std=std,
        ir=ir,
        t_stat=t_stat,
        p_value=p_value,
        sharpe=sharpe,
        n_obs=n,
        min=np.min(valid_ic),
        max=np.max(valid_ic),
        median=np.median(valid_ic),
        skew=stats.skew(valid_ic),
        kurtosis=stats.kurtosis(valid_ic)
    )
```

---

## 标准化算法

### 🧮 Z-score标准化

**公式**:
```
Z_i = (X_i - μ) / σ
```

其中:
- `X_i`: 原始因子值
- `μ`: 因子均值
- `σ`: 因子标准差

### 🔧 实现细节

```python
def standardize_cross_section(factors):
    """
    横截面标准化 (Z-score)
    
    Args:
        factors: 因子数据 (N, M) - N个交易日，M个标的
    
    Returns:
        standardized: 标准化后的因子 (N, M)
    
    处理规则:
    1. 每个交易日单独处理
    2. 只使用有效数据计算均值和标准差
    3. 如果std=0，则设为0
    4. 保留NaN值
    """
    standardized = np.zeros_like(factors, dtype=float)
    
    for t in range(len(factors)):
        factor_t = factors[t]  # (M,)
        
        # 获取有效数据
        valid_idx = ~np.isnan(factor_t)
        valid_data = factor_t[valid_idx]
        
        if len(valid_data) > 0:
            mean = np.mean(valid_data)
            std = np.std(valid_data)
            
            # 标准化
            if std > EPSILON:
                standardized[t][valid_idx] = (valid_data - mean) / std
                standardized[t][~valid_idx] = np.nan
            else:
                # std=0时，所有有效值标准化为0
                standardized[t][valid_idx] = 0.0
                standardized[t][~valid_idx] = np.nan
        else:
            standardized[t] = np.nan
    
    return standardized
```

### 💡 关键设计

- **逐日处理**: 消除时间序列偏差
- **有效数据**: 忽略NaN值
- **零标准差处理**: std=0时设为0，避免除零
- **NaN保留**: 保持缺失值信息

---

## 极值截断算法

### 🧮 Winsorize方法

**原理**: 将极端值替换为分位数

**公式**:
```
X'_i = clip(X_i, Q_lower, Q_upper)
```

其中:
- `Q_lower`: 下界分位数 (2.5%)
- `Q_upper`: 上界分位数 (97.5%)

### 🔧 实现细节

```python
def winsorize_factors(factors, lower_pct=2.5, upper_pct=97.5):
    """
    极值截断 (Winsorize)
    
    Args:
        factors: 因子数据 (N, M)
        lower_pct: 下界百分位 (default: 2.5%)
        upper_pct: 上界百分位 (default: 97.5%)
    
    Returns:
        winsorized: 截断后的因子 (N, M)
    
    处理规则:
    1. 每个交易日单独处理
    2. 计算有效数据的分位数
    3. 有界因子跳过截断
    4. 保留NaN值
    """
    winsorized = np.zeros_like(factors, dtype=float)
    
    for t in range(len(factors)):
        factor_t = factors[t]
        
        # 获取有效数据
        valid_idx = ~np.isnan(factor_t)
        valid_data = factor_t[valid_idx]
        
        if len(valid_data) > 0:
            # 计算分位数
            lower_bound = np.percentile(valid_data, lower_pct)
            upper_bound = np.percentile(valid_data, upper_pct)
            
            # 截断
            clipped = np.clip(valid_data, lower_bound, upper_bound)
            winsorized[t][valid_idx] = clipped
            winsorized[t][~valid_idx] = np.nan
        else:
            winsorized[t] = np.nan
    
    return winsorized
```

### 💡 关键设计

- **有界因子跳过**: BOUNDED_FACTORS中的因子不截断
- **逐日处理**: 消除时间序列偏差
- **分位数计算**: 使用有效数据计算
- **NaN保留**: 保持缺失值信息

---

## FDR校正算法

### 🧮 Benjamini-Hochberg方法

**目标**: 控制假发现率 (False Discovery Rate)

**步骤**:

```
1. 计算所有因子的p-value
2. 按p-value升序排列
3. 计算调整后的p-value
4. 选择满足条件的因子
```

### 🔧 实现细节

```python
def benjamini_hochberg_fdr(p_values, alpha=0.1):
    """
    Benjamini-Hochberg FDR校正
    
    Args:
        p_values: 原始p-value数组
        alpha: FDR阈值 (default: 0.1)
    
    Returns:
        rejected: 布尔数组，True表示拒绝零假设
    
    算法:
    1. 按p-value升序排列
    2. 计算调整阈值: alpha * i / m
    3. 找到最大的i满足 p_value[i] <= 阈值
    4. 拒绝所有p-value <= p_value[i]的因子
    """
    m = len(p_values)
    
    # 获取排序索引
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    
    # 计算调整阈值
    thresholds = alpha * np.arange(1, m + 1) / m
    
    # 找到最大的i满足条件
    valid_idx = sorted_p <= thresholds
    if np.any(valid_idx):
        max_i = np.where(valid_idx)[0][-1]
        threshold = sorted_p[max_i]
    else:
        threshold = -1  # 没有因子通过
    
    # 创建拒绝数组
    rejected = np.zeros(m, dtype=bool)
    rejected[sorted_idx[:max_i + 1]] = True if threshold >= 0 else False
    
    return rejected
```

### 📊 示例

```
原始p-value: [0.001, 0.008, 0.039, 0.041, 0.042]
排序后:     [0.001, 0.008, 0.039, 0.041, 0.042]
阈值:       [0.02, 0.04, 0.06, 0.08, 0.10]

比较:
  0.001 <= 0.02 ✓
  0.008 <= 0.04 ✓
  0.039 <= 0.06 ✓
  0.041 <= 0.08 ✓
  0.042 <= 0.10 ✓

最大i = 4，所有因子通过
```

---

## WFO窗口划分

### 🧮 滑动窗口算法

**参数**:
- `IS_WINDOW`: 252天 (样本内)
- `OOS_WINDOW`: 60天 (样本外)
- `STEP`: 20天 (步进)

### 🔧 实现细节

```python
def create_wfo_windows(n_days, is_window=252, oos_window=60, step=20):
    """
    创建WFO窗口
    
    Args:
        n_days: 总交易日数
        is_window: 样本内窗口大小
        oos_window: 样本外窗口大小
        step: 步进大小
    
    Returns:
        windows: WFOWindow列表
    
    算法:
    1. 从t=0开始
    2. IS: [t, t+is_window)
    3. OOS: [t+is_window, t+is_window+oos_window)
    4. 步进: t += step
    5. 重复直到OOS结束超过总天数
    """
    windows = []
    window_id = 0
    t = 0
    
    while t + is_window + oos_window <= n_days:
        is_start = t
        is_end = t + is_window
        oos_start = is_end
        oos_end = oos_start + oos_window
        
        window = WFOWindow(
            window_id=window_id,
            is_start=is_start,
            is_end=is_end,
            oos_start=oos_start,
            oos_end=oos_end,
            selected_factors=[],
            is_ic_stats={},
            oos_ic_stats={},
            oos_performance={}
        )
        windows.append(window)
        
        t += step
        window_id += 1
    
    return windows
```

### 📊 示例

```
总天数: 500天
IS_WINDOW: 252天
OOS_WINDOW: 60天
STEP: 20天

Window 0: IS[0:252]   OOS[252:312]
Window 1: IS[20:272]  OOS[272:332]
Window 2: IS[40:292]  OOS[292:352]
...
Window N: IS[...] OOS[...]
```

---

## 因子选择算法

### 🧮 多阶段筛选

**阶段1**: IC/IR筛选

```python
def select_by_ic_ir(ic_stats, min_ic=0.01, min_ir=0.05):
    """
    基于IC/IR筛选因子
    
    规则:
    - IC > min_ic
    - IR > min_ir
    """
    selected = []
    for factor, stats in ic_stats.items():
        if stats.mean > min_ic and stats.ir > min_ir:
            selected.append(factor)
    return selected
```

**阶段2**: 显著性检验

```python
def apply_significance_test(ic_stats, alpha=0.05):
    """
    t-test显著性检验
    
    规则:
    - p-value < alpha
    """
    selected = []
    for factor, stats in ic_stats.items():
        if stats.p_value < alpha:
            selected.append(factor)
    return selected
```

**阶段3**: FDR校正

```python
def apply_fdr_correction(ic_stats, alpha=0.1):
    """
    Benjamini-Hochberg FDR校正
    """
    p_values = np.array([stats.p_value for stats in ic_stats.values()])
    rejected = benjamini_hochberg_fdr(p_values, alpha)
    
    selected = []
    for (factor, stats), is_rejected in zip(ic_stats.items(), rejected):
        if is_rejected:
            selected.append(factor)
    return selected
```

**阶段4**: 相关性过滤

```python
def filter_by_correlation(factors, factor_corr, max_corr=0.7):
    """
    相关性过滤
    
    规则:
    - 因子间相关系数 < max_corr
    - 保留IC最高的因子
    """
    selected = []
    remaining = set(factors)
    
    # 按IC排序
    factors_by_ic = sorted(
        factors, 
        key=lambda f: ic_stats[f].mean, 
        reverse=True
    )
    
    for factor in factors_by_ic:
        if factor not in remaining:
            continue
        
        selected.append(factor)
        remaining.remove(factor)
        
        # 移除高相关的因子
        for other in list(remaining):
            if abs(factor_corr[factor][other]) > max_corr:
                remaining.remove(other)
    
    return selected
```

---

## 性能优化

### ⚡ 向量化计算

**原则**: 避免Python循环，使用NumPy向量操作

**示例**:

```python
# ❌ 慢: Python循环
result = []
for i in range(len(data)):
    result.append(data[i] * 2)

# ✅ 快: NumPy向量化
result = data * 2
```

### 💾 缓存策略

**缓存层次**:

```
1. 内存缓存: 当前会话的计算结果
2. 磁盘缓存: pickle格式的因子数据
3. 数据库缓存: (可选) 历史数据
```

### 🔄 并行处理

**并行化**:

```python
from joblib import Parallel, delayed

# 并行计算因子
factors = Parallel(n_jobs=8)(
    delayed(compute_factor)(symbol, prices)
    for symbol in symbols
)
```

---

## 关键参数参考

| 参数 | 值 | 说明 |
|------|-----|------|
| EPSILON | 1e-10 | 除零保护 |
| DEFAULT_COVERAGE_THRESHOLD | 0.97 | 数据覆盖率 |
| IC_MIN_OBSERVATIONS | 20 | IC计算最小观察数 |
| WINSORIZE_LOWER_PCT | 2.5 | 下界百分位 |
| WINSORIZE_UPPER_PCT | 97.5 | 上界百分位 |
| DEFAULT_IS_WINDOW | 252 | 样本内窗口 |
| DEFAULT_OOS_WINDOW | 60 | 样本外窗口 |
| DEFAULT_STEP | 20 | WFO步进 |
| DEFAULT_IC_THRESHOLD | 0.05 | IC筛选阈值 |

---

**更新日期**: 2025-10-27  
**维护者**: ETF Rotation System Team
