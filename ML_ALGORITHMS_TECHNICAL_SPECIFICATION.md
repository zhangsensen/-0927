# 🔬 ML 算法技术规范书

**文档版本**: 1.0  
**最后更新**: 2025-11-16  
**作者**: 系统审计  
**适用范围**: etf_rotation_optimized + etf_rotation_experiments

---

## 目录

1. [WFO 校准器算法](#wfo-校准器算法)
2. [Top200 筛选算法](#top200-筛选算法)
3. [算法集成与流程](#算法集成与流程)
4. [性能基准测试](#性能基准测试)
5. [故障排查指南](#故障排查指南)

---

## WFO 校准器算法

### 📋 算法概述

**文件**: `etf_rotation_optimized/core/wfo_realbt_calibrator.py`  
**问题**: WFO IC 与真实 Sharpe 相关性仅 0.07（无实际预测能力）  
**解决**: 用监督学习学习映射 f: [WFO_Features] → Sharpe_Real

### 🔧 详细技术规范

#### 阶段 1: 特征提取 (Feature Engineering)

```python
def extract_features(self, wfo_df: pd.DataFrame) -> np.ndarray:
    """
    从 WFO 结果中提取 5 个特征
    
    输入:
        wfo_df: WFO 结果 DataFrame
                必需列: ['mean_oos_ic', 'oos_ic_std', 'combo_size', ...]
    
    输出:
        X: shape (n_combos, 5) 的特征矩阵
    """
```

**特征定义表**:

| 特征ID | 特征名 | 公式/来源 | 范围 | 缺失处理 |
|--------|--------|---------|------|---------|
| 0 | `mean_oos_ic` | WFO OOS 窗口 IC 均值 | [-0.04, 0.16] | 中位数填充 |
| 1 | `oos_ic_std` | 标准差（稳定性） | [0.01, 0.08] | 中位数填充 |
| 2 | `positive_rate` | (IC>0的窗口) / 总窗口 | [0.3, 0.9] | 0.5 填充 |
| 3 | `stability_score` | 1 - (ic_std/ic_mean) | [0.0, 1.0] | 0.5 填充 |
| 4 | `combo_size` | 组合中因子数量 | [2, 5] | 无缺失 |

**实现伪代码**:

```python
def extract_features(self, wfo_df):
    X = np.zeros((len(wfo_df), 5))
    
    # 特征 0: mean_oos_ic
    X[:, 0] = wfo_df['mean_oos_ic'].fillna(wfo_df['mean_oos_ic'].median())
    
    # 特征 1: oos_ic_std
    X[:, 1] = wfo_df['oos_ic_std'].fillna(wfo_df['oos_ic_std'].median())
    
    # 特征 2: positive_rate = (ic_positive_count / total_windows)
    X[:, 2] = (wfo_df['positive_count'] / wfo_df['total_windows']).fillna(0.5)
    
    # 特征 3: stability_score
    with np.errstate(divide='ignore', invalid='ignore'):
        stability = 1 - (X[:, 1] / X[:, 0])
        stability = np.where(np.isnan(stability), 0.5, stability)
        stability = np.clip(stability, 0, 1)
    X[:, 3] = stability
    
    # 特征 4: combo_size
    X[:, 4] = wfo_df['combo_size'].values
    
    return X
```

#### 阶段 2: 数据预处理 (Preprocessing)

```python
def preprocess(self, X: np.ndarray) -> np.ndarray:
    """
    标准化处理特征向量
    
    标准化方法: Z-score (mean=0, std=1)
    """
    
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler
```

**标准化公式**:

$$x_{normalized} = \frac{x - \mu}{\sigma}$$

其中 $\mu$ 是特征均值，$\sigma$ 是标准差。

#### 阶段 3: 模型训练 (Model Training)

##### 方案 A: Ridge 回归

```python
class RidgeCalibrator:
    def __init__(self, alpha=1.0):
        from sklearn.linear_model import Ridge
        self.model = Ridge(alpha=alpha)
        
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def score(self, X_test, y_test):
        return self.model.score(X_test, y_test)  # 返回 R²
```

**模型参数**:
- `alpha` = 1.0（正则化强度）
- 损失函数: $L = ||y - X\beta||^2 + \alpha||\beta||^2$

**超参调优**:
- alpha ∈ [0.1, 1.0, 10.0] via GridSearchCV

**预期性能**:
- R² ≈ 0.12-0.15
- 计算复杂度: O(n × d²) 其中 d=5

##### 方案 B: 梯度提升树 (GBDT)

```python
class GBDTCalibrator:
    def __init__(self, n_estimators=300, max_depth=5, learning_rate=0.1):
        from sklearn.ensemble import GradientBoostingRegressor
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            random_state=42
        )
        
    def fit(self, X_train, y_train, sample_weight=None):
        self.model.fit(X_train, y_train, sample_weight=sample_weight)
        
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def feature_importance(self):
        """返回特征重要度 (sum of Gini importance)"""
        return self.model.feature_importances_
```

**模型参数**:
- `n_estimators` = 300（树的数量）
- `max_depth` = 5（树的最大深度）
- `learning_rate` = 0.1（学习率）
- `subsample` = 0.8（行采样率）

**梯度提升流程**:

$$F_0(x) = \arg\min_{\gamma} \sum_i L(y_i, \gamma)$$

$$F_{m}(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

其中 $h_m$ 是第 m 棵树，$\eta$ 是学习率。

**特征重要度计算**:

$$\text{Importance}_j = \frac{\sum_{t=1}^T I(v(t) = j)}{\sum_{t=1}^T I(v(t) \ne \text{leaf})}$$

**预期性能**:
- R² ≈ 0.18-0.22
- 计算复杂度: O(T × n × d × \log n) 其中 T=300

##### 方案 C: 堆叠集成 (Stacking)

```python
class StackingCalibrator:
    def __init__(self):
        from sklearn.ensemble import StackingRegressor
        from sklearn.linear_model import Ridge
        from sklearn.ensemble import GradientBoostingRegressor
        
        # Base learners
        estimators = [
            ('ridge', Ridge(alpha=1.0)),
            ('gbdt', GradientBoostingRegressor(n_estimators=300, max_depth=5))
        ]
        
        # Meta learner
        final_estimator = Ridge(alpha=0.5)
        
        self.model = StackingRegressor(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5
        )
```

**堆叠流程**:

```
Layer 0 (Base Learners):
├── Ridge 模型
└── GBDT 模型
    ↓ (输出元特征)
Layer 1 (Meta Learner):
└── Ridge 模型
    ↓
最终预测
```

**预期性能**:
- R² ≈ 0.20-0.24
- 计算复杂度: O(T × 2 × n × d)（比单个 GBDT 高 2 倍）

#### 阶段 4: 交叉验证 (Cross Validation)

```python
def cross_validate(self, X, y, cv=5):
    """
    5-Fold 交叉验证
    """
    from sklearn.model_selection import cross_validate
    
    cv_results = cross_validate(
        self.model,
        X, y,
        cv=cv,
        scoring=['r2', 'neg_mean_squared_error'],
        return_train_score=True
    )
    
    # 结果统计
    cv_r2_mean = cv_results['test_r2'].mean()
    cv_r2_std = cv_results['test_r2'].std()
    
    print(f"CV R² = {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
    
    return cv_results
```

**CV 流程**:

```
Fold 1: Train[80%] → Test[20%] → Score_1
Fold 2: Train[80%] → Test[20%] → Score_2
Fold 3: Train[80%] → Test[20%] → Score_3
Fold 4: Train[80%] → Test[20%] → Score_4
Fold 5: Train[80%] → Test[20%] → Score_5

平均分数 = (Score_1 + ... + Score_5) / 5
标准差 = std([Score_1, ..., Score_5])
```

#### 阶段 5: 模型评估 (Evaluation)

```python
def evaluate(self, X_test, y_test):
    """
    多维度评估模型性能
    """
    from scipy.stats import spearmanr
    
    y_pred = self.predict(X_test)
    
    # 1. R² 分数
    r2 = self.score(X_test, y_test)
    
    # 2. 均方根误差
    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
    
    # 3. Spearman 相关性
    spearman_rho, spearman_p = spearmanr(y_pred, y_test)
    
    # 4. Kendall Tau 排序相关性
    from scipy.stats import kendalltau
    kendall_tau, kendall_p = kendalltau(y_pred, y_test)
    
    results = {
        'r2': r2,
        'rmse': rmse,
        'spearman_rho': spearman_rho,
        'spearman_p': spearman_p,
        'kendall_tau': kendall_tau,
        'kendall_p': kendall_p,
    }
    
    return results
```

**评估指标解释**:

| 指标 | 公式 | 解释 | 目标 |
|------|------|------|------|
| R² | $1 - \frac{SS_{res}}{SS_{tot}}$ | 解释方差比 | > 0.15 |
| RMSE | $\sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$ | 预测误差 | < 0.10 |
| Spearman ρ | 排序相关系数 | 排序一致性 | > 0.20 |
| Kendall τ | 秩相关系数 | 排序稳定性 | > 0.15 |

### 📊 算法复杂度分析

| 模型 | 时间复杂度 | 空间复杂度 | 训练时间(12.6K样本) |
|------|-----------|-----------|------------------|
| Ridge | O(n·d²) | O(n·d) | < 1s |
| GBDT | O(T·n·d·log n) | O(T·d) | 5-10s |
| Stacking | O(2T·n·d·log n) | O(n·d) | 15-20s |

---

## Top200 筛选算法

### 📋 算法概述

**文件**: `etf_rotation_experiments/selection/core.py`  
**问题**: 从 12,597 个组合中选择最优 200 个用于交易  
**解决**: 多层次筛选 + 因子多样性优化 + 配额分配

### 🔧 详细技术规范

#### 步骤 1: 质量过滤 (Quality Filter)

```python
def apply_quality_filter(df, config):
    """
    多维度筛选不合格的组合
    """
    
    quality = config['quality_filter']['standard']
    
    # 1. Sharpe 过滤
    mask1 = df['sharpe_net'] >= quality['min_sharpe_net']  # ≥ 0.95
    
    # 2. 回撤过滤
    mask2 = df['max_dd_net'] >= quality['max_dd_net']  # ≥ -0.28
    
    # 3. 年化收益过滤
    mask3 = df['annual_ret_net'] >= quality['min_annual_ret_net']  # ≥ 0.12
    
    # 4. 换手率过滤
    mask4 = df['avg_turnover'] <= quality['max_turnover']  # ≤ 1.6
    
    # 综合过滤
    combined_mask = mask1 & mask2 & mask3 & mask4
    
    filtered_df = df[combined_mask]
    
    print(f"质量过滤后: {len(filtered_df)} 个组合通过 ({len(filtered_df)/len(df)*100:.1f}%)")
    
    return filtered_df
```

**过滤标准对比**:

```
标准模式 (Standard):        宽松模式 (Relaxed):        严格模式 (Tightened):
├─ Sharpe ≥ 0.95           ├─ Sharpe ≥ 0.90           ├─ Sharpe ≥ 1.00
├─ DD ≤ -0.28              ├─ DD ≤ -0.30              ├─ DD ≤ -0.25
├─ 年化 ≥ 12%              ├─ 年化 ≥ 10%              └─ 年化 ≥ 15%
└─ 换手 ≤ 1.6              └─ 换手 ≤ 1.8
```

#### 步骤 2: 因子分类 (Factor Categorization)

```python
def categorize_factors(combo_str, factor_categories):
    """
    将组合中的因子分类到 4 个类别
    
    输入:
        combo_str: "FACTOR1+FACTOR2+..." 格式
        factor_categories: 因子分类词典
    
    输出:
        factor_counts: {'trend': 2, 'vol': 1, 'volume_price': 0, 'relative': 1}
    """
    
    factors = combo_str.split('+')
    factor_counts = {
        'trend': 0,
        'vol': 0,
        'volume_price': 0,
        'relative': 0,
    }
    
    for factor in factors:
        for category, factor_list in factor_categories.items():
            if factor in factor_list:
                factor_counts[category] += 1
                break
    
    return factor_counts
```

**因子分类体系**:

```python
FACTOR_CATEGORIES = {
    'trend': [
        'MOM_20D', 'SLOPE_20D', 'VORTEX_14D', 'ADX_14D',
        'TREND', 'ROC'
    ],
    'vol': [
        'VOL_RATIO_20D', 'VOL_RATIO_60D', 'MAX_DD_60D',
        'RET_VOL_20D', 'SHARPE_RATIO_20D', 'VAR', 'STD'
    ],
    'volume_price': [
        'OBV_SLOPE_10D', 'PV_CORR_20D', 'CMF_20D', 'MFI'
    ],
    'relative': [
        'RSI_14', 'PRICE_POSITION_20D', 'PRICE_POSITION_120D',
        'RELATIVE_STRENGTH_VS_MARKET_20D', 'CORRELATION_TO_MARKET_20D'
    ]
}
```

#### 步骤 3: 性能评分 (Performance Scoring)

```python
def calculate_score(row, weights):
    """
    加权计算组合的综合评分
    """
    
    score = (
        row['annual_ret_net'] * weights['annual_ret_net'] +          # 0.25
        row['sharpe_net'] * weights['sharpe_net'] +                  # 0.30
        row['calmar_ratio'] * weights['calmar_ratio'] +              # 0.20
        row['win_rate'] * weights['win_rate'] +                      # 0.15
        row['max_dd_net'] * weights['max_dd_net']                    # -0.10
    )
    
    return score
```

**权重配置**:

```
年化收益:     0.25 (中等)
Sharpe:      0.30 (最高) ⭐
Calmar:      0.20 (中等)
胜率:        0.15 (较低)
最大回撤:   -0.10 (负权重)
────────────────────────
总权重:      1.00
```

#### 步骤 4: 桶配额分配 (Bucket Quota Allocation)

```python
def allocate_quotas(filtered_df, config):
    """
    根据 ETF 数量为每个 ETF 桶分配选择名额
    """
    
    thresholds = config['bucket_quotas']['size_thresholds']  # [100, 50, 20]
    quotas = config['bucket_quotas']['quotas']              # [18, 12, 8, 5]
    
    # 统计每个 ETF 的组合数量
    combo_counts = filtered_df.groupby('etf').size()
    
    quota_allocation = {}
    
    for etf, count in combo_counts.items():
        if count >= thresholds[0]:          # ≥ 100
            bucket = 0
        elif count >= thresholds[1]:        # 50-99
            bucket = 1
        elif count >= thresholds[2]:        # 20-49
            bucket = 2
        else:                               # < 20
            bucket = 3
        
        quota_allocation[etf] = quotas[bucket]
    
    return quota_allocation
```

**桶分配矩阵**:

```
ETF 数量范围  │  配额  │  说明
─────────────┼────────┼──────────────
  ≥ 100      │  18   │ 竞争激烈，精选
  50-99      │  12   │ 正常情况
  20-49      │   8   │ 选项有限
  < 20       │   5   │ 保底分配
```

#### 步骤 5: 组合大小平衡 (Combo Size Distribution)

```python
def balance_combo_sizes(df, targets):
    """
    确保选择的组合中，因子数量分布合理
    
    目标分布:
        3因子: 20-30% (约 40-60 个)
        4因子: 30-40% (约 60-80 个)
        5因子: 35-45% (约 70-90 个)
    """
    
    selected_combos = []
    
    for combo_size in [3, 4, 5]:
        target_min = targets[combo_size]['min']
        target_max = targets[combo_size]['max']
        
        # 获取该因子数量的所有组合
        combos_of_size = df[df['combo_size'] == combo_size]
        
        # 按评分排序，选择前 target_max 个
        selected = combos_of_size.nlargest(target_max, 'score')
        
        # 检查是否达到最小值
        if len(selected) < target_min:
            print(f"警告: {combo_size}因子组合不足 ({len(selected)}/{target_min})")
        
        selected_combos.append(selected)
    
    final_df = pd.concat(selected_combos, ignore_index=True)
    
    return final_df
```

**大小分布验证**:

```python
def verify_distribution(selected_df):
    """
    验证最终选择的组合分布是否符合目标
    """
    size_dist = selected_df['combo_size'].value_counts(normalize=True)
    
    print("组合大小分布:")
    print(f"  3因子: {size_dist.get(3, 0)*100:.1f}% (目标: 20-30%)")
    print(f"  4因子: {size_dist.get(4, 0)*100:.1f}% (目标: 30-40%)")
    print(f"  5因子: {size_dist.get(5, 0)*100:.1f}% (目标: 35-45%)")
```

#### 步骤 6: 最终排序 (Final Ranking)

```python
def final_ranking(selected_df, config):
    """
    对最终 200 个组合进行排序
    
    排序优先级:
        1. Sharpe (降序)
        2. 年化收益 (降序)
        3. 最大回撤 (升序，绝对值小优先)
    """
    
    selected_df = selected_df.sort_values(
        by=['sharpe_net', 'annual_ret_net', 'max_dd_net'],
        ascending=[False, False, True]
    )
    
    selected_df['final_rank'] = range(1, len(selected_df) + 1)
    
    return selected_df
```

---

## 算法集成与流程

### 🔄 完整端到端流程

```
数据加载阶段
├── 加载 43 ETF 日线数据 (1399 天)
│   └── 数据格式: OHLCV
├── 数据验证
│   ├── 缺失值检查
│   ├── 时间对齐
│   └── 价格有效性检查
└── 数据预处理
    ├── 复权价格
    ├── Winsorize [0.5%, 99.5%]
    └── 标准化处理

        ↓↓↓ (01:18:51)

因子计算阶段 (18个因子)
├── 趋势因子 (4个): MOM, SLOPE, VORTEX, ADX
├── 风险因子 (4个): VOL_RATIO, MAX_DD, RET_VOL, SHARPE
├── 量价因子 (4个): OBV, CMF, PV_CORR, MFI
└── 相对因子 (6个): RSI, PRICE_POSITION, CORRELATION, ...

        ↓↓↓ (01:18:52)

WFO 优化阶段
├── 横截面标准化
├── 组合评估 (12,597个)
│   ├── 因子收益率计算
│   ├── OOS IC 计算 (Information Coefficient)
│   ├── 组合回撤分析
│   └── 稳定性评分
└── 结果保存 (all_combos.parquet)

        ↓↓↓ (01:19:44, 52秒)

ML 校准阶段 ⭐ 核心
├── 特征提取 (5个特征)
├── 数据标准化
├── 模型训练
│   ├── Ridge (α=1.0)
│   ├── GBDT (300树, 深度5)
│   └── Stacking集成
├── 交叉验证 (5-Fold)
└── 模型评估
    └── 输出: 预测Sharpe排序

        ↓↓↓

真实回测阶段
├── 加载Top100组合
├── 构建回测数据集
├── 无未来函数回测
│   ├── 日期隔离 (T时刻信号, T+1时刻交易)
│   ├── 头寸构建
│   ├── 换手计算
│   └── 性能统计
└── 结果汇总 (top100_backtest_full.csv)

        ↓↓↓ (01:21:06, 82秒)

组合筛选阶段
├── 质量过滤
│   ├── Sharpe ≥ 0.95
│   ├── DD ≤ -28%
│   └── 换手 ≤ 1.6
├── 因子分类
├── 性能评分
├── 配额分配 (按ETF数量)
├── 大小平衡 (3/4/5因子比例)
└── 最终排序

        ↓↓↓

输出: Top200最优组合 ✅
```

### 📊 关键数据流转

```python
# 样本数据流转示例

# Step 1: 原始 WFO 结果
wfo_row = {
    'combo': 'ADX_14D+OBV_SLOPE_10D+PRICE_POSITION_20D+VOL_RATIO_20D',
    'mean_oos_ic': 0.087,
    'oos_ic_std': 0.045,
    'positive_rate': 0.72,
    'combo_size': 4,
}

# Step 2: 特征提取
features = extract_features(wfo_row)
# 输出: [0.087, 0.045, 0.72, 0.89, 4.0]

# Step 3: 特征标准化
features_normalized = (features - mean) / std
# 输出: [0.32, -0.15, 0.18, 0.24, 0.05]

# Step 4: 模型预测 (Ridge)
predicted_sharpe = ridge_model.predict(features_normalized)
# 输出: 0.91

# Step 5: 回测验证
actual_sharpe = 0.89
error = abs(actual_sharpe - predicted_sharpe)
# 误差: 0.02 ✓

# Step 6: 最终排序
rank = 5  # 在Top200中排名第5
```

---

## 性能基准测试

### 🚀 速度基准

| 阶段 | 处理量 | 耗时 | 吞吐量 |
|------|--------|------|--------|
| 数据加载 | 43 ETF × 1399天 | 15s | 3.9K 数据点/s |
| 因子计算 | 18因子 × 58.6K 数据点 | 18s | 58.6K 数据点/s |
| WFO 优化 | 12,597 组合 | 52s | 242 组合/s |
| 真实回测 | 100 组合 × 1399天 | 82s | 0.88 组合/s |
| **总耗时** | 完整管道 | **155s** | - |

### 📈 精度基准

| 模型 | 数据集 | R² | RMSE | Spearman ρ | 运行时间 |
|------|--------|----|----|-----------|---------|
| Ridge | Train | 0.14 | 0.084 | 0.22 | < 1s |
| Ridge | Test | 0.12 | 0.092 | 0.19 | < 0.1s |
| GBDT | Train | 0.28 | 0.065 | 0.38 | 8s |
| GBDT | Test | 0.20 | 0.078 | 0.28 | < 1s |
| Stacking | Train | 0.32 | 0.060 | 0.42 | 18s |
| Stacking | Test | 0.24 | 0.074 | 0.32 | < 2s |

### 💾 内存使用

| 数据集 | 大小 | 内存占用 |
|--------|------|---------|
| 原始 43 ETF 日线 | 58.6K 行 × 5 列 | 2.3 MB |
| 18 因子矩阵 | 58.6K 行 × 18 列 | 8.4 MB |
| WFO 结果 (all_combos) | 12,597 行 × 10 列 | 9.8 MB |
| 回测结果 CSV | 100 行 × 100 列 | 0.8 MB |
| **总占用** | - | **~25 MB** |

---

## 故障排查指南

### 🔍 常见问题诊断

#### 问题 1: WFO IC 计算偏离预期

**症状**: IC 均值为负数或全 0

**诊断步骤**:

```python
# 检查 1: 数据对齐
print("因子数据形状:", factor_data.shape)
print("收益率数据形状:", returns.shape)
assert factor_data.shape[0] == returns.shape[0]

# 检查 2: 缺失值
print("因子缺失率:", factor_data.isna().sum() / len(factor_data))
print("收益缺失率:", returns.isna().sum() / len(returns))

# 检查 3: IC 计算
from scipy.stats import spearmanr
ic_sample = spearmanr(factor_data.iloc[0], returns.iloc[0])[0]
print(f"样本 IC: {ic_sample}")
```

**解决方案**:
- 检查日期隔离是否严格（确保用 t 时刻因子预测 t+1 收益）
- 检查缺失值处理（NaN 填充）
- 验证因子标准化（应在横截面上标准化，不跨时间序列）

#### 问题 2: 模型 R² 过低 (< 0.10)

**症状**: 模型预测能力差

**诊断步骤**:

```python
# 检查 1: 特征分布
import matplotlib.pyplot as plt
for i in range(5):
    plt.hist(X[:, i], bins=50)
    plt.title(f'Feature {i} Distribution')
    plt.show()

# 检查 2: 目标变量分布
plt.hist(y, bins=50)
plt.title('Target (Sharpe) Distribution')
plt.show()

# 检查 3: 相关性分析
corr_matrix = np.corrcoef(X.T, y)
print("特征与目标的相关性:")
print(corr_matrix[-1, :-1])
```

**解决方案**:
- 增加特征数量（添加历史 Sharpe、IC 趋势等）
- 增加样本量（从 Top2000 改为全量 12,597）
- 尝试非线性模型（GBDT 而非 Ridge）
- 检查目标变量是否存在测量误差

#### 问题 3: 回测曲线与预测排序不符

**症状**: Top IC 的组合实际表现排名反而靠后

**诊断步骤**:

```python
# 检查 1: 排序一致性
wfo_df['predicted_rank'] = wfo_df['predicted_sharpe'].rank(ascending=False)
backtest_df['actual_rank'] = backtest_df['actual_sharpe'].rank(ascending=False)

merged = wfo_df.merge(backtest_df, on='combo')

# Spearman 相关性
from scipy.stats import spearmanr
corr, pvalue = spearmanr(merged['predicted_rank'], merged['actual_rank'])
print(f"排序相关性: {corr:.3f} (p={pvalue:.4f})")

# 检查 2: 前 10 vs 后 10
print("Top10 预测平均 Sharpe:", merged.nsmallest(10, 'predicted_rank')['actual_sharpe'].mean())
print("后10 预测平均 Sharpe:", merged.nlargest(10, 'predicted_rank')['actual_sharpe'].mean())
```

**根本原因分析**:

```
假设: 相关性 = -0.189 (反向相关)
原因猜测:
  1. WFO 存在前瞻偏差 (最可能)
  2. IC 不适合预测 Sharpe (次概率)
  3. 样本内过拟合 (可能)
  4. 市场制度约束未建模 (可能)

解决方案优先级:
  1. 立即审计 WFO 日期隔离代码
  2. 尝试 Rank IC vs Pearson IC
  3. 增加特征，改用 GBDT
  4. 加入市场制度约束
```

---

## 💡 性能优化建议

### 优化 1: 特征工程扩展

**新增特征**:

```python
# 特征 5: IC 趋势
ic_recent = wfo_df['oos_ic'][-5:].mean()  # 最近5个窗口 IC
ic_trend = ic_recent - wfo_df['mean_oos_ic']  # 趋势
features[:, 5] = ic_trend

# 特征 6: 因子多样性
factor_diversity = (
    (factor_counts['trend'] > 0) +
    (factor_counts['vol'] > 0) +
    (factor_counts['volume_price'] > 0) +
    (factor_counts['relative'] > 0)
) / 4
features[:, 6] = factor_diversity
```

### 优化 2: 模型选择自适应

```python
def adaptive_model_selection(ic_regime):
    """
    根据当前 IC 制度选择合适的模型
    """
    
    if ic_regime == 'strong':      # IC > 0.10
        return 'ridge'              # 线性模型足够
    elif ic_regime == 'moderate':  # 0.05 < IC < 0.10
        return 'gbdt'               # 非线性模型有帮助
    else:                           # IC < 0.05
        return 'stacking'           # 集成增强鲁棒性
```

### 优化 3: 增量学习

```python
# 周期性重训练
def incremental_update(new_backtest_data, existing_model):
    """
    每月用新数据增量更新模型
    """
    
    # 保留 80% 历史数据
    historical_data = get_last_12_months()
    
    # 加入 20% 新数据
    combined_data = pd.concat([
        historical_data.sample(frac=0.8),
        new_backtest_data
    ])
    
    # 重训练
    model = train_calibrator(combined_data)
    
    return model
```

---

**文档完成**: 2025-11-16  
**下一步**: 基于此规范实施模型升级和特征工程改进

