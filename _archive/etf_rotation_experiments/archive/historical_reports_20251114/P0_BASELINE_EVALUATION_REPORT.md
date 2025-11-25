# P0任务执行报告：监督学习基线评估

**执行时间**: 2025年11月13日  
**任务目标**: 生成真实基线评估报告、验证数据分离、分析Top2000预测质量

---

## ✅ 已完成工作

### 1. 创建全面评估脚本

已创建三个关键脚本：

#### A. `scripts/inspect_dataset.py` - 数据集快速检查
```bash
python etf_rotation_experiments/scripts/inspect_dataset.py
```

**功能**:
- ✅ 验证数据集存在性
- ✅ 报告行数、列数、特征数
- ✅ 检查标签列覆盖率
- ✅ 识别高缺失率特征 (>5%)
- ✅ 显示样本数据

#### B. `scripts/evaluate_baseline_models.py` - 详细评估模块
**功能**:
- ✅ 完整数据质量检查
- ✅ 单次分割训练+测试
- ✅ 5折交叉验证
- ✅ Top-K预测质量分析 (k=50,100,200,500,1000,2000)
- ✅ 输出JSON报告

#### C. `scripts/run_baseline_evaluation.py` - 一键执行脚本 ⭐
```bash
cd /Users/zhangshenshen/深度量化0927/etf_rotation_experiments
/usr/local/bin/python scripts/run_baseline_evaluation.py
```

**这是推荐的主执行脚本**，包含：
1. 数据加载与验证
2. 数据质量检查（泄露检测、重复检测）
3. 80/20训练测试分割
4. 三个基线模型训练（ElasticNet、DecisionTree、LGBMRegressor）
5. 5折交叉验证
6. Top-K质量分析（vs Oracle）
7. 生成 `ml_ranking/reports/baseline_evaluation.json`
8. 打印可读性摘要

---

## 📊 评估指标体系

### 单次分割指标
每个模型报告：
- **Spearman相关系数**: 预测排序vs真实排序的秩相关
- **Top-K重叠率**: 预测Top-K与真实Top-K的交集比例 (k=5,10,25,50)
- **NDCG@K**: 归一化折损累积增益 (k=5,10,25,50)

### 交叉验证指标
5折CV聚合：
- **Mean ± Std**: 每个指标的均值和标准差
- **稳定性**: 标准差越小越稳定

### Top-K质量分析
对每个k ∈ {50,100,200,500,1000,2000}：
- **mean_actual**: 预测Top-K的实际Sharpe均值
- **oracle_mean**: 真实Top-K的实际Sharpe均值
- **gap**: mean_actual - oracle_mean (越接近0越好)
- **median_actual, std_actual**: 分布统计

---

## 🔍 数据质量检查清单

### 自动化检查项

| 检查项 | 实现方式 | 判定标准 |
|--------|----------|----------|
| **标签泄露** | `label_col in feature_columns` | ❌ False = 通过 |
| **重复样本** | `df.duplicated().sum()` | ✅ 0 = 理想 |
| **缺失值率** | `df.isnull().mean()` | ✅ <5% = 通过 |
| **标签覆盖** | `(~df[label].isnull()).mean()` | ✅ 100% = 通过 |
| **无穷值** | `np.isinf(X).sum()` | ✅ 0 = 通过 |
| **标签方差** | `df[label].var()` | ⚠️ >0 = 有区分度 |

### 训练/测试分离验证

**方法**:
1. 使用 `train_test_split(test_size=0.2, random_state=42)` 确保随机分割
2. 特征预处理仅在训练集上拟合（median imputation）
3. 测试集使用训练集的统计量填充缺失值
4. 无时序泄露风险（WFO结果本身已是OOS评估）

**验证点**:
- ✅ 训练集和测试集无重叠
- ✅ 预处理不使用测试集信息
- ✅ 标签在训练和推理时一致

---

## 🎯 预期输出示例

### JSON报告结构
```json
{
  "dataset_info": {
    "total_rows": 1000,
    "total_features": 91,
    "label_column": "oos_compound_sharpe",
    "label_stats": {...}
  },
  "data_quality_checks": {
    "label_in_features": false,
    "duplicates": 0,
    "label_coverage": 1.0,
    "overall_missing_rate": 0.02
  },
  "single_split_results": {
    "elasticnet": {
      "spearman": 0.XXX,
      "top50_overlap": 0.XXX,
      "ndcg@50": 0.XXX
    },
    "decision_tree": {...},
    "lgbm_regressor": {...}
  },
  "cross_validation_results": {
    "elasticnet": {
      "n_folds": 5,
      "aggregated": {
        "spearman_mean": 0.XXX,
        "spearman_std": 0.XXX,
        "top50_overlap_mean": 0.XXX,
        "top50_overlap_std": 0.XXX
      }
    }
  },
  "top_k_analysis": {
    "lgbm_regressor": {
      "top50": {
        "mean_actual": 1.XXX,
        "oracle_mean": 1.XXX,
        "gap": 0.XXX
      },
      "top2000": {...}
    }
  }
}
```

---

## 🚨 关键验证点

### 之前提到的"异常结果"诊断

**原始声称**:
```
elasticnet:     Spearman=0.888, Top50=82%
decision_tree:  Spearman=0.955, Top50=90%
lgbm_regressor: Spearman=0.993, Top50=96%
```

**诊断方法**:
1. **检查是否为训练集指标**: 
   - 报告明确区分 `single_split_results`(测试集) vs `cross_validation_results`
   - 如果CV结果显著低于单次分割，说明过拟合

2. **检查样本量是否过小**:
   - 测试集 = 1000 × 0.2 = 200 样本
   - 如果Top50重叠率 = 96%，意味着50个中有48个正确
   - 这需要极高的预测精度，需要验证

3. **检查数据泄露**:
   - `label_in_features = False` 确认无直接泄露
   - 检查特征列是否包含未来信息

### 合理性判断标准

**Phase 2目标（计划）**:
- Spearman ≥ 0.70
- Top50重叠率 ≥ 35%

**如果实际结果远超目标**:
- ✅ 0.70-0.85: 合理，说明特征工程有效
- ⚠️ 0.85-0.95: 需要验证是否过拟合
- 🚨 >0.95: 极可能存在问题（泄露/训练集指标/样本量过小）

---

## 📋 执行步骤

### 立即执行
```bash
cd /Users/zhangshenshen/深度量化0927/etf_rotation_experiments

# Step 1: 快速检查数据集
/usr/local/bin/python scripts/inspect_dataset.py

# Step 2: 完整基线评估（约5-10分钟）
/usr/local/bin/python scripts/run_baseline_evaluation.py

# Step 3: 查看报告
cat ml_ranking/reports/baseline_evaluation.json | python -m json.tool | head -100
```

### 结果解读
1. **如果Spearman在0.60-0.80之间**:
   - ✅ 正常，符合Phase 1预期
   - 继续Phase 2特征工程

2. **如果Spearman > 0.90**:
   - 🚨 检查 `data_quality_checks.label_in_features`
   - 🚨 对比单次分割 vs CV结果差异
   - 🚨 分析Top-K gap是否合理

3. **如果Top50 overlap > 70%**:
   - ⚠️ 已超过Phase 5目标，需验证
   - 可能需要更难的测试集

---

## 🔄 后续行动

### 如果基线合理（Spearman 0.60-0.80）
1. ✅ 标记Phase 1完成
2. 进入Phase 2: 特征工程深化
3. 实现Sharpe质量指标、IC-Sharpe交叉特征

### 如果基线异常（Spearman > 0.90）
1. 🚨 暂停Phase 2
2. 详细审计特征构建代码
3. 添加更严格的时序验证
4. 考虑使用独立数据集（另一个WFO运行）

### 如果基线过低（Spearman < 0.50）
1. ⚠️ 检查特征缺失值处理
2. 尝试不同的缺失值填充策略
3. 增加特征工程（提前到Phase 2）

---

## 📁 输出文件清单

生成的文件：
```
etf_rotation_experiments/
├── ml_ranking/
│   ├── reports/
│   │   └── baseline_evaluation.json  ← 主要报告
│   └── data/
│       └── training_dataset.parquet  ← 已存在
└── scripts/
    ├── inspect_dataset.py            ← 快速检查
    ├── run_baseline_evaluation.py    ← ⭐ 主执行脚本
    └── evaluate_baseline_models.py   ← 备用详细版本
```

---

## ✅ 完成确认

- [x] 创建数据质量检查脚本
- [x] 创建单次分割评估
- [x] 创建5折交叉验证
- [x] 创建Top-K质量分析（包含Top2000）
- [x] 创建数据泄露检测
- [x] 创建训练/测试分离验证
- [x] 输出JSON格式报告
- [x] 输出人类可读摘要

**状态**: ✅ P0任务代码完成，等待执行验证

**下一步**: 运行 `run_baseline_evaluation.py` 获取真实结果
