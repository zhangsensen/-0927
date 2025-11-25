# ML Ranker 统一训练Pipeline指南

## 📋 文档概览

本文档是ML Ranker排序训练系统的完整使用指南,包含:
- 系统架构和模块边界
- 快速上手教程
- 多换仓周期数据接入SOP
- 配置文件详解
- 常见问题FAQ

**目标读者**: 量化策略研究员、ML工程师

---

## 🏗️ 一、系统架构概览

### 1.1 整体架构

ML Ranker排序训练系统与WFO主流程的关系:

```
┌─────────────────────────────────────────────────────────────────┐
│                       ETF轮动策略系统                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐          ┌──────────────────┐            │
│  │   WFO主流程       │          │  ML Ranker模块    │            │
│  │                  │          │                  │            │
│  │  • 策略参数搜索   │─────────▶│  • 排序模型训练   │            │
│  │  • OOS窗口评估   │  输出     │  • 稳健性验证     │            │
│  │  • IC/Sharpe计算 │ all_combos│  • 模型应用      │            │
│  └──────────────────┘  .parquet└──────────────────┘            │
│          │                              │                       │
│          │                              │                       │
│          ▼                              ▼                       │
│  ┌──────────────────┐          ┌──────────────────┐            │
│  │  真实回测         │          │ 新WFO结果排序     │            │
│  │                  │          │                  │            │
│  │  • 滑点2bps      │          │  • 预测分数      │            │
│  │  • 实际收益      │          │  • Top-K选择     │            │
│  │  • Sharpe/DD     │          │  • 策略组合      │            │
│  └──────────────────┘          └──────────────────┘            │
│          │                                                      │
│          │                                                      │
│          ▼                                                      │
│  ┌──────────────────────────────────────────┐                  │
│  │         训练集构建                        │                  │
│  │  WFO特征 + 真实回测标签 → 训练样本        │                  │
│  └──────────────────────────────────────────┘                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 数据流

```
[WFO实验] ──→ all_combos.parquet ──┐
                                   ├──→ [特征工程] ──→ [模型训练] ──→ [保存模型]
[真实回测] ──→ profit_backtest.csv ─┘
                                      ↓                  ↓
                               [合并标签]         [评估报告]
                                                        ↓
                                                 [稳健性验证]
```

### 1.3 模块边界

| 模块 | 职责 | 输入 | 输出 |
|------|------|------|------|
| **WFO主流程** | 策略参数搜索和OOS评估 | 策略参数空间 | all_combos.parquet |
| **真实回测** | 完整回测验证WFO结果 | WFO输出+历史数据 | profit_backtest.csv |
| **ML Ranker** | 学习WFO→真实表现映射 | WFO特征+真实标签 | 排序模型+评估报告 |
| **模型应用** | 对新WFO结果排序 | 新WFO+训练好的模型 | Top-K策略列表 |

### 1.4 文件组织

```
etf_rotation_experiments/
├── ml_ranker/                    # 排序模型模块
│   ├── config.py                 # [NEW] 配置类
│   ├── pipeline.py               # [NEW] 统一训练Pipeline
│   ├── data_loader.py            # 数据加载(已扩展)
│   ├── feature_engineer.py       # 特征工程
│   ├── ltr_model.py              # LightGBM模型
│   ├── evaluator.py              # 评估指标
│   ├── robustness_eval.py        # 稳健性验证
│   ├── models/                   # 训练好的模型
│   │   ├── ltr_ranker.txt
│   │   ├── ltr_ranker_meta.pkl
│   │   └── ltr_ranker_features.json
│   └── evaluation/               # 评估报告
│       ├── evaluation_report.json
│       ├── robustness_report.json
│       └── ranking_comparison_top100.csv
├── configs/                      # [NEW] 配置文件
│   └── ranking_datasets.yaml     # 数据源配置
├── run_ranking_pipeline.py       # [NEW] 统一训练入口
├── train_ranker.py               # 单数据源训练(保留)
├── apply_ranker.py               # 模型应用
├── results/                      # WFO结果
│   └── run_20251114_155420/
│       └── all_combos.parquet
└── results_combo_wfo/            # 真实回测结果
    └── 20251114_155420_20251114_161032/
        └── top_profit_backtest.csv
```

---

## 🚀 二、快速上手

### 2.1 单数据源训练 (当前场景)

**场景**: 仅使用一个换仓周期的WFO+真实回测数据

```bash
# 方法1: 使用统一Pipeline (推荐)
cd etf_rotation_experiments
python run_ranking_pipeline.py --config configs/ranking_datasets.yaml

# 方法2: 使用传统脚本 (向后兼容)
python train_ranker.py
```

**输出**:
- 模型: `ml_ranker/models/ltr_ranker.txt`
- 评估报告: `ml_ranker/evaluation/evaluation_report.json`
- 稳健性报告: `ml_ranker/evaluation/robustness_report.json`
- 排序对比表: `ml_ranker/evaluation/ranking_comparison_top100.csv`

**预计耗时**: ~7分钟 (训练2分钟 + 稳健性评估5分钟)

### 2.2 多数据源训练 (未来场景)

**场景**: 使用多个不同换仓周期的WFO实验数据

**Step 1**: 编辑配置文件

```bash
vi configs/ranking_datasets.yaml
```

取消注释需要的数据源:

```yaml
datasets:
  - wfo_dir: "results/run_20251114_155420"
    real_dir: "results_combo_wfo/20251114_155420_20251114_161032"
    rebalance_days: 8
    label: "8天换仓基准实验"
  
  # 取消下面的注释以启用新数据源
  - wfo_dir: "results/run_xxx_1d"
    real_dir: "results_combo_wfo/xxx_1d"
    rebalance_days: 1
    label: "1天高频换仓实验"
```

**Step 2**: 运行训练

```bash
python run_ranking_pipeline.py --config configs/ranking_datasets.yaml
```

**结果**: 模型会学习到不同换仓周期的共同规律,泛化能力更强

### 2.3 快速训练 (跳过稳健性评估)

```bash
# 适合快速迭代调试
python run_ranking_pipeline.py --no-robustness
```

**耗时**: ~2分钟 (仅训练,不做稳健性验证)

### 2.4 自定义参数训练

```bash
python run_ranking_pipeline.py \
  --config configs/ranking_datasets.yaml \
  --n-estimators 1000 \
  --learning-rate 0.03 \
  --robustness-folds 10 \
  --robustness-repeats 10
```

---

## 📖 三、多换仓周期数据接入SOP

### 3.1 完整流程

```
┌─────────────────────────────────────────────────────────────────┐
│                   多换仓周期数据接入流程                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 1: 运行新换仓周期的WFO实验                                │
│  ├─ python run_combo_wfo.py --rebalance-freq 5                 │
│  └─ 输出: results/run_xxx_5d/all_combos.parquet                │
│                                                                 │
│  Step 2: 运行相应的真实回测                                     │
│  ├─ python real_backtest/run_profit_backtest.py \              │
│  │    --ranking-file results/run_xxx_5d/all_combos.parquet     │
│  └─ 输出: results_combo_wfo/xxx_5d/profit_backtest.csv         │
│                                                                 │
│  Step 3: 在配置文件中添加数据源                                 │
│  ├─ vi configs/ranking_datasets.yaml                           │
│  └─ 取消注释或新增5天换仓配置                                   │
│                                                                 │
│  Step 4: 重新训练模型                                           │
│  ├─ python run_ranking_pipeline.py \                           │
│  │    --config configs/ranking_datasets.yaml                   │
│  └─ 输出: 新的多数据源训练模型                                  │
│                                                                 │
│  Step 5: 对比新旧模型性能                                       │
│  ├─ cat ml_ranker/evaluation/evaluation_report.json            │
│  └─ 检查Spearman、NDCG、稳健性指标                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 详细步骤

#### Step 1: 运行新换仓周期的WFO实验

```bash
# 示例: 运行5天换仓周期WFO
cd etf_rotation_experiments
python run_combo_wfo.py --rebalance-freq 5
```

**检查输出**:
```bash
ls results/run_*/all_combos.parquet
# 应该看到新的run_xxx目录
```

#### Step 2: 运行真实回测

```bash
# 获取最新WFO运行目录
LATEST_WFO=$(ls -t results/run_* | head -1)
echo "最新WFO目录: $LATEST_WFO"

# 运行真实回测
python real_backtest/run_profit_backtest.py \
  --ranking-file ${LATEST_WFO}/all_combos.parquet \
  --slippage-bps 2
```

**检查输出**:
```bash
ls -lh results_combo_wfo/
# 应该看到新的时间戳目录,包含profit_backtest.csv
```

#### Step 3: 更新配置文件

编辑 `configs/ranking_datasets.yaml`:

```yaml
datasets:
  # 现有8天数据源
  - wfo_dir: "results/run_20251114_155420"
    real_dir: "results_combo_wfo/20251114_155420_20251114_161032"
    rebalance_days: 8
    label: "8天换仓基准实验"
  
  # [NEW] 添加5天数据源
  - wfo_dir: "results/run_20251114_171234"  # ← 替换为实际目录名
    real_dir: "results_combo_wfo/20251114_171234_20251114_172145"  # ← 替换为实际目录名
    rebalance_days: 5
    label: "5天换仓实验"
```

**配置验证**:
```bash
# 检查WFO目录是否存在
ls results/run_20251114_171234/all_combos.parquet

# 检查真实回测目录是否存在
ls results_combo_wfo/20251114_171234_20251114_172145/*profit_backtest*.csv
```

#### Step 4: 重新训练模型

```bash
python run_ranking_pipeline.py --config configs/ranking_datasets.yaml
```

**观察输出**:
```
📦 加载多数据源训练集 (共2个)

[1/2] 8天换仓基准实验 (8天)
  WFO目录: results/run_20251114_155420
  回测目录: results_combo_wfo/20251114_155420_20251114_161032
  ✓ 加载 12597 个样本

[2/2] 5天换仓实验 (5天)
  WFO目录: results/run_20251114_171234
  回测目录: results_combo_wfo/20251114_171234_20251114_172145
  ✓ 加载 12597 个样本

✓ 合并完成: 25194 个样本

来源分布:
  -  5天: 12597 样本 (均值= 0.1234, std=0.0567)
  -  8天: 12597 样本 (均值= 0.1456, std=0.0623)
```

#### Step 5: 对比新旧模型性能

```bash
# 查看评估报告
cat ml_ranker/evaluation/evaluation_report.json | jq '.model_metrics'

# 查看稳健性报告
cat ml_ranker/evaluation/robustness_report.json | jq '.summary'
```

**关注指标**:
- Spearman相关性: 是否仍然>0.85
- 稳健性std: 是否<0.05
- Top-10命中率: 是否≥3/10

### 3.3 批量添加多个数据源

如果一次性运行了多个换仓周期的WFO实验:

```yaml
datasets:
  - {wfo_dir: "results/run_xxx_1d", real_dir: "results_combo_wfo/xxx_1d", rebalance_days: 1}
  - {wfo_dir: "results/run_xxx_2d", real_dir: "results_combo_wfo/xxx_2d", rebalance_days: 2}
  - {wfo_dir: "results/run_xxx_3d", real_dir: "results_combo_wfo/xxx_3d", rebalance_days: 3}
  - {wfo_dir: "results/run_xxx_5d", real_dir: "results_combo_wfo/xxx_5d", rebalance_days: 5}
  - {wfo_dir: "results/run_xxx_8d", real_dir: "results_combo_wfo/xxx_8d", rebalance_days: 8}
  - {wfo_dir: "results/run_xxx_10d", real_dir: "results_combo_wfo/xxx_10d", rebalance_days: 10}
```

**优势**:
- 更多样化的训练数据
- 模型学习到跨时间尺度的规律
- 对新换仓周期泛化能力更强

**风险**:
- 训练时间线性增加
- 不同周期数据质量差异可能影响模型
- 需要验证各数据源的匹配率

---

## ⚙️ 四、配置文件详解

### 4.1 YAML配置结构

`configs/ranking_datasets.yaml` 完整结构:

```yaml
# 全局配置
target_col: "annual_ret_net"      # 主目标列
secondary_target: "sharpe_net"    # 次要目标列

# 数据源列表
datasets:
  - wfo_dir: "..."                # WFO结果目录 (必填)
    real_dir: "..."               # 真实回测目录 (必填)
    rebalance_days: 8             # 换仓周期 (必填)
    weight: 1.0                   # 权重 (可选,默认1.0)
    label: "..."                  # 标签 (可选,用于日志)

# 元数据 (可选)
metadata:
  description: "..."
  version: "..."
```

### 4.2 参数说明

#### 全局参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `target_col` | string | "annual_ret_net" | 训练目标列,用于排序学习 |
| `secondary_target` | string | "sharpe_net" | 次要目标列,用于验证和分析 |

#### DataSource参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `wfo_dir` | string | ✅ | WFO结果目录路径,包含all_combos.parquet |
| `real_dir` | string | ✅ | 真实回测目录路径,包含profit_backtest.csv |
| `rebalance_days` | int | ✅ | 换仓周期天数,会作为特征列加入训练集 |
| `weight` | float | ❌ | 数据集权重,保留字段暂未使用(默认1.0) |
| `label` | string | ❌ | 数据源标签,用于日志显示和调试 |

#### rebalance_days的作用

`rebalance_days` 字段会作为元数据列加入训练集:

```python
# 训练集中会包含:
df['rebalance_days'] = [8, 8, 8, ..., 5, 5, 5, ...]
```

**用途**:
1. **数据追溯**: 知道每个样本来自哪个换仓周期
2. **分组分析**: 可以按换仓周期分组评估模型性能
3. **未来扩展**: 可作为特征输入模型(目前未启用)

**是否应该作为特征?**

目前`rebalance_days`仅作为元数据,不输入模型。未来可考虑:

```python
# 在feature_engineer.py中添加:
if 'rebalance_days' in df.columns:
    features['rebalance_days_log'] = np.log(df['rebalance_days'])
    features['is_high_freq'] = (df['rebalance_days'] <= 3).astype(int)
```

**权衡**:
- ✅ 优势: 模型可以学习换仓周期的影响
- ❌ 劣势: 对未见过的换仓周期泛化能力未知

### 4.3 配置示例

#### 示例1: 单数据源配置

```yaml
target_col: "annual_ret_net"
datasets:
  - wfo_dir: "results/run_20251114_155420"
    real_dir: "results_combo_wfo/20251114_155420_20251114_161032"
    rebalance_days: 8
```

#### 示例2: 多数据源配置

```yaml
target_col: "annual_ret_net"
secondary_target: "sharpe_net"

datasets:
  - wfo_dir: "results/run_20251114_155420"
    real_dir: "results_combo_wfo/20251114_155420_20251114_161032"
    rebalance_days: 8
    weight: 1.0
    label: "8天基准"
  
  - wfo_dir: "results/run_20251114_171234"
    real_dir: "results_combo_wfo/20251114_171234_20251114_172145"
    rebalance_days: 5
    weight: 1.0
    label: "5天实验"
  
  - wfo_dir: "results/run_20251114_182345"
    real_dir: "results_combo_wfo/20251114_182345_20251114_183456"
    rebalance_days: 10
    weight: 1.0
    label: "10天实验"
```

#### 示例3: 使用不同权重

```yaml
datasets:
  - wfo_dir: "results/run_xxx_8d"
    real_dir: "results_combo_wfo/xxx_8d"
    rebalance_days: 8
    weight: 2.0  # 重点关注8天数据
  
  - wfo_dir: "results/run_xxx_1d"
    real_dir: "results_combo_wfo/xxx_1d"
    rebalance_days: 1
    weight: 0.5  # 降低1天数据权重
```

**注意**: 权重功能尚未实现,预留给未来版本

---

## 🔧 五、Pipeline内部流程

### 5.1 Pipeline执行流程

```python
run_training_pipeline(config)
    │
    ├──▶ STEP 1: 加载数据
    │    ├─ load_multi_source_data(config)
    │    │  ├─ 遍历config.datasets
    │    │  ├─ load_wfo_features(wfo_dir)
    │    │  ├─ load_real_backtest_results(real_dir)
    │    │  ├─ build_training_dataset(wfo, real)
    │    │  └─ 添加rebalance_days元数据列
    │    └─ 返回: merged_df, y, metadata
    │
    ├──▶ STEP 2: 特征工程
    │    ├─ build_feature_matrix(merged_df)
    │    │  ├─ extract_scalar_features() # 16个标量特征
    │    │  ├─ expand_sequence_features() # 21个序列特征
    │    │  ├─ build_cross_features() # 6个交叉特征
    │    │  └─ parse_combo_features() # 4个combo特征
    │    └─ 返回: X (n_samples × 44特征)
    │
    ├──▶ STEP 3: 模型训练
    │    ├─ LTRRanker(objective='regression')
    │    ├─ 5-Fold交叉验证训练
    │    ├─ StandardScaler特征标准化
    │    └─ 返回: trained_model
    │
    ├──▶ STEP 4: 模型评估
    │    ├─ model.predict(X) → scores, ranks
    │    ├─ compute_spearman(y_true, scores)
    │    ├─ compute_ndcg(y_true, scores)
    │    ├─ compute_topk_metrics(y_true, scores)
    │    └─ generate_evaluation_report()
    │
    ├──▶ STEP 5: 稳健性评估 (可选)
    │    ├─ evaluate_kfold_cv(X, y, 5折)
    │    │  └─ 每折训练独立模型并评估
    │    ├─ evaluate_repeated_holdout(X, y, 5次)
    │    │  └─ 每次随机80/20划分
    │    └─ generate_robustness_report()
    │
    ├──▶ STEP 6: 保存模型
    │    ├─ model.save("ml_ranker/models/ltr_ranker")
    │    │  ├─ ltr_ranker.txt (LightGBM模型)
    │    │  ├─ ltr_ranker_meta.pkl (scaler等)
    │    │  └─ ltr_ranker_features.json (特征列表)
    │    └─
    │
    └──▶ STEP 7: 生成报告
         ├─ evaluation_report.json (评估指标)
         ├─ robustness_report.json (稳健性分析)
         └─ ranking_comparison_top100.csv (对比表)
```

### 5.2 数据合并逻辑

多数据源合并流程:

```python
# 伪代码
all_merged = []
for ds in config.datasets:
    # 加载单个数据源
    wfo_df = load_wfo_features(ds.wfo_dir)
    real_df = load_real_backtest_results(ds.real_dir)
    
    # 按combo列匹配
    merged = pd.merge(wfo_df, real_df, on='combo', how='inner')
    
    # 添加元数据列
    merged['rebalance_days'] = ds.rebalance_days
    merged['source_label'] = ds.label
    merged['source_id'] = idx
    
    all_merged.append(merged)

# 纵向拼接所有数据源
combined_df = pd.concat(all_merged, ignore_index=True)
```

**关键点**:
- 使用`pd.merge(..., on='combo', how='inner')`确保WFO和真实回测匹配
- 每个数据源独立匹配,匹配率应>95%
- `rebalance_days`列用于标记样本来源

### 5.3 特征工程流程

44维特征构建:

```python
# 1. 标量特征 (16个)
scalar_feats = [
    'combo_size', 'mean_oos_ic', 'oos_ic_std', 'oos_ic_ir',
    'positive_rate', 'best_rebalance_freq', 'stability_score',
    'mean_oos_sharpe', 'oos_sharpe_std', 'mean_oos_sample_count',
    'oos_compound_sharpe', 'oos_compound_mean', 'oos_compound_std',
    'oos_compound_sample_count', 'p_value', 'q_value'
]

# 2. 序列特征展开 (21个)
# 从oos_ic_list, oos_sharpe_list, oos_ir_list提取:
seq_feats = [
    'ic_seq_mean', 'ic_seq_std', 'ic_seq_min', 'ic_seq_max',
    'ic_seq_median', 'ic_positive_ratio', 'ic_seq_trend', 'ic_seq_cv',
    'sharpe_seq_mean', 'sharpe_seq_std', 'sharpe_seq_min', 'sharpe_seq_max',
    'sharpe_seq_trend', 'sharpe_seq_cv',
    'ir_seq_mean', 'ir_seq_std', 'ir_seq_min', 'ir_seq_max',
    'ir_positive_ratio', 'ir_seq_trend', 'ir_seq_cv'
]

# 3. 交叉特征 (6个)
cross_feats = [
    'ic_sharpe_ratio', 'ic_ir_ratio', 'sharpe_ir_ratio',
    'stability_ic_product', 'compound_sharpe_ic_ratio',
    'positive_rate_stability_product'
]

# 4. Combo解析特征 (4个)
combo_feats = [
    'top_n', 'factor_count', 'has_ret_factor', 'has_sharp_factor'
]

# 总计: 16 + 21 + 6 + 4 = 47个特征
# (实际44个,部分特征可能缺失或合并)
```

### 5.4 模型训练细节

**为什么使用regression而非lambdarank?**

```python
# LambdaRank限制: 单query不能超过10000行
# 我们的训练集: 12597行 (单数据源) 或 25194行 (双数据源)
# 解决方案: 使用regression预测分数,再按分数排序

model = LTRRanker(
    objective="regression",  # 而非"lambdarank"
    metric="rmse",
    n_estimators=500,
    learning_rate=0.05
)

# 训练后预测
scores = model.predict(X)  # 回归分数
ranks = scores.argsort()[::-1]  # 按分数降序排名
```

**模型参数选择**:

| 参数 | 值 | 说明 |
|------|-----|------|
| n_estimators | 500 | 树数量,平衡精度和速度 |
| learning_rate | 0.05 | 学习率,较小值防止过拟合 |
| max_depth | 6 | 树深度,控制复杂度 |
| num_leaves | 31 | 叶子数,LightGBM特有参数 |
| min_data_in_leaf | 20 | 叶子最小样本数,防止过拟合 |
| lambda_l1 | 0.1 | L1正则化 |
| lambda_l2 | 0.1 | L2正则化 |

### 5.5 稳健性评估逻辑

**K-Fold交叉验证**:

```python
# 5折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=2025)

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # 训练独立模型
    model = train_single_model(X_train, y_train)
    
    # 预测并评估
    scores = model.predict(X_val)
    spearman = compute_spearman(y_val, scores)
    
    results.append({'fold': fold, 'spearman': spearman})

# 计算稳定性
mean_spearman = np.mean([r['spearman'] for r in results])
std_spearman = np.std([r['spearman'] for r in results])
```

**Repeated Holdout**:

```python
# 5次随机80/20划分
for repeat in range(5):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=2025+repeat
    )
    
    model = train_single_model(X_train, y_train)
    scores = model.predict(X_val)
    spearman = compute_spearman(y_val, scores)
    
    results.append({'repeat': repeat, 'spearman': spearman})
```

**稳定性判断标准**:

| std范围 | 评价 | 建议 |
|---------|------|------|
| < 0.03 | ✅ 优秀 | 可以放心部署 |
| 0.03 - 0.08 | ✅ 良好 | 可以部署,持续监控 |
| 0.08 - 0.15 | ⚠️  一般 | 需要调参或增加数据 |
| > 0.15 | ❌ 较差 | 过拟合风险高,需重新设计 |

---

## ❓ 六、常见问题FAQ

### Q1: 什么时候需要重训模型?

**建议重训时机**:
1. ✅ 新增换仓周期数据源 (如添加5天、10天数据)
2. ✅ 现有数据源大幅更新 (如重跑了所有WFO实验)
3. ✅ 模型性能明显下降 (Spearman < 0.80)
4. ✅ 特征工程有重大改进

**不需要重训的情况**:
- ❌ 只是对新的WFO结果排序 → 使用`apply_ranker.py`
- ❌ 微调超参数 (如n_estimators 500→600) → 提升有限
- ❌ 数据源仅增加少量样本 (< 5%)

### Q2: 如何判断新数据源质量?

**数据质量检查清单**:

```bash
# 1. 检查文件完整性
ls results/run_xxx/all_combos.parquet  # WFO结果
ls results_combo_wfo/xxx/*profit_backtest*.csv  # 真实回测

# 2. 检查样本数量
python -c "import pandas as pd; \
  wfo = pd.read_parquet('results/run_xxx/all_combos.parquet'); \
  real = pd.read_csv('results_combo_wfo/xxx/profit_backtest.csv'); \
  print(f'WFO: {len(wfo)}, Real: {len(real)}')"

# 3. 检查目标列分布
python -c "import pandas as pd; \
  df = pd.read_csv('results_combo_wfo/xxx/profit_backtest.csv'); \
  print(f'annual_ret_net: mean={df.annual_ret_net.mean():.4f}, std={df.annual_ret_net.std():.4f}')"
```

**合格标准**:
- ✅ 样本数 > 10000
- ✅ annual_ret_net均值在[-0.2, 0.5]范围
- ✅ annual_ret_net标准差 > 0.01 (有区分度)
- ✅ WFO与真实回测匹配率 > 95%

### Q3: rebalance_days应该作为特征吗?

**当前方案**: 仅作为元数据,不输入模型

**未来可选方案**:

```python
# 方案A: 直接作为数值特征
features['rebalance_days'] = df['rebalance_days']

# 方案B: 对数变换
features['rebalance_days_log'] = np.log(df['rebalance_days'])

# 方案C: 分类编码
features['is_high_freq'] = (df['rebalance_days'] <= 3).astype(int)
features['is_mid_freq'] = ((df['rebalance_days'] > 3) & (df['rebalance_days'] <= 10)).astype(int)
features['is_low_freq'] = (df['rebalance_days'] > 10).astype(int)

# 方案D: One-Hot编码
rebalance_dummies = pd.get_dummies(df['rebalance_days'], prefix='rebal')
features = pd.concat([features, rebalance_dummies], axis=1)
```

**权衡**:
- ✅ 优点: 模型可以学习换仓周期的影响,泛化能力可能提升
- ❌ 缺点: 对未见过的换仓周期(如15天)泛化能力未知
- ❌ 风险: 可能过拟合于训练集中的换仓周期分布

**推荐**: 先用当前方案(不作为特征),等积累5+个换仓周期数据后再考虑

### Q4: 多数据源合并后样本不平衡怎么办?

**场景**:
```
8天数据: 12597个样本 (60%)
5天数据: 8432个样本 (40%)
```

**解决方案**:

**方案A: 使用weight参数(未来版本)**
```yaml
datasets:
  - rebalance_days: 8
    weight: 0.67  # 12597 / (12597+8432)
  - rebalance_days: 5
    weight: 1.0
```

**方案B: 重采样**
```python
# 在load_multi_source_data()中
from sklearn.utils import resample

# 对少数类上采样
if len(merged) < target_size:
    merged = resample(merged, n_samples=target_size, random_state=42)

# 或对多数类下采样
if len(merged) > target_size:
    merged = merged.sample(n=target_size, random_state=42)
```

**方案C: 分层采样 (最优)**
```python
from sklearn.model_selection import StratifiedKFold

# 按rebalance_days分层
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(X, df['rebalance_days']):
    # 确保每折都包含各种换仓周期
    ...
```

**当前建议**: 暂时不处理,LightGBM对样本不平衡有一定鲁棒性。等数据源达到5+个时再考虑分层采样。

### Q5: 如何调试Pipeline执行失败?

**常见错误及解决方案**:

#### 错误1: 配置文件不存在
```
❌ FileNotFoundError: 配置文件不存在: configs/ranking_datasets.yaml
```

**解决**:
```bash
ls configs/ranking_datasets.yaml  # 检查文件是否存在
python run_ranking_pipeline.py --config configs/ranking_datasets.yaml  # 确保路径正确
```

#### 错误2: 数据源目录不存在
```
❌ FileNotFoundError: WFO结果文件不存在: results/run_xxx/all_combos.parquet
```

**解决**:
```bash
# 检查YAML中的路径是否正确
vi configs/ranking_datasets.yaml

# 列出所有WFO目录
ls -d results/run_*

# 更新YAML中的wfo_dir为实际存在的目录
```

#### 错误3: WFO与真实回测匹配率过低
```
❌ ValueError: 匹配率过低 (45.3%)，请检查数据源是否一致
```

**原因**: WFO和真实回测使用的ranking文件不一致

**解决**:
```bash
# 确认真实回测使用的是正确的WFO输出
python real_backtest/run_profit_backtest.py \
  --ranking-file results/run_xxx/all_combos.parquet  # ← 确保与WFO目录一致
```

#### 错误4: 内存不足
```
❌ MemoryError: Unable to allocate ...
```

**解决**:
```bash
# 方案1: 减少数据源数量
# 方案2: 降低模型复杂度
python run_ranking_pipeline.py --n-estimators 300  # 从500降到300

# 方案3: 跳过稳健性评估
python run_ranking_pipeline.py --no-robustness
```

#### 错误5: YAML格式错误
```
❌ YAMLError: mapping values are not allowed here
```

**解决**:
```bash
# 检查YAML语法
python -c "import yaml; yaml.safe_load(open('configs/ranking_datasets.yaml'))"

# 常见错误: 缩进不一致、缺少引号、特殊字符未转义
```

### Q6: 如何查看训练过程的中间结果?

**查看数据加载日志**:
```python
# Pipeline会自动打印:
📦 加载多数据源训练集 (共2个)
[1/2] 8天换仓基准实验 (8天)
  ✓ 加载 12597 个样本
  目标均值: 0.1234
```

**查看特征矩阵**:
```python
# 在pipeline.py中添加调试代码:
X_df.to_csv("debug_features.csv", index=False)
print(X_df.describe())
```

**查看模型预测分数分布**:
```python
# 在evaluation_report.json中查看:
cat ml_ranker/evaluation/evaluation_report.json | jq '.score_distribution'
```

**查看稳健性详细结果**:
```bash
# 每折的详细指标
cat ml_ranker/evaluation/robustness_detail.csv
```

### Q7: 生产环境使用建议

**模型更新频率**:
- 建议: 每次新增换仓周期数据源时重训
- 或者: 每季度重训一次,更新最新数据

**版本管理**:
```bash
# 保存带时间戳的模型版本
mv ml_ranker/models/ltr_ranker.txt \
   ml_ranker/models/ltr_ranker_v1.0_20251114.txt

# 创建符号链接指向最新版本
ln -s ltr_ranker_v1.0_20251114.txt ml_ranker/models/ltr_ranker.txt
```

**性能监控**:
```python
# 定期评估模型性能
python apply_ranker.py --model ml_ranker/models/ltr_ranker --wfo-dir results/run_latest

# 对比预测Top-10与实际Top-10的Spearman相关性
# 如果Spearman < 0.80,考虑重训模型
```

---

## 📚 七、附录

### 7.1 命令速查表

| 命令 | 说明 |
|------|------|
| `python run_ranking_pipeline.py` | 使用默认配置训练 |
| `python run_ranking_pipeline.py --no-robustness` | 快速训练(跳过稳健性评估) |
| `python run_ranking_pipeline.py --n-estimators 1000` | 自定义树数量 |
| `python train_ranker.py` | 单数据源训练(传统方式) |
| `python apply_ranker.py --model ml_ranker/models/ltr_ranker --wfo-dir results/run_xxx` | 应用模型排序 |
| `cat ml_ranker/evaluation/evaluation_report.json` | 查看评估报告 |
| `cat ml_ranker/evaluation/robustness_report.json` | 查看稳健性报告 |

### 7.2 输出文件说明

| 文件 | 内容 | 用途 |
|------|------|------|
| `ltr_ranker.txt` | LightGBM模型 | 用于预测排序 |
| `ltr_ranker_meta.pkl` | StandardScaler等元数据 | 特征标准化 |
| `ltr_ranker_features.json` | 特征名列表 | 确保特征对齐 |
| `evaluation_report.json` | 完整评估指标 | 模型性能分析 |
| `robustness_report.json` | 稳健性统计 | 过拟合风险评估 |
| `robustness_detail.csv` | 每折详细结果 | 深度分析 |
| `ranking_comparison_top100.csv` | Top-100对比表 | 可视化排序效果 |

### 7.3 关键指标解释

| 指标 | 含义 | 好坏判断 |
|------|------|----------|
| Spearman相关性 | 预测排序与真实排序的一致性 | >0.85优秀, 0.7-0.85良好, <0.7需改进 |
| NDCG@10 | Top-10排序质量(考虑位置权重) | >0.90优秀, 0.80-0.90良好 |
| Top-10命中率 | 预测Top-10中真正的Top-10数量 | ≥3/10及格, ≥5/10优秀 |
| 稳健性std | 不同切分上Spearman的标准差 | <0.03优秀, <0.08良好, >0.15过拟合 |

### 7.4 相关文档链接

- [ML Ranker README](ml_ranker/README.md) - 模块整体介绍
- [QUICKSTART](ml_ranker/QUICKSTART.md) - 5分钟快速上手
- [ROBUSTNESS_GUIDE](ml_ranker/ROBUSTNESS_GUIDE.md) - 稳健性评估详解
- [IMPLEMENTATION_SUMMARY](ml_ranker/IMPLEMENTATION_SUMMARY.md) - 实现总结

---

## 📝 八、更新日志

### v1.1 (2025-11-14)
- ✅ 新增统一Pipeline系统
- ✅ 支持多数据源/多换仓周期训练
- ✅ 新增配置文件管理(YAML)
- ✅ 重构训练流程,提升代码复用性
- ✅ 完善文档和使用指南

### v1.0 (2024-11-15)
- ✅ 初始版本: 单数据源训练
- ✅ LightGBM排序模型
- ✅ 稳健性评估模块
- ✅ 模型应用脚本

---

**编写**: ML Ranker Team  
**最后更新**: 2025-11-14  
**文档版本**: v1.1
