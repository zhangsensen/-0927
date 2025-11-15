# PLAN 阶段

基于方案B (Learning-to-Rank模型)，我现在制定详细的实施计划。

---

## 实施计划概览

**目标**: 构建LightGBM LambdaRank模型，学习从WFO特征到真实回测排序的映射

**模块结构**:
```
etf_rotation_experiments/
├── ml_ranker/
│   ├── __init__.py
│   ├── data_loader.py      # 数据加载与预处理
│   ├── feature_engineer.py # 特征工程 (序列特征展开)
│   ├── ltr_model.py        # LTR模型训练与推理
│   ├── evaluator.py        # 排序评估指标
│   └── cli.py              # 命令行入口
├── train_ranker.py         # 主训练脚本
└── apply_ranker.py         # 应用脚本 (新WFO结果排序)
```

---

## IMPLEMENTATION CHECKLIST

### Phase 1: 基础设施搭建 (30分钟)

- [ ] **1.1** 创建 `ml_ranker/` 目录结构
- [ ] **1.2** 创建 `ml_ranker/__init__.py` (模块初始化)
- [ ] **1.3** 检查依赖: lightgbm, scikit-learn, pandas, numpy

### Phase 2: 数据加载模块 (45分钟)

- [ ] **2.1** 实现 `data_loader.py::load_wfo_features()`
  - 读取 `all_combos.parquet`
  - 提取标量特征 (27列中的数值列)
  - 返回 DataFrame
  
- [ ] **2.2** 实现 `data_loader.py::load_real_backtest_results()`
  - 读取真实回测CSV
  - 提取目标列: `annual_ret_net`, `sharpe_net`
  - 返回 DataFrame

- [ ] **2.3** 实现 `data_loader.py::build_training_dataset()`
  - 按 `combo` 字段 merge WFO和真实结果
  - 验证100%匹配
  - 返回 (X, y, metadata)

### Phase 3: 特征工程模块 (60分钟)

- [ ] **3.1** 实现 `feature_engineer.py::extract_scalar_features()`
  - 选择基础标量特征 (IC, Sharpe, stability等)
  - 处理缺失值 (fillna或dropna)
  
- [ ] **3.2** 实现 `feature_engineer.py::expand_sequence_features()`
  - 从 `oos_ic_list` 提取: mean, std, min, max, trend
  - 从 `oos_sharpe_list` 提取: mean, std, cv
  - 计算稳定性指标: positive_rate_in_oos
  
- [ ] **3.3** 实现 `feature_engineer.py::create_cross_features()`
  - IC × Sharpe 交互特征
  - stability_score × positive_rate
  - (可选) combo字符串解析 (因子数量)

- [ ] **3.4** 实现 `feature_engineer.py::build_feature_matrix()`
  - 整合所有特征
  - 返回最终特征矩阵 X

### Phase 4: LTR模型模块 (90分钟)

- [ ] **4.1** 实现 `ltr_model.py::LTRRanker` 类
  - 初始化LightGBM参数
  - 设置 `objective='lambdarank'`
  - 配置 `metric='ndcg'`

- [ ] **4.2** 实现 `LTRRanker.train()`
  - 构造 query group (单个group包含所有策略)
  - K-Fold交叉验证 (5折)
  - 训练集/验证集分割
  - 保存最佳模型

- [ ] **4.3** 实现 `LTRRanker.predict()`
  - 对新数据预测排序分数
  - 返回 predicted_score 和 rank

- [ ] **4.4** 实现 `LTRRanker.save()` / `LTRRanker.load()`
  - 模型序列化 (joblib)
  - 保存特征列名和预处理器

### Phase 5: 评估模块 (45分钟)

- [ ] **5.1** 实现 `evaluator.py::compute_spearman_correlation()`
  - 预测排序 vs 真实排序的Spearman相关系数

- [ ] **5.2** 实现 `evaluator.py::compute_ndcg()`
  - NDCG@5, NDCG@10, NDCG@50

- [ ] **5.3** 实现 `evaluator.py::compute_topk_metrics()`
  - Top-K命中率
  - Top-K平均收益 vs baseline

- [ ] **5.4** 实现 `evaluator.py::generate_evaluation_report()`
  - 综合评估报告
  - 对比WFO原始排序

### Phase 6: 主流程脚本 (60分钟)

- [ ] **6.1** 实现 `train_ranker.py`
  - 加载数据
  - 特征工程
  - 训练模型
  - 交叉验证
  - 保存模型和报告

- [ ] **6.2** 实现 `apply_ranker.py`
  - 加载训练好的模型
  - 读取新WFO结果
  - 预测排序
  - 输出排序结果CSV

- [ ] **6.3** 实现 `ml_ranker/cli.py`
  - 命令行参数解析
  - 子命令: train, apply, evaluate

### Phase 7: 测试与验证 (45分钟)

- [ ] **7.1** 端到端测试: 训练流程
  - 验证数据加载正确
  - 验证特征数量和类型
  - 验证模型训练无报错

- [ ] **7.2** 端到端测试: 推理流程
  - 加载模型
  - 对同一批数据预测
  - 验证排序输出格式

- [ ] **7.3** 评估指标验证
  - 计算Spearman相关性
  - 对比Top-10策略
  - 生成完整报告

### Phase 8: 文档与封装 (30分钟)

- [ ] **8.1** 编写 `ml_ranker/README.md`
  - 快速开始指南
  - 训练命令示例
  - 应用命令示例

- [ ] **8.2** 添加代码注释和docstring
  - 每个函数的输入输出
  - 关键假设和约束

- [ ] **8.3** 创建示例配置文件 (可选)
  - `ml_ranker_config.yaml`

---

## 关键技术决策

### 决策1: 目标变量选择
**选择**: `annual_ret_net` (年化收益净值) 作为主目标
**理由**: 
- 最直接反映策略盈利能力
- 分布较均匀 (std=4.8%)
- 可以用`sharpe_net`作为次要目标验证

### 决策2: Query Group构造
**选择**: 整个实验作为一个query group
**理由**:
- 单次实验的所有策略需要相对排序
- 符合LTR的listwise learning范式

### 决策3: 交叉验证策略
**选择**: 5-Fold Stratified KFold
**理由**:
- 单时间段数据无法时序切分
- Stratified保证各折目标分布平衡
- 5折平衡训练效率和验证稳定性

### 决策4: 特征选择
**包含特征**:
- ✅ 基础WFO指标: IC, Sharpe, stability
- ✅ 序列统计: OOS窗口的均值/方差/趋势
- ✅ 交叉特征: IC×Sharpe
- ❌ 暂不解析combo字符串 (第二阶段优化)

### 决策5: 评估基线
**Baseline**: WFO的`mean_oos_ic`原始排序
**对比指标**:
1. Spearman相关性提升
2. NDCG@10提升
3. Top-10平均收益提升

---

## 边界情况处理

1. **缺失值**: 用0填充或列中位数填充
2. **异常值**: 不做winsorize (保留真实分布)
3. **特征共线性**: LightGBM自动处理
4. **过拟合**: 通过early stopping + CV控制
5. **新combo出现**: 特征提取流程需泛化

---

## 预期输出文件

训练后生成:
```
ml_ranker/
├── models/
│   ├── ltr_ranker_20251114.pkl        # 模型文件
│   ├── feature_columns.json           # 特征列表
│   └── training_report.json           # 训练报告
├── evaluation/
│   ├── cv_results.csv                 # 交叉验证结果
│   ├── feature_importance.png         # 特征重要性图
│   └── ranking_comparison.csv         # 排序对比
```

应用后生成:
```
results/
└── ranked_by_ltr_20251114.csv         # 重新排序的策略列表
```

---

现在进入 **EXECUTE** 阶段，我将按照checklist逐步实现...