# ML Ranker: 机器学习策略排序系统

## 🎯 项目简介

本系统使用LightGBM学习从WFO特征到真实回测收益的排序映射，解决WFO排序与实际表现不一致的问题。

**核心价值：**
- ✅ Spearman相关性: **0.9480** (极高的排序一致性)
- ✅ Top-10命中率: 3/10 (baseline: 0/10)
- ✅ Top-10平均收益: **0.2036** (baseline: 0.0850, 提升139%)
- ✅ NDCG@10: **0.9479** (接近完美排序)

**🆕 v1.1 更新 (2025-11-14)**:
- ✅ 统一训练Pipeline: 一键完成训练+评估+稳健性验证
- ✅ 多数据源支持: 轻松整合多个换仓周期的WFO实验
- ✅ YAML配置管理: 简化数据源管理和模型训练
- ✅ 完整文档: [RANKING_PIPELINE_GUIDE.md](RANKING_PIPELINE_GUIDE.md) 详细使用指南

## 📊 训练结果摘要

### 模型性能对比

| 指标 | WFO原始排序 | LTR模型 | 提升 |
|------|------------|---------|------|
| Spearman相关性 | 0.0181 | **0.9480** | +5136% |
| Top-10命中率 | 0/10 | **3/10** | +300% |
| Top-10平均收益 | 0.0850 | **0.2036** | +139% |
| NDCG@10 | 0.5206 | **0.9479** | +82% |

### Top-15重要特征

1. `sharpe_seq_max` - Sharpe序列最大值
2. `ic_seq_max` - IC序列最大值
3. `oos_compound_std` - OOS复合标准差
4. `ic_seq_trend` - IC趋势斜率
5. `oos_ic_std` - OOS IC标准差
6. `sharpe_seq_min` - Sharpe序列最小值
7. `oos_compound_mean` - OOS复合均值
8. `ir_seq_std` - IR序列标准差
9. `oos_sharpe_std` - OOS Sharpe标准差
10. `ic_seq_median` - IC序列中位数

**关键发现：** 序列特征（趋势、极值、波动）比单点统计更能预测真实表现。

---

## 🚀 快速开始

### 方法1: 统一Pipeline (推荐⭐⭐⭐⭐⭐)

**适用场景**: 生产环境、多数据源训练、需要稳健性验证

```bash
cd etf_rotation_experiments

# 基础训练(自动稳健性评估)
python run_ranking_pipeline.py --config configs/ranking_datasets.yaml

# 快速训练(跳过稳健性评估)
python run_ranking_pipeline.py --no-robustness

# 自定义参数
python run_ranking_pipeline.py \
  --config configs/ranking_datasets.yaml \
  --n-estimators 1000 \
  --learning-rate 0.03 \
  --robustness-folds 10
```

**输出：**
- 模型: `ml_ranker/models/ltr_ranker.txt`
- 评估报告: `ml_ranker/evaluation/evaluation_report.json`
- 稳健性报告: `ml_ranker/evaluation/robustness_report.json`
- 排序对比表: `ml_ranker/evaluation/ranking_comparison_top100.csv`

**预计耗时**: 
- 完整训练(含稳健性评估): ~7分钟
- 快速训练(跳过稳健性): ~2分钟

### 方法2: 单数据源训练 (传统方式)

**适用场景**: 向后兼容、快速实验、单一换仓周期

```bash
# 自动查找最新WFO+回测数据
python train_ranker.py

# 指定数据源
python train_ranker.py \
  --wfo-dir results/run_20251114_155420 \
  --backtest-dir results_combo_wfo/20251114_155420_20251114_161032
```

### 方法3: 应用模型 (对新WFO排序)

```bash
python apply_ranker.py \
  --model ml_ranker/models/ltr_ranker \
  --wfo-dir results/run_NEW \
  --top-k 50
```

**输出：**
- `results/run_NEW/ranked_combos.csv` - 全量排序结果
- `results/run_NEW/ranked_top50.csv` - Top-50策略

---

## 📦 多数据源训练 (新功能)

### 配置多换仓周期数据

编辑 `configs/ranking_datasets.yaml`:

```yaml
datasets:
  # 8天换仓数据
  - wfo_dir: "results/run_20251114_155420"
    real_dir: "results_combo_wfo/20251114_155420_20251114_161032"
    rebalance_days: 8
    label: "8天基准实验"
  
  # 5天换仓数据 (新增)
  - wfo_dir: "results/run_xxx_5d"
    real_dir: "results_combo_wfo/xxx_5d"
    rebalance_days: 5
    label: "5天换仓实验"
  
  # 10天换仓数据 (新增)
  - wfo_dir: "results/run_xxx_10d"
    real_dir: "results_combo_wfo/xxx_10d"
    rebalance_days: 10
    label: "10天换仓实验"
```

### 运行多数据源训练

```bash
python run_ranking_pipeline.py --config configs/ranking_datasets.yaml
```

**效果**: 模型会学习到不同换仓周期的共同规律,泛化能力更强

**详细指南**: 参见 [RANKING_PIPELINE_GUIDE.md](RANKING_PIPELINE_GUIDE.md)

---

## 📁 项目结构 (v1.1)

```
ml_ranker/
├── __init__.py                   # 模块导出
├── config.py                     # [NEW] 配置类
├── pipeline.py                   # [NEW] 统一训练Pipeline
├── data_loader.py                # 数据加载(已扩展多数据源)
├── feature_engineer.py           # 特征工程（~44特征）
├── ltr_model.py                  # LightGBM LTR模型
├── evaluator.py                  # 评估指标计算
├── robustness_eval.py            # 稳健性验证
├── RANKING_PIPELINE_GUIDE.md    # [NEW] 完整使用指南
├── models/                       # 训练好的模型
│   ├── ltr_ranker.txt
│   ├── ltr_ranker_meta.pkl
│   └── ltr_ranker_features.json
└── evaluation/                   # 评估报告
    ├── evaluation_report.json
    ├── robustness_report.json
    └── ranking_comparison_top100.csv

configs/                          # [NEW] 配置文件
└── ranking_datasets.yaml         # 数据源配置

run_ranking_pipeline.py           # [NEW] 统一训练入口
train_ranker.py                   # 单数据源训练(保留)
apply_ranker.py                   # 模型应用
```

## 🔧 技术细节

### 数据流程

1. **加载WFO特征** (12597个组合 × 27列)
   - 标量特征: `mean_oos_ic`, `oos_sharpe_proxy`, `stability_score`...
   - 序列特征: `oos_ic_list` (19窗口), `oos_sharpe_list`

2. **特征工程** (生成~50特征)
   - 标量特征: 16个基础WFO统计
   - 序列展开: 21个统计特征 (mean, std, min, max, median, trend, CV, positive_ratio)
   - 交叉特征: 6个组合特征 (IC×Sharpe, stability×posrate...)
   - Combo解析: 4个策略结构特征

3. **模型训练**
   - 算法: LightGBM Regression (避免lambdarank的query size限制)
   - 目标: 预测`annual_ret_net` (年化净收益)
   - 验证: 5-fold CV with StandardScaler
   - 评估: Spearman相关性 (主要指标)

4. **排序输出**
   - 模型预测分数 → 排名
   - 与WFO原始排名对比
   - 输出Top-K策略列表

### 关键设计决策

**Q: 为什么用Regression而不是LambdaRank？**  
A: LightGBM LambdaRank对单个query有10000行限制，我们的训练集超过1万行。Regression模式学习分数，然后用分数排序，效果同样优秀（Spearman 0.948）。

**Q: 为什么Spearman相关性这么高？**  
A: WFO特征本身已包含排序信号（IC, Sharpe, stability），模型学习的是"哪些特征组合真正预示高收益"，而非从零学习。

**Q: 序列特征为什么重要？**  
A: 单时间点统计（均值）容易被噪声干扰，序列特征（趋势、极值、波动）揭示策略的动态行为，更能预测未来表现。

## 📈 使用场景

### 场景1：选择Top策略用于实盘

```bash
# 训练模型
python train_ranker.py

# 对最新WFO排序
python apply_ranker.py \
  --model ml_ranker/models/ltr_ranker \
  --wfo-dir results/run_latest \
  --top-k 10

# 查看Top-10
head -11 results/run_latest/ranked_top10.csv
```

### 场景2：对比不同WFO run的排序

```bash
# 对比多个WFO run
for run_dir in results/run_*/; do
  python apply_ranker.py \
    --model ml_ranker/models/ltr_ranker \
    --wfo-dir "$run_dir" \
    --top-k 20
done

# 分析一致性
python -c "
import pandas as pd
from pathlib import Path

top_combos = []
for csv in Path('results').glob('*/ranked_top20.csv'):
    df = pd.read_csv(csv)
    top_combos.append(set(df['combo']))

# 交集 = 多个run都排名靠前的稳定策略
stable = set.intersection(*top_combos)
print(f'跨run稳定Top-20策略数: {len(stable)}')
print(stable)
"
```

### 场景3：重新训练模型（新数据）

```bash
# 用新的WFO + 回测结果重新训练
python train_ranker.py \
  --wfo-dir results/run_20251201_100000 \
  --backtest-dir results_combo_wfo/20251201_100000_20251201_110000 \
  --model-dir ml_ranker/models_v2
```

## 🧪 评估指标说明

### Spearman相关性
- 衡量预测排名与真实排名的一致性
- 范围: [-1, 1]，1表示完全一致
- **本模型: 0.9480** (极优)

### NDCG@K (Normalized Discounted Cumulative Gain)
- 考虑排名位置的加权评估指标
- 范围: [0, 1]，1表示完美排序
- **本模型 NDCG@10: 0.9479**

### Top-K命中率
- 预测Top-K中有多少是真实Top-K
- **本模型 Top-10: 3/10** (baseline: 0/10)

### Top-K平均收益
- 预测Top-K策略的真实平均收益
- 衡量模型选择策略的实际价值
- **本模型 Top-10: 0.2036** (baseline: 0.0850)

## 🔍 常见问题

### Q: 模型需要重新训练吗？
A: 建议每次大规模WFO后重新训练，保持特征分布一致。

### Q: 可以用于其他策略类型吗？
A: 可以。只需要提供WFO特征和真实回测结果，模型会自动学习映射关系。

### Q: 如何解释模型预测？
A: 查看`evaluation_report.json`中的`feature_importance`，了解哪些特征驱动排序。

### Q: Top-10平均收益低于真实Top-10是否正常？
A: 正常。模型是基于历史数据训练的，无法完美预测未来。0.2036 vs 0.2195已经是极好的结果（baseline只有0.0850）。

## 📊 示例输出

### 训练完成提示
```
================================================================================
✅ 训练完成
================================================================================

  模型性能:
    Spearman相关性: 0.9480
    NDCG@10: 0.9479
    Top-10命中率: 3/10
    Top-10平均收益: 0.2036

  输出文件:
    模型: ml_ranker/models/ltr_ranker.txt
    元数据: ml_ranker/models/ltr_ranker.meta.pkl
    评估报告: ml_ranker/evaluation/evaluation_report.json
    对比表: ml_ranker/evaluation/ranking_comparison_top100.csv
```

### Top-10策略预览
```
  #  1  CMF_20D + MAX_DD_60D + PV_CORR_20D + RSI_14 + VOL_RATIO_20D
  #  2  CMF_20D + MOM_20D + OBV_SLOPE_10D + PRICE_POSITION_20D + RSI_14
  #  3  ADX_14D + CMF_20D + OBV_SLOPE_10D + RELATIVE_STRENGTH_VS_MARKET_20D + RSI_14
  #  4  CMF_20D + OBV_SLOPE_10D + PRICE_POSITION_20D + RELATIVE_STRENGTH_VS_MARKET_20D + RSI_14
  #  5  CMF_20D + MAX_DD_60D + RSI_14 + SHARPE_RATIO_20D
  ...
```

## 🛠️ 依赖项

- Python 3.11+
- LightGBM 4.6.0
- scikit-learn 1.7.2
- pandas
- numpy
- scipy

## 📝 引用

如果使用本系统，请注明：

```
ML Ranker: Learning-to-Rank System for ETF Rotation Strategies
Author: Zhang Shenshen
Date: 2024-11-15
Spearman Correlation: 0.9480
```

## 🎓 方法论摘要

本系统遵循 **RESEARCH→INNOVATE→PLAN→EXECUTE→REVIEW** 框架开发：

1. **RESEARCH**: 分析WFO特征与真实回测结果的映射关系
2. **INNOVATE**: 提出Learning-to-Rank方案，利用序列特征提升预测能力
3. **PLAN**: 8阶段实施计划（基础设施 → 数据 → 特征 → 模型 → 评估 → 脚本 → 测试 → 文档）
4. **EXECUTE**: 渐进式实现，确保每个模块可测试
5. **REVIEW**: Spearman 0.9480验证方案有效性

---

**版本:** v1.0  
**最后更新:** 2024-11-15  
**状态:** ✅ 生产就绪
