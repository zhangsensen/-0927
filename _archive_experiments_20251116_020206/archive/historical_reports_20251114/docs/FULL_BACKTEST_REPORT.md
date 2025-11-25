# 全量回测与模型训练报告

## 执行总结

已完成全部 12597 个组合的真实回测，生成完整训练数据集（22597 样本），并训练了两阶段排序模型。

## 完成的任务

### 1. 准备排名文件 ✅
- 从 `all_combos.parquet` 生成完整排名文件
- 总组合数: 12597
- 输出: `results/run_20251111_145454/ranking_baseline_full.parquet`

### 2. 批量回测全部组合 ✅
- 分批执行真实回测（2000/批）
- 已完成全部 12597 个组合的回测
- 回测结果分散在多个批次目录中
- 所有结果已落盘到 `results_combo_wfo/`

### 3. 重建训练数据集 ✅
- 运行 `build_rank_dataset.py` 合并所有回测结果
- 生成数据集: `results/rank_dataset_full_20251111_145454.parquet`
- 样本数: 22597（包含重复组合）
- 特征总数: 194
- 包含所有新增特征：
  - 市场环境适应性特征
  - 组合构成特征
  - 历史表现一致性特征
  - 极端风险指标特征

### 4. 训练两阶段排序模型 ✅
- 使用全量数据训练
- Stage1 模型: `results/models/calibrator_sharpe_filter.txt`
- Stage2 模型: `results/models/calibrator_profit_ranker.txt`
- 交叉验证指标:
  - NDCG@1000: 0.9916
  - Spearman: 0.8294
  - Kendall: 0.7203
  - Top100 overlap: 84.20%
  - Top500 overlap: 89.80%
  - Top1000 overlap: 90.36%

### 5. 应用模型生成新排名 ✅
- 生成 blend 排名（alpha=0.0 到 1.0，步长 0.1）
- 输出目录: `results/run_20251111_145454/ranking_blends/`
- 生成文件:
  - `ranking_baseline.parquet` (alpha=0.0)
  - `ranking_blend_0.10.parquet` 到 `ranking_blend_0.90.parquet`
  - `ranking_lightgbm.parquet` (alpha=1.0)

### 6. 验证模型效果 ✅
- 运行 Top1000 真实回测对比
- 对比策略: baseline, blend_0.30, ml_pure
- 评估报告: `results/calibrator_ab_report_full_20251111_145454.json`

## 关键发现

### 问题 1: 排名完全相同
所有排名策略（baseline、blend、ml_pure）的 Top1000 组合完全相同，性能指标也完全一致。

**原因分析:**
- ML 模型输出分数与 baseline 分数的相关性为 1.0
- 所有组合的 ML 分数都被替换为 baseline 分数

### 问题 2: Sharpe 阈值过滤失效
- Sharpe 预测分数范围: [-4.90, -0.50]
- 设置的阈值: -0.3
- 结果: 所有 12597 个组合的 Sharpe 预测都低于阈值
- 导致: 所有组合的 ML 分数被强制替换为 baseline 分数（见 `apply_rank_calibrator.py` 第 322 行）

### 问题 3: 特征不匹配
推理时缺失以下特征（在训练时存在，但 `all_combos.parquet` 中不存在）:
- `test_position_size`
- `freq`
- `ret_net_minus_gross`
- `ret_net_over_gross`
- `ret_cost_ratio`
- `cost_drag`
- `breakeven_turnover_est`
- `sharpe_net_minus_gross`
- `max_dd_net_minus_gross`
- `dd_ratio_net_over_gross`
- 等 10+ 个特征

这些特征在推理时被填充为 0，导致模型预测不准确。

### 问题 4: 单 Run 训练的局限性
- 训练数据只来自一个 run (`20251111_145454`)
- 虽然有 22597 个样本，但都来自同一次 WFO
- 导致模型可能过拟合到这个特定 run 的特征分布
- 缺乏跨 run 的泛化能力

## 性能基准（Baseline）

### Top100
- 平均年化收益: 21.46%
- 中位数年化收益: 21.38%
- 平均 Sharpe: 1.0427
- 盈利率: 100.00%

### Top200
- 平均年化收益: 20.97%
- 中位数年化收益: 20.79%
- 平均 Sharpe: 1.0249
- 盈利率: 100.00%

### Top500
- 平均年化收益: 20.21%
- 中位数年化收益: 20.05%
- 平均 Sharpe: 1.0021
- 盈利率: 100.00%

### Top1000
- 平均年化收益: 19.17%
- 中位数年化收益: 19.21%
- 平均 Sharpe: 0.9724
- 盈利率: 100.00%

## 根本原因总结

1. **特征工程问题**: 训练数据包含真实回测特征，但推理时这些特征不可用
2. **两阶段架构问题**: Sharpe 过滤阶段的阈值设置不当，导致所有组合被过滤
3. **数据来源单一**: 只有一个 run 的数据，缺乏多样性
4. **目标不一致**: 训练目标（预测 Sharpe/年化收益）与实际应用场景（WFO 排序）不完全匹配

## 建议的改进方向

### 短期修复（立即可行）

1. **调整 Sharpe 阈值**
   - 当前: -0.3
   - 建议: -5.0 或更低（或完全移除 Sharpe 过滤）
   - 理由: 让更多组合通过第一阶段过滤

2. **修复特征不匹配**
   - 方案 A: 在推理时只使用 WFO 特征训练模型
   - 方案 B: 在推理时模拟缺失特征（使用历史平均值或模型预测）
   - 推荐: 方案 A，重新训练只使用 WFO 特征的模型

3. **简化模型架构**
   - 移除两阶段架构
   - 直接训练单阶段排序模型
   - 目标: 预测 WFO 分数或年化收益

### 中期优化（需要更多数据）

1. **扩展训练数据**
   - 收集多个历史 run 的数据（至少 3-5 个 run）
   - 确保数据覆盖不同市场环境
   - 增加数据多样性和泛化能力

2. **改进目标函数**
   - 使用多目标学习（年化收益 + Sharpe + 最大回撤）
   - 引入 Listwise 排序损失（LambdaMART）
   - 优化 Top-K 准确率而非单点预测

3. **特征工程优化**
   - 只使用 WFO 阶段可获得的特征
   - 添加更多组合结构特征（因子类型、周期分布等）
   - 引入市场环境特征（当前市场状态、波动率等）

### 长期方向（架构重构）

1. **在线学习框架**
   - 每次 WFO 后更新模型
   - 使用增量学习避免遗忘
   - 适应市场变化

2. **集成学习**
   - 训练多个模型（不同特征子集、不同算法）
   - 使用 Stacking 或 Voting 集成
   - 提高稳定性和鲁棒性

3. **强化学习**
   - 将策略选择建模为序列决策问题
   - 使用强化学习优化长期收益
   - 考虑组合之间的相关性和风险分散

## 文件清单

### 数据文件
- `results/run_20251111_145454/ranking_baseline_full.parquet` - 全量排名文件（12597 组合）
- `results/rank_dataset_full_20251111_145454.parquet` - 完整训练数据集（22597 样本，194 特征）

### 模型文件
- `results/models/calibrator_sharpe_filter.txt` - Stage1 Sharpe 过滤模型
- `results/models/calibrator_profit_ranker.txt` - Stage2 收益排序模型

### 排名文件
- `results/run_20251111_145454/ranking_blends/ranking_baseline.parquet` - Baseline 排名
- `results/run_20251111_145454/ranking_blends/ranking_blend_0.30.parquet` - 混合排名（30% ML）
- `results/run_20251111_145454/ranking_lightgbm.parquet` - 纯 ML 排名

### 回测结果
- `results_combo_wfo/20251111_145454_*/top*_profit_backtest_slip0bps_*.csv` - 各批次回测结果

### 评估报告
- `results/calibrator_ab_report_full_20251111_145454.json` - A/B 测试评估报告

## 下一步行动

### 立即执行
1. 重新训练只使用 WFO 特征的单阶段模型
2. 移除 Sharpe 过滤逻辑或大幅降低阈值
3. 验证新模型是否能产生不同的排名

### 后续计划
1. 收集更多历史 run 的数据
2. 实现多 run 交叉验证
3. 优化特征工程和模型架构

## 结论

本次全量回测和模型训练成功完成了所有技术步骤，但由于特征不匹配和阈值设置问题，模型未能产生有效的排名改进。核心问题在于训练数据和推理数据的特征不一致，以及两阶段架构的 Sharpe 过滤过于严格。

建议优先修复特征不匹配问题，重新训练只使用 WFO 特征的模型，并简化为单阶段架构，以快速验证模型的有效性。

---

**报告生成时间**: 2025-11-11 16:20  
**项目路径**: `/Users/zhangshenshen/深度量化0927/etf_rotation_experiments`

