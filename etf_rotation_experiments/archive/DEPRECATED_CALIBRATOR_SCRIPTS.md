# Calibrator相关脚本废弃说明

**时间**: 2025-11-13  
**原因**: 已从WFO流水线移除ML校准器(WFORealBacktestCalibrator)，回归IC排序策略

## 废弃脚本清单

以下脚本依赖已删除的calibrator模块或ranking_blends目录结构，**不再维护**：

### 1. Calibrator训练/应用

- `scripts/train_calibrator_full.py` - 训练GBDT Sharpe预测模型
- `scripts/train_calibrator_profit.py` - 训练GBDT年化收益预测模型
- `scripts/apply_rank_calibrator.py` - 应用两阶段校准器生成ranking_blends
- `scripts/diagnose_calibrator.py` - 校准器诊断分析

### 2. LambdaRank排序实验
- `scripts/train_simple_lambdarank.py`
- `scripts/apply_lambdarank_model.py`
- `scripts/quick_lambdarank_experiment.py`

### 3. 增强特征实验
- `scripts/generate_enhanced_features.py`
- `scripts/train_enhanced_gbdt.py`
- `scripts/apply_enhanced_gbdt.py`
- `scripts/train_multi_objective_gbdt.py`

### 4. 验证/对比脚本
- `scripts/validate_ranking_predictive_power.py` - 依赖ranking_blends/ranking_baseline.parquet
- `scripts/validate_ml_ranking_accuracy.py` - 同上
- `scripts/export_top100_from_blends.py` - 从ranking_blends导出
- `scripts/compare_topk_backtests.py` - 对比IC vs Calibrator排序
- `scripts/generate_ensemble_rankings.py` - IC∩Calibrator集成策略
- `scripts/generate_comprehensive_report.py` - 综合报告生成器
- `scripts/evaluate_rank_calibrator.py` - 校准器评估

### 5. 调试脚本
- `debug_ranking_diagnosis.py` - 加载calibrator_gbdt_profit.joblib

## 新流水线说明

**核心变化**:
1. 移除 `core/wfo_realbt_calibrator.py`
2. `combo_wfo_optimizer.py` 改用 `scoring_strategy` 参数（默认`ic`）
3. 输出文件改为:
   - `results/run_XX/top_combos.parquet` (Top5000)
   - `results/run_XX/ranking_ic_top5000.parquet` (IC排序全集)
   - `results/run_XX/top100_by_ic.parquet` (兼容旧流程)

**后续使用**:
- 真实回测请使用 `--ranking-file results/run_XX/ranking_ic_top5000.parquet`
- 验证脚本需直接对接真实回测CSV，而非ranking_blends

## 替代方案

如需验证WFO→RealBacktest映射质量：
1. 运行 `run_combo_wfo.py` 生成IC排序
2. 运行 `real_backtest/run_profit_backtest.py --topk 5000`
3. 直接计算Spearman相关性：`mean_oos_ic` vs `sharpe_net`

## 归档位置

旧calibrator逻辑已归档至:

```text
archive/failed_experiments/v2_breadth_市场广度风控失败_20251110/core/wfo_realbt_calibrator.py
```

