# LTR 模型重训计划

**当前模型**: `strategies/ml_ranker/models/ltr_ranker`  
**训练日期**: 未知（待查档案）  
**状态**: ⚠️ 需重训

---

## 重训触发条件

自动触发（满足任一条件）:
1. 距上次训练超过 **90 天**
2. 最近 3 次 WFO run 的 Top-2000 平均年化低于 **15%** baseline
3. ML 排序 Top-1 OOS IC 连续 5 次低于 **0.02**

手动触发（建议场景）:
- 市场结构性变化（如政策/风格切换）
- 累积新 WFO 结果数量 ≥ 5 次
- 回测指标出现明显退化

---

## 重训流程

### 1. 数据准备
```bash
# 收集最近 N 次 WFO 结果作为训练集
cd etf_rotation_experiments
find results/run_* -name "all_combos.parquet" -mtime -180 | head -5
```

**要求**:
- 最少 3 次 WFO run，覆盖不同时间窗口
- 每次 run 至少 5000 个候选组合
- 包含完整回测结果（用于计算真实收益标签）

### 2. 执行训练
```bash
# 方式1: 使用统一 pipeline（推荐）
python applications/run_ranking_pipeline.py \
  --config configs/ranking_datasets.yaml \
  --output-dir ml_ranker/models_v2 \
  --enable-robustness

# 方式2: 单数据源训练（快速迭代）
python applications/train_ranker.py \
  --wfo-dir results/run_20251116_035732 \
  --backtest-dir results_combo_wfo/20251116_035732_20251116_035758 \
  --model-dir ml_ranker/models_v2 \
  --n-folds 5 \
  --n-estimators 500
```

### 3. 验证评估
```bash
# 在测试集上验证性能
python applications/apply_ranker.py \
  --model ml_ranker/models_v2/ltr_ranker \
  --wfo-dir results/run_LATEST \
  --output ranked_combos_v2.csv \
  --top-k 200

# 对比新旧模型指标
python tools/compare_model_versions.py \
  --model-old ml_ranker/models/ltr_ranker \
  --model-new ml_ranker/models_v2/ltr_ranker \
  --test-data results/run_LATEST
```

**验收标准**:
- Spearman 相关性 ≥ 0.60 (训练集)
- NDCG@10 ≥ 0.70
- Top-100 组合平均年化 ≥ 旧模型 + 2%
- 稳健性测试通过（跨时间窗口 Sharpe 标准差 < 0.2）

### 4. 版本管理
```bash
# 标记版本号和训练日期
echo "v2.0-$(date +%Y%m%d)" > ml_ranker/models_v2/VERSION
cp ml_ranker/models_v2/ltr_ranker* ml_ranker/models_archive/v2.0/

# 更新生产配置（验证通过后）
sed -i 's|models/ltr_ranker|models_v2/ltr_ranker|' configs/combo_wfo_config.yaml

# 提交变更
git add ml_ranker/models_v2/ configs/combo_wfo_config.yaml
git commit -m "feat: update LTR model to v2.0-$(date +%Y%m%d)"
```

---

## 下次重训计划

**目标日期**: 2025-12-01 (距今 15 天)  
**数据集**: 收集 2025-11-01 至 2025-11-30 期间的 WFO 结果  
**预期改进**: 纳入最新市场特征，目标 Top-2000 平均 Sharpe 提升至 0.95+

**行动清单**:
- [ ] 2025-11-25: 执行一次完整 WFO（43 ETF，2~5 因子组合）
- [ ] 2025-11-28: 收集历史 WFO 结果，合并训练集
- [ ] 2025-11-29: 执行训练 + 验证
- [ ] 2025-12-01: 切换生产配置，记录版本信息

---

## 历史记录

| 版本 | 训练日期 | 数据集 | 性能指标 | 备注 |
|------|----------|--------|----------|------|
| v1.0 | 未知 | 未知 | Spearman=? | 当前生产模型 |
| v2.0 | 待定 | 5 次 WFO | - | 计划中 |

---

## 自动化脚本（TODO）

创建 `tools/retrain_ltr_model.sh`:
```bash
#!/bin/bash
# 自动化 LTR 模型重训流程

set -e

# 1. 检查触发条件
# 2. 收集数据
# 3. 执行训练
# 4. 验证评估
# 5. 版本管理
# 6. 发送通知
```

**cron 定时任务**:
```cron
# 每月 1 号凌晨 2 点检查是否需要重训
0 2 1 * * cd /path/to/project && bash tools/check_retrain_trigger.sh
```
