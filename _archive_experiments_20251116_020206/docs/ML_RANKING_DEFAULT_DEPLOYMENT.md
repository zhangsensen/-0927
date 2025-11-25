# ML 排序默认化部署说明

**部署日期**: 2025-11-14  
**状态**: ✅ 已完成

---

## 变更摘要

基于 A/B 测试验证结果 (Top-200 平均 Sharpe +69%, 年化收益 +7.87%), 已将 **ML 排序设为 WFO 系统的默认排序方式**。

---

## 核心改动

### 1. 配置文件 (`configs/combo_wfo_config.yaml`)

**变更**:
- `ranking.method` 默认值: `"wfo"` → `"ml"` ✅
- 新增详细注释说明 ML 为推荐的生产排序方式
- WFO 排序保留为显式备用选项

**当前配置**:
```yaml
ranking:
  method: "ml"  # 生产默认: "ml" (ML模型排序) | 备用: "wfo" (原始WFO排序)
  top_n: 200
  ml_model_path: "ml_ranker/models/ltr_ranker"
```

### 2. 主流程 (`run_combo_wfo.py`)

**变更**:
- 默认 `ranking_method` 从 `"wfo"` 改为 `"ml"`
- 增强日志输出, 明确标识当前排序方式:
  - ML 模式: `📊 排序方式: ML (LTR 模型) ✅ 生产推荐`
  - WFO 模式: `📊 排序方式: WFO (mean_oos_ic) ⚠️ 备用模式`
- 优化错误提示, 回退时清晰标注 `⚠️ 自动回退到 WFO 排序模式`

### 3. 文档更新

**更新文件**:
- `README.md`: 新增排序模式说明和快速开始, 标注 ML 为默认
- `docs/ML_RANKING_INTEGRATION_GUIDE.md`: 重构文档结构, ML 排序置顶

**废弃配置**:
- `configs/combo_wfo_config_ml_test.yaml`: 标记为"历史测试用, 已废弃"

---

## 使用方式

### 默认使用 (ML 排序)

```bash
cd /Users/zhangshenshen/深度量化0927/etf_rotation_experiments
python run_combo_wfo.py  # 默认使用 ML 排序
```

**输出文件**:
- `results/run_XXXXXX/ranking_ml_top200.parquet` - ML 排序结果
- `results/run_XXXXXX/top_combos.parquet` - Top-200 组合

**日志标识**:
```
🔀 排序模式选择
  📊 排序方式: ML (LTR 模型) ✅ 生产推荐
  TopN: 200
  模型路径: ml_ranker/models/ltr_ranker
⚡ 执行ML排序...
✅ ML排序完成: 12597 个组合
```

### 备用模式 (WFO 排序)

如需使用 WFO 原始排序, 修改 `configs/combo_wfo_config.yaml`:

```yaml
ranking:
  method: "wfo"  # 改为 wfo
```

然后运行:
```bash
python run_combo_wfo.py
```

**输出文件**:
- `results/run_XXXXXX/ranking_ic_top200.parquet` - WFO 排序结果

---

## 自动回退机制

系统具备完善的容错机制, 当 ML 模式遇到问题时会自动回退到 WFO 排序:

**回退场景**:
1. ML 模块 (`apply_ranker.py`) 不可用
2. ML 模型文件不存在 (`ml_ranker/models/ltr_ranker/`)
3. ML 排序执行失败 (Exception)

**回退日志**:
```
❌ ML模型不存在: ml_ranker/models/ltr_ranker
   💡 提示: 请先运行 python run_ranking_pipeline.py 训练模型
   ⚠️ 自动回退到 WFO 排序模式
```

---

## 验证测试

### 测试 1: ML 排序 (默认) ✅

```bash
# 确认配置
grep "method:" configs/combo_wfo_config.yaml
# 输出: method: "ml"

# 运行 WFO
python run_combo_wfo.py

# 验证输出
ls results/run_*/ranking_ml_top200.parquet  # 应该存在
```

**结果**: ✅ ML 排序成功, 生成 `ranking_ml_top200.parquet`, 日志显示 "ML (LTR 模型) ✅ 生产推荐"

### 测试 2: WFO 备用模式 ✅

```bash
# 修改配置为 wfo
sed -i '' 's/method: "ml"/method: "wfo"/' configs/combo_wfo_config.yaml

# 运行 WFO
python run_combo_wfo.py

# 验证输出
ls results/run_*/ranking_ic_top200.parquet  # 应该存在

# 恢复配置
sed -i '' 's/method: "wfo"/method: "ml"/' configs/combo_wfo_config.yaml
```

**结果**: ✅ WFO 排序成功, 生成 `ranking_ic_top200.parquet`, 日志显示 "WFO (mean_oos_ic) ⚠️ 备用模式"

---

## ML 模型管理

### 当前模型

**路径**: `ml_ranker/models/ltr_ranker/`

**文件**:
- `ltr_ranker.txt` (543KB) - LightGBM 模型
- `ltr_ranker_meta.pkl` (2.8KB) - 元数据
- `ltr_ranker_features.json` (902B) - 特征列表

**训练指标** (基于历史 WFO 数据):
- Spearman 相关系数: 0.948
- 样本数: 12,597 个策略组合
- 特征数: 44 维

### 重训模型 (可选)

定期 (每季度) 重训模型以适应市场变化:

```bash
# 更新训练数据配置
vim configs/ranking_datasets.yaml  # 添加新的 WFO run 目录

# 重新训练
python run_ranking_pipeline.py --config configs/ranking_datasets.yaml

# 验证模型
python ml_ranker/robustness_eval.py
```

---

## 性能对比 (A/B 测试)

### Top-200 组合

| 指标 | WFO 排序 | ML 排序 | 提升 |
|------|---------|---------|------|
| 平均年化收益 | 11.20% | 19.06% | **+7.87%** (+70%) |
| 平均 Sharpe | 0.548 | 0.927 | **+0.379** (+69%) |
| 平均最大回撤 | -30.20% | -21.65% | **改善 8.56%** |

### Top-2000 组合

| 指标 | WFO 排序 | ML 排序 | 提升 |
|------|---------|---------|------|
| 平均年化收益 | 11.20% | 18.34% | **+7.13%** (+64%) |
| 平均 Sharpe | 0.534 | 0.905 | **+0.371** (+69%) |
| 平均最大回撤 | -30.08% | -20.97% | **改善 9.11%** |

**结论**: ML 排序在不同规模的组合池中均表现稳定优势, 适合生产环境。

---

## 故障排除

### 问题 1: 模型文件不存在

**症状**:
```
❌ ML模型不存在: ml_ranker/models/ltr_ranker
   ⚠️ 自动回退到 WFO 排序模式
```

**解决**:
```bash
# 训练模型
python run_ranking_pipeline.py --config configs/ranking_datasets.yaml

# 验证
ls -lh ml_ranker/models/ltr_ranker/
```

### 问题 2: 想临时使用 WFO 排序

**解决**: 修改配置文件 `ranking.method: "wfo"`, 无需卸载 ML 模型

### 问题 3: ML 排序失败

**排查步骤**:
1. 查看详细错误日志
2. 验证模型文件完整性: `ls ml_ranker/models/ltr_ranker/`
3. 测试 apply_ranker 独立运行: `python apply_ranker.py --model ml_ranker/models/ltr_ranker --wfo-dir results/run_latest --top-k 10`

---

## 后续维护

### 定期任务

- [ ] **每季度**: 重训 ML 模型 (新增换仓周期数据后)
- [ ] **每月**: 对比 ML vs WFO 排序效果 (运行 `analysis/compare_wfo_vs_ml.py`)
- [ ] **每周**: 检查模型文件完整性

### 参考文档

- 完整使用指南: `docs/ML_RANKING_INTEGRATION_GUIDE.md`
- 对比报告: `analysis/WFO_vs_ML_comparison_top2000_20251114.md`
- 实施总结: `docs/ML_RANKING_IMPLEMENTATION_SUMMARY.md`

---

**部署负责人**: GitHub Copilot  
**审核状态**: ✅ 已验证
