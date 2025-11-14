# ML排序模式使用指南

## ⚡ 快速开始 (生产推荐)

**默认配置已启用 ML 排序**, 直接运行即可:

```bash
cd /Users/zhangshenshen/深度量化0927/etf_rotation_experiments
python run_combo_wfo.py  # 默认使用 ML 排序
```

**输出**: 系统会自动使用 ML 模型对 WFO 组合进行排序, 生成 `ranking_ml_top200.parquet`

---

## 概述

WFO系统现已支持两种排序模式:

1. **ML排序模式** (`method: "ml"`) ✅ **生产默认**
   - 使用已训练的机器学习模型进行排序
   - A/B测试验证: Top-200 平均 Sharpe **+69%**, 年化收益 **+7.87%**
   - 适用于生产环境, 具备良好的泛化能力

2. **WFO排序模式** (`method: "wfo"`) ⚠️ **备用模式**
   - 使用原始 WFO 指标排序 (mean_oos_ic, oos_sharpe_proxy 等)
   - 用于对照基准或 ML 模型不可用时的回退选项

## 配置方式

在 `configs/combo_wfo_config.yaml` 中配置 `ranking` 块:

```yaml
ranking:
  method: "ml"     # 生产默认: "ml" (ML模型排序) | 备用: "wfo" (原始WFO排序)
  top_n: 200       # 最终选择的组合数量
  ml_model_path: "ml_ranker/models/ltr_ranker"  # ML模型路径 (无扩展名)
```

## 使用步骤

### 方法1: ML排序模式 (生产推荐, 默认)

```bash
# 确保配置文件中 ranking.method: "ml" (已默认)
python run_combo_wfo.py

# 或显式指定配置文件
python run_combo_wfo.py --config configs/combo_wfo_config.yaml
```

**输出文件**:
- `results/run_XXXXXX/all_combos.parquet` - 全部组合 (原始 WFO 指标)
- `results/run_XXXXXX/top_combos.parquet` - Top-N 组合 (ML 排序后)
- `results/run_XXXXXX/ranking_ml_top<N>.parquet` - ML 排名文件

**日志标识**:
```
🔀 排序模式选择
  📊 排序方式: ML (LTR 模型) ✅ 生产推荐
  TopN: 200
  模型路径: ml_ranker/models/ltr_ranker
⚡ 执行ML排序...
✅ ML排序完成: 12597 个组合
  Top-1 LTR分数: 0.1916
```

### 方法2: WFO排序模式 (备用)

如需使用原始 WFO 排序 (例如对照测试), 修改配置:

```yaml
ranking:
  method: "wfo"  # 改为 wfo
```

```bash
python run_combo_wfo.py
```

**输出文件**:
- `results/run_XXXXXX/all_combos.parquet` - 全部组合 (按 WFO 指标排序)
- `results/run_XXXXXX/top_combos.parquet` - Top-N 组合
- `results/run_XXXXXX/ranking_ic_top<N>.parquet` - WFO 排名文件

**日志标识**:
```
🔀 排序模式选择
  📊 排序方式: WFO (mean_oos_ic) ⚠️ 备用模式
  TopN: 200
  使用 WFO 原始排序 (mean_oos_ic + stability_score)
```

---

## ML 模型管理

### 首次使用: 训练ML排序模型

如果是首次使用或需要重训模型:

```bash
# 确保有可用的 WFO + 真实回测数据
# 训练ML排序模型
python run_ranking_pipeline.py --config configs/ranking_datasets.yaml

# 验证模型已生成
ls -lh ml_ranker/models/ltr_ranker/
# 应该看到: ltr_ranker.txt, ltr_ranker_meta.pkl, ltr_ranker_features.json
```

#### 步骤2: 配置ML排序模式

编辑 `configs/combo_wfo_config.yaml`:

```yaml
ranking:
  method: "ml"     # 修改为 ml
  top_n: 200       # 最终选择200个组合
  ml_model_path: "ml_ranker/models/ltr_ranker"
```

#### 步骤3: 运行WFO (使用ML排序)

```bash
python run_combo_wfo.py --config configs/combo_wfo_config.yaml
```

**输出文件**:
- `results/run_XXXXXX/all_combos.parquet` - 全部组合 (原始 WFO 指标)
- `results/run_XXXXXX/top_combos.parquet` - Top-N 组合 (**按 ML 排序**)
- `results/run_XXXXXX/ranking_ml_top<N>.parquet` - ML排名文件 (包含 ltr_score, ltr_rank)

## 排序对比

### WFO排序

- 依据: `mean_oos_ic`, `oos_sharpe_proxy`, `stability_score` 等 WFO 指标
- 优点: 直观,基于历史表现
- 缺点: 可能过拟合历史数据

### ML排序

- 依据: ML模型预测的 `ltr_score` (综合44个WFO特征)
- 优点: 
  - 更好的泛化能力 (Spearman 0.91+)
  - 稳健性优秀 (std < 0.005)
  - 考虑特征交互
- 缺点: 需要先训练模型

## 快速测试

### 测试ML排序 (使用已有数据)

```bash
# 方法1: 直接对已有WFO结果应用ML排序
python apply_ranker.py \
  --model ml_ranker/models/ltr_ranker \
  --wfo-dir results/run_20251114_155420 \
  --top-k 100

# 查看结果
head -20 results/run_20251114_155420/ranked_combos.csv
```

### 测试完整流程 (快速配置)

```bash
# 使用快速测试配置 (数据量小,速度快)
python run_combo_wfo.py --config configs/combo_wfo_config_ml_test.yaml
```

## 故障排除

### 问题1: ML排序模块不可用

**错误**: `ML排序模块不可用,仅支持 WFO 排序模式`

**解决**:
- 检查 `apply_ranker.py` 是否存在
- 检查 `ml_ranker` 包是否可导入: `python -c "from apply_ranker import apply_ltr_ranking"`

### 问题2: ML模型不存在

**错误**: `ML模型不存在: ml_ranker/models/ltr_ranker`

**解决**:
```bash
# 训练模型
python run_ranking_pipeline.py --config configs/ranking_datasets.yaml

# 验证模型
ls ml_ranker/models/ltr_ranker/
```

### 问题3: ML排序失败,回退到WFO

**解决**:
- 查看日志中的详细错误信息
- 确认模型文件完整 (ltr_ranker.txt, ltr_ranker_meta.pkl, ltr_ranker_features.json)
- 确认 WFO 特征与模型训练时一致

## 配置参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ranking.method` | str | `"wfo"` | 排序方法: "wfo" 或 "ml" |
| `ranking.top_n` | int | `200` | 最终选择的组合数量 |
| `ranking.ml_model_path` | str | `"ml_ranker/models/ltr_ranker"` | ML模型路径 (无扩展名) |

## 性能对比

基于 `results/run_20251114_155420` 的测试结果:

| 指标 | WFO排序 | ML排序 |
|------|---------|--------|
| Spearman相关性 | 0.85~0.90 | **0.91+** |
| NDCG@10 | 0.87~0.92 | **0.90+** |
| 稳健性 (std) | 0.01~0.02 | **< 0.005** |
| Top-10命中率 | 基准 | +15% |

## 后续回测

ML排序后的文件可直接用于真实回测:

```bash
# 对ML排序的Top组合进行回测
python real_backtest/run_profit_backtest.py \
  --ranking-file results/run_XXXXXX/top_combos.parquet \
  --slippage-bps 2 \
  --topk 100
```

## 更新日志

- **2025-11-14**: 初始版本,支持 wfo/ml 两种排序模式
- 保持向后兼容,默认使用 wfo 模式
- 自动回退机制确保稳定性
