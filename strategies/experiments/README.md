# Experiments - 实验管线

## 📁 目录结构

```
experiments/
├── experiment_configs/          # YAML 配置文件
│   ├── p0_weight_grid_coarse.yaml
│   ├── p0_weight_grid_fine.yaml
│   ├── p0_topn_scan.yaml
│   └── p0_cost_sensitivity.yaml
├── run_experiments.py           # 实验运行器
├── aggregate_results.py         # 结果聚合工具
└── README.md                    # 本文档
```

---

## 🚀 快速开始

### 1. 运行单个实验

```bash
# P0 阶段 - 粗网格扫描
python strategies/experiments/run_experiments.py \
    --config strategies/experiments/experiment_configs/p0_weight_grid_coarse.yaml
```

### 2. 运行所有 P0 实验

```bash
python strategies/experiments/run_experiments.py \
    --pattern "p0_*.yaml"
```

### 3. 聚合实验结果

```bash
# 聚合所有 P0 实验结果
python strategies/experiments/aggregate_results.py \
    --pattern "strategies/results/experiments/p0_*/run_*/results.csv" \
    --output strategies/results/experiments/p0_summary.csv \
    --top-n 100 \
    --plot
```

---

## 📋 P0 阶段实验清单

### Phase 1: 粗网格权重扫描
**配置**: `p0_weight_grid_coarse.yaml`  
**目标**: 权重0.2步长粗扫，快速定位高夏普区域  
**参数**:
- 权重网格: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
- Top-N: [8]
- 费率: [0.0028]
- 最大组合数: 50,000

**运行命令**:
```bash
python strategies/experiments/run_experiments.py \
    --config strategies/experiments/experiment_configs/p0_weight_grid_coarse.yaml
```

---

### Phase 2: 精细网格权重扫描
**配置**: `p0_weight_grid_fine.yaml`  
**目标**: 权重0.1步长精扫，精确定位最优权重  
**参数**:
- 权重网格: [0.0, 0.1, 0.2, ..., 1.0]
- Top-N: [8]
- 费率: [0.0028]
- 最大组合数: 50,000

**运行命令**:
```bash
python strategies/experiments/run_experiments.py \
    --config strategies/experiments/experiment_configs/p0_weight_grid_fine.yaml
```

---

### Phase 3: Top-N 扫描
**配置**: `p0_topn_scan.yaml`  
**目标**: 测试不同持仓数量对策略表现的影响  
**参数**:
- 权重网格: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
- Top-N: [6, 8, 10, 12, 15]
- 费率: [0.0028]
- 最大组合数: 50,000

**运行命令**:
```bash
python strategies/experiments/run_experiments.py \
    --config strategies/experiments/experiment_configs/p0_topn_scan.yaml
```

---

### Phase 4: 成本敏感性分析
**配置**: `p0_cost_sensitivity.yaml`  
**目标**: 测试不同交易费率对策略表现的影响  
**参数**:
- 权重网格: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
- Top-N: [8]
- 费率: [0.001, 0.002, 0.0028, 0.003, 0.004, 0.005]
- 最大组合数: 50,000

**运行命令**:
```bash
python strategies/experiments/run_experiments.py \
    --config strategies/experiments/experiment_configs/p0_cost_sensitivity.yaml
```

---

## 📊 结果分析

### 查看实验日志

```bash
# 查看最新实验日志
ls -lh strategies/results/experiments/experiment_log_*.json

# 查看日志内容
cat strategies/results/experiments/experiment_log_20251017_*.json | jq .
```

### 聚合并生成报表

```bash
# 聚合所有 P0 结果，生成 Top-100 策略
python strategies/experiments/aggregate_results.py \
    --pattern "strategies/results/experiments/p0_*/run_*/results.csv" \
    --output strategies/results/experiments/p0_summary.csv \
    --top-n 100 \
    --plot

# 查看汇总统计
cat strategies/results/experiments/p0_summary_summary.csv
```

### 可视化图表

聚合工具会自动生成：
- `p0_summary_sharpe_topn.png` - 夏普-TopN 热力图
- `p0_summary_sharpe_fee.png` - 夏普-费率曲线

---

## 🔧 高级用法

### Dry Run（仅打印命令）

```bash
python strategies/experiments/run_experiments.py \
    --pattern "p0_*.yaml" \
    --dry-run
```

### 对比历史最优

```bash
python strategies/experiments/aggregate_results.py \
    --pattern "strategies/results/experiments/p0_*/run_*/results.csv" \
    --output strategies/results/experiments/p0_summary_new.csv \
    --history strategies/results/experiments/p0_summary_old.csv \
    --top-n 100
```

---

## 📝 YAML 配置说明

### 配置文件结构

```yaml
experiment:
  name: "实验名称"
  description: "实验描述"
  phase: "P0/P1/P2"
  version: "1.0"

parameters:
  # 因子选择
  top-factors-json: "production_factor_results/factor_screen_f5_*.json"
  top-k: 10
  
  # 权重网格
  weight-grid: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
  max-active-factors: 6
  
  # Top-N 选股
  top-n-list: [8]
  min-score-list: [null]
  
  # 费率
  fees: [0.0028]
  
  # 回测参数
  init-cash: 1000000.0
  freq: "1D"
  norm-method: "zscore"
  
  # 执行控制
  max-total-combos: 50000
  batch-size: 10000
  num-workers: 1
  
  # 输出控制
  top-k-results: 100
  output: "results/experiments/p0_test"
```

### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `weight-grid` | List[float] | 权重候选值列表，支持任意浮点数 |
| `top-n-list` | List[int] | Top-N 候选值列表 |
| `fees` | List[float] | 费率候选值列表（支持成本敏感性分析） |
| `max-total-combos` | int | 最大组合数限制 |
| `max-active-factors` | int | 最大非零因子数量 |
| `top-k-results` | int | 仅保留夏普最高的前K个结果 |

---

## 🎯 最佳实践

### 1. 分阶段执行
- **P0**: 粗扫 → 精扫 → Top-N 扫描 → 成本敏感性
- **P1**: 动态权重 → Regime 策略
- **P2**: 多策略组合 → 风控模块

### 2. 资源控制
- 单轮组合数 ≤ 50,000
- 使用 `--top-k-results` 控制输出规模
- 大规模实验使用 `--batch-size` 分批执行

### 3. 结果管理
- 每次实验自动生成时间戳目录
- 保留实验日志（JSON + CSV）
- 定期聚合结果，生成总榜

---

## 🐛 故障排查

### 问题1: 组合数爆炸

**症状**: 组合数超过 50,000，内存不足

**解决**:
```yaml
parameters:
  max-total-combos: 50000  # 限制总组合数
  max-active-factors: 6    # 限制非零因子数
```

### 问题2: 实验运行失败

**检查**:
1. 配置文件路径是否正确
2. 因子面板文件是否存在
3. 输出目录是否有写权限

**调试**:
```bash
# 使用 dry-run 模式检查命令
python strategies/experiments/run_experiments.py \
    --config xxx.yaml \
    --dry-run
```

---

## 📚 相关文档

- [vectorbt_multifactor_grid.py](../vectorbt_multifactor_grid.py) - 回测引擎
- [ETF_ROTATION_GOLDEN_RHYTHM.md](../ETF_ROTATION_GOLDEN_RHYTHM.md) - 策略文档
- [README.md](../README.md) - Strategies 目录说明

---

**最后更新**: 2025-10-17  
**维护人**: Linus-Style 量化工程助手
