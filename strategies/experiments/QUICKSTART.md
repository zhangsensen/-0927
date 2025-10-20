# 快速开始 - P0 阶段实验

## ⚡ 5分钟上手

### 1. 验证环境

```bash
cd /Users/zhangshenshen/深度量化0927
bash strategies/experiments/verify_setup.sh
```

### 2. 运行第一个实验（Dry Run）

```bash
# 仅打印命令，不实际执行
python strategies/experiments/run_experiments.py \
    --config strategies/experiments/experiment_configs/p0_weight_grid_coarse.yaml \
    --dry-run
```

### 3. 运行真实实验（粗网格扫描）

```bash
# 预计耗时：5-10分钟
python strategies/experiments/run_experiments.py \
    --config strategies/experiments/experiment_configs/p0_weight_grid_coarse.yaml
```

**输出位置**: `strategies/results/experiments/p0_coarse/run_YYYYMMDD_HHMMSS/`

---

## 📊 P0 完整流程

### Step 1: 粗网格扫描（5-10分钟）

```bash
python strategies/experiments/run_experiments.py \
    --config strategies/experiments/experiment_configs/p0_weight_grid_coarse.yaml
```

**目标**: 快速定位高夏普区域  
**参数**: 权重0.2步长，Top-N=8，费率=0.0028

---

### Step 2: 精细网格扫描（10-20分钟）

```bash
python strategies/experiments/run_experiments.py \
    --config strategies/experiments/experiment_configs/p0_weight_grid_fine.yaml
```

**目标**: 精确定位最优权重  
**参数**: 权重0.1步长，Top-N=8，费率=0.0028

---

### Step 3: Top-N 扫描（15-30分钟）

```bash
python strategies/experiments/run_experiments.py \
    --config strategies/experiments/experiment_configs/p0_topn_scan.yaml
```

**目标**: 测试不同持仓数量  
**参数**: Top-N=[6,8,10,12,15]

---

### Step 4: 成本敏感性分析（20-40分钟）

```bash
python strategies/experiments/run_experiments.py \
    --config strategies/experiments/experiment_configs/p0_cost_sensitivity.yaml
```

**目标**: 测试不同费率影响  
**参数**: 费率=[0.001, 0.002, 0.0028, 0.003, 0.004, 0.005]

---

## 📈 结果分析

### 聚合所有 P0 结果

```bash
python strategies/experiments/aggregate_results.py \
    --pattern "strategies/results/experiments/p0_*/run_*/results.csv" \
    --output strategies/results/experiments/p0_summary.csv \
    --top-n 100 \
    --plot
```

**输出**:
- `p0_summary.csv` - Top-100 策略
- `p0_summary_summary.csv` - 汇总统计
- `plots/p0_summary_sharpe_topn.png` - 夏普-TopN 热力图
- `plots/p0_summary_sharpe_fee.png` - 夏普-费率曲线

---

### 查看最优策略

```bash
python -c "
import pandas as pd
df = pd.read_csv('strategies/results/experiments/p0_summary.csv')
print('🏆 Top-10 策略:')
print(df.head(10)[['weights', 'top_n', 'fee', 'sharpe', 'annual_return', 'max_drawdown']])
"
```

---

## 🔧 常用命令

### 运行所有 P0 实验

```bash
python strategies/experiments/run_experiments.py \
    --pattern "p0_*.yaml" \
    --config-dir strategies/experiments/experiment_configs
```

### 查看实验日志

```bash
# 列出所有日志
ls -lh strategies/results/experiments/experiment_log_*.json

# 查看最新日志
cat $(ls -t strategies/results/experiments/experiment_log_*.json | head -1) | jq .
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

## 🎯 预期结果

### P0 阶段目标

- ✅ 完成 4 个实验配置
- ✅ 生成 Top-100 策略榜单
- ✅ 识别最优权重组合
- ✅ 确定最佳 Top-N 值
- ✅ 评估成本敏感性

### 关键指标

| 指标 | 目标值 |
|------|--------|
| 夏普比率 | > 0.4 |
| 年化收益 | > 6% |
| 最大回撤 | < 35% |
| 换手率 | < 50 |

---

## 🐛 常见问题

### Q1: 找不到因子面板文件

**错误**: `FileNotFoundError: production_factor_results/...`

**解决**:
```bash
# 检查文件是否存在
ls -lh production_factor_results/factor_screen_f5_*.json

# 或修改配置文件中的路径
vim strategies/experiments/experiment_configs/p0_weight_grid_coarse.yaml
```

---

### Q2: 内存不足

**错误**: `MemoryError` 或系统卡死

**解决**:
```yaml
# 在配置文件中降低组合数
parameters:
  max-total-combos: 10000  # 从 50000 降低到 10000
  max-active-factors: 4    # 从 6 降低到 4
```

---

### Q3: 实验运行太慢

**优化**:
```yaml
# 启用多进程（谨慎使用）
parameters:
  num-workers: 4  # M4 Pro 可用 4 个进程
```

---

## 📚 下一步

完成 P0 后，进入 P1 阶段：

1. **动态权重调整** - 基于波动率/ATR 调整因子权重
2. **Regime 分类器** - 牛市/熊市/震荡市场状态识别
3. **动态 Top-N** - 根据市场状态调整持仓数量

详见: [P1 开发计划](../ETF_ROTATION_GOLDEN_RHYTHM.md)

---

**最后更新**: 2025-10-17  
**维护人**: Linus-Style 量化工程助手
