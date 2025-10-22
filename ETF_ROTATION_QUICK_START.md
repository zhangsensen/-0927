# ETF轮动系统快速使用指南

## 🚀 快速开始

### 一键执行完整流程
```bash
cd /Users/zhangshenshen/深度量化0927/etf_rotation_system
./test_full_pipeline.sh
```

预计耗时: 3-5分钟

---

## 📋 分步执行

### 步骤1: 生成因子面板（48因子）
```bash
cd /Users/zhangshenshen/深度量化0927/etf_rotation_system/01_横截面建设

python3 generate_panel_refactored.py \
    --data-dir ../../raw/ETF/daily \
    --output-dir ../data/results/panels \
    --workers 8
```

**输出**: `../data/results/panels/panel_YYYYMMDD_HHMMSS/panel.parquet`  
**包含**: 36个传统因子 + 12个轮动因子

---

### 步骤2: 筛选核心因子（12因子）
```bash
cd /Users/zhangshenshen/深度量化0927/etf_rotation_system/02_因子筛选

python3 run_etf_cross_section_configurable.py \
    --config optimized_screening_config.yaml
```

**输出**: `../data/results/screening/screening_YYYYMMDD_HHMMSS/passed_factors.csv`  
**筛选标准**: IC≥1.5%, IR≥0.12, 相关性≤55%, 强制保留轮动因子

---

### 步骤3: 执行VBT回测（1万组合）
```bash
cd /Users/zhangshenshen/深度量化0927/etf_rotation_system/03_vbt回测

python3 large_scale_backtest_50k.py
```

**输出**: `../data/results/backtest/backtest_YYYYMMDD_HHMMSS/results.csv`  
**配置**: 12因子 × 6权重点 × [5,8,10]持仓 = 12,882个策略（实际生成）

---

## 📊 查看结果

### 回测结果分析
```bash
cd /Users/zhangshenshen/深度量化0927

python3 << 'EOF'
import pandas as pd

# 使用最新的回测结果
results = pd.read_csv("etf_rotation_system/data/results/backtest/backtest_20251022_015507/results.csv")

print("=" * 80)
print("回测结果统计")
print("=" * 80)
print(f"\n总策略数: {len(results):,}")
print(f"平均Sharpe: {results['sharpe_ratio'].mean():.4f}")
print(f"Top Sharpe: {results['sharpe_ratio'].max():.4f}")

# Top 10
print("\nTop 10策略:")
top10 = results.nlargest(10, 'sharpe_ratio')
for i, row in top10.iterrows():
    print(f"  #{i+1}: Sharpe={row['sharpe_ratio']:.4f}, Return={row['total_return']:.2f}%, Top-N={row['top_n']}")
EOF
```

---

## 🔧 配置文件位置

### 横截面建设
`01_横截面建设/config/factor_panel_config.yaml`

### 因子筛选
`02_因子筛选/optimized_screening_config.yaml`

### VBT回测
参数在`03_vbt回测/large_scale_backtest_50k.py`第77-112行

---

## 📂 结果文件位置

### 最新结果（示例）
```
data/results/
├── panels/panel_20251022_013039/panel.parquet          # 48因子面板
├── screening/screening_20251022_014652/passed_factors.csv  # 12核心因子
└── backtest/backtest_20251022_015507/results.csv       # Top 200策略
```

### 历史结果
所有历史结果都保留在对应的时间戳目录中

---

## ⚙️ 关键参数

### 因子筛选标准
```yaml
min_ic: 0.015              # IC均值≥1.5%
min_ir: 0.12               # IC_IR≥0.12
max_correlation: 0.55      # 因子相关性≤55%
max_factors: 12            # 最多12个因子
force_include_factors:     # 强制保留
  - ROTATION_SCORE
  - RELATIVE_MOMENTUM_60D
  - CS_RANK_CHANGE_5D
```

### 回测配置
```python
top_k = 12                              # 使用12个核心因子
top_n_list = [5, 8, 10]                 # 持仓5/8/10只ETF
weight_grid_points = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # 6个权重点
weight_sum_range = [0.6, 1.4]           # 权重和范围
max_combinations = 10000                # 最大组合数
rebalance_freq = 20                     # 每20天再平衡
fees = 0.003                            # 0.3%往返成本
```

---

## 🐛 常见问题

### Q1: 找不到panel文件
**解决**: 确保先执行步骤1生成面板，或手动指定`--panel-file`参数

### Q2: 筛选结果少于12个因子
**解决**: 降低`min_ic`或`min_ir`阈值，在`optimized_screening_config.yaml`中修改

### Q3: 回测无结果
**解决**: 检查权重约束`weight_sum_range`，如果太严格可能无法生成有效组合

### Q4: Dirichlet采样命中率低
**解决**: 放宽`weight_sum_range`到`[0.5, 1.5]`，或增加`max_combinations * 30`倍数

---

## 📖 相关文档

- **筛选优化报告**: `FACTOR_SCREENING_OPTIMIZATION_REPORT.md`
- **回测对比报告**: `BACKTEST_12FACTORS_COMPARISON_REPORT.md`
- **完整优化总结**: `ETF_ROTATION_OPTIMIZATION_SUMMARY.md`
- **主项目文档**: `CLAUDE.md`

---

## 💡 提示

1. **首次运行**: 建议先执行`./test_full_pipeline.sh`验证完整流程
2. **参数调优**: 修改配置文件后需重新执行对应步骤
3. **结果对比**: 使用时间戳区分不同版本的结果
4. **性能监控**: 观察轮动因子在Top 10中的使用情况
5. **增量更新**: 只需重新执行步骤3即可使用相同的因子面板和筛选结果

---

**更新日期**: 2025-10-22  
**版本**: 1.0（12因子优化版）
