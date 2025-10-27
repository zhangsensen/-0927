# WFO实验系统使用指南

## 🎯 核心理念

**你的理解完全正确！**

```
横截面 + 因子筛选 → 建设1次（除非改因子公式/筛选逻辑）
                  ↓
                缓存到本地（100x加速）
                  ↓
         WFO参数实验 → 频繁测试（50+次）
```

### 性能对比

| 步骤 | 首次运行 | 使用缓存 | 加速比 |
|------|---------|---------|--------|
| 因子计算 | 11秒 | 0.1秒 | **100x** |
| 因子标准化 | 6秒 | 0.1秒 | **60x** |
| WFO回测 | 0.3秒 | 0.3秒 | 1x |
| **总计** | **18秒** | **7秒** | **2.6x** |

---

## 📁 目录结构

```
etf_rotation_optimized/
├── scripts/
│   ├── production_backtest.py      # 完整回测（Step 1-6）
│   ├── wfo_experiments.py          # 批量WFO实验
│   └── compare_runs.py             # 结果对比工具
│
├── configs/
│   ├── wfo_grid.yaml              # 参数网格配置
│   └── wfo_grid_test.yaml         # 测试用配置
│
├── utils/
│   └── factor_cache.py            # 智能缓存系统
│
├── cache/
│   └── factor_engine/             # 因子缓存目录
│       ├── raw_*.parquet          # 原始因子缓存
│       └── standardized_*.parquet # 标准化因子缓存
│
└── results/
    ├── 20251025_154432/           # 完整回测结果
    │   ├── wfo_results.csv
    │   ├── top100_portfolios.csv
    │   └── metadata.json
    │
    └── wfo_experiments_20251025_161617/  # 批量实验结果
        ├── experiments_summary.csv        # 汇总文件 ⭐
        ├── experiments_summary.json
        ├── exp_001_wfo_results.csv       # 实验1详细结果
        └── exp_002_wfo_results.csv       # 实验2详细结果
```

---

## 🚀 快速开始

### 1️⃣ 首次完整回测（建立因子缓存）

```bash
# 执行完整流程：加载数据 → 计算因子 → 标准化 → WFO → 组合优化
python scripts/production_backtest.py
```

**首次运行时间**: ~18秒
- ✅ 数据验证: 2020-2025, 前复权价格
- ✅ 10个因子计算 + 标准化
- ✅ 55个WFO窗口
- ✅ 1000个组合测试，保存TOP 100
- ✅ **自动缓存因子到本地**

### 2️⃣ 批量WFO实验（使用缓存）

```bash
# 快速测试（16个实验）
python scripts/wfo_experiments.py --grid basic_grid

# 详细测试（81个实验）
python scripts/wfo_experiments.py --grid full_grid

# 保守策略（1个实验）
python scripts/wfo_experiments.py --grid conservative

# 激进策略（1个实验）
python scripts/wfo_experiments.py --grid aggressive
```

**后续运行时间**: ~7秒/次
- ✅ 因子从缓存加载（0.1秒）
- ✅ 只重新运行WFO部分
- ✅ 自动保存所有实验结果

### 3️⃣ 结果对比

```bash
# 对比两次回测结果
python scripts/compare_runs.py \
  results/20251025_154432 \
  results/20251025_160000
```

**输出示例**:
```
================================================================================
WFO结果对比
================================================================================

基本信息对比
--------------------------------------------------------------------------------
指标                                   Run 1                Run 2
--------------------------------------------------------------------------------
窗口数                                    55                   55
样本内天数                                252                  504
样本外天数                                 60                  120

IC统计对比
--------------------------------------------------------------------------------
指标                                   Run 1                Run 2                差值
--------------------------------------------------------------------------------
平均 OOS IC                           0.1826               0.1903              +0.0077
OOS IC 标准差                         0.0421               0.0389              -0.0032
平均 IC 衰减                          0.0032               0.0028              -0.0004

结论
--------------------------------------------------------------------------------
✅ Run 2 显著更优 (IC提升 +0.0077)
================================================================================
```

---

## ⚙️ 参数配置

### 配置文件: `configs/wfo_grid.yaml`

```yaml
# 基础网格（2×2×2×2 = 16个实验）
basic_grid:
  is_period: [252, 504]          # 样本内: 1年, 2年
  oos_period: [60, 120]          # 样本外: 3月, 6月
  step_size: [20, 40]            # 滚动步长
  target_factor_count: [5, 8]    # 因子数

# 完整网格（3×3×3×3 = 81个实验）
full_grid:
  is_period: [126, 252, 504]
  oos_period: [30, 60, 120]
  step_size: [10, 20, 40]
  target_factor_count: [3, 5, 8]

# 焦点测试 - IS周期影响
test_is_period:
  is_period: [126, 189, 252, 315, 378, 441, 504]  # 0.5~2年
  oos_period: [60]                                 # 固定
  step_size: [20]
  target_factor_count: [5]

# 焦点测试 - 因子数影响
test_factor_count:
  is_period: [252]
  oos_period: [60]
  step_size: [20]
  target_factor_count: [2, 3, 4, 5, 6, 7, 8, 9, 10]
```

### 自定义配置

创建新配置文件 `configs/my_grid.yaml`:

```yaml
my_custom_grid:
  is_period: [180, 360]
  oos_period: [45, 90]
  step_size: [15, 30]
  target_factor_count: [4, 6]
```

运行:
```bash
python scripts/wfo_experiments.py \
  --config configs/my_grid.yaml \
  --grid my_custom_grid
```

---

## 📊 实验结果分析

### 汇总文件: `experiments_summary.csv`

```csv
experiment,is_period,oos_period,step_size,target_factor_count,num_windows,avg_oos_ic,avg_ic_drop,top_factor,top_factor_freq,result_file
exp_001,252,60,20,5,55,0.1826,0.0032,PRICE_POSITION_20D,0.98,exp_001_wfo_results.csv
exp_002,252,60,20,8,55,0.1745,0.0024,PRICE_POSITION_20D,0.98,exp_002_wfo_results.csv
exp_003,252,120,20,5,55,0.1903,0.0028,MOM_20D,0.95,exp_003_wfo_results.csv
```

**关键指标**:
- `avg_oos_ic`: 平均样本外IC（越高越好）
- `avg_ic_drop`: 平均IC衰减（越小越好）
- `top_factor`: 选中频率最高的因子
- `top_factor_freq`: 选中频率（接近1.0说明因子稳定）

### Python分析示例

```python
import pandas as pd

# 加载汇总结果
summary = pd.read_csv("results/wfo_experiments_20251025_161617/experiments_summary.csv")

# 按OOS IC排序
top10 = summary.nlargest(10, 'avg_oos_ic')
print(top10[['experiment', 'is_period', 'oos_period', 'avg_oos_ic']])

# 分析IS周期影响（固定其他参数）
is_impact = summary[
    (summary['oos_period'] == 60) & 
    (summary['step_size'] == 20) & 
    (summary['target_factor_count'] == 5)
].sort_values('is_period')

print(is_impact[['is_period', 'avg_oos_ic', 'avg_ic_drop']])

# 分析因子数影响
factor_impact = summary[
    (summary['is_period'] == 252) & 
    (summary['oos_period'] == 60) & 
    (summary['step_size'] == 20)
].sort_values('target_factor_count')

print(factor_impact[['target_factor_count', 'avg_oos_ic']])
```

---

## 🔧 缓存管理

### 缓存机制

系统使用**哈希验证**自动管理缓存:

```python
cache_key = f"{stage}_{data_hash}_{code_hash}.parquet"
```

- `data_hash`: 数据形状 + 最后一行 + 最后日期
- `code_hash`: 因子库源代码MD5

**自动失效条件**:
1. 数据变化（新增ETF、日期更新）
2. 因子公式修改
3. 缓存超过7天（可配置）

### 手动清理缓存

```bash
# 清理所有缓存
rm -rf cache/factor_engine/*.parquet

# 清理特定类型
rm cache/factor_engine/raw_*.parquet          # 只清理原始因子
rm cache/factor_engine/standardized_*.parquet # 只清理标准化因子
```

### 缓存统计

```python
from utils.factor_cache import FactorCache

cache = FactorCache()
stats = cache.get_cache_stats()

print(f"缓存文件数: {stats['file_count']}")
print(f"总大小: {stats['total_size_mb']:.2f} MB")
print(f"最老文件: {stats['oldest_file']}")
```

---

## 🎓 高级用法

### 1. 单独运行某个实验

```python
from pathlib import Path
from scripts.wfo_experiments import WFOExperiments
from scripts.production_backtest import ProductionBacktest

# 加载数据
backtest = ProductionBacktest(output_base_dir=Path("results"))
backtest.load_data()

# 创建实验
exp = WFOExperiments(
    ohlcv=backtest.ohlcv,
    output_dir=Path("results/my_experiment")
)

# 运行单个实验
result = exp.run_single_experiment(
    exp_name="test_1year_3month",
    is_period=252,
    oos_period=60,
    step_size=20,
    target_factor_count=5
)

print(f"OOS IC: {result['avg_oos_ic']:.4f}")
```

### 2. 批量对比多个实验

```bash
# 生成对比报告
for dir in results/wfo_experiments_*/; do
    echo "=== $(basename $dir) ==="
    cat "$dir/experiments_summary.csv" | head -n 2
done
```

### 3. 可视化参数影响

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 加载结果
summary = pd.read_csv("experiments_summary.csv")

# 热力图：IS vs OOS周期
pivot = summary.pivot_table(
    values='avg_oos_ic',
    index='is_period',
    columns='oos_period',
    aggfunc='mean'
)

plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlGnBu')
plt.title('OOS IC Heatmap: IS Period vs OOS Period')
plt.savefig('results/ic_heatmap.png', dpi=300)
```

---

## ✅ 完整工作流示例

```bash
# 第1天：首次建设
python scripts/production_backtest.py
# 耗时: 18秒
# 输出: results/20251025_154432/
#      - wfo_results.csv
#      - top100_portfolios.csv
#      - 10个因子缓存

# 第2天：快速测试16个参数组合
python scripts/wfo_experiments.py --grid basic_grid
# 耗时: 7秒 × 16 = 112秒（因子已缓存）
# 输出: results/wfo_experiments_20251026_xxx/
#      - experiments_summary.csv (TOP 16参数)
#      - exp_001~016_wfo_results.csv

# 第3天：详细测试81个组合
python scripts/wfo_experiments.py --grid full_grid
# 耗时: 7秒 × 81 = 567秒 ≈ 9.5分钟
# 输出: 81个实验的完整结果

# 第4天：焦点测试IS周期影响
python scripts/wfo_experiments.py --grid test_is_period
# 耗时: 7秒 × 7 = 49秒
# 输出: IS周期从0.5年到2年的影响曲线

# 第5天：对比最优vs基准
python scripts/compare_runs.py \
  results/wfo_experiments_20251026_xxx \
  results/20251025_154432
```

---

## 💡 最佳实践

### DO ✅

1. **首次运行**: 用 `production_backtest.py` 验证数据和建立缓存
2. **参数调优**: 先用 `basic_grid` 快速测试，再用 `full_grid` 精细搜索
3. **焦点测试**: 固定3个参数，只变1个参数，分析单一影响
4. **结果对比**: 用 `compare_runs.py` 对比实验结果
5. **缓存清理**: 数据更新后手动删除缓存
6. **版本管理**: 提交代码前记录最优参数组合

### DON'T ❌

1. ❌ 不要频繁修改因子公式（会导致缓存失效）
2. ❌ 不要同时运行多个实验（会重复计算因子）
3. ❌ 不要忽略缓存失效警告
4. ❌ 不要删除 `metadata.json`（记录了参数和Git版本）
5. ❌ 不要在实验中混用不同数据集
6. ❌ 不要过度拟合参数（警惕OOS IC过高）

---

## 🔍 故障排查

### Q1: 缓存不生效，每次都重新计算？

**检查**:
```bash
ls -lh cache/factor_engine/
```

**原因**:
- 数据更新了（时间范围变化）
- 因子公式改了
- 缓存超过7天

**解决**:
- 正常现象，自动重新缓存
- 如果需要强制使用旧缓存：`FactorCache(ttl_days=30)`

### Q2: 实验结果不一致？

**检查**:
```bash
cat results/*/metadata.json | grep git_commit
```

**原因**:
- 代码版本不同
- 数据版本不同
- 随机种子未固定

**解决**:
- 对比 `metadata.json` 中的 `git_commit`
- 确保数据时间范围一致
- 固定随机种子（如有用到）

### Q3: 内存不足？

**症状**: `MemoryError` 或进程被杀

**解决**:
```python
# 方法1: 减少并行实验数（逐个运行）
# 方法2: 减少因子数
# 方法3: 缩短时间范围

# 监控内存
import psutil
print(f"Memory: {psutil.virtual_memory().percent}%")
```

---

## 📈 性能优化

### 当前性能（43 ETF × 1399天）

| 操作 | 时间 | 优化空间 |
|------|------|---------|
| 数据加载 | 0.7秒 | ⚪ 无需优化 |
| 因子计算（首次） | 11秒 | ✅ 缓存后0.1秒 |
| 因子标准化（首次） | 6秒 | ✅ 缓存后0.1秒 |
| WFO单次 | 0.3秒 | ⚪ 已优化 |
| 组合优化 | 0.7秒 | ⚪ 已优化 |

### 100个实验的总时间

- **无缓存**: 18秒 × 100 = 30分钟
- **有缓存**: 7秒 × 100 = 11.7分钟
- **加速比**: **2.6x**

---

## 🎯 总结

```
原始方案（被拒绝）: 500+ 行代码，阶段分离，血缘追踪 ❌
实际方案（已实现）: 100 行代码，智能缓存，参数化实验 ✅

核心理念: Cache what's slow, parameterize what varies

结果:
  ✅ 因子只算1次（11秒）
  ✅ WFO测试N次（0.3秒/次）
  ✅ 100x缓存加速
  ✅ 支持81种参数组合
  ✅ 自动保存所有结果
  ✅ 对比工具
  ✅ YAML配置管理
```

**Linus会满意的实现**: 简单、高效、实用！🚀
