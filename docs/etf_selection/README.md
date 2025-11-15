<!-- ALLOW-MD -->
# Top-200 组合筛选工具

## 概述

从约 2000 个回测组合中，按「方案 C：强调风格与因子多样化的均衡型」筛选出 Top-200 组合，形成"精开发池"。

## 快速开始

### 最简单用法（使用默认配置）

```bash
python scripts/run_top200_selection.py \
    --input results_combo_wfo/.../top2000.csv \
    --output selection/top200_selected.csv
```

### 使用自定义配置文件

```bash
python scripts/run_top200_selection.py \
    --input data.csv \
    --output top200.csv \
    --config my_config.yaml
```

### 通过命令行覆盖关键参数

```bash
python scripts/run_top200_selection.py \
    --input data.csv \
    --output top200.csv \
    --min-sharpe-net 1.0 \
    --max-turnover 1.4 \
    --verbose
```

## 筛选流程

1. **数据预处理**：检查关键字段，剔除缺失行
2. **质量过滤**：按 Sharpe、回撤、收益、换手设定门槛（自适应调整）
3. **因子结构解析**：从 combo 字段解析因子列表，按关键词分类
4. **综合评分**：加权计算 selection_score
5. **分桶配额采样**：根据桶样本数动态分配名额
6. **combo_size 调整**：控制 3/4/5 因子的比例
7. **高换手控制**：限制高换手（>1.4）组合比例不超过 30%
8. **最终截断**：输出 Top-200

## 配置说明

### 默认配置

所有默认配置在 `selection/core.py` 的 `DEFAULT_CONFIG` 中定义。

**质量过滤阈值：**
- 标准：sharpe_net ≥ 0.95, max_dd_net ≥ -0.28, annual_ret_net ≥ 0.12, avg_turnover ≤ 1.6
- 放松（样本<300）：sharpe_net ≥ 0.90, max_dd_net ≥ -0.30, annual_ret_net ≥ 0.10, avg_turnover ≤ 1.8
- 收紧（样本>1500）：sharpe_net ≥ 1.0, avg_turnover ≤ 1.4

**因子分类关键词：**
- trend: MOM, SLOPE, VORTEX, ADX, TREND, ROC
- vol: VOL_RATIO, MAX_DD, RET_VOL, SHARPE, VAR, STD
- volume_price: OBV, PV_CORR, CMF, MFI
- relative: RSI, PRICE_POSITION, RELATIVE, CORRELATION, BETA

**综合评分权重：**
- annual_ret_net: 0.25
- sharpe_net: 0.30
- calmar_ratio: 0.20
- win_rate: 0.15
- max_dd_net: -0.10（回撤越小越好）

**combo_size 分布目标：**
- size=3: 40-60 个（20-30%）
- size=4: 60-80 个（30-40%）
- size=5: 70-90 个（35-45%）

**高换手控制：**
- 阈值：1.4
- 最大比例：30%

### 自定义配置文件

创建 YAML 文件（例如 `my_config.yaml`）：

```yaml
quality_filter:
  standard:
    min_sharpe_net: 1.0  # 覆盖默认 0.95
    max_turnover: 1.5    # 覆盖默认 1.6

scoring_weights:
  annual_ret_net: 0.3   # 更重视收益
  sharpe_net: 0.25

turnover_control:
  max_ratio: 0.25       # 高换手最多 25%
```

### 命令行参数覆盖

支持以下参数：
- `--min-sharpe-net`: 最低 Sharpe 比率
- `--max-dd-net`: 最大回撤
- `--min-annual-ret-net`: 最低年化收益
- `--max-turnover`: 最大换手率
- `--max-high-turnover-ratio`: 高换手最大比例

## 输出文件说明

输出 CSV 包含原始字段 + 以下新增字段：

- `dominant_factor`: 主导因子类别（trend/vol/volume_price/relative/mixed）
- `bucket`: 桶标识（格式：`{combo_size}_{dominant_factor}`）
- `selection_score`: 综合评分
- `final_rank`: 最终排名（1-200）

## 单组合分析

在 Python 中分析单个组合：

```python
from selection import analyze_single_combo
from selection.analyzer import format_combo_profile
import pandas as pd

# 加载结果
df = pd.read_csv('top200_selected.csv')

# 分析第 1 名组合
profile = analyze_single_combo(df, 1)  # 按 final_rank
print(format_combo_profile(profile))

# 或按 combo 字符串
profile = analyze_single_combo(df, "ADX_14D + CMF_20D + MAX_DD_60D")
```

## 在 Notebook 中使用

```python
from selection import select_top200, DEFAULT_CONFIG
import pandas as pd

# 加载数据
df = pd.read_csv('data/top2000.csv')

# 自定义配置
config = DEFAULT_CONFIG.copy()
config['quality_filter']['standard']['min_sharpe_net'] = 1.0

# 执行筛选
result = select_top200(df, config, verbose=True)

# 保存
result.to_csv('top200.csv', index=False)
```

## 依赖

- pandas >= 1.3.0
- pyyaml >= 5.4（可选，用于配置文件支持）

## 故障排除

### "数据缺少关键字段"错误

确保输入 CSV 包含以下字段：
- combo, combo_size, annual_ret_net, sharpe_net, max_dd_net, avg_turnover

### "未安装 pyyaml"警告

如需使用配置文件功能，安装：
```bash
pip install pyyaml
```

### 筛选结果少于 200 个

检查日志，可能原因：
1. 质量过滤后样本不足
2. 数据池中可用组合不足

解决方法：
- 放松质量阈值（通过配置文件或 CLI 参数）
- 检查输入数据质量

## 示例工作流

```bash
# 1. 筛选 Top-200
python scripts/run_top200_selection.py \
    --input results_combo_wfo/20251114_185809_20251114_185823/top2000_profit_backtest_slip2bps_*.csv \
    --output selection/top200_selected.csv \
    --verbose

# 2. 在 Python 中分析
python
>>> from selection import analyze_single_combo
>>> from selection.analyzer import format_combo_profile
>>> import pandas as pd
>>> df = pd.read_csv('selection/top200_selected.csv')
>>> profile = analyze_single_combo(df, 1)
>>> print(format_combo_profile(profile))
```

## 版本

- v1.0.0: 初始版本
