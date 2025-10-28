# 分步执行脚本使用说明

## 📋 概述

现在有4个独立的执行脚本，每个都有详细的过程日志：

1. **step1_cross_section.py** - 横截面建设
2. **step2_factor_selection.py** - 因子筛选（标准化）
3. **step3_run_wfo.py** - WFO优化
4. **run_all_steps.py** - 一次性执行所有步骤（可选）

## 🚀 使用方式

### 方式1：分步执行（推荐用于调试）

```bash
cd /Users/zhangshenshen/深度量化0927/etf_rotation_optimized

# Step 1: 构建横截面
python scripts/step1_cross_section.py

# Step 2: 因子筛选
python scripts/step2_factor_selection.py

# Step 3: WFO优化
python scripts/step3_run_wfo.py
```

### 方式2：一次性执行（自动传递数据目录）

```bash
cd /Users/zhangshenshen/深度量化0927/etf_rotation_optimized

python scripts/run_all_steps.py
```

## 📂 输出目录结构

```
results/
├── cross_section/
│   └── 20251025/
│       └── 20251025_201234/
│           ├── ohlcv/
│           │   ├── open.parquet
│           │   ├── high.parquet
│           │   ├── low.parquet
│           │   ├── close.parquet
│           │   └── volume.parquet
│           ├── factors/
│           │   ├── RSI_14.parquet
│           │   ├── MOM_20D.parquet
│           │   └── ...（10个因子）
│           ├── metadata.json
│           └── step1_cross_section.log
│
├── factor_selection/
│   └── 20251025/
│       └── 20251025_201245/
│           ├── standardized/
│           │   ├── RSI_14.parquet
│           │   ├── MOM_20D.parquet
│           │   └── ...（10个因子）
│           ├── metadata.json
│           └── step2_factor_selection.log
│
└── wfo/
    └── 20251025_201256/
        ├── wfo_results.pkl
        ├── wfo_report.txt
        ├── metadata.json
        └── step3_wfo.log
```

## 📊 日志内容

### Step 1 日志示例
```
[20:12:34] INFO - 阶段 1/4: 加载ETF数据
[20:12:35] INFO - ✅ 数据加载完成
[20:12:35] INFO -    日期: 1399 天
[20:12:35] INFO -    标的: 43 只
[20:12:35] INFO - 阶段 2/4: 计算精确因子
[20:12:36] INFO - ✅ 使用因子缓存（跳过计算）
[20:12:36] INFO - 因子数量: 10
[20:12:36] INFO -   01. RSI_14                     NaN率: 1.00%
...
```

### Step 2 日志示例
```
[20:12:45] INFO - 阶段 1/4: 加载横截面数据
[20:12:45] INFO - ✅ open.parquet: (1399, 43)
[20:12:46] INFO - 阶段 2/4: 因子标准化
[20:12:46] INFO - ✅ 使用标准化因子缓存
[20:12:46] INFO - 📊 标准化验证:
[20:12:46] INFO -   RSI_14                     均值=-0.0000  标准差= 1.0000  NaN率= 1.00%
...
```

### Step 3 日志示例
```
[20:12:56] INFO - 阶段 1/3: 加载标准化因子
[20:12:57] INFO - 阶段 2/3: WFO优化
[20:12:57] INFO - WFO参数配置:
[20:12:57] INFO -   样本内窗口: 252 天
[20:12:57] INFO -   样本外窗口: 60 天
[20:12:57] INFO -   滑动步长: 20 天
[20:12:58] INFO - 🔄 开始WFO优化...
[20:15:23] INFO - ✅ WFO优化完成（耗时 145.2秒）
[20:15:23] INFO - 📊 WFO结果统计:
[20:15:23] INFO -   窗口总数: 55
[20:15:23] INFO -   平均OOS IC: 0.1826
...
```

## 🔧 特性说明

### 1. 自动查找最新数据
- Step 2 会自动查找最新的横截面数据
- Step 3 会自动查找最新的因子筛选数据
- 如果手动指定路径，可以传入参数

### 2. 缓存系统
- 原始因子计算结果会缓存（约4.6MB）
- 标准化因子结果会缓存（约4.6MB）
- 缓存文件位于 `cache/factors/`
- 使用时间戳命名，7天TTL

### 3. 详细日志
- 每个步骤都有独立的日志文件
- 控制台同步输出
- 包含进度、耗时、统计信息

### 4. 元数据追踪
- 每个步骤保存完整元数据
- 包含时间戳、参数、结果摘要
- JSON格式，便于后续分析

## ⚙️ 参数配置

### 数据加载参数
```python
etf_codes = 43个ETF
start_date = "2020-01-01"
end_date = "2025-10-14"
```

### WFO参数
```python
in_sample_days = 252      # 样本内窗口
out_of_sample_days = 60   # 样本外窗口
step_days = 20            # 滑动步长
target_factor_count = 5   # 目标因子数
ic_threshold = 0.05       # IC阈值
```

## 🧹 清理缓存

如果需要重新计算所有数据：

```bash
# 清理历史结果
rm -rf results/* cache/*

# 或者只清理缓存
rm -rf cache/factors/*
```

## 📝 修改建议

### 修改ETF代码列表
编辑 `scripts/step1_cross_section.py`，修改 `etf_codes` 列表。

### 修改WFO参数
编辑 `scripts/step3_run_wfo.py`，修改 `run_wfo_optimization()` 函数中的参数。

### 修改因子库
编辑 `core/precise_factor_library_v2.py`，添加或删除因子。

## ❓ 常见问题

### Q1: Step 2 找不到横截面数据？
**A**: 确保先运行 `step1_cross_section.py`，或手动指定路径：
```python
from pathlib import Path
from scripts.step2_factor_selection import main

cross_section_dir = Path("results/cross_section/20251025/20251025_201234")
main(cross_section_dir=cross_section_dir)
```

### Q2: 缓存什么时候失效？
**A**: 
- 时间戳超过7天
- OHLCV数据MD5变化
- 因子库类名变化

### Q3: 如何只重算标准化，不重算原始因子？
**A**: 只删除标准化缓存：
```bash
rm cache/factors/standardized_*.pkl
```

## 🎯 开发模式建议

1. **首次运行**: 使用 `run_all_steps.py` 确保流程畅通
2. **调试阶段**: 分步执行，检查中间结果
3. **修改参数**: 清理对应步骤的缓存和结果
4. **生产环境**: 考虑添加错误处理和告警

## 📞 支持

如有问题，请检查日志文件：
- `results/cross_section/{date}/{timestamp}/step1_cross_section.log`
- `results/factor_selection/{date}/{timestamp}/step2_factor_selection.log`
- `results/wfo/{timestamp}/step3_wfo.log`
