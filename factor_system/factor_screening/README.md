# 快速启动配置系统

## 🚀 概述

这是一个基于专业级因子筛选器的快速启动配置系统，支持：

- **多时间框架批量筛选**
- **多股票并行处理**
- **灵活的配置管理**
- **简化的命令行接口**
- **预设配置模板**

## 📁 文件结构

```
factor_screening/
├── professional_factor_screener.py  # 核心筛选引擎
├── config_manager.py               # 配置管理器
├── batch_screener.py              # 批量筛选器
├── cli.py                         # 命令行接口
├── quick_start.py                 # 快速启动脚本
├── README.md                      # 说明文档
└── configs/                       # 配置文件目录
    └── templates/                 # 配置模板
```

## 🎯 快速开始

### 1. 最简单的使用方式

```bash
# 单股票快速筛选
python quick_start.py single 0700.HK 60min

# 多时间框架筛选
python quick_start.py multi_tf 0700.HK

# 多股票筛选
python quick_start.py multi_stocks
```

### 2. 命令行接口

```bash
# 单个筛选
python cli.py single 0700.HK 60min --preset quick

# 批量筛选
python cli.py batch 0700.HK,0005.HK 30min,60min --preset default

# 使用配置文件
python cli.py config batch_config.yaml

# 列出预设配置
python cli.py presets

# 创建配置模板
python cli.py templates
```

### 3. Python API

```python
from config_manager import ConfigManager
from batch_screener import BatchScreener

# 创建管理器
config_manager = ConfigManager()
batch_screener = BatchScreener(config_manager)

# 创建批量配置
batch_config = config_manager.create_batch_config(
    task_name="my_screening",
    symbols=["0700.HK", "0005.HK"],
    timeframes=["30min", "60min"],
    preset="default"
)

# 运行筛选
batch_result = batch_screener.run_batch(batch_config)

# 保存结果
saved_files = batch_screener.save_results(batch_result)
```

## ⚙️ 预设配置

### 1. default - 默认配置
- **用途**: 平衡的参数设置，适合大多数场景
- **IC周期**: [1, 3, 5, 10, 20]
- **最小样本**: 100
- **并行数**: 4

### 2. quick - 快速配置
- **用途**: 测试和快速验证
- **IC周期**: [1, 3, 5]
- **最小样本**: 50
- **并行数**: 2

### 3. deep - 深度配置
- **用途**: 全面的因子分析
- **IC周期**: [1, 3, 5, 10, 20, 30]
- **最小样本**: 200
- **并行数**: 6

### 4. high_freq - 高频配置
- **用途**: 优化短周期因子
- **时间框架**: [1min, 5min, 15min]
- **IC周期**: [1, 2, 3, 5]
- **权重**: 更重视短周期适应性

### 5. multi_timeframe - 多时间框架配置
- **用途**: 多时间框架分析
- **时间框架**: [5min, 15min, 30min, 60min, 1day]
- **并行数**: 8

## 📝 配置文件格式

### 单个筛选配置 (YAML)

```yaml
name: "my_screening"
description: "自定义筛选配置"
symbols: ["0700.HK"]
timeframes: ["60min"]
ic_horizons: [1, 3, 5, 10, 20]
min_sample_size: 100
significance_level: 0.05
weights:
  predictive_power: 0.35
  stability: 0.25
  independence: 0.20
  practicality: 0.10
  short_term_fitness: 0.10
```

### 批量筛选配置 (YAML)

```yaml
task_name: "batch_screening"
description: "批量筛选任务"
max_concurrent_tasks: 2
continue_on_error: true
screening_configs:
  - name: "0700_60min"
    symbols: ["0700.HK"]
    timeframes: ["60min"]
    # ... 其他配置
  - name: "0005_60min"
    symbols: ["0005.HK"]
    timeframes: ["60min"]
    # ... 其他配置
```

## 🔧 高级用法

### 1. 自定义配置

```python
from config_manager import ScreeningConfig

# 创建自定义配置
custom_config = ScreeningConfig(
    name="custom",
    symbols=["0700.HK"],
    timeframes=["30min"],
    ic_horizons=[1, 3, 5],
    weights={
        "predictive_power": 0.4,
        "stability": 0.3,
        "independence": 0.15,
        "practicality": 0.1,
        "short_term_fitness": 0.05
    }
)

# 保存配置
config_manager.save_config(custom_config, "my_custom_config")
```

### 2. 并行处理优化

```python
# 设置并行参数
batch_config.max_concurrent_tasks = 4  # 最大并发任务数
batch_config.enable_task_parallel = True  # 启用并行

# 为每个筛选配置设置并行
for config in batch_config.screening_configs:
    config.max_workers = 6  # 单个任务的并行数
    config.enable_parallel = True
```

### 3. 结果分析

```python
# 生成对比报告
comparison_df = batch_screener.generate_comparison_report([batch_result])

# 查看汇总统计
print(batch_result.summary_stats)

# 查看性能统计
print(batch_result.performance_stats)
```

## 📊 输出结果

### 1. 批量结果目录结构

```
output/
└── batch_[ID]_[timestamp]/
    ├── batch_summary.json      # 批量摘要
    ├── task_results.json       # 任务结果详情
    ├── batch_report.csv        # 汇总报告
    └── detailed_results/       # 详细结果
        ├── 0700.HK_60min_details.json
        └── 0005.HK_30min_details.json
```

### 2. 结果内容

- **batch_summary.json**: 批量任务的整体统计
- **task_results.json**: 每个子任务的详细结果
- **batch_report.csv**: 便于Excel查看的汇总报告
- **detailed_results/**: 每个任务的完整因子筛选结果

## 🚨 注意事项

### 1. 数据路径配置

确保数据路径正确设置：
- `data_root`: 因子数据目录 (默认: `./output`)
- `raw_data_root`: 原始价格数据目录 (默认: `../raw`)

### 2. 内存和性能

- 大批量任务建议设置合理的并发数
- 监控内存使用，避免OOM
- 深度配置需要更多计算资源

### 3. 错误处理

- 设置 `continue_on_error=True` 在遇到错误时继续
- 查看日志文件了解详细错误信息
- 使用 `max_retries` 设置重试次数

## 🔍 故障排除

### 1. 数据文件不存在

```
错误: No factor data found for 0700.HK 60min
解决: 检查数据路径和文件名格式
```

### 2. 内存不足

```
错误: MemoryError
解决: 减少并发数或使用更小的数据集
```

### 3. 配置验证失败

```
错误: 配置验证失败
解决: 检查配置参数的有效性和范围
```

## 📞 支持

如有问题，请查看：
1. 日志文件中的详细错误信息
2. 配置文件的参数设置
3. 数据文件的路径和格式

---

**版本**: 1.0.0  
**作者**: 量化首席工程师  
**日期**: 2025-09-30
