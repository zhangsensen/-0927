# ETF轮动回测引擎 - 向量化并行计算版本

高性能ETF轮动策略回测引擎，支持完全配置化和大规模并行计算。

## 🚀 核心特性

- **完全向量化计算**: 使用3D矩阵操作和`np.einsum`消除所有循环
- **多进程并行**: 支持自定义工作进程数和块大小优化
- **配置化设计**: 通过YAML配置文件实现完全参数化
- **预设场景**: 提供6种优化预设配置
- **高精度计算**: 支持自定义数值精度保护

## 📁 项目结构

```
03_vbt回测/
├── 📄 README.md                           # 项目文档
├── 📄 parallel_backtest_engine.py         # 核心向量化引擎
├── 📄 parallel_backtest_configurable.py   # 配置化版本
├── 📄 config_loader_parallel.py           # 配置加载系统
├── 📄 parallel_backtest_config.yaml       # 主配置文件
├── 📁 archive_docs/                       # 文档归档
├── 📁 archive_tests/                      # 测试和工具归档
│   ├── 📁 legacy_engines/                 # 旧版引擎
│   ├── 📁 config_files/                   # 配置文件归档
│   ├── 📁 test_files/                     # 测试文件
│   ├── 📁 utilities/                      # 工具脚本
│   └── 📁 reports/                        # 性能报告
└── 📁 archive_tasks/                      # 任务归档
```

## 🔧 核心文件说明

### 主要引擎文件

- **`parallel_backtest_engine.py`**: 高性能向量化并行引擎
  - 完全消除循环的3D矩阵操作
  - 支持大规模权重组合并行优化
  - 实现了7.72x加速比，85.8%并行效率

- **`parallel_backtest_configurable.py`**: 配置化版本
  - 集成完整配置抽象系统
  - 支持多种预设配置
  - 保持与原版完全一致的数值精度

- **`config_loader_parallel.py`**: 配置管理
  - 支持YAML配置文件加载
  - 提供6种预设配置场景
  - 自动数据类型转换和验证

- **`parallel_backtest_config.yaml`**: 主配置文件
  - 完整的参数配置
  - 多种优化预设
  - 支持自定义参数覆盖

## ⚡ 性能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| 处理速度 | 128+ 策略/秒 | 10,000组合测试 |
| 并行效率 | 85.8% | 9个工作进程 |
| 加速比 | 7.72x | 相比串行版本 |
| 内存效率 | <1GB | 中等规模数据 |
| 数值精度 | 100%一致 | 与原版完全一致 |

## 🎯 使用方法

### 基本使用

```bash
# 使用默认配置运行
python parallel_backtest_engine.py

# 使用配置化版本
python parallel_backtest_configurable.py

# 指定预设配置
python parallel_backtest_configurable.py --preset high_performance

# 指定最大组合数
python parallel_backtest_configurable.py --max-combinations 50000
```

### 配置预设

- `quick_test`: 快速测试 (500组合)
- `standard`: 标准配置 (5,000组合)
- `high_performance`: 高性能 (15,000组合)
- `memory_safe`: 内存安全 (3,000组合)
- `fine_grid`: 精细网格 (8,000组合)
- `vectorized_optimized`: 向量化优化 (10,000组合)

### 自定义配置

编辑 `parallel_backtest_config.yaml` 文件或创建自定义配置：

```yaml
parallel_config:
  n_workers: 16
  chunk_size: 50
  max_combinations: 50000

backtest_config:
  top_n_list: [3, 5, 8, 10, 12]
  fees: 0.0015
  rebalance_freq: 15

weight_grid_config:
  grid_points: [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
  weight_sum_range: [0.7, 1.3]
```

## 📊 技术实现

### 向量化优化

- **3D矩阵重塑**: `(dates, symbols, factors)` 结构优化
- **批量矩阵乘法**: 使用 `np.einsum('cf,dsf->cds')` 消除循环
- **内存访问优化**: 连续内存布局提高缓存效率
- **并行处理**: 多进程独立块处理

### 配置抽象

- **类型安全**: 自动数据类型转换
- **验证机制**: 配置参数完整性检查
- **预设管理**: 场景化配置快速切换
- **向后兼容**: 与硬编码版本100%兼容

## 📈 性能优化总结

1. **循环消除**: 从O(n³)循环复杂度优化到O(1)矩阵操作
2. **并行计算**: 9进程并行实现接近理论最大性能
3. **内存优化**: 向量化操作减少内存分配开销
4. **配置灵活**: 零性能损失的完整配置抽象

## 🗂️ 归档说明

已归档的文件按功能分类存储在 `archive_tests/` 目录：

- **legacy_engines/**: 旧版引擎实现
- **config_files/**: 历史配置文件
- **test_files/**: 性能测试和对比测试
- **utilities/**: 辅助工具和分析脚本
- **reports/**: 性能分析报告

核心项目保持纯净，只包含生产级代码文件。