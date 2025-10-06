# HK量化因子架构方案

## 概述

本架构方案是一个专业的量化因子系统，采用三层分离设计，支持多时间框架策略开发和实盘交易。基于Linus工程哲学设计：简洁、实用、高效。

## 架构设计

### 核心理念

- **数据分离**：原始数据、因子筛选、因子输出三层分离
- **预计算优化**：优秀因子预存储，避免重复计算
- **即时加载**：运行时高效桥接，支持策略快速执行
- **模块化设计**：每个组件职责单一，接口标准化

### 系统架构图

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   原始数据层     │    │   因子筛选层     │    │   因子输出层     │
│                │    │                │    │                │
│ /raw/HK/       │───▶│/factor_system/  │───▶│/factor_system/  │
│ • 0700HK_1min   │    │factor_ready/    │    │因子输出/         │
│ • 0700HK_5min   │    │• best_factors   │    │• 5min/         │
│ • 0700HK_60m    │    │• 0700_HK_best   │    │• 15min/        │
│ • 多时间框架     │    │• 筛选评估结果    │    │• 60min/        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   运行时桥接     │
                    │                │
                    │ hk_midfreq/    │
                    │ • price_loader  │
                    │ • factor_interface│
                    │ • strategy_core │
                    └─────────────────┘
```

## 数据存储架构

### 1. 原始数据层 (`/raw/HK/`)

**用途**：存储原始OHLCV价格数据
**格式**：Parquet文件，按股票代码和时间框架组织
**特点**：
- 标准化文件命名：`{SYMBOL}_{TIMEFRAME}_{DATE_RANGE}.parquet`
- 支持多时间框架：1min, 2min, 3min, 5min, 15min, 30min, 60min, daily
- 高效列式存储，支持快速查询

**示例文件结构**：
```
/raw/HK/
├── 0700HK_1min_2025-03-05_2025-09-01.parquet
├── 0700HK_5min_2025-03-05_2025-09-01.parquet
├── 0700HK_60m_2025-03-05_2025-09-01.parquet
├── 0005HK_1min_2025-03-06_2025-09-02.parquet
└── ...
```

### 2. 因子筛选层 (`/factor_system/factor_ready/`)

**用途**：存储经过严格筛选的优秀因子
**格式**：JSON + Parquet混合存储
**特点**：
- 预筛选的5维度评估因子
- 包含完整的统计验证结果
- 支持快速策略开发

**文件结构**：
```
/factor_system/factor_ready/
├── 0700_HK_best_factors.parquet    # 优秀因子数据
├── build_screened_factors.py       # 构建脚本
└── session_20251005/               # 筛选会话目录
    ├── timeframes/
    │   ├── 0700_HK_60m/
    │   │   └── top_factors_detailed.json
    │   └── 0005_HK_5min/
    │       └── top_factors_detailed.json
    └── metadata.json
```

### 3. 因子输出层 (`/factor_system/因子输出/`)

**用途**：存储计算的因子时间序列数据
**格式**：按时间框架分组的Parquet文件
**特点**：
- 仅包含因子列，去除原始价格数据
- 支持多时间框架因子值存储
- 便于策略直接使用

**目录结构**：
```
/factor_system/因子输出/
├── 5min/
│   ├── 0700_HK_5min_factors_2025-10-05_14-30-00.parquet
│   └── 0005_HK_5min_factors_2025-10-05_14-30-00.parquet
├── 15min/
├── 30min/
├── 60min/
└── daily/
```

## 运行时桥接设计

### 核心组件

#### 1. PriceDataLoader (`price_loader.py`)

**职责**：统一的价格数据加载接口
**特性**：
- 标准化的符号和时间框架映射
- 自动文件发现和加载
- 数据格式标准化处理
- 错误处理和验证

```python
# 使用示例
loader = PriceDataLoader()
price_data = loader.load_price("0700.HK", "60min")
```

#### 2. FactorScoreLoader (`factor_interface.py`)

**职责**：因子评分和筛选数据加载
**特性**：
- 支持多符号、多时间框架批量加载
- 智能会话管理和数据发现
- 灵活的聚合评分计算
- 因子时间序列数据加载

```python
# 使用示例
loader = FactorScoreLoader()
factor_scores = loader.load_scores_as_series(["0700.HK", "0005.HK"])
factor_data = loader.load_factor_time_series("0700.HK", "60min", ["RSI_14", "MACD"])
```

#### 3. StrategyCore (`strategy_core.py`)

**职责**：策略信号生成引擎
**特性**：
- 多时间框架候选选择
- 因子驱动的信号生成
- 趋势和确认过滤器
- 风险管理集成

```python
# 使用示例
strategy = StrategyCore()
candidates = strategy.select_candidates(["0700.HK", "0005.HK"])
signals = strategy.build_signal_universe(price_data)
```

## 数据流设计

### 1. 因子计算流程

```
原始价格数据 → 因子计算引擎 → 统计验证 → 5维度评估 → 优秀因子筛选 → 存储到factor_ready
```

### 2. 策略执行流程

```
选择候选股票 → 加载优秀因子 → 即时因子计算 → 信号生成 → 风险管理 → 交易执行
```

### 3. 数据访问模式

- **读取优化**：预存储优秀因子，减少运行时计算
- **写入优化**：批量计算和存储，提升效率
- **缓存策略**：内存缓存热门数据，加速访问

## 技术特性

### 性能优化

1. **列式存储**：Parquet格式，高效压缩和查询
2. **预计算**：优秀因子预存储，避免重复计算
3. **向量化**：使用VectorBT，10-50x性能提升
4. **内存管理**：智能缓存，减少内存占用

### 数据质量

1. **标准化**：统一的数据格式和命名规范
2. **验证机制**：完整性和准确性检查
3. **版本控制**：因子数据版本管理
4. **监控告警**：数据质量实时监控

### 扩展性设计

1. **模块化**：组件独立，易于扩展和维护
2. **配置驱动**：参数化配置，适应不同策略
3. **接口标准化**：统一的API接口，支持插件扩展
4. **多市场支持**：架构支持港股、A股、美股

## 改进建议

### P0级优化（立即实施）

1. **统一配置管理**
   - 将硬编码路径集中到config.py
   - 提升系统可移植性
   - 便于环境切换

2. **路径解耦**
   - 移除price_loader.py和factor_interface.py中的硬编码路径
   - 使用配置文件管理所有路径
   - 支持相对路径和绝对路径

3. **错误处理标准化**
   - 统一异常处理机制
   - 标准化错误码和消息
   - 提升调试效率

### P1级改进（中期规划）

1. **缓存机制优化**
   ```python
   # 添加内存缓存
   @lru_cache(maxsize=1000)
   def load_factor_data(symbol, timeframe):
       # 因子数据加载逻辑
   ```

2. **并行数据加载**
   ```python
   # 多线程并行加载
   from concurrent.futures import ThreadPoolExecutor

   def load_multiple_symbols(symbols, timeframe):
       with ThreadPoolExecutor(max_workers=4) as executor:
           futures = [executor.submit(load_price, sym, timeframe) for sym in symbols]
           return {sym: future.result() for sym, future in zip(symbols, futures)}
   ```

3. **数据版本管理**
   ```python
   # 因子数据版本控制
   class FactorDataManager:
       def __init__(self, version="v1.0"):
           self.version = version
           self.data_dir = Path(f"/factor_system/factor_ready/{version}")
   ```

### P2级演进（长期规划）

1. **分布式存储支持**
   - 支持云存储（S3、OSS等）
   - 数据分片和负载均衡
   - 容灾备份机制

2. **实时数据更新**
   - 增量数据更新机制
   - 实时因子计算
   - 消息队列集成

3. **监控体系**
   - 性能监控仪表板
   - 数据质量监控
   - 告警和通知系统

## 使用指南

### 快速开始

```python
from hk_midfreq.price_loader import PriceDataLoader
from hk_midfreq.factor_interface import FactorScoreLoader
from hk_midfreq.strategy_core import StrategyCore

# 1. 加载价格数据
price_loader = PriceDataLoader()
price_data = price_loader.load_price("0700.HK", "60min")

# 2. 加载因子评分
factor_loader = FactorScoreLoader()
factor_scores = factor_loader.load_scores_as_series(["0700.HK"])

# 3. 生成交易信号
strategy = StrategyCore()
signals = strategy.build_signal_universe({"0700.HK": {"60m": price_data}})
```

### 自定义策略开发

```python
# 自定义信号生成逻辑
def custom_signal_generator(close_price, volume_data, factor_scores):
    # 基于因子评分的自定义逻辑
    top_factors = factor_scores.nlargest(5)
    # 生成入场和出场信号
    return entries, exits

# 集成到StrategyCore
strategy = StrategyCore()
strategy.register_signal_generator(custom_signal_generator)
```

### 批量处理

```python
# 批量股票处理
symbols = ["0700.HK", "0005.HK", "0941.HK"]
timeframe = "60min"

# 并行加载价格数据
price_data = {symbol: price_loader.load_price(symbol, timeframe)
              for symbol in symbols}

# 批量生成信号
all_signals = strategy.build_signal_universe(price_data, timeframe=timeframe)
```

## 最佳实践

### 开发规范

1. **命名规范**
   - 文件名：小写字母+下划线
   - 类名：大驼峰命名
   - 函数名：小写字母+下划线

2. **错误处理**
   - 使用具体的异常类型
   - 提供有意义的错误消息
   - 记录详细的调试信息

3. **性能优化**
   - 使用向量化操作
   - 避免重复计算
   - 合理使用缓存

### 部署建议

1. **环境配置**
   - 使用虚拟环境隔离依赖
   - 配置文件管理不同环境
   - 监控系统资源使用

2. **数据备份**
   - 定期备份因子数据
   - 版本控制配置文件
   - 灾难恢复计划

3. **性能监控**
   - 监控数据加载性能
   - 跟踪内存使用情况
   - 记录策略执行时间

## 总结

本架构方案是一个专业的量化因子系统，具备以下核心优势：

- ✅ **设计清晰**：三层分离架构，职责明确
- ✅ **性能优秀**：预计算+即时加载，高效执行
- ✅ **扩展性强**：模块化设计，易于扩展
- ✅ **实用导向**：解决实际问题，支持生产使用

该架构完全符合Linus工程哲学：**简洁、实用、高效**，是一个可以直接投入生产的专业级量化系统。