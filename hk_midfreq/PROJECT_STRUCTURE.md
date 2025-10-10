# HK中频交易系统 - 项目结构文档

**版本**: 1.0  
**更新日期**: 2025-10-06  
**哲学**: Linus Torvalds风格 - 简洁、实用、无垃圾

---

## 📋 目录结构

```
hk_midfreq/
├── 核心模块 (11个)
│   ├── __init__.py                 # 包初始化和公共API
│   ├── config.py                   # 配置管理 (PathConfig, RuntimeConfig)
│   ├── price_loader.py             # 价格数据加载器
│   ├── factor_interface.py         # 因子数据接口
│   ├── fusion.py                   # 多时间框架因子融合
│   ├── strategy_core.py            # 策略核心逻辑
│   ├── backtest_engine.py          # VectorBT回测引擎
│   ├── result_manager.py           # 结果管理器
│   ├── log_formatter.py            # 结构化日志格式化
│   ├── settings_loader.py          # 设置加载器
│   └── run_multi_tf_backtest.py    # 主程序入口
│
├── 可选模块 (2个)
│   ├── combination_backtest.py     # 因子组合回测引擎
│   └── session_index_manager.py    # 会话索引管理
│
├── 配置文件
│   ├── settings.yaml               # 系统配置
│   └── ARCHITECTURE.md             # 架构设计文档
│
├── 工具目录
│   ├── utils/                      # 工具函数
│   │   ├── __init__.py
│   │   └── signal_utils.py
│   └── examples/                   # 示例代码
│       └── p0_optimization_demo.py
│
└── 输出目录
    ├── backtest_results/           # 回测结果存储
    │   ├── RUNS_INDEX.json         # 会话索引
    │   └── {session_id}/           # 各个回测会话
    └── __pycache__/                # Python缓存
```

---

## 🎯 核心模块详解

### 1. `__init__.py` - 包初始化
**行数**: 35  
**职责**: 定义公共API，导出核心类和函数  
**导出**:
- `run_single_asset_backtest`
- `PathConfig`, `TradingConfig`, `ExecutionConfig`, `StrategyRuntimeConfig`
- `StrategyCore`, `StrategySignals`
- `PriceDataLoader`, `FactorScoreLoader`
- `BacktestResultManager`
- `DataLoadError`, `FactorLoadError`

### 2. `config.py` - 配置管理
**行数**: 175  
**职责**: 统一路径管理、策略参数配置  
**核心类**:
- `PathConfig`: 项目路径管理（项目根、数据目录、输出目录）
- `TradingConfig`: 交易参数（止损、止盈、持仓天数）
- `ExecutionConfig`: 执行参数（交易成本、滑点）
- `StrategyRuntimeConfig`: 策略运行时配置

**设计原则**:
- 单一真相来源（SSoT）
- 自动路径发现（向上搜索项目根）
- 无硬编码路径

### 3. `price_loader.py` - 价格数据加载
**行数**: 151  
**职责**: 从 `raw/HK/` 加载原始价格数据  
**功能**:
- 支持多时间框架（5min, 15min, 30min, 60min, daily）
- 自动发现最新数据文件
- 标准化列名和索引
- 异常处理（`DataLoadError`）

**数据格式**:
```python
DataFrame[timestamp, open, high, low, close, volume, turnover]
```

### 4. `factor_interface.py` - 因子数据接口
**行数**: 412  
**职责**: 管理因子数据的加载和查询  
**核心类**: `FactorScoreLoader`  
**功能**:
- 从 `factor_ready/` 加载因子评分（筛选后的优秀因子）
- 从 `factor_output/` 加载因子时间序列
- 支持因子面板聚合
- 相关性分析支持

**三层数据架构**:
1. `raw/HK/` → 价格数据
2. `factor_ready/` → 因子评分（50个最佳因子）
3. `factor_output/{tf}/` → 因子时间序列（233个因子）

### 5. `fusion.py` - 因子融合
**行数**: 244  
**职责**: 多时间框架因子融合  
**核心类**: `FactorFusionEngine`  
**算法**:
- 因子权重计算（IC加权或等权重）
- 时间框架聚合
- 趋势/确认过滤
- 复合评分计算

### 6. `strategy_core.py` - 策略核心
**行数**: 923  
**职责**: 信号生成和候选筛选  
**核心类**: `StrategyCore`  
**功能**:
- 多时间框架候选筛选
- 信号生成（因子驱动 or 传统反转）
- 风险参数管理
- 入场/出场逻辑

### 7. `backtest_engine.py` - 回测引擎
**行数**: 186  
**职责**: VectorBT向量化回测  
**核心函数**: `run_single_asset_backtest`, `run_portfolio_backtest`  
**特性**:
- 向量化计算（高性能）
- 多标的组合回测
- 交易成本建模
- 完整统计指标

### 8. `result_manager.py` - 结果管理
**行数**: 1078  
**职责**: 回测结果的保存、管理和可视化  
**核心类**: `BacktestResultManager`  
**功能**:
- 时间戳会话目录创建
- 结果保存（Parquet, CSV, JSON）
- 图表生成（性能概览、交易分布）
- 日志管理（RotatingFileHandler, 10MB）
- 环境快照（pip freeze, 系统信息）
- 列名清洗（移除非法字符）

**会话目录结构**:
```
{session_id}/
├── charts/
│   ├── performance_overview.png
│   └── trade_distribution.png
├── data/
│   ├── portfolio_stats.parquet
│   ├── trades.parquet
│   └── positions.parquet
├── logs/
│   ├── debug.log
│   ├── stdout.log
│   └── stderr.log
├── env/
│   ├── pip_freeze.txt
│   └── system_info.json
├── backtest_config.json
├── backtest_metrics.json
└── summary_report.md
```

### 9. `log_formatter.py` - 日志格式化
**行数**: 172  
**职责**: 标准化日志输出  
**核心类**: `StructuredLogger`  
**格式**: `{session_id}|{symbol}|{tf}|{direction}|{metric}={value}`

### 10. `settings_loader.py` - 设置加载
**行数**: 121  
**职责**: 从 `settings.yaml` 加载配置  
**功能**:
- 单例模式
- 日志级别解析
- 配置缓存

### 11. `run_multi_tf_backtest.py` - 主程序
**行数**: 666  
**职责**: 多时间框架回测的主入口  
**流程**:
1. 初始化配置和会话
2. 加载多时间框架价格数据
3. 加载因子数据（可选）
4. 生成多时间框架组合信号
5. 执行VectorBT回测
6. 保存结果和生成报告

---

## 🔧 可选模块

### 1. `combination_backtest.py` - 因子组合回测
**行数**: 711  
**用途**: 批量测试因子组合，寻找最优组合  
**核心类**:
- `FactorCombinationEngine`: 生成因子组合
- `VectorBTBatchBacktester`: 批量回测
- `CombinationOptimizer`: 组合优化
- `PerformanceAnalyzer`: 绩效分析

**何时使用**: 需要系统性寻找最佳因子组合时

### 2. `session_index_manager.py` - 会话索引
**行数**: 233  
**用途**: 维护所有回测会话的中心化索引  
**核心类**: `SessionIndexManager`  
**功能**:
- 会话注册
- 会话查询（按symbol/type过滤）
- 索引重建
- 摘要报告

**何时使用**: 管理大量回测会话时

---

## ⚙️ 配置文件

### `settings.yaml`
**行数**: 77  
**用途**: 系统全局配置  
**配置项**:
- 日志配置（级别、格式、轮转）
- 路径配置
- 回测配置
- 图表配置
- 因子配置
- 环境快照配置
- 数据验证配置

---

## 📊 数据流架构

```
原始数据层 (raw/HK/)
    ↓ PriceDataLoader
价格数据 (OHLCV)
    ↓
    ├─→ 因子计算 → 因子输出层 (factor_output/{tf}/)
    │       ↓
    │   因子筛选 → 因子准备层 (factor_ready/)
    │       ↓ FactorScoreLoader
    │   因子评分 + 时间序列
    │       ↓
    └─→ StrategyCore → 信号生成
            ↓
        BacktestEngine → VectorBT回测
            ↓
        ResultManager → 保存结果
            ↓
        backtest_results/{session_id}/
```

---

## 🎯 使用指南

### 快速开始

```python
# 1. 导入核心模块
from hk_midfreq import (
    run_single_asset_backtest,
    PathConfig,
    StrategyCore,
    PriceDataLoader,
)

# 2. 简单回测
from hk_midfreq.run_multi_tf_backtest import main
main()

# 3. 因子组合回测
from hk_midfreq.combination_backtest import run_combination_backtest
result = run_combination_backtest(
    symbol="0700.HK",
    timeframe="15min"
)
```

### 关键依赖

```
核心:
- pandas >= 1.5.0
- numpy >= 1.23.0
- vectorbt >= 0.25.0

可选:
- matplotlib >= 3.5.0 (图表生成)
- joblib >= 1.2.0 (组合回测并行)
- psutil >= 5.9.0 (性能监控)
```

---

## 🚀 性能指标

### 回测性能
- **单标的回测**: < 5秒（8000+ bars）
- **多时间框架**: 5-10秒（5个时间框架）
- **因子组合回测**: 10-60秒（1000-10000组合）

### 内存使用
- **基础回测**: < 200MB
- **多因子回测**: < 500MB
- **组合回测**: 500MB - 2GB（视组合数量）

---

## 🔄 维护指南

### 添加新功能
1. 遵循现有模块结构
2. 保持函数简短（< 50行）
3. 添加类型提示
4. 更新 `__init__.py` 导出

### 删除代码原则
按Linus哲学，删除条件：
- ✅ 无任何引用
- ✅ 功能重复
- ✅ 超过6个月未使用
- ✅ 测试/临时代码

### 代码质量标准
- 🟢 优秀: 简洁、单一职责、向量化
- 🟡 可接受: 功能正确但可优化
- 🔴 需重构: 复杂、重复、性能差

---

## 📝 变更历史

### v1.0 (2025-10-06)
- ✅ 完成P0优化（路径解耦、配置统一）
- ✅ 实现多时间框架回测
- ✅ 实现因子组合回测
- ✅ 标准化日志格式
- ✅ 环境快照功能
- ✅ 清理临时文件和未使用模块
- ❌ 删除: diagnose_missing_factors.py
- ❌ 删除: data_integrity_validator.py
- ❌ 删除: test_new_features.py
- ❌ 删除: validate_architecture.py

---

## 🎓 设计哲学

### Linus原则
1. **简洁至上**: 代码越少越好，但不能过简
2. **实用主义**: 解决真实问题，不过度设计
3. **性能优先**: 向量化 > 循环，内存效率 > 70%
4. **无破坏性**: API稳定，向后兼容
5. **消灭特殊情况**: 用数据结构和流程统一，减少if/else

### 量化纪律
1. **严禁未来函数**: 任何前视偏差都是致命的
2. **交易成本真实**: 0.2%佣金 + 0.05 HKD滑点
3. **数据完整性**: 统一时区、明确复权、固定schema
4. **可重现性**: 固定随机种子、完整环境快照

---

## 📮 联系

**项目**: HK中频交易系统  
**架构**: 三层分离（原始数据 → 因子 → 回测）  
**文档**: 本文件持续更新

**最后更新**: 2025-10-06 15:35 UTC+8

