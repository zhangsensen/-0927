# iFlow CLI 项目指南 - 深度量化0927

## 项目概述

**深度量化0927** 是一个专业级量化交易开发环境，提供统一的因子计算引擎，支持多市场算法化交易研究。系统核心是 **FactorEngine** 架构，确保研究、回测和生产环境的100%一致性。

### 核心特性

- **统一因子计算引擎**: 单例模式，研究、回测、批量生成全部通过统一API
- **多市场支持**: A股、港股、ETF市场，154个技术指标 + 15个资金流因子
- **向量化计算**: 100%采用Pandas/NumPy向量化，无循环
- **双层缓存系统**: 内存+磁盘缓存，大幅提升性能
- **严格风控**: 无前视偏差，T+1执行约束，可交易性过滤

### 支持的时间框架

| 时间框架 | 说明 | 用途 |
|---------|------|------|
| `1min` | 1分钟 | 高频策略 |
| `5min` | 5分钟 | 中高频策略 |
| `15min` | 15分钟 | 中频策略 |
| `30min` | 30分钟 | 中频策略 |
| `60min` | 60分钟 | 日内策略 |
| `120min` | 2小时 | 日内策略 |
| `240min` | 4小时 | 日内策略 |
| `daily` | 日线 | 日频策略 |
| `weekly` | 周线 | 周频策略 |
| `monthly` | 月线 | 月频策略 |

## 快速开始

### 环境安装

```bash
# 使用uv现代包管理器
uv sync

# 激活虚拟环境
source .venv/bin/activate

# 开发安装（包含所有工具）
uv sync --group all
```

### 核心使用示例

```python
from factor_system.factor_engine import api
from datetime import datetime

# 计算技术指标
factors = api.calculate_factors(
    factor_ids=["RSI14", "MACD", "STOCH"],
    symbols=["0700.HK", "0005.HK"],
    timeframe="15min",
    start_date=datetime(2025, 9, 1),
    end_date=datetime(2025, 9, 30)
)

# 计算资金流因子（A股）
money_flow_factors = api.calculate_factors(
    factor_ids=["MainNetInflow_Rate", "LargeOrder_Ratio"],
    symbols=["000001.SZ", "600036.SH"],
    timeframe="daily",
    start_date=datetime(2025, 9, 1),
    end_date=datetime(2025, 9, 30)
)
```

## 项目结构

```
深度量化0927/
├── factor_system/                # 核心因子系统
│   ├── factor_engine/           # 统一因子计算引擎
│   │   ├── api.py               # 统一API入口
│   │   ├── core/                # 核心引擎组件
│   │   ├── providers/           # 数据提供者
│   │   └── factors/             # 因子实现
│   ├── factor_generation/       # 批量因子生成
│   ├── factor_screening/        # 专业因子筛选
│   └── utils/                   # 工具函数
├── a_shares_strategy/           # A股策略框架
├── etf_download_manager/        # ETF数据管理
├── hk_midfreq/                  # 港股中频策略
├── examples/                    # 使用示例
├── scripts/                     # 工具脚本
├── tests/                       # 测试套件
├── raw/                         # 原始数据存储
└── docs/                        # 详细文档
```

## 开发命令

### 使用Makefile

```bash
make help           # 显示所有可用命令
make install        # 安装开发依赖
make format         # 格式化代码（black + isort）
make lint           # 运行代码检查（flake8 + mypy）
make test           # 运行测试（pytest）
make test-cov       # 运行测试并生成覆盖率报告
make clean          # 清理缓存和临时文件
make check          # 运行所有质量检查（pre-commit）
```

### 手动命令

```bash
# 代码格式化
black .
isort .

# 类型检查
mypy factor_system/

# 运行测试
pytest -v

# 运行pre-commit钩子
pre-commit run --all-files
```

## 因子系统详解

### 可用因子类别

1. **移动平均类** (33个): MA, EMA, DEMA, TEMA, KAMA, WMA等
2. **MACD指标类** (4个): MACD, MACD_SIGNAL, MACD_HIST等
3. **RSI指标类** (10个): RSI3, RSI6, RSI9, RSI14等
4. **随机指标类** (3个): STOCH, STOCHF, STOCHRSI
5. **布林带类** (1个): BBANDS
6. **威廉指标类** (2个): WILLR
7. **商品通道指标类** (2个): CCI
8. **ATR指标类** (1个): ATR
9. **成交量指标类** (6个): OBV, VWAP, Volume_Momentum等
10. **方向性指标类** (6个): ADX, PLUS_DI, MINUS_DI等
11. **K线形态类** (60+个): CDL3WHITESOLDIERS, CDLMORNINGSTAR等
12. **资金流因子类** (12个): MainNetInflow_Rate, LargeOrder_Ratio等

### 核心因子集

```python
# 计算核心因子集（便捷函数）
core_factors = api.calculate_core_factors(
    symbols=["0700.HK"],
    timeframe="15min",
    start_date=datetime(2025, 9, 1),
    end_date=datetime(2025, 9, 30)
)

# 计算动量因子集
momentum_factors = api.calculate_momentum_factors(
    symbols=["0700.HK"],
    timeframe="15min",
    start_date=datetime(2025, 9, 1),
    end_date=datetime(2025, 9, 30)
)
```

## 配置管理

### 环境变量配置

```bash
# 数据路径
export FACTOR_ENGINE_RAW_DATA_DIR="raw"
export FACTOR_ENGINE_CACHE_DIR="cache/factor_engine"

# 缓存配置
export FACTOR_ENGINE_MEMORY_MB="500"
export FACTOR_ENGINE_TTL_HOURS="24"
export FACTOR_ENGINE_ENABLE_MEMORY="true"
export FACTOR_ENGINE_ENABLE_DISK="true"

# 引擎配置
export FACTOR_ENGINE_N_JOBS="1"
export FACTOR_ENGINE_LOG_LEVEL="INFO"
```

### 预定义配置模板

```python
from factor_system.factor_engine.settings import (
    get_development_config,    # 开发环境
    get_research_config,       # 研究环境
    get_production_config      # 生产环境
)

# 使用研究环境配置
settings = get_research_config()
```

## 性能基准

- **因子计算速度**: 300-800+ 因子/秒
- **内存效率**: >70%利用率（Polars优化）
- **缓存命中率**: >90%（智能预热）
- **总因子数**: 258+ (包括参数化变体)
- **核心因子**: 约112个 (不含参数化变体)

## 质量控制

### 代码质量标准

- **类型检查**: 100%类型注解，mypy严格模式
- **代码格式化**: black + isort，88字符行宽
- **静态分析**: flake8, vulture, bandit安全扫描
- **测试覆盖**: pytest + coverage，核心组件全覆盖

### 数据质量标准

- **一致性验证**: FactorEngine vs factor_generation对齐
- **无前视偏差**: 严格遵循14:30信号冻结，T+1执行
- **可交易性约束**: `tradability_mask`有效过滤不可交易样本
- **统一口径**: 所有占比因子分母锁死为`turnover_amount`

## 开发最佳实践

### 1. 使用统一API

```python
# ✅ 推荐：使用统一API
from factor_system.factor_engine import api
factors = api.calculate_factors(...)

# ❌ 避免：直接使用引擎内部组件
from factor_system.factor_engine.core.engine import FactorEngine
```

### 2. 缓存预热

```python
# 预热常用因子缓存
api.prewarm_cache(
    factor_ids=["RSI", "MACD", "STOCH"],
    symbols=["0700.HK", "0005.HK"],
    timeframe="15min",
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 12, 31),
)
```

### 3. 错误处理

```python
from factor_system.factor_engine import api, UnknownFactorError

try:
    factors = api.calculate_factors(["UNKNOWN_FACTOR"], ...)
except UnknownFactorError as e:
    print(f"可用因子: {e.available_factors}")
```

## 相关文档

- **核心指南**: `docs/FACTOR_ENGINE_DEPLOYMENT_GUIDE.md` - 部署指南
- **A股系统**: `docs/MONEYFLOW_INTEGRATION_GUIDE.md` - 资金流集成
- **因子集**: `factor_system/FACTOR_REGISTRY.md` - 因子注册表
- **项目结构**: `docs/README.md` - 详细文档索引

## 环境信息

- **操作系统**: Darwin 24.3.0
- **Python版本**: ≥3.11
- **包管理器**: uv (现代Python包管理)
- **项目路径**: `/Users/zhangshenshen/深度量化0927`
- **Git仓库**: https://github.com/zhangsensen/-0927.git

## 维护信息

- **维护者**: 量化工程团队
- **更新日期**: 2025-10-15
- **版本**: v3.0
- **状态**: 生产就绪

---

*本指南为iFlow CLI提供项目上下文，帮助AI助手更好地理解和协助量化交易开发工作。*