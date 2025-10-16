# iFlow CLI 项目指南 - 深度量化0927

## 项目概述

**深度量化0927** 是一个专业级量化交易开发环境，提供统一的因子计算引擎，支持多市场算法化交易研究。系统核心是 **FactorEngine** 架构，确保研究、回测和生产环境的100%一致性。项目采用 Linus Torvalds 工程原则，消除特殊案例，提供实用解决方案。

### 核心特性

- **统一因子计算引擎**: 单例模式，研究、回测、批量生成全部通过统一API
- **多市场支持**: A股、港股、ETF市场，154个技术指标 + 15个资金流因子 + ETF横截面因子系统
- **向量化计算**: 100%采用Pandas/NumPy向量化，无循环
- **双层缓存系统**: 内存+磁盘缓存，大幅提升性能
- **严格风控**: 无前视偏差，T+1执行约束，可交易性过滤
- **ETF横截面系统**: 完整的ETF轮动因子计算框架，支持生产级横截面分析
- **生产就绪**: 完整的一致性验证、质量控制和生产流水线

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

# ETF横截面因子计算
from factor_system.factor_engine.factors.etf_cross_section import create_etf_cross_section_manager
manager = create_etf_cross_section_manager()
etf_factors = manager.calculate_factors(
    symbols=["510300.SH", "510500.SH", "159915.SZ"],
    start_date=datetime(2025, 9, 1),
    end_date=datetime(2025, 9, 30)
)
```

### ETF横截面系统示例

```python
# 使用统一管理器
from factor_system.factor_engine.factors.etf_cross_section import create_etf_cross_section_manager

# 创建管理器
manager = create_etf_cross_section_manager()

# 计算横截面因子
result = manager.calculate_factors(
    symbols=["510300.SH", "510500.SH", "159915.SZ"],
    start_date=datetime(2025, 9, 1),
    end_date=datetime(2025, 9, 30)
)

# 获取因子注册表
registry = manager.get_factor_registry()
print(f"可用因子数: {len(registry)}")
```

## 项目结构

```
深度量化0927/
├── factor_system/                # 核心因子系统
│   ├── factor_engine/           # 统一因子计算引擎
│   │   ├── api.py               # 统一API入口
│   │   ├── core/                # 核心引擎组件
│   │   ├── providers/           # 数据提供者
│   │   ├── adapters/            # 生产适配器
│   │   ├── factors/             # 因子实现
│   │   │   ├── etf_cross_section/  # ETF横截面因子系统
│   │   │   ├── money_flow/      # 资金流因子
│   │   │   ├── technical/       # 技术指标因子
│   │   │   └── factor_registry.py  # 统一因子注册表
│   │   └── settings.py          # 引擎配置管理
│   ├── factor_generation/       # 批量因子生成
│   ├── factor_screening/        # 专业因子筛选
│   ├── research/                # 研究工具
│   ├── shared/                  # 共享组件
│   └── utils/                   # 工具函数
├── a_shares_strategy/           # A股策略框架
├── etf_download_manager/        # ETF数据管理
├── etf_rotation/                # ETF轮动策略
├── etf_cross_section_production/ # ETF横截面生产系统
├── hk_midfreq/                  # 港股中频策略
├── examples/                    # 使用示例
├── scripts/                     # 工具脚本
│   ├── production_pipeline.py   # 生产流水线
│   ├── ci_checks.py            # CI检查
│   ├── cache_cleaner.py        # 缓存清理
│   └── comprehensive_smoke_test.py # 冒烟测试
├── tests/                       # 测试套件
├── production/                  # 生产配置
├── raw/                         # 原始数据存储
├── cache/                       # 缓存目录
├── factor_output/               # 因子输出
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
make update-deps    # 更新依赖
make setup-dev      # 初始化开发环境
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

### 生产命令

```bash
# 运行生产流水线
python scripts/production_pipeline.py

# 运行一致性检查
python scripts/ci_checks.py

# 清理缓存
python scripts/cache_cleaner.py --etf-cross-section

# 运行冒烟测试
python scripts/comprehensive_smoke_test.py
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
12. **资金流因子类** (15个): MainNetInflow_Rate, LargeOrder_Ratio等
13. **ETF横截面因子类**: 专门的ETF轮动因子系统，支持生产级横截面分析

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

### ETF横截面因子系统

```python
# 专业5维度因子筛选
from factor_system.factor_screening import professional_factor_screener

screener = professional_factor_screener.FactorScreener(
    symbol="0700.HK",
    timeframe="60min",
    factors=["RSI14", "MACD", "STOCH"]
)

# 运行筛选
results = screener.screen_factors()

# ETF轮动策略
from etf_rotation.scorer import ETFScorer
from etf_rotation.portfolio import ETFPortfolio

scorer = ETFScorer()
portfolio = ETFPortfolio()

# ETF横截面生产系统
from factor_system.factor_engine.factors.etf_cross_section import create_etf_cross_section_manager
manager = create_etf_cross_section_manager()
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
- **ETF横截面因子**: 专门的ETF轮动因子系统，支持生产级分析
- **生产验证**: 完整的一致性验证和质量控制体系

## 质量控制

### 代码质量标准

- **类型检查**: 100%类型注解，mypy严格模式
- **代码格式化**: black + isort，88字符行宽
- **静态分析**: flake8, vulture, bandit安全扫描
- **测试覆盖**: pytest + coverage，核心组件全覆盖
- **依赖管理**: uv现代包管理器，支持依赖分组
- **生产验证**: 自动化CI检查和一致性验证

### 数据质量标准

- **一致性验证**: FactorEngine vs factor_generation对齐
- **无前视偏差**: 严格遵循14:30信号冻结，T+1执行
- **可交易性约束**: `tradability_mask`有效过滤不可交易样本
- **统一口径**: 所有占比因子分母锁死为`turnover_amount`
- **ETF横截面一致性**: 专门的冒烟测试验证系统
- **生产监控**: 实时质量监控和异常检测

## 开发最佳实践

### 1. 使用统一API

```python
# ✅ 推荐：使用统一API
from factor_system.factor_engine import api
factors = api.calculate_factors(...)

# ✅ 推荐：使用ETF横截面统一管理器
from factor_system.factor_engine.factors.etf_cross_section import create_etf_cross_section_manager
manager = create_etf_cross_section_manager()

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

# ETF横截面缓存预热
manager.prewarm_cache(
    symbols=["510300.SH", "510500.SH"],
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 12, 31)
)
```

### 3. 错误处理

```python
from factor_system.factor_engine import api, UnknownFactorError

try:
    factors = api.calculate_factors(["UNKNOWN_FACTOR"], ...)
except UnknownFactorError as e:
    print(f"可用因子: {e.available_factors}")

# ETF横截面错误处理
from factor_system.factor_engine.factors.etf_cross_section import ETFCrossSectionError
try:
    result = manager.calculate_factors(...)
except ETFCrossSectionError as e:
    print(f"ETF横截面错误: {e}")
```

### 4. 系统验证

```bash
# 运行冒烟测试
python scripts/comprehensive_smoke_test.py

# 清理ETF横截面缓存
python scripts/cache_cleaner.py --etf-cross-section

# 验证系统一致性
python tests/test_factor_engine_consistency.py

# 运行生产验证
python scripts/production_cross_section_validation.py

# 运行质量检查
make check
```

## 相关文档

- **核心指南**: `docs/FACTOR_ENGINE_DEPLOYMENT_GUIDE.md` - 部署指南
- **A股系统**: `docs/MONEYFLOW_INTEGRATION_GUIDE.md` - 资金流集成
- **因子集**: `factor_system/FACTOR_REGISTRY.md` - 因子注册表
- **项目结构**: `docs/README.md` - 详细文档索引
- **CLAUDE指南**: `CLAUDE.md` - 完整项目指导
- **ETF横截面**: `scripts/smoke_test_report.md` - 系统测试报告
- **生产验证**: `FINAL_PRODUCTION_REPORT.md` - 生产就绪报告

## 环境信息

- **操作系统**: Darwin 24.3.0
- **Python版本**: ≥3.11
- **包管理器**: uv (现代Python包管理)
- **项目路径**: `/Users/zhangshenshen/深度量化0927`
- **Git仓库**: https://github.com/zhangsensen/-0927.git
- **项目名称**: factor-engine
- **当前版本**: 0.2.0

## 维护信息

- **维护者**: 量化工程团队
- **更新日期**: 2025-10-16
- **版本**: v0.2.0
- **状态**: 生产就绪
- **新增特性**: 
  - ETF横截面因子系统增强
  - 生产级一致性验证
  - 完整的质量控制体系
  - 自动化CI/CD流水线
  - 路径管理工具集成
  - 增强的错误处理和监控

---

*本指南为iFlow CLI提供项目上下文，帮助AI助手更好地理解和协助量化交易开发工作。*