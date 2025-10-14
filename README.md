# 深度量化0927 - 量化因子系统

## 项目概述

本项目是一个专业级量化交易开发环境，提供统一的因子计算引擎，支持多市场算法化交易研究。系统核心是**FactorEngine**架构，确保研究、回测和生产环境的100%一致性。系统涵盖A股、港股和ETF市场，提供154个技术指标和完整的资金流因子系统。

**核心理念**：Linus Torvalds工程原则 - 消除特殊案例，提供实用解决方案，确保代码在实际市场中可靠运行。

## ✅ 核心系统组件

### 1. **FactorEngine 统一架构**
- ✅ **API接口**: `factor_system/factor_engine/api.py` - 单一入口点
- ✅ **核心引擎**: 支持双重缓存，高性能因子计算
- ✅ **因子注册表**: 154个技术指标 + 15个资金流因子
- ✅ **数据提供者**: Parquet、CSV、分钟数据、资金流数据

### 2. **多市场支持**
- ✅ **港股市场**: 276+股票，1分钟到日线数据
- ✅ **A股市场**: 资金流因子系统，T+1执行约束
- ✅ **ETF市场**: 19个核心ETF，2年历史数据

### 3. **因子计算系统**
- ✅ **154个技术指标**: 36核心 + 118增强指标
- ✅ **15个资金流因子**: 8核心 + 4增强 + 3约束因子
- ✅ **多时间框架**: 1min到monthly，自动重采样
- ✅ **向量化实现**: >95%向量化率

### 4. **专业因子筛选**
- ✅ **5维度评估**: 预测力、稳定性、独立性、实用性、适应性
- ✅ **统计严谨性**: Benjamini-Hochberg FDR校正，VIF分析
- ✅ **成本建模**: 港股佣金、印花税、滑点模型

### 5. **完整测试体系**
- ✅ **单元测试**: 核心组件覆盖
- ✅ **集成测试**: 多组件协作验证
- ✅ **一致性测试**: FactorEngine vs factor_generation对齐

## 📊 技术特性

- **向量化计算**: 100%采用Pandas/NumPy向量化，无循环。
- **统一口径**: 所有占比因子分母锁死为`turnover_amount`。
- **无前视偏差**: 严格遵循14:30信号冻结，T+1执行。
- **可交易性约束**: `tradability_mask`有效过滤不可交易样本。
- **自动化测试**: 覆盖数据口径、因子计算、T+1执行等关键环节。
- **多时间框架**: 支持 1min/5min/15min/30min/60min/120min/240min/daily/weekly/monthly

### 支持的时间框架 (timeframes)

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

## 🚀 快速开始

### 环境安装
```bash
# 使用uv现代包管理器
uv sync

# 激活虚拟环境
source .venv/bin/activate

# 开发安装（包含所有工具）
uv sync --group all
```

### 核心使用

#### 1. 因子计算（推荐方式）
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

#### 2. 批量因子生成
```bash
# 港股单股票因子生成
cd factor_system/factor_generation
python run_single_stock.py 0700.HK

# A股因子生成
python a_shares_strategy/generate_a_share_factors.py 600036.SH --timeframe 5min

# ETF数据处理
python etf_download_manager/download_etf_final.py
```

#### 3. 因子筛选
```bash
# 专业5维度因子筛选
cd factor_system/factor_screening
python professional_factor_screener.py --symbol 0700.HK --timeframe 60min
```

### 系统验证
```bash
# 运行测试套件
pytest -v

# FactorEngine一致性验证
python tests/test_factor_engine_consistency.py

# 路径管理系统验证
python -c "from factor_system.utils import get_project_root; print('✅ 系统正常')"
```

## 📁 项目结构

```
深度量化0927/
├── factor_system/                # 核心因子系统
│   ├── factor_engine/           # 统一因子计算引擎
│   ├── factor_generation/       # 批量因子生成
│   ├── factor_screening/        # 专业因子筛选
│   └── utils/                   # 路径管理和工具
├── a_shares_strategy/           # A股策略框架
├── etf_download_manager/        # ETF数据管理
├── hk_midfreq/                  # 港股中频策略
├── examples/                    # 使用示例
├── scripts/                     # 工具脚本
├── tests/                       # 测试套件
├── raw/                         # 原始数据存储
└── docs/                        # 详细文档
```

## 🔧 开发环境

### 代码质量工具
```bash
# 代码格式化
black factor_system/
isort factor_system/

# 类型检查
mypy factor_system/

# 运行pre-commit钩子
pre-commit run --all-files

# 安装开发钩子
pre-commit install
```

### 性能基准
- **因子计算速度**: 300-800+ 因子/秒
- **内存效率**: >70%利用率（Polars优化）
- **缓存命中率**: >90%（智能预热）

## 📚 相关文档

- **核心指南**: `CLAUDE.md` - 完整项目指导
- **A股系统**: `README_A_SHARES.md` - A股专门说明
- **因子集**: `README_FACTOR_SETS.md` - 因子集管理
- **ETF数据**: ETF下载和管理文档
- **MCP设置**: `MCP_SETUP.md` - AI助手集成

## 🎯 适用场景

- **量化研究**: 154个技术指标 + 15个资金流因子
- **策略开发**: 多市场、多时间框架支持
- **因子筛选**: 5维度专业评估体系
- **回测分析**: 与VectorBT深度集成
- **生产部署**: 统一架构确保一致性

---
**维护者**: 量化工程团队 | **更新**: 2025-10-14 | **版本**: v3.0
