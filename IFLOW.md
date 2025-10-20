# iFlow CLI - 深度量化交易开发环境

## 🎯 项目概览

**深度量化0927** 是一个专业级量化交易开发环境，提供统一的因子计算引擎，支持多市场算法化交易研究。系统核心是**FactorEngine**架构，确保研究、回测和生产环境的100%一致性。

**核心理念**：Linus Torvalds工程原则 - 消除特殊案例，提供实用解决方案，确保代码在实际市场中可靠运行。

## 🚀 核心系统组件

### 1. **FactorEngine 统一架构**
- ✅ **API接口**: `factor_system/factor_engine/api.py` - 单一入口点
- ✅ **核心引擎**: 支持双重缓存，高性能因子计算
- ✅ **因子注册表**: 154个技术指标 + 15个资金流因子 + 206个ETF横截面因子
- ✅ **数据提供者**: Parquet、CSV、分钟数据、资金流数据

### 2. **多市场支持**
- ✅ **港股市场**: 276+股票，1分钟到月线数据
- ✅ **A股市场**: 资金流因子系统，T+1执行约束
- ✅ **ETF市场**: 19个核心ETF，2年历史数据，横截面因子分析

### 3. **因子计算系统**
- ✅ **154个技术指标**: 36核心 + 118增强指标
- ✅ **15个资金流因子**: 8核心 + 4增强 + 3约束因子
- ✅ **206个ETF横截面因子**: 20传统 + 174动态因子
- ✅ **多时间框架**: 1min到monthly，自动重采样
- ✅ **向量化实现**: >95%向量化率

### 4. **专业因子筛选**
- ✅ **5维度评估**: 预测力、稳定性、独立性、实用性、适应性
- ✅ **统计严谨性**: Benjamini-Hochberg FDR校正，VIF分析
- ✅ **成本建模**: 港股佣金、印花税、滑点模型

### 5. **质量保障体系**
- ✅ **统一质量检查**: pyscn + Vulture + 量化安全检查
- ✅ **Git自动化钩子**: pre-commit基础检查 + pre-push全面检查
- ✅ **性能基准**: 300-800+ 因子/秒计算速度
- ✅ **架构合规**: 73%架构合规率，目标>90%

## 📊 技术特性

- **向量化计算**: 100%采用Pandas/NumPy向量化，无循环
- **统一口径**: 所有占比因子分母锁死为`turnover_amount`
- **无前视偏差**: 严格遵循14:30信号冻结，T+1执行
- **可交易性约束**: `tradability_mask`有效过滤不可交易样本
- **自动化测试**: 覆盖数据口径、因子计算、T+1执行等关键环节
- **多时间框架**: 支持 1min/5min/15min/30min/60min/120min/240min/daily/weekly/monthly

## 🛠️ 开发环境

### 环境安装
```bash
# 使用uv现代包管理器
uv sync

# 激活虚拟环境
source .venv/bin/activate

# 开发安装（包含所有工具）
uv sync --group all
```

### 代码质量工具
```bash
# 统一代码质量检查 (推荐)
bash scripts/unified_quality_check.sh

# 代码格式化
black factor_system/
isort factor_system/

# 类型检查
mypy factor_system/

# 运行pre-commit钩子
pre-commit run --all-files
```

### 性能基准
- **因子计算速度**: 300-800+ 因子/秒
- **内存效率**: >70%利用率（Polars优化）
- **缓存命中率**: >90%（智能预热）

## 🎯 快速开始

### 1. 因子计算（推荐方式）
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

# 计算ETF横截面因子（206个因子）
from factor_system.factor_engine.factors.etf_cross_section import create_etf_cross_section_manager
etf_manager = create_etf_cross_section_manager()
etf_cross_section = etf_manager.calculate_factors(
    symbols=["510300.SH", "159919.SZ", "512100.SH"],
    timeframe="daily",
    start_date=datetime(2025, 9, 1),
    end_date=datetime(2025, 9, 30)
)
```

### 2. 批量因子生成
```bash
# 港股单股票因子生成
cd factor_system/factor_generation
python run_single_stock.py 0700.HK

# A股因子生成
python a_shares_strategy/generate_a_share_factors.py 600036.SH --timeframe 5min

# ETF数据处理
python etf_download_manager/download_etf_final.py
```

### 3. 因子筛选
```bash
# 专业5维度因子筛选
cd factor_system/factor_screening
python professional_factor_screener.py --symbol 0700.HK --timeframe 60min

# ETF横截面因子筛选
python etf_factor_engine_production/scripts/legacy_production_full_cross_section.py
```

### 4. 系统验证
```bash
# 运行测试套件
pytest -v

# FactorEngine一致性验证
python tests/test_factor_engine_consistency.py

# ETF横截面系统冒烟测试
python scripts/comprehensive_smoke_test.py

# 路径管理系统验证
python -c "from factor_system.utils import get_project_root; print('✅ 系统正常')"
```

## 📁 项目结构

```
深度量化0927/
├── factor_system/                # 核心因子系统
│   ├── factor_engine/           # 统一因子计算引擎
│   │   └── factors/etf_cross_section/  # ETF横截面因子系统 (206因子)
│   ├── factor_generation/       # 批量因子生成
│   ├── factor_screening/        # 专业因子筛选
│   └── utils/                   # 路径管理和工具
├── a_shares_strategy/           # A股策略框架
├── etf_download_manager/        # ETF数据管理
├── etf_factor_engine_production/ # ETF因子引擎生产环境
├── hk_midfreq/                  # 港股中频策略
├── examples/                    # 使用示例
├── scripts/                     # 工具脚本
│   ├── unified_quality_check.sh # 统一质量检查
│   └── comprehensive_smoke_test.py # ETF系统测试
├── tests/                       # 测试套件
├── raw/                         # 原始数据存储
└── docs/                        # 详细文档
```

## 🔧 核心配置文件

### pyproject.toml
- **Python版本**: ≥3.11
- **核心依赖**: vectorbt, pandas, numpy, ta-lib, polars
- **开发工具**: pytest, black, mypy, pre-commit
- **质量工具**: pyscn≥1.1.1, vulture≥2.14

### Makefile
```bash
make help     # 显示帮助信息
make install  # 安装开发依赖
make format   # 格式化代码
make lint     # 运行代码检查
make test     # 运行测试
make check    # 运行所有质量检查
```

## 📈 性能指标

### 质量评分体系
- **当前健康评分**: 85/100 (A级)
- **复杂度评分**: 70/100 (平均9.45，目标≤8)
- **代码重复**: 6.4% (目标<2%)
- **架构合规**: 73% (目标>90%)

### 计算性能
- **小规模化** (500×20因子): 831+ 因子/秒
- **中等规模** (1000×50因子): 864+ 因子/秒
- **大规模** (2000×100因子): 686+ 因子/秒
- **超大规模** (5000×200因子): 370+ 因子/秒

## 🛡️ 安全与合规

### 核心安全红线
- **未来函数检测**: 严禁使用未来数据，确保回测有效性
- **T+1时间安全**: 严格执行交易日延迟，防止时间泄露
- **因子清单合规**: FactorEngine必须严格遵循官方因子清单

### Git自动化质量门
- **Pre-commit检查**: 基础安全检查和质量验证
- **Pre-push检查**: 完整质量分析套件（pyscn + Vulture + 量化检查）

## 📚 相关文档

- **核心指南**: `CLAUDE.md` - 完整项目指导和技术细节
- **A股系统**: `README_A_SHARES.md` - A股专门说明
- **因子集**: `README_FACTOR_SETS.md` - 因子集管理
- **ETF横截面**: `etf_factor_engine_production/` - ETF因子系统文档
- **质量检查**: `scripts/unified_quality_check.sh` - 质量检查工具

## 🎯 适用场景

- **量化研究**: 154个技术指标 + 15个资金流因子 + 206个ETF横截面因子
- **策略开发**: 多市场、多时间框架支持
- **因子筛选**: 5维度专业评估体系
- **回测分析**: 与VectorBT深度集成
- **生产部署**: 统一架构确保一致性
- **质量保障**: 专业级代码质量检查体系

---

**维护者**: 量化工程团队 | **更新**: 2025-10-17 | **版本**: v3.1

**核心优势**: 统一架构 + 多市场支持 + 专业质量保障 + 生产级可靠性