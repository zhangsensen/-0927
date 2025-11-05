# 深度量化0927 - 量化交易研究平台

## 项目概述

本项目是一个专业级量化交易开发环境,包含多个独立子系统,每个系统专注于特定市场或策略类型。

**核心理念**: Linus Torvalds工程原则 - 消除特殊案例,提供实用解决方案,确保代码在实际市场中可靠运行。

## 🎯 核心子系统

### 1. **ETF轮动系统** (`etf_rotation_optimized/`) ⭐ 主力系统
- **状态**: ✅ 生产就绪
- **市场**: 中国ETF市场 (43只核心ETF)
- **策略**: 横截面因子轮动 + WFO优化
- **特性**: 
  - 18个精选因子 (趋势/动量/波动/资金流)
  - IC加权 + 多策略枚举
  - 交易级胜率计算
  - 事件驱动组合构建
- **文档**: [详见 etf_rotation_optimized/README.md](etf_rotation_optimized/README.md)

### 2. **FactorEngine统一框架** (`factor_system/`)
- **状态**: 维护模式
- **功能**: 154个技术指标 + 15个资金流因子
- **市场**: A股、港股、ETF多市场支持
- **特性**: 统一API、双重缓存、向量化计算

### 3. **A股策略** (`a_shares_strategy/`)
- **状态**: 研发中
- **市场**: A股市场
- **特性**: T+1约束、资金流因子

### 4. **港股中频** (`hk_midfreq/`)
- **状态**: 研发中  
- **市场**: 港股市场 (276+股票)
- **频率**: 1分钟到日线

### 5. **ETF数据管理** (`etf_download_manager/`)
- **功能**: ETF数据下载和管理
- **支持**: 自定义日期范围、增量更新

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

## 🚀 快速开始 - ETF轮动系统

ETF轮动系统是当前主力系统,推荐从这里开始:

```bash
# 进入ETF轮动目录
cd etf_rotation_optimized

# 安装依赖
make install

# 运行完整pipeline
make run
```

详细文档: [etf_rotation_optimized/README.md](etf_rotation_optimized/README.md)

## 📁 项目结构

```
深度量化0927/
├── README.md                      # 本文件
├── zen_mcp_使用指南.md            # MCP工具使用指南
│
├── etf_rotation_optimized/        # ⭐ 主力系统: ETF轮动
│   ├── core/                      # 核心引擎 (28个模块)
│   ├── configs/                   # 配置文件
│   ├── docs/                      # 详细文档
│   ├── tests/                     # 单元测试
│   └── results/                   # 运行结果
│
├── factor_system/                 # FactorEngine框架 (维护)
├── a_shares_strategy/             # A股策略 (研发中)
├── hk_midfreq/                    # 港股中频 (研发中)
├── etf_download_manager/          # ETF数据管理
│
├── production/                    # 生产配置
├── cache/                         # 缓存目录
└── results/                       # 历史结果
```

## 📖 文档导航

### ETF轮动系统核心文档
- [快速开始](etf_rotation_optimized/README.md) - 5分钟上手
- [项目结构](etf_rotation_optimized/PROJECT_STRUCTURE.md) - 代码架构
- [交易指南](etf_rotation_optimized/EVENT_DRIVEN_TRADING_GUIDE.md) - 事件驱动交易
- [胜率功能](etf_rotation_optimized/DELIVERY_DOCUMENT.md) - 交易级胜率计算

### 完整文档索引
- [文档索引](etf_rotation_optimized/docs/INDEX.md) - 所有文档列表
- [部署指南](etf_rotation_optimized/docs/DEPLOYMENT.md) - 生产部署
- [运维手册](etf_rotation_optimized/docs/OPERATIONS.md) - 日常运维

## 🔧 技术栈

- **语言**: Python 3.11+
- **数据处理**: Pandas, NumPy, Numba
- **配置管理**: YAML
- **测试**: pytest
- **构建**: Make, uv

## 📊 系统对比

| 系统 | 状态 | 市场 | 策略类型 | 推荐度 |
|------|------|------|----------|--------|
| ETF轮动 | ✅ 生产 | ETF | 横截面轮动 | ⭐⭐⭐⭐⭐ |
| FactorEngine | 🔧 维护 | A股/港股/ETF | 因子框架 | ⭐⭐⭐ |
| A股策略 | 🚧 研发 | A股 | 资金流 | ⭐⭐ |
| 港股中频 | 🚧 研发 | 港股 | 中高频 | ⭐⭐ |

## 💡 开发建议

1. **新用户**: 从ETF轮动系统开始,文档完善、代码清晰
2. **策略研发**: 使用`etf_rotation_optimized/configs/experiments/`进行实验
3. **因子开发**: 在`core/precise_factor_library_v2.py`添加新因子
4. **回测验证**: 使用`vectorbt_backtest/`进行暴力回测

## 🤝 贡献指南

- 代码风格: 遵循PEP 8
- 提交信息: 简洁明了,描述清晰
- 测试要求: 新功能必须包含单元测试
- 文档更新: 代码修改同步更新文档

---

**最后更新**: 2025-11-04
**维护者**: 深度量化团队

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
