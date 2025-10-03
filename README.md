# 🎯 深度量化交易系统

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![VectorBT](https://img.shields.io/badge/VectorBT-0.28.1+-orange.svg)](https://vectorbt.io/)
[![Pandas](https://img.shields.io/badge/Pandas-2.3.2+-blue.svg)](https://pandas.pydata.org/)

一个专业级的量化交易开发环境，专为多市场分析和综合因子筛选而设计。本系统提供154个技术指标的完整技术分析框架，支持A股、港股和美股市场，具备专业级的因子筛选和回测功能。

## ✨ 核心特性

### 🏆 专业级因子筛选系统
- **5维评估体系**: 预测力(35%) + 稳定性(25%) + 独立性(20%) + 实用性(15%) + 短期适应性(5%)
- **154项技术指标**: 涵盖趋势、动量、波动率、成交量等完整指标体系
- **统计严谨性**: Benjamini-Hochberg FDR校正、VIF多重共线性检测、滚动窗口验证

### 🚀 高性能计算
- **VectorBT集成**: 相比传统pandas性能提升10-50倍
- **向量化计算**: 消除Python循环，内存使用优化40-60%
- **多时间框架**: 支持1min到daily的8种时间框架自动对齐

### 🌍 多市场支持
- **A股市场**: 中国股票市场专门分析框架
- **港股市场**: 276+股票，分钟级精度数据
- **美股市场**: 172+股票，多时间框架支持

## 📦 快速安装

### 环境要求
- Python 3.11+
- 推荐使用 [uv](https://github.com/astral-sh/uv) 现代Python包管理器

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/zhangsensen/-0927.git
cd -0927
```

2. **安装依赖**
```bash
# 使用 uv (推荐)
uv sync

# 或使用 pip
pip install -e .
```

3. **安装可选依赖**
```bash
# 性能优化版本
uv sync --extra performance

# 完整版本(包含可视化、Web界面等)
uv sync --extra all
```

4. **激活环境**
```bash
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate     # Windows
```

## 🚀 快速开始

### A股技术分析

```bash
# 运行A股技术分析
python a股/stock_analysis/sz_technical_analysis.py 000001

# 下载A股数据
python a股/data_download/simple_download.py

# 筛选热门A股
python a股/screen_top_stocks.py
```

### 专业因子筛选

```bash
# 单股票因子筛选
python factor_system/factor_screening/cli.py screen 0700.HK 60min

# 批量处理多股票
python factor_system/factor_screening/cli.py batch --symbols 0700.HK,0005.HK,0941.HK --timeframe 60min

# 快速开始
python factor_system/factor_generation/quick_start.py 0700.HK

# 154指标增强计算器
python factor_system/factor_generation/enhanced_factor_calculator.py
```

### 配置管理

```bash
# 列出可用配置
python factor_system/factor_screening/cli.py config list

# 创建自定义配置
python factor_system/factor_screening/cli.py config create my_config.yaml

# 验证配置文件
python factor_system/factor_screening/cli.py config validate my_config.yaml
```

## 📊 系统架构

### 核心组件

```
深度量化0927/
├── a股/                          # A股分析框架
│   ├── stock_analysis/           # 技术分析引擎
│   ├── data_download/           # 数据下载模块
│   └── screen_top_stocks.py     # 股票筛选工具
├── factor_system/               # 专业因子系统
│   ├── factor_generation/       # 因子计算模块
│   │   ├── enhanced_factor_calculator.py  # 154指标引擎
│   │   ├── multi_tf_vbt_detector.py      # VectorBT分析器
│   │   └── quick_start.py                 # 快速入口
│   └── factor_screening/        # 因子筛选模块
│       ├── professional_factor_screener.py  # 5维筛选引擎
│       ├── cli.py                         # 命令行界面
│       └── batch_screener.py              # 批量处理
└── 简单实用.md                    # 使用指南
```

### 154项技术指标体系

#### 核心技术指标 (36项)
- **移动平均线**: MA5, MA10, MA20, MA30, MA60, EMA5, EMA12, EMA26
- **动量指标**: MACD, RSI, Stochastic, Williams %R, CCI, MFI
- **波动率指标**: Bollinger Bands, ATR, Standard Deviation
- **成交量指标**: OBV, Volume SMA, Volume Ratio

#### 增强指标 (118项)
- **高级均线**: DEMA, TEMA, T3, KAMA, Hull MA
- **振荡器**: TRIX, ROC, CMO, ADX, DI+, DI-
- **趋势指标**: Parabolic SAR, Aroon, Chande Momentum
- **统计指标**: Z-Score, Correlation, Beta, Alpha
- **周期指标**: Hilbert Transform, Sine Wave, Trendline

## 📈 因子筛选评估体系

### 5维综合评分

| 维度 | 权重 | 评估内容 |
|------|------|----------|
| **预测力** | 35% | 多周期IC分析、IC衰减、持续性指标 |
| **稳定性** | 25% | 滚动窗口IC、横截面稳定性、一致性 |
| **独立性** | 20% | VIF检测、因子相关性、信息增量 |
| **实用性** | 15% | 交易成本、换手率、流动性要求 |
| **短期适应性** | 5% | 反转效应、动量持续性、波动率敏感性 |

### 因子质量分级

- 🥇 **Tier 1** (综合评分 ≥ 0.8): 核心因子，强烈推荐
- 🥈 **Tier 2** (0.6-0.8): 重要因子，推荐使用
- 🥉 **Tier 3** (0.4-0.6): 备用因子，谨慎使用
- ❌ **不推荐** (< 0.4): 不建议使用

### 统计显著性

- ***** p < 0.001: 高度显著
- **** p < 0.01: 显著
- *** p < 0.05: 边缘显著
- 无标记: 不显著

## 🛠️ 开发指南

### 代码质量

```bash
# 代码格式化
black factor_system/ a股/

# 导入排序
isort factor_system/ a股/

# 类型检查
mypy factor_system/ a股/

# 运行测试
pytest

# 测试覆盖率
pytest --cov=factor_system
```

### 性能基准

| 规模 | 样本数×因子数 | 处理速度 |
|------|---------------|----------|
| 小规模 | 500×20 | 831+ 因子/秒 |
| 中规模 | 1000×50 | 864+ 因子/秒 |
| 大规模 | 2000×100 | 686+ 因子/秒 |
| 超大规模 | 5000×200 | 370+ 因子/秒 |

### 完整筛选流程性能
- **处理速度**: 5.7 因子/秒 (80个因子完整分析)
- **内存使用**: < 1MB (中等规模数据)
- **主要瓶颈**: 滚动IC计算 (94.2%时间占比)

## 📚 使用示例

### 示例1: 港股因子分析

```python
from factor_system.factor_screening.professional_factor_screener import ProfessionalFactorScreener

# 创建筛选器实例
screener = ProfessionalFactorScreener()

# 运行因子筛选
results = screener.screen_factor(
    symbol="0700.HK",
    timeframe="60min",
    start_date="2024-01-01",
    end_date="2024-12-31"
)

# 查看顶级因子
top_factors = results.get_top_factors(n=10)
print(f"顶级因子: {top_factors}")
```

### 示例2: A股技术分析

```python
from a股.stock_analysis.sz_technical_analysis import SZTechnicalAnalysis

# 创建分析器
analyzer = SZTechnicalAnalysis("000001")

# 计算154项技术指标
indicators = analyzer.calculate_all_indicators()

# 生成分析报告
report = analyzer.generate_report(indicators)
print(report)
```

### 示例3: 批量股票筛选

```python
from factor_system.factor_screening.batch_screener import BatchScreener

# 定义股票池
symbols = ["0700.HK", "0005.HK", "0941.HK", "1398.HK", "2318.HK"]

# 批量筛选
batch_screener = BatchScreener()
results = batch_screener.screen_multiple(
    symbols=symbols,
    timeframe="daily",
    factor_count=50
)

# 获取综合排名
ranking = results.get_comprehensive_ranking()
print("股票综合排名:", ranking)
```

## 🎯 性能优化建议

### 计算优化
- 使用VectorBT进行向量化操作
- 避免DataFrame.apply，使用内置函数
- 启用numba JIT编译加速
- 合理使用缓存机制

### 内存优化
- 使用parquet格式存储数据
- 及时释放不需要的变量
- 分批处理大型数据集
- 监控内存使用情况

## 📋 数据要求

### 数据格式
- **A股**: `{SYMBOL_CODE}_1d_YYYY-MM-DD.csv`
- **港股**: `{SYMBOL}_1min_YYYY-MM-DD_YYYY-MM-DD.parquet`
- **输出**: `{SYMBOL}_{TIMEFRAME}_factors_YYYY-MM-DD_HH-MM-SS.parquet`

### 数据质量
- 无未来函数偏差
- 处理幸存者偏差
- 真实市场数据(非模拟数据)
- 跨时间框架正确对齐

## 🔧 配置说明

### 策略配置模板
- `long_term_config.yaml`: 长期投资策略
- `conservative_config.yaml`: 保守交易策略
- `high_frequency_config.yaml`: 高频交易策略
- `aggressive_config.yaml`: 激进交易策略

### 自定义配置
```yaml
# 示例配置文件
data:
  market: "HK"
  symbols: ["0700.HK", "0005.HK"]
  timeframe: "60min"

factors:
  enable_all: true
  custom_weights:
    predictability: 0.4
    stability: 0.3
    independence: 0.2
    practicality: 0.1

screening:
  min_ic_mean: 0.02
  max_turnover: 12.0
  significance_level: 0.05
```

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 开发规范
- 遵循PEP 8代码风格
- 添加适当的测试用例
- 更新相关文档
- 确保所有测试通过

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🙏 致谢

- [VectorBT](https://vectorbt.io/) - 高性能回测框架
- [pandas](https://pandas.pydata.org/) - 数据处理框架
- [TA-Lib](https://ta-lib.org/) - 技术分析库
- [yfinance](https://pypi.org/project/yfinance/) - 金融数据接口

## 📞 联系方式

- 项目主页: [GitHub](https://github.com/zhangsensen/-0927)
- 问题反馈: [Issues](https://github.com/zhangsensen/-0927/issues)
- 文档: [Wiki](https://github.com/zhangsensen/-0927/wiki)

---

⚡ **开始您的量化交易之旅！**

*本系统专为严肃的算法交易研究而设计，提供专业级因子分析能力，针对多个市场进行优化。*
