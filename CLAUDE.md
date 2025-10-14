# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **professional-grade quantitative trading development environment** with a unified factor calculation engine designed for multi-market algorithmic trading research. The system ensures 100% consistency between research, backtesting, and production environments through a unified FactorEngine architecture.

**Core Philosophy**: Linus Torvalds engineering principles - eliminate special cases, practical solutions, clean code that works in real markets.

## Key Commands

### Environment Setup
```bash
# Install dependencies using uv (modern Python package manager)
uv sync

# Activate virtual environment
source .venv/bin/activate

# Development installation with all tools
uv sync --group all

# Install FactorEngine in development mode
pip install -e .
```

### Code Quality & Testing
```bash
# 统一代码质量检查 (推荐)
bash scripts/unified_quality_check.sh

# 运行测试
pytest

# Run tests with coverage
pytest --cov=factor_system

# Type checking
mypy factor_system/

# Code formatting (Black - 88 char line length)
black factor_system/ data-resampling/

# Import sorting
isort factor_system/ data-resampling/

# Run pre-commit hooks
pre-commit run --all-files

# FactorEngine consistency validation
python tests/test_factor_engine_consistency.py
```

### Core System Validation
```bash
# Verify path management system
python -c "from factor_system.utils import get_project_root, get_raw_data_dir; print('✅ Path system OK')"

# Verify exception handling
python -c "from factor_system.utils import safe_operation; print('✅ Exception handling OK')"

# Verify configuration loading
python -c "from factor_system.factor_generation.config_loader import load_config; print('✅ Config system OK')"

# Run comprehensive system validation
python scripts/test_root_cause_fixes.py
```

### Factor Generation & Analysis
```bash
# Single stock factor generation (154 indicators, multiple timeframes)
cd factor_system/factor_generation
python run_single_stock.py 0700.HK

# Batch factor processing
python run_batch_processing.py

# Complete pipeline with data resampling
python run_complete_pipeline.py 0700.HK

# Quick start analysis
python quick_start.py 0700.HK
```

### Factor Screening System
```bash
# Professional 5-dimension factor screening
cd factor_system/factor_screening
python professional_factor_screener.py --symbol 0700.HK --timeframe 60min

# Batch screening multiple stocks
python batch_screener.py --symbols 0700.HK,0005.HK,0941.HK

# CLI interface
python cli.py screen 0700.HK 60min
python cli.py batch --symbols 0700.HK,0005.HK --timeframe 60min
```

### A-Share Market Analysis
```bash
# Technical analysis with 154 indicators
python a股/stock_analysis/sz_technical_analysis.py <STOCK_CODE>

# Data download
python a股/data_download/download_603920.py

# Market screening
python a股/实施快速启动.py

# Comprehensive analysis
python a股/stock_analysis/ultimate_300450_analysis.py
```

### Money Flow Analysis (A-Share Focus)
```bash
# Quick start with money flow factors
python examples/moneyflow_quickstart.py

# Generate money flow factors
python scripts/produce_money_flow_factors.py

# Run comprehensive money flow integration test
python scripts/test_moneyflow_integration_comprehensive.py

# Verify T+1 execution constraints
python scripts/verify_t_plus_1.py

# Production run for money flow factors
python scripts/run_money_flow_only.py
```

### ETF Data Management
```bash
# Download ETF daily data (2 years of historical data)
python download_etf_final.py

# Generate ETF money flow estimates (multi-factor model)
python etf_moneyflow_final_solution.py

# Test ETF money flow detection (debugging interface limitations)
python test_single_day_etf.py

# Analyze ETF code formats and data availability
python analyze_etf_codes.py

# Investigate 301-series codes (创业板 stocks, not ETFs)
python investigate_301_codes.py
```

## Architecture Overview

### Unified Path Management System
**Critical**: All paths must use the unified path management system in `factor_system/utils/project_paths.py`:

```python
from factor_system.utils import get_project_root, get_raw_data_dir, get_factor_output_dir

# Get project root directory
project_root = get_project_root()

# Get data directories
raw_data_dir = get_raw_data_dir()
factor_output_dir = get_factor_output_dir()
screening_results_dir = get_screening_results_dir()
```

**Never use hardcoded paths** like `"../raw"` or `"../factor_output"`.

### FactorEngine: Core Architecture

**Unified Calculation Core** (`factor_system/factor_engine/`):
- `api.py` - **Single entry point** for all factor calculations
- `core/engine.py` - Main calculation engine with dual-layer caching
- `core/registry.py` - Factor registration and metadata management
- `providers/` - Pluggable data providers (Parquet, CSV)
- `factors/` - 100+ technical indicators organized by category
- `settings.py` - Environment configuration with Pydantic models

**Key Principle**: Research, backtesting, and production all use the exact same FactorEngine instance to eliminate calculation discrepancies.

### Factor Generation System (`factor_system/factor_generation/`)
- **154 Technical Indicators**: Complete TA-Lib implementation plus custom indicators
- **Multi-Timeframe Support**: 1min to daily, automatic resampling
- **Batch Processing**: Parallel processing with configurable workers
- **Configuration**: YAML-based with comprehensive parameter control

### Professional Factor Screening (`factor_system/factor_screening/`)
- **5-Dimension Framework**: Predictive Power, Stability, Independence, Practicality, Short-term Adaptability
- **Statistical Rigor**: Benjamini-Hochberg FDR correction, VIF analysis, IC decay
- **Cost Modeling**: Commission, slippage, market impact for Hong Kong market
- **Performance Optimization**: VectorBT integration, parallel screening
- **Money Flow Integration**: A-share money flow factors with T+1 execution constraints

### Exception Handling Framework
Use the unified exception handling system in `factor_system/utils/error_utils.py`:

```python
from factor_system.utils import safe_operation, FactorSystemError, ConfigurationError

@safe_operation
def calculate_factors():
    # Your factor calculation logic
    return result

# Custom exceptions
raise ConfigurationError("Missing required configuration: data_path")
raise FactorSystemError("Factor calculation failed")
```

## Key Design Principles

### Linus-Style Engineering Standards
1. **Eliminate Special Cases**: Use data structures instead of if/else branches
2. **Never Break Userspace**: API stability is paramount
3. **Practical Solutions**: Solve real problems, don't create concepts
4. **Simplicity as Weapon**: Short functions, clear naming, minimal complexity
5. **Code as Truth**: All assumptions must be verifiable in backtesting

### Quantitative Engineering Standards (from Cursor Rules)
- **No Future Function**: Strict temporal alignment, no lookahead bias (CRITICAL)
- **Statistical Rigor**: Benjamini-Hochberg FDR correction mandatory
- **154 Indicators**: 36 core + 118 enhanced, vectorized implementation
- **5-Dimension Screening**: Predictive power, stability, independence, practicality, adaptability
- **Performance**: VectorBT > loops, memory >70% efficiency
- **Code Quality**: Functions <50 lines, complexity <10, type hints required

### Performance Requirements
- **Vectorization Rate**: >95% of operations must be vectorized
- **Critical Path**: <1ms for single factor calculations
- **Memory Efficiency**: Use VectorBT and Polars for large datasets
- **Never use DataFrame.apply()**: Use built-in vectorized operations

### Code Quality Standards
- **Indentation**: Maximum 3 levels
- **Function Length**: Keep functions short and focused
- **No Magic Numbers**: All parameters must be explicit and configurable
- **Logging Over Comments**: Systems should be self-documenting through structured logs

## Configuration Management

### Environment Variables
FactorEngine uses environment variables for deployment flexibility:

```bash
export FACTOR_ENGINE_RAW_DATA_DIR="/data/market/raw"
export FACTOR_ENGINE_MEMORY_MB="1024"
export FACTOR_ENGINE_N_JOBS="-1"
export FACTOR_ENGINE_CACHE_DIR="/cache/factors"
```

### Pre-configured Environments
- **Development**: 200MB cache, 2-hour TTL, single-thread, verbose logging
- **Research**: 512MB cache, 24-hour TTL, 4-core parallel, info logging
- **Production**: 1GB cache, 7-day TTL, all-core parallel, warning-only logging

### Factor Generation Configuration
Configuration is managed through Python classes with YAML support:

```python
from factor_system.factor_generation.config_loader import load_config

# Load configuration
config = load_config()

# Create custom configurations
full_config = create_full_config()  # All indicators enabled
basic_config = create_basic_config()  # Core indicators only
```

## Data Structure & File Organization

### File Naming Conventions
- **A-Share Data**: `{SYMBOL_CODE}_1d_YYYY-MM-DD.csv`
- **Hong Kong Data**: `{SYMBOL}_1min_YYYY-MM-DD_YYYY-MM-DD.parquet`
- **Factor Output**: `{SYMBOL}_{TIMEFRAME}_factors_YYYY-MM-DD_HH-MM-SS.parquet`
- **ETF Daily Data**: `{SYMBOL}_daily_YYYYMMDD_YYYYMMDD.parquet`
- **ETF Money Flow**: `{SYMBOL}_moneyflow_YYYYMMDD_YYYYMMDD.parquet`

### Directory Structure
```
深度量化0927/
├── raw/                          # Raw OHLCV data by market
│   ├── HK/                       # Hong Kong stocks (276+ stocks)
│   ├── US/                       # US stocks (172+ stocks)
│   ├── A股/                      # A-share stocks
│   │   └── SH/money_flow/        # A-share money flow data (parquet)
│   └── ETF/                      # ETF data storage
│       ├── daily/                # ETF daily price data (parquet)
│       ├── moneyflow/            # ETF money flow estimates (parquet)
│       ├── moneyflow_market/     # Market money flow data (parquet)
│       └── summary/              # Download summaries (json)
├── factor_system/
│   ├── factor_engine/            # Unified factor calculation core
│   │   └── factors/money_flow/   # Money flow factor implementations
│   ├── factor_generation/        # Factor generation pipeline
│   ├── factor_screening/         # Professional factor screening
│   └── utils/                    # Path management & error handling
├── factor_system/factor_output/  # Generated factor files
├── examples/                     # Usage examples (moneyflow_quickstart.py)
├── scripts/                      # Utility and testing scripts
└── factor_system/factor_screening/screening_results/  # Screening results
```

## FactorEngine API Usage

### Recommended Usage Pattern
```python
from factor_system.factor_engine import api
from datetime import datetime

# ✅ RECOMMENDED: Use unified API
factors = api.calculate_factors(
    factor_ids=["RSI", "MACD", "STOCH"],
    symbols=["0700.HK", "0005.HK"],
    timeframe="15min",
    start_date=datetime(2025, 9, 1),
    end_date=datetime(2025, 9, 30)
)

# List available factors
available_factors = api.list_available_factors()
factor_categories = api.list_factor_categories()

# Calculate money flow factors (A-share)
money_flow_factors = api.calculate_factors(
    factor_ids=["MainNetInflow_Rate", "LargeOrder_Ratio", "Flow_Price_Divergence"],
    symbols=["000001.SZ", "600036.SH"],
    timeframe="daily",
    start_date=datetime(2025, 9, 1),
    end_date=datetime(2025, 9, 30)
)
```

### Performance Optimization
```python
# Pre-warm cache for frequently used factors
api.prewarm_cache(
    factor_ids=["RSI", "MACD", "STOCH"],
    symbols=["0700.HK", "0005.HK"],
    timeframe="15min",
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 12, 31)
)

# Monitor cache performance
stats = api.get_cache_stats()
print(f"Cache hit rate: {stats['memory_hit_rate']:.2%}")
```

## Testing Strategy

### Test Categories
```bash
# Unit tests - Fast, isolated component tests
pytest -m unit

# Integration tests - Component interaction tests
pytest -m integration

# Slow tests - Full pipeline tests with real data
pytest -m slow

# Coverage reporting
pytest --cov=factor_system --cov-report=html
```

### Key Validation Scripts
- `tests/test_factor_engine_consistency.py` - Ensure FactorEngine matches factor_generation
- `tests/test_factor_consistency_v2.py` - Factor calculation consistency validation
- `scripts/test_root_cause_fixes.py` - End-to-end system validation
- `scripts/migrate_parquet_schema.py` - Data migration utilities
- `scripts/test_moneyflow_integration_comprehensive.py` - Money flow integration testing
- `scripts/verify_t_plus_1.py` - T+1 execution constraint verification

### ETF Data Validation Scripts
- `download_etf_final.py` - Main ETF data download with 2-year historical coverage (19 ETFs)
- `etf_moneyflow_final_solution.py` - Advanced ETF money flow estimation system (multi-factor model)
- `test_single_day_etf.py` - Debug interface limitations and data availability
- `analyze_etf_codes.py` - Comprehensive ETF code format analysis (discovered 301-series = ChiNext stocks)
- `investigate_301_codes.py` - Analysis of 301-series codes (创业板 stocks, not ETFs)
- `etf_download_list.py` - Complete ETF inventory with priority ratings and metadata

## Market-Specific Considerations

### Hong Kong Market (Primary Focus)
- **Commission**: 0.2%
- **Stamp Duty**: 0.1%
- **Slippage**: 0.05 HKD per share
- **Trading Hours**: 9:30-12:00, 13:00-16:00 HKT
- **Settlement**: T+2

### A-Share Market (Money Flow Focus)
- **Money Flow Factors**: 12 core + enhanced factors with T+1 execution constraints
- **Signal Freeze**: 14:30 signal generation freeze for next-day execution
- **Tradability Mask**: Automatic filtering of non-tradable samples
- **Data Source**: SH money flow data in parquet format
- **Settlement**: T+1

### ETF Market (Exchange-Traded Funds)
- **ETF Daily Data**: Downloaded via Tushare Pro `fund_daily` interface with `asset='FD'`
- **Money Flow Estimation**: Multi-factor model using volume, price, and trading patterns
- **Available ETFs**: 19 core ETFs across market caps, sectors, and themes
- **Data Coverage**: 2 years of historical data with signal strength classification
- **Estimation Method**: Volume anomaly (40%), volume factor (30%), price breakout (20%), momentum (10%)
- **Important Note**: Direct ETF money flow data not available through standard Tushare moneyflow interfaces

### Data Requirements
- **No Look-ahead Bias**: All calculations must use only historical data
- **Survivorship Bias**: Proper handling of delisted stocks
- **Time Zone Consistency**: All timestamps in HKT
- **Corporate Actions**: Proper adjustment for splits, dividends

## 代码质量保障体系

### 统一质量检查工具
项目集成了基于pyscn和Vulture的专业代码质量检查工具：

```bash
# 运行统一质量检查 (推荐)
bash scripts/unified_quality_check.sh

# 查看质量标准文档
cat CODE_QUALITY_STANDARDS.md
```

### 质量检查覆盖范围
- **pyscn深度分析**: 使用CFG和APTED算法分析代码复杂度和架构合规性
- **Vulture死代码检测**: 自动识别未使用的代码和导入
- **量化安全检查**: 未来函数检测、时间安全验证、因子清单合规
- **基础质量检查**: Python语法、代码格式、导入排序
- **性能分析**: 代码复杂度和性能瓶颈识别

### Git自动化钩子
项目配置了多层Git钩子确保代码质量：

```bash
# 安装pre-commit钩子
pre-commit install

# Pre-commit检查 (快速检查)
git commit  # 自动运行基础安全检查和质量验证

# Pre-push检查 (全面检查)
git push  # 自动运行完整质量分析套件
```

### 质量评分体系
- **当前健康评分**: 85/100 (A级)
- **复杂度评分**: 70/100 (平均9.45，目标≤8)
- **代码重复**: 6.4% (目标<2%)
- **架构合规**: 73% (目标>90%)

### 核心安全红线
- **未来函数检测**: 严禁使用未来数据，确保回测有效性
- **T+1时间安全**: 严格执行交易日延迟，防止时间泄露
- **因子清单合规**: FactorEngine必须严格遵循官方因子清单

## Performance Benchmarks

### Factor Calculation Performance
- **Small Scale** (500 samples × 20 factors): 831+ factors/second
- **Medium Scale** (1000 samples × 50 factors): 864+ factors/second
- **Large Scale** (2000 samples × 100 factors): 686+ factors/second
- **Extra Large** (5000 samples × 200 factors): 370+ factors/second

### Complete Screening Process
- **Processing Speed**: 5.7 factors/second (80 factors full analysis)
- **Memory Usage**: < 1MB (medium scale data)
- **Main Bottleneck**: Rolling IC calculation (94.2% of time)

## Development Workflow

1. **Always use unified path management** - Never hardcode paths
2. **Add exception handling** using the framework decorators
3. **Write tests** for new functionality
4. **Run validation scripts** before committing
5. **Use FactorEngine API** for all factor calculations
6. **Follow Linus-style principles** - eliminate complexity

## Troubleshooting

### Common Issues
- **Import Errors**: Check path management system usage
- **Configuration Issues**: Verify YAML syntax and required fields
- **Performance Issues**: Check for DataFrame.apply usage
- **Data Issues**: Verify file naming conventions and directory structure

### Validation Commands
```bash
# Quick system health check
python -c "
from factor_system.utils import get_project_root, safe_operation
from factor_system.factor_engine import api
print('✅ System healthy')
"
```

This quantitative trading platform is designed for serious algorithmic trading research with professional-grade factor analysis capabilities. The unified FactorEngine ensures complete consistency across research, backtesting, and production environments while following Linus Torvalds engineering principles of simplicity, practicality, and reliability.

## ETF Data Management

**Important Discovery**: Tushare Pro's standard `moneyflow` interface does not include ETF data - it only covers individual stocks. The 301-series codes found in moneyflow data are 创业板 (ChiNext) stocks, not ETFs.

**Solution Implemented**: Advanced multi-factor ETF money flow estimation system using:
- Volume anomaly detection (40% weight)
- Volume factor analysis (30% weight)
- Price breakout patterns (20% weight)
- Momentum continuity (10% weight)

**Available ETF Data**: 19 core ETFs with 2 years of historical data, signal strength classification, and large order estimation. Data stored in `raw/ETF/` with separate directories for daily price data and estimated money flow data.

## Money Flow Factor System (A-Share Focus)

The project includes a complete production-grade money flow factor system for A-shares with:

**Core Architecture**:
- **MoneyFlowProvider**: Unified data loading with standardized processing
- **12 Core Factors**: Including MainNetInflow_Rate, LargeOrder_Ratio, Flow_Price_Divergence
- **4 Enhanced Factors**: Institutional absorption, flow tier analysis, reversal ratios
- **3 Constraint Factors**: Gap signals, end-of-day trading, tradability masks

**Key Features**:
- **T+1 Execution Compliance**: 14:30 signal generation freeze for next-day execution
- **Tradability Filtering**: Automatic filtering of non-tradable samples
- **Multi-Timeframe Support**: 1min to monthly data processing
- **Vectorized Implementation**: 100% vectorized calculations for performance

**Available Timeframes**: 1min, 5min, 15min, 30min, 60min, 120min, 240min, daily, weekly, monthly

**Usage**:
```bash
python examples/moneyflow_quickstart.py  # Quick start demo
pytest -v  # Run comprehensive tests
```