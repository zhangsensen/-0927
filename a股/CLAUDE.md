# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **comprehensive A-share (Chinese stock market) technical analysis framework** designed for systematic quantitative analysis of Chinese stocks. The project provides a complete technical analysis workflow including data downloading, technical indicator calculation, performance evaluation, and investment recommendation generation.

## Key Commands

### Environment Setup
```bash
# Install dependencies using uv (modern Python package manager)
uv sync

# Activate virtual environment
source .venv/bin/activate

# Run tests
pytest

# Run tests with coverage
pytest --cov=factor_system

# Type checking
mypy factor_system/ data-resampling/

# Code formatting
black factor_system/ data-resampling/

# Import sorting
isort factor_system/ data-resampling/
```

### Data Processing
```bash
# Download A-share stock data
python simple_download.py

# Run technical analysis for specific stock
python 300450_sz_technical_analysis.py 300450.SZ

# Batch resample all HK 1-minute data to higher timeframes (parent project)
python batch_resample_hk.py
```

## Architecture Overview

### Core Components

**A-Share Technical Analysis (`A股/`)**
- `300450_sz_technical_analysis.py` - Comprehensive technical analysis with 154 indicators
- `simple_download.py` - Data downloading tool for A-share stocks
- `A股技术分析框架.md` - Complete technical analysis framework documentation
- Stock-specific directories: `002074.SZ/`, `300450.SZ/` containing downloaded data

**Factor System (`factor_system/`)**
- `multi_tf_vbt_detector.py` - Multi-timeframe analysis engine with VectorBT
- `enhanced_factor_calculator.py` - 154-indicator calculation engine
- `config.py` and `config.yaml` - Configuration management
- `robust_factor_screener.py` - Factor screening and validation

**Data Processing Pipeline (`data-resampling/`)**
- `resampling/` - Market data resampling utilities
- Production-ready resampling modules for different markets

**Data Storage**
- `raw/HK/` - Hong Kong stock data (276+ stocks, multiple timeframes)
- `raw/US/` - US stock data (172+ stocks)
- `cache/` - Computed factor data caching
- A-share data stored in individual stock directories

### Technical Indicator System

The framework implements **154 technical indicators** organized into categories:

**Core Technical Indicators (36 factors)**
- **Moving Averages**: MA5, MA10, MA20, MA30, MA60, EMA5, EMA12, EMA26
- **Momentum**: MACD, RSI, Stochastic, Williams %R, CCI, MFI
- **Volatility**: Bollinger Bands, ATR, Standard Deviation
- **Volume**: OBV, Volume SMA, Volume Ratio

**Enhanced Indicators (118 factors)**
- **Advanced MA**: DEMA, TEMA, T3, KAMA, Hull MA
- **Oscillators**: TRIX, ROC, CMO, ADX, DI+, DI-
- **Trend**: Parabolic SAR, Aroon, Chande Momentum
- **Statistical**: Z-Score, Correlation, Beta, Alpha
- **Cycle**: Hilbert Transform, Sine Wave, Trendline

### Performance Analysis Framework

**Technical Signal Generation**
- Multi-dimensional scoring system (trend, momentum, volume, risk)
- Support/resistance level identification using Fibonacci retracements
- MACD divergence detection and pattern recognition
- Volume-price trend analysis

**Risk Management**
- Value at Risk (VaR) and Conditional VaR calculations
- Maximum drawdown analysis and recovery metrics
- Volatility-based position sizing
- Dynamic stop-loss recommendations

**Investment Recommendations**
- Comprehensive scoring algorithm (0-5 scale)
- Multi-factor signal validation
- Risk-adjusted position sizing
- Target price projections using Fibonacci levels

## Data Structure

### File Naming Convention
- A-share data: `{SYMBOL}_{TIMEFRAME}_YYYY-MM-DD.csv`
- Example: `300450.SZ_1d_2025-09-28.csv` (daily data)
- Example: `300450.SZ_1h_2025-09-28.csv` (hourly data)

### Data Format
```csv
Date,Close,High,Low,Open,Volume
股票代码,股票代码,股票代码,股票代码,股票代码,股票代码
2025-04-01,20.87,21.14,20.86,21.00,9805664
```

### Supported Stocks
- **300450.SZ** (先导智能 - Lead Intelligent)
- **002074.SZ** (国轩高科 - Guoxuan High-Tech)
- Framework supports unlimited stock additions

## Configuration Management

### Project Configuration (`pyproject.toml`)
```toml
[project]
name = "quant-trading"
version = "0.1.0"
description = "量化交易开发环境"
requires-python = ">=3.11"
```

### Key Dependencies
- **vectorbt>=0.28.1** - High-performance backtesting engine
- **pandas>=2.3.2** - Data manipulation
- **numpy>=2.3.3** - Numerical computing
- **ta-lib>=0.6.7** - Technical indicators
- **yfinance>=0.2.66** - Financial data download
- **matplotlib>=3.10.6** - Data visualization
- **scikit-learn>=1.7.2** - Machine learning

## Development Guidelines

### Code Style
- Black formatting with 88-character line length
- isort for import sorting
- mypy for strict type checking
- pytest for testing with coverage

### Technical Analysis Best Practices
- **Data Quality**: Automatic validation and cleaning of OHLCV data
- **Bias Prevention**: No look-ahead bias in technical calculations
- **Risk Management**: Integrated risk metrics and position sizing
- **Signal Validation**: Multi-factor confirmation for trading signals

### Performance Optimization
- Vectorized operations using NumPy and pandas
- Memory-efficient data structures
- Cached computations for repeated analysis
- Parallel processing for batch operations

## Key Features

### A-Share Market Specialization
- Chinese stock code format support (SZ, SH exchanges)
- A-share trading hours and holidays consideration
- Market-specific technical patterns and indicators
- Southbound flow integration capabilities

### Advanced Analytics
- **Multi-timeframe Analysis**: Synchronized signals across different timeframes
- **Pattern Recognition**: Candlestick patterns and technical formations
- **Risk Metrics**: Comprehensive risk assessment and management
- **Portfolio Integration**: Multi-stock analysis and correlation

### Production Readiness
- Comprehensive error handling and logging
- Automated report generation in Markdown format
- Batch processing capabilities for multiple stocks
- Configurable analysis parameters and thresholds

## Specialized Modules

### Technical Analysis Engine (`300450_sz_technical_analysis.py`)
- 154 technical indicators calculation
- Advanced signal generation algorithms
- Performance metrics computation
- Investment recommendation system

### Data Downloader (`simple_download.py`)
- yfinance integration for A-share data
- Multiple timeframe support (daily, hourly)
- Automated data storage and organization
- Error handling and data validation

### Framework Documentation (`A股技术分析框架.md`)
- Complete technical analysis methodology
- Mathematical formulas for all indicators
- Risk management frameworks
- Practical implementation examples

## Usage Examples

### Basic Technical Analysis
```bash
# Analyze specific stock
python 300450_sz_technical_analysis.py 300450.SZ

# Custom analysis with parameters
python 300450_sz_technical_analysis.py 300450.SZ --data-dir /path/to/data --output-dir /path/to/output
```

### Data Download
```bash
# Download data for configured stocks
python simple_download.py

# Extensible for additional stocks
# Edit stocks list in simple_download.py
```

### Batch Processing
```python
# Example batch analysis (framework supports)
stocks = ["300450.SZ", "002074.SZ"]
for stock in stocks:
    run_technical_analysis(stock)
```

## Quality Assurance

### Testing Framework
- pytest configuration with coverage reporting
- Unit tests for technical indicators
- Integration tests for data processing
- Performance benchmarks for critical operations

### Code Quality
- Strict type checking with mypy
- Code formatting with black
- Import sorting with isort
- Comprehensive documentation

### Data Validation
- Automatic data integrity checks
- Missing value handling and imputation
- Outlier detection and treatment
- Date range validation

## Integration with Parent Project

This A-share framework integrates with a larger quantitative trading ecosystem:
- **Hong Kong Market Analysis**: 276+ HK stocks with sophisticated factor systems
- **US Market Coverage**: 172+ US stocks for multi-market analysis
- **Shared Infrastructure**: Common data processing and analysis tools
- **Unified Framework**: Consistent methodology across markets

## Advanced Features

### Machine Learning Integration
- Scikit-learn for predictive modeling
- Feature engineering from technical indicators
- Pattern recognition algorithms
- Risk prediction models

### Visualization Capabilities
- Interactive charts with matplotlib
- Technical indicator plotting
- Performance visualization
- Risk metric dashboards

### Reporting System
- Automated Markdown report generation
- Performance summary tables
- Risk assessment reports
- Investment recommendation documentation

## Error Handling

### Data Processing
- Robust error handling for data download failures
- Automatic retry mechanisms for network issues
- Data validation and integrity checks
- Graceful degradation for missing data

### Analysis Quality
- Validation of technical indicator calculations
- Cross-verification of signal generation
- Risk assessment for all recommendations
- Confidence scoring for investment suggestions

This framework provides a complete solution for A-share technical analysis, combining quantitative rigor with practical investment insights. It's designed for both research and production use, with comprehensive documentation and testing infrastructure.
