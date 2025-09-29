# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **comprehensive quantitative trading development environment** specialized for multi-market analysis with two main components:

1. **A-Share Technical Analysis Framework** (`a股/`) - Chinese stock market analysis system with 154 technical indicators
2. **Hong Kong/US Market Factor System** (`factor_system/`) - Multi-timeframe factor analysis using VectorBT

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

### A-Share Analysis
```bash
# Run technical analysis on A-share stocks
python a股/300450_sz_technical_analysis.py 300450.SZ

# Download A-share data using yfinance
python a股/simple_download.py
```

### Factor System Analysis
```bash
# Run multi-timeframe factor analysis with 154 indicators
python factor_system/multi_tf_vbt_detector.py 0700.HK

# Demo factor system
python demo_factor_system.py

# Run VectorBT professional detector
python factor_system/vbt_professional_detector.py
```

### Data Processing
```bash
# Batch resample Hong Kong 1-minute data to higher timeframes
python batch_resample_hk.py
```

## Architecture Overview

### Core Components

**A-Share Framework (`a股/`)**
- `300450_sz_technical_analysis.py` - Main technical analysis engine with comprehensive indicators
- `simple_download.py` - A-share data downloader using yfinance
- `A股技术分析框架.md` - Detailed framework documentation
- Individual stock directories with OHLCV data

**Factor System (`factor_system/`)**
- `multi_tf_vbt_detector.py` - Multi-timeframe analysis engine with 154 indicators
- `enhanced_factor_calculator.py` - Advanced indicator calculation engine
- `vbt_professional_detector.py` - Professional VectorBT-based signal detection
- `config.py` - Configuration management system
- `data/` - Data loading utilities

**Data Processing (`data-resampling/`)**
- `resampling/hk_resampler.py` - Hong Kong market data resampling
- `production_resampler_simple.py` - Production-ready resampler

### 154-Indicator System

The system implements **154 technical indicators** organized into categories:

**Core Technical Indicators (36 factors)**
- Moving Averages: MA5, MA10, MA20, MA30, MA60, EMA5, EMA12, EMA26
- Momentum: MACD, RSI, Stochastic, Williams %R, CCI, MFI
- Volatility: Bollinger Bands, ATR, Standard Deviation
- Volume: OBV, Volume SMA, Volume Ratio

**Enhanced Indicators (118 factors)**
- Advanced MA: DEMA, TEMA, T3, KAMA, Hull MA
- Oscillators: TRIX, ROC, CMO, ADX, DI+, DI-
- Trend: Parabolic SAR, Aroon, Chande Momentum
- Statistical: Z-Score, Correlation, Beta, Alpha
- Cycle: Hilbert Transform, Sine Wave, Trendline

### Performance Optimization

**VectorBT Integration**
- 10-50x performance improvement over traditional pandas
- Vectorized computations eliminating Python loops
- Memory optimization: 40-60% reduction in usage

**Multi-Timeframe Analysis**
- Supported timeframes: 1min, 2min, 3min, 5min, 15min, 30min, 60min, daily
- Automatic timeframe alignment and signal synchronization
- Smart data resampling preserving market microstructure

## Data Structure

### File Naming Convention
- A-Share: `{SYMBOL_CODE}_1d_YYYY-MM-DD.csv`
- Hong Kong: `{SYMBOL}_1min_YYYY-MM-DD_YYYY-MM-DD.parquet`
- Output: `{SYMBOL}_{TIMEFRAME}_factors_YYYY-MM-DD_HH-MM-SS.parquet`

### Market-Specific Data
- **A-Share**: Individual stock directories with daily data
- **Hong Kong**: 276+ stocks with minute-level data
- **US Market**: 172+ stocks with various timeframes

## Configuration Management

### Project Configuration
- **pyproject.toml**: Modern Python project configuration with uv
- **Python 3.11+**: Required for all components
- **Dependencies**: VectorBT, pandas, NumPy, TA-Lib, scikit-learn, etc.

### Factor System Config
Configuration is managed through Python classes with YAML support:
- Data paths and directory structures
- Indicator enable/disable flags
- Performance optimization settings
- Memory efficiency modes

## Development Guidelines

### Code Style
- Black formatting with 88-character line length
- isort for import sorting
- mypy for strict type checking
- pytest for testing with coverage

### Performance Considerations
- Use VectorBT for all vectorized operations
- Avoid loops in favor of NumPy/VectorBT built-ins
- Implement memory-efficient data structures
- Cache computed factors to avoid redundant calculations

### Quantitative Rigor
- Bias prevention: No look-ahead bias or survivorship bias
- Real market data only (no simulated data)
- Proper data alignment across timeframes
- Statistical validation of all indicators

## Key Dependencies

### Core Quantitative Stack
- **vectorbt>=0.28.1** - High-performance backtesting engine
- **pandas>=2.3.2** - Data manipulation
- **numpy>=2.3.3** - Numerical computing
- **polars>=1.0.0** - High-performance dataframes
- **numba>=0.60.0** - JIT compilation

### Technical Analysis
- **ta-lib>=0.6.7** - Technical indicators
- **scikit-learn>=1.7.2** - Machine learning
- **scipy>=1.16.2** - Scientific computing

### Data & Storage
- **pyarrow>=21.0.0** - Columnar storage
- **fastparquet** - Parquet file handling
- **yfinance>=0.2.66** - Market data
- **sqlalchemy>=2.0.0** - Database ORM

### Visualization
- **matplotlib>=3.10.6** - Static plotting
- **seaborn>=0.13.2** - Statistical visualization
- **plotly>=5.24.0** - Interactive charts
- **dash>=2.18.0** - Web dashboards

## Special Features

### Multi-Market Support
- **A-Share**: Chinese stock market with specialized patterns
- **Hong Kong**: 276+ stocks with minute-level precision
- **US Market**: 172+ stocks with various timeframes

### Production Readiness
- Comprehensive error handling and logging
- Performance monitoring and alerting
- Scalable architecture for large datasets
- Configurable deployment settings

### Advanced Analytics
- Multi-timeframe signal generation
- Cross-indicator correlation analysis
- Market regime detection
- Factor performance attribution

## Risk Management

### Built-in Risk Metrics
- VaR (Value at Risk) calculations
- Maximum drawdown analysis
- Sharpe and Sortino ratios
- Volatility-adjusted position sizing

### Position Sizing
- Risk-based position calculation
- Volatility-adjusted sizing
- Stop-loss recommendations
- Portfolio risk metrics

## Error Handling

### Data Validation
- Automatic data integrity checks
- Missing data handling and imputation
- Outlier detection and treatment
- Data alignment across timeframes

### Performance Monitoring
- Memory usage tracking
- Calculation time benchmarking
- Data quality metrics
- System health monitoring

This quantitative trading platform is designed for serious algorithmic trading research with professional-grade factor analysis capabilities optimized for multiple markets.