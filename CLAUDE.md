# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **professional-grade quantitative trading development environment** specializing in multi-market analysis with comprehensive factor screening capabilities. The system features:

1. **A-Share Technical Analysis Framework** (`a股/`) - Chinese stock market analysis with 154 technical indicators
2. **Professional Factor Screening System** (`factor_system/`) - 5-dimension factor analysis with statistical significance testing
3. **Multi-Market Support** - Hong Kong (276+ stocks), US (172+ stocks), and A-Share markets
4. **VectorBT Integration** - 10-50x performance improvement over traditional pandas

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
python a股/stock_analysis/sz_technical_analysis.py <STOCK_CODE>

# Download A-share data using yfinance
python a股/data_download/simple_download.py

# Screen top A-share stocks
python a股/screen_top_stocks.py
```

### Factor System Analysis
```bash
# Quick start multi-timeframe analysis
python factor_system/factor_generation/quick_start.py <STOCK_CODE>

# Professional factor screening (CLI)
python factor_system/factor_screening/cli.py screen <STOCK_CODE> <TIMEFRAME>

# Enhanced factor calculator with 154 indicators
python factor_system/factor_generation/enhanced_factor_calculator.py

# Professional factor screener demo
python factor_system/factor_screening/quick_start.py

# Batch processing multiple stocks
python factor_system/factor_screening/batch_screener.py
```

### Data Processing
```bash
# Batch resample Hong Kong 1-minute data to higher timeframes
python batch_resample_hk.py
```

## Architecture Overview

### Core Components

**A-Share Framework (`a股/`)**
- `stock_analysis/sz_technical_analysis.py` - Main technical analysis engine with 154 indicators
- `data_download/simple_download.py` - A-share data downloader using yfinance
- `screen_top_stocks.py` - Stock screening utility
- `batch_storage_analysis.py` - Storage analysis tool
- Individual stock directories with OHLCV data

**Factor System (`factor_system/`)**
- `factor_generation/` - Multi-timeframe factor calculation and analysis
  - `enhanced_factor_calculator.py` - 154 technical indicators engine
  - `multi_tf_vbt_detector.py` - VectorBT-based analysis
  - `quick_start.py` - Quick start entry point
  - `config.py` - Configuration management
- `factor_screening/` - Professional factor screening system
  - `professional_factor_screener.py` - 5-dimension screening engine
  - `cli.py` - Command-line interface
  - `batch_screener.py` - Batch processing
  - `config_manager.py` - Configuration management
- `config/` - Strategy configuration templates
- `tests/` - Comprehensive test suite

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

### 5-Dimension Screening Framework

The professional factor screener implements a comprehensive 5-dimension evaluation system:

**1. Predictive Power (35% weight)**
- Multi-horizon IC analysis (1, 3, 5, 10, 20 days)
- IC decay analysis and persistence metrics
- Information Coefficient ratio (risk-adjusted)

**2. Stability (25% weight)**
- Rolling window IC analysis
- Cross-sectional stability assessment
- IC consistency measurement

**3. Independence (20% weight)**
- Variance Inflation Factor (VIF) detection
- Factor correlation analysis
- Information increment calculation

**4. Practicality (15% weight)**
- Trading cost assessment (commission, slippage, impact)
- Turnover rate analysis
- Liquidity requirements evaluation

**5. Short-term Adaptability (5% weight)**
- Reversal effect detection
- Momentum persistence analysis
- Volatility sensitivity assessment

### Performance Optimization

**VectorBT Integration**
- 10-50x performance improvement over traditional pandas
- Vectorized computations eliminating Python loops
- Memory optimization: 40-60% reduction in usage

**Multi-Timeframe Analysis**
- Supported timeframes: 1min, 2min, 3min, 5min, 15min, 30min, 60min, daily
- Automatic timeframe alignment and signal synchronization
- Smart data resampling preserving market microstructure

### Statistical Rigor

**Advanced Statistical Methods**
- Benjamini-Hochberg FDR correction for multiple comparisons
- Strict significance testing (α = 0.01, 0.05, 0.10)
- Variance Inflation Factor (VIF) for multicollinearity detection
- Rolling window validation for stability assessment

**Bias Prevention**
- No look-ahead bias in factor calculations
- Proper survivorship bias handling
- Real market data only (no simulated data)
- Proper data alignment across timeframes

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

**Strategy Configuration Templates**
- `long_term_config.yaml` - Long-term investment strategies
- `conservative_config.yaml` - Conservative trading strategies
- `high_frequency_config.yaml` - High-frequency trading strategies
- `aggressive_config.yaml` - Aggressive trading strategies

### Development Standards

**Code Quality Requirements**
- Black formatting with 88-character line length
- isort for import sorting
- mypy for strict type checking (disallow_untyped_defs)
- pytest for testing with 95%+ coverage requirement

**Performance Standards**
- Memory efficiency > 70%
- Critical path operations < 1ms
- VectorBT for all vectorized operations
- Avoid DataFrame.apply in favor of built-ins

**Cursor Rules Integration**
The project includes Cursor rules (`.cursor/rules/my-linus.mdc`) that define:
- **Identity**: Quantitative Chief Engineer with Linus Torvalds engineering philosophy
- **Core Principles**: Eliminate special cases, API compatibility, practicality, simplicity
- **Tech Stack**: Python data science ecosystem, VectorBT, 154 indicators
- **Quality Standards**: Code taste ratings (🟢/🟡/🔴), 3-layer indent limit, >80% test coverage

## Development Guidelines

### Testing Commands
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=factor_system

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m slow          # Slow tests only

# Performance benchmarking
python factor_system/factor_screening/performance_benchmark.py

# Basic functionality test
python factor_system/factor_screening/tests/test_basic_functionality.py
```

### Performance Considerations
- Use VectorBT for all vectorized operations
- Avoid loops in favor of NumPy/VectorBT built-ins
- Implement memory-efficient data structures
- Cache computed factors to avoid redundant calculations
- Monitor memory usage with memory-profiler for large datasets

### Quantitative Rigor
- Bias prevention: No look-ahead bias or survivorship bias
- Real market data only (no simulated data)
- Proper data alignment across timeframes
- Statistical validation of all indicators
- Use multiple random seeds for robustness testing

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

### CLI Usage Examples

**Professional Factor Screening**
```bash
# Screen factors for a specific stock
python factor_system/factor_screening/cli.py screen 0700.HK 60min

# Batch process multiple stocks
python factor_system/factor_screening/cli.py batch --symbols 0700.HK,0005.HK,0941.HK --timeframe 60min

# Generate screening report
python factor_system/factor_screening/cli.py report 0700.HK --output screening_report.pdf
```

**Configuration Management**
```bash
# List available configurations
python factor_system/factor_screening/cli.py config list

# Create custom configuration
python factor_system/factor_screening/cli.py config create custom_config.yaml

# Validate configuration
python factor_system/factor_screening/cli.py config validate screening_config.yaml
```

### Performance Benchmarks

**Factor Calculation Performance**
- Small scale (500 samples × 20 factors): 831+ factors/second
- Medium scale (1000 samples × 50 factors): 864+ factors/second
- Large scale (2000 samples × 100 factors): 686+ factors/second
- Extra large scale (5000 samples × 200 factors): 370+ factors/second

**Complete Screening Process**
- Processing speed: 5.7 factors/second (80 factors full analysis)
- Memory usage: < 1MB (medium scale data)
- Main bottleneck: Rolling IC calculation (94.2% of time)

### Result Interpretation

**Factor Quality Tiers**
- **Comprehensive Score ≥ 0.8**: 🥇 Tier 1 - Core factors, highly recommended
- **Comprehensive Score 0.6-0.8**: 🥈 Tier 2 - Important factors, recommended
- **Comprehensive Score 0.4-0.6**: 🥉 Tier 3 - Backup factors, use with caution
- **Comprehensive Score < 0.4**: ❌ Not recommended

**Statistical Significance**
- ***** p < 0.001: Highly significant
- **** p < 0.01: Significant
- *** p < 0.05: Marginally significant
- No marker: Not significant

This quantitative trading platform is designed for serious algorithmic trading research with professional-grade factor analysis capabilities optimized for multiple markets.
