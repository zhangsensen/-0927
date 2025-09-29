# Factor Generation Code

This directory contains the core factor generation components:

- `quick_start.py` - Simple launcher for factor generation
- `multi_tf_vbt_detector.py` - Main multi-timeframe factor detection engine
- `enhanced_factor_calculator.py` - 154 technical indicators calculation core
- `config_loader.py` - Configuration management for factor generation
- `config.py` - Simple configuration loader
- `config.yaml` - Multi-timeframe detector configuration

## Usage

```bash
# Generate factors for a specific stock
python quick_start.py <STOCK_CODE>

# Example:
python quick_start.py 0700.HK
```

## Files Structure

```
factor_generation/
├── quick_start.py              # Entry point launcher
├── multi_tf_vbt_detector.py     # Main detection engine
├── enhanced_factor_calculator.py # Core 154 indicators
├── config_loader.py             # Configuration management
├── config.py                    # Simple configuration loader
├── config.yaml                  # Multi-timeframe detector config
└── README.md                   # This file
```

## Dependencies

All dependencies are managed through uv (see ../uv.lock)