# 因子筛选模块依赖图

```
professional_factor_screener.py
├── config_manager.ScreeningConfig
├── enhanced_result_manager.EnhancedResultManager
├── utils.temporal_validator (TemporalValidator)
├── utils.time_series_protocols (SafeTimeSeriesProcessor)
├── utils (FactorFileAligner, find_aligned_factor_files, validate_factor_alignment)
├── pandas / numpy / psutil / yaml / scipy
└── logging / pathlib / typing

enhanced_result_manager.py
├── professional_factor_screener.FactorMetrics (序列化依赖)
├── matplotlib.pyplot
├── pandas / numpy
└── json / pathlib / logging

config_manager.py
├── yaml / json
├── dataclasses
└── pathlib / logging

scripts/check_future_functions.py
├── ast / pathlib / re
└── logging / typing

utils/time_series_protocols.py
├── pandas / numpy
└── logging / typing
```

> 备注：现阶段 `enhanced_result_manager` 仍需通过局部导入使用 `FactorMetrics`；后续架构拆分时需打破该循环依赖。
