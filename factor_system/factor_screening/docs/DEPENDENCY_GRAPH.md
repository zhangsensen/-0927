# 因子筛选模块依赖图

```
professional_factor_screener.py
├── config_manager.ScreeningConfig
├── enhanced_result_manager.EnhancedResultManager
├── utils.temporal_validator (TemporalValidator)
├── utils.time_series_protocols (SafeTimeSeriesProcessor)
├── utils (FactorFileAligner 等)
├── pandas / numpy / scipy / psutil / yaml
└── 标准库：logging、pathlib、typing

enhanced_result_manager.py
├── professional_factor_screener.FactorMetrics
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

> 后续重构目标：按照 core/data/computation/screening/reporting/config 分层拆分，逐步打散当前单体依赖。
