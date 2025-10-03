# 系统输入输出契约

## 配置输入
- `ScreeningConfig.data_root`: `Path | str`，指向因子文件目录；允许相对路径，初始化时转换为 `Path`
- `ScreeningConfig.raw_data_root`: 价格原始数据目录，必须存在或在运行前创建
- `ScreeningConfig.output_root`: 输出结果根目录；创建失败需抛出 `OSError`
- 兼容字段：`output_dir`（旧版路径），当同时存在时以 `output_root` 优先

## 运行参数
- `symbol`: 必须为 `<代码>.<市场>` 格式（如 `0700.HK`）；非法格式视为错误
- `timeframe`: 必须包含时间单位（`5min`/`60min`/`daily` 等），不可为空
- `ic_horizons`: 列表，全部为正整数
- `min_sample_size`: 正整数

## 数据契约
- 因子数据 (`load_factors`)
  - `DataFrame` 索引为 `DatetimeIndex`
  - 列均为可数值转换类型，允许 `float32/float64`
  - 至少保留 10 列有效因子
- 价格数据 (`load_price_data`)
  - 必须包含 `open/high/low/close/volume` 列
  - 索引为 `DatetimeIndex`
- 统一时区：输入数据需事先对齐，若检测到 `tzinfo` 不一致直接中止

## 输出契约
- `screen_factors_comprehensive` 返回 `Dict[str, FactorMetrics]`
  - `FactorMetrics` 字段不可缺失
  - `is_significant`、`comprehensive_score` 等均为 `float/bool`
- 会话目录结构
  - `<output_root>/<symbol>_<timeframe>_<timestamp>/`
    - `screening_report.csv`
    - `screening_statistics.json`
    - `executive_summary.txt`
    - `detailed_analysis.md`
    - `charts/`
- 多时间框架批量任务共用同一会话根目录

## 向后兼容行为
- 若旧配置仅提供 `output_dir`，系统需自动兼容
- 当日志目录不可写时，必须回退到控制台输出并给出警告
- 缺失文件需提供清晰错误日志，不允许静默失败
- 所有异常均通过 `logger.error(..., exc_info=True)` 输出

