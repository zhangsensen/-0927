# 系统输入输出契约

## 配置输入
- `ScreeningConfig.data_root`: 因子文件根目录，允许相对路径，初始化时转换为 `Path`
- `ScreeningConfig.raw_data_root`: 价格数据根目录，若不存在需提前创建
- `ScreeningConfig.output_root`: 会话输出根目录，无法创建时必须抛出 `OSError`
- 向后兼容：当仅提供 `output_dir` 时仍需支持，但以 `output_root` 优先

## 运行参数
- `symbol`: 必须为 `<代码>.<市场>` 格式（例 `0700.HK`），非法格式直接拒绝
- `timeframe`: 必须包含时间单位（如 `5min`、`60min`、`daily`），不可为空
- `ic_horizons`: 正整数列表
- `min_sample_size`: 正整数

## 数据契约
- 因子数据 (`load_factors`)
  - 返回 `pd.DataFrame`，索引为 `pd.DatetimeIndex`
  - 数值列类型统一为浮点型（`float32/float64`）
  - 经过数据质量验证后至少保留 10 列有效因子
- 价格数据 (`load_price_data`)
  - 包含 `open/high/low/close/volume` 列
  - 索引为 `pd.DatetimeIndex`
  - 未对齐时需进行时间标准化处理（日线支持 `.normalize()`）

## 输出契约
- `screen_factors_comprehensive` 返回 `Dict[str, FactorMetrics]`
  - `FactorMetrics` 字段完整，数值类型为 `float` / `bool` / `int`
- 会话目录结构
  - `<output_root>/<symbol>_<timeframe>_<timestamp>/`
    - `screening_report.csv`
    - `screening_statistics.json`
    - `executive_summary.txt`
    - `detailed_analysis.md`
    - `charts/`
- 批量运行共享同一主会话目录，默认生成 `multi_timeframe_summary.json`

## 向后兼容行为
- 老配置字段（如 `output_dir`）必须继续支持，但内部统一映射到新字段
- 当日志目录不可写时需回退到控制台输出并给出警告
- 任何文件缺失都必须写入清晰的错误日志，不允许静默失败
- 异常统一通过 `logger.error(..., exc_info=True)` 记录并抛出

