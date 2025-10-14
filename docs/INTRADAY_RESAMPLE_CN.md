# A股分钟数据重采样（避开午休与收盘）

问题：直接使用 `resample('60min', origin="start")` 会产生 11:30、12:30、15:30 等不在交易时段内的时间点。

解决：使用会话感知（Session-aware）的重采样工具，仅在上午(09:30-11:30)与下午(13:00-15:00)两个会话内按 `5/15/30/60min` 分箱聚合。

## 使用方法

```python
from factor_system.utils.session_resample import resample_ashare_intraday

# df: 1分钟原始数据，DatetimeIndex（可多级索引(symbol, datetime)）
# 需要包含列：open, high, low, close, volume[, amount]

df_60 = resample_ashare_intraday(df, '60min')
df_30 = resample_ashare_intraday(df, '30min')
df_15 = resample_ashare_intraday(df, '15min')
df_05 = resample_ashare_intraday(df, '5min')
```

注意：函数会严格截断到 11:30 与 15:00 的右边界（不产生 11:30-12:30、12:30-13:30、14:30-15:30 等跨时段与越界分箱）。

## 自定义节假日

可在 `config/cn_holidays.txt` 填写节假日，一行一个日期：

```
2025-01-01
2025-02-10
```

或 `YYYYMMDD`：

```
20250101
```

未提供时，默认仅剔除周末。

