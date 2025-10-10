# 数据结构和格式规范

## 文件命名约定

### 市场数据格式
```
A股数据格式:
{SYMBOL_CODE}_1d_YYYY-MM-DD.csv
示例: 000001_1d_2025-10-07.csv

港股原始数据:
{SYMBOL}_1min_YYYY-MM-DD_YYYY-MM-DD.parquet
示例: 0700.HK_1min_2025-09-01_2025-09-30.parquet

港股重采样数据:
{SYMBOL}_{TIMEFRAME}_YYYY-MM-DD.parquet
示例: 0700.HK_15min_2025-09-01.parquet

因子输出数据:
{SYMBOL}_{TIMEFRAME}_factors_YYYY-MM-DD_HH-MM-SS.parquet
示例: 0700.HK_15min_factors_2025-10-07_14-30-00.parquet
```

### 筛选结果格式
```
筛选会话目录:
{SYMBOL}_multi_tf_YYYYMMDD_HHMMSS/
示例: 0700.HK_multi_tf_20251007_020832/

时间框架子目录:
timeframes/{SYMBOL}_{TIMEFRAME}_{TIMESTAMP}/
示例: timeframes/0700.HK_15min_20251007_020832/

分析报告:
detailed_analysis.md
README.md
```

## 数据结构定义

### OHLCV数据结构
```python
columns = {
    'open': 'float64',      # 开盘价
    'high': 'float64',      # 最高价
    'low': 'float64',       # 最低价
    'close': 'float64',     # 收盘价
    'volume': 'int64'       # 成交量
}
index = 'datetime64[ns]'   # 时间索引
```

### 因子数据结构
```python
# 技术指标因子 (37个)
technical_factors = {
    # 动量指标
    'RSI_14': 'float64',           # 相对强弱指数
    'RSI_6': 'float64',            # 短期RSI
    'RSI_24': 'float64',           # 长期RSI
    'MACD_12_26_9': 'float64',     # MACD
    'MACD_signal_12_26_9': 'float64', # MACD信号线
    'MACD_hist_12_26_9': 'float64',  # MACD柱状图
    'STOCH_14_3_3_k': 'float64',   # 随机指标K值
    'STOCH_14_3_3_d': 'float64',   # 随机指标D值
    'WILLR_14': 'float64',         # 威廉指标
    'CCI_14': 'float64',           # 商品通道指数
    'MFI_14': 'float64',           # 资金流量指数
    'ADX_14': 'float64',           # 平均趋向指数
    'ATR_14': 'float64',           # 真实波动率

    # 动量变化指标
    'MOM_10': 'float64',           # 动量
    'ROC_10': 'float64',           # 变化率
    'ROCP_10': 'float64',          # 变化百分比
    'TRIX_14': 'float64',          # 三重平滑移动平均
    'DEMA_14': 'float64',          # 双指数移动平均
    'TEMA_14': 'float64',          # 三重指数移动平均

    # 振荡器
    'AROON_up_14': 'float64',      # 阿隆指标上行
    'AROON_down_14': 'float64',    # 阿隆指标下行
    'AROON_osc_14': 'float64',     # 阿隆振荡器
    'CMO_14': 'float64',           # 钱德动量摆动指标
    'DX_14': 'float64',            # 趋向指数
    'MINUS_DI_14': 'float64',      # 负向指标
    'PLUS_DI_14': 'float64',       # 正向指标
}

# 移动平均类因子 (12个)
overlap_factors = {
    'SMA_5': 'float64',            # 简单移动平均
    'SMA_10': 'float64',
    'SMA_20': 'float64',
    'SMA_30': 'float64',
    'SMA_60': 'float64',
    'EMA_5': 'float64',            # 指数移动平均
    'EMA_12': 'float64',
    'EMA_26': 'float64',
    'WMA_14': 'float64',           # 加权移动平均
    'DEMA_14': 'float64',          # 双指数移动平均
    'TEMA_14': 'float64',          # 三重指数移动平均
    'KAMA_14': 'float64',          # 考夫曼自适应移动平均
}

# 波动率因子
volatility_factors = {
    'BBANDS_upper_20_2': 'float64',  # 布林带上轨
    'BBANDS_middle_20_2': 'float64', # 布林带中轨
    'BBANDS_lower_20_2': 'float64',  # 布林带下轨
    'STDDEV_20': 'float64',          # 标准差
    'ATR_14': 'float64',             # 平均真实波动率
}

# 成交量因子
volume_factors = {
    'OBV': 'float64',               # 能量潮
    'VOLUME_SMA_20': 'float64',     # 成交量移动平均
    'VOLUME_RATIO': 'float64',      # 成交量比率
    'MFI_14': 'float64',            # 资金流量指数
    'AD': 'float64',                # 累积派发线
    'ADOSC_3_10': 'float64',        # 振荡器
}

# 价格模式因子 (60+个)
pattern_factors = {
    # 蜡烛图模式
    'CDL2CROWS': 'float64',         # 两只乌鸦
    'CDL3BLACKCROWS': 'float64',    # 三只乌鸦
    'CDL3INSIDE': 'float64',        # 三内含
    'CDL3LINESTRIKE': 'float64',    # 三线打击
    'CDL3OUTSIDE': 'float64',       # 三外部
    'CDL3STARSINSOUTH': 'float64',  # 南方三星
    'CDL3WHITESOLDIERS': 'float64', # 三个白兵
    'CDLABANDONEDBABY': 'float64',  # 弃婴
    'CDLADVANCEBLOCK': 'float64',   # 大敌当前
    'CDLBELTHOLD': 'float64',       # 捉腰带线
    'CDLBREAKAWAY': 'float64',      # 突破
    'CDLCLOSINGMARUBOZU': 'float64', # 收盘光头/光脚
    'CDLCONCEALBABYSWALL': 'float64', # 藏婴吞没
    'CDLCOUNTERATTACK': 'float64',  # 反击线
    'CDLDARKCLOUDCOVER': 'float64', # 乌云压顶
    'CDLDOJI': 'float64',           # 十字星
    'CDLDOJISTAR': 'float64',       # 十字星
    'CDLDRAGONFLYDOJI': 'float64',  # 蜻蜓十字
    'CDLENGULFING': 'float64',      # 吞没模式
    'CDLEVENINGDOJISTAR': 'float64', # 黄昏之星十字
    'CDLEVENINGSTAR': 'float64',    # 黄昏之星
    'CDLGAPSIDESIDEWHITE': 'float64', # 向上跳空并列阳线
    'CDLGRAVESTONEDOJI': 'float64', # 墓碑十字
    'CDLHAMMER': 'float64',         # 锤头
    'CDLHANGINGMAN': 'float64',     # 上吊线
    'CDLHARAMI': 'float64',         # 孕线
    'CDLHARAMICROSS': 'float64',    # 十字孕线
    'CDLHIGHWAVE': 'float64',       # 高浪
    'CDLHIKKAKE': 'float64',        # 拉锯
    'CDLHOMINGPIGEON': 'float64',   # 归雁
    'CDLIDENTICAL3CROWS': 'float64', # 三胞胎乌鸦
    'CDLINNECK': 'float64',         # 颈内线
    'CDLINVERTEDHAMMER': 'float64', # 倒锤头
    'CDLKICKING': 'float64',        # 反击
    'CDLKICKINGBYLENGTH': 'float64', # 反击由长度决定
    'CDLLADDERBOTTOM': 'float64',   # 梯底
    'CDLLONGLEGGEDDOJI': 'float64', # 长腿十字
    'CDLLONGLINE': 'float64',       # 长蜡烛
    'CDLMARUBOZU': 'float64',       # 光头/光脚
    'CDLMATCHINGLOW': 'float64',    # 镊子底
    'CDLMATHOLD': 'float64',        # 捉腰带线
    'CDLMORNINGDOJISTAR': 'float64', # 早星十字
    'CDLMORNINGSTAR': 'float64',    # 早星
    'CDLONNECK': 'float64',         # 颈上线
    'CDLPIERCING': 'float64',       # 刺透
    'CDLRICKSHAWMAN': 'float64',    # 黄包车夫
    'CDLRISEFALL3METHODS': 'float64', # 上升/下降三法
    'CDLSEPARATINGLINES': 'float64', # 分离线
    'CDLSHOOTINGSTAR': 'float64',   # 射击之星
    'CDLSHORTLINE': 'float64',      # 短蜡烛
    'CDLSPINNINGTOP': 'float64',    # 陀螺线
    'CDLSTALLEDPATTERN': 'float64', # 停顿模式
    'CDLSTICKSANDWICH': 'float64',  # 三明治
    'CDLTAKURI': 'float64',         # 探底
    'CDLTASUKIGAP': 'float64',      # 跳空
    'CDLTHRUSTING': 'float64',      # 插入
    'CDLTRISTAR': 'float64',        # 三星
    'CDLUNIQUE3RIVER': 'float64',   # 三川
    'CDLUPSIDEGAP2CROWS': 'float64', # 向上跳空两只乌鸦
    'CDLXSIDEGAP3METHODS': 'float64' # 上升/下降跳空三法
}

# 统计因子 (15个)
statistical_factors = {
    'BETA': 'float64',              # 贝塔系数
    'CORREL': 'float64',            # 相关系数
    'LINEARREG': 'float64',         # 线性回归
    'LINEARREG_ANGLE': 'float64',   # 线性回归角度
    'LINEARREG_INTERCEPT': 'float64', # 线性回归截距
    'LINEARREG_SLOPE': 'float64',   # 线性回归斜率
    'STDDEV': 'float64',            # 标准差
    'TSF': 'float64',               # 时间序列预测
    'VAR': 'float64',               # 方差
}
```

### 筛选评估结果结构
```python
screening_results = {
    'factor_metadata': {
        'factor_id': str,           # 因子标识符
        'factor_category': str,     # 因子类别
        'parameters': dict,         # 参数配置
        'calculation_time': float,  # 计算时间
        'data_quality': float       # 数据质量评分
    },
    'predictive_power': {
        'ic_1d': float,             # 1日信息系数
        'ic_3d': float,             # 3日信息系数
        'ic_5d': float,             # 5日信息系数
        'ic_10d': float,            # 10日信息系数
        'ic_20d': float,            # 20日信息系数
        'ic_mean': float,           # 平均IC
        'ic_std': float,            # IC标准差
        'ic_ir': float,             # 信息比率
        'ic_decay_rate': float,     # IC衰减率
        'significance_1d': float,   # 1日显著性p值
        'significance_5d': float,   # 5日显著性p值
        'significance_20d': float   # 20日显著性p值
    },
    'stability': {
        'rolling_ic_mean': float,   # 滚动IC均值
        'rolling_ic_std': float,    # 滚动IC标准差
        'stability_score': float,   # 稳定性评分
        'consistency_ratio': float  # 一致性比率
    },
    'independence': {
        'vif_score': float,         # 方差膨胀因子
        'correlation_matrix': np.ndarray, # 相关性矩阵
        'independence_score': float, # 独立性评分
        'information_increment': float # 信息增量
    },
    'practicality': {
        'turnover_rate': float,     # 换手率
        'trading_cost': float,      # 交易成本
        'liquidity_requirement': float, # 流动性要求
        'practicality_score': float # 实用性评分
    },
    'adaptability': {
        'reversal_effect': float,   # 反转效应
        'momentum_persistence': float, # 动量持续性
        'volatility_sensitivity': float, # 波动率敏感性
        'adaptability_score': float # 适应性评分
    },
    'comprehensive_score': float,   # 综合评分
    'quality_tier': str,           # 质量等级 (T1/T2/T3/Not Recommended)
    'recommendations': list,       # 建议
    'warnings': list              # 警告
}
```

## 配置文件结构

### FactorEngine配置
```yaml
# factor_engine/configs/engine_config.yaml
engine:
  cache:
    memory_limit_mb: 500
    disk_cache_dir: "cache/factor_engine"
    ttl_hours: 24

  data_provider:
    type: "parquet"
    data_root: "/Users/zhangshenshen/深度量化0927/raw"

  performance:
    n_jobs: -1
    chunk_size: 1000
    enable_parallel: true

  logging:
    level: "INFO"
    format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### 因子生成配置
```yaml
# factor_generation/config.yaml
data:
  root_path: "/Users/zhangshenshen/深度量化0927/raw/HK"
  file_pattern: "{symbol}_1min_{start_date}_{end_date}.parquet"

analysis:
  timeframes: ["5min", "15min", "30min", "60min", "daily"]
  memory_limit_mb: 8192
  enable_parallel: true

factors:
  technical:
    enabled: true
    indicators: ["RSI", "MACD", "STOCH", "WILLR", "CCI"]

  overlap:
    enabled: true
    indicators: ["SMA", "EMA", "BBANDS"]

  volume:
    enabled: true
    indicators: ["OBV", "VOLUME_SMA", "VOLUME_RATIO"]

  pattern:
    enabled: true
    indicators: ["CDLENGULFING", "CDLHAMMER", "CDLDOJI"]
```

### 筛选配置
```yaml
# factor_screening/configs/screening_config.yaml
screening:
  dimensions:
    predictive_power:
      weight: 0.35
      horizons: [1, 3, 5, 10, 20]
      significance_levels: [0.01, 0.05, 0.10]

    stability:
      weight: 0.25
      window_size: 252
      min_observations: 60

    independence:
      weight: 0.20
      vif_threshold: 5.0
      correlation_threshold: 0.7

    practicality:
      weight: 0.15
      max_turnover: 0.20
      cost_threshold: 0.002

    adaptability:
      weight: 0.05
      short_window: 5
      long_window: 20

statistics:
  fdr_method: "benjamini_hochberg"
  min_observations: 50
  bootstrap_samples: 1000

output:
  save_results: true
  generate_plots: true
  export_format: ["parquet", "json"]
```

## 缓存结构

### 内存缓存格式
```python
memory_cache = {
    'factor_key': {
        'data': pd.DataFrame,        # 因子数据
        'metadata': dict,            # 元数据
        'timestamp': datetime,       # 计算时间
        'ttl': int,                  # 生存时间
        'access_count': int,         # 访问次数
        'last_access': datetime      # 最后访问时间
    }
}
```

### 磁盘缓存格式
```
cache/factor_engine/
├── factors/
│   ├── {symbol}/
│   │   ├── {timeframe}/
│   │   │   ├── {factor_id}_{start_date}_{end_date}.parquet
│   │   │   └── metadata.json
├── registry/
│   └── factor_registry.json
└── cache_stats.json
```

## 市场特定数据结构

### A股数据结构
```
A股/
├── 000001/
│   ├── 000001_1d_2025-10-07.csv
│   ├── technical_analysis_20251007.json
│   └── screening_results/
├── 000002/
│   └── ...
└── screen_results/
    └── top_stocks_20251007.json
```

### 港股数据结构
```
raw/HK/
├── 0700.HK/
│   ├── 0700.HK_1min_2025-09-01_2025-09-30.parquet
│   ├── 0700.HK_5min_2025-09-01_2025-09-30.parquet
│   └── 0700.HK_15min_2025-09-01_2025-09-30.parquet
├── 0005.HK/
│   └── ...
└── symbols_list.json
```

### 筛选结果结构
```
factor_system/factor_screening/screening_results/
├── {SYMBOL}_multi_tf_{TIMESTAMP}/
│   ├── session_metadata.json
│   ├── comprehensive_report.md
│   ├── timeframes/
│   │   ├── {SYMBOL}_{TIMEFRAME}_{TIMESTAMP}/
│   │   │   ├── factors.parquet
│   │   │   ├── analysis.json
│   │   │   ├── plots/
│   │   │   │   ├── ic_analysis.png
│   │   │   │   ├── factor_correlation.png
│   │   │   │   └── performance_metrics.png
│   │   │   ├── detailed_analysis.md
│   │   │   └── README.md
│   └── summary.json
└── screening_sessions_index.json
```

## 数据质量标准

### 数据完整性要求
- 开盘价、最高价、最低价、收盘价必须 > 0
- 成交量必须 >= 0
- 价格关系: low <= open,close <= high
- 时间序列必须按时间排序
- 重复时间戳必须去重

### 缺失数据处理
```python
missing_data_handling = {
    'price_data': 'forward_fill',   # 价格数据前向填充
    'volume_data': 'zero_fill',     # 成交量置零
    'missing_threshold': 0.05,      # 缺失阈值5%
    'min_observations': 100,        # 最小观测数
    'outlier_detection': 'iqr',     # 异常值检测
    'outlier_threshold': 3.0        # 异常值阈值
}
```

### 数据验证规则
```python
validation_rules = {
    'price_positive': lambda df: (df[['open', 'high', 'low', 'close']] > 0).all().all(),
    'volume_non_negative': lambda df: (df['volume'] >= 0).all(),
    'price_order': lambda df: (df['low'] <= df[['open', 'close']]).all().all() &
                           (df[['open', 'close']] <= df['high']).all().all(),
    'time_sequence': lambda df: df.index.is_monotonic_increasing,
    'no_duplicates': lambda df: not df.index.duplicated().any()
}
```