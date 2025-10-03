# 配置文件模板说明

## 模板文件

1. **single_screening_template.yaml** - 单个筛选任务配置模板
2. **batch_screening_template.yaml** - 批量筛选任务配置模板
3. **high_freq_template.yaml** - 高频交易优化配置模板
4. **deep_analysis_template.yaml** - 深度分析配置模板

## 使用方法

1. 复制模板文件
2. 修改相关参数（股票代码、时间框架等）
3. 使用batch_screener.py加载配置运行

## 配置参数说明

### 基础参数
- `symbols`: 股票代码列表
- `timeframes`: 时间框架列表
- `ic_horizons`: IC计算周期

### 筛选参数
- `min_ic_threshold`: 最小IC阈值
- `vif_threshold`: VIF阈值
- `correlation_threshold`: 相关性阈值

### 权重配置
- `weights`: 5维度评分权重分配

详细参数说明请参考主文档。
