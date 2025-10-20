# ETF因子面板配置指南

## 概述

本文档详细说明了ETF因子面板生成系统的配置架构和使用方法。通过配置驱动的设计，用户可以灵活调整所有计算参数，而无需修改代码。

## 配置文件结构

配置文件采用YAML格式，位于 `config/factor_panel_config.yaml`。主要包含以下模块：

### 1. 交易参数 (trading)
```yaml
trading:
  days_per_year: 252          # 年化交易日数
  epsilon_small: 1.0e-10      # 避免除零的小数值
  min_periods: 1              # 滚动窗口最小值
```

**说明**:
- `days_per_year`: 用于波动率年化计算，默认252个交易日
- `epsilon_small`: 数值计算中的小常数，防止除零错误
- `min_periods`: 滚动窗口计算的最小有效数据点数

### 2. 因子时间窗口 (factor_windows)
```yaml
factor_windows:
  momentum: [20, 63, 126, 252]     # 动量因子周期
  volatility: [20, 60, 120]         # 波动率因子窗口
  drawdown: [63, 126]               # 回撤因子窗口
  rsi: [6, 14, 24]                   # RSI计算窗口
  price_position: [20, 60, 120]      # 价格位置窗口
  volume_ratio: [5, 20, 60]         # 成交量比率窗口

  # 技术指标特定窗口
  atr_period: 14                     # ATR计算周期
  vpt_trend_window: 20               # 量价趋势窗口
  vol_volatility_window: 20          # 成交量波动窗口
  amount_surge_short: 5              # 成交额突增短期窗口
  amount_surge_long: 20              # 成交额突增长期窗口
  vol_ratio_window: 20               # 成交量比率基准窗口
  intraday_position_window: 5        # 日内位置平滑窗口
  price_volume_div_window: 5         # 量价背离平滑窗口
```

**说明**:
- 数组中的每个值对应一个独立的因子
- 窗口长度影响因子的计算周期和敏感度
- 短期窗口反应更快，长期窗口更稳定

### 3. 阈值参数 (thresholds)
```yaml
thresholds:
  large_order_volume_ratio: 1.2      # 大单流入成交量比率阈值
  doji_body_threshold: null          # 十字星身体阈值（null使用epsilon）
  hammer_lower_shadow_ratio: 2.0     # 锤子线下影线倍数
  hammer_upper_shadow_ratio: 1.0     # 锤子线上影线倍数
```

**说明**:
- `large_order_volume_ratio`: 识别大单流入的成交量放大倍数
- `hammer_lower_shadow_ratio`: 锤子线形态的下影线与实体比例
- `hammer_upper_shadow_ratio`: 锤子线形态的上影线与实体比例

### 4. 路径配置 (paths)
```yaml
paths:
  data_dir: "raw/ETF/daily"                           # 数据目录
  output_dir: "etf_rotation_system/data/results/panels" # 输出目录
  config_file: "config/etf_config.yaml"               # 默认配置文件
```

### 5. 并行处理配置 (processing)
```yaml
processing:
  max_workers: 4               # 默认并行进程数
  continue_on_symbol_error: true # 单个标的失败时继续处理
  max_failure_rate: 0.1        # 最大失败率容忍
```

### 6. 因子开关 (factor_enable)
```yaml
factor_enable:
  # 原有18个因子
  momentum: true
  volatility: true
  drawdown: true
  momentum_acceleration: true
  rsi: true
  price_position: true
  volume_ratio: true

  # 新增技术因子
  overnight_return: true
  atr: true
  doji_pattern: true
  intraday_range: true
  bullish_engulfing: true
  hammer_pattern: true
  price_impact: true
  volume_price_trend: true
  vol_ma_ratio_5: true
  vol_volatility_20: true
  true_range: true
  buy_pressure: true

  # 资金流因子
  vwap_deviation: true
  amount_surge_5d: true
  price_volume_div: true
  intraday_position: true
  large_order_signal: true
```

**说明**:
- 每个因子都可以独立启用或禁用
- 禁用不需要的因子可以提升计算性能
- 资金流因子需要成交额数据支持

### 7. 数据处理配置 (data_processing)
```yaml
data_processing:
  required_columns:
    - "date"
    - "open"
    - "high"
    - "low"
    - "close"
    - "volume"
    - "symbol"

  optional_columns:
    - "amount"  # 资金流数据

  volume_column_alias: "vol"  # 成交量列别名
  fallback_estimation: true   # 缺少amount时使用估算
```

### 8. 输出配置 (output)
```yaml
output:
  save_execution_log: true    # 保存执行日志
  save_metadata: true         # 保存元数据
  timestamp_subdirectory: true # 使用时间戳子目录
```

### 9. 显示配置 (display)
```yaml
display:
  log_separator_length: 80    # 日志分隔符长度
  progress_bar_desc: "计算因子" # 进度条描述
  factor_list_start: 1       # 因子列表起始编号
  json_indent: 2             # JSON文件缩进
  coverage_format: ".2%"     # 覆盖率显示格式
```

### 10. 计算常量 (constants)
```yaml
constants:
  rsi_multiplier: 100         # RSI计算乘数
  concat_axis: 1             # DataFrame连接轴
  astype_float: "float"      # 类型转换
  shift_days: 1              # 位移天数
  concat_max_axis: 1         # max()轴参数
  sign_multiplier: 1         # np.sign乘数
```

## 使用方法

### 基本使用
```bash
# 使用默认配置
python generate_panel_refactored.py

# 指定配置文件
python generate_panel_refactored.py --config config/factor_panel_config.yaml

# 覆盖特定参数
python generate_panel_refactored.py --workers 8 --data-dir custom_data
```

### 自定义配置示例

#### 1. 调整因子窗口
```yaml
# 短期策略配置
factor_windows:
  momentum: [5, 10, 20]          # 更短的动量周期
  volatility: [5, 10, 20]        # 更短的波动率窗口
  rsi: [3, 7, 14]                # 更敏感的RSI
```

#### 2. 启用/禁用因子
```yaml
# 仅使用核心因子
factor_enable:
  momentum: true
  volatility: true
  rsi: true
  # 禁用其他所有因子
  drawdown: false
  doji_pattern: false
  hammer_pattern: false
  # ... 其他因子设为false
```

#### 3. 调整阈值
```yaml
# 更严格的大单识别
thresholds:
  large_order_volume_ratio: 1.5  # 提高阈值
  hammer_lower_shadow_ratio: 2.5  # 更严格的锤子线标准
```

#### 4. 性能优化配置
```yaml
# 高性能配置
processing:
  max_workers: 8                    # 增加并行进程
  continue_on_symbol_error: true    # 容错处理
  max_failure_rate: 0.05            # 降低失败容忍度

factor_enable:
  # 禁用计算密集的因子
  hammer_pattern: false
  bullish_engulfing: false
  intraday_range: false
```

### 在代码中使用配置
```python
from config.config_classes import FactorPanelConfig

# 加载配置
config = FactorPanelConfig.from_yaml('config/factor_panel_config.yaml')

# 访问配置参数
print(f"动量窗口: {config.factor_windows.momentum}")
print(f"工作进程数: {config.processing.max_workers}")

# 动态修改配置
config.trading.days_per_year = 250
config.factor_windows.momentum = [10, 20, 30]

# 验证配置
if config.validate():
    print("配置有效")
```

## 配置验证

配置系统包含多层验证机制：

### 1. 类型验证
- 确保数值类型正确
- 检查数组元素类型一致

### 2. 范围验证
- 检查窗口大小是否为正数
- 验证比率参数是否合理

### 3. 逻辑验证
- 确保短期窗口小于长期窗口
- 检查因子开关设置

### 4. 完整性验证
- 验证必需配置项存在
- 检查配置项之间的一致性

## 最佳实践

### 1. 配置管理
- 为不同策略创建专门的配置文件
- 使用版本控制管理配置变更
- 定期备份关键配置

### 2. 参数调优
- 从默认参数开始逐步调整
- 使用回测验证参数效果
- 记录参数调整的历史和原因

### 3. 性能优化
- 禁用不需要的因子提升性能
- 根据硬件调整并行进程数
- 监控内存和CPU使用情况

### 4. 错误处理
- 设置合理的失败容忍度
- 保存详细的执行日志
- 定期检查输出结果质量

## 故障排除

### 常见问题

#### 1. 配置加载失败
```
错误: 配置加载失败，使用默认配置
解决: 检查YAML语法和文件路径
```

#### 2. 参数验证失败
```
错误: Window size must be positive
解决: 检查窗口参数是否为正数
```

#### 3. 因子计算失败
```
错误: 无有效因子数据
解决: 检查数据质量和因子开关设置
```

#### 4. 性能问题
```
问题: 计算速度过慢
解决: 调整max_workers或禁用部分因子
```

### 调试技巧

1. **启用详细日志**
   ```yaml
   logging:
     level: "DEBUG"
   ```

2. **使用测试数据**
   ```bash
   python test_equivalence.py
   ```

3. **验证配置**
   ```python
   config = FactorPanelConfig.from_yaml('config.yaml')
   config.validate()
   ```

## 扩展配置

如需添加新的配置项：

1. 在YAML文件中添加配置
2. 在相应的配置类中添加字段
3. 添加验证逻辑
4. 更新使用该配置的代码

示例：
```yaml
# 添加新配置
new_feature:
  enabled: true
  parameter1: 100
  parameter2: 0.5
```

```python
# 在配置类中添加
@dataclass
class NewFeatureConfig:
    enabled: bool = True
    parameter1: int = 100
    parameter2: float = 0.5

# 在主配置类中添加
new_feature: NewFeatureConfig = field(default_factory=NewFeatureConfig)
```

通过这套配置系统，ETF因子面板生成实现了高度的灵活性和可维护性，满足不同策略和环境的需要。