# 配置修改示例

## 概述

通过修改 `config/factor_panel_config.yaml` 文件，可以灵活调整因子计算的所有参数，无需修改代码。

## 常见配置修改

### 1. 调整因子时间窗口

#### 短期交易策略配置
```yaml
factor_windows:
  momentum: [5, 10, 20]          # 短期动量周期
  volatility: [5, 10, 20]        # 短期波动率
  rsi: [3, 7, 14]                # 短期RSI
  price_position: [5, 10, 20]     # 短期价格位置
  volume_ratio: [3, 10, 20]      # 短期成交量比率
```

#### 长期投资策略配置
```yaml
factor_windows:
  momentum: [60, 120, 252]       # 长期动量周期
  volatility: [60, 120, 252]     # 长期波动率
  rsi: [14, 30, 60]              # 长期RSI
  price_position: [60, 120, 252]  # 长期价格位置
  volume_ratio: [20, 60, 120]    # 长期成交量比率
```

### 2. 启用/禁用特定因子

#### 仅使用核心因子
```yaml
factor_enable:
  # 基础因子
  momentum: true
  volatility: true
  rsi: true

  # 禁用其他因子
  drawdown: false
  momentum_acceleration: false
  price_position: false
  volume_ratio: false

  # 技术因子全部禁用
  overnight_return: false
  atr: false
  doji_pattern: false
  intraday_range: false
  bullish_engulfing: false
  hammer_pattern: false
  price_impact: false
  volume_price_trend: false
  vol_ma_ratio_5: false
  vol_volatility_20: false
  true_range: false
  buy_pressure: false

  # 资金流因子全部禁用
  vwap_deviation: false
  amount_surge_5d: false
  price_volume_div: false
  intraday_position: false
  large_order_signal: false
```

#### 技术分析重点配置
```yaml
factor_enable:
  # 基础因子保持开启
  momentum: true
  volatility: true
  rsi: true

  # 重点启用技术因子
  overnight_return: true
  atr: true
  doji_pattern: true
  intraday_range: true
  bullish_engulfing: true
  hammer_pattern: true

  # 禁用资金流因子（如果数据不足）
  vwap_deviation: false
  amount_surge_5d: false
  large_order_signal: false
```

### 3. 调整计算阈值

#### 更严格的信号识别
```yaml
thresholds:
  large_order_volume_ratio: 1.5      # 提高大单识别阈值
  doji_body_threshold: 0.001         # 十字星实体阈值
  hammer_lower_shadow_ratio: 2.5     # 更严格的锤子线标准
  hammer_upper_shadow_ratio: 0.8     # 更严格的上影线限制
```

#### 更宽松的信号识别
```yaml
thresholds:
  large_order_volume_ratio: 1.1      # 降低大单识别阈值
  hammer_lower_shadow_ratio: 1.5     # 放宽锤子线标准
  hammer_upper_shadow_ratio: 1.2     # 放宽上影线限制
```

### 4. 性能优化配置

#### 高性能配置（多核CPU）
```yaml
processing:
  max_workers: -1                    # 使用所有CPU核心
  continue_on_symbol_error: true    # 容错处理
  max_failure_rate: 0.05            # 降低失败容忍度

# 禁用计算密集的因子
factor_enable:
  hammer_pattern: false
  bullish_engulfing: false
  intraday_range: false
  price_impact: false
```

#### 低内存配置
```yaml
processing:
  max_workers: 2                     # 减少并行进程

# 仅保留最重要的因子
factor_enable:
  momentum: true
  volatility: true
  rsi: true

  # 禁用其他所有因子
  drawdown: false
  momentum_acceleration: false
  price_position: false
  volume_ratio: false
  overnight_return: false
  atr: false
  doji_pattern: false
  intraday_range: false
  bullish_engulfing: false
  hammer_pattern: false
  price_impact: false
  volume_price_trend: false
  vol_ma_ratio_5: false
  vol_volatility_20: false
  true_range: false
  buy_pressure: false
  vwap_deviation: false
  amount_surge_5d: false
  price_volume_div: false
  intraday_position: false
  large_order_signal: false
```

### 5. 路径配置

#### 开发环境路径
```yaml
paths:
  data_dir: "/Users/zhangshenshen/深度量化0927/raw/ETF/daily"
  output_dir: "/Users/zhangshenshen/深度量化0927/etf_rotation_system/data/results/panels"
  config_file: "config/factor_panel_config.yaml"
```

#### 生产环境路径
```yaml
paths:
  data_dir: "/data/etf/daily"
  output_dir: "/data/etf/results/panels"
  config_file: "/etc/etf/factor_panel_config.yaml"
```

## 配置验证

修改配置后，务必进行验证：

```bash
# 验证配置文件语法
python migrate_to_config.py --validate

# 测试配置加载
python migrate_to_config.py --test

# 运行完整测试
python migrate_to_config.py --all
```

## 配置最佳实践

1. **备份配置**: 修改前备份原始配置文件
2. **逐步调整**: 一次只修改一类参数，便于定位问题
3. **测试验证**: 每次修改后运行验证确保配置正确
4. **版本控制**: 使用Git管理配置变更历史
5. **文档记录**: 记录重要配置变更的原因和效果

## 常见配置错误

### YAML语法错误
```yaml
# 错误示例
factor_windows:
  momentum: [20, 63, 126, 252  # 缺少闭合括号

# 正确示例
factor_windows:
  momentum: [20, 63, 126, 252]
```

### 参数类型错误
```yaml
# 错误示例
processing:
  max_workers: "8"  # 字符串而非数字

# 正确示例
processing:
  max_workers: 8    # 数字
```

### 参数范围错误
```yaml
# 错误示例
factor_windows:
  momentum: [0, -5, 10]  # 包含负数和零

# 正确示例
factor_windows:
  momentum: [5, 10, 20]   # 所有为正数
```
