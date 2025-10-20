# 使用自定义配置

## 创建自定义配置文件

### 1. 复制默认配置
```bash
cp config/factor_panel_config.yaml config/my_custom_config.yaml
```

### 2. 修改自定义配置
编辑 `config/my_custom_config.yaml` 文件：

```yaml
# 自定义策略配置示例
trading:
  days_per_year: 252
  epsilon_small: 1.0e-10
  min_periods: 1

# 短期交易策略
factor_windows:
  momentum: [5, 10, 20]          # 短期动量
  volatility: [5, 10, 20]        # 短期波动率
  drawdown: [10, 20]             # 短期回撤
  rsi: [3, 7, 14]                # 短期RSI
  price_position: [5, 10, 20]     # 短期价格位置
  volume_ratio: [3, 10, 20]      # 短期成交量比率

# 技术指标窗口
atr_period: 10
vpt_trend_window: 15
vol_volatility_window: 15

# 更严格的阈值
thresholds:
  large_order_volume_ratio: 1.3
  doji_body_threshold: null
  hammer_lower_shadow_ratio: 2.2
  hammer_upper_shadow_ratio: 0.9

# 高性能配置
processing:
  max_workers: 8
  continue_on_symbol_error: true
  max_failure_rate: 0.05

# 核心因子仅启用
factor_enable:
  momentum: true
  volatility: true
  rsi: true
  atr: true

  # 禁用其他因子
  drawdown: false
  momentum_acceleration: false
  price_position: false
  volume_ratio: false
  overnight_return: false
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

# 自定义路径
paths:
  data_dir: "/Users/zhangshenshen/深度量化0927/raw/ETF/daily"
  output_dir: "/Users/zhangshenshen/深度量化0927/custom_results/panels"
  config_file: "config/my_custom_config.yaml"
```

## 使用自定义配置

### 基本用法
```bash
python generate_panel_refactored.py --config config/my_custom_config.yaml
```

### 覆盖特定参数
命令行参数会覆盖配置文件中的相应设置：

```bash
# 覆盖工作进程数
python generate_panel_refactored.py \
  --config config/my_custom_config.yaml \
  --workers 4

# 覆盖数据目录
python generate_panel_refactored.py \
  --config config/my_custom_config.yaml \
  --data-dir "/path/to/custom/data"

# 覆盖输出目录
python generate_panel_refactored.py \
  --config config/my_custom_config.yaml \
  --output-dir "/path/to/custom/output"

# 组合多个覆盖参数
python generate_panel_refactored.py \
  --config config/my_custom_config.yaml \
  --workers 6 \
  --data-dir "/custom/data/path" \
  --output-dir "/custom/output/path"
```

## 配置文件最佳实践

### 1. 命名规范
```bash
# 推荐的配置文件命名
config/short_term_config.yaml      # 短期策略
config/long_term_config.yaml       # 长期策略
config/high_performance_config.yaml # 高性能配置
config/low_memory_config.yaml      # 低内存配置
config/production_config.yaml      # 生产环境
config/development_config.yaml     # 开发环境
```

### 2. 配置文件模板

#### 短期交易配置 (`short_term_config.yaml`)
```yaml
factor_windows:
  momentum: [3, 5, 10, 20]
  volatility: [3, 5, 10, 20]
  rsi: [3, 5, 10, 15]

processing:
  max_workers: -1  # 最大性能
```

#### 长期投资配置 (`long_term_config.yaml`)
```yaml
factor_windows:
  momentum: [60, 120, 252]
  volatility: [60, 120, 252]
  rsi: [30, 60, 120]

factor_enable:
  # 禁用短期噪音因子
  overnight_return: false
  intraday_range: false
  buy_pressure: false
```

#### 回测配置 (`backtest_config.yaml`)
```yaml
# 确保结果可重现
processing:
  max_workers: 1  # 单线程避免随机性
  continue_on_symbol_error: false  # 严格模式

output:
  save_execution_log: true
  save_metadata: true
  timestamp_subdirectory: true
```

## 配置验证

创建自定义配置后，务必进行验证：

```bash
# 1. 验证YAML语法
python -c "import yaml; yaml.safe_load(open('config/my_custom_config.yaml'))"

# 2. 使用项目工具验证
python migrate_to_config.py --config config/my_custom_config.yaml --validate

# 3. 测试配置加载
python -c "
from config.config_classes import FactorPanelConfig
config = FactorPanelConfig.from_yaml('config/my_custom_config.yaml')
print('配置加载成功')
print(f'因子数量: {len([k for k, v in config.factor_enable.__dict__.items() if v])}')
"

# 4. 运行小规模测试
python generate_panel_refactored.py \
  --config config/my_custom_config.yaml \
  --workers 2
```

## 配置管理

### 版本控制
```bash
# 提交配置文件
git add config/my_custom_config.yaml
git commit -m "添加短期交易策略配置"

# 查看配置变更历史
git log -p config/my_custom_config.yaml

# 比较不同配置
git diff HEAD config/my_custom_config.yaml config/short_term_config.yaml
```

### 配置备份
```bash
# 创建配置备份
cp config/my_custom_config.yaml config/my_custom_config.yaml.backup

# 批量备份
mkdir -p config/backups/$(date +%Y%m%d)
cp config/*.yaml config/backups/$(date +%Y%m%d)/
```

## 故障排除

### 配置文件找不到
```
错误: 配置文件不存在: config/my_custom_config.yaml
解决: 检查文件路径和文件名是否正确
```

### YAML语法错误
```
错误: 配置加载失败，使用默认配置
解决: 使用YAML验证器检查语法，注意缩进和引号
```

### 参数冲突
```
现象: 命令行参数没有生效
解决: 确保命令行参数在配置文件参数之后
```

### 性能问题
```
现象: 自定义配置运行很慢
解决: 检查max_workers设置，禁用不需要的因子
```
