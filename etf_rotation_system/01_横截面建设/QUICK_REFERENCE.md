# ETF因子面板配置 - 快速参考

## 🚀 快速开始

```bash
# 1. 使用默认配置
python generate_panel_refactored.py

# 2. 自定义配置文件
python generate_panel_refactored.py --config config/my_config.yaml

# 3. 覆盖参数
python generate_panel_refactored.py --workers 8 --data-dir custom_data

# 4. 验证配置
python migrate_to_config.py --validate --test
```

## 📝 核心配置项

### 因子窗口调整
```yaml
factor_windows:
  momentum: [20, 63, 126, 252]     # 动量周期
  volatility: [20, 60, 120]         # 波动率窗口
  rsi: [6, 14, 24]                   # RSI窗口
```

### 因子开关控制
```yaml
factor_enable:
  momentum: true          # 启用动量因子
  hammer_pattern: false   # 禁用锤子线
  vwap_deviation: true    # 启用VWAP偏离度
```

### 性能调优
```yaml
processing:
  max_workers: 8         # 并行进程数
  max_failure_rate: 0.1  # 失败容忍度
```

### 阈值调整
```yaml
thresholds:
  large_order_volume_ratio: 1.5  # 大单阈值
  hammer_lower_shadow_ratio: 2.0 # 锤子线标准
```

## 🎯 常用配置模板

### 短期交易策略
```yaml
factor_windows:
  momentum: [5, 10, 20]
  volatility: [5, 10, 20]
  rsi: [3, 7, 14]

thresholds:
  large_order_volume_ratio: 1.3
```

### 长期投资策略
```yaml
factor_windows:
  momentum: [60, 120, 252]
  volatility: [60, 120, 252]
  rsi: [14, 30, 60]

factor_enable:
  # 禁用短期噪音因子
  overnight_return: false
  intraday_range: false
```

### 高性能配置
```yaml
processing:
  max_workers: -1  # 使用所有CPU核心

factor_enable:
  # 仅保留核心因子
  momentum: true
  volatility: true
  rsi: true
  # 其他因子设为false
```

## 🧪 测试和验证

```bash
# 功能等价性测试
python test_equivalence.py

# 配置验证
python migrate_to_config.py --all

# 性能测试
time python generate_panel_refactored.py --config config/factor_panel_config.yaml
```

## 📊 配置效果

| 配置项 | 影响 | 建议范围 |
|--------|------|----------|
| momentum窗口 | 动量敏感度 | 5-252天 |
| volatility窗口 | 波动率平滑度 | 20-252天 |
| workers | 计算速度 | 1-16 |
| 大单阈值 | 信号强度 | 1.1-2.0 |

## ⚠️ 注意事项

1. **数据质量**: 确保OHLCV数据完整
2. **内存使用**: 大量因子和高频率数据可能消耗大量内存
3. **参数验证**: 配置修改后务必运行验证
4. **备份配置**: 重要配置变更前先备份

## 🔧 故障排除

| 问题 | 检查项 | 解决方案 |
|------|--------|----------|
| 配置加载失败 | YAML语法 | 使用YAML验证器 |
| 计算速度慢 | workers设置 | 增加并行进程数 |
| 内存不足 | 因子数量 | 禁用部分因子 |
| 结果异常 | 数据质量 | 检查输入数据完整性 |

## 📚 相关文档

- [详细配置指南](docs/configuration_guide.md)
- [API文档](docs/api_reference.md)
- [迁移指南](migrate_to_config.py)

## 💡 专业提示

- 使用版本控制管理配置变更
- 为不同策略创建专门的配置文件
- 定期运行性能测试监控系统状态
- 利用因子开关快速测试不同因子组合