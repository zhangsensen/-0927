# 基本使用示例

## 使用默认配置

最简单的使用方式，所有参数使用配置文件中的默认值：

```bash
python generate_panel_refactored.py
```

这将：
- 读取 `config/factor_panel_config.yaml` 配置文件
- 从配置的数据目录加载ETF数据
- 计算所有启用的因子（默认36个）
- 将结果保存到配置的输出目录
- 生成执行日志和元数据文件

## 指定数据目录和输出目录

如果需要自定义数据路径和输出位置：

```bash
python generate_panel_refactored.py \
  --data-dir "/Users/zhangshenshen/深度量化0927/raw/ETF/daily" \
  --output-dir "/Users/zhangshenshen/深度量化0927/etf_rotation_system/data/results/panels"
```

## 调整并行进程数

根据硬件配置优化性能：

```bash
# 使用8个并行进程
python generate_panel_refactored.py --workers 8

# 使用所有CPU核心
python generate_panel_refactored.py --workers -1
```

## 完整示例

```bash
# 完整参数示例
python generate_panel_refactored.py \
  --config config/factor_panel_config.yaml \
  --data-dir "/Users/zhangshenshen/深度量化0927/raw/ETF/daily" \
  --output-dir "/Users/zhangshenshen/深度量化0927/etf_rotation_system/data/results/panels" \
  --workers 8
```

## 预期输出

成功运行后，将在输出目录创建：

```
panel_YYYYMMDD_HHMMSS/
├── panel.parquet         # 因子数据（约15-20MB）
├── metadata.json         # 元数据文件
└── execution_log.txt     # 执行日志
```

示例输出信息：
- 标的数量：43个ETF
- 因子数量：36个
- 数据点：56,575条
- 覆盖率：98.47%
- 执行时间：1-3秒（取决于硬件配置）

## 常见问题

### 配置文件未找到
```
错误: 配置加载失败，使用默认配置
解决: 确保 config/factor_panel_config.yaml 文件存在
```

### 数据目录为空
```
错误: 未找到任何ETF数据文件
解决: 检查数据目录路径和文件格式（.parquet）
```

### 内存不足
```
解决: 减少并行进程数 --workers 4
或禁用部分因子以降低内存使用
```
