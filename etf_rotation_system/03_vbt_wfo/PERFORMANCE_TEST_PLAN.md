# WFO 性能测试计划

**测试日期**: 2025-10-24  
**测试目标**: 验证 1000 策略/Period 的回测性能

---

## 🎯 测试配置

### 策略数量计算
```
250 权重组合 × 2 Top-N × 2 调仓频率 = 1000 策略/Period
```

### 详细参数

**回测参数**:
- `top_n_list`: [3, 5]
- `rebalance_freq_list`: [5, 10]
- `max_combinations`: 250

**权重网格**:
- `grid_points`: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] (6个点)
- `weight_sum_range`: [0.8, 1.2]

**IS/OOS 设置**:
- `run_is`: true ✅
- `run_oos`: true ✅
- 预计总策略数：**1000 × 2 (IS+OOS) × Period数**

---

## 📊 预期性能

### 基于历史数据推测

**旧版性能** (嵌套并行问题前):
- 速度: ~2000 策略/秒
- 1000策略/Period: ~0.5秒

**修复后性能** (顺序 Period + 策略级并行):
- 数据加载: 一次性 (~10秒)
- 回测速度: ~2000 策略/秒 (8 workers)
- 单 Period (IS+OOS): ~1秒

**假设 10 个 Period**:
- 总策略: 1000 × 2 × 10 = 20,000
- 数据加载: 10秒
- 回测计算: 10秒
- **总耗时**: ~20秒
- **平均速度**: 1000 策略/秒

---

## 🧪 测试步骤

### 1. 清理环境
```bash
cd /Users/zhangshenshen/深度量化0927/etf_rotation_system/03_vbt_wfo
rm -rf /Users/zhangshenshen/深度量化0927/etf_rotation_system/data/results/vbtwfo/wfo_*
```

### 2. 运行测试
```bash
python3 production_runner_optimized.py
```

### 3. 观察指标
- 数据加载时间
- 每个 Period 耗时
- IS vs OOS 速度
- 整体吞吐量 (策略/秒)

---

## 📈 性能指标

### 关键指标

| 指标 | 目标值 | 单位 |
|------|--------|------|
| 数据加载时间 | < 15秒 | 秒 |
| Period 处理速度 | > 800 | 策略/秒 |
| 整体吞吐量 | > 1000 | 策略/秒 |
| 内存使用 | < 8GB | GB |

### 瓶颈分析

**可能瓶颈**:
1. 数据加载 IO (Parquet 读取)
2. 权重组合生成 (250 组合)
3. Portfolio 计算 (vectorbt)
4. 结果聚合和排序

**优化方向**:
- 减少 Top-N 数量 (当前 2 个)
- 减少调仓频率 (当前 2 个)
- 调整 chunk_size (当前 100)
- 减少 max_combinations (当前 250)

---

## 🔧 配置对比

### 测试配置 (1000 策略)
```yaml
backtest_config:
  top_n_list: [3, 5]                    # 2个
  rebalance_freq_list: [5, 10]         # 2个
  weight_grid:
    max_combinations: 250               # 250组合
    # → 250 × 2 × 2 = 1000 策略
```

### 快速配置 (100 策略)
```yaml
backtest_config:
  top_n_list: [5]                       # 1个
  rebalance_freq_list: [10]             # 1个
  weight_grid:
    max_combinations: 100               # 100组合
    # → 100 × 1 × 1 = 100 策略
```

### 完整配置 (10000+ 策略)
```yaml
backtest_config:
  top_n_list: [1, 2, 3, 5, 8]          # 5个
  rebalance_freq_list: [5, 10, 20]     # 3个
  weight_grid:
    max_combinations: 1000              # 1000组合
    # → 1000 × 5 × 3 = 15000 策略
```

---

## 📝 改进记录

### 本次修改 (2025-10-24)

**1. 开启 IS 回测**:
```yaml
run_is: true   # 原来 false
run_oos: true
```

**2. 删除普通回测代码**:
```bash
rm parallel_backtest_configurable.py
```
- 理由: 避免混淆，只保留 WFO 版本
- 保留文件: `production_runner_optimized.py`

**3. 调整为 1000 策略配置**:
- Top-N: 5 → 2 个
- 调仓频率: 3 → 2 个
- max_combinations: 10000 → 250

---

## ✅ 测试检查清单

运行前检查:
- [ ] 配置文件正确 (`simple_config.yaml`)
- [ ] 数据文件存在 (panel, prices, screening)
- [ ] 磁盘空间充足 (> 1GB)

运行中观察:
- [ ] IS 和 OOS 都在运行
- [ ] 每个 Period 显示策略数量
- [ ] 速度符合预期 (> 800 策略/秒)
- [ ] 内存使用正常 (< 8GB)

运行后验证:
- [ ] 结果保存成功 (summary.json, results.pkl)
- [ ] 日志完整 (logs/wfo.log)
- [ ] IS vs OOS Sharpe 对比合理

---

## 🎯 预期输出

### 日志示例
```
🚀 WFO生产环境回测启动
📁 结果目录: .../vbtwfo/wfo_20251024_XXXXXX
📝 日志文件: .../logs/wfo.log

[1/4] 初始化引擎
[2/4] 加载数据 (一次性)
  数据加载完成，耗时: 10.5秒

[3/4] 顺序处理每个Period (内部策略级并行)
🔧 配置: run_is=True, run_oos=True
------------------------------------------------------------
Period 1/10: 2022-01-01 ~ 2023-03-31
  IS回测完成: 1,000个结果, 耗时1.2秒
  OOS回测完成: 200个结果 (Top 200), 耗时0.5秒
  Period完成: 总耗时1.8秒, 速度1111策略/秒

Period 2/10: 2022-04-01 ~ 2023-06-30
  ...

[4/4] 保存结果
  摘要已保存: .../summary.json
  详细结果已保存: .../results.pkl

============================================================
WFO生产环境回测完成
============================================================
总Period数:     10
总策略数:       20,000
  - IS策略:     10,000
  - OOS策略:    10,000
总耗时:         0.3分钟
数据加载:       10.5秒 (53%)
回测计算:       0.2分钟
整体速度:       1111 策略/秒
============================================================

✅ 完成 | 所有结果保存在: .../vbtwfo/wfo_20251024_XXXXXX
```

---

## 🚀 开始测试

运行命令:
```bash
cd /Users/zhangshenshen/深度量化0927/etf_rotation_system/03_vbt_wfo
python3 production_runner_optimized.py
```

或使用快捷脚本:
```bash
./run_wfo.sh
```
