# WFO (Walk-Forward Optimization) 模块

## 📋 概述

WFO（Walk-Forward Optimization，滚动前向优化）是业界公认的抗过拟合验证方法。本模块将精细策略系统改造为WFO模式，通过时间序列上的滚动训练和验证，检测策略的真实可用性。

## 🎯 核心功能

### 1. 时间窗口管理
- **训练窗口(IS)**: 12个月，用于优化参数
- **测试窗口(OOS)**: 3个月，用于验证效果
- **步进**: 3个月，滚动前进

### 2. 过拟合检测
- **IS/OOS性能比**: 衡量过拟合程度
- **性能衰减率**: 评估OOS相对IS的下降幅度
- **参数稳定性**: 检查不同周期参数一致性
- **OOS胜率**: 统计通过阈值的周期比例

### 3. 分析报告
- JSON格式完整结果
- Markdown格式可读报告
- IS/OOS对比分析
- 部署建议和风险评估

## 🚀 快速开始

### 运行WFO

```bash
cd /Users/zhangshenshen/深度量化0927/etf_rotation_system/04_精细策略

python run_wfo.py \
    --vbt-results "../data/results/backtest/backtest_merged_20251022_192014" \
    --config "./config/fine_strategy_config.yaml" \
    --output "./output/wfo" \
    --top-k 200
```

### 参数说明

- `--vbt-results`: VBT回测结果目录（包含results.csv）
- `--config`: 配置文件路径
- `--output`: WFO结果输出目录
- `--top-k`: 从VBT结果提取Top-K策略进行WFO验证

## 📊 输出文件

### 1. `wfo_results_YYYYMMDD_HHMMSS.json`
完整的WFO结果，包含：
- 配置信息
- 每个周期的IS/OOS结果
- 汇总统计
- 过拟合分析

### 2. `wfo_analysis_YYYYMMDD_HHMMSS.json`
过拟合分析详情：
- 核心指标（IS/OOS Sharpe、过拟合比等）
- 周期明细
- 综合评级
- 风险和建议

### 3. `WFO_REPORT_YYYYMMDD_HHMMSS.md`
Markdown格式的可读报告，包含：
- 核心指标表格
- 过拟合检测结果
- 周期明细
- 综合评级和建议

## 🔬 WFO工作流程

```
1. 加载VBT回测结果（Top 200策略）
2. 生成WFO周期（根据数据范围）
3. 对每个周期：
   a. IS阶段：在训练窗口优化参数
   b. 选择Top-N策略
   c. OOS阶段：在测试窗口验证
   d. 记录IS/OOS表现
4. 汇总所有周期结果
5. 计算过拟合指标
6. 生成分析报告
```

## 📈 评估指标

### 核心指标

| 指标 | 描述 | 优秀 | 良好 | 需改进 |
|------|------|------|------|--------|
| 过拟合比 | IS/OOS Sharpe | ≤1.2 | ≤1.5 | >1.5 |
| 性能衰减 | (IS-OOS)/IS | <10% | <25% | >25% |
| OOS胜率 | OOS≥阈值比例 | >80% | >60% | <60% |
| IS-OOS相关性 | 参数稳定性 | >0.7 | >0.5 | <0.5 |

### 综合评级

- **A级**: 优秀，可部署
- **B级**: 良好，可谨慎部署
- **C级**: 及格，需要进一步优化
- **D级**: 需改进
- **F级**: 不及格，不建议部署

## ⚙️ 配置说明

在 `config/fine_strategy_config.yaml` 中配置WFO参数：

```yaml
wfo_config:
  enabled: true
  
  time_windows:
    train_window_months: 12     # 训练窗口
    test_window_months: 3       # 测试窗口
    step_months: 3              # 步进
  
  data_range:
    start_date: "2022-01-01"    # 数据开始（需根据实际调整）
    end_date: "2024-12-31"      # 数据结束
  
  in_sample:
    top_n_strategies: 10        # IS期选择Top-N进入OOS
    min_sharpe: 0.40            # IS最小Sharpe要求
  
  out_of_sample:
    min_sharpe: 0.30            # OOS最小Sharpe阈值
  
  overfit_detection:
    max_overfit_ratio: 1.5      # 最大过拟合比
    max_decay_rate: 0.25        # 最大衰减率
```

## 🧪 测试示例

### 快速测试（使用merged结果）

```bash
python run_wfo.py \
    --vbt-results "../data/results/backtest/backtest_merged_20251022_192014" \
    --top-k 50
```

这将对Top 50策略运行WFO验证，大约需要5-10分钟。

## 📝 解读报告

### 示例输出

```
【核心结论】
  总周期数: 8
  OOS平均Sharpe: 0.456
  过拟合比: 1.18
  性能衰减: 12.3%
  综合评级: B
  可部署: ✅ 是
```

### 解读要点

1. **过拟合比 <1.5**: 说明IS和OOS表现差距可控
2. **性能衰减 <25%**: OOS性能没有大幅下降
3. **评级B**: 策略稳健性良好，可谨慎部署
4. **建议**: 查看详细报告了解风险和具体建议

## 🔧 高级用法

### 自定义时间窗口

修改 `fine_strategy_config.yaml`:

```yaml
wfo_config:
  time_windows:
    train_window_months: 18  # 更长的训练窗口
    test_window_months: 6    # 更长的测试窗口
    step_months: 6           # 更大的步进
```

### 调整筛选标准

```yaml
wfo_config:
  in_sample:
    top_n_strategies: 5      # 只选Top-5进入OOS
    min_sharpe: 0.50         # 提高IS Sharpe要求
  
  out_of_sample:
    min_sharpe: 0.40         # 提高OOS阈值
```

## ⚠️ 重要说明

### 当前实现

本版本使用**统计模拟**生成IS/OOS结果，主要用于：
1. 验证WFO架构的正确性
2. 快速评估参数组合的稳健性
3. 提供过拟合检测的框架

### 生产部署

实际部署时需要：
1. 集成真实的VectorBT回测引擎
2. 加载分时段的ETF价格数据
3. 计算分时段的因子值
4. 运行真实的分时段回测

代码中已标注需要替换的部分（见 `wfo_backtest_runner.py`）。

## 🎓 理论背景

### 为什么需要WFO？

传统回测问题：
- 使用全样本数据优化参数
- 相当于"用未来信息优化过去"
- 结果往往过度优化，实盘失效

WFO优势：
- 严格的时间隔离
- 只用过去数据优化，用未来数据验证
- 模拟真实交易环境
- 有效降低过拟合风险

### 何时使用WFO？

适用场景：
- ✅ 大规模参数搜索（如本项目420K策略）
- ✅ 多因子组合优化
- ✅ 需要实盘部署的策略
- ✅ 对稳健性要求高的场景

不适用场景：
- ❌ 数据量太少（<2年）
- ❌ 单一简单策略
- ❌ 仅用于学术研究

## 🤝 贡献

如需改进或建议，请联系开发团队。

---

**文档版本**: v1.0  
**最后更新**: 2025-10-22  
**作者**: AI编码代理
