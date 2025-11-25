# ETF轮动WFO排序系统

> **核心目标**: 通过Walk-Forward优化找到最优的因子组合排序，实现ETF轮动策略的超额收益
> 
> **排序方式**: ✅ 默认使用 **ML 排序** (A/B测试验证: 平均 Sharpe +69%, 年化收益 +7.87%)

[![运行状态](https://img.shields.io/badge/status-production-green)]()
[![排序模式](https://img.shields.io/badge/ranking-ML%20(default)-brightgreen)]()
[![Python版本](https://img.shields.io/badge/python-3.11-blue)]()
[![数据范围](https://img.shields.io/badge/data-2020--2025-orange)]()

---

## 🚀 快速开始

```bash
cd /Users/zhangshenshen/深度量化0927/etf_rotation_experiments

# 运行 WFO 优化 (默认使用 ML 排序)
python run_combo_wfo.py

# 结果输出至: results/run_YYYYMMDD_HHMMSS/
```

**默认行为**: 系统会使用 ML 模型对 WFO 组合进行排序, 自动生成 `ranking_ml_top200.parquet`

---

## 🎯 项目目标

- **收益目标**: Top100组合年化收益率 > 15%
- **风险目标**: Sharpe比率 > 1.2，最大回撤 < 15%
- **数据基础**: 43只ETF，2020-2025年，1400+交易日
- **排序方式**: ML 模型排序 (生产默认) | WFO 指标排序 (备用)

---

## 📊 排序模式说明

### ML 排序 (生产默认) ✅

- **配置**: `configs/combo_wfo_config.yaml` 中 `ranking.method: "ml"`
- **优势**: A/B 测试验证 (Top-200)
  - 平均年化收益: **+7.87%** (19.06% vs 11.20%)
  - 平均 Sharpe: **+0.379** (0.927 vs 0.548)
  - 平均回撤改善: **-8.56%** (-21.65% vs -30.20%)
- **适用**: 生产环境, 具备良好的泛化能力和风险控制

### WFO 排序 (备用) ⚠️

- **配置**: 修改 `configs/combo_wfo_config.yaml` 中 `ranking.method: "wfo"`
- **用途**: 对照基准或 ML 模型不可用时的回退选项
- **切换**: 改配置后重新运行 `python run_combo_wfo.py` 即可

**详细文档**: 参见 `docs/ML_RANKING_INTEGRATION_GUIDE.md`

---

## 📂 项目结构

```
etf_rotation_experiments/
├── README.md                      # 本文档
├── PROJECT_BASELINE.md            # 详细技术文档
├── run_combo_wfo.py               # ⭐ WFO主入口
├── apply_ranker.py                # ⭐ ML排序脚本
├── core/                          # ⭐ 核心模块
│   ├── data_loader.py             # ETF数据加载
│   ├── precise_factor_library_v2.py # 因子计算库
│   ├── cross_section_processor.py # 横截面处理
│   ├── combo_wfo_optimizer.py     # WFO优化器
│   └── pipeline.py                # 流程编排
├── ml_ranker/                     # ⭐ ML排序模块
│   ├── models/ltr_ranker/         # 训练好的ML模型
│   ├── data_loader.py             # 数据加载
│   ├── feature_engineer.py        # 特征工程
│   └── ltr_model.py               # LTR模型
├── configs/
│   ├── combo_wfo_config.yaml      # 完整WFO配置 (ML排序默认)
│   └── ranking_datasets.yaml      # ML模型训练配置
├── real_backtest/
│   └── run_profit_backtest.py     # 利润回测脚本
├── analysis/
│   └── compare_wfo_vs_ml.py       # WFO vs ML 对比脚本
├── docs/
│   └── ML_RANKING_INTEGRATION_GUIDE.md # ML排序使用指南
├── data/
│   └── etf_prices_template.csv    # 市场基准数据
├── results/                       # WFO运行结果
├── results_combo_wfo/             # 组合回测结果
└── logs/                          # 运行日志
```

**已归档目录**:
- `archive/historical_reports_20251114/` - 历史分析报告

---

## 🚀 快速开始

### 前置要求

```bash
# Python 3.11+
# 数据位置: /Users/zhangshenshen/深度量化0927/raw/ETF/daily (43只ETF parquet文件)
```

### 运行完整WFO

```bash
cd /Users/zhangshenshen/深度量化0927/etf_rotation_experiments

# 1. 运行WFO优化 (约30-60分钟)
python run_combo_wfo.py

# 输出位置（所有输出保持在项目内部）:
# - etf_rotation_experiments/results/run_YYYYMMDD_HHMMSS/
#   ├── all_combos.parquet          # 所有组合的WFO统计
#   ├── ranking_ic_top5000.parquet  # IC排序Top5000组合
#   └── wfo_summary.json            # 运行摘要
# - etf_rotation_experiments/logs/wfo_run_YYYYMMDD_HHMMSS.log  # 运行日志
```

### 回测Top100组合

```bash
# 2. 运行利润回测 (约10分钟)
python real_backtest/run_profit_backtest.py \
  --topk 100 \
  --ranking-file results/run_latest/ranking_ic_top5000.parquet \
  --slippage-bps 2 \
  --output-csv results/run_latest/top100_backtest.csv

# 输出位置（所有输出保持在项目内部）:
# - etf_rotation_experiments/results_combo_wfo/<timestamp>/
#   ├── top100_backtest.csv         # 回测结果汇总
#   └── 其他分析文件

# 查看结果
python -c "
import pandas as pd
df = pd.read_csv('results_combo_wfo/*/top100_backtest.csv')
print(f'年化收益: {df[\"annual_ret_net\"].mean():.2%}')
print(f'Sharpe比率: {df[\"sharpe_net\"].mean():.2f}')
print(f'最大回撤: {df[\"max_dd_net\"].mean():.2%}')
"
```

### 快速测试（验证环境）

```bash
# 使用最小配置验证核心流程（实测 < 5秒）
python run_combo_wfo.py -c configs/combo_wfo_config_minimal_test.yaml

# 验证输出:
# - 目录: results/run_YYYYMMDD_HHMMSS/
# - 关键文件: all_combos.parquet, top100_by_ic.parquet, wfo_summary.json
```

**✅ 最小测试已验证通过** (2024-11-14):
- 数据加载: 4只ETF × 429天 ✅
- 因子计算: 18个因子 × 4只ETF ✅  
- WFO优化: 153个2因子组合 × 17个滚动窗口 ✅
- 运行时长: < 5秒 ✅
- **详细验证报告**: 见 `CLEANUP_AND_VALIDATION_FINAL_REPORT.md`

---

## 🔬 核心流程

### 1. 数据加载
- 从 `raw/ETF/daily` 加载43只ETF的日线数据（2020-2025）
- 使用Tushare标准格式（trade_date, adj_close等）
- 自动缓存已处理数据到内存

### 2. 因子计算
- **动量类**: RET_5D, RET_20D, SLOPE_20D, ADX_14D
- **波动率类**: VOL_RATIO_20D, RET_VOL_20D, MAX_DD_60D
- **量价类**: CMF_20D, OBV_SLOPE_20D
- **综合类**: SHARPE_RATIO_20D, CORRELATION_TO_MARKET_20D

### 3. WFO优化
- **滚动窗口**: is_period=252天, oos_period=60天
- **组合生成**: 2-5因子随机组合
- **评估指标**: IC均值、IC_IR、稳定性得分
- **FDR控制**: Benjamini-Hochberg方法，alpha=0.05

### 4. 排序筛选
- 基于样本外IC均值排序
- 考虑IC稳定性（IC_IR）
- 输出Top5000组合

### 5. 回测验证
- 等权持仓，定期调仓
- 扣除滑点（2bps）和手续费（0.05%）
- 计算年化收益、Sharpe、最大回撤等指标

---

## 📊 成功标准

| 指标 | 目标值 | 备注 |
|------|--------|------|
| Top100年化收益 | >15% | 扣除交易成本 |
| Sharpe比率 | >1.2 | 净值曲线计算 |
| 最大回撤 | <15% | 净值最大回撤 |
| 样本外IC均值 | >0.03 | WFO所有窗口平均 |
| IC_IR | >0.5 | IC稳定性指标 |

---

## 📖 详细文档

- **[PROJECT_BASELINE.md](PROJECT_BASELINE.md)** - 完整技术文档（数据源、因子库、WFO配置、成功标准）
- **[CLEANUP_EXECUTION_REPORT_20251114.md](CLEANUP_EXECUTION_REPORT_20251114.md)** - 项目清理报告

---

## 🛠️ 配置说明

### WFO核心参数 (`configs/combo_wfo_config.yaml`)

```yaml
combo_wfo:
  combo_sizes: [2, 3, 4, 5]       # 因子组合大小
  is_period: 252                   # 样本内窗口（天）
  oos_period: 60                   # 样本外窗口（天）
  rebalance_frequencies: [8]       # 调仓频率（天）
  enable_fdr: true                 # 启用FDR控制
  fdr_alpha: 0.05                  # FDR显著性水平

data:
  start_date: "2020-01-01"         # 数据起始日期
  end_date: "2025-10-14"           # 数据结束日期
  symbols: [...]                   # 43只ETF代码列表
```

---

## ⚠️ 注意事项

1. **内存需求**: WFO过程需要~4GB内存
2. **运行时间**: 完整WFO约30-60分钟，取决于CPU核心数
3. **数据依赖**: 确保 `raw/ETF/daily` 目录存在且包含完整数据
4. **Python版本**: 需要Python 3.11+，依赖pandas、numpy、scipy等

---

## 📝 版本历史

- **2025-11-14**: 项目清理，移除无效ML实验，建立基准文档
- **2025-11-13**: 完成WFO核心流程优化
- **2025-11-12**: 初始版本

---

## 📧 联系方式

如有问题，请参考 `PROJECT_BASELINE.md` 或检查 `logs/` 目录的运行日志。

---

**最后更新**: 2025-11-14  
**状态**: ✅ 生产就绪
