# ETF轮动优化系统 🎯

> 重要：请先阅读 `docs/PROJECT_OVERVIEW.md` 与 `docs/LLM_GUARDRAILS.md`
>
> - 规范入口：`real_backtest/`（同名脚本以该目录为准）
> - WFO 主入口：`run_combo_wfo.py`（根目录唯一副本）
> - 输出契约：见 `docs/OUTPUT_SCHEMA.md`，修改前务必同步更新文档与读取脚本

## 🧩 启用「Markdown 新建限制」

为避免大模型随意新建无意义的 `.md` 文件，本仓库提供 pre-commit 钩子进行约束：

- 仅允许在 `docs/` 目录新建 `.md`；
- 新建文档需在前 20 行加入允许标记（任选其一）：`<!-- ALLOW-MD -->` / `[ALLOW-MD]` / `ALLOW_MD: true`；
- 已存在的 `.md` 不受影响（可自由修改）。

安装钩子：

```bash
bash scripts/install_git_hooks.sh
```

说明：具体策略见 `docs/LLM_GUARDRAILS.md`。

一个高性能、生产就绪的ETF轮动策略系统，基于因子投资和Walk Forward Optimization (WFO) 框架。系统支持从单因子分析到组合级深度挖掘的全流程，专为量化交易和个人投资者设计。

## ✨ 核心特性

### 🚀 性能优势
- **Numba JIT编译**：核心计算函数加速10-100倍
- **向量化处理**：避免Python循环，充分利用NumPy优化
- **并行计算**：多核CPU并行处理大规模组合测试
- **内存优化**：智能缓存和数据流管理

### 📊 策略能力
- **18个精选因子**：动量、波动率、相对强度、技术指标全覆盖
- **组合级WFO**：支持2-5因子的深度协同效应挖掘
- **多频率测试**：自动测试5-30天换仓频率的最优选择
- **FDR控制**：假发现率校正，确保统计显著性

### 🛡️ 稳健性保证
- **滑动窗口验证**：避免过拟合，确保策略泛化能力
- **严格数据处理**：Z-score标准化、Winsorize极值处理
- **未来函数防护**：彻底排查前瞻偏差，确保结果可信

## 🎯 快速开始

### 基础运行（单因子WFO）
```bash
# 运行基础WFO流程
python run_final_production.py

# 查看结果
cat results/wfo/wfo_summary.csv
```

### 深度挖掘（组合级WFO）
```bash
# 运行组合级深度优化（推荐）
python run_combo_wfo.py

# 查看Top组合
head -5 results_combo_wfo/top_combos.csv
```

### 无未来函数回测验证
```bash
# 严格回测验证（无前瞻偏差）
python test_freq_no_lookahead.py

# 查看回测报告
open results_combo_wfo/未来函数排查与修复完整报告.md
```

### 快速测试
```bash
# 5分钟快速验证（只跑2-3因子组合）
python run_combo_wfo.py --quick

# 查看完整报告
open results_combo_wfo/REPORT.md
```

## 📈 核心结果

### 🏆 最优组合发现
**Top 1**: `RELATIVE_STRENGTH_VS_MARKET_20D + RET_VOL_20D + SLOPE_20D + VOL_RATIO_20D`
- **OOS IC**: 0.199 ± 0.137
- **信息比率(IR)**: 1,743.79 (超高!)
- **稳定性得分**: 523.33
- **最优换仓**: 30天
- **FDR q值**: 0.0000 (高度显著)

### 📊 系统性能
- **测试规模**: 12,597个组合 × 19个WFO窗口 × 6个频率 = 143万次测试
- **运行时间**: 仅3.8分钟（10核并行）
- **显著率**: 99.1%通过FDR检验(α=0.05)
- **数据覆盖**: 43只主流ETF × 1399个交易日

### 🎯 实盘验证结果
**最优配置：4因子组合 + 6天换仓**
- **年化收益**: 12.9%
- **Sharpe比率**: 0.486
- **最大回撤**: -33.6%
- **组合**: CORRELATION_TO_MARKET_20D + OBV_SLOPE_10D + PV_CORR_20D + RELATIVE_STRENGTH_VS_MARKET_20D + VOL_RATIO_20D

## 🏗️ 技术架构

### 核心模块
```
etf_rotation_optimized/
├── core/                           # 核心算法引擎
│   ├── data_loader.py               # 高性能数据加载器
│   ├── precise_factor_library_v2.py  # 精选因子库(18个)
│   ├── cross_section_processor.py   # 横截面标准化处理
│   ├── ic_calculator_numba.py       # JIT编译的IC计算器
│   ├── direct_factor_wfo_optimizer.py # 单因子WFO优化器
│   ├── combo_wfo_optimizer.py      # 组合级WFO优化器
│   └── pipeline.py                  # 统一流程编排
├── configs/                        # 配置管理
│   ├── default.yaml                # 基础配置模板
│   └── combo_wfo_config.yaml      # 组合WFO配置
├── results*/                       # 结果输出
│   ├── wfo/                       # 单因子结果
│   └── results_combo_wfo/           # 组合级结果
├── run_combo_wfo.py               # WFO优化主入口
├── run_final_production.py        # 生产运行脚本
├── test_freq_no_lookahead.py      # 无未来函数回测
└── docs/                          # 技术文档
```

### 数据流处理
```mermaid
graph LR
    A[原始数据] --> B[数据加载器]
    B --> C[因子计算引擎]
    C --> D[横截面处理器]
    D --> E[WFO优化器]
    E --> F[组合挖掘器]
    F --> G[结果分析器]

    style A fill:#f9f,stroke:#333
    style G fill:#9f9,stroke:#333
```

## 🔧 配置指南

### 单因子WFO配置 (`configs/default.yaml`)
```yaml
data:
  symbols: [...]  # 43只主流ETF
  start_date: "2020-01-01"
  end_date: "2025-10-14"

cross_section:
  winsorize_lower: 0.025
  winsorize_upper: 0.975

wfo:
  is_period: 252    # 1年训练期
  oos_period: 60    # 60天测试期
  step_size: 60     # 60天步长
```

### 组合WFO配置 (`configs/combo_wfo_config.yaml`)
```yaml
combo_wfo:
  combo_sizes: [2, 3, 4, 5]           # 2-5因子组合
  rebalance_frequencies: [5, 10, 15, 20, 25, 30]  # 多频率测试
  enable_fdr: true                      # FDR控制
  fdr_alpha: 0.05                        # 5%显著性水平
  complexity_penalty_lambda: 0.15         # 复杂度惩罚
```

## 📚 使用场景

### 🎯 个人量化投资者
- **快速验证策略想法**: 5分钟获得统计显著性结果
- **深度因子挖掘**: 发现4-5因子的高效协同组合
- **低频交易**: 30天换仓，适合个人资金管理

### 🏢 机构研究团队
- **因子库扩展**: 在18个精选因子基础上添加自定义因子
- **风险控制集成**: 接入组合优化器和风险模型
- **多资产类别**: 扩展到股票、期货等其他资产

### 🎓 学术研究
- **因子有效性验证**: 严格的WFO流程和FDR控制
- **组合优化理论**: 多因子协同效应的实证研究
- **机器学习**: 为AI策略提供高质量特征工程

## 🔍 核心算法

### 1. Walk Forward Optimization
```python
# 滑动窗口训练验证
for window in sliding_windows:
    # In-Sample训练
    factors_selected = select_by_ic(window.train_data)
    # Out-of-Sample测试
    oos_ic = evaluate(window.test_data, factors_selected)
    # 累积统计
    update_performance(oos_ic)
```

### 2. 组合级挖掘
```python
# 组合枚举与评估
for combo_size in [2, 3, 4, 5]:
    for combo in combinations(18_factors, combo_size):
        for freq in [5, 10, 15, 20, 25, 30]:
            score = evaluate_combo(combo, freq)
            # FDR校正
            corrected_p = fdr_correction(score)
```

### 3. 因子库 (18个精选)
| 类别 | 因子 | 说明 |
|------|------|------|
| 动量 | MOM_20D, RELATIVE_STRENGTH_VS_MARKET_20D | 20日动量、相对强度 |
| 波动率 | RET_VOL_20D, VOL_RATIO_20D/60D | 收益波动、成交量比率 |
| 趋势 | SLOPE_20D, PRICE_POSITION_20D/120D | 价格斜率、位置指标 |
| 技术 | RSI_14, ADX_14D, VORTEX_14D | RSI、ADX、涡度指标 |
| 质量 | SHARPE_RATIO_20D, CALMAR_RATIO_60D | 夏普比率、卡玛比率 |
| 资金流 | CMF_20D, OBV_SLOPE_10D | 资金流指标 |

## 🚀 性能优化

### 计算性能
- **Numba JIT加速**: IC计算提升100倍
- **向量化操作**: 避免Python循环，利用NumPy优化
- **并行计算**: 10核CPU并行处理12K组合
- **内存管理**: 智能缓存，避免重复计算

### 统计严谨性
- **FDR多重检验**: Benjamini-Hochberg校正
- **滑动窗口**: 避免过拟合，确保泛化能力
- **前瞻偏差防护**: 严格的未来函数排查
- **稳定性评分**: 考虑IC、IR、胜率、复杂度

## 📊 结果解读

### 关键指标
- **IC (Information Coefficient)**: 因子与未来收益的相关系数
- **IR (Information Ratio)**: IC均值/IC标准差，衡量风险调整后表现
- **胜率**: 正IC占比，评估方向准确性
- **稳定性得分**: 综合评分(0.5*IC + 0.3*IR + 0.2*胜率 - 复杂度惩罚)

### 优化建议
1. **换仓频率**: 10天(高收益) vs 30天(低成本)
2. **因子数量**: 4-5因子(协同效应) vs 2-3因子(简单性)
3. **风险控制**: 结合波动率和最大回撤指标
4. **市场适应**: 牛熊市可能需要不同因子组合

## 📁 结果目录说明

- `results_combo_wfo/` - WFO优化结果
  - `REPORT.md` - 完整分析报告和Top 10组合
  - `top_combos.csv` - Top 50组合详细数据
  - `all_combos.csv` - 所有12,597个组合结果
  - `freq_test_no_lookahead.csv` - 无未来函数回测结果
  - `未来函数排查与修复完整报告.md` - 回测验证报告

- `results/wfo/` - 单因子WFO结果
  - `wfo_summary.csv` - 每个窗口的统计摘要
  - `window_results.json` - 详细窗口结果

## 🛠️ 高级用法

### 自定义因子
```python
# 在precise_factor_library_v2.py中添加
@njit
def custom_factor(close, volume, window=20):
    # 实现自定义因子逻辑
    return factor_values
```

### 风险集成
```yaml
# 添加风险约束
risk_management:
  max_position_size: 0.05    # 单个ETF最大5%
  max_drawdown: 0.15         # 最大回撤15%
  turnover_limit: 0.3         # 年换手率30%
```

### 实盘部署
```bash
# 1. 回测验证
python run_combo_wfo.py

# 2. 策略评估
python test_freq_no_lookahead.py

# 3. 生产运行
python run_final_production.py
```

## 📈 路线图

### 2024年发展历程
- **Q1**: 系统架构重构，移除历史遗留代码
- **Q2**: Numba JIT优化，性能提升10倍
- **Q3**: 组合级WFO实现，支持深度挖掘
- **Q4**: FDR控制集成，统计严谨性提升

### 未来规划
- [ ] 多周期扩展(周频、月频)
- [ ] 跨资产类别支持(股票、期货)
- [ ] 机器学习特征工程
- [ ] 实盘交易接口集成

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进系统：

1. **Bug报告**: 提供复现步骤和期望结果
2. **功能建议**: 详细描述使用场景
3. **性能优化**: 提供基准测试结果
4. **文档改进**: 帮助其他用户理解系统

## 📄 许可证

MIT License

---

<div align="center">

**ETF轮动优化系统**
*让量化交易更简单、更高效、更可靠*

⭐ 如果这个项目对您有帮助，请考虑给我们一个Star！

</div>
