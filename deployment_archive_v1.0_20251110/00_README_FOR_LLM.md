# ETF轮动策略归档 v1.0 - LLM阅读指南

**生成时间**: 2025-11-10  
**策略版本**: v1.0_wfo_20251109  
**用途**: 为大模型提供完整、可复现的策略文档

---

## 快速导航

本归档文件夹包含从0到1的完整策略开发流程，按以下顺序阅读：

### 第一步：理解全流程（必读）
📄 **01_PIPELINE_AUDIT_REPORT.md** (重要性: ⭐⭐⭐⭐⭐)
- 完整审核报告，涵盖WFO→回测→筛选→配置→部署
- 关键结论：12,597组合→5个生产策略，全部验证通过
- 阅读时间：15分钟

### 第二步：查看策略配置（核心）
📄 **configs/strategy_config_v1.json** (重要性: ⭐⭐⭐⭐⭐)
- 5个生产策略的完整配置
- 包含因子列表、权重、风控参数
- 阅读时间：5分钟

📄 **configs/allocation_config_v1.json** (重要性: ⭐⭐⭐⭐)
- 权重分配：[25%, 25%, 20%, 15%, 15%]
- 5x5相关性矩阵
- 阅读时间：3分钟

### 第三步：深入WFO过程（可选）
📄 **configs/combo_wfo_config.yaml** (重要性: ⭐⭐⭐⭐)
- WFO参数：IS=252天, OOS=60天, step=60天
- 组合规模：[2,3,4,5], 调仓频率：8天
- 阅读时间：3分钟

📄 **docs/WFO_ANALYSIS_REPORT_20251109.md** (重要性: ⭐⭐⭐)
- WFO结果详细分析
- Top30因子频率统计
- Rank-Sharpe相关性分析
- 阅读时间：10分钟

### 第四步：理解代码逻辑（高级）
📄 **code/run_combo_wfo.py** (357行, 重要性: ⭐⭐⭐⭐)
- WFO主脚本，负责加载数据、计算因子、执行WFO
- 关键函数：`main()`, 数据加载、因子计算、WFO执行
- 阅读时间：20分钟

📄 **code/run_production_backtest.py** (2192行, 重要性: ⭐⭐⭐⭐⭐)
- 无未来函数回测引擎，严格时间隔离
- 关键机制：close-to-close收益率，日级IC预计算
- 性能：241ms/策略，IC占98.7%
- 阅读时间：30分钟

### 第五步：验证数据一致性（审计）
📄 **wfo_audit_summary.json** (重要性: ⭐⭐⭐)
- WFO运行的统计数据
- 绩效分布、异常值检查
- 阅读时间：5分钟

📄 **strategy_validation_report.json** (重要性: ⭐⭐⭐⭐⭐)
- 5个策略配置与WFO结果的交叉验证
- 验证结果：5/5通过（100%匹配）
- 阅读时间：3分钟

📄 **data/strategy_candidates_selected.csv** (重要性: ⭐⭐⭐)
- 6个候选策略（最终部署5个）
- 包含rank、因子、绩效指标
- 阅读时间：3分钟

---

## 文件结构总览

```
deployment_archive_v1.0_20251110/
├── 00_README_FOR_LLM.md (本文档)
├── 01_PIPELINE_AUDIT_REPORT.md (全流程审核报告)
├── 02_CODE_SNAPSHOT_INDEX.md (代码快照索引)
├── wfo_audit_summary.json (WFO审核数据)
├── strategy_validation_report.json (策略验证报告)
├── code/ (核心代码)
│   ├── run_combo_wfo.py (WFO主脚本, 357行)
│   └── run_production_backtest.py (回测引擎, 2192行)
├── configs/ (配置文件)
│   ├── combo_wfo_config.yaml (WFO参数)
│   ├── strategy_config_v1.json (5个策略配置)
│   └── allocation_config_v1.json (权重+相关性)
├── data/ (数据快照)
│   └── strategy_candidates_selected.csv (6个候选策略)
└── docs/ (分析文档)
    └── WFO_ANALYSIS_REPORT_20251109.md (WFO分析报告)
```

---

## 关键指标速查

### 策略组合绩效
| 指标 | 预期值 | 数据来源 |
|------|--------|----------|
| 组合Sharpe | 1.096 | allocation_config_v1.json |
| 组合年化收益 | 21.83% | allocation_config_v1.json |
| 组合MaxDD | -22.01% | allocation_config_v1.json |
| 策略数量 | 5 | strategy_config_v1.json |
| 调仓频率 | 8天 | combo_wfo_config.yaml |
| 持仓数 | 5 | strategy_config_v1.json |

### WFO统计
| 指标 | 值 | 数据来源 |
|------|-----|----------|
| 总组合数 | 12,597 | wfo_audit_summary.json |
| IS窗口 | 252天 | combo_wfo_config.yaml |
| OOS窗口 | 60天 | combo_wfo_config.yaml |
| 滚动步长 | 60天 | combo_wfo_config.yaml |
| FDR方法 | BH, α=0.05 | combo_wfo_config.yaml |
| Sharpe中位数 | 0.555 | wfo_audit_summary.json |
| MaxDD中位数 | -26.8% | wfo_audit_summary.json |

### 5个生产策略
| 策略ID | Rank | Sharpe | MaxDD | 权重 | 核心因子 |
|--------|------|--------|-------|------|----------|
| strat_001 | 1843 | 1.121 | -17.4% | 25% | RSI_14, MAX_DD_60D, SLOPE_20D |
| strat_002 | 693 | 1.132 | -21.0% | 25% | ADX_14D, CMF_20D, RSI_14 |
| strat_003 | 2772 | 1.063 | -23.9% | 20% | VOL_RATIO_60D, PRICE_POSITION_20D |
| strat_004 | 3189 | 1.102 | -26.9% | 15% | CMF_20D, OBV_SLOPE_10D |
| strat_005 | 3006 | 1.030 | -24.0% | 15% | ADX_14D, PRICE_POSITION_20D |

---

## 验证检查清单

如果你是LLM，需要验证策略配置的正确性，请按以下顺序检查：

### ✅ 第1步：配置一致性
- [ ] 读取`strategy_validation_report.json`
- [ ] 确认`strategies_passed`字段 = 5（全部通过）
- [ ] 检查每个策略的`validation`字段 = "PASS"

### ✅ 第2步：绩效合理性
- [ ] 读取`strategy_config_v1.json`
- [ ] 确认5个策略的Sharpe全部 > 1.0
- [ ] 确认5个策略的MaxDD全部 > -27%（除strat_004为-26.9%）

### ✅ 第3步：权重总和
- [ ] 读取`allocation_config_v1.json`
- [ ] 计算`weights`数组总和 = 1.0 (100%)
- [ ] 确认`correlation_matrix`为5x5对称矩阵

### ✅ 第4步：WFO数据完整性
- [ ] 读取`wfo_audit_summary.json`
- [ ] 确认`total_combos` = 12597
- [ ] 确认`config.IS_period` = 252
- [ ] 确认`config.OOS_period` = 60

### ✅ 第5步：代码无未来函数
- [ ] 读取`code/run_production_backtest.py` 第1-120行
- [ ] 确认注释中包含"无未来函数保障"
- [ ] 确认代码中使用`factors[day_idx-1]`获取信号
- [ ] 确认代码中使用`returns[day_idx]`计算收益

---

## 常见问题（FAQ for LLM）

### Q1: 如何快速理解策略逻辑？
**A**: 阅读顺序：`strategy_config_v1.json`（5分钟）→ `01_PIPELINE_AUDIT_REPORT.md`第3节（策略筛选）→ `allocation_config_v1.json`（权重分配）

### Q2: 如何验证策略是否过拟合？
**A**: 检查`01_PIPELINE_AUDIT_REPORT.md`第1.4节（绩效分布），确认：
- Sharpe中位数0.555（温和）
- Top10 Sharpe分布0.96-1.13（集中，非离群值）
- Sharpe<0仅0.2%（异常少）

### Q3: 如何确认无未来函数？
**A**: 检查`01_PIPELINE_AUDIT_REPORT.md`第2.1节（无未来函数保障），验证：
- close-to-close收益率（`close[t]/close[t-1]-1`）
- NAV差异=0（重算验证）
- 环境变量`RB_ENFORCE_NO_LOOKAHEAD=1`自检

### Q4: 如何理解5个策略的区别？
**A**: 查看`strategy_config_v1.json`中的`factors`字段：
- strat_001/002：RSI族策略（RSI_14核心）
- strat_003：VOL族策略（VOL_RATIO_60D核心）
- strat_004/005：ADX/CMF族策略（趋势+资金流）

### Q5: WFO的12,597是如何计算的？
**A**: 查看`combo_wfo_config.yaml`：
- 组合规模：[2,3,4,5]
- 18个因子：C(18,2) + C(18,3) + C(18,4) + C(18,5) = 153+816+3060+8568 = 12,597
- 验证：`wfo_audit_summary.json`中`total_combos` = 12597 ✅

### Q6: 策略何时需要调整？
**A**: 查看`strategy_config_v1.json`中的`risk_control`：
- 策略级：60天MaxDD<-30% → 权重减半
- 策略级：60天Sharpe<0.3 → 权重减半
- 组合级：总MaxDD<-28% → 全部减仓50%
- 紧急停止：连续10天亏损 → 暂停交易

---

## 时间线（关键节点）

| 日期 | 事件 | 产出 |
|------|------|------|
| 2025-11-09 03:25 | WFO运行启动 | - |
| 2025-11-10 00:13 | WFO运行完成 | 12,597组合结果 |
| 2025-11-10 02:00 | 策略筛选 | 6候选→5部署 |
| 2025-11-10 02:30 | 配置生成 | strategy_config_v1.json |
| 2025-11-10 03:00 | 深度审核 | 本归档文件夹 |
| 2025-11-11 | 数据接入测试 | ⏳ 待执行 |
| 2025-11-11→11-17 | 模拟盘运行 | ⏳ 7天测试 |
| 2025-11-18 | 实盘启动 | ⏳ 10万元初始资金 |

---

## 版本历史

| 版本 | 日期 | 变更内容 |
|------|------|----------|
| v1.0_wfo_20251109 | 2025-11-10 | 初始版本，基于20251109 WFO运行 |

---

## 联系与支持

- **策略版本**: v1.0_wfo_20251109
- **下次复审**: 2025-12-10（模拟盘30天后）
- **归档位置**: `deployment_archive_v1.0_20251110/`

---

**最后更新**: 2025-11-10  
**文档状态**: ✅ 完整，可复现  
**LLM可读性**: ⭐⭐⭐⭐⭐ (优化完成)
