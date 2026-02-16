# v6.0 Production Seal

**封板版本**: v6.0_20260212
**封板时间**: 2026-02-12
**状态**: PRODUCTION (C2 confirmed champion, ready for shadow deploy)
**代码基线**: foundation-repair branch (P0-1 WFO hysteresis fix applied)

---

## 策略定义

**Production Strategy**: C2 (3因子)
**因子**: `AMIHUD_ILLIQUIDITY + CALMAR_RATIO_60D + CORRELATION_TO_MARKET_20D`

| 因子 | 维度桶 | 含义 |
|------|--------|------|
| AMIHUD_ILLIQUIDITY | D (MICROSTRUCTURE) | 流动性溢价 — 低流动性ETF有更高风险补偿 |
| CALMAR_RATIO_60D | B (SUSTAINED_POSITION) | 60日Calmar比率 — 收益/回撤效率 |
| CORRELATION_TO_MARKET_20D | E (TREND_STRENGTH_RISK) | 市场相关性 — 低相关ETF提供分散化 |

**跨桶覆盖**: 3桶 (B+D+E), 通过 cross-bucket constraint (min_buckets=3)

## v5.0 → v6.0 变更

| 项目 | v5.0 | v6.0 | 原因 |
|------|------|------|------|
| 策略 | S1 (ADX+OBV+SHARPE+SLOPE) | **C2 (AMIHUD+CALMAR+CORR_MKT)** | S1结果是bounded_factors bug产物 |
| ETF池 | 43 (38A+5Q) | **49 (41A+8Q)** | 扩展覆盖 |
| bounded_factors | 4个 | **7个** (+ADX,CMF,CORR_MKT) | 修复3-way不一致 |
| WFO OOS评分 | IC-based (无hysteresis) | **execution_score (有IS warm-up)** | P0-1修复, rho从-0.28→+0.64 |
| active_factors | 18 OHLCV | **24 (18 OHLCV + 6 non-OHLCV)** | fund_share + margin |

## 执行框架

| 参数 | 值 |
|------|-----|
| FREQ | 5 |
| POS_SIZE | 2 |
| LOOKBACK | 252 |
| COMMISSION | 0.0002 (2bp) |
| 迟滞 | ON (dr=0.10, mh=9, max 1 swap) |
| Regime Gate | ON (vol mode, 510300, 25/30/40%) |
| 执行模型 | T1_OPEN |
| 成本模型 | SPLIT_MARKET med (a_share=0.0020, qdii=0.0050) |
| Universe | A_SHARE_ONLY (41 traded + 8 QDII monitored) |

## 绩效指标

### VEC 回测 (49-ETF, F5+Exp4, T1_OPEN, med cost)

| 策略 | 全期收益 | HO收益 | HO MDD | HO Sharpe | HO最差月 | 交易数 |
|------|---------|--------|--------|-----------|---------|--------|
| **C2** | +48.2% | **+71.9%** | **11.6%** | **+2.75** | -5.9% | 30 |
| 6F替代 | +59.5% | +66.9% | 10.8% | +2.69 | -6.1% | 27 |
| Holy Trinity | +55.5% | +44.4% | 11.9% | +2.20 | -6.6% | 45 |
| S1 (修复后) | +24.5% | +30.4% | 10.1% | +1.84 | -2.8% | 71 |

### BT Ground Truth (49-ETF, F5+Exp4, T1_OPEN, med cost, 0 margin failures)

| 策略 | HO收益 | HO MDD | HO Sharpe | HO最差月 | 交易数 |
|------|--------|--------|-----------|---------|--------|
| **C2** | **+71.6%** | **11.9%** | **+2.61** | -7.1% | 37 |
| S1 (修复后) | +3.8% | 23.5% | +0.29 | -10.2% | 155 |

### VEC-BT Gap

| 策略 | VEC HO | BT HO | Gap | 说明 |
|------|--------|-------|-----|------|
| C2 | +71.9% | +71.6% | **-0.3pp** | 极度对齐 (交易少→rounding小) |
| S1 | +30.4% | +3.8% | -26.6pp | S1在BT下进一步崩溃 |

## Alpha Decomposition (C2)

| 来源 | 贡献 | 占比 |
|------|------|------|
| 选股alpha | +49.1pp | 68.3% |
| Gate×选股交互 | +17.2pp | 23.9% |
| 纯择时beta | -0.3pp | ~0% |
| 总计 | +71.9pp | 100% |

C2 是纯选股策略, regime gate通过放大好选股的仓位间接贡献。

## 验证通过项

- [x] Gate 1: 滚动OOS 3/4窗口正收益, 4/4最差月达标
- [x] Gate 2: PnL集中度 Gini=0.448, Top-20%贡献=92%
- [x] Gate 3: BT Ground Truth 0 margin failures, HO +71.6%
- [x] Q1 正交性: S1-C2 Top-2 Jaccard=0.000
- [x] Q2 尾部: P(both<0|crash)=0.300, worst week overlap=1/5
- [x] HLZ-adjusted Sharpe: 2.31 (significant after N=20,160 haircut)
- [x] WFO-VEC Spearman rho=+0.636 (p=0.048) — hysteresis-aware WFO validated

## 已知风险

1. **C2池扩展+12.5pp是normalization artifact**: 0新ETF被选中, 仅6/39 rebalance点diverge
2. **30笔HO交易**: 统计显著性有限, 需100+交易 (~6个月shadow)
3. **残差相关0.497**: S1-C2 去掉A股beta后仍有中等正相关, 非完全独立
4. **Circuit breaker disabled**: thresholds=0.0, 依赖regime gate做风控

## Shadow Deploy

- **基础设施**: `generate_today_signal.py --shadow-config configs/shadow_strategies.yaml`
- **停止规则**: MDD>24%紧急停止, 5连亏审查, O'Brien-Fleming分段检验
- **目标**: 176笔交易 (~3.5年) for Sharpe>1.5 at p<0.05
- **6F监控**: AMIHUD+MARGIN_BUY+MARGIN_CHG+PP120+SHARE_ACCEL+SLOPE (Sharpe 2.69)

## 文件清单

```
v6.0_20260212/
├── SEAL_SUMMARY.md                    # 本文档
├── CHECKSUMS.sha256                   # 完整性校验
├── artifacts/
│   └── strategy_params.yaml           # C2 冻结参数
├── locked/
│   ├── configs/
│   │   └── combo_wfo_config.yaml      # 生产配置快照
│   └── src/
│       └── frozen_params.py           # 参数冻结模块快照
└── scripts/
    └── README.md                      # 复现步骤
```

---

**Sealed by**: Claude Code
**Seal Timestamp**: 2026-02-12
**Previous Version**: v5.0_20260211 (DEPRECATED — S1 results were bounded_factors bug artifacts)
