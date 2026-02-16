# ETF策略项目状态汇总

**最后更新**: 2026-02-16
**当前版本**: v8.0 (sealed 2026-02-15)
**生产策略**: composite_1 (5F) — ADX_14D + BREAKOUT_20D + MARGIN_BUY_RATIO + PRICE_POSITION_120D + SHARE_CHG_5D
**回退策略**: core_4f (4F) — MARGIN_CHG_10D + PRICE_POSITION_120D + SHARE_CHG_20D + SLOPE_20D

---

## 1. 项目概述

### 1.1 核心目标
构建ETF轮动策略研究平台，通过三层验证体系 (WFO → VEC → BT) 筛选因子组合，生成生产级策略信号。

### 1.2 三层验证体系
```
WFO (筛选层)  →  VEC (精度层)   →  BT (真实层)
~2分钟           ~5分钟            ~30-60分钟
IC gate+评分     Numba JIT内核     Backtrader事件驱动
无状态           浮点份额          整数手、资金约束
```

**关键原则**: 每一层必须在生产执行框架下运行 (FREQ=5 + Exp4 hysteresis + regime gate)，否则结果无效。

### 1.3 生产参数 (v8.0, sealed)
| 参数 | 值 | 说明 |
|-----|-----|-----|
| FREQ | 5 | 5个交易日换仓 |
| POS_SIZE | 2 | 同时持有2只ETF |
| COMMISSION | 0.0002 | 2bp基础手续费 |
| delta_rank | 0.10 | Hysteresis: rank01差≥0.10才换仓 |
| min_hold_days | 9 | Hysteresis: 最少持有9天 |
| 因子池 | 23 (17 OHLCV + 6 non-OHLCV) | 活跃因子 |
| 标的池 | 49只 | 41只A股ETF + 8只QDII(仅监控) |
| Regime Gate | ON | 510300波动率门控 (25/30/40%) |
| 成本模型 | SPLIT_MARKET | A股20bp, QDII 50bp |

---

## 2. 当前策略表现

### 2.1 composite_1 生产策略 (v8.0, BT ground truth)

**因子组合**: ADX_14D(+1) + BREAKOUT_20D(+1) + MARGIN_BUY_RATIO(-1) + PRICE_POSITION_120D(+1) + SHARE_CHG_5D(-1)

| 指标 | Train | Holdout | 全期 | 说明 |
|------|-------|---------|------|------|
| 收益率 | +51.6% | **+53.9%** | +105.5% | 2020-01~2026-02 |
| 最大回撤 | 10.8% | 10.8% | 10.8% | |
| Sharpe | 1.46 | **1.38** | — | 风险调整最优 |
| Calmar | 4.78 | **7.41** | — | |
| 交易次数 | — | 77 | — | F5+Exp4控制换手 |
| Margin Failures | 0 | 0 | — | BT执行可行 |
| Rolling 正率 | — | 61% (11/18) | — | |

**v8.0 核心特征**:
- 5因子组合，2个负向因子 (MARGIN_BUY_RATIO, SHARE_CHG_5D)
- 全候选中 **最高 Sharpe (1.38)** 和 **最高 HO Calmar (7.41)**
- 低 MDD (10.8%)，风险调整后表现最优

### 2.2 core_4f 回退策略 (v8.0, BT ground truth)

**因子组合**: MARGIN_CHG_10D(-1) + PRICE_POSITION_120D(+1) + SHARE_CHG_20D(-1) + SLOPE_20D(+1)

| 指标 | Train | Holdout | 全期 |
|------|-------|---------|------|
| 收益率 | +53.0% | **+67.4%** | +120.4% |
| 最大回撤 | 14.9% | 14.9% | 14.9% |
| Sharpe | 1.12 | 1.09 | — |
| Calmar | 3.56 | 4.56 | — |
| 交易次数 | — | 75 | — |
| Rolling 正率 | — | **78% (14/18)** | — |

**特征**: 纯 4 因子组合，**绝对收益最高** (+67.4%)，Rolling 稳定性最强 (78%)

### 2.3 VEC-BT 对齐验证 (v8.0)

| 策略 | VEC HO | BT HO | Gap | 状态 |
|------|--------|-------|-----|------|
| composite_1 | +55.7% | +53.9% | **-1.9pp** | PASS (<5pp) |
| core_4f | +68.0% | +67.4% | **-0.6pp** | PASS (<5pp) |

**管线健康**: VEC-BT train gap mean 0.07pp, median 0.00pp — 完美对齐

### 2.4 实盘表现 (2025-12-18 ~ 2026-02-09, 6周)

- 收益: **+6.37%** (49,178 CNY)
- 交易: 22笔, 胜率 83.3%, 盈亏比 2.33
- 持仓: 100% A股, 零QDII (市场环境导致, 非bug)

---

## 3. 研究历史与关键发现

### 3.1 核心认知: 执行设计 > 信号质量

| 优化维度 | 回报乘数 | 证据 |
|----------|---------|------|
| **执行优化** | **3.6x** | 同一信号, F3_OFF→F5_ON, HO +11.8%→+42.7% |
| **信号改进** | 1.25x | 不同信号, 相同执行框架 |
| **因子重组** | ~1x | 代数因子组合, 信息空间已饱和 |

**原理**: A股ETF宇宙PC1=59.8%, Kaiser有效维度=5/17。大部分因子冗余，执行框架决定哪些因子能存活。

### 3.2 已完成研究 (按时间倒序)

#### Phase 1: Non-OHLCV Factor Optimization (2026-02-16) — EXHAUSTED

- **方法**: 200 combos × 4-gate 验证，vs v8.0 baseline
- **Exp-A1**: SHARE_CHG_10D vs SHARE_CHG_5D → **REJECTED** (-67.4pp gap)
- **Exp-A2**: SHARE_ACCEL 集成 → **PARTIAL** (稳定性换收益，无风险调整提升)
- **Exp-A3**: 全网格搜索 → **v8.0 confirmed optimal**
- **结论**: 当前 23 因子空间已饱和，突破需新数据源 (Phase 2: IOPV, FX, northbound)
- **文档**: `docs/research/phase1_non_ohlcv_optimization_20260216.md`

#### v8.0 三大管线修复 (2026-02-14~15) — COMPLETED

1. **IC-sign factor direction fix**: 5/6 非OHLCV 因子被系统性反向使用 → stability-gated sign flip
2. **VEC metadata propagation**: factor_signs/factor_icirs 丢失 → VEC-BT gap 20.3pp→1.47pp
3. **BT execution-side hysteresis**: 信号态反馈环 → 执行态驱动，gap +25.7pp→-0.8pp

**教训**: 修 bug 不要"修一个封一版"，等全部修完再封板 (Rule 26)

#### 代数因子VEC验证 — MARGINAL (2026-02-12)

- **方法**: GP挖掘78个代数因子 → WFO 1.2M组合 → 27 VEC候选 → 6 BT候选
- **结论**: 代数组合是OHLCV重排, 信息空间已被17因子覆盖
- **文档**: `docs/research/algebraic_factor_vec_validation.md`

#### 行业约束研究 — NEGATIVE (2026-02-12)

- **结论**: 同行业双持是最优配置; MDD反而恶化
- **文档**: `docs/research/sector_constraint_negative_results.md`

#### 条件因子研究 — NEGATIVE (2026-02-11)

- **5个假设全部推翻**: +15pp是路径依赖复利artifact
- **文档**: `docs/research/conditional_factor_negative_results.md`

#### 跨桶约束 — POSITIVE (2026-02-11)

- **验证**: HO中位数 +4.9pp
- **文档**: `docs/research/bucket_constraints_ablation.md`

### 3.3 待研究方向

#### Phase 2: 新信息源因子开发

| 实验 | 数据源 | 因子候选 | 状态 |
|------|--------|---------|------|
| B4 汇率 | AkShare BOC FX | USD_CNY_MOM_5D, FX_CARRY | **数据已有** |
| B2 北向资金 | Tushare moneyflow_hsgt | NORTHBOUND_NET_5D | 需确认映射 |
| B1 IOPV折溢价 | QMT/Wind实时 | IOPV_PREMIUM_5D | 数据管道待建 |
| B3 期权IV | Tushare opt_daily | IV_RANK_20D | 覆盖面窄 |

---

## 4. 系统架构

### 4.1 关键文件
```
configs/combo_wfo_config.yaml        # 单一配置源 (ETF池, 因子, 引擎参数)
src/etf_strategy/core/frozen_params.py  # v8.0 版本锁定参数
src/etf_strategy/core/hysteresis.py     # @njit hysteresis内核
src/etf_strategy/run_combo_wfo.py       # WFO筛选主流程
scripts/batch_vec_backtest.py           # VEC批量回测
scripts/batch_bt_backtest.py            # BT ground truth
scripts/generate_today_signal.py        # 每日信号生成 (有状态)
sealed_strategies/v8.0_20260215/        # v8.0 封存策略快照
```

### 4.2 信号评估原则 (v5.0+)
**任何新信号/因子必须在生产执行框架下评估 (FREQ=5 + Exp4 + regime gate), 否则不是有效候选。**

### 4.3 Alpha维度分析
- **Return PCA**: PC1中位数 59.8% → 强单因子主导
- **因子空间Kaiser维度**: 5/17 → 大部分因子冗余
- **含义**: OHLCV衍生因子的信息空间已近饱和

---

## 5. 版本历史

| 版本 | 日期 | 策略 | 说明 |
|------|------|------|------|
| **v8.0 sealed** | 2026-02-15 | composite_1 (5F) + core_4f (4F) | 管线修复后首个 clean seal |
| v7.0 | 2026-02-13 | — | 废弃: IC-sign/metadata/exec-side bugs |
| v6.0 | 2026-02-12 | — | 从未使用: train gate fail |
| v5.0 | 2026-02-11 | S1 (4F) | 废弃: ADX Winsorize bug |
| v3.4 | 2025-12-16 | S1 (4F) | FREQ=3 基线 |

---

## 6. 快速启动

### 6.1 运行管线
```bash
make wfo                             # WFO筛选 (~2min)
make vec                             # VEC回测 (~5min)
make bt                              # BT审计 (~30-60min)
make pipeline                        # 全流程 WFO→VEC→BT
uv run python scripts/generate_today_signal.py  # 每日信号
```

### 6.2 查看封存策略
```bash
ls sealed_strategies/v8.0_20260215/
cat sealed_strategies/v8.0_20260215/SEAL_SUMMARY.md
```

### 6.3 接手checklist
- [ ] 已阅读 `CLAUDE.md` 了解项目规范
- [ ] 已阅读本文件了解当前状态
- [ ] 已阅读 `memory/rules.md` 了解 26 条硬性规则
- [ ] 已理解三层验证体系和信号评估原则
- [ ] 已理解 v8.0 生产参数 (FREQ=5, POS_SIZE=2, Exp4 hysteresis)

---

## 7. 关键教训

### 7.1 执行 > 信号
```
同一信号, 执行优化 → 3.6x回报乘数
不同信号, 同一执行 → 1.25x回报乘数
更多因子组合       → ~1x (信息饱和)
```

### 7.2 管线一致性 (Rule 24)
```
VEC必须传递 hysteresis 参数 (delta_rank, min_hold_days)
VEC必须透传 factor_signs / factor_icirs 元数据
BT必须使用执行态 (shadow_holdings) 而非信号态驱动 hysteresis
```

### 7.3 因子-执行兼容性
```
稳定rank因子 (ADX, SLOPE, CMF) → Exp4过滤噪声, 表现好
不稳定rank因子 (PV_CORR, PP_20D) → Exp4锁死错误仓位, 崩溃
信号质量 ≠ 生产表现; rank稳定性才是关键
```

---

**文档维护**: 每次重大研究后更新
**封存策略**: `sealed_strategies/v8.0_20260215/`
**研究文档**: `docs/research/`
**经验教训**: `memory/rules.md` (26条)
