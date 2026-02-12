# ETF策略项目状态汇总

**最后更新**: 2026-02-12
**当前版本**: v5.0 (sealed 2026-02-11)
**生产策略**: S1 (ADX_14D + OBV_SLOPE_10D + SHARPE_RATIO_20D + SLOPE_20D)
**影子策略**: C2 (AMIHUD + CALMAR_60D + CORR_MKT_20D) — BT验证通过，待shadow

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

### 1.3 生产参数 (v5.0, sealed)
| 参数 | 值 | 说明 |
|-----|-----|------|
| FREQ | 5 | 5个交易日换仓 |
| POS_SIZE | 2 | 同时持有2只ETF |
| COMMISSION | 0.0002 | 2bp基础手续费 |
| delta_rank | 0.10 | Hysteresis: rank01差≥0.10才换仓 |
| min_hold_days | 9 | Hysteresis: 最少持有9天 |
| 因子池 | 17 (base) + 1 (PREMIUM_DEVIATION_20D) | 活跃因子 |
| 标的池 | 43只 | 38只A股ETF + 5只QDII(仅监控) |
| Regime Gate | ON | 510300波动率门控 (25/30/40%) |
| 成本模型 | SPLIT_MARKET | A股20bp, QDII 50bp |

---

## 2. 当前策略表现

### 2.1 S1 生产策略 (v5.0 VEC验证)
| 指标 | 全期 | 训练期 | 样本外 | 说明 |
|------|------|--------|--------|------|
| 收益率 | +54.1% | +8.0% | **+42.7%** | 2020-01~2026-02 |
| 最大回撤 | — | — | **11.8%** | |
| 最差月份 | — | — | **-2.9%** | |
| 交易次数 | — | — | 75 | F5+Exp4控制换手 |
| 季度正收益率 | — | — | 75% | |

### 2.2 C2 影子候选 (BT ground truth验证)
| 指标 | VEC | BT | 说明 |
|------|-----|-----|------|
| 样本外收益 | +63.9% | +45.9% | VEC-BT gap: -18.1pp |
| 最大回撤 | 10.4% | 12.2% | |
| 最差月份 | -9.4% | -7.1% | |
| 交易次数 | 28 | 34 | |
| Margin failures | — | **0** | |
| vs BT-S1 | — | **+13.4pp** | BT下仍优于S1 |

### 2.3 实盘表现 (2025-12-18 ~ 2026-02-09, 6周)
- 收益: **+6.37%** (49,178 CNY)
- 交易: 22笔, 胜率 83.3%, 盈亏比 2.33
- 持仓: 100% A股, 零QDII (市场环境导致, 非bug)

---

## 3. 研究历史与关键发现

### 3.1 核心认知: 执行设计 > 信号质量

| 优化维度 | 回报乘数 | 证据 |
|----------|---------|------|
| **执行优化** | **3.6x** | S1 F5_OFF→F5_ON, 信号不变, HO +11.8%→+42.7% |
| **信号改进** | 1.25x | C2 vs S1, 相同执行框架 |
| **因子重组** | ~1x | 代数因子组合, 信息空间已饱和 |

**原理**: A股ETF宇宙PC1=59.8%, Kaiser有效维度=5/17。大部分因子冗余，执行框架决定哪些因子能存活。

### 3.2 已完成研究 (按时间倒序)

#### 代数因子VEC验证 — 边际价值 (2026-02-12)
- **方法**: GP挖掘78个代数因子 → WFO 1.2M组合 → 27 VEC候选 → 6 BT候选
- **关键发现**: Hysteresis反转分组排名 (Group A在Exp4下反超Group B)
- **发现**: CMF家族因子rank稳定, 与Exp4兼容
- **产出**: 修复VEC hysteresis参数遗漏bug (最高价值产出)
- **结论**: 代数组合是OHLCV重排, 信息空间已被17因子覆盖
- **文档**: `docs/research/algebraic_factor_vec_validation.md`

#### 行业约束研究 — NEGATIVE (2026-02-12)
- **测试**: 同行业双持诊断, COMMODITY max_1 A/B测试, 全7行业约束
- **结论**: 同行业双持是最优配置, 非尾部风险放大器; MDD反而恶化
- **文档**: `docs/research/sector_constraint_negative_results.md`

#### 条件因子研究 — NEGATIVE (2026-02-11)
- **测试**: Drop ADX, Drop OBV, regime条件切换, differential trade attribution
- **5个假设全部推翻**: +15pp是路径依赖复利artifact, 非per-trade alpha
- **文档**: `docs/research/conditional_factor_negative_results.md`

#### 跨桶约束 — POSITIVE (2026-02-11)
- **验证**: 5桶 × min_buckets=3 × max_per_bucket=2
- **Phase 1 (VEC)**: HO中位数 +4.84pp, PASS
- **Phase 2 (F5+Exp4)**: HO中位数 +4.92pp, MDD -1.63pp, CONFIRMED
- **决策**: 研究默认开启, 生产配置保持关闭 (向后兼容)
- **文档**: `docs/research/bucket_constraints_ablation.md`

#### C2 Ground Truth验证 — POSITIVE (2026-02-11)
- **BT验证**: HO +45.9%, 0 margin failures, 比BT-S1多+13.4pp
- **Gate 1 (滚动)**: PASS (3/4窗口正收益)
- **Gate 2 (集中度)**: PASS (Gini 0.448)
- **决策**: Shadow 8-12周

#### 折溢价因子 (PREMIUM_DEVIATION_20D) — REJECTED by WFO (2026-02-12)
- 独立IC -0.107, WFO框架下缩水至 -0.044 (< 0.05 gate)
- 正样本率 41% (< 55% gate)
- 已加入config active_factors但不影响S1/C2 (数据仅覆盖2025-02起)

### 3.3 管线基础设施修复

#### VEC hysteresis参数遗漏 (2026-02-12, CRITICAL)
- **症状**: S1 VEC结果 +3.2% vs sealed +54.1% (50pp差距)
- **根因**: `batch_vec_backtest.py` 从未传递 `delta_rank`/`min_hold_days` → 默认0 (禁用)
- **影响**: 所有历史VEC批量结果均无hysteresis, 排名完全错误
- **修复**: 从config读取hysteresis参数并传递给 `run_vec_backtest()`

#### BT sizing commission率不匹配 (2026-02-11, CRITICAL)
- **症状**: 所有组合 ~99个 margin failures, 0% holdout收益
- **根因**: Engine `sizing_commission_rate` 默认2bp, 但SPLIT_MARKET broker收20bp
- **修复**: 传递 `max(a_share, qdii)` 费率

#### BT regime gate双重应用 (2026-02-11)
- **根因**: `vol_regime_series` 与 `gate_arr` 使用相同阈值, engine内部相乘
- **修复**: 设置 `vol_regime_series = 1.0`

---

## 4. 系统架构

### 4.1 关键文件
```
configs/combo_wfo_config.yaml        # 单一配置源 (ETF池, 因子, 引擎参数)
src/etf_strategy/core/frozen_params.py  # 版本锁定参数
src/etf_strategy/core/hysteresis.py     # @njit hysteresis内核
src/etf_strategy/run_combo_wfo.py       # WFO筛选主流程
scripts/batch_vec_backtest.py           # VEC批量回测
scripts/batch_bt_backtest.py            # BT ground truth
scripts/generate_today_signal.py        # 每日信号生成 (有状态)
sealed_strategies/v5.0_20260211/        # 封存策略快照
```

### 4.2 信号评估原则 (v5.0+)
**任何新信号/因子必须在生产执行框架下评估 (FREQ=5 + Exp4 + regime gate), 否则不是有效候选。**

验证证据:
- 相同信号 (S1), 仅执行优化 → HO +35.8pp
- 切换信号, 相同执行 → 价值被破坏 (冠军因子在Exp4下崩溃)
- 因子rank稳定性决定Exp4兼容性 (稳定rank → Exp4过滤噪声; 不稳定rank → Exp4锁死错误仓位)

### 4.3 Alpha维度分析
- **Return PCA**: PC1中位数 59.8%, >50%在77%窗口 → 强单因子主导
- **因子空间Kaiser维度**: 5/17 → 大部分因子冗余
- **S1因子IC相关性**: 均值0.31 (中等), ADX与其他正交, SHARPE×SLOPE=0.84
- **含义**: OHLCV衍生因子的信息空间已近饱和

---

## 5. 待办事项与方向

### 5.1 高优先级 (High ROI)
- [ ] **Shadow C2**: 已BT验证, +13.4pp over S1, 零工程量, 最高ROI
- [ ] **BT验证6个代数因子候选**: 30分钟跑完, 期望值需调低

### 5.2 中优先级 (需要新数据)
- [ ] **Exp7 QDII折溢价**: 需IOPV/NAV数据 (Tushare Pro可用, QMT无)
- [ ] **Exp6 FX汇率归因**: 需forex_daily()数据
- [ ] **Exp5 Rank EMA平滑**: 纯算法改进, 不需新数据

### 5.3 已完结方向 (不再投入)
- [x] 条件因子切换 (S1 4F家族内) — EXHAUSTED
- [x] 行业约束 (POS_SIZE=2下) — DEAD
- [x] 代数因子信号搜索 — 边际递减, 转向新数据源

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
ls sealed_strategies/v5.0_20260211/
cat sealed_strategies/v5.0_20260211/SEAL_SUMMARY.md
```

### 6.3 接手checklist
- [ ] 已阅读 `CLAUDE.md` 了解项目规范
- [ ] 已阅读本文件了解当前状态
- [ ] 已理解三层验证体系和信号评估原则
- [ ] 已理解v5.0生产参数 (FREQ=5, POS_SIZE=2, Exp4 hysteresis)
- [ ] 知道VEC必须传递hysteresis参数

---

## 7. 关键教训

### 7.1 执行 > 信号
```
同一信号, 执行优化 → 3.6x回报乘数
不同信号, 同一执行 → 1.25x回报乘数
更多因子组合       → ~1x (信息饱和)
```

### 7.2 管线一致性
```
VEC必须传递hysteresis参数 (delta_rank, min_hold_days)
BT必须传递正确的sizing_commission_rate
Regime gate只能在timing_arr应用一次, 不能重复
```

### 7.3 因子-执行兼容性
```
稳定rank因子 (ADX, SLOPE, CMF) → Exp4过滤噪声, 表现好
不稳定rank因子 (PV_CORR, PP_20D) → Exp4锁死错误仓位, 崩溃
信号质量 ≠ 生产表现; rank稳定性才是关键
```

---

**文档维护**: 每次重大研究后更新
**封存策略**: `sealed_strategies/v5.0_20260211/`
**研究文档**: `docs/research/`
