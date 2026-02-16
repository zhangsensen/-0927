# ETF 轮动策略研究平台 - 完整项目说明

> **版本**: v5.0 | **更新日期**: 2026-02-12 | **状态**: 生产运行中 (S1 上线 2025-12-18)

---

## 目录

1. [项目定位与目标](#1-项目定位与目标)
2. [系统架构总览](#2-系统架构总览)
3. [核心引擎模块](#3-核心引擎模块)
4. [因子系统](#4-因子系统)
5. [ETF 池架构](#5-etf-池架构)
6. [完整业务流程](#6-完整业务流程)
7. [验证流水线](#7-验证流水线)
8. [风控与择时机制](#8-风控与择时机制)
9. [数据层架构](#9-数据层架构)
10. [配置体系](#10-配置体系)
11. [封存与版本管理](#11-封存与版本管理)
12. [上线策略详情](#12-上线策略详情)
13. [风险分析与已知局限](#13-风险分析与已知局限)
14. [关键文件索引](#14-关键文件索引)
15. [附录](#15-附录)

---

## 1. 项目定位与目标

### 1.1 一句话描述

**基于横截面因子排名的 ETF 轮动策略系统**：从 49 只 ETF 中，每 5 个交易日用多因子打分选出最强的 2 只持有（A_SHARE_ONLY 模式），通过三层引擎 (WFO → VEC → BT) + Exp4 Hysteresis 确保策略可复现、可审计。

### 1.2 核心理念

```
输入: 49 只 ETF 的日线 OHLCV (2020-2026, ~6 年)
      ↓
因子: 23 个因子 (17 OHLCV + 6 non-OHLCV: fund_share + margin)
      ↓
信号: 横截面标准化 (有界→rank, 无界→Z-Score) → 多因子等权合成 → Top-K 排名
      ↓
执行: 每 5 日调仓 + Hysteresis (delta_rank=0.10, min_hold=9) → 持有 2 只 ETF
      ↓
输出: HO +42.7% (S1), MDD 11.8%, 夏普 ~2.15 (VEC)
```

### 1.3 关键数字

| 指标 | 数值 | 说明 |
|------|------|------|
| 回测期 | 2020-01 ~ 2026-02 | ~6 年，覆盖牛熊震荡 |
| ETF 池 | 49 只 | 41 A股 + 8 QDII (A_SHARE_ONLY 模式) |
| 因子库 | 23 个 | 17 OHLCV + 6 non-OHLCV (fund_share + margin) |
| 组合搜索空间 | ~245,157 | C(23, 2..7)，跨桶约束后 ~120,000 |
| 调仓频率 | 5 天 | 锁死参数，禁止修改 |
| 持仓数量 | 2 只 | 集中持仓，高弹性 |
| Hysteresis | delta_rank=0.10, min_hold=9 | Exp4 噪声过滤 |
| 初始资金 | 100 万 | 足够覆盖实盘最小单位 |
| 佣金 | SPLIT_MARKET | A股 20bp, QDII 50bp (med tier) |

### 1.4 项目演进简史

```
2024-Q1  v0.1  探索期: 20只纯A股ETF, 基础因子
2024-Q2  v0.5  扩展期: 30只ETF + 行业ETF
2024-Q3  v0.8  成熟期: 38只 + 债券/商品
2024-Q4  v1.0  锁定: 43只 (加入5只QDII), 单引擎
2025-11  v2.0  工程化: 三层引擎 (WFO/VEC/BT), VEC-BT对齐
2025-12  v3.0  高频化: FREQ=3, POS=2, 收益237%
2025-12  v3.1  审计: ETF池深度分析, QDII贡献确认
2025-12  v3.2  交付: BT Ground Truth + 四重验证
2025-12  v3.4  上线: 震荡市精选双策略
2026-01  v4.0  重构: 16因子正交集, T1_OPEN执行模型
2026-02  v4.1  成本: SPLIT_MARKET成本模型 (A/QDII分级)
2026-02  v4.2  扩展: +13新因子候选, GPU加速IC
2026-02  v5.0  生产: FREQ=5+Exp4 hysteresis, 49 ETF, A_SHARE_ONLY (当前生产版本)
```

---

## 2. 系统架构总览

### 2.1 三层引擎架构

```
┌─────────────────────────────────────────────────────────────┐
│  第一层: WFO (Walk-Forward Optimization) — 粗筛             │
│  ─────────────────────────────────────────────────────────   │
│  脚本: src/etf_strategy/run_combo_wfo.py                    │
│  引擎: combo_wfo_optimizer.py (582行, Numba JIT加速)        │
│  输入: 23因子 × 49 ETF × ~1500天                             │
│  处理: 滚动窗口 IC 计算, ~120,000种因子组合遍历 (跨桶约束)  │
│  输出: Top-100 候选组合 (按 IC/稳定性排序)                   │
│  耗时: ~2 分钟                                               │
│  定位: 快速筛选, 数值可能与VEC/BT有偏差 (正常)             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  第二层: VEC (Vectorized Backtester) — 精算                 │
│  ─────────────────────────────────────────────────────────   │
│  脚本: scripts/batch_vec_backtest.py                        │
│  引擎: Numba @njit 内核 (vec_backtest_kernel)               │
│  输入: WFO Top-100 组合                                      │
│  处理: 完整模拟持仓/调仓/手续费/择时/止损/风控              │
│  输出: 收益率/夏普/Calmar/最大回撤/交易次数/胜率             │
│  耗时: ~5 分钟 (100组合)                                     │
│  定位: 开发主力引擎, 快且精确                               │
│  对齐: 必须与 BT < 0.01pp 差异                              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  第三层: BT (Backtrader) — 审计真值                          │
│  ─────────────────────────────────────────────────────────   │
│  脚本: scripts/batch_bt_backtest.py                         │
│  引擎: Backtrader GenericStrategy (事件驱动, 逐K线模拟)     │
│  输入: 所有候选组合 (或 Top-K 子集)                          │
│  处理: 真实撮合: 滑点/手续费/资金管理/涨跌停检查            │
│  输出: BT收益 (含Train/Holdout分段), 交易明细               │
│  耗时: ~30-60 分钟 (24核并行, 100组合)                      │
│  定位: Ground Truth, 对外交付的最终口径                     │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 模块依赖关系

```
pyproject.toml (editable 安装)
        │
  ┌─────┴─────────────────────┐
  │                           │
  ▼                           ▼
etf_strategy                etf_data
(核心策略模块)              (数据下载, 独立模块)
  │                           └── QMT Bridge SDK
  │
  ├── core/                   ← 核心引擎 (锁定, 禁止修改)
  │   ├── combo_wfo_optimizer.py
  │   ├── precise_factor_library_v2.py
  │   ├── cross_section_processor.py
  │   ├── data_loader.py
  │   ├── ic_calculator_numba.py
  │   ├── market_timing.py
  │   ├── regime_detector.py
  │   ├── wfo_realbt_calibrator.py
  │   ├── factor_stability_tester.py
  │   ├── category_factors.py
  │   ├── data_contract.py
  │   └── utils/rebalance.py   ← VEC/BT对齐的关键共享工具
  │
  ├── auditor/                ← BT审计模块
  │   ├── core/engine.py      ← Backtrader策略实现
  │   └── runners/parallel_audit.py
  │
  └── regime_gate.py          ← 体制门控 (可选)

scripts/                      ← 操作脚本
  ├── batch_vec_backtest.py
  ├── batch_bt_backtest.py
  ├── run_full_space_vec_backtest.py
  ├── run_rolling_oos_consistency.py
  ├── run_holdout_validation.py
  ├── final_triple_validation.py
  ├── generate_production_pack.py
  ├── generate_today_signal.py
  └── run_full_pipeline.py    ← 一键全流程
```

---

## 3. 核心引擎模块

### 3.1 代码量统计

| 模块 | 行数 | 核心职责 |
|------|------|---------|
| `precise_factor_library_v2.py` | 1,616 | 17 OHLCV因子计算 (Numba加速) |
| `category_factors.py` | 759 | 分类因子 (债券/商品/QDII专用) |
| `combo_wfo_optimizer.py` | 582 | 滚动WFO优化器 |
| `cross_section_processor.py` | 551 | 横截面标准化 |
| `factor_stability_tester.py` | 359 | 因子稳定性测试 |
| `wfo_realbt_calibrator.py` | 348 | WFO-实盘校准器 (ML) |
| `data_loader.py` | 258 | 数据加载 + 缓存 |
| `regime_detector.py` | 252 | 市场体制检测 |
| `market_timing.py` | 213 | 择时模块 (轻量/双层) |
| `data_contract.py` | 142 | 数据质量契约 |
| `ic_calculator_numba.py` | 125 | IC计算 (Numba) |
| `utils/rebalance.py` | ~120 | 调仓日程/信号滞后/价格视图 |
| **核心总计** | **~5,325** | |

### 3.2 ComboWFOOptimizer — 滚动优化器

**核心算法**:

1. **窗口滑动**: IS=180天(训练) + OOS=60天(测试), 步长=60天
2. **信号计算**: 多因子等权平均, NaN自适应 (仅对有效因子加权)
3. **IC计算**: Spearman Rank IC, Signal(t-1) vs Return(t:t+freq)
4. **评分公式**:
   ```
   stability_score = 0.5×mean_ic + 0.3×mean_ir + 0.2×positive_rate
                   - 0.1×ic_std - λ×combo_size
   ```
5. **ML校准** (可选): 用GBT模型学习 WFO指标 → 真实夏普的映射关系

**关键配置**:
```yaml
wfo:
  combo_sizes: [2, 3, 4, 5, 6, 7]
  is_period: 180        # 样本内窗口
  oos_period: 60        # 样本外窗口
  step_size: 60         # 滑动步长
  n_jobs: 16            # 并行核数
  enable_fdr: true      # 多重检验校正
  fdr_alpha: 0.05       # 假发现率阈值
  complexity_penalty_lambda: 0.15  # 复杂度惩罚
```

### 3.3 VEC Backtest Kernel — 向量化回测内核

**Numba @njit 编译, 核心循环实现**:

```
对每个调仓日 t (由 generate_rebalance_schedule 生成):
  1. 计算因子合成分数: Σ(standardized_factor[f] × weight[f])
  2. 信号排名: stable_topk (分数降序, 同分按ETF索引升序)
  3. 择时过滤: timing_signal(t-1) × regime_gate(t-1)
  4. 卖出: 不在Top-K中的旧持仓 → 按收盘价卖出, 扣手续费
  5. 买入: 新进Top-K的ETF → 等权分配可用资金, 按收盘价买入
  6. 记录: equity_curve[t], daily_return, 胜率/盈亏比统计
```

**关键特性**:
- 零前视偏差: `shift_timing_signal()` 滞后1天
- VEC-BT对齐: 使用相同的 `generate_rebalance_schedule()`
- 风控模块: ATR止损 / 固定%止损 / 阶梯止盈 / 熔断 / 冷却期 / 杠杆上限
- 波动率体制: HS300 HV分档缩放持仓比例

### 3.4 BT Auditor — Backtrader审计引擎

**架构**: Backtrader `GenericStrategy` + 多进程并行 (24 workers)

**流程**:
```
Worker初始化: 加载OHLCV + 因子数据 (共享)
  ↓
接收任务: combo_str (如 "ADX_14D + SLOPE_20D")
  ↓
构建信号: 解析因子 → 横截面标准化 → 合成分数
  ↓
BT回测: Backtrader逐K线模拟 (Cheat-On-Close模式)
  ↓
输出: {bt_return, bt_train_return, bt_holdout_return,
       bt_max_drawdown, bt_sharpe, bt_calmar,
       bt_total_trades, bt_win_rate, bt_profit_factor}
```

**VEC-BT对齐保证**:
- 平均差异: **0.06pp** (100个策略)
- MAX_DD_60D组合: **0.015pp** (达到0.01pp级别)
- 差异来源: 浮点精度累积 (每笔交易0.4~2.8元), 非逻辑错误

### 3.5 CrossSectionProcessor — 横截面标准化

**对每个交易日, 对每个因子**:

```
无界因子 (MOM, SLOPE, VOL等):
  Winsorize: clip to [P2.5, P97.5]
  Z = (x - mean_cs) / std_cs

有界因子 (ADX, RSI, PRICE_POSITION等):
  → Rank 标准化到 [-0.5, 0.5]
  (不 Winsorize: 天然有界; 不原值透传: 避免尺度差异主导 VEC 内核求和)
```

> **v5.0 关键变更**: 有界因子从"原值透传"改为"rank 标准化 [-0.5, 0.5]"。此修复消除了 bounded factors 与 Z-scored factors 的尺度不匹配问题 (ADX [7,92] vs Z-score [-3,+3])。

**有界因子清单 (7 个)**:
```
ADX_14D  [0, 100]
CMF_20D  [-1, 1]
CORRELATION_TO_MARKET_20D  [-1, 1]
PRICE_POSITION_20D  [0, 1]
PRICE_POSITION_120D  [0, 1]
PV_CORR_20D  [-1, 1]
RSI_14  [0, 100]
```

### 3.6 Rebalance Utilities — VEC/BT对齐核心

三个关键函数 (所有引擎必须使用):

```python
# 1. 生成统一调仓日程 (VEC和BT共用)
schedule = generate_rebalance_schedule(
    total_periods=T,          # 总K线数
    lookback_window=252,      # 回看期
    freq=5,                   # 调仓频率
)
# 输出: [253, 256, 259, 262, ...]

# 2. 信号滞后 (防前视偏差)
timing_arr = shift_timing_signal(raw_timing)
# shifted[0] = 1.0, shifted[t] = raw[t-1]

# 3. 价格视图统一 (close_t-1 / open_t / close_t)
close_prev, open_t, close_t = ensure_price_views(close, open)
```

---

## 4. 因子系统

### 4.1 23 个活跃因子全景

#### OHLCV 衍生因子 (17 个)

| # | 因子名 | 类别 | 窗口 | 有界 | 方向 | S1 使用 |
|---|--------|------|------|------|------|:-------:|
| 1 | **ADX_14D** | 趋势强度 | 14d | [0,100] | 越高越好 | **S1** |
| 2 | **SLOPE_20D** | 趋势 | 20d | 无 | 越高越好 | **S1** |
| 3 | VORTEX_14D | 趋势 | 14d | 无 | 中性 | |
| 4 | MOM_20D | 动量 | 20d | 无 | 越高越好 | |
| 5 | BREAKOUT_20D | 动量 | 20d | 无 | 越高越好 | |
| 6 | PRICE_POSITION_20D | 价格位置 | 20d | [0,1] | 中性 | |
| 7 | PRICE_POSITION_120D | 价格位置 | 120d | [0,1] | 中性 | |
| 8 | MAX_DD_60D | 风险 | 60d | 无 | 越低越好 | |
| 9 | CALMAR_RATIO_60D | 风险调整 | 60d | 无 | 越高越好 | |
| 10 | **SHARPE_RATIO_20D** | 风险调整 | 20d | 无 | 越高越好 | **S1** |
| 11 | CORRELATION_TO_MARKET_20D | 相关性 | 20d | [-1,1] | 越低越好 | |
| 12 | **OBV_SLOPE_10D** | 资金流 | 10d | 无 | 越高越好 | **S1** |
| 13 | PV_CORR_20D | 量价耦合 | 20d | [-1,1] | 越高越好 | |
| 14 | VOL_RATIO_20D | 成交量 | 20d | 无 | 越高越好 | |
| 15 | GK_VOL_RATIO_20D | 波动率微结构 | 20d | 无 | 中性 | |
| 16 | UP_DOWN_VOL_RATIO_20D | 量能方向性 | 20d | 无 | 越高越好 | |
| 17 | AMIHUD_ILLIQUIDITY | 流动性 | 20d | 无 | 越低越好 | |

#### Non-OHLCV 因子 (6 个, v5.0 新增)

| # | 因子名 | 数据源 | 说明 |
|---|--------|--------|------|
| 18 | SHARE_CHG_5D | fund_share | 5 日基金份额变化率 |
| 19 | SHARE_CHG_10D | fund_share | 10 日基金份额变化率 |
| 20 | SHARE_CHG_20D | fund_share | 20 日基金份额变化率 |
| 21 | SHARE_ACCEL | fund_share | 份额变化加速度 (5D/20D) |
| 22 | MARGIN_CHG_10D | margin | 10 日融资余额变化率 |
| 23 | MARGIN_BUY_RATIO | margin | 融资买入占比 |

> Non-OHLCV 因子通过 `extra_factors` 机制从预计算 parquet 加载，需先运行 `uv run python scripts/precompute_non_ohlcv_factors.py`。 |

### 4.2 因子计算特征

- **100% 向量化**: 无 Python 循环, 无 `.apply()`, 全部 Pandas/NumPy 批操作
- **性能**: 17 OHLCV因子 × 49 ETF × 1500天 < 100ms
- **NaN策略**: 严格保留, 不填充, 不插值 (窗口不足→NaN)
- **Numba加速**: IC计算、信号合成使用 @njit 编译
- **非OHLCV因子**: 预计算后通过 extra_factors 机制加载，与 OHLCV 因子统一参与标准化

### 4.3 因子组合搜索

```
C(23,2) =     253 种 2因子组合
C(23,3) =   1,771 种 3因子组合
C(23,4) =   8,855 种 4因子组合
C(23,5) =  33,649 种 5因子组合
C(23,6) = 100,947 种 6因子组合
C(23,7) = 245,157 种 7因子组合
──────────────────────────────
合计     = ~245,157 种组合 (无约束搜索空间)
跨桶约束 (min_buckets=3, max_per_bucket=2) → ~120,000 种 (-51%)
```

### 4.4 已知不稳定因子

| 因子 | 问题 | VEC-BT偏差 |
|------|------|-----------|
| OBV_SLOPE_10D | BT中有累积漂移 | 61pp |
| CMF_20D | BT计算路径不同 | 35pp |

> 这两个因子标记为 `production_ready=False`, 但在v3.4上线策略中仍然使用了OBV_SLOPE_10D (因为其收益贡献显著)。需要注意VEC-BT对齐监控。

---

## 5. ETF 池架构

### 5.1 49 只 ETF 池结构 (v5.0)

| 分类 | 数量 | 核心作用 |
|------|------|---------|
| A股宽基 | 7 | Beta来源 (沪深300/中证500/科创50等) |
| A股成长 | 17 | 行业轮动 (半导体/新能源/AI/医药等) |
| A股周期 | 7 | 板块轮动 (证券/银行/有色/军工/房地产等) |
| A股防御 | 3 | 避险 (消费/红利) |
| 债券 | 3 | 熊市避风港 (国债/可转债) |
| 商品 | 3 | 通胀对冲 (黄金/白银/豆粕) |
| A股资源 | 1 | 能源 (煤炭) |
| **QDII** | **8** | **横截面校准 + 跨市场监控** |
| **合计** | **49** | |

> **v5.0 新增 6 只**: 159985 豆粕 ETF, 512200 房地产 ETF, 515220 煤炭 ETF (A 股), 513180 恒生科技 ETF, 513400 道琼斯 ETF, 513520 日经 ETF (QDII)

### 5.2 A_SHARE_ONLY 模式

v5.0 生产使用 `A_SHARE_ONLY` 模式: 8 只 QDII 参与因子计算和横截面标准化，但硬屏蔽不进入实盘持仓。
- **原因**: QDII 交易成本高 (50-80bp)、结算差异、额度限制
- **验证**: 实盘 6 周 (2025-12-18 ~ 2026-02-09) 100% A 股持仓, +6.37%

### 5.3 QDII 历史贡献 (v3.1 GLOBAL 模式参考)

> 以下数据基于 v3.1 (FREQ=3, 43 ETF, GLOBAL 模式)。v5.0 不交易 QDII，数据仅供参考横截面校准价值。

| 代码 | 名称 | 市场 | 历史选中 | 历史胜率 | 历史贡献 |
|------|------|------|---------|---------|---------|
| 513500 | 标普500 ETF | 美股 | 45 | 68.9% | +25.37% |
| 513130 | 恒生科技(港元) | 港股 | 15 | 53.3% | +23.69% |
| 513100 | 纳指100 ETF | 美股 | 31 | 61.3% | +22.03% |
| 159920 | 恒生指数 ETF | 港股 | 10 | 70.0% | +17.13% |
| 513050 | 中概互联 ETF | 港股 | 18 | 44.4% | +2.01% |

---

## 6. 完整业务流程

### 6.1 六阶段流水线

```
阶段 0: 数据更新
  ↓ scripts/update_daily_from_qmt_bridge.py --all
  ↓ (QMT Bridge SDK → Parquet 文件)
  ↓
阶段 1: WFO 因子组合挖掘 (粗筛)
  ↓ src/etf_strategy/run_combo_wfo.py
  ↓ 输入: 18因子 × 43ETF × ~1500天
  ↓ 输出: results/run_YYYYMMDD/top100_by_ic.parquet
  ↓ 耗时: ~2分钟
  ↓
阶段 2: VEC 向量化精算 (Screening)
  ↓ scripts/run_full_space_vec_backtest.py
  ↓ 输入: WFO Top-100 组合
  ↓ 输出: results/vec_full_backtest_YYYYMMDD/vec_all_combos.parquet
  ↓ 耗时: ~5分钟
  ↓
阶段 3: Rolling + Holdout 无泄漏验证
  ↓ scripts/final_triple_validation.py
  ↓ (内部调用 Rolling OOS + Holdout Validation)
  ↓ 三重门控: 训练稳定性 + 样本外盈利 + 风险过滤
  ↓ 输出: results/final_triple_validation_YYYYMMDD/final_candidates.parquet
  ↓
阶段 4: BT 审计 (Ground Truth)
  ↓ scripts/batch_bt_backtest.py
  ↓ 输入: final_candidates (或 Top-K)
  ↓ 输出: results/bt_backtest_full_YYYYMMDD/bt_results.parquet
  ↓ 含: bt_train_return / bt_holdout_return 分段收益
  ↓ 耗时: ~30-60分钟 (24核并行)
  ↓
阶段 5: 生产包 + 封存
  ↓ archive/scripts/legacy_reports/generate_production_pack.py
  ↓ 评分: prod_score = 0.45×Holdout + 0.20×Calmar + 0.20×(-MaxDD) + 0.15×Train
  ↓ 输出: production_candidates.parquet + PRODUCTION_REPORT.md
  ↓
  ↓ scripts/seal_release.py  (封存归档)
  ↓ 输出: sealed_strategies/v3.X_YYYYMMDD/ (含CHECKSUMS.sha256)
  ↓
阶段 6: 每日信号生成 (实盘)
  scripts/generate_today_signal.py
  输入: production_candidates.parquet + 当日收盘数据
  输出: CSV 目标持仓 (策略ID → ETF → 数量 → 买/卖)
```

### 6.2 一键全流程

```bash
uv run python scripts/run_full_pipeline.py \
  --top-n 200 \
  --n-jobs 24 \
  --regime-gate auto
# 耗时: ~1-2小时, 输出完整的 production_pack
```

### 6.3 数据流转图

```
原始数据 (Parquet)
    ↓ DataLoader (缓存到Pickle, 43x加速)
OHLCV DataFrame (T × N)
    ↓ PreciseFactorLibrary.compute_all_factors()
原始因子矩阵 (T × N × 18)
    ↓ CrossSectionProcessor.process_all_factors()
标准化因子 (Z-Score/Winsorize后, T × N × 18)
    ↓ ComboWFOOptimizer.run_combo_search()
    │  ├── 滚动窗口 IC 计算 (Numba)
    │  └── 评分排名 (stability_score / calibrated_sharpe)
Top-100 候选组合
    ↓ batch_vec_backtest.py
    │  ├── 因子合成 → 信号排名 → Top-K选股
    │  ├── 模拟持仓/调仓/手续费
    │  └── 输出: equity_curve + risk_metrics
VEC 精确回测结果
    ↓ final_triple_validation.py
    │  ├── Rolling OOS: 分季度一致性检验
    │  ├── Holdout: 训练期后盲测
    │  └── AND 门控: 三重全通才保留
Final Candidates (无泄漏, 全通过)
    ↓ batch_bt_backtest.py
    │  └── Backtrader 事件驱动审计 (Ground Truth)
BT Audit Results
    ↓ generate_production_pack.py
    │  └── 综合评分排名 → Top-20
Production Pack → 实盘部署
```

---

## 7. 验证流水线

### 7.1 四重验证体系

```
                ┌────────────────────┐
                │  WFO 粗筛 (L1)     │
                │  ~12,597 → Top-100 │
                └─────────┬──────────┘
                          ↓
                ┌────────────────────┐
                │  VEC 精算 (L2)     │
                │  全量回测指标计算   │
                └─────────┬──────────┘
                          ↓
          ┌───────────────┼───────────────┐
          ↓               ↓               ↓
   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
   │ Rolling OOS  │ │   Holdout    │ │  风险过滤    │
   │ 分段一致性   │ │ 盲测数据     │ │ 因子黑名单   │
   │ ≥60%正收益   │ │ 收益>0       │ │ 排除不稳定组 │
   └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
          └───────────────┬┘               │
                          ↓                │
                ┌────────────────────┐     │
                │  Triple AND门控    │←────┘
                │  三重全通才保留    │
                └─────────┬──────────┘
                          ↓
                ┌────────────────────┐
                │  BT审计 (L3)      │
                │  Ground Truth 确认│
                └─────────┬──────────┘
                          ↓
                ┌────────────────────┐
                │  Production Pack   │
                │  综合评分 Top-20   │
                └────────────────────┘
```

### 7.2 三道门控

**Gate 1: Rolling OOS 一致性**
```
数据: 仅训练期 (防止Holdout泄漏)
切片: 按季度/月/年切分
通过条件:
  ✓ ≥60% 分段正收益 (pos_rate ≥ 0.60)
  ✓ 最差分段收益 > -8%
  ✓ 平均Calmar > 0.80
```

**Gate 2: Holdout 盲测**
```
数据: training_end_date → end_date (完全未见)
通过条件:
  ✓ Holdout收益 > 0% (必须盈利)
  ✓ 最大回撤 < 50% (灾难排除)
过拟合检测:
  ⚠️ 如果 训练Calmar / Holdout Calmar > 1.5 → 显著过拟合
```

**Gate 3: 风险过滤**
```
排除: 已知不稳定因子组合 (EXCLUDE_FACTORS集合)
排除: 特定因子对 (EXCLUDE_FACTOR_PAIRS)
排除: BT审计失败的组合
```

### 7.3 综合评分

**Triple Validation 评分**:
```
composite_score = 0.3 × rank(vec_calmar)
               + 0.4 × rank(worst_segment_return)  ← 强调稳健性
               + 0.3 × rank(holdout_calmar)
```

**Production Pack 评分 (BT口径)**:
```
prod_score_bt = 0.45 × rank(bt_holdout_return)     ← 强调盲测
             + 0.20 × rank(bt_calmar_ratio)
             + 0.20 × rank(-bt_max_drawdown)       ← 控回撤
             + 0.15 × rank(bt_train_return)
```

### 7.4 典型筛选漏斗

```
全空间:     ~12,597 组合
  ↓ (WFO)
粗筛:       100 候选
  ↓ (VEC)
精算:       100 (含完整指标)
  ↓ (Rolling)
一致性通过: ~52 候选
  ↓ (Holdout)
盲测通过:   ~35 候选
  ↓ (Triple AND)
最终金选:   ~15-20 候选
  ↓ (Basket)
上线组合:   2-7 策略
```

---

## 8. 风控与择时机制

### 8.1 择时模块

**轻量择时 (LightTimingModule)** — 默认模式:
```
composite = 0.4 × MA信号 + 0.4 × 动量信号 + 0.2 × 黄金信号

仓位 = 1.0 (满仓)     如果 composite ≥ -0.4
仓位 = 0.3 (防御)     如果 composite < -0.4
```

**双层择时 (DualTimingModule)** — 高级模式:
```
层级1 (宏观防御): 大盘 < MA200 → 仓位降至 10%
层级2 (微观过滤): 个股 < MA20 → 禁止买入/强制卖出
```

### 8.2 Regime Gate (体制门控)

可选功能, 通过 `combo_wfo_config.yaml` 配置开关:

**波动率模式** (默认):
```
代理: 沪深300 (510300)
计算: 20日HV年化
分档:
  HV < 25%  → 满仓 (1.0x)
  HV 25-30% → 70%仓位 (0.7x)
  HV 30-40% → 40%仓位 (0.4x)
  HV ≥ 40%  → 10%仓位 (0.1x, 近似空仓)
```

### 8.3 止损机制

| 模式 | 参数 | 说明 |
|------|------|------|
| 固定%止损 | 8% | 回撤超8%自动止损 |
| ATR动态止损 | 3倍ATR(14日) | 波动自适应止损 |
| 阶梯止盈 | 收益>20%→止损收紧到8%, >40%→5% | 保护利润 |
| 熔断 | 单日-5%或总-20%暂停 | 极端行情保护 |
| 冷却期 | 止损后N天禁止买入同标的 | 防止追跌 |
| 杠杆上限 | 1.0 (零杠杆原则) | 永不加杠杆 |

### 8.4 防前视偏差设计

| 环节 | 机制 | 保障 |
|------|------|------|
| 因子计算 | 仅使用历史数据 | 窗口只看过去 |
| 信号执行 | `shift_timing_signal()` | t日信号→t+1日执行 |
| IC计算 | Signal(t-1) vs Return(t:t+freq) | 不访问当日数据 |
| 调仓日程 | `generate_rebalance_schedule()` | VEC/BT共用, 无偏差 |
| Holdout数据 | 严格在training_end_date之后 | 训练期不可见 |

---

## 9. 数据层架构

### 9.1 数据来源

```
QMT Trading Terminal (VM: 192.168.122.132:8001)
    ↓ qmt-data-bridge SDK (异步API)
    ↓
scripts/update_daily_from_qmt_bridge.py --all
    ↓
raw/ETF/daily/*.parquet  (原始日线数据)
    ↓
DataLoader (src/etf_strategy/core/data_loader.py)
    ↓ 缓存: .cache/ohlcv_*.pkl (Pickle, 43x加速)
    ↓
内存中的 OHLCV 字典:
  {'close': DataFrame(T×N), 'open': ..., 'high': ..., 'low': ..., 'volume': ...}
```

### 9.2 数据特征

| 属性 | 值 |
|------|-----|
| 格式 | Parquet (列式存储) |
| 时间范围 | 2020-01-01 ~ 2025-12-12 |
| 覆盖 | 49 只 ETF (41 A股 + 8 QDII), 日线级别 |
| 价格 | 前复权 (adj_close/adj_open等) |
| 缺失处理 | NaN保留, 不填充 |
| 缺失率 | < 20% (DataContract约束) |
| 最小天数 | ≥ 100 个交易日 |

### 9.3 etf_data 模块 (独立)

```
src/etf_data/
├── core/
│   ├── data_manager.py    # 数据持久化 (保存/加载Parquet)
│   ├── downloader.py      # QMT Bridge下载器
│   ├── models.py          # 数据模型 (ETFInfo, ETFStatus等)
│   └── config.py          # 配置管理
├── config/
│   └── etf_config_manager.py
└── scripts/
    ├── batch_download.py  # 批量下载
    └── quick_download.py  # 快速下载
```

> 重要: `etf_data` 是独立的数据工具, 不参与策略运行。策略仅依赖 `etf_strategy` 模块 + `raw/ETF/` 数据。

---

## 10. 配置体系

### 10.1 主配置 (`configs/combo_wfo_config.yaml`)

```yaml
data:
  start_date: "2020-01-01"
  end_date: "2025-12-12"
  training_end_date: "2025-04-30"    # Holdout分界
  symbols: [510050, 510300, ..., 513130]  # 43只ETF
  data_dir: "raw/ETF/daily"
  cache_dir: ".cache"
  symbols: [49 只 ETF]       # 41 A股 + 8 QDII

backtest:
  freq: 5                    # 调仓频率 (交易日) ← 锁死
  pos_size: 2                # 持仓数量 ← 锁死
  initial_capital: 1000000
  execution_model: "T1_OPEN"
  cost_model:
    mode: "SPLIT_MARKET"     # A股/QDII分级费率
    tier: "med"              # A股 20bp, QDII 50bp

  hysteresis:                # Exp4 噪声过滤 ← v5.0 核心变更
    delta_rank: 0.10         # 最小 rank01 差才换仓
    min_hold_days: 9         # 最短持有天数

  regime_gate:
    enabled: true
    mode: "volatility"
    volatility:
      proxy_symbol: "510300"
      window: 20
      thresholds_pct: [25, 30, 40]
      exposures: [1.0, 0.7, 0.4, 0.1]

universe:
  mode: "A_SHARE_ONLY"      # QDII 参与标准化但不交易

wfo:
  combo_sizes: [2, 3, 4, 5, 6, 7]
  is_period: 180
  oos_period: 60
  step_size: 60
  n_jobs: -1
  enable_fdr: true
  fdr_alpha: 0.05
  complexity_penalty_lambda: 0.15
  bucket_constraints:
    enabled: true
    min_buckets: 3
    max_per_bucket: 2
  scoring_weights:
    annual_return: 0.4
    sharpe_ratio: 0.3
    max_drawdown: 0.3
```

### 10.2 ETF池配置 (`configs/etf_pools.yaml`)

定义7个战略子池 (宽基/成长/周期/防御/债券/商品/QDII), 含每只ETF的:
- 代码、名称、类别
- 选择原因
- 数据起始日期
- 资金配比建议

---

## 11. 封存与版本管理

### 11.1 封存流程

```bash
uv run python scripts/seal_release.py \
  --version v3.4 --date 20251216 \
  --final-candidates results/.../final_candidates.parquet \
  --bt-results results/.../bt_results.parquet \
  --production-dir results/production_pack_YYYYMMDD_HHMMSS \
  --force
```

### 11.2 封存包结构

```
sealed_strategies/v3.X_YYYYMMDD/
├── locked/                 # 源码快照
│   ├── src/                # 核心代码
│   ├── scripts/            # 关键脚本
│   ├── configs/            # 配置文件
│   ├── pyproject.toml      # 依赖规格
│   └── uv.lock             # 锁定依赖版本
├── artifacts/              # 产出物
│   ├── production_candidates.csv
│   ├── PRODUCTION_REPORT.md
│   └── DEPLOYMENT_GUIDE.md
├── MANIFEST.json           # 版本元数据
├── CHECKSUMS.sha256        # SHA256完整性校验
├── REPRODUCE.md            # 一键复现指南
└── README.md
```

### 11.3 版本历史

| 版本 | 日期 | 策略数 | 说明 |
|------|------|--------|------|
| v3.1 | 2025-12-15 | 多策略 | BT Ground Truth 基线 |
| v3.2 | 2025-12-14 | 152 → 120 | 四重验证交付版 |
| v3.4 | 2025-12-16 | 2 | 震荡市精选双策略 |
| v4.0 | 2026-01-31 | - | 16 因子正交集 |
| v4.1 | 2026-02-03 | - | SPLIT_MARKET 成本模型 |
| v4.2 | 2026-02-05 | - | 因子扩展研究 |
| **v5.0** | **2026-02-11** | **1 (S1)** | **FREQ=5 + Exp4 + 49 ETF, 当前生产版** |
| c2_shadow | 2026-02-11 | 1 (C2) | Shadow 候选 (AMIHUD+CALMAR+CORR_MKT) |

---

## 12. 上线策略详情 (v5.0)

### 12.1 生产策略 S1 (4因子)

```
因子: ADX_14D + OBV_SLOPE_10D + SHARPE_RATIO_20D + SLOPE_20D
执行: FREQ=5, Exp4 Hysteresis (delta_rank=0.10, min_hold_days=9), Regime Gate ON
```

| 指标 | VEC (v5.0) | BT (med cost) |
|------|-----------|---------------|
| HO 收益 | **+42.7%** | +32.5% |
| HO MDD | 11.8% | 11.0% |
| HO Sharpe | 2.15 | - |
| Worst Month | -2.7% | - |
| Trades | 75 | 120 |
| Margin Failures | - | 0 |

### 12.2 Shadow 候选 C2 (3因子)

```
因子: AMIHUD_ILLIQUIDITY + CALMAR_RATIO_60D + CORRELATION_TO_MARKET_20D
状态: Shadow 部署中, 8-12 周后评估
```

| 指标 | VEC | BT (med cost) |
|------|-----|---------------|
| HO 收益 | +63.9% | +45.9% |
| HO MDD | 10.4% | 12.2% |
| Trades | 28 | 34 |

### 12.3 S1 因子含义

| 因子 | 含义 | 策略逻辑 |
|------|------|---------|
| **ADX_14D** | 趋势强度 | 选择正在形成趋势的 ETF (不关心方向) |
| **SLOPE_20D** | 线性回归斜率 | 选择价格有上升趋势的 ETF |
| **SHARPE_RATIO_20D** | 近 20 日风险调整收益 | 选择"性价比"最高的 ETF |
| **OBV_SLOPE_10D** | OBV 斜率 (资金流) | 选择有持续资金流入的 ETF |

**策略思路**: 寻找"趋势明确(ADX)、方向向上(SLOPE)、资金流入(OBV)、风险调整收益好(SHARPE)"的 ETF, 即**强势且有资金支撑**的品种。

### 12.4 为何 S1 + F5_ON

- **因子排名稳定性**: ADX/OBV/SHARPE/SLOPE 的横截面排名高度稳定 (rank autocorrelation > 0.8), 与 Exp4 hysteresis 高度兼容
- **"冠军"因子不兼容**: AMIHUD+PP20D+PV_CORR+SLOPE 排名波动大 → Exp4 锁仓 → 训练期负收益
- **执行 > 信号**: v3.4→v5.0 同一信号 (S1), 仅执行改进 (FREQ=5 + Exp4) 就提升 +35.8pp holdout 收益
- **实盘验证**: 2025-12-18 ~ 2026-02-09, +6.37%, 22 trades, 83.3% WR

---

## 13. 风险分析与已知局限

### 13.1 核心风险

| 风险 | 等级 | 说明 |
|------|------|------|
| **QDII集中度** | 高 | 5只海外ETF贡献90%+收益, 美股崩盘→策略同步崩溃 |
| **策略同质性** | 中高 | 两策略因子重叠80%, 持仓重叠>80%, 实质约1.5个策略 |
| **换手敏感** | 中 | 平均持有9天, 年化~4000%换手, 滑点影响大 |
| **过拟合风险** | 中 | 12,597种组合搜索, 虽有FDR校正但统计风险仍存 |
| **OBV_SLOPE不稳定** | 中 | 该因子VEC-BT偏差61pp, 但上线策略仍使用 |
| **市场结构变化** | 低中 | 策略基于2020-2025训练, 市场结构变化可能导致失效 |

### 13.2 性能边界

| 场景 | 预期表现 |
|------|---------|
| 趋势市 (如2020H2) | 优秀, 因子能捕捉趋势 |
| 震荡市 (如2025Q4) | 微亏或持平 (-0.23%) |
| 急跌 (如2020Q1) | 体制门控缩仓至10%, 但仍有损失 |
| 流动性枯竭 | 43只ETF日均成交>1亿, 风险可控 |
| 美股闪崩 | QDII标的可能跳空低开, 无法及时止损 |

### 13.3 已知技术债务

1. v3.4封存包包含 `.venv/` 目录 (1.2GB), 违反封存指南, 应清理
2. OBV_SLOPE_10D 因子在BT中有61pp漂移, 上线策略仍在使用
3. Makefile中的路径部分指向旧目录名 (`factor_system/`, `etf_rotation_optimized/`)

---

## 14. 关键文件索引

### 14.1 核心引擎 (禁止修改)

| 文件 | 行数 | 职责 |
|------|------|------|
| `src/etf_strategy/core/precise_factor_library_v2.py` | 1,616 | 17 OHLCV 因子计算引擎 |
| `src/etf_strategy/core/combo_wfo_optimizer.py` | 582 | WFO滚动优化器 |
| `src/etf_strategy/core/cross_section_processor.py` | 551 | 横截面标准化 |
| `src/etf_strategy/core/data_loader.py` | 258 | 数据加载 + 缓存 |
| `src/etf_strategy/core/ic_calculator_numba.py` | 125 | Numba IC计算 |
| `src/etf_strategy/core/market_timing.py` | 213 | 择时模块 |
| `src/etf_strategy/core/utils/rebalance.py` | ~120 | VEC/BT对齐工具 |
| `src/etf_strategy/auditor/core/engine.py` | ~500 | Backtrader策略 |

### 14.2 操作脚本

| 文件 | 职责 |
|------|------|
| `src/etf_strategy/run_combo_wfo.py` | WFO入口 (阶段1) |
| `scripts/batch_vec_backtest.py` | VEC批量回测 (阶段2) |
| `scripts/run_full_space_vec_backtest.py` | VEC全空间回测 |
| `scripts/final_triple_validation.py` | 三重验证 (阶段3) |
| `scripts/batch_bt_backtest.py` | BT批量审计 (阶段4) |
| `archive/scripts/legacy_reports/generate_production_pack.py` | 生产包生成 (阶段5) |
| `scripts/generate_today_signal.py` | 每日信号 (阶段6) |
| `scripts/run_full_pipeline.py` | 一键全流程 |
| `scripts/seal_release.py` | 封存归档 |

### 14.3 配置与文档

| 文件 | 职责 |
|------|------|
| `configs/combo_wfo_config.yaml` | 主配置 (参数/ETF池/风控) |
| `configs/etf_pools.yaml` | ETF池详细定义 |
| `CLAUDE.md` | LLM 开发指南 |
| `docs/ETF_POOL_ARCHITECTURE.md` | ETF 池深度分析 (49 ETF) |
| `docs/ETF_DATA_GUIDE.md` | 数据目录与数据源指南 |
| `sealed_strategies/v5.0_20260211/` | 当前生产版本封存 |

---

## 15. 附录

### 15.1 技术栈

| 技术 | 用途 | 版本要求 |
|------|------|---------|
| Python | 主语言 | ≥ 3.11 |
| UV | 包管理 | 必须使用 |
| Numba | JIT编译 (因子/IC/回测内核) | ≥ 0.62 |
| Pandas | 数据处理 | ≥ 2.3 |
| NumPy | 数值计算 | ≥ 2.3 |
| Backtrader | 事件驱动回测 (审计层) | ≥ 1.9 |
| vectorbt | 向量化回测 (参考) | ≥ 0.28 |
| LightGBM | WFO校准器 | 可选 |
| Optuna | 超参优化 | 可选 |
| TA-Lib | 技术指标 | ≥ 0.6 |
| PyArrow | Parquet读写 | ≥ 21.0 |
| Joblib | 并行计算 | ≥ 1.5 |
| Matplotlib/Plotly | 可视化 | 可选 |
| QMT Bridge | 实时数据源 | SDK |

### 15.2 硬件环境

| 组件 | 规格 |
|------|------|
| CPU | AMD Ryzen 9 (16核/32线程) |
| GPU | NVIDIA RTX 5070 Ti (ML加速, 非策略核心) |
| 内存 | ≥ 32GB |
| 存储 | NVMe SSD (数据读写) |

### 15.3 常用命令速查

```bash
# 环境管理
uv sync --dev              # 安装依赖
uv run python <script.py>  # 运行脚本 (必须用uv run)

# 生产流水线
uv run python src/etf_strategy/run_combo_wfo.py           # 1. WFO
uv run python scripts/run_full_space_vec_backtest.py       # 2. VEC
uv run python scripts/final_triple_validation.py           # 3. 验证
uv run python scripts/batch_bt_backtest.py                 # 4. BT审计
uv run python archive/scripts/legacy_reports/generate_production_pack.py          # 5. 生产包

# 每日操作
uv run python scripts/update_daily_from_qmt_bridge.py --all  # 数据更新
uv run python scripts/generate_today_signal.py \              # 生成信号
  --candidates results/.../production_candidates.parquet \
  --asof 2025-12-12 --trade-date 2025-12-15 --capital 50000

# 代码质量
make format   # black + isort
make lint     # flake8 + mypy
make test     # pytest
```

### 15.4 禁止操作清单

| 操作 | 原因 |
|------|------|
| 修改 FREQ/POS/Hysteresis 参数 | 策略核心参数已锁定 (v5.0) |
| 增删 ETF 池成员 | 横截面标准化会改变所有信号 |
| 修改核心因子库 | 可能导致所有回测结果失效 |
| 修改回测引擎逻辑 | VEC-BT对齐依赖一致的逻辑 |
| 使用 pip install | 必须使用 uv 管理依赖 |
| 删除 ARCHIVE 目录 | 保留历史最佳结果 |
| 移除任何 QDII ETF | 8 只 QDII 参与横截面校准 |

---

*最后更新: 2026-02-12*
*文档生成: 基于全项目深度分析, v5.0 参数更新*
