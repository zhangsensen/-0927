# v8.0 封板复盘：从 v5.0 到 v8.0 的教训

> 日期: 2026-02-15
> 类型: 教训 / 决策记录

## 事件时间线

| 日期 | 事件 | 影响 |
|------|------|------|
| 02-11 | v5.0 封板 (S1 4F, HO +42.7%) | S1 上线基准 |
| 02-12 | 发现 ADX Winsorize artifact → v5.0 废弃 | S1 真实 HO 仅 +30.4% (-12.3pp) |
| 02-12 | v6.0 封板 (C2 3F) | C2 被推为 successor |
| 02-12 | C2 Train Gate 失败 (train -20.9%) | C2 从 successor 降级 |
| 02-13 | v7.0 封板 (6F #1) | VEC-BT gap +25.6pp 未被发现 |
| 02-14 | 三大管线修复 (IC-sign / metadata / exec-side) | Gap 从 25pp 降到 <2pp |
| 02-14 | 155 候选全量 BT 验证 | 首次 pipeline 完全正确的结果 |
| 02-15 | v8.0 封板 | **首个管线修复后的 clean seal** |

## 核心教训

### 教训 1: 封板前必须检查 VEC-BT gap

v7.0 的 VEC-BT gap 是 +25.6pp，远超 10pp 红线，但封板时没有系统检查。
v8.0 的 155 个候选全部 gap < 2pp。

**Rule 25 (新增)**: 封板前 VEC-BT gap 检查是硬性前置条件，gap > 10pp = 不允许封板。

### 教训 2: 3 个 bug 累积效应远大于单个

| Bug | 单独影响 | 累积影响 |
|-----|---------|---------|
| IC-sign 反向 | ~20pp gap | 非OHLCV因子反向 → 选错ETF |
| VEC metadata 丢失 | ~20pp gap | Holdout/Rolling 用错权重 |
| BT signal-side hysteresis | +25.7pp gap | 33次多余交易 |
| **三者叠加** | — | v7.0 全部结果不可信 |

**教训**: Bug 是乘法不是加法。每个单独"还行"的问题组合后致命。**修一个不够，必须全修完再封板。**

### 教训 3: 版本快速迭代的代价

```
v5.0 (02-11) → 废弃 (02-12, 1天)
v6.0 (02-12) → 从未正式使用
v7.0 (02-13) → 废弃 (02-15, 2天)
v8.0 (02-15) → 首个 clean seal
```

5天内封了4个版本，前3个全废弃。根因：**急于封板，修一个 bug 就封一版，没有等全部修完。**

**Rule 26 (新增)**: 发现 pipeline bug 后，不要"修一个封一版"。等全部 bug 修完、full pipeline 跑通后再封板。封板是昂贵操作（创建 locked/ 快照、写 SEAL_SUMMARY、生成 CHECKSUMS），不应频繁执行。

### 教训 4: 策略表现不等于策略强，要看基准

用户问"5年 +51.6% 是不是很差？"。答案取决于基准：
- 510300 同期 -0.1%（5年零收益），策略超额 +51.7pp
- 但如果和美股比（SPY 5年 +80%+），就不突出

**教训**: 永远带基准讨论策略收益。绝对收益无意义，超额收益才有意义。A股 2022-2023 大熊市让所有策略的绝对收益看起来不高。

### 教训 5: composite_1 选择 vs core_4f 的权衡

| | composite_1 | core_4f |
|--|------------|---------|
| HO Return | +53.9% | **+67.4%** |
| HO MDD | **10.8%** | 14.9% |
| Sharpe | **1.38** | 1.09 |
| Rolling 正率 | 61% (边界) | **78%** |
| Composite Score | **#1** | #中等 |

选 composite_1 作为主策略因为：
1. 风险调整后收益更好（Sharpe 1.38 vs 1.09）
2. MDD 更低（10.8% vs 14.9%）— 实盘 MDD 容忍度比回测低
3. Composite score #1（综合 Train+Rolling+Holdout 最均衡）

但 core_4f 绝对收益更高且 rolling 稳定性更强。这是"收益 vs 稳定性"的典型权衡。
**实盘应该两个都 shadow 跑。**

## 检查清单：未来封板流程

```
封板前置检查：
□ Pipeline 全部已知 bug 已修复
□ Full pipeline (WFO→VEC→Rolling→Holdout→BT) 端到端跑通
□ VEC-BT gap 全部 < 10pp（target < 5pp）
□ 四关验证全通过 (Train > 0, Rolling ≥ 60%, Holdout > 0, BT 0 MF)
□ Train → Holdout 方向一致（都为正）

封板执行：
□ frozen_params.py: 新版本 config + CURRENT_VERSION 更新
□ test_frozen_params.py: 版本断言更新 + 新版本测试
□ shadow_strategies.yaml: 更新 shadow 策略
□ sealed_strategies/vX.Y/: 完整封板包
□ 前序版本 DEPRECATED.md
□ make test 全通过
□ CHECKSUMS.sha256 全验证
□ 包大小 < 50MB
```

## 相关文件

- `src/etf_strategy/core/frozen_params.py` — v8.0 配置
- `sealed_strategies/v8.0_20260215/SEAL_SUMMARY.md` — 完整封板文档
- `memory/rules.md` — Rule 24 (VEC metadata), Rule 22 (exec-side hysteresis)
- `memory/2026-02_vec_bt_alignment_fix.md` — BT gap 根因分析
- `memory/2026-02_pipeline_metadata_loss.md` — metadata 丢失模式
