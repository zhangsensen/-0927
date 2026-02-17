# WHEN/HOW 维度研究 — 完结报告

> 日期: 2026-02-17
> 状态: **已关闭 — 三方向全部 KILL**
> 前置知识: 读 CLAUDE.md + memory/MEMORY.md
> 详细数据: `docs/research/when_how_dimension_research_20260217.md`
> 脚本: `scripts/research_when_how_stage01.py`

---

## 背景

v8.0 封板后，WHAT 维度（选哪个 ETF）已系统性穷尽。两个未探索的优化轴：
- **WHEN**: 什么时候该信任选股（市场环境识别）
- **HOW**: 两个策略怎么配合（ensemble/切换）

## 实验结果

### Stage 0: Ensemble 失败相关性 (HOW)

| 指标 | Train (16Q) | Holdout (9M) |
|------|-------------|--------------|
| Pearson rho | **0.586** | 0.637 |
| composite_1 win rate | 68.8% | 55.6% |
| core_4f win rate | 56.2% | 66.7% |
| P(both_fail) | 0.188 | 0.111 |
| P(both_fail\|one_fail) | 0.333 | 0.167 |

**判定**: MARGINAL (rho 0.5-0.7 区间)
- 不是完全冗余（rho<0.7），有一定互补性
- 但也不是强互补（rho>0.5），改善幅度有限
- 季度 blend Sharpe 仅提升 +0.023

### Stage 1: 截面收益离散度 (WHEN)

| 检验 | 结果 | 判定 |
|------|------|------|
| 正交性: disp vs regime_vol | rho=0.538 (20D), 0.369 (5D) | **KILL** (>0.5) |
| 预测力: disp → strategy ret | Pearson<0.05, p>0.3 (全部) | **KILL** (零信号) |
| Quartile 单调性 | NONE (5D/20D × Train/HO 全部) | **KILL** |

**关键发现 (Rule 33)**: 截面收益离散度 ⊂ 市场波动率。高波动本质上产生高离散——这是数学恒等式，不是经济信号。

### Stage 2a: Ensemble Capital Split

| 指标 | composite_1 (PS=2) | Blend (2×PS=1) | 变化 |
|------|-------------------|----------------|------|
| Sharpe | 1.577 | 0.363 | **-77%** |
| Max DD | 10.8% | 16.8% | +6pp |
| Total Return | 136.1% | 26.4% | -110pp |

**关键发现 (Rule 32)**: PS=2→PS=1 导致 Sharpe 暴跌75-85%。策略在低于优化目标的 POS_SIZE 下不可用。Capital Split 路径彻底关闭。

---

## 结论

```
WHAT (选什么)  → 已穷尽 (Phase 1 + moneyflow + 得分离散度)
WHEN (何时信任) → 已关闭 (离散度 ⊂ 波动率 + 零预测力)
HOW  (怎么组合) → 已关闭 (Capital Split: PS降级崩塌)
```

**v8.0 = 当前数据和方法论下的天花板。**

---

## 未来方向 (当前不可执行): 策略级条件切换

### 概念

不分资金（避开 Rule 32），而是**整体切换**使用哪个策略：

```
某个信号 → 高值时: 全仓用 composite_1 (PS=2)
          → 低值时: 全仓用 core_4f (PS=2)
```

### 为什么没被关闭

1. **两策略有互补性** (rho=0.586 < 1.0)：合适的切换信号理论上可利用
2. **保持 PS=2** → 避开 Rule 32 的 PS 降级崩塌
3. **只增加一个决策参数** (切换阈值) → 过拟合风险可控

### 为什么目前不可执行

**没有可用的切换信号**：
- 收益离散度 → KILLED (Rule 33, 与 regime gate 同维度)
- 得分离散度 → KILLED (rho<0.08, 零信号)
- 因子近期 IC → 49 ETF 信噪比太低 (见下方风险分析)
- regime gate 本身 → 不区分两策略（两者都含 regime gate）

### 何时可以重启

当有**新的正交市场环境信号**可用时：
- 北向资金净流入趋势（编码外资情绪，vs 波动率正交）
- 期权 IV 曲面斜率（前瞻预期，vs 历史波动率正交）
- 行业集中度指标（截面结构，可能与波动率部分正交）

**判定标准**:
1. 与 regime_vol 的 |rho| < 0.3 (严格正交)
2. 与两策略收益的 rank correlation 方向相反 (一高一低)
3. Train/Holdout 方向一致 (Rule 4)

---

## 新增规则

- **Rule 32**: POS_SIZE 不可降级。PS=2→PS=1: Sharpe -75%。Ensemble 必须在目标 PS 下独立验证。
- **Rule 33**: 截面收益离散度 ⊂ 市场波动率 (rho=0.538)。不是独立 WHEN 信号。

---

## 行动建议

1. **停止所有研究性探索** — WHAT/WHEN/HOW 三维度天花板已确认
2. **进入纯运维模式** — shadow 监控 + 日常信号 + 数据更新
3. **等待数据突破** — 新正交数据源可得时，可考虑"策略切换"框架（非活跃研究）
4. **Shadow 验证时间表** — 8-12 周后 (~2026-04-15) 评估 S1→v8.0 切换
