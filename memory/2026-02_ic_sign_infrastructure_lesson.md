# IC-Sign 基础设施教训：水管通了，水一直都在

> 日期: 2026-02-16
> 类型: 教训 / 元认知
> 关联: Rule 24, Phase 1-3 IC-sign fix, v8.0 封板

## 事件

2026-02-14 完成 IC-sign 三阶段修复后，v8.0 的两个封板策略都包含负向因子：
- **composite_1** (5F): MARGIN_BUY_RATIO(-1), SHARE_CHG_5D(-1) — 5 因子中 2 个负向
- **core_4f** (4F): MARGIN_CHG_10D(-1), SHARE_CHG_20D(-1) — 4 因子中 2 个负向

而在修复前（v5.0~v7.0），这些因子要么被排除，要么被反向使用，导致"Non-OHLCV all failed"的错误结论。

## 核心问题

管线有一个隐含假设：**所有因子 "higher = better"**。

具体表现：
1. WFO 用 `abs(IC)` 做权重 → **丢弃了方向信息**
2. VEC/BT 求和 `score = Σ factor_i` → 假设所有因子正向
3. 结果：SHARE_CHG_5D (IC = -0.101, stability = 1.00) 被当正向用
4. 系统选出"高 SHARE_CHG"的 ETF → 这些 ETF 恰恰表现差
5. WFO 认为这些因子"没用" → 排除出候选

**这不是策略问题，是基础设施 bug。**

## 教训

### 核心认知

> **v8.0 策略能胜出，不是因为发现了新因子，而是同样的因子池，管线能正确处理了。**
>
> 修复前：系统只能使用 "higher=better" 的因子（约一半的搜索空间）
> 修复后：正向、负向因子都能正确使用（完整的搜索空间）
>
> 水管通了，水一直都在。

### 推论

1. **"因子无效"的结论必须先排除管线 bug** — 在断言因子没用之前，检查方向处理是否正确
2. **基础设施 bug 的影响是乘法级的** — 一个方向 bug 同时废掉所有负向因子，不是损失一两个因子，是损失整个方向维度
3. **Non-OHLCV 因子并非无效** — v8.0 两个策略各有 2 个非OHLCV 负向因子，都是 stability=1.00 的强信号
4. **搜索空间的完整性比搜索算法的优化更重要** — 50% 搜索空间被 bug 切掉，任何优化都无法弥补

### 检查清单

- [ ] 因子评估前：确认 factor_signs 是否正确传递到 VEC/BT
- [ ] "因子无效"结论前：检查该因子的 IC 方向和 stability
- [ ] 新因子接入时：确认 WFO→VEC→BT 全链路的方向处理
- [ ] 管线重构时：回归测试必须包含已知负向因子（如 SHARE_CHG_5D）

## 统计验证修正 (2026-02-16)

Phase 2 验证中记录的统计数据存在虚高，已从归档数据重新计算：

| 指标 | 原声称 | 实际值 (Active 22F) | 说明 |
|------|--------|---------------------|------|
| IC rho | 0.966 | **0.813** | Pearson, active factors |
| ICIR rho | 0.959 | **0.502** | Pearson; Spearman=0.294 (p=0.18, 不显著) |
| 方向一致率 | 96% (54/56) | **91% (20/22)** | 原值包含 28 个 neutral 因子充水 |
| A/B +51.2pp | 无归档数据 | — | 临时脚本结果未保存 |

**教训**: 临时计算的统计声称必须归档到可复现的输出文件，否则不写入 memory。

## 相关文件

- `src/etf_strategy/core/combo_wfo_optimizer.py` — stability gating (L831-838)
- `scripts/batch_vec_backtest.py` — sign * weight pre-multiply (L1870-1881)
- `scripts/batch_bt_backtest.py` — ICIR-weighted scoring (L359-394)
- `sealed_strategies/v8.0_20260215/SEAL_SUMMARY.md` — factor_signs 记录
- `results/factor_direction_stability_20260214_013357/direction_analysis.csv` — 原始验证数据
