# 🐛 关键Bug：Set遍历不确定性导致回测结果不可复现

**发现日期**: 2025-11-27  
**严重程度**: 🔴 Critical  
**影响范围**: `scripts/full_wfo_backtest_v2.py`, 历史所有回测结果  

---

## 问题描述

旧回测系统使用Python `set` 存储目标持仓，并直接遍历set进行买入操作。当资金不足以买入所有目标ETF时，**哪个ETF先被买入取决于set的遍历顺序**。

Python set的遍历顺序是**哈希依赖的、不确定的**，导致：
- 同一策略在不同运行环境下产生不同结果
- 结果不可复现、不可追溯

---

## 根因代码

```python
# scripts/full_wfo_backtest_v2.py 原问题代码
target_indices = set(valid_indices[top_k_local].tolist())
...
for idx in target_indices:  # ← set遍历顺序不确定！
    if idx in holdings:
        continue
    # 买入逻辑 - 先遍历到的先买
```

当资金不足时，例如目标是 `{0, 7, 9}`：
- 实际遍历可能是 `[0, 9, 7]` (hash顺序)
- 如果只够买2个，就买了 `[0, 9]`，漏掉 `7`

---

## 量化影响

| 回测版本 | 策略 | Return | 可复现 |
|---------|------|--------|--------|
| 原set遍历 | ADX+CORR+PP+SHARPE | 96.0% | ❌ 不可复现 |
| 确定性顺序 | ADX+CORR+PP+SHARPE | **72.6%** | ✅ 可复现 |

**差距**: 23.4个百分点的收益差异完全来自于随机的买入顺序！

---

## 修复方案

改用**确定性的得分降序**买入：

```python
# 修复后代码
if np.sum(valid_mask) >= POS_SIZE:
    valid_indices = np.where(valid_mask)[0]
    valid_scores = combined_score[valid_mask]
    # 按得分从高到低排序，确保确定性顺序
    sorted_order = np.argsort(valid_scores)[::-1]  # 降序
    target_list = valid_indices[sorted_order[:POS_SIZE]].tolist()
    target_indices = set(target_list)
else:
    target_list = []
    target_indices = set()

# Buy (按得分从高到低顺序，确保确定性)
for idx in target_list:  # 使用有序列表而非set
    ...
```

---

## 修复文件

1. `scripts/full_wfo_backtest_v2.py` - 改用 `target_list` 有序列表
2. `etf_rotation_optimized/run_unified_wfo.py` - 新系统已使用确定性顺序

---

## 教训总结

1. **禁止直接遍历set进行有副作用的操作**
2. **涉及资金分配的循环必须用确定性顺序**
3. **所有回测结果必须可复现验证**

---

## 验证命令

```bash
# 运行两次，结果必须完全一致
.venv/bin/python scripts/full_wfo_backtest_v2.py
# 保存结果1
.venv/bin/python scripts/full_wfo_backtest_v2.py  
# 保存结果2
# diff 结果1 结果2 应该完全一致
```

---

## 相关修复

- ADX_14D TR计算bug (2025-11-27): `np.maximum` 替代 `pd.concat().max(axis=1)`
- VORTEX_14D TR计算bug (2025-11-27): 同上

---

## 修复后的权威结果 (2025-11-27 最终版)

### Bug 2: NaN 价格未处理导致 MaxDD 异常

修复前 89% 策略的 MaxDD 为 nan，原因是 close_prices 中存在 nan 值。

**修复方案**: 在 Mark-to-Market 和 Sell 时检查 nan，使用 entry_price 作为备用。

### 修复后 Top 10 策略 (确定性回测)
| Rank | Return | MaxDD | WR | 策略 |
|------|--------|-------|-----|------|
| 1 | **134.2%** | -19.3% | 51.0% | ADX_14D + OBV_SLOPE_10D + PRICE_POSITION_20D + PV_CORR_20D + SHARPE_RATIO_20D |
| 2 | 123.4% | -18.6% | 51.0% | ADX_14D + OBV_SLOPE_10D + PRICE_POSITION_120D + PV_CORR_20D + SHARPE_RATIO_20D |
| 3 | 116.1% | -18.1% | 52.4% | ADX_14D + OBV_SLOPE_10D + PRICE_POSITION_20D + RET_VOL_20D + SHARPE_RATIO_20D |
| 4 | 115.7% | -18.0% | 52.4% | ADX_14D + OBV_SLOPE_10D + PRICE_POSITION_120D + RET_VOL_20D + SHARPE_RATIO_20D |
| 5 | 101.7% | -18.8% | 50.8% | ADX_14D + OBV_SLOPE_10D + PRICE_POSITION_120D + PRICE_POSITION_20D + SHARPE_RATIO_20D |
| 6 | 100.9% | -19.2% | 50.5% | ADX_14D + CALMAR_RATIO_60D + MAX_DD_60D + PV_CORR_20D + SHARPE_RATIO_20D |
| 7 | 99.8% | -12.7% | 52.6% | ADX_14D + MAX_DD_60D + OBV_SLOPE_10D + PV_CORR_20D + SHARPE_RATIO_20D |
| 8 | 93.2% | -17.8% | 54.1% | CORRELATION_TO_MARKET_20D + MAX_DD_60D + OBV_SLOPE_10D + PV_CORR_20D + SHARPE_RATIO_20D |
| 9 | 92.2% | -19.0% | 52.6% | ADX_14D + OBV_SLOPE_10D + PV_CORR_20D + RET_VOL_20D + SHARPE_RATIO_20D |
| 10 | 91.1% | -18.5% | 52.9% | ADX_14D + OBV_SLOPE_10D + RET_VOL_20D |

### 统计对比
- 总策略数: 12,269
- 收益 > 100%: 6个
- 收益 > 90%: 10个  
- 收益 > 80%: 30个
- 最高收益: **134.2%**
- 平均收益: 提升至约 30%+

### 可复现性验证
✅ 两次运行结果**完全一致**，验证通过 (max diff = 0.0)
