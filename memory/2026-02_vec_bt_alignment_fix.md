# LESSON: VEC-BT 滞后链对齐 — 信号态 vs 执行态

> 最后更新: 2026-02-14
> 涉及文件: `src/etf_strategy/auditor/core/engine.py`, `scripts/batch_bt_backtest.py`

---

## 事件：+25.7pp VEC-BT Holdout Gap 根因定位与修复

### 背景

v7.0 封存策略 #1 (6F: ADX+AMIHUD+PP120+PP20+SHARE_ACCEL+SLOPE) 出现严重的 VEC-BT 不对齐：

| 指标 | VEC | BT (修复前) | BT (修复后) |
|------|-----|-------------|------------|
| Holdout 收益 | +25.2% | +50.9% | **+24.4%** |
| 交易次数 | 62 | 95 | **64** |
| Gap | — | **+25.7pp** | **-0.8pp** |

### 错误的根因假说（shadow commission）

**最初假说**：BT 的 shadow accounting 使用 `sizing_commission_rate = max(a_share, qdii) = 50bp`，
而 VEC 使用逐 ticker 的 `cost_arr`（A股=20bp, QDII=50bp）。既然全部交易都是 A 股，
BT 每笔多估 30bp，积累导致现金偏差 → 交易分歧。

**实现了 per-ticker cost_rates**：
- engine.py 新增 `cost_rates` 参数
- sell/buy 都用逐 ticker 费率
- batch_bt_backtest.py 构建 cost_rates 字典传入

**结果：零效果**。Gap 完全没变。

**原因**：shadow sizing 的佣金差异只影响单次仓位估算 ~0.3%，不足以改变任何交易决策。
broker.getcash() 每次 rebalance 都重置为真实现金，shadow 误差不累积。

### 真正的根因：Exp4.1 信号态反馈环

**关键消融实验**：
```
Hysteresis OFF: VEC +20.1% (195t), BT +18.9% (197t), gap -1.3pp, trade_diff=2
Hysteresis ON:  VEC +25.2% (62t),  BT +50.9% (95t),  gap +25.7pp, trade_diff=33
```
→ 滞后机制是 gap 的唯一放大器。

**架构差异**：

```
VEC 内核（正确）:
  h_mask[n] = holdings[n] > 0.0     ← 执行态（实际持仓）
  hold_days = hold_days_arr          ← 执行态（实际持有天数）

BT 引擎 Exp4.1（错误）:
  hmask[i] = _signal_portfolio[t]    ← 信号态（上一次滞后输出）
  hdays = _signal_hold_days          ← 信号态（信号层持有天数）
```

**自引用反馈环**：`_signal_portfolio` 是滞后函数的**输出**，然后在下一次 rebalance 作为**输入**
传回滞后函数。当 BT 的执行态（broker 实际成交）与信号态有微小差异（浮点精度、整手凑整），
信号态不会自我修正，而是将偏差放大：

```
信号态说"持有A" → 滞后保护A → 执行态可能因凑整微调了B
→ 下次信号态仍基于"持有A" → 执行态已经偏离
→ 一个不同的 swap 决策 → 链式分歧
→ 33次多余交易 → +25.7pp gap
```

VEC 不存在这个问题，因为 VEC 的 h_mask 始终从 `holdings[]`（执行态）构建。

### 修复

将 `_compute_rebalance_targets()` 中的 hmask/hdays 构建从信号态切换为执行态：

```python
# 修复前（信号态）:
hmask = np.array([t in self._signal_portfolio for t in etf_list])
hdays = np.array([self._signal_hold_days.get(t, 0) for t in etf_list])

# 修复后（执行态）:
hmask = np.zeros(N, dtype=np.bool_)
for i, t in enumerate(etf_list):
    if self.shadow_holdings.get(t, 0.0) > 0:
        hmask[i] = True
hdays = np.zeros(N, dtype=np.int64)
for t, d in self._hold_days.items():
    if t in etf_list:
        hdays[etf_list.index(t)] = d
```

---

## 核心教训

### 教训 1: 路径依赖系统中信号态 ≠ 执行态

在有滞后/动量/状态的策略中，信号层和执行层会因微小差异（浮点精度、整手凑整、现金余额）
而逐渐分离。**必须用执行态（actual holdings）驱动状态依赖逻辑**，否则状态漂移不可控。

### 教训 2: 消融实验是定位根因的最快手段

不要猜根因。用消融法：
1. 关掉滞后 → gap 消失 → 根因在滞后
2. 切换 hmask 来源 → gap 消失 → 根因在 hmask 来源

比逐行 debug 快 10x。

### 教训 3: shadow accounting 对 gap 的影响远小于预期

`sizing_commission_rate` 只影响单次仓位估算（~0.3%），不影响累积状态。
BT 的 `broker.getcash()` 每次 rebalance 重置为真实现金，shadow 误差不累积。
**不要在 shadow 层找根因，看状态层。**

### 教训 4: 路径依赖的链式放大效应

Hysteresis OFF 时 float/int 差异只造成 1-2pp（正常）。
Hysteresis ON 时同样的微小差异被链式放大到 25+pp，因为一次不同的 swap
改变了未来所有的持仓路径。

**红旗判断标准更新**：
- Hysteresis OFF gap > 5pp → 引擎 bug
- Hysteresis ON gap > 10pp → 状态追踪 bug（大概率不是 float/int 差异）

### 教训 5: 先看数据模式，后看代码

最快的诊断路径：
1. 跑消融（hyst ON/OFF）→ 5分钟 → 定位到滞后
2. 比较月度收益 → 5分钟 → 确认分歧起点
3. 猴子补丁切换状态源 → 10分钟 → 确认根因
4. 正式修复 + 测试 → 20分钟

总计 40 分钟。如果直接逐行 debug engine.py（600+ 行）→ 可能花数小时。

---

## 检查清单：未来 VEC-BT 对齐诊断

```
□ 1. 先跑 Hyst OFF 对比 → gap < 5pp 说明引擎基本对齐
□ 2. 再跑 Hyst ON 对比 → gap > 10pp 说明状态追踪有分歧
□ 3. 检查 hmask 来源：是执行态 (shadow_holdings) 还是信号态 (_signal_portfolio)?
□ 4. 检查 hdays 来源：是执行态 (_hold_days) 还是信号态 (_signal_hold_days)?
□ 5. VEC 和 BT 的 h_mask/hdays 是否从同类状态构建？
□ 6. 月度对比找分歧起点 → 缩小到具体 rebalance 日
□ 7. 猴子补丁验证假说 → 确认后再改正式代码
```

---

## 附：per-ticker cost_rates 改动

虽然对 gap 无效，但 cost_rates 改动**技术上更准确**（A股 20bp, QDII 50bp 分别计算），
保留作为正确性改进。`cost_rates=None` 时回退到原有 `sizing_commission_rate` 行为。
