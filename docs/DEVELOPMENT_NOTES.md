# 开发注意事项与陷阱

> **最后更新**: 2025-11-28

---

## 🚨 关键陷阱

### 1. Set 遍历不确定性

**问题**：Python set 遍历顺序不确定，导致回测结果不可复现

```python
# ❌ 错误
for ticker in target_set:  # 顺序不确定！
    buy(ticker)

# ✅ 正确
for ticker in sorted(target_set):  # 确定顺序
    buy(ticker)
```

**影响范围**：所有涉及 set 遍历的调仓逻辑

---

### 2. 未来函数泄露

**问题**：使用当日数据做决策

```python
# ❌ 错误
signal = timing_values[t]      # 使用 T 日信号
score = factors_3d[t, n, :]    # 使用 T 日因子

# ✅ 正确
signal = timing_values[t-1]    # 使用 T-1 日信号
score = factors_3d[t-1, n, :]  # 使用 T-1 日因子
```

**检查方法**：
```bash
grep -n "timing_values\[t\]" **/*.py
grep -n "factors_3d\[t," **/*.py
```

---

### 3. BT 资金计算时序

**问题**：卖出订单提交后，broker 现金未立即更新

```python
# ❌ 错误
current_equity = self.broker.getvalue()  # 卖出前净值
available = current_equity * timing_ratio

# ✅ 正确
cash_after_sells = self.broker.getcash()
for ticker, shares in sells:
    cash_after_sells += shares * price * (1 - comm)
available = cash_after_sells * timing_ratio
```

---

### 4. 浮点精度问题

**问题**：浮点比较导致合法买入被拒绝

```python
# ❌ 错误
if cost > available:  # 可能因浮点误差拒绝
    reject()

# ✅ 正确
if cost > available * 1.0001:  # 留 0.01% 容差
    reject()
```

---

## 📋 开发规范

### 修改前检查清单

- [ ] 是否涉及调仓逻辑？→ 需要 VEC/BT 双重验证
- [ ] 是否涉及信号计算？→ 检查是否使用 T-1 数据
- [ ] 是否涉及 set/dict 遍历？→ 使用 sorted()
- [ ] 是否涉及资金计算？→ 检查时序正确性

### 修改后验证

```bash
# 1. 清理缓存
python scripts/cache_cleaner.py

# 2. 单组合验证
python scripts/full_vec_bt_comparison.py --combo "..."

# 3. 批量验证（可选）
python scripts/batch_vec_backtest.py
python scripts/batch_bt_backtest.py
```

---

## 🔧 调试技巧

### 逐日对比

```python
# 在 full_vec_bt_comparison.py 中添加
print(f"Day {t}: VEC={vec_equity:.2f}, BT={bt_equity:.2f}")
```

### 检查调仓日

```python
rebalance_set = generate_rebalance_schedule(T, LOOKBACK, FREQ)
print(f"调仓日: {sorted(list(rebalance_set))[:10]}...")
```

### 检查择时信号

```python
print(f"T={t}: raw={timing_raw[t]:.4f}, shifted={timing_shifted[t]:.4f}")
```

---

## 📁 文件修改影响范围

| 文件 | 影响 | 需要验证 |
|------|------|----------|
| `core/utils/rebalance.py` | 所有引擎 | 全量 VEC/BT 对比 |
| `core/market_timing.py` | 择时信号 | 全量验证 |
| `core/precise_factor_library_v2.py` | 因子计算 | 清理缓存 + 验证 |
| `strategy_auditor/core/engine.py` | BT 引擎 | VEC/BT 对比 |
| `scripts/batch_vec_backtest.py` | VEC 引擎 | VEC/BT 对比 |

---

## 🚀 性能优化建议

### Numba 编译

```python
from numba import njit

@njit
def _compute_score(factors_3d, t, n, factor_indices):
    score = 0.0
    for f_idx in factor_indices:
        val = factors_3d[t-1, n, f_idx]
        if not np.isnan(val):
            score += val
    return score
```

### 向量化优先

```python
# ❌ 慢
for i in range(len(arr)):
    result[i] = arr[i] * 2

# ✅ 快
result = arr * 2
```

### 缓存策略

```python
# 大数据加载使用 lru_cache
from functools import lru_cache

@lru_cache(maxsize=1)
def load_factors():
    return pd.read_parquet("factors.parquet")
```

---

## 📝 代码风格

### 命名规范

- **文件**: `snake_case.py`
- **类**: `PascalCase`
- **函数/变量**: `snake_case`
- **常量**: `UPPER_SNAKE_CASE`

### 注释规范

```python
def rebalance(t: int, holdings: np.ndarray) -> np.ndarray:
    """
    执行调仓逻辑
    
    Args:
        t: 当前时间索引
        holdings: 当前持仓数组
        
    Returns:
        更新后的持仓数组
        
    Note:
        - 使用 T-1 日信号
        - 确保 set 遍历有序
    """
```

---

## 🔗 相关文档

- [架构设计](ARCHITECTURE.md)
- [VEC/BT 对齐历史](VEC_BT_ALIGNMENT_HISTORY_AND_FIXES.md)
