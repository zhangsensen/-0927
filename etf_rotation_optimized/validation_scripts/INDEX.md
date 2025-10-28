# 因子验证框架 - 快速索引

## 📂 文件清单

| 文件 | 用途 | 目标用户 |
|------|------|---------|
| **README_AI.md** | 🤖 AI 快速入门指南 | **大模型优先阅读** |
| **README_FULL.md** | 📖 完整使用文档 | 需要详细参考的开发者 |
| **README.md** | 📝 原有文档 | 历史保留 |
| **factor_validator.py** | 🏗️ 核心框架代码 | 必须导入的基础类 |
| **example_evaluate_factors.py** | 💡 3个完整示例 | 快速上手参考 |
| **evaluate_candidate_factors.py** | 📊 历史评估案例 | 2025-10-27 评估记录 |
| **verify_factor_implementation.py** | 🧪 单元测试工具 | 因子实现调试 |
| **analyze_zero_usage_factors.py** | 🔍 因子使用分析 | 因子池清理 |

---

## 🎯 快速导航

### 👉 我是 AI / 大模型
**直接阅读**：`README_AI.md`（10 分钟快速上手）

### 👉 我想评估新因子
1. 阅读：`README_AI.md`（快速入门）
2. 参考：`example_evaluate_factors.py`（完整示例）
3. 运行：复制模板代码，修改因子逻辑

### 👉 我想了解详细用法
**阅读**：`README_FULL.md`（高级用法、扩展、FAQ）

### 👉 我想看历史案例
**查看**：
- `evaluate_candidate_factors.py`（代码）
- `candidate_factors_evaluation_20251027_185650.csv`（结果）
- `README_FULL.md` 中的"历史评估案例"章节

---

## ⚡ 3 步快速验证（极简版）

### Step 1: 创建因子类

```python
from validation_scripts.factor_validator import FactorValidator

class MyFactor(FactorValidator):
    def compute_factor(self):
        factor = self.close.pct_change(periods=20)  # 你的因子逻辑
        return self._cross_sectional_standardize(factor)  # 必须标准化
```

### Step 2: 加载数据并运行

```python
from pathlib import Path

# 数据路径
results_dir = Path("etf_rotation_optimized/results")
ohlcv_dir = sorted((results_dir / "cross_section" / "20251027").glob("*"))[-1] / "ohlcv"
factors_dir = sorted((results_dir / "factor_selection" / "20251027").glob("*"))[-1] / "standardized"

# 执行评估
validator = MyFactor(str(ohlcv_dir), str(factors_dir))
result = validator.evaluate('MY_FACTOR')
```

### Step 3: 查看结果

```python
if result['PASS_ALL']:
    print("✅ 通过准入")
else:
    print("❌ 拒绝")
```

---

## 📊 准入门槛（一览表）

| 指标 | 门槛 | 含义 |
|------|------|------|
| OOS IC | ≥ 0.010 | 样本外预测力 |
| 衰减比 | ≤ 50% | IS→OOS 稳定性 |
| 失败率 | ≤ 30% | OOS IC<0 的窗口占比 |
| Top3 相关 | < 0.7 | 与现有强因子的冗余度 |

**全部满足** → `PASS_ALL = True` → 可考虑集成生产系统

---

## 🛠️ 核心工具方法

| 方法 | 功能 |
|------|------|
| `_cross_sectional_standardize(df)` | 横截面标准化（每日去均值/标准差） |
| `_compute_cross_sectional_ic(...)` | 横截面 IC 计算（T-1 对齐） |
| `_run_wfo_evaluation(...)` | WFO 滚动窗口评估（55 窗口） |
| `_check_correlation_with_top3(df)` | Top3 因子冗余检查 |

---

## 📝 典型因子模板

### 动量因子
```python
def compute_factor(self):
    ret = self.close.pct_change(periods=20, fill_method=None)
    return self._cross_sectional_standardize(ret)
```

### 波动率因子
```python
def compute_factor(self):
    vol = self.returns.rolling(window=20).std()
    return self._cross_sectional_standardize(vol)
```

### 价格位置因子
```python
def compute_factor(self):
    high_20 = self.high.rolling(window=20).max()
    low_20 = self.low.rolling(window=20).min()
    position = (self.close - low_20) / (high_20 - low_20 + 1e-8)
    return self._cross_sectional_standardize(position)
```

### 成交量因子
```python
def compute_factor(self):
    dollar_vol = self.close * self.volume
    avg_vol = dollar_vol.rolling(window=20).mean()
    return self._cross_sectional_standardize(avg_vol)
```

---

## 🚨 常见错误与修复

### 错误 1: 未横截面标准化
```python
# ❌ 错误
def compute_factor(self):
    return self.close.pct_change(periods=20)

# ✅ 正确
def compute_factor(self):
    ret = self.close.pct_change(periods=20)
    return self._cross_sectional_standardize(ret)
```

### 错误 2: 手动 T-1 对齐（不需要）
```python
# ❌ 不需要
def compute_factor(self):
    ret = self.close.pct_change(periods=20).shift(1)  # 多余
    return self._cross_sectional_standardize(ret)

# ✅ 正确（框架自动处理）
def compute_factor(self):
    ret = self.close.pct_change(periods=20)
    return self._cross_sectional_standardize(ret)
```

### 错误 3: 返回值格式不正确
```python
# ❌ 错误：返回 Series
def compute_factor(self):
    return self.close.mean(axis=1)  # 单列数据

# ✅ 正确：返回 DataFrame（横截面）
def compute_factor(self):
    ret = self.close.pct_change(periods=20)
    return self._cross_sectional_standardize(ret)
```

---

## 📞 获取帮助

1. **查看示例**：`example_evaluate_factors.py`（3 个完整案例）
2. **阅读文档**：`README_AI.md`（快速）或 `README_FULL.md`（详细）
3. **运行测试**：
   ```bash
   cd /Users/zhangshenshen/深度量化0927
   python etf_rotation_optimized/validation_scripts/example_evaluate_factors.py
   ```

---

## 📅 更新日志

- **2025-10-27**：创建标准化验证框架
- **2025-10-27**：评估反转/波动/成交额 3 个因子（全部拒绝）
- **2025-10-27**：发布 AI 快速入门指南 + 完整文档

---

**维护者**：深度量化团队  
**最后更新**：2025-10-27
