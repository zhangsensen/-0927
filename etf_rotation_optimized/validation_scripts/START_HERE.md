# 🚀 开始使用因子验证框架

> **快速上手指南 - 5分钟学会评估新因子**

---

## 📖 第一次使用？从这里开始

### 🤖 如果你是 AI / 大模型
👉 **直接阅读**：[README_AI.md](README_AI.md)（10分钟快速上手）

### 👨‍💻 如果你是开发者
👉 **快速入门**：[README_AI.md](README_AI.md)  
👉 **详细文档**：[README_FULL.md](README_FULL.md)  
👉 **完整索引**：[INDEX.md](INDEX.md)

---

## ⚡ 3步快速验证新因子

### 步骤 1: 创建你的因子类

```python
from validation_scripts.factor_validator import FactorValidator

class MyFactor(FactorValidator):
    def compute_factor(self):
        # 你的因子逻辑（示例：20日动量）
        factor = self.close.pct_change(periods=20, fill_method=None)
        
        # 必须横截面标准化
        return self._cross_sectional_standardize(factor)
```

### 步骤 2: 运行评估

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

### 步骤 3: 查看结果

```python
if result['PASS_ALL']:
    print("✅ 通过准入，可考虑集成")
else:
    print("❌ 拒绝")
```

---

## 📊 准入门槛（所有条件必须满足）

| 指标 | 门槛 | 含义 |
|------|------|------|
| **OOS IC** | ≥ 0.010 | 样本外预测力 |
| **衰减比** | ≤ 50% | IS→OOS 稳定性 |
| **失败率** | ≤ 30% | OOS IC<0 的窗口占比 |
| **Top3相关** | < 0.7 | 与现有强因子的冗余度 |

---

## 📁 文件导航

| 文件 | 用途 | 优先级 |
|------|------|--------|
| [README_AI.md](README_AI.md) | AI快速入门 | ⭐⭐⭐⭐⭐ |
| [example_evaluate_factors.py](example_evaluate_factors.py) | 完整示例代码 | ⭐⭐⭐⭐⭐ |
| [factor_validator.py](factor_validator.py) | 核心框架代码 | ⭐⭐⭐⭐ |
| [INDEX.md](INDEX.md) | 快速索引 | ⭐⭐⭐⭐ |
| [README_FULL.md](README_FULL.md) | 详细文档 | ⭐⭐⭐ |
| [SUMMARY.md](SUMMARY.md) | 使用总结 | ⭐⭐⭐ |

---

## 🧪 运行测试（验证环境）

```bash
cd /Users/zhangshenshen/深度量化0927
python etf_rotation_optimized/validation_scripts/example_evaluate_factors.py
```

**预期输出**：3个因子评估报告，全部未通过准入门槛（与历史评估一致）

---

## 💡 典型因子模板

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

---

## ⚠️ 常见错误

### ❌ 错误：未横截面标准化
```python
def compute_factor(self):
    return self.close.pct_change(periods=20)  # 缺少标准化
```

### ✅ 正确：横截面标准化
```python
def compute_factor(self):
    ret = self.close.pct_change(periods=20, fill_method=None)
    return self._cross_sectional_standardize(ret)
```

---

## 📞 获取帮助

1. **查看示例**：[example_evaluate_factors.py](example_evaluate_factors.py)
2. **阅读文档**：[README_AI.md](README_AI.md)（快速）或 [README_FULL.md](README_FULL.md)（详细）
3. **查看索引**：[INDEX.md](INDEX.md)（导航）

---

## 🎯 核心原则

✅ **先离线验证，后集成代码** - 不污染生产系统  
✅ **严守准入门槛，宁缺毋滥** - 保持系统简洁  
✅ **记录评估过程，可追溯** - 每次评估生成独立报告

---

**维护者**：深度量化团队  
**最后更新**：2025-10-27  
**状态**：✅ 生产就绪
