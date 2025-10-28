# 🤖 AI 快速入门指南 - 因子验证框架

> **本文档专为大模型设计，快速理解如何使用因子验证工具**

---

## 📋 核心原则

**目的**：离线验证新因子，不污染生产系统  
**口径**：日频横截面 Spearman + T-1 对齐  
**门槛**：OOS IC ≥ 0.010，衰减≤50%，失败率≤30%，Top3相关<0.7

---

## ⚡ 3 步快速验证

### Step 1: 继承 `FactorValidator` 并实现 `compute_factor()`

```python
from validation_scripts.factor_validator import FactorValidator

class MyFactor(FactorValidator):
    def compute_factor(self) -> pd.DataFrame:
        # 计算因子
        factor = self.close.pct_change(periods=20)  # 示例：20日收益率
        
        # 横截面标准化（必须）
        return self._cross_sectional_standardize(factor)
```

### Step 2: 加载数据并运行

```python
from pathlib import Path

# 数据路径（自动查找最新）
results_dir = Path("etf_rotation_optimized/results")
ohlcv_dir = sorted((results_dir / "cross_section" / "20251027").glob("*"))[-1] / "ohlcv"
factors_dir = sorted((results_dir / "factor_selection" / "20251027").glob("*"))[-1] / "standardized"

# 创建验证器并执行
validator = MyFactor(str(ohlcv_dir), str(factors_dir))
result = validator.evaluate('MY_FACTOR')
```

### Step 3: 解读结果

```python
if result['PASS_ALL']:
    print("✅ 通过准入，可考虑集成")
else:
    print("❌ 拒绝，原因见日志")
```

---

## 🔑 关键信息

### 数据访问（自动加载）

| 属性 | 说明 | 类型 |
|------|------|------|
| `self.close` | 收盘价 | DataFrame (1399×43) |
| `self.high` | 最高价 | DataFrame |
| `self.low` | 最低价 | DataFrame |
| `self.open` | 开盘价 | DataFrame |
| `self.volume` | 成交量 | DataFrame |
| `self.returns` | 收益率（已 pct_change） | DataFrame |
| `self.top3_factors` | Top3 因子（字典） | Dict[str, DataFrame] |

### 工具方法

| 方法 | 说明 |
|------|------|
| `_cross_sectional_standardize(df)` | 横截面标准化（每日去均值/标准差） |
| `_compute_cross_sectional_ic(...)` | 计算横截面 IC（T-1 对齐） |
| `_run_wfo_evaluation(...)` | WFO 滚动窗口评估 |
| `_check_correlation_with_top3(df)` | 与 Top3 因子冗余检查 |

### 准入门槛（类属性）

```python
MIN_OOS_IC = 0.010          # OOS IC 下限
MAX_DECAY_RATIO = 0.50      # 衰减比上限
MAX_FAILURE_RATIO = 0.30    # 失败率上限
MAX_TOP3_CORR = 0.70        # Top3 相关性上限
```

**可覆盖**：子类中重新定义即可

---

## 📚 完整示例（复制即用）

```python
#!/usr/bin/env python3
"""验证 20 日动量因子"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from validation_scripts.factor_validator import FactorValidator


class Momentum20D(FactorValidator):
    """20日动量因子"""
    
    def compute_factor(self):
        # 20日收益率
        ret_20d = self.close.pct_change(periods=20, fill_method=None)
        
        # 横截面标准化
        return self._cross_sectional_standardize(ret_20d)


def main():
    # 数据路径
    results_dir = Path(__file__).parent.parent / "results"
    ohlcv_dir = sorted((results_dir / "cross_section" / "20251027").glob("*"))[-1] / "ohlcv"
    factors_dir = sorted((results_dir / "factor_selection" / "20251027").glob("*"))[-1] / "standardized"
    
    # 执行评估
    validator = Momentum20D(str(ohlcv_dir), str(factors_dir))
    result = validator.evaluate('MOMENTUM_20D')
    
    # 保存结果
    import json
    from datetime import datetime
    
    output = Path(__file__).parent / f"momentum_20d_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(output, 'w') as f:
        json.dump({k: v for k, v in result.items() if k != 'wfo_windows'}, f, indent=2, default=str)
    
    print(f"\n💾 结果: {output}")


if __name__ == "__main__":
    main()
```

**运行**：
```bash
cd /Users/zhangshenshen/深度量化0927
python etf_rotation_optimized/validation_scripts/validate_momentum_20d.py
```

---

## 📊 批量评估（多个因子）

```python
from validation_scripts.factor_validator import BatchFactorValidator

# 创建多个验证器
validators = [
    Momentum20D(ohlcv_dir, factors_dir),
    Reversal10D(ohlcv_dir, factors_dir),
    Volatility30D(ohlcv_dir, factors_dir)
]

factor_names = ['MOM_20D', 'REV_10D', 'VOL_30D']

# 批量评估
batch = BatchFactorValidator(ohlcv_dir, factors_dir)
results_df = batch.evaluate_batch(validators, factor_names)

# 保存汇总
results_df.to_csv('batch_results.csv', index=False)
```

---

## ⚠️ 必须遵守的规则

### 1. 横截面标准化（必须）

```python
# ❌ 错误
def compute_factor(self):
    return self.close.pct_change(periods=20)

# ✅ 正确
def compute_factor(self):
    ret = self.close.pct_change(periods=20)
    return self._cross_sectional_standardize(ret)
```

### 2. T-1 对齐（框架自动处理）

- **你不需要**手动 `shift(1)`
- IC 计算时框架会自动对齐：因子[t-1] 预测收益[t]

### 3. 返回值要求

```python
def compute_factor(self) -> pd.DataFrame:
    """
    Returns:
        pd.DataFrame:
            - Index: 必须与 self.close.index 一致
            - Columns: 必须与 self.close.columns 一致
            - Values: 横截面标准化后的因子值（允许前期 NaN）
    """
```

### 4. 有效样本数

- 每日横截面至少 5 个有效样本（非 NaN）
- 否则该日 IC 不计入统计

---

## 🎯 评估标准速查

| 指标 | 门槛 | 说明 |
|------|------|------|
| **OOS IC** | ≥ 0.010 | 样本外预测力下限 |
| **衰减比** | ≤ 50% | (IS IC - OOS IC) / IS IC |
| **失败率** | ≤ 30% | OOS IC < 0 的窗口占比 |
| **Top3 相关** | < 0.7 | 与 CALMAR/PRICE_POS/CMF 的中位相关 |

**全部满足** → `PASS_ALL = True` → 可考虑集成

---

## 📁 文件结构

```
validation_scripts/
├── factor_validator.py          # 核心框架（必读）
├── example_evaluate_factors.py  # 3个因子示例（参考）
├── README_AI.md                 # 本文档
└── README_FULL.md               # 完整文档（详细版）
```

---

## 🔗 快速参考

- **参考示例**：`example_evaluate_factors.py`（3 个完整案例）
- **详细文档**：`README_FULL.md`（高级用法、扩展、FAQ）
- **历史案例**：`evaluate_candidate_factors.py`（2025-10-27 评估反转/波动/成交额因子，全部拒绝）

---

## 💡 常见因子模板

### 动量类

```python
def compute_factor(self):
    ret = self.close.pct_change(periods=20, fill_method=None)
    return self._cross_sectional_standardize(ret)
```

### 波动率类

```python
def compute_factor(self):
    vol = self.returns.rolling(window=20).std()
    return self._cross_sectional_standardize(vol)
```

### 趋势类

```python
def compute_factor(self):
    slope = (self.close - self.close.shift(20)) / 20
    return self._cross_sectional_standardize(slope)
```

### 成交量类

```python
def compute_factor(self):
    dollar_vol = self.close * self.volume
    avg_vol = dollar_vol.rolling(window=20).mean()
    return self._cross_sectional_standardize(avg_vol)
```

### 价格位置类

```python
def compute_factor(self):
    high_20 = self.high.rolling(window=20).max()
    low_20 = self.low.rolling(window=20).min()
    position = (self.close - low_20) / (high_20 - low_20 + 1e-8)
    return self._cross_sectional_standardize(position)
```

---

## 🚨 典型失败案例（警示）

### 案例：REVERSAL_FACTOR_5D（2025-10-27）

```
OOS IC: 0.0109 ✅ 看似达标
衰减比: 1000% ❌ 严重过拟合
失败率: 45% ❌ 近一半窗口无效

原因：IS IC ≈ 0（样本内无效），OOS 突然变正（噪音碰撞）
结论：虚假正 IC，典型数据窥探陷阱，拒绝
```

**教训**：不能只看 OOS IC，必须综合评估 IS-OOS 一致性与稳定性

---

## ✅ 检查清单

使用前请确认：

- [ ] 继承 `FactorValidator` 类
- [ ] 实现 `compute_factor()` 方法
- [ ] 横截面标准化（调用 `_cross_sectional_standardize()`）
- [ ] 返回值 Index/Columns 与 OHLCV 对齐
- [ ] 运行评估（调用 `evaluate()`）
- [ ] 解读结果（检查 `PASS_ALL` 与具体指标）
- [ ] 保存报告（JSON/CSV）

---

**最后更新**：2025-10-27  
**维护者**：深度量化团队  
**如有疑问**：查看 `example_evaluate_factors.py` 完整示例
