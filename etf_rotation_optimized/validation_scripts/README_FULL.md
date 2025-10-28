# 因子验证工具集 - 完整使用指南

本目录提供**完整的因子离线验证框架**，用于在不影响生产系统的前提下，严格评估新因子的有效性。

---

## 📚 目录结构

```
validation_scripts/
├── README_FULL.md                     # 本文档（完整使用指南）
├── README.md                          # 原有文档（保留）
├── factor_validator.py                # 因子验证基础框架（核心类）✨ NEW
├── example_evaluate_factors.py        # 示例：如何评估 3 个候选因子 ✨ NEW
├── evaluate_candidate_factors.py      # 历史案例：2025-10-27 评估反转/波动/成交额因子
├── verify_factor_implementation.py    # 因子实现正确性验证（单元测试类）
└── analyze_zero_usage_factors.py      # 零使用频率因子分析工具
```

---

## 🚀 快速开始（3 步评估新因子）

### Step 1: 创建因子类（继承 `FactorValidator`）

```python
from validation_scripts.factor_validator import FactorValidator

class MyNewFactor(FactorValidator):
    """你的新因子"""
    
    def compute_factor(self) -> pd.DataFrame:
        """
        实现因子计算逻辑
        
        Returns:
            pd.DataFrame: 横截面标准化后的因子值
                - Index: 时间序列（与 OHLCV 对齐）
                - Columns: 资产代码
                - Values: 标准化后的因子值
        """
        # 示例：20日动量
        ret_20d = self.close.pct_change(periods=20, fill_method=None)
        
        # 横截面标准化（必须）
        factor_std = self._cross_sectional_standardize(ret_20d)
        
        return factor_std
```

### Step 2: 运行评估

```python
# 加载数据（自动查找最新的 OHLCV 与标准化因子目录）
validator = MyNewFactor(ohlcv_dir, existing_factors_dir)

# 执行完整评估（WFO + 准入门槛检查）
result = validator.evaluate('MY_NEW_FACTOR')
```

### Step 3: 查看结果

```
📊 评估结果
================================================================================
因子名称: MY_NEW_FACTOR
  - IS IC 均值: 0.0156
  - OOS IC 均值: 0.0123
  - IC 衰减比: 21.15%
  - 失败窗口率: 25.45%
  - Top3 中位相关: 0.4521

🎯 准入门槛检查
================================================================================
  ✅ OOS IC ≥ 0.01: 0.0123
  ✅ 衰减比 ≤ 50%: 21.15%
  ✅ 失败率 ≤ 30%: 25.45%
  ✅ Top3相关 < 0.7: 0.4521

================================================================================
✅ 通过准入门槛！
================================================================================
```

---

## 📖 核心组件说明

### 1. `FactorValidator` 基础类

**作用**：提供标准化的因子评估流程

**核心方法**：
- `compute_factor()`: **抽象方法**，子类必须实现（计算因子值）
- `evaluate(factor_name)`: **主方法**，执行完整评估流程
- `_cross_sectional_standardize()`: 横截面标准化工具
- `_compute_cross_sectional_ic()`: 严格 T-1 对齐的横截面 IC 计算
- `_run_wfo_evaluation()`: WFO 滚动窗口评估
- `_check_correlation_with_top3()`: 与 Top3 因子冗余检查

**准入门槛（类属性，可覆盖）**：
```python
MIN_OOS_IC = 0.010          # OOS 平均 RankIC ≥ 0.010
MAX_DECAY_RATIO = 0.50      # IS→OOS 衰减比例 ≤ 50%
MAX_FAILURE_RATIO = 0.30    # 失败窗口比例 ≤ 30%
MAX_TOP3_CORR = 0.70        # 与 Top3 因子相关性 < 0.7
```

**WFO 配置（沿用生产系统）**：
```python
IS_WINDOW = 252   # 样本内窗口（1年交易日）
OOS_WINDOW = 60   # 样本外窗口（3个月）
STEP = 20         # 滚动步长（1个月）
```

**数据访问（自动加载）**：
- `self.close`, `self.high`, `self.low`, `self.open`, `self.volume`：OHLCV 数据
- `self.returns`：收益率（已计算，T-1 对齐）
- `self.top3_factors`：Top3 稳定因子（`CALMAR_RATIO_60D`, `PRICE_POSITION_120D`, `CMF_20D`）

---

### 2. `BatchFactorValidator` 批量验证器

**作用**：一次性评估多个因子，输出汇总报告

**用法示例**：
```python
from validation_scripts.factor_validator import BatchFactorValidator

validators = [
    ReversalFactor5D(ohlcv_dir, factors_dir),
    VolatilitySkew20D(ohlcv_dir, factors_dir),
    DollarVolumeAccel10D(ohlcv_dir, factors_dir)
]

factor_names = ['REVERSAL_5D', 'VOL_SKEW_20D', 'ACCEL_10D']

batch = BatchFactorValidator(ohlcv_dir, factors_dir)
results_df = batch.evaluate_batch(validators, factor_names)
```

**输出**：
```
📊 批量评估结果汇总
================================================================================
       factor_name  oos_ic_mean  ic_decay_ratio  failure_ratio  PASS_ALL
0     REVERSAL_5D     0.010879       10.005363       0.454545     False
1    VOL_SKEW_20D     0.006826        0.171411       0.400000     False
2      ACCEL_10D    -0.005638        0.123171       0.563636     False

✅ 通过准入: 0 个
❌ 未通过: 3 个
================================================================================
```

---

## 🔧 高级用法

### 自定义准入门槛

```python
class StrictValidator(FactorValidator):
    # 覆盖类属性
    MIN_OOS_IC = 0.015          # 更严格的 IC 要求
    MAX_DECAY_RATIO = 0.30      # 更严格的衰减要求
    MAX_FAILURE_RATIO = 0.20    # 更严格的失败率要求
    
    def compute_factor(self):
        # ... 你的因子计算逻辑
        pass
```

### 访问详细窗口数据

```python
result = validator.evaluate('MY_FACTOR')

# 获取每个 WFO 窗口的 IC 值
for window in result['wfo_windows']:
    print(f"Window {window['is_start']}-{window['oos_end']}: "
          f"IS IC={window['is_ic']:.4f}, OOS IC={window['oos_ic']:.4f}")
```

### 使用工具方法

```python
# 横截面标准化
factor_raw = self.close.pct_change(periods=10)
factor_std = self._cross_sectional_standardize(factor_raw)

# 计算特定窗口的 IC
ic = self._compute_cross_sectional_ic(
    factor=my_factor,
    returns=self.returns,
    window_start=100,
    window_end=200
)
```

---

## 📝 完整示例（从零开始评估新因子）

假设你想评估一个**30日趋势强度因子**：

### 1. 创建验证脚本 `validate_trend_strength.py`

```python
#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from validation_scripts.factor_validator import FactorValidator


class TrendStrength30D(FactorValidator):
    """30日趋势强度因子"""
    
    def compute_factor(self) -> pd.DataFrame:
        """
        计算逻辑：
        - 线性回归斜率 / 残差标准差
        - 衡量趋势的"信噪比"
        """
        window = 30
        factor = pd.DataFrame(index=self.close.index, columns=self.close.columns)
        
        for col in self.close.columns:
            prices = self.close[col]
            
            for i in range(window, len(prices)):
                y = prices.iloc[i-window:i].values
                x = np.arange(window)
                
                # 线性回归
                slope, intercept = np.polyfit(x, y, 1)
                fitted = slope * x + intercept
                residuals = y - fitted
                
                # 趋势强度 = 斜率 / 残差标准差
                trend_strength = slope / (residuals.std() + 1e-8)
                factor.iloc[i, factor.columns.get_loc(col)] = trend_strength
        
        # 横截面标准化
        return self._cross_sectional_standardize(factor)


def main():
    # 数据路径（自动查找最新）
    results_dir = Path(__file__).parent.parent / "results"
    cross_section_base = results_dir / "cross_section" / "20251027"
    latest_cross = sorted(cross_section_base.glob("*"))[-1]
    ohlcv_dir = latest_cross / "ohlcv"
    
    factor_sel_base = results_dir / "factor_selection" / "20251027"
    latest_factor = sorted(factor_sel_base.glob("*"))[-1]
    factors_dir = latest_factor / "standardized"
    
    # 创建验证器
    validator = TrendStrength30D(str(ohlcv_dir), str(factors_dir))
    
    # 执行评估
    result = validator.evaluate('TREND_STRENGTH_30D')
    
    # 保存结果
    import json
    from datetime import datetime
    
    output_file = Path(__file__).parent / f"trend_strength_30d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # 简化输出（去除窗口详细数据）
    summary = {k: v for k, v in result.items() if k != 'wfo_windows'}
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n💾 结果已保存: {output_file}")


if __name__ == "__main__":
    main()
```

### 2. 运行评估

```bash
cd /Users/zhangshenshen/深度量化0927
python etf_rotation_optimized/validation_scripts/validate_trend_strength.py
```

### 3. 解读结果

如果通过所有门槛，可考虑纳入生产系统；否则直接归档，避免污染因子池。

---

## 🎯 评估标准详解

### 1. **OOS IC ≥ 0.010**
- **含义**：样本外平均横截面 Spearman IC 不低于 0.010
- **理由**：系统现有最弱因子（ADX_14D）的 OOS IC ≈ 0.0081，新因子需超越基准
- **典型失败案例**：VOLATILITY_SKEW_20D (0.0068)

### 2. **衰减比 ≤ 50%**
- **含义**：(IS IC - OOS IC) / IS IC ≤ 0.50
- **理由**：IS→OOS 衰减过大说明过拟合噪音，泛化能力差
- **典型失败案例**：REVERSAL_5D (1000%，IS≈0 但 OOS 突然变正)

### 3. **失败窗口率 ≤ 30%**
- **含义**：OOS IC < 0 的窗口数占比不超过 30%
- **理由**：稳定性要求，避免"看天吃饭"型因子
- **典型失败案例**：DOLLAR_VOLUME_ACCEL_10D (57%)

### 4. **Top3 相关性 < 0.7**
- **含义**：与 `CALMAR_RATIO_60D`, `PRICE_POSITION_120D`, `CMF_20D` 的中位相关性低于 0.7
- **理由**：避免因子冗余，确保增量信息
- **计算方式**：展开为时间×资产向量，全局 Spearman 相关

---

## 📊 历史评估案例

### 案例 1：2025-10-27 反转/波动/成交额因子评估

**背景**：收到 AI 生成的因子建议，需验证是否适用于 ETF 轮动系统

**评估因子**：
1. `REVERSAL_FACTOR_5D`：5日短期反转
2. `VOLATILITY_SKEW_20D`：波动率偏斜（下跌日波动/上涨日波动）
3. `DOLLAR_VOLUME_ACCELERATION_10D`：美元成交额加速度

**结果**：全部未通过准入门槛

| 因子 | OOS IC | 衰减比 | 失败率 | Top3相关 | 结论 |
|------|--------|--------|--------|----------|------|
| REVERSAL_5D | 0.0109✅ | 1000%❌ | 45%❌ | -0.28✅ | 拒绝 |
| VOL_SKEW_20D | 0.0068❌ | 17%✅ | 40%❌ | -0.10✅ | 拒绝 |
| ACCEL_10D | -0.0056❌ | 12%✅ | 57%❌ | 0.04✅ | 拒绝 |

**核心发现**：
- 反转逻辑在 ETF 资产类不适用（ETF 篮子分散导致反转效应弱化）
- 波动率偏斜对 ETF 横截面区分度不足（行业/风格主导）
- 成交额加速度与收益负相关且幅度太弱（可能反映短期炒作→回撤）

**详细报告**：`candidate_factors_evaluation_20251027_185650.csv`

**代码**：`evaluate_candidate_factors.py`

---

## 🛠️ 工具脚本说明

### `verify_factor_implementation.py`
- **用途**：单元测试类，验证因子计算正确性
- **方法**：使用模拟数据，比对预期输出
- **适用场景**：因子实现调试阶段

### `analyze_zero_usage_factors.py`
- **用途**：分析 WFO 中零使用频率的因子
- **输出**：因子选择统计、相关性矩阵
- **适用场景**：因子池清理、冗余分析

---

## ⚠️ 注意事项

### 1. **横截面标准化必须执行**
```python
# ❌ 错误：未标准化
def compute_factor(self):
    return self.close.pct_change(periods=20)

# ✅ 正确：横截面标准化
def compute_factor(self):
    ret_20d = self.close.pct_change(periods=20)
    return self._cross_sectional_standardize(ret_20d)
```

### 2. **T-1 对齐由框架自动处理**
- `compute_factor()` 只需返回因子值，无需手动 shift
- IC 计算时框架会自动执行 T-1 对齐（因子[t-1] 预测收益[t]）

### 3. **NaN 处理**
- 允许前期有 NaN（窗口预热期）
- IC 计算时自动跳过 NaN 样本
- 但确保有效样本数 ≥ 5（否则该日 IC 不计入）

### 4. **数据对齐**
- 确保返回的 DataFrame index 与 `self.close.index` 一致
- 确保 columns 与 `self.close.columns` 一致

---

## 📞 扩展与贡献

### 添加新工具函数

如需在 `FactorValidator` 中添加通用工具（如 winsorize、分位数归一化），可直接扩展基类：

```python
class FactorValidator(ABC):
    # ... 现有代码 ...
    
    def _winsorize(self, factor: pd.DataFrame, lower=0.01, upper=0.99) -> pd.DataFrame:
        """横截面 winsorize（每日）"""
        lower_bound = factor.quantile(lower, axis=1)
        upper_bound = factor.quantile(upper, axis=1)
        
        return factor.clip(lower=lower_bound, upper=upper_bound, axis=0)
```

### 自定义评估指标

继承并覆盖 `_print_report()` 方法，添加自定义输出：

```python
class MyValidator(FactorValidator):
    def _print_report(self, result: Dict):
        super()._print_report(result)  # 调用父类报告
        
        # 添加自定义指标
        logger.info(f"\n📌 自定义指标:")
        logger.info(f"  - IC 标准差: {np.std([w['oos_ic'] for w in result['wfo_windows']]):.4f}")
```

---

## 🔗 相关文档

- **生产系统主流程**：`../scripts/step3_run_wfo.py`（WFO 优化）
- **因子库**：`../core/factor_calculator.py`（18个基础因子）
- **准入门槛来源**：`../core/constrained_walk_forward_optimizer.py`（IC 计算口径）

---

## 📅 更新日志

- **2025-10-27**：创建标准化验证框架，评估反转/波动/成交额因子（全部拒绝）
- **2025-10-27**：发布 `FactorValidator` 基础类与 `BatchFactorValidator` 批量工具

---

## 💡 最佳实践

1. **先离线验证，再集成代码**：避免因子池污染
2. **严守准入门槛**：宁缺毋滥，保持系统简洁
3. **记录评估过程**：每次评估生成独立报告文件
4. **参数扫描**：若单参数失败，尝试小幅调整（如 5D→10D→20D）
5. **状态分层**：若全局 IC 不足，可尝试牛/熊/震荡分层（但需额外研究）

---

**维护者**：深度量化团队  
**最后更新**：2025-10-27  
**联系方式**：如有疑问请查看 `example_evaluate_factors.py` 完整示例
