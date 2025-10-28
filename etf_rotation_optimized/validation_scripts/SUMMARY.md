# 因子验证框架 - 使用总结

## ✅ 已完成整理

### 📦 核心框架
- ✅ `factor_validator.py` - 标准化验证基础类
  - `FactorValidator` 基类（必须继承）
  - `BatchFactorValidator` 批量验证器
  - 严格执行"横截面 Spearman + T-1"口径
  - 自动 WFO 评估（IS=252, OOS=60, step=20）
  - 准入门槛检查（OOS IC ≥ 0.010，衰减≤50%，失败率≤30%，Top3相关<0.7）

### 📚 文档体系
- ✅ `INDEX.md` - 快速索引（首页导航）
- ✅ `README_AI.md` - **AI/大模型优先阅读**（10分钟快速上手）
- ✅ `README_FULL.md` - 完整文档（高级用法、扩展、FAQ）
- ✅ `README.md` - 原有文档（保留历史）

### 💡 示例代码
- ✅ `example_evaluate_factors.py` - 3个完整案例
  - `ReversalFactor5D` - 短期反转
  - `VolatilitySkew20D` - 波动率偏斜
  - `DollarVolumeAccel10D` - 成交额加速度
  - 批量评估演示

### 📊 历史评估记录
- ✅ `evaluate_candidate_factors.py` - 2025-10-27 评估代码
- ✅ `candidate_factors_evaluation_20251027_185650.csv` - 评估结果
- ✅ `validation_results_20251027_190923.csv` - 示例运行结果

### 🔧 辅助工具
- ✅ `verify_factor_implementation.py` - 单元测试工具
- ✅ `analyze_zero_usage_factors.py` - 零使用频率分析

---

## 🚀 快速开始（复制即用）

### 方式 1: 使用模板快速创建

```python
#!/usr/bin/env python3
"""验证我的新因子"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from validation_scripts.factor_validator import FactorValidator


class MyNewFactor(FactorValidator):
    """我的新因子"""
    
    def compute_factor(self):
        # TODO: 替换为你的因子逻辑
        factor = self.close.pct_change(periods=20, fill_method=None)
        
        # 必须横截面标准化
        return self._cross_sectional_standardize(factor)


def main():
    # 数据路径（自动查找最新）
    results_dir = Path(__file__).parent.parent / "results"
    ohlcv_dir = sorted((results_dir / "cross_section" / "20251027").glob("*"))[-1] / "ohlcv"
    factors_dir = sorted((results_dir / "factor_selection" / "20251027").glob("*"))[-1] / "standardized"
    
    # 执行评估
    validator = MyNewFactor(str(ohlcv_dir), str(factors_dir))
    result = validator.evaluate('MY_NEW_FACTOR')
    
    # 保存结果
    if result['PASS_ALL']:
        print("\n✅ 通过准入门槛，可考虑集成生产系统")
    else:
        print("\n❌ 未通过准入门槛，拒绝集成")


if __name__ == "__main__":
    main()
```

**运行**：
```bash
cd /Users/zhangshenshen/深度量化0927
python etf_rotation_optimized/validation_scripts/validate_my_factor.py
```

### 方式 2: 直接修改示例代码

```bash
# 复制示例文件
cd /Users/zhangshenshen/深度量化0927/etf_rotation_optimized/validation_scripts
cp example_evaluate_factors.py validate_my_factor.py

# 编辑 validate_my_factor.py，修改因子逻辑

# 运行
cd /Users/zhangshenshen/深度量化0927
python etf_rotation_optimized/validation_scripts/validate_my_factor.py
```

---

## 📖 推荐阅读顺序

### 对于 AI / 大模型
1. **先读**：`README_AI.md`（10分钟）
2. **参考**：`example_evaluate_factors.py`（完整代码）
3. **运行测试**：确认环境可用

### 对于人类开发者
1. **快速入门**：`README_AI.md`（快速上手）
2. **深入学习**：`README_FULL.md`（详细文档）
3. **参考示例**：`example_evaluate_factors.py`
4. **查看历史**：`evaluate_candidate_factors.py` + CSV 结果

---

## 🎯 核心设计原则

### 1. 严格准入门槛（宁缺毋滥）
- OOS IC ≥ 0.010（超越现有最弱因子）
- 衰减比 ≤ 50%（IS-OOS 稳定性）
- 失败率 ≤ 30%（避免"看天吃饭"）
- Top3 相关 < 0.7（避免冗余）

### 2. 离线验证（不污染生产）
- 先验证，后集成
- 独立评估，无副作用
- 完整记录，可追溯

### 3. 标准化流程（统一口径）
- 横截面 Spearman IC
- 严格 T-1 对齐（因子[t-1] 预测收益[t]）
- WFO 滚动窗口（IS=252, OOS=60, step=20）
- 与生产系统完全一致

### 4. 自动化报告（透明评估）
- 详细日志输出
- 结构化结果保存
- 通过/拒绝明确判定

---

## 📊 历史案例总结

### 2025-10-27 评估：反转/波动/成交额因子

**背景**：收到 AI 生成的因子建议，严格评估是否适用于 ETF 轮动系统

**评估因子**：
1. `REVERSAL_FACTOR_5D` - 5日短期反转
2. `VOLATILITY_SKEW_20D` - 波动率偏斜
3. `DOLLAR_VOLUME_ACCELERATION_10D` - 美元成交额加速度

**结果**：**全部未通过准入门槛**

| 因子 | OOS IC | 衰减比 | 失败率 | Top3相关 | 结论 |
|------|--------|--------|--------|----------|------|
| REVERSAL_5D | 0.0109✅ | **1000%❌** | **45%❌** | -0.28✅ | **拒绝** |
| VOL_SKEW_20D | **0.0068❌** | 17%✅ | **40%❌** | -0.10✅ | **拒绝** |
| ACCEL_10D | **-0.0056❌** | 12%✅ | **57%❌** | 0.04✅ | **拒绝** |

**核心发现**：
- 反转逻辑在 ETF 不适用（篮子分散弱化反转效应）
- 波动率偏斜横截面区分度不足（行业/风格主导）
- 成交额加速度与收益负相关且太弱（炒作→回撤）

**教训**：
1. IS-OOS 符号反转 = 虚假正 IC（过拟合噪音）
2. OOS IC 不足 = 预测力低于现有基准
3. 失败率高 = 不稳定，无法依赖

**详细报告**：
- 代码：`evaluate_candidate_factors.py`
- 结果：`candidate_factors_evaluation_20251027_185650.csv`

---

## 🛠️ 常见任务速查

### 评估单个因子
```bash
cd /Users/zhangshenshen/深度量化0927
python etf_rotation_optimized/validation_scripts/example_evaluate_factors.py
```

### 批量评估多个因子
修改 `example_evaluate_factors.py` 中的 `validators` 列表，添加你的因子类

### 查看历史评估
```bash
cat validation_scripts/candidate_factors_evaluation_20251027_185650.csv
```

### 调试因子实现
```bash
python etf_rotation_optimized/validation_scripts/verify_factor_implementation.py
```

### 分析零使用频率因子
```bash
python etf_rotation_optimized/validation_scripts/analyze_zero_usage_factors.py
```

---

## ⚠️ 重要提醒

### 必须遵守的规则
1. **横截面标准化**：必须调用 `_cross_sectional_standardize()`
2. **T-1 对齐**：框架自动处理，不要手动 shift
3. **返回格式**：必须返回 DataFrame（横截面）
4. **有效样本数**：每日至少 5 个非 NaN 样本

### 典型错误示例
```python
# ❌ 错误：未标准化
def compute_factor(self):
    return self.close.pct_change(periods=20)

# ❌ 错误：手动 shift（不需要）
def compute_factor(self):
    ret = self.close.pct_change(periods=20).shift(1)
    return self._cross_sectional_standardize(ret)

# ✅ 正确
def compute_factor(self):
    ret = self.close.pct_change(periods=20, fill_method=None)
    return self._cross_sectional_standardize(ret)
```

---

## 📞 获取帮助

### 查看文档
- **快速上手**：`README_AI.md`
- **详细文档**：`README_FULL.md`
- **快速索引**：`INDEX.md`

### 参考示例
- **完整案例**：`example_evaluate_factors.py`
- **历史评估**：`evaluate_candidate_factors.py`

### 运行测试
```bash
cd /Users/zhangshenshen/深度量化0927
python etf_rotation_optimized/validation_scripts/example_evaluate_factors.py
```

预期输出：3 个因子全部未通过准入门槛（与历史评估一致）

---

## 📅 维护记录

- **2025-10-27 19:09**：创建标准化验证框架
- **2025-10-27 19:09**：完成文档体系（INDEX + README_AI + README_FULL）
- **2025-10-27 19:09**：示例代码测试通过（3个因子，全部拒绝）
- **2025-10-27 19:09**：框架验收完成，可投入使用

---

## 🎉 总结

### 已完成工作
✅ 核心框架代码（`factor_validator.py`）  
✅ 完整文档体系（3个 README + INDEX）  
✅ 示例代码（3个完整案例）  
✅ 历史评估记录（代码 + CSV 结果）  
✅ 测试验收（运行正常，结果一致）

### 下一步建议
1. **后续加因子**：使用本框架先离线验证，通过后再集成生产
2. **保持克制**：严守准入门槛，宁缺毋滥
3. **记录归档**：每次评估保存独立报告（CSV/JSON）
4. **定期复盘**：回顾拒绝的因子，避免重复错误

---

**维护者**：深度量化团队  
**最后更新**：2025-10-27  
**状态**：✅ 生产就绪
