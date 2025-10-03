# 因子筛选系统使用指南

> **版本**: 2.0.0  
> **更新日期**: 2025-10-03  
> **作者**: 量化首席工程师

---

## 📋 快速开始

### 安装依赖

```bash
# 使用uv（推荐）
uv sync --dev

# 或使用pip
pip install -r requirements.txt
```

### 最简示例

```python
from professional_factor_screener import ProfessionalFactorScreener

# 1. 初始化筛选器
screener = ProfessionalFactorScreener()

# 2. 执行筛选
results = screener.screen_factors_comprehensive(
    symbol="0700.HK",
    timeframe="60min"
)

# 3. 获取顶级因子
top_factors = screener.get_top_factors(results, top_n=10)

# 4. 输出结果
for i, factor in enumerate(top_factors, 1):
    print(f"{i}. {factor.name}: {factor.comprehensive_score:.3f}")
```

---

## 🎯 核心概念

### 5维度筛选框架

量化因子筛选系统基于**5维度评估体系**：

| 维度 | 权重 | 核心指标 | 目标 |
|------|------|---------|------|
| **1. 预测能力** | 30% | IC均值、IC IR、IC衰减率 | 因子对未来收益的预测能力 |
| **2. 稳定性** | 25% | 滚动IC均值、滚动IC标准差、IC一致性 | 因子在不同市场环境下的稳定表现 |
| **3. 独立性** | 20% | VIF、最大相关系数、信息增量 | 因子提供的独特信息价值 |
| **4. 实用性** | 15% | 换手率、交易成本、成本效率 | 因子的实际交易可行性 |
| **5. 短周期适应性** | 10% | 反转效应、动量持续性、波动敏感度 | 因子对短周期市场特征的适应能力 |

**综合得分计算**:
```
综合得分 = 0.3×预测能力 + 0.25×稳定性 + 0.2×独立性 + 0.15×实用性 + 0.1×适应性
```

---

### IC（信息系数）

**定义**: 因子值与未来收益的Spearman等级相关系数

**解读**:
- `|IC| > 0.05`: 具有一定预测能力
- `|IC| > 0.10`: 预测能力较强
- `|IC| > 0.15`: 预测能力优秀

**IC IR（信息比率）**:
```
IC IR = IC均值 / IC标准差
```
- 衡量因子预测能力的稳定性
- 越高越好，通常 `IC IR > 1.0` 为优秀

---

### VIF（方差膨胀因子）

**定义**: 衡量因子间多重共线性的指标

**解读**:
- `VIF < 5`: 独立性良好
- `5 ≤ VIF < 10`: 存在一定共线性，可接受
- `VIF ≥ 10`: 严重共线性，需要剔除

**计算公式**:
```
VIF_i = 1 / (1 - R²_i)
```
其中 `R²_i` 是因子i对其他所有因子的回归决定系数。

---

### FDR校正

**Benjamini-Hochberg方法**：控制假发现率（False Discovery Rate）

**为什么需要**:
- 测试217个因子时，即使随机数据也可能有 `217×0.05 ≈ 11` 个因子通过p<0.05检验
- FDR校正确保"显著因子"中真实有效的比例

**自适应阈值**:
- 小样本（n<300）：`alpha = 0.05 / 2 = 0.025`（更严格）
- 大样本（n>1000）：`alpha = 0.05 × 1.2 = 0.06`（稍放宽）

---

## 📖 使用场景

### 场景1: 单股票多周期筛选

**目标**: 对同一股票在不同时间框架下进行因子筛选

```python
from professional_factor_screener import ProfessionalFactorScreener
from enhanced_result_manager import EnhancedResultManager

# 初始化
screener = ProfessionalFactorScreener()
result_manager = EnhancedResultManager()

# 多周期筛选
symbol = "0700.HK"
timeframes = ["15min", "30min", "60min"]

for tf in timeframes:
    print(f"\n{'='*80}")
    print(f"筛选: {symbol} {tf}")
    print(f"{'='*80}")
    
    # 执行筛选
    results = screener.screen_factors_comprehensive(symbol, tf)
    
    # 保存结果
    session_id = result_manager.create_screening_session(
        symbol=symbol,
        timeframe=tf,
        results=results,
        screening_stats=screener.screening_stats,
        config=screener.config
    )
    
    # 统计输出
    significant = sum(1 for m in results.values() if m.is_significant)
    high_score = sum(1 for m in results.values() if m.comprehensive_score > 0.6)
    
    print(f"✅ 总因子: {len(results)}")
    print(f"✅ 显著因子: {significant}")
    print(f"✅ 高分因子: {high_score}")
    print(f"📁 会话ID: {session_id}")
```

---

### 场景2: 多股票批量筛选

**目标**: 批量筛选多只股票，找出共性优质因子

```python
from collections import defaultdict

# 股票池
symbols = ["0700.HK", "9988.HK", "0941.HK", "1810.HK"]
timeframe = "60min"

# 存储每个因子在不同股票上的表现
factor_performance = defaultdict(list)

for symbol in symbols:
    try:
        results = screener.screen_factors_comprehensive(symbol, timeframe)
        
        for factor_name, metrics in results.items():
            factor_performance[factor_name].append({
                'symbol': symbol,
                'score': metrics.comprehensive_score,
                'ic_mean': metrics.ic_mean,
                'is_significant': metrics.is_significant
            })
        
        print(f"✅ {symbol}: {len(results)} factors")
    
    except Exception as e:
        print(f"❌ {symbol}: {e}")

# 找出跨股票稳定的优质因子
stable_factors = []
for factor_name, performances in factor_performance.items():
    # 在所有股票上都显著且高分
    if len(performances) == len(symbols):
        avg_score = sum(p['score'] for p in performances) / len(performances)
        significant_ratio = sum(p['is_significant'] for p in performances) / len(performances)
        
        if avg_score > 0.6 and significant_ratio > 0.8:
            stable_factors.append({
                'name': factor_name,
                'avg_score': avg_score,
                'significant_ratio': significant_ratio
            })

# 排序并输出
stable_factors.sort(key=lambda x: x['avg_score'], reverse=True)

print(f"\n🏆 跨股票稳定优质因子 (Top 10):")
for i, factor in enumerate(stable_factors[:10], 1):
    print(f"{i:2d}. {factor['name']:<30} "
          f"平均得分={factor['avg_score']:.3f} "
          f"显著率={factor['significant_ratio']:.1%}")
```

---

### 场景3: 自定义配置筛选

**目标**: 根据具体策略需求调整筛选参数

```python
from config_manager import ScreeningConfig

# 激进配置 - 追求高预测能力
aggressive_config = ScreeningConfig(
    ic_horizons=[1, 3, 5],  # 只关注短周期
    alpha_level=0.10,  # 放宽显著性要求
    min_sample_size=150,  # 较小样本量
    vif_threshold=10.0,  # 允许一定共线性
    weight_predictive_power=0.50,  # 提高预测能力权重
    weight_stability=0.20,
    weight_independence=0.10,
    weight_practicality=0.10,
    weight_short_term_adaptability=0.10
)

# 保守配置 - 追求稳定性
conservative_config = ScreeningConfig(
    ic_horizons=[5, 10, 20],  # 关注中长周期
    alpha_level=0.01,  # 严格显著性
    min_sample_size=500,  # 大样本量
    vif_threshold=3.0,  # 严格独立性
    weight_predictive_power=0.25,
    weight_stability=0.35,  # 提高稳定性权重
    weight_independence=0.25,
    weight_practicality=0.10,
    weight_short_term_adaptability=0.05
)

# 使用不同配置
screener_aggressive = ProfessionalFactorScreener(config=aggressive_config)
screener_conservative = ProfessionalFactorScreener(config=conservative_config)

# 对比筛选结果
results_aggressive = screener_aggressive.screen_factors_comprehensive("0700.HK", "60min")
results_conservative = screener_conservative.screen_factors_comprehensive("0700.HK", "60min")

print(f"激进策略: {len(results_aggressive)} 因子")
print(f"保守策略: {len(results_conservative)} 因子")
```

---

### 场景4: 因子相关性分析

**目标**: 分析顶级因子之间的相关性，构建因子组合

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 筛选因子
results = screener.screen_factors_comprehensive("0700.HK", "60min")

# 获取顶级因子
top_factors = screener.get_top_factors(results, top_n=20, min_score=0.6)

# 加载因子数据
factors_df = screener.load_factor_data("0700.HK", "60min")

# 提取顶级因子的数据
top_factor_names = [f.name for f in top_factors]
top_factor_data = factors_df[top_factor_names]

# 计算相关性矩阵
correlation_matrix = top_factor_data.corr()

# 可视化
plt.figure(figsize=(14, 12))
sns.heatmap(
    correlation_matrix,
    annot=False,
    cmap='coolwarm',
    center=0,
    square=True,
    linewidths=0.5
)
plt.title('Top 20 Factors Correlation Matrix')
plt.tight_layout()
plt.savefig('factor_correlation.png', dpi=300)

# 找出低相关性的因子组合
low_corr_pairs = []
for i in range(len(top_factor_names)):
    for j in range(i+1, len(top_factor_names)):
        corr = correlation_matrix.iloc[i, j]
        if abs(corr) < 0.3:  # 低相关性阈值
            low_corr_pairs.append({
                'factor1': top_factor_names[i],
                'factor2': top_factor_names[j],
                'correlation': corr
            })

print(f"\n发现 {len(low_corr_pairs)} 对低相关性因子组合")
```

---

## 🛠️ 高级技巧

### 1. 性能优化

**并行处理**:
```python
from concurrent.futures import ProcessPoolExecutor

def screen_symbol(symbol):
    screener = ProfessionalFactorScreener()
    return screener.screen_factors_comprehensive(symbol, "60min")

symbols = ["0700.HK", "9988.HK", "0941.HK"]

with ProcessPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(screen_symbol, symbols))
```

**内存优化**:
```python
# 筛选后立即释放内存
import gc

results = screener.screen_factors_comprehensive("0700.HK", "60min")
top_factors = screener.get_top_factors(results, top_n=10)

# 释放大对象
del results
gc.collect()
```

---

### 2. 异常处理

```python
from pathlib import Path

def safe_screen_factors(symbol, timeframe):
    """安全的因子筛选函数，包含完整异常处理"""
    try:
        screener = ProfessionalFactorScreener()
        results = screener.screen_factors_comprehensive(symbol, timeframe)
        return results
    
    except FileNotFoundError as e:
        print(f"❌ 数据文件不存在: {e}")
        print("建议: 检查data_root路径和文件名")
        return None
    
    except ValueError as e:
        print(f"❌ 数据验证失败: {e}")
        print("建议: 检查数据质量和样本量")
        return None
    
    except MemoryError:
        print(f"❌ 内存不足")
        print("建议: 减少并行度或增加系统内存")
        return None
    
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        import traceback
        traceback.print_exc()
        return None
```

---

### 3. 数据质量检查

```python
# 加载数据前预检查
def check_data_quality(symbol, timeframe):
    screener = ProfessionalFactorScreener()
    
    try:
        # 加载因子数据
        factors = screener.load_factor_data(symbol, timeframe)
        
        # 检查1: 样本量
        sample_size = len(factors)
        print(f"样本量: {sample_size}")
        if sample_size < 200:
            print("⚠️  样本量不足，建议至少200个样本")
        
        # 检查2: 缺失值
        missing_ratio = factors.isnull().sum().sum() / (factors.shape[0] * factors.shape[1])
        print(f"缺失值比例: {missing_ratio:.2%}")
        if missing_ratio > 0.1:
            print("⚠️  缺失值过多，建议数据清洗")
        
        # 检查3: 无穷值
        inf_count = np.isinf(factors.select_dtypes(include=[np.number])).sum().sum()
        print(f"无穷值数量: {inf_count}")
        if inf_count > 0:
            print("⚠️  存在无穷值，将自动处理")
        
        # 检查4: 时间覆盖
        time_span = (factors.index.max() - factors.index.min()).days
        print(f"时间跨度: {time_span} 天")
        
        return True
    
    except Exception as e:
        print(f"❌ 数据质量检查失败: {e}")
        return False

# 使用
if check_data_quality("0700.HK", "60min"):
    results = screener.screen_factors_comprehensive("0700.HK", "60min")
```

---

### 4. 因子筛选流水线

```python
class FactorScreeningPipeline:
    """完整的因子筛选流水线"""
    
    def __init__(self, config=None):
        self.screener = ProfessionalFactorScreener(config=config)
        self.result_manager = EnhancedResultManager()
    
    def run(self, symbol, timeframe):
        """执行完整流水线"""
        print(f"\n{'='*80}")
        print(f"🚀 开始因子筛选流水线: {symbol} {timeframe}")
        print(f"{'='*80}\n")
        
        # 步骤1: 数据质量检查
        print("📊 步骤1: 数据质量检查...")
        if not self._check_data_quality(symbol, timeframe):
            return None
        
        # 步骤2: 执行筛选
        print("\n🔍 步骤2: 执行5维度因子筛选...")
        results = self.screener.screen_factors_comprehensive(symbol, timeframe)
        
        # 步骤3: 结果分析
        print("\n📈 步骤3: 结果分析...")
        self._analyze_results(results)
        
        # 步骤4: 保存结果
        print("\n💾 步骤4: 保存结果...")
        session_id = self.result_manager.create_screening_session(
            symbol=symbol,
            timeframe=timeframe,
            results=results,
            screening_stats=self.screener.screening_stats,
            config=self.screener.config
        )
        
        print(f"\n✅ 流水线执行完成！")
        print(f"📁 会话ID: {session_id}")
        
        return results
    
    def _check_data_quality(self, symbol, timeframe):
        """数据质量检查"""
        try:
            factors = self.screener.load_factor_data(symbol, timeframe)
            sample_size = len(factors)
            missing_ratio = factors.isnull().sum().sum() / (factors.shape[0] * factors.shape[1])
            
            print(f"  ✓ 样本量: {sample_size}")
            print(f"  ✓ 缺失值比例: {missing_ratio:.2%}")
            
            if sample_size < 200:
                print("  ⚠️  样本量不足")
                return False
            
            return True
        except Exception as e:
            print(f"  ❌ 数据质量检查失败: {e}")
            return False
    
    def _analyze_results(self, results):
        """结果分析"""
        total = len(results)
        significant = sum(1 for m in results.values() if m.is_significant)
        high_score = sum(1 for m in results.values() if m.comprehensive_score > 0.6)
        
        print(f"  ✓ 总因子数: {total}")
        print(f"  ✓ 显著因子: {significant} ({significant/total:.1%})")
        print(f"  ✓ 高分因子: {high_score} ({high_score/total:.1%})")
        
        # 输出Top 5
        top_5 = sorted(results.values(), key=lambda x: x.comprehensive_score, reverse=True)[:5]
        print(f"\n  🏆 Top 5 因子:")
        for i, factor in enumerate(top_5, 1):
            print(f"    {i}. {factor.name:<30} {factor.comprehensive_score:.3f}")

# 使用流水线
pipeline = FactorScreeningPipeline()
results = pipeline.run("0700.HK", "60min")
```

---

## 🚨 常见问题

### Q1: 为什么有些因子IC值为0？

**原因**:
1. 因子值常数或近似常数
2. 样本量不足导致无法计算相关性
3. 时间对齐失败

**解决方案**:
```python
# 检查因子值分布
factor_series = factors["your_factor"]
print(f"唯一值数量: {factor_series.nunique()}")
print(f"标准差: {factor_series.std()}")
print(f"缺失值: {factor_series.isnull().sum()}")
```

---

### Q2: VIF计算失败怎么办？

**原因**:
- 因子间完全共线性（相关系数=1）
- 样本量过小

**解决方案**:
```python
# 检查相关性矩阵
corr_matrix = factors.corr()
high_corr_pairs = []

for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.95:
            high_corr_pairs.append((
                corr_matrix.columns[i],
                corr_matrix.columns[j],
                corr_matrix.iloc[i, j]
            ))

print(f"发现 {len(high_corr_pairs)} 对高相关性因子")
```

---

### Q3: 如何处理"样本量不足"错误？

**解决方案**:
```python
# 方案1: 降低最小样本量要求
config = ScreeningConfig(min_sample_size=100)  # 默认200
screener = ProfessionalFactorScreener(config=config)

# 方案2: 使用更长时间跨度的数据
# 检查可用数据
factors = screener.load_factor_data("0700.HK", "60min")
print(f"可用样本量: {len(factors)}")
print(f"时间跨度: {factors.index.min()} 至 {factors.index.max()}")
```

---

### Q4: 如何解读综合得分？

**得分区间**:
- `0.8~1.0`: **优秀** - 核心因子，强烈推荐使用
- `0.6~0.8`: **良好** - 可用因子，建议组合使用
- `0.4~0.6`: **一般** - 可考虑因子，需谨慎验证
- `0.0~0.4`: **较弱** - 不推荐使用

**注意事项**:
- 综合得分需结合统计显著性判断
- 不同市场环境下表现可能不同
- 建议进行样本外验证

---

## 📚 最佳实践

### 1. 因子筛选流程

```
1. 数据准备与质量检查 → 2. 初步筛选（IC>0.03） → 3. 统计显著性检验（FDR校正）
→ 4. VIF独立性检验 → 5. 实用性评估 → 6. 综合打分排序 → 7. 样本外验证
```

### 2. 配置选择建议

| 策略类型 | IC周期 | alpha水平 | VIF阈值 | 样本量 |
|---------|--------|-----------|---------|--------|
| 高频策略 | [1,3,5] | 0.10 | 10.0 | 150+ |
| 中频策略 | [3,5,10] | 0.05 | 5.0 | 200+ |
| 低频策略 | [5,10,20] | 0.01 | 3.0 | 500+ |

### 3. 因子组合构建

**步骤**:
1. 筛选综合得分>0.6的因子
2. 计算因子间相关性矩阵
3. 选择低相关性（|ρ|<0.3）的因子组合
4. 根据预测能力加权构建多因子模型

---

## 🔗 相关文档

- [API_REFERENCE.md](API_REFERENCE.md) - 完整API文档
- [CONTRACT.md](CONTRACT.md) - 系统契约文档
- [README.md](../README.md) - 项目概览

---

**文档维护**: 量化首席工程师  
**最后更新**: 2025-10-03

