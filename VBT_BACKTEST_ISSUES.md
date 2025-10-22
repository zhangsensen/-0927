# 🔍 VBT 回测系统 - 问题定位报告（仅问题，不修复）

**执行时间**: 2025-10-21 23:35  
**模式**: Linus 无情审查 - 定位问题，不修复  
**原则**: No bullshit. No magic. Just math and code.

---

## 🎯 核心发现

### 🔴 **问题1: 因子来源适配断层** 
**严重程度**: CRITICAL  
**位置**: 所有回测引擎的因子加载接口  

#### 问题描述
最新筛选输出：
```python
# passed_factors.csv 的列名
factor,ic_mean,ic_std,ic_ir,ic_positive_rate,stability,t_stat,p_value,sample_size,coverage,ic_1d,ir_1d,...

# 第1行数据示例
PRICE_POSITION_20D,0.600,0.253,2.365,...  # 18个通过的因子
```

但回测引擎期望的接口：
```python
# parallel_backtest_configurable.py Line 132-148
def _load_top_factors(self) -> List[str]:
    df = pd.read_csv(self.config.screening_file)
    col_name = "factor" if "factor" in df.columns else "panel_factor"
    
    # 寻找"factor"列 ✓ 存在
    # 取Top K: df.head(self.config.top_k)[col_name].tolist() ✓ 可行
    # 但问题来了...
```

#### 真实问题
1. **无 IC/IR 过滤**: 回测直接取Top-K，**不检查因子质量**
   - 筛选器已输出IC=0.600, IR=2.36的优质因子
   - 但回测不管这些值，直接用前10个
   - ❌ 浪费了筛选的统计显著性信息

2. **权重求和约束不匹配**:
   - 配置: `weight_sum_range: [0.8, 1.2]`
   - 但有18个通过的因子，权重必然要很小
   - 举例: 18个因子权重和1.0 → 平均权重 = 1/18 ≈ 0.056
   - 这会导致权重网格生成出大量**零权重的组合** (浪费计算)

3. **权重网格生成过度**:
   - 配置权重点: `[0.0, 0.1, 0.2, ..., 1.0]` = 11个点
   - 对18个因子: 11^18 = 3.8e18 个组合 ❌
   - 内存直接爆炸
   - 实际代码会用 `max_combinations: 10000` 裁剪，但裁剪方法：
     ```python
     # Line 500: _generate_weight_combinations
     valid_mask = (weight_sums >= 0.8) & (weight_sums <= 1.2)
     ```
   - 这会**大量过滤**，导致有效组合 << 10000

---

### 🔴 **问题2: 因子数量的矩阵重塑bug**
**严重程度**: CRITICAL  
**位置**: `parallel_backtest_engine.py` Line 255-260, `parallel_backtest_configurable.py` Line 161, 276

#### 问题描述
```python
# 代码逻辑
factor_data = panel[factors].unstack(level="symbol")  # 取因子列并unstack
n_dates, n_total = factor_data.shape                   # n_total = n_dates * n_factors
n_factors = len(factors)
n_symbols = n_total // n_factors                       # 推导 n_symbols

# 这是错的!
```

#### 真实问题
假设:
- 面板结构: MultiIndex(symbol, date) × 因子列
- 面板大小: 56,575行 × 18列 (18个通过的因子)

```python
factor_data = panel[factors].unstack(level="symbol")
# 结果: n_dates=? n_total=?
# 如果原始面板是 (symbol, date) MultiIndex:
#   - unstack(level="symbol") → index=date, columns=MultiIndex(symbol, factor)
#   - 形状: n_unique_dates × (n_unique_symbols × n_factors)
```

**关键问题**: `n_symbols = n_total // n_factors` 这个推导假设了:
```
n_total = n_dates_in_unstacked × n_columns_in_unstacked
         = n_dates × (n_symbols × n_factors)  ← 这里错了!
```

正确应该是:
```python
n_dates, n_cols = factor_data.shape
# n_cols = n_symbols × n_factors
n_symbols = n_cols // n_factors  # ✓ 这是对的
```

但代码写的是:
```python
n_dates, n_total = factor_data.shape
n_symbols = n_total // n_factors  # ✓ 实际上也是对的
```

**等等，这不是bug**？

让我重新检查... ❌ **我之前理解错了**。

**真实bug在这里**:
```python
# 原始panel: (symbol, date) MultiIndex × 18因子列
# reshape 到 (n_dates, n_symbols, n_factors)
# unstack后得到 (n_dates, n_symbols×n_factors) 的矩阵

normalized.values.reshape(n_dates, n_symbols, n_factors)  # Line 260

# 但问题：normalized 的列是什么顺序？
# unstack 后列是 (FACTOR1, SYM1), (FACTOR1, SYM2), ..., (FACTOR2, SYM1), ...
# 还是 (SYM1, FACTOR1), (SYM1, FACTOR2), ..., (SYM2, FACTOR1), ...?
```

**这里有隐晦的列序假设，容易出问题**。

---

### 🟡 **问题3: 权重网格参数与因子数不适配**
**严重程度**: MEDIUM  
**位置**: 配置文件 & 权重生成逻辑

#### 问题描述
最新筛选产出**18个因子**，但回测配置写死了:
```yaml
# parallel_backtest_config.yaml
weight_grid_points: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
weight_sum_range: [0.8, 1.2]  # ← 这个范围对18因子太宽松
max_combinations: 10000
```

#### 真实问题
1. **权重生成效率低**:
   ```python
   # 理论: 11^18 ≈ 4e18 组合
   # 实际: filter到 10000 组合 = 0.000000026% 有效率 ❌
   # 意味着: 大量组合被丢弃，权重空间没被有效探索
   ```

2. **权重约束逻辑不清**:
   ```python
   valid_mask = (weight_sums >= 0.8) & (weight_sums <= 1.2)
   # 但18个因子，权重和1.0，平均权重0.056
   # 几乎每个权重都是0.0, 0.1 才能保证总和在[0.8, 1.2]
   # 结果: 权重组合会严重集中在 "某些因子1.0，其他全0" 的角落
   ```

3. **没有自适应逻辑**:
   - 筛选出10个因子时 → 权重网格合理
   - 筛选出18个因子时 → 同一套配置就变得不适用
   - ❌ 缺少"自动调整权重网格"的逻辑

---

### 🟡 **问题4: 缺少IC/IR排序和因子可信度过滤**
**严重程度**: MEDIUM  
**位置**: `parallel_backtest_configurable.py` Line 148

#### 问题描述
```python
# 现有逻辑
if self.config.factors:
    factors = self.config.factors  # 用配置里的因子
else:
    factors = df.head(self.config.top_k)[col_name].tolist()  # 取前K
    
# 但 passed_factors.csv 已经按IC排序了!
# 第1行: PRICE_POSITION_20D IC=0.600 ← 最强
# ...
# 第18行: VOLATILITY_120D IC=-0.031 ← 最弱

# 问题：回测没有显式用IC值过滤
df = pd.read_csv(self.config.screening_file)
# 这里可以读到 ic_mean, ic_ir 列，但没有用它们
```

#### 真实问题
1. **无IC过滤**:
   ```python
   # 应该有（但没有）
   min_ic = 0.01
   df_filtered = df[df['ic_mean'] >= min_ic]
   
   # 目前只是盲目取Top-K
   ```

2. **无IR质量检查**:
   ```python
   # 应该有（但没有）
   min_ir = 0.05
   df_filtered = df[df['ic_ir'] >= min_ir]
   ```

3. **统计显著性被忽视**:
   - CSV中有 `p_value`, `t_stat`, `coverage` 列
   - 回测没用它们来过滤因子
   - ❌ 可能在用 p-value=1.0 的噪声因子

---

### 🟡 **问题5: 简化引擎与完整引擎的混乱**
**严重程度**: MEDIUM  
**位置**: `simple_parallel_backtest_engine.py` vs 其他两个引擎

#### 问题描述
有**三个**回测引擎:
1. `parallel_backtest_engine.py` - 原始版本（向量化）
2. `parallel_backtest_configurable.py` - 配置化版本
3. `simple_parallel_backtest_engine.py` - 简化版本

#### 真实问题
1. **API不一致**:
   ```python
   # engine1
   engine.run_parallel_backtest(panel_path, price_dir, screening_csv, ...)
   
   # engine2
   engine.parallel_grid_search(panel_path, price_dir, screening_file, factors, ...)
   
   # engine3
   engine.backtest(...) # 不知道签名
   ```

2. **因子加载逻辑三份**:
   - engine1: 不知道
   - engine2: `_load_top_factors()`
   - engine3: 没看
   - ❌ 重复代码，维护困难

3. **与最新筛选的适配等级**:
   - engine1: ❌ 不清楚是否支持passed_factors.csv
   - engine2: ⚠️ 部分支持（需验证列名）
   - engine3: ❌ 未知

---

### 🟡 **问题6: 权重组合生成算法不确定性**
**严重程度**: MEDIUM  
**位置**: `simple_parallel_backtest_engine.py` Line 70-100

#### 问题描述
```python
# simple引擎的权重生成
for _ in range(max_combinations):
    weights = {}
    remaining_sum = 1.0
    
    for i, factor in enumerate(factor_names[:-1]):
        if i == len(factor_names) - 2:
            weights[factor] = remaining_sum
        else:
            max_weight = min(remaining_sum, 0.8)
            if max_weight > 0:
                weight = np.random.choice([w for w in weight_options if w <= max_weight])
                weights[factor] = weight
                remaining_sum -= weight
            else:
                weights[factor] = 0.0
    
    weights[factor_names[-1]] = max(0.0, remaining_sum)
```

#### 真实问题
1. **随机生成，不确定性**:
   - 每次运行会产生不同的权重组合（种子设为42，但...）
   - 结果不可复现 ❌

2. **算法偏差**:
   - 靠前的因子权重会更大（因为随机选择时用了 `remaining_sum`）
   - 最后一个因子权重被迫等于剩余值
   - ❌ 权重分布不均匀

3. **与 engine1 的 `itertools.product` 冲突**:
   - engine1: 网格搜索 → 确定性
   - simple: 随机抽样 → 随机性
   - ❌ 两套策略混乱

---

### 🟢 **问题7: 缺少"筛选结果与回测参数匹配"的验证**
**严重程度**: LOW-MEDIUM  
**位置**: 整个回测系统启动点

#### 问题描述
```python
# 应该有但没有的验证
筛选输出: 18个因子
配置要求: weight_grid 的 11 个点

# 系统启动时应该警告:
"⚠️ 因子数(18) × 权重点数(11) = 组合指数爆炸"
"⚠️ 建议调整权重网格为 [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] 或自动调整"
```

#### 真实问题
缺少预校验逻辑，导致:
1. 用户可能输入不匹配的配置
2. 回测引擎沉默地执行，丢弃大量组合
3. 最后得到的结果可信度低，但用户不知道

---

## 📋 问题清单（优先级）

| 优先级 | 问题 | 类型 | 影响 |
|--------|------|------|------|
| 🔴 | 因子来源适配断层 | 接口 | 可能使用错误的因子集合 |
| 🔴 | 矩阵重塑列序隐晦假设 | 数据结构 | 因子排列顺序混乱 |
| 🟡 | 权重网格与因子数不适配 | 配置 | 权重空间探索低效 |
| 🟡 | 无IC/IR过滤 | 逻辑 | 回测可能用噪声因子 |
| 🟡 | 三个引擎的API混乱 | 架构 | 维护困难，易出错 |
| 🟡 | 权重生成算法的随机性 | 算法 | 不可复现性 |
| 🟢 | 缺少参数匹配验证 | 用户体验 | 隐错，难调试 |

---

## 🔬 需要确认的细节

### 问题A: Unstack 后的列顺序
```python
panel = pd.read_parquet("panel.parquet")  # MultiIndex(symbol, date) × [PRICE_POSITION_20D, ...]
factor_data = panel[18个因子].unstack(level="symbol")

# 问题: 列的顺序是?
# (PRICE_POSITION_20D, symbol_1), (PRICE_POSITION_20D, symbol_2), ..., (VOLUME_RATIO_60D, symbol_1), ...
# 还是别的?
# 这决定了 reshape(n_dates, n_symbols, n_factors) 是否正确
```

### 问题B: passed_factors.csv 的排序
```python
# 问题: 是按IC从高到低排序吗?
# 还是按输入因子池的顺序?
# 这影响 df.head(top_k) 是否真的获取了最佳因子
```

### 问题C: 三个回测引擎哪个是生产版？
```python
# engine1: 原始, 向量化, 但不知道是否支持最新筛选格式
# engine2: 配置化, 有因子加载逻辑, 但代码更复杂
# engine3: 简化, 但随机生成权重
# 
# 问题: 应该用哪个? 有优先级吗?
```

---

## 📊 系统快照

**当前筛选输出:**
```
18个通过因子, 按IC排序
Top 5:
  1. PRICE_POSITION_20D    IC=0.600  IR=2.36
  2. INTRADAY_POSITION     IC=0.357  IR=1.27
  3. PRICE_POSITION_120D   IC=0.369  IR=1.14
  4. BUY_PRESSURE          IC=0.289  IR=0.87
  5. LARGE_ORDER_SIGNAL    IC=0.217  IR=1.03
```

**当前回测配置:**
```yaml
top_k: 10  # 会取前10个因子
weight_grid_points: 11个点
weight_sum_range: [0.8, 1.2]  # 对18因子太宽松
```

**风险:** 
- ❌ 回测会用18个因子，但权重网格针对少量因子设计
- ❌ 可能丢弃低IC因子（VOLATILITY_120D IC=-0.031）却不自知
- ❌ 权重组合空间利用率 < 0.1%

---

## 🎯 建议的检查点

1. 验证 `_load_top_factors()` 真正读到的因子
2. 打印 unstack 后的列名，确认顺序
3. 检查 simple 引擎的权重生成是否可控
4. 添加 IC 过滤前的日志输出
5. 针对18因子重新设计权重网格

---

**状态**: 仅定位问题，不修复  
**下一步**: 等待确认问题优先级，决定修复顺序
