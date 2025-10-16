# 🪓 ETF横截面因子系统使用指南

**更新时间**: 2025-10-16 19:40  
**状态**: ✅ 系统正常工作，370个因子  

---

## 📊 系统现状

### **当前产出**
```
✅ ETF数量: 43个
✅ 因子数量: 370个
✅ 数据点: 8,084行
✅ 日期范围: 2024-01-01 ~ 2024-10-16
✅ 面板大小: 21MB
```

### **因子分类**
```
VBT内置指标: 152个
TA-Lib指标: 193个
自定义指标: 25个
总计: 370个
```

---

## 🎯 正确的调用方式

### **核心架构**

```
factor_system/factor_generation/
  └── enhanced_factor_calculator.py (154个指标引擎)
       ↓
factor_system/factor_engine/adapters/
  └── VBTIndicatorAdapter (统一适配器)
       ↓
etf_factor_engine_production/scripts/
  └── produce_full_etf_panel.py (ETF面板生产)
```

### **关键代码** (Line 136-222)

```python
def calculate_all_factors(self, data: pd.DataFrame) -> pd.DataFrame:
    """计算所有因子 - 使用factor_generation批量计算"""
    
    # 1. 加载VBT适配器（复用factor_generation引擎）
    from factor_system.factor_engine.adapters import VBTIndicatorAdapter
    
    calculator = VBTIndicatorAdapter(
        price_field=self.price_field,
        engine_version=self.engine_version
    )
    
    # 2. 按symbol分组计算
    symbols = data.index.get_level_values("symbol").unique()
    
    for symbol in symbols:
        # 提取单个ETF数据
        symbol_data = data.xs(symbol, level="symbol")
        calc_input = symbol_data.reset_index()
        
        # 3. 批量计算所有因子（370个）
        factors_df = calculator.compute_all_indicators(calc_input)
        
        # 4. 添加symbol和date
        factors_df["symbol"] = symbol
        factors_df["date"] = calc_input["date"].values
        
        panel_list.append(factors_df)
    
    # 5. 合并为横截面面板
    panel = pd.concat(panel_list, ignore_index=True)
    panel = panel.set_index(["symbol", "date"]).sort_index()
    
    return panel
```

---

## 🚀 使用方法

### **1. 生成全量因子面板**

```bash
# 从项目根目录运行
cd /Users/zhangshenshen/深度量化0927

# 生成面板
python etf_factor_engine_production/scripts/produce_full_etf_panel.py \
  --start-date 20240101 \
  --end-date 20241016 \
  --data-dir raw/ETF/daily \
  --output-dir factor_output/etf_rotation

# 输出:
# - panel_FULL_20240101_20241016.parquet (21MB, 370个因子)
# - panel_FULL_20240101_20241016_summary.csv (因子概要)
```

### **2. 筛选高质量因子**

```bash
# 从全量面板筛选
python etf_factor_engine_production/scripts/filter_factors_from_panel.py \
  --panel-file factor_output/etf_rotation/panel_FULL_20240101_20241016.parquet \
  --summary-file factor_output/etf_rotation/panel_FULL_20240101_20241016_summary.csv \
  --mode production \
  --output-dir factor_output/etf_rotation

# 输出:
# - panel_filtered_production.parquet (高质量因子)
# - factor_correlation_matrix.csv (相关性矩阵)
```

### **3. 测试验证**

```bash
# 测试面板结构
python etf_factor_engine_production/scripts/test_one_pass_panel.py
```

---

## 📋 数据格式

### **输入数据格式**

```python
# ETF数据文件: raw/ETF/daily/*.parquet
# 必需列:
- date: datetime64
- open: float64
- high: float64
- low: float64
- close: float64
- volume: float64
- adj_close: float64 (可选，优先使用)
```

### **输出面板格式**

```python
# MultiIndex: (symbol, date)
# Columns: 370个因子
# 示例:
                    VBT_MA_5  VBT_MA_10  VBT_MA_20  ...
symbol     date                                      
159801.SZ  2024-01-02  1.234     1.235      1.236  ...
           2024-01-03  1.235     1.236      1.237  ...
159819.SZ  2024-01-02  2.345     2.346      2.347  ...
```

---

## 🔧 核心组件说明

### **VBTIndicatorAdapter**

```python
# 位置: factor_system/factor_engine/adapters/vbt_indicator_adapter.py
# 功能: 统一适配器，调用factor_generation引擎

from factor_system.factor_engine.adapters import VBTIndicatorAdapter

adapter = VBTIndicatorAdapter(
    price_field="adj_close",  # 优先使用复权价格
    engine_version="v2"       # 引擎版本
)

# 计算所有指标
factors = adapter.compute_all_indicators(data)
# 返回: DataFrame with 370 columns
```

### **factor_generation引擎**

```python
# 位置: factor_system/factor_generation/enhanced_factor_calculator.py
# 功能: 154个技术指标计算引擎

# 指标分类:
- VBT内置: 152个 (MA, RSI, MACD, BBANDS, ATR等)
- TA-Lib: 193个 (WILLR, CCI, ADX, AROON等)
- 自定义: 25个 (组合指标、派生指标)
```

---

## ⚠️ 注意事项

### **1. 运行目录**

```bash
# ❌ 错误: 在scripts目录运行
cd etf_factor_engine_production/scripts
python produce_full_etf_panel.py  # 会找不到数据

# ✅ 正确: 在项目根目录运行
cd /Users/zhangshenshen/深度量化0927
python etf_factor_engine_production/scripts/produce_full_etf_panel.py
```

### **2. 数据路径**

```python
# 数据必须在: raw/ETF/daily/*.parquet
# 文件命名格式: {symbol}_daily_{start}_{end}.parquet
# 例如: 510300.SH_daily_20200102_20251014.parquet
```

### **3. 内存使用**

```python
# 43个ETF × 370个因子 × 188天 ≈ 21MB
# 如果ETF数量增加，注意内存使用
# 建议: 分批处理，每批50个ETF
```

---

## 🎯 扩展方式

### **如果需要新增因子**

```python
# ❌ 错误: 在ETF系统里重新开发
# ✅ 正确: 在factor_generation引擎里添加

# 1. 编辑: factor_system/factor_generation/enhanced_factor_calculator.py
# 2. 添加新指标到相应类别
# 3. 重新运行produce_full_etf_panel.py
# 4. 新因子自动出现在面板中
```

### **如果需要自定义指标**

```python
# 在VBTIndicatorAdapter中添加
# 位置: factor_system/factor_engine/adapters/vbt_indicator_adapter.py

def compute_custom_indicators(self, data):
    # 添加自定义计算逻辑
    custom_factors = {}
    
    # 例如: 价格动量
    custom_factors['PRICE_MOMENTUM_5'] = data['close'].pct_change(5)
    
    return pd.DataFrame(custom_factors)
```

---

## 📊 性能指标

```
计算速度: ~2秒/ETF (370个因子)
总耗时: 43个ETF × 2秒 ≈ 86秒
内存峰值: ~500MB
输出大小: 21MB (parquet压缩)
```

---

## 🪓 Linus式原则

### **DO**
✅ 复用factor_generation引擎  
✅ 使用VBTIndicatorAdapter统一接口  
✅ 从项目根目录运行脚本  
✅ 保持数据格式一致  

### **DON'T**
❌ 不要重复开发指标计算  
❌ 不要在ETF系统里造轮子  
❌ 不要创建新的计算引擎  
❌ 不要忽视现有架构  

---

## 📝 相关文档

- `factor_system/factor_generation/README.md` - 因子引擎文档
- `etf_factor_engine_production/DELIVERY_REPORT.md` - 交付报告
- `CORRECT_APPROACH.md` - 正确方法总结（教训）

---

**最后更新**: 2025-10-16  
**维护者**: Linus式量化工程师  
**原则**: 复用 > 重写，修复 > 重建  

🪓 **代码要干净、逻辑要可证、系统要能跑通**
