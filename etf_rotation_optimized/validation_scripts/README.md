# 因子验证脚本库

本文件夹包含用于验证新增因子质量的标准化脚本。

## 脚本列表

### 1. verify_factor_implementation.py
**用途**: 验证因子实现的参数传递和计算逻辑正确性

**使用场景**:
- 添加新因子后,验证参数映射是否正确
- 检查因子计算逻辑是否符合预期
- 模拟测试因子输出是否合理

**运行方式**:
```bash
cd /Users/zhangshenshen/深度量化0927/etf_rotation_optimized
python3 validation_scripts/verify_factor_implementation.py
```

**输出内容**:
- 参数传递验证
- 计算逻辑验证
- 模拟数据测试结果
- 因子数值分布统计

---

### 2. analyze_zero_usage_factors.py
**用途**: 深度分析因子的预测能力和使用率

**使用场景**:
- WFO结果显示某因子使用率为0%
- 需要分析因子IC(信息系数)
- 检查因子间相关性
- 诊断因子未被选中的原因

**运行方式**:
```bash
cd /Users/zhangshenshen/深度量化0927/etf_rotation_optimized
python3 validation_scripts/analyze_zero_usage_factors.py
```

**输出内容**:
- 每个因子的平均IC和IC>0比例
- 因子间相关性矩阵
- 因子数值分布统计
- 0%使用率原因诊断

---

## 标准验证流程

### 阶段1: 代码实现验证(开发后立即执行)
```bash
python3 validation_scripts/verify_factor_implementation.py
```
**检查点**:
- ✅ 参数传递正确
- ✅ 计算逻辑合理
- ✅ 无语法错误
- ✅ 数值分布正常

### 阶段2: 完整测试流程
```bash
python3 scripts/step1_cross_section.py
python3 scripts/step2_factor_selection.py
python3 scripts/step3_run_wfo.py
python3 scripts/step4_backtest_1000_combinations.py
```

### 阶段3: 结果深度分析(如有0%使用率因子)
```bash
python3 validation_scripts/analyze_zero_usage_factors.py
```
**诊断点**:
- IC是否低于阈值(0.02)
- 是否与高频使用因子高度相关
- 是否在5因子配额竞争中落败

---

## 验证标准

### 因子合格标准
1. **代码质量**:
   - 参数传递正确
   - 计算逻辑符合金融定义
   - 无NaN/Inf异常值(或妥善处理)

2. **预测能力**:
   - 平均IC ≥ 0.02
   - IC>0比例 > 50%
   - IC t统计量显著

3. **独立性**:
   - 与现有因子相关性 < 0.8
   - 提供新的信息维度

### 0%使用率诊断逻辑
| 情况 | IC | 相关性 | 结论 |
|------|-------|---------|------|
| IC < 0.02 | ❌ | - | 预测能力不足,需优化 |
| IC ≥ 0.02 | ✅ | > 0.6 | 被相似因子压制,正常 |
| IC ≥ 0.02 | ✅ | < 0.6 | 在5因子配额竞争中落败,可保留观察 |

---

## 历史验证案例

### 案例1: 18因子Codex审查验证(2024-10-27)
**背景**: Codex指控RELATIVE_STRENGTH_VS_MARKET_20D和CORRELATION_TO_MARKET_20D存在"严重逻辑错误"

**验证过程**:
1. 运行`verify_factor_implementation.py`: 证明参数传递完全正确
2. 运行`analyze_zero_usage_factors.py`: 证明因子计算逻辑正确,IC正常

**结论**: ✅ Codex误判,所有代码实现正确

**关键发现**:
- RELATIVE_STRENGTH: IC=0.0238, 使用率90.9% (优秀)
- CORRELATION: IC=0.0194 (略低于0.02阈值), 数值范围[-0.896, 0.995] (不是恒为1)

---

## 维护说明

- 新增验证脚本请更新此README
- 保持脚本独立性,无需修改主代码
- 所有脚本应支持从项目根目录运行
- 输出格式保持清晰易读

**最后更新**: 2024-10-27
