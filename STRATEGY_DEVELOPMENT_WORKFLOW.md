# 🚀 ETF轮动策略开发流程完整指南

> **版本**: v3.1 | **更新日期**: 2025-12-10  
> **核心思想**: 横截面相对强弱 + 高频轮动 + 风险控制  
> **目标**: 开发可复现、高胜率的ETF轮动策略

---

## 📋 目录

1. [整体流程概览](#整体流程概览)
2. [详细步骤分解](#详细步骤分解)
3. [组合数量统计](#组合数量统计)
4. [关键验证点](#关键验证点)
5. [时间复杂度分析](#时间复杂度分析)
6. [常见问题与解决方案](#常见问题与解决方案)

---

## 🎯 整体流程概览

```mermaid
graph TD
    A[数据加载] --> B[因子计算]
    B --> C[横截面标准化]
    C --> D[WFO优化]
    D --> E[VEC批量回测]
    E --> F[Rolling一致性验证]
    F --> G[Holdout冷数据验证]
    G --> H[BT事件驱动审计(Ground Truth)]
    H --> I[封板归档(Sealed Release)]
    I --> J[生产部署]

    style A fill:#e1f5fe
    style B fill:#e1f5fe
    style C fill:#e1f5fe
    style D fill:#fff3e0
    style E fill:#fff3e0
    style F fill:#fff8e1
    style G fill:#ffebee
    style H fill:#ffebee
    style I fill:#e8f5e8
    style J fill:#e8f5e8
```

**核心原则**:
- 🔒 **锁死交易规则**: FREQ=3, POS=2, 不止损, 不cash
- 🎯 **IC门槛过滤**: IC > 0.05 OR positive_rate > 55%
- 📊 **综合得分排序**: OOS收益(40%) + Sharpe(30%) + 回撤(30%)

**交付标准（v3.2 起）**：策略对外交付必须通过 **VEC + Rolling + Holdout + BT** 四重验证，且通过后必须封板归档。
- **VEC（Screening）**：用于快速精算与筛选，不作为最终对外口径。
- **Rolling（Stability）**：滚动一致性 gate 必须使用 train-only summary，禁止混入 holdout 信息。
- **Holdout（Cold Data）**：冷数据表现必须单独输出并作为门槛之一。
- **BT（Ground Truth）**：事件驱动审计输出为最终对外口径，并输出 Train/Holdout 分段收益。
- **封板（Seal）**：冻结产物 + 配置 + 关键脚本 + 源码快照 + 依赖锁定，并生成 sha256 校验。

---

## 🔄 详细步骤分解

### 1. 数据加载 📊
**输入**: 43只ETF的OHLCV数据 (2020-01-01 ~ 2025-12-08)  
**输出**: 标准化的DataFrame字典

```python
# 数据规格
- 时间范围: 1,399个交易日
- ETF数量: 43只 (38A股 + 5QDII)
- 数据频率: 日线
- 字段: open, high, low, close, volume
```

**关键检查**:
- ✅ 数据连续性 (无跳日)
- ✅ 前复权处理
- ✅ 价格合理性 (无负数/异常值)

### 2. 因子计算 🔬
**输入**: OHLCV数据  
**输出**: 18个因子 × 43只ETF × 1,399天的矩阵

**因子列表** (18个):
```
趋势类 (4): ADX_14D, SLOPE_20D, VORTEX_14D, MOM_20D
动量类 (3): RSI_14, PRICE_POSITION_20D, PRICE_POSITION_120D
风险类 (3): MAX_DD_60D, RET_VOL_20D, CALMAR_RATIO_60D
相关性 (2): CORRELATION_TO_MARKET_20D, RELATIVE_STRENGTH_VS_MARKET_20D
成交量 (3): VOL_RATIO_20D, VOL_RATIO_60D, CMF_20D
价量耦合 (3): OBV_SLOPE_10D, PV_CORR_20D
```

**计算特点**:
- ✅ Numba加速
- ✅ 滚动窗口计算
- ✅ 自动处理NaN

### 3. 横截面标准化 📐
**输入**: 原始因子矩阵  
**输出**: 标准化因子矩阵 (均值≈0, 标准差≈1)

**处理步骤**:
1. **有界因子透传**: PRICE_POSITION_20D, PRICE_POSITION_120D (已在[0,1]范围内)
2. **无界因子处理**:
   - Winsorize截断: [2.5%, 97.5%]分位数
   - Z-score标准化: (x - μ) / σ
3. **NaN保持**: 不填充缺失值

**输出规格**: (1,399天 × 43ETF × 18因子)

### 4. WFO优化 ⚡
**输入**: 标准化因子矩阵  
**输出**: Top 100个组合 (按IC排序)

**WFO参数**:
- **样本内(IS)**: 252个交易日 (约1年)
- **样本外(OOS)**: 60个交易日 (约3个月)
- **步长**: 60天
- **总窗口数**: 17个

**组合生成**:
- 因子组合大小: 2, 3, 4, 5因子
- 总组合数: 12,597个
- 评分指标: IC均值, IC标准差, ICIR

**输出**: `top100_by_ic.parquet` (100个最佳组合)

### 5. VEC批量回测 🚀
**输入**: Top 100组合 + 全量因子数据  
**输出**: 精确的回测结果 (收益, 风险指标)

**回测参数**:
- **调仓频率**: 3个交易日
- **持仓数量**: 2只ETF
- **初始资金**: 100万
- **手续费**: 0.02%
- **择时**: Light Timing (阈值-0.1)

**输出指标**:
- 总收益率
- 年化收益/波动率/Sharpe
- 最大回撤/Calmar
- 胜率/盈亏比
- 交易次数

### 6. 策略筛选 🎯
**输入**: VEC回测结果  
**输出**: Top策略 (按综合得分排序)

**筛选步骤**:
1. **IC门槛过滤**: IC > 0.05 或 positive_rate > 55%
2. **风险因子过滤**: 移除高风险因子组合
3. **综合得分计算**:
   - OOS收益: 40%
   - Sharpe: 30%
   - MaxDD: 30%

**输出**: `top100_by_composite.csv` (按得分排序)

### 7. Holdout验证 🔍
**输入**: Top策略 + Holdout数据 (2025-06-01 ~ 2025-12-08)  
**输出**: 样本外表现验证

**验证方法**:
- 重新计算因子 (Holdout期)
- 使用训练集参数标准化
- 完整VEC回测 (相同参数)
- 对比训练集表现

**关键指标**:
- Holdout收益 vs 训练集收益
- IC稳定性
- 因子衰减分析

### 8. BT 审计（Ground Truth）🧾
**输入**: final candidates + 完整 OHLCV 数据
**输出**: BT 审计结果（包含 Train/Holdout 分段收益）

**原则**:
- 最终收益/回撤等“对外口径”以 BT 为准。
- 必须输出 `bt_train_return` / `bt_holdout_return`，避免跨区间对比争议。

### 9. 封板归档（Sealed Release）🔒
**输入**: final candidates / BT results / production pack
**输出**: sealed_strategies/<version>_<yyyymmdd>/（含 MANIFEST.json + CHECKSUMS.sha256 + locked 代码快照）

---

## 📊 组合数量统计

| 阶段 | 组合数量 | 说明 |
|------|----------|------|
| **因子组合** | 12,597 | C(18,2)+C(18,3)+C(18,4)+C(18,5) |
| **WFO窗口** | 17 | IS 252天 + OOS 60天, 步长60天 |
| **WFO总计算** | 214,149 | 12,597 × 17 |
| **Top组合** | 100 | 按IC排序筛选 |
| **VEC回测** | 100 | 每个组合完整回测 |
| **筛选后** | 20-50 | 通过IC门槛和风险过滤 |
| **最终策略** | 1-5 | 按综合得分Top |

**详细组合数**:
- 2因子组合: C(18,2) = 153
- 3因子组合: C(18,3) = 816
- 4因子组合: C(18,4) = 3,060
- 5因子组合: C(18,5) = 8,568
- **总计**: 12,597

---

## ✅ 关键验证点

### 数据层验证
- [ ] **数据完整性**: 无缺失日期, 价格合理
- [ ] **前复权正确**: 除权除息处理准确
- [ ] **缓存一致性**: 缓存文件与源数据匹配

### 因子层验证
- [ ] **计算正确性**: 与定义公式一致
- [ ] **NaN处理**: 窗口不足时正确设NaN
- [ ] **标准化效果**: 均值≈0, 标准差≈1

### WFO层验证
- [ ] **IC显著性**: FDR校正后p值 < 0.05
- [ ] **OOS稳定性**: OOS IC > 0
- [ ] **无前视偏差**: 信号不使用未来数据

### VEC层验证
- [ ] **回测准确性**: 与Backtrader结果对齐 (<0.1pp差异)
- [ ] **交易逻辑**: 调仓日程, 持仓计算正确
- [ ] **风险指标**: Sharpe, MaxDD计算准确

### 筛选层验证
- [ ] **IC门槛有效**: 过滤低质量策略
- [ ] **得分计算**: 权重分配合理
- [ ] **排序稳定**: Top策略确定性

### Holdout层验证
- [ ] **样本外表现**: Holdout收益 > 0
- [ ] **因子稳定性**: IC衰减 < 50%
- [ ] **过拟合检查**: 训练集过高估计

### BT审计层验证（交付真值）
- [ ] **审计口径**: 收益/回撤/交易统计由 BT 输出
- [ ] **分段收益**: Train/Holdout 分段字段齐全
- [ ] **资金约束**: margin failure/交易次数等审计字段合理

### 封板验证（可复现与防篡改）
- [ ] **快照完整**: 产物 + 配置 + 关键脚本 + 源码快照 + 依赖锁定
- [ ] **校验通过**: sha256sum -c CHECKSUMS.sha256 全部 OK

---

## ⏱️ 时间复杂度分析

| 阶段 | 时间复杂度 | 典型耗时 | 并行化 |
|------|------------|----------|--------|
| 数据加载 | O(N×T) | 30秒 | ✅ |
| 因子计算 | O(N×T×F) | 2分钟 | ✅ |
| 标准化 | O(T×N×F) | 10秒 | ✅ |
| WFO优化 | O(C×W×T) | 1分钟 | ✅ |
| VEC回测 | O(C×T) | 5秒 | ✅ |
| 策略筛选 | O(C×logC) | 1秒 | - |
| Holdout验证 | O(T_holdout) | 30秒 | ✅ |

**总耗时**: ~5分钟 (单机, 8核并行)

---

## 🚨 常见问题与解决方案

### 数据问题
**Q: 数据不连续**  
A: 检查数据源, 补充缺失日

**Q: 前复权错误**  
A: 验证复权因子计算

### 因子问题
**Q: IC为NaN**  
A: 检查因子计算逻辑, 确认窗口长度

**Q: 标准化后分布异常**  
A: 验证Winsorize分位数设置

### WFO问题
**Q: 组合数太多**  
A: 限制组合大小 (2-4因子)

**Q: IC不显著**  
A: 检查因子质量, 调整FDR阈值

### VEC问题
**Q: 与BT结果不符**  
A: 检查交易时序, 手续费计算

**Q: 收益异常**  
A: 验证调仓逻辑, 持仓计算

### 筛选问题
**Q: Top策略太多**  
A: 提高IC门槛, 增加风险过滤

**Q: 得分计算错误**  
A: 检查权重分配, 指标计算

### Holdout问题
**Q: 表现严重恶化**  
A: 重新优化, 减少过拟合

**Q: 因子衰减**  
A: 更新因子库, 添加新因子

---

## 📁 输出文件结构

```
results/
├── run_YYYYMMDD_HHMMSS/           # WFO结果
│   ├── all_combos.parquet         # 全部12,597组合IC
│   ├── top100_by_ic.parquet       # Top100组合
│   └── factors/                   # 标准化因子数据
├── vec_full_space_YYYYMMDD_HHMMSS/ # VEC结果
│   ├── full_space_results.csv     # 全部组合回测结果
│   └── vec_all_combos.parquet     # 详细结果
└── selection_v2_YYYYMMDD_HHMMSS/   # 筛选结果
    ├── top100_by_composite.csv    # 按得分排序
    ├── all_combos_scored.csv      # 全部打分结果
    └── SELECTION_REPORT.md        # 详细报告
```

---

## 🎯 最佳实践

1. **版本控制**: 每次运行生成时间戳目录
2. **参数锁定**: 生产策略参数禁止修改
3. **结果备份**: ARCHIVE重要结果
4. **定期验证**: 每月检查因子有效性
5. **风险控制**: 始终保持止损和熔断机制

---

## 📞 技术支持

**核心脚本**:
- `src/etf_strategy/run_combo_wfo.py`: WFO主流程
- `scripts/batch_vec_backtest.py`: VEC回测
- `scripts/select_strategy_v2.py`: 策略筛选

**配置文件**:
- `configs/combo_wfo_config.yaml`: 主要参数
- `configs/etf_pools.yaml`: ETF池定义

**文档**:
- `docs/BEST_STRATEGY_43ETF_UNIFIED.md`: 策略详情
- `docs/VEC_BT_ALIGNMENT_GUIDE.md`: 对齐指南

---

**🔒 流程标准化 | 12,597组合验证 | 17窗口WFO | Holdout防过拟合**