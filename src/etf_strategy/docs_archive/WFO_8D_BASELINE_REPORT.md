<!-- ALLOW-MD --># WFO 8天基线运行报告
**时间**: 2025-11-08 19:37  
**运行目录**: `results/run_20251108_193712/`

---

## 📊 核心指标

| 指标 | 数值 |
|------|------|
| 总组合数 | 12,597 |
| 显著组合 | 11,357 (90.2%) |
| 平均 OOS IC | 0.083 |
| IC 标准差 | 0.027 |
| IC 范围 | [-0.039, 0.155] |

## 🏆 Top 10 组合（按 IC 排序）

| 排名 | 组合 | OOS IC | 稳定性 | 频率 |
|------|------|--------|--------|------|
| 1 | ADX_14D + PRICE_POSITION_20D | 0.1553 | 0.08 | 8天 |
| 2 | ADX_14D + PRICE_POSITION_20D + VOL_RATIO_20D | 0.1530 | -0.07 | 8天 |
| 3 | ADX_14D + PRICE_POSITION_20D + PV_CORR_20D | 0.1511 | -0.07 | 8天 |
| 4 | ADX_14D + PRICE_POSITION_20D + VOL_RATIO_60D | 0.1506 | -0.08 | 8天 |
| 5 | PRICE_POSITION_20D + VOL_RATIO_60D | 0.1484 | 0.07 | 8天 |
| 6 | PRICE_POSITION_20D + PV_CORR_20D | 0.1478 | 0.08 | 8天 |
| 7 | PRICE_POSITION_20D + VOL_RATIO_20D | 0.1477 | 0.07 | 8天 |
| 8 | ADX_14D + PRICE_POSITION_20D + VOL_RATIO_20D + VOL_RATIO_60D | 0.1477 | -0.24 | 8天 |
| 9 | ADX_14D + PRICE_POSITION_20D + PV_CORR_20D + VOL_RATIO_20D | 0.1473 | -0.23 | 8天 |
| 10 | ADX_14D + CORRELATION_TO_MARKET_20D + PRICE_POSITION_20D | 0.1472 | -0.09 | 8天 |

## 📈 IC 分布分析

```
IC > 0.3:  0 组合
IC > 0.2:  0 组合
IC > 0.1:  2,182 组合 (17.3%)
IC > 0.0:  12,277 组合 (97.5%)
IC < 0.0:  320 组合 (2.5%)
```

**关键观察**:
- IC 中位数: 0.092
- IC 75分位: 0.097
- 无极端高IC组合（>0.3），说明无过拟合迹象
- 97.5%组合IC为正，基线稳健

## 🔧 因子主导性分析

**高频出现因子（Top组合中）**:
1. **PRICE_POSITION_20D** — 出现10次，核心因子
2. **ADX_14D** — 出现8次，趋势强度
3. **VOL_RATIO_20D / VOL_RATIO_60D** — 成交量结构
4. **PV_CORR_20D** — 价量耦合

**被忽略因子**:
- RSI_14, MOM_20D, SLOPE_20D 在Top10中未出现
- 说明短期动量因子在8天频率下不占优

## 📦 组合规模分布

| 规模 | 数量 | 占比 |
|------|------|------|
| 2因子 | 153 | 1.2% |
| 3因子 | 816 | 6.5% |
| 4因子 | 3,060 | 24.3% |
| 5因子 | 8,568 | 68.0% |

**观察**: 5因子组合占主导，但Top1是2因子简洁组合。

## 🎯 频率一致性

- **100%组合最优频率 = 8天**（符合预期，配置锁定单频）
- 无频率切换，排除了频率干扰

## ✅ 产物完整性

```
results/run_20251108_193712/
├── READY (标记文件)
├── all_combos.parquet (4.5MB, 12597行)
├── top100_by_ic.parquet (51.5KB)
├── top_combos.parquet (51.5KB)
├── factors/ (18个因子parquet)
├── wfo_summary.json
├── factor_selection_summary.json
└── run_config.json
```

## 🚀 下一步建议

1. **真实回测验证**:
   - 对Top100执行8天频率回测
   - 计算realized Sharpe, annual_ret, max_dd
   - 验证learned_score排序 ↔ realized性能一致性

2. **泛化测试**（可选）:
   - 对Top100执行16/24天频率回测
   - 观察IC稳定性跨频率衰减

3. **稳健性增强**:
   - 生成learned_ranking（Ridge回归WFO特征→IC）
   - 构建Top500白名单
   - 执行concordance分析（Precision@K, Top-decile recall）

---

## 🔬 技术改进（已完成）

### 1. tqdm进度条监控
- ✅ WFO组合评估：`tqdm(all_combos, desc="WFO组合评估", unit="combo")`
- ✅ 横截面标准化：`tqdm(factor_items, desc="横截面标准化", unit="因子")`
- 位置：
  - `core/combo_wfo_optimizer.py:246`
  - `core/cross_section_processor.py:343`

### 2. 完整清理与重建
- 清空 `.cache/*`, `results/run_*`, `results_combo_wfo/*`
- 锁定配置：`rebalance_frequencies: [8]`
- 原子落盘：`pending_run_<ts>` → `run_<ts>` + `READY`标记
- 自动维护：`.latest_run`文件 + `run_latest`符号链接

### 3. 配置简化
- 去除多余频率候选
- `combo_wfo_config.yaml`仅保留`[8]`
- 消除环境变量`RB_FREQ_SUBSET`复杂性

---

**结论**: 8天WFO基线已稳定建立，产物齐全，IC分布合理，无过拟合迹象。进度监控已优化（tqdm），后续可直接用于真实回测验证与排序学习。
