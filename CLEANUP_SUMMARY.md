# 项目清理总结 - 2025-11-10

## 🗑️ 删除的失败项目

| 项目 | 原因 | 教训 |
|------|------|------|
| `etf_rotation_v2_breadth/` | 显式风控（止损止盈）破坏动量趋势，年化从 12.9% 崩到 2% | 风控要在信号层而非仓位层 |
| `etf_rotation_stable_iter/` | 薄封装，只是设置环境变量，无新价值 | 不要为了"隔离"而过度工程化 |
| `etf_rotation_adaptive/` | 未验证的自适应风控实验，从未跑通完整回测 | 代码先跑起来再迭代 |

**归档位置**: `archive/failed_experiments/`

---

## ✅ 保留的生产系统

### `etf_rotation_optimized/` (只读，uchg 标志)
- **状态**: 生产就绪，性能冻结（2025-11-09）
- **年化**: 12.9% | Sharpe 0.486 | 回撤 -20%
- **工作流**: 18个因子 → IC加权 → WFO优化 → 8天频率轮动
- **禁止改动**: 已用 `chmod u-w -R` 及 `chflags uchg` 锁定

---

## 🧪 新建实验项目

### `etf_rotation_experiments/` (可写)
- **用途**: 多样性优化、排名改进、新算法验证
- **来源**: 完整复制自 `etf_rotation_optimized/`
- **隔离**: 独立结果目录，不影响生产

#### 当前实验
1. **因子多样性** (`real_backtest/run_diversity_experiment.py`)
   - 目标: 打破 RSI_14 >80% 集中
   - 方法: 按类别覆盖约束、相似度去冗
   - 输出: `experiments/diversity_v1.csv` (4个演示组合)

2. **多目标排名** (规划中)
   - 目标: 改善 WFO 排序与实盘相关性
   - 方法: 多目标评分 (Sharpe + 稳定性 + 成本)

---

## 📊 项目架构最终形态

```
深度量化0927/
  ├── etf_rotation_optimized/    ← 生产主线 (只读)
  │   ├── real_backtest/
  │   │   ├── run_production_backtest.py
  │   │   └── ...
  │   ├── core/
  │   ├── configs/
  │   └── results_combo_wfo/
  │
  ├── etf_rotation_experiments/   ← 实验基地 (可写)
  │   ├── real_backtest/
  │   │   ├── run_production_backtest.py (复制)
  │   │   ├── run_diversity_experiment.py (新)
  │   │   └── ...
  │   ├── experiments/
  │   │   ├── diversity_v1.csv
  │   │   └── diversity_backtest_results/
  │   ├── EXPERIMENTS.md (本实验计划)
  │   └── ...
  │
  ├── archive/
  │   └── failed_experiments/
  │       ├── v2_breadth_市场广度风控失败_20251110/
  │       └── ... (留作教训)
  │
  └── README.md (根目录)
```

---

## 🚀 下一步工作流

### 1. 生成多样化组合
```bash
cd etf_rotation_experiments
python real_backtest/run_diversity_experiment.py \
  --topk 100 \
  --output experiments/diversity_v1.csv
```

### 2. 回测多样化组合
```bash
RB_COMBO_FILE=experiments/diversity_v1.csv \
RB_FORCE_FREQ=8 \
python real_backtest/run_production_backtest.py
```

### 3. 对比基线 vs 实验
```bash
# 生成对比报告：Sharpe / 回撤 / 因子覆盖 / 换手
python scripts/compare_results.py \
  --baseline etf_rotation_optimized/results_combo_wfo/*/top100* \
  --experiment etf_rotation_experiments/results_combo_wfo/*/top100*
```

---

## ✏️ 修改政策

- **etf_rotation_optimized**: 禁改，需要改就复制到 experiments
- **etf_rotation_experiments**: 自由开发，结果独立存储
- **共享代码**: 修改需同步更新两个副本

---

**清理完成时间**: 2025-11-10 19:26  
**总节省**: 删除 ~500MB 冗余代码 + 3个未验证实验
