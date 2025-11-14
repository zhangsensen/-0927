# 回测脚本统一配置接入文档

**日期**: 2024-11-14  
**状态**: ✅ 已部署  
**影响范围**: `real_backtest/run_profit_backtest.py`

---

## 📋 变更概述

将真实回测脚本 `run_profit_backtest.py` 接入统一的排序配置系统(`combo_wfo_config.yaml`),确保"WFO产出→排序→回测"整条链路使用同一套配置,避免手动指定 `--ranking-file` 时的遗漏风险。

---

## 🎯 核心改动

### 1. **统一TopK默认值来源**
**优先级**: `--topk 参数` > `ranking.top_n 配置` > `None (全部)`

```python
# 修改前: 仅从环境变量或None
default_topk = int(env_topk) if env_topk else None

# 修改后: 配置文件 > 环境变量 > None
ranking_config = cfg.get("ranking", {})
config_top_n = ranking_config.get("top_n", None)
final_topk = args.topk if args.topk else config_top_n
```

**效果**:
- 命令行不指定 `--topk` 时,自动读取配置文件的 `ranking.top_n: 200`
- 显式传入 `--topk` 时优先使用参数值
- 向后兼容原有的环境变量 `RB_TOPK`

---

### 2. **根据 ranking.method 自动选择排序文件**
**配置文件**: `configs/combo_wfo_config.yaml`

```yaml
ranking:
  method: "ml"   # 或 "wfo"
  top_n: 200
```

#### 场景1: ML排序 (method="ml")
自动查找最新run目录的ML排序文件:
```
ranking_ml_top{topk}.parquet  # 首选,TopK特定
ranking_ml_top200.parquet     # 默认top200
ranked_top{topk}.parquet      # 备用命名
ranked_combos.parquet         # 全量ML排序
```

**日志示例**:
```
✓ 排序方式: ML (LTR 模型) ✅ 生产推荐
  样本数: 200
```

#### 场景2: WFO排序 (method="wfo")
使用原有的内部排序逻辑 `load_top_combos_from_run()`:
```python
top_df = load_top_combos_from_run(latest_run, top_n=final_topk)
top_df_cal = maybe_apply_profit_calibrator(top_df)
```

**日志示例**:
```
✓ 排序方式: WFO 内部排序 ⚠️ 备用模式
  排序指标: IC_or_calibrated_default
```

---

### 3. **保持 --ranking-file 参数最高优先级**
显式指定 `--ranking-file` 时,覆盖配置文件的 `ranking.method`:

```bash
python real_backtest/run_profit_backtest.py \
  --slippage-bps 2 \
  --ranking-file results/run_xxx/ranking_ml_top200.parquet
```

**日志示例**:
```
✓ 使用排序文件: ranking_ml_top200.parquet (样本=200)
  来源: --ranking-file 参数 (显式指定)
```

---

### 4. **增强日志输出**
现在日志清晰标识:
- TopK 来源 (参数/配置文件/默认)
- 排序模式 (ML/WFO)
- 排序文件来源
- 是否为生产推荐配置

```
====================================================================================================
盈利优先回测 (含滑点 + 利润校准排序)
====================================================================================================
参数: TopK=200 (来源: 配置文件), 滑点=2.0bps, 强制频率=无

✓ 配置文件: /path/to/combo_wfo_config.yaml
✓ 滑点率: 0.0200%

读取 WFO 组合...
✓ 最新 run: /path/to/results/run_20251114_184946
  排序模式: ML (来源: 配置文件 ranking.method)
✓ 找到 ML 排序文件: ranking_ml_top200.parquet
✓ 排序方式: ML (LTR 模型) ✅ 生产推荐
  样本数: 200
```

---

## ✅ 验证测试

### 测试1: 默认ML模式 (无 --ranking-file)
```bash
python real_backtest/run_profit_backtest.py --slippage-bps 2
```

**预期行为**:
- TopK=200 (来自配置文件)
- 排序模式: ML
- 自动使用 `ranking_ml_top200.parquet`
- 日志标识 "✅ 生产推荐"

**实际结果**: ✅ 通过
```
参数: TopK=200 (来源: 配置文件)
排序模式: ML (来源: 配置文件 ranking.method)
✓ 排序方式: ML (LTR 模型) ✅ 生产推荐
Top1年化(净): 22.62% | Sharpe(净): 1.096
```

---

### 测试2: WFO备用模式
修改配置 `ranking.method: "wfo"` 后:
```bash
python real_backtest/run_profit_backtest.py --slippage-bps 2
```

**预期行为**:
- TopK=200 (来自配置文件)
- 排序模式: WFO
- 使用内部排序逻辑
- 日志标识 "⚠️ 备用模式"

**实际结果**: ✅ 通过
```
参数: TopK=200 (来源: 配置文件)
排序模式: WFO (来源: 配置文件 ranking.method)
✓ 排序方式: WFO 内部排序 ⚠️ 备用模式
Top1年化(净): 19.87% | Sharpe(净): 0.988
```

---

### 测试3: 显式指定 --ranking-file
```bash
python real_backtest/run_profit_backtest.py \
  --slippage-bps 2 \
  --ranking-file results/run_20251114_184946/ranking_ml_top200.parquet
```

**预期行为**:
- 使用指定的排序文件
- 覆盖配置文件的 `ranking.method`
- 日志标识 "显式指定"

**实际结果**: ✅ 通过
```
✓ 使用排序文件: ranking_ml_top200.parquet (样本=200)
  来源: --ranking-file 参数 (显式指定)
Top1年化(净): 22.62% | Sharpe(净): 1.096
```

---

## 🔧 使用指南

### 场景A: 生产环境 (推荐)
保持配置文件默认 `method: "ml"`,直接运行:
```bash
python real_backtest/run_profit_backtest.py --slippage-bps 2
```
✅ 自动使用ML排序结果,TopK=200

---

### 场景B: 对照基准 (WFO排序)
临时测试WFO排序效果:
```bash
# 方法1: 修改配置文件 ranking.method="wfo"
python real_backtest/run_profit_backtest.py --slippage-bps 2

# 方法2: 显式指定ranking-file
python real_backtest/run_profit_backtest.py \
  --ranking-file results/run_xxx/ranking_ic_top200.parquet \
  --slippage-bps 2
```

---

### 场景C: 自定义TopK
```bash
# 使用配置文件的 ranking.method,但覆盖TopK
python real_backtest/run_profit_backtest.py --topk 500 --slippage-bps 2
```

---

### 场景D: 完全自定义
```bash
# 同时指定TopK和排序文件
python real_backtest/run_profit_backtest.py \
  --topk 1000 \
  --ranking-file results/run_xxx/custom_ranking.parquet \
  --slippage-bps 2
```

---

## 📊 优先级汇总

### TopK 决策链:
1. `--topk` 参数 (最高优先级)
2. `ranking.top_n` 配置
3. `RB_TOPK` 环境变量
4. `None` (全部组合)

### 排序文件决策链:
1. `--ranking-file` 参数 (最高优先级)
2. `ranking.method="ml"` → `ranking_ml_top{topk}.parquet`
3. `ranking.method="wfo"` → `load_top_combos_from_run()` (内部排序)
4. ML文件缺失 → 自动回退到WFO

---

## 🎯 核心价值

### ✅ 避免人为错误
- 不再需要手动指定 `--ranking-file`
- 忘记传参时自动使用生产配置
- 减少"本地测试用了旧排序"的问题

### ✅ 配置统一
- WFO主流程和回测脚本使用同一配置文件
- 排序模式切换只需修改一处 (`ranking.method`)
- TopK设置全局一致

### ✅ 向后兼容
- 保留所有CLI参数和环境变量
- 原有脚本不需修改即可运行
- `--ranking-file` 参数仍可覆盖配置

### ✅ 可观测性
- 日志清晰标识TopK来源
- 明确区分ML/WFO模式
- 标注生产推荐配置 (✅) 和备用模式 (⚠️)

---

## 🔗 相关文档

- [ML排序接入文档](./ml_ranking_integration.md)
- [ML排序默认化部署文档](./ml_ranking_default_deployment.md)
- [WFO配置参考](../configs/combo_wfo_config.yaml)
- [回测脚本使用指南](../real_backtest/README.md)

---

## 📝 变更历史

| 日期       | 版本 | 变更内容                    | 作者  |
|------------|------|----------------------------|-------|
| 2024-11-14 | 1.0  | 初始版本,回测脚本统一配置接入 | AI    |

---

## 💡 后续优化建议

1. **环境变量清理**: 考虑逐步淘汰 `RB_TOPK`, `RB_RANKING_FILE` 等环境变量,统一到配置文件
2. **配置校验**: 增加配置文件的合法性检查 (如 method 只能是 "ml" 或 "wfo")
3. **日志结构化**: 将关键日志输出到JSON文件,便于后续分析
4. **多配置支持**: 支持通过 `--config` 参数切换不同的配置文件
