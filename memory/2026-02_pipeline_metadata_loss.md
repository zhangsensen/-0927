# META-LESSON: 管线信息瓶颈 — 元数据在中间阶段丢失

> 最后更新: 2026-02-14
> 关联规则: Rule 14, 19, 24

---

## 模式识别：同一个 bug 的三次变体

| 日期 | 丢失的信息 | 瓶颈位置 | Gap / 症状 | Rule |
|------|-----------|---------|-----------|------|
| 02-12 | non-OHLCV 因子名 | holdout/rolling 用旧 API | 177/200 静默丢弃 | 14 |
| 02-13 | delta_rank / min_hold_days | holdout/rolling 没传参 | 12.1pp → 8.9pp | 19 |
| 02-14 | factor_signs / factor_icirs | VEC 输出只存性能指标 | **20.3pp → 1.47pp** | 24 |

三次 bug 的**根因完全一样**：多阶段管线中，中间阶段只传递了"自己关心的数据"，
丢弃了下游需要的元数据。

## 抽象模式

```
WFO（生产元数据 A+B+C）
  → 中间阶段 X（只用 A，丢弃 B+C）
    → 输出只含 A + 性能指标
      → 下游 Y 需要 B → 拿不到 → 默认值 → 错误行为
        → 错误行为 vs BT（有 B）= 人为 gap
```

## 为什么这个模式反复出现？

1. **阶段独立开发**：VEC、Holdout、Rolling 是不同时期写的，各自只关心自己需要的输入
2. **默认值静默降级**：缺列时 fallback 到默认值（all-positive, equal-weight, no-hysteresis），
   不报错不告警，结果"看起来正常"
3. **新功能后只改了起点和终点**：IC-sign fix 改了 WFO（生产端）和 BT（消费端），
   但中间的 VEC 管线作为"传递者"被忽略了
4. **测试用旧数据**：旧策略全是 OHLCV + all-positive-sign，bug 永远不触发

## 防御原则

### 原则 1: 管线是数据传送带，不是漏斗

每个阶段不仅要处理自己的逻辑，还要**透传所有上游元数据**。
阶段 X 不用 column B 不代表可以丢掉它——下游 Y 可能需要。

```python
# 错误：只保存自己计算的列
results.append({"combo": combo_str, "vec_return": ret, ...})

# 正确：自己的列 + 上游透传
row_result = {"combo": combo_str, "vec_return": ret, ...}
if "factor_signs" in input_df.columns:
    row_result["factor_signs"] = input_df.iloc[i]["factor_signs"]
results.append(row_result)
```

### 原则 2: "修复无效果"= 数据流断裂，不是"问题不存在"

当一个理论上应该有效的 fix 产生零变化时：
- ❌ 错误结论："这个问题不重要"
- ✅ 正确反应："fix 的数据通路是否畅通？"

本次验证：ICIR fix 在 holdout/rolling 脚本中已经写了 pre-multiply 逻辑，
但 VEC 输出没有 factor_signs 列 → `has_signs = False` → 逻辑从未执行 → 零变化。
**零变化本身就是最大的红旗。**

### 原则 3: 新元数据必须做"端到端穿透测试"

每次给 WFO 输出加新列时，验证方法：

```bash
# 1. 确认 WFO 输出有这个列
uv run python -c "import pandas as pd; df=pd.read_csv('results/run_*/full_combo_results.csv', nrows=1); print('col' in df.columns)"

# 2. 确认 VEC 输出也有这个列
uv run python -c "import pandas as pd; df=pd.read_parquet('results/vec_from_wfo_*/full_space_results.parquet', nrows=1); print('col' in df.columns)"

# 3. 确认 Holdout 脚本能读到
# 在 holdout 脚本的入口打印 training_df.columns

# 4. VEC-BT gap 在正常范围（<5pp）
```

### 原则 4: VEC-BT gap 是管线健康指标

| Gap 范围 | 含义 | 行动 |
|---------|------|------|
| < 2pp | 正常（float/int 差异） | 无需行动 |
| 2-5pp | 轻微偏差 | 检查参数一致性 |
| 5-10pp | 可能有参数遗漏 | Rule 19 检查清单 |
| > 10pp | 几乎确定有结构性 bug | 消融法定位（Rule 22） |
| > 20pp | 数据流断裂 | 检查元数据传播（Rule 24） |

---

## 通用检查清单：WFO 新增列后

```
□ 1. WFO 输出有新列？（csv/parquet 验证）
□ 2. run_full_space_vec_backtest.py 读取并使用了新列？
□ 3. VEC 输出 parquet 透传了新列？
□ 4. run_holdout_validation.py 从 VEC 输出读到了新列？
□ 5. run_rolling_oos_consistency.py 从 VEC 输出读到了新列？
□ 6. batch_bt_backtest.py 从 WFO 输出读到了新列？
□ 7. generate_today_signal.py（生产脚本）也支持新列？
□ 8. parallel_audit.py 也支持新列？
□ 9. VEC-BT holdout gap < 5pp？
□ 10. Pipeline 端到端跑过，输出数量合理？
```

## 一句话总结

**管线的每个中间阶段都是潜在的信息瓶颈。
每次给上游加新元数据，必须手动确认每个下游都能收到。
"fix 无效果"不是"问题不重要"，而是"数据流断了"。**
