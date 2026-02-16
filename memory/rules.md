# Rules: 策略验证诊断规则

## Rule 1: 策略结果可疑时的诊断流程

```
第1步：拆解看数字
  "把 train / holdout / full 三期收益分开显示"
  → train 和 holdout 方向不一致 = 红旗
  → train 亏损的策略不能进 holdout 验证

第2步：审计验证关卡
  "验证流程每一关具体检查什么？"
  → 逐关列出检查条件，对照"三好原则"找缺失
  → 三好 = train好 + rolling好 + holdout好

第3步：看漏斗通过率
  → 某关 99% 通过 = 形同虚设
  → 某关 1% 通过 = 门槛可能不合理或过时

第4步：参数溯源
  → 门槛值是什么时候、什么代码版本下设的？
  → 代码改了但门槛没跟着调 = 必须重新校准
```

## Rule 2: 验证流程必须包含四关

| 关卡 | 检查内容 | 门槛 |
|------|---------|------|
| Train Gate | train_return > 0, train_mdd < 25% | 硬性淘汰 |
| Rolling Gate | pos_rate ≥ 60%, worst ≥ -8%, calmar ≥ 0.30 | 一致性 |
| Holdout Gate | holdout_return > 0 | 冷数据 |
| BT Gate | margin failures = 0, 执行可行 | 事件驱动 |

缺任何一关 = 验证不完整，结论不可信。

## Rule 3: 代码修复后必须重新校准门槛

- 2026-02-12 教训：修了 4 个 bug（bounded factor、VEC hysteresis、BT double-gate、sizing commission）后，Rolling Calmar 中位数从 ~0.8 降到 0.22
- 旧门槛 Calmar ≥ 0.80 在新代码下杀死 99% 候选
- 校准方法：跑敏感性分析（sweep 门槛值），找到有效区分度的阈值

## Rule 4: 永远先看 train/holdout 拆分再下结论

- C2 全期 +35.8% 看起来不错，拆开是 train -20.9% + holdout +71.6% = 不可信
- S1 全期 +26.3% 看起来不错，拆开是 train +21.7% + holdout +3.8% = 样本外失效
- **全期收益掩盖了 train/holdout 不一致**，必须拆开看

## Rule 5: 门槛调整必须有敏感性分析支撑

不能拍脑袋改门槛。正确做法：
1. 扫描一组门槛值（如 Calmar 从 0.0 到 0.8）
2. 看每个门槛下通过数和最终存活数
3. 找到"膝盖点"——门槛再严存活数骤降的位置
4. 记录选择依据

## Rule 6: Bug fix + 门槛校准是原子操作

修代码后指标分布会变。修了 bug 但没校准门槛 = 工作只做了一半。
- 每次修复影响策略结果的 bug 后，必须重新跑敏感性分析校准所有门槛
- 检查清单：修 bug → 跑管线 → 看漏斗通过率 → 校准门槛 → 重新验证

## Rule 7: 聚合指标必须拆分到子周期

任何看起来不错的聚合数字都可能掩盖问题：
- 全期收益 = train + holdout 复合，可能方向相反
- 平均 Sharpe = 好周期和差周期的混合
- 总交易胜率 = 前半段 90% + 后半段 40% 的平均
- **永远先拆分，后下结论**

## Rule 8: 设计原则必须落地为代码中的硬性检查

"三好原则"写在文档里但代码里没有 `train_return > 0` gate = 原则不存在。
- 每条设计原则必须对应一个代码中的 assert / gate / check
- 如果找不到对应代码 → 要么补上，要么承认原则没有被执行

## Rule 9: 越好看的结果越要先做 sanity check

C2 被宣布为 successor 的路径：HO +71.9% → Sharpe 2.75 → alpha 68% → shadow deploy ready。每一步都在加固信心，但最基本的 train_return 都没看过。
- **先做 sanity check，再做深入分析**
- Sanity check 清单：train/ho 拆分、漏斗通过率、和基准对比

## Rule 10: 先看数据，后看代码

诊断问题时的优先级：
1. 看数据（train/ho 拆分表）→ 5分钟定位
2. 看漏斗（每关通过率）→ 10分钟定位
3. 看代码（diff sealed vs current）→ 1小时

**错误路径**：先怀疑代码改坏了 → 对比 sealed → 讨论 WFO 排序 → 最后才看 train/ho 拆分
**正确路径**：先看 train/ho 拆分 → 发现不一致 → 审计 gate → 定位缺失

---

## Rule 11: 代码重构/新模块交付前必须完成三层验证

**教训 (2026-02-13)**：factor_registry 重构中，只跑了现有单元测试就声称"验证通过"，
被用户追问后才发现遗漏。现有测试对新代码零覆盖。

三层验证缺一不可：

```
第1层：新代码必须有自己的测试
  "你新建/修改的每个模块，有没有针对性的测试？"
  → 新文件(factor_registry.py) → 必须有 test_factor_registry.py
  → 修改签名(FactorCache.loader) → 必须有测试覆盖新参数
  → 消除同步(bounded_factors 三处→一处) → 必须有一致性断言
  → 现有测试通过 ≠ 新代码被测试了

第2层：真实数据端到端
  "拿1728天×49ETF的生产数据跑一遍，新旧路径输出一致吗？"
  → 不是 print 一下"看起来对"，是 max_diff < 1e-10 的硬断言
  → 数量级也要对：因子数、有效值数、shape 全部核对
  → 意外差异（如56 vs 44因子）必须解释清楚来源

第3层：管线端到端
  "make pipeline 跑得通吗？"
  → WFO → VEC → Rolling → Holdout → Final 全流程无报错
  → 不是"应该能跑"，是"跑完了，这是输出"
```

**反面案例**：
- ❌ "158个测试全过 → 验证通过" — 这158个测试没有一个测 factor_registry
- ❌ "import 没报错 → 代码正确" — import 通过只说明语法对
- ❌ "parquet vs inline 应该一致" — 实际跑了才发现56 vs 44

## Rule 12: 重构交付物清单（逐项打勾才算完）

每次基础设施重构，交付前逐项确认：

```
□ 新文件有对应测试文件
□ 修改的函数签名有测试覆盖新参数
□ 消除的同步点有一致性断言（A == B == C）
□ 真实数据跑过，输出有硬断言验证
□ make pipeline / make test 端到端通过
□ 所有 config 文件同步更新（不只是主 config）
□ CLAUDE.md / MEMORY.md 文档同步
□ git diff 检查无意外文件被修改
```

**核心原则**："说验证了"和"验证了"是两回事。只有带输出的硬断言才算验证。

## Rule 13: 不要用"旧测试全过"代替"新代码测试了"

现有测试套件测的是现有功能的回归。它们不会自动覆盖你刚写的新代码。

判断标准：
- `git diff --name-only` 列出的修改文件，每个都有对应测试覆盖修改点吗？
- 新建的文件，有对应的 `test_*.py` 吗？
- 如果答案是"没有"，那"全部通过"毫无意义。

## Rule 14: 中台化改造必须覆盖全链路，不能只改上游

**教训 (2026-02-13)**：FactorCache 中台化后，WFO 和 VEC 都接入了（23 因子含 non-OHLCV），
但 `run_holdout_validation.py` 和 `run_rolling_oos_consistency.py` 仍直接调用 `PreciseFactorLibrary`（只有 17 个 OHLCV 因子）。
结果：177/200 个含 non-OHLCV 因子的组合在验证阶段被 `KeyError → return None` **静默丢弃**，零日志输出。

```
故障链：
  VEC（FactorCache, 23因子）→ Top 200（177含non-OHLCV）
  → Holdout（PreciseFactorLibrary, 17因子）→ KeyError → return None
  → 只剩 23 个纯 OHLCV 组合
  → 结论"non-OHLCV 没效果" ← 错误！它们从未被评估过
```

**检查清单**：中台/统一入口改造时——
```
□ 列出所有消费方（grep 旧 API 调用）
□ 逐个替换为新 API
□ 确认 0 处旧 API 残留（grep 验证）
□ 端到端跑管线，确认新因子在每个阶段都出现
□ 检查 except/return None 路径——有没有静默吞错误
```

**核心原则**：改了"数据源"但没改"数据消费方" = 改了一半。管线断裂最危险的形式就是静默丢弃——不报错、不告警、结果看起来正常但少了 88% 的数据。

## Rule 15: 静默丢弃是最危险的 bug 模式

任何 `except: return None` / `except: pass` / `except: continue` 都是定时炸弹。

**本次案例**：
```python
try:
    combo_indices = [factor_index_map[f] for f in factors_in_combo]
except KeyError:
    return None  # 177 个组合在这里消失，零日志
```

**修复原则**：
- 所有 `except → 静默跳过` 必须打印警告（至少一行 `print(f"⚠️ skipped: {reason}")`）
- 定期审计管线中的 `except` 语句：`grep -n "except.*:" scripts/*.py`
- 输入数量 vs 输出数量必须在日志中显示（如 "200 combos in → 23 results out"），差距过大自动告警

## Rule 16: 建立在有 bug 管线上的研究结论不可信

**教训 (2026-02-13)**：MEMORY.md 写了 "Non-OHLCV Factor Integration — EXHAUSTED (P0-P4 全部失败)"。
但整个研究期间，管线的验证阶段就没有正确加载过 non-OHLCV 因子。
"不 work"的结论建立在"从未被测试过"的事实之上。

```
错误推理链：
  管线结果：0 个 non-OHLCV 进入最终验证
  → 研究结论："non-OHLCV 因子没效果，方向 EXHAUSTED"
  → 实际原因：验证脚本 KeyError 静默丢弃，non-OHLCV 从未被评估

正确做法：
  结论异常（"所有新因子都没用"）→ 先验证管线本身 → 再下结论
```

**原则**：当研究结论是"全部失败/没效果"时，第一反应不应该是"这个方向走不通"，
而应该是**"是不是管线/测试本身有问题导致了假阴性？"**

## Rule 17: "看起来正常"是最危险的管线状态

管线跑完无报错，输出 23 个结果，有通过有淘汰——一切看起来合理。
但实际上 177/200 个组合被静默丢弃了。

**危险信号识别**：
- 新功能/新因子加了，但结果和之前一模一样 → 新东西可能根本没生效
- 输入 200 输出 23，但没有日志说明为什么少了 177 个
- "所有新因子都没效果" — 如果连一个都没用，更可能是测试有问题

**防御手段**：
- 每个管线阶段打印 `输入 N 个 → 输出 M 个（过滤 K 个）`
- 加一个 non-OHLCV 因子存在性断言：`assert any(non_ohlcv in factor_names for ...)`
- 用户直觉（"有点反常"）永远比管线输出优先——**直觉和数据冲突时，先审计管线**

## Rule 18: 基础设施变更后，第一个检查生产脚本

**教训 (2026-02-13)**：FactorCache 替代 PreciseFactorLibrary 时，研究/验证脚本陆续更新了，
但 `generate_today_signal.py`（**每天跑的生产信号生成器**）从未改过。
它用 `if f not in std_factors: continue` 静默跳过缺失因子——信号降级但不报错。

当前策略（S1/C2）恰好全是 OHLCV 因子，bug 永远不触发。
**直到有一天部署 non-OHLCV 策略，生产信号和回测结果不一致，且零告警。**

```
危险度排序（改完底层后检查顺序）：
  1. 生产脚本（generate_today_signal.py）— 天天跑，出错后果最严重
  2. 验证管线（holdout/rolling）— 决定策略上线，出错导致误判
  3. 回测引擎（BT/VEC）— 研究用，出错浪费时间但不影响生产
  4. 研究脚本 — 一次性分析，影响最小
```

**检查方法**：
```bash
# 变更 FactorCache/CrossSectionProcessor/PreciseFactorLibrary 后立即跑：
grep -rn "PreciseFactorLibrary\|CrossSectionProcessor" scripts/*.py
# 返回 0 行才算迁移完成
```

## Rule 19: 验证脚本必须与 BT 共享执行参数

**教训 (2026-02-13)**：`run_holdout_validation.py` 和 `run_rolling_oos_consistency.py` 调用
`run_vec_backtest()` 时没有传 `delta_rank`/`min_hold_days`，而 `batch_bt_backtest.py` 从
config 读取了这些参数。导致 VEC-BT gap 被人为放大（median 12.1pp → 修复后 8.9pp）。

```
故障链：
  VEC holdout（无hysteresis）→ 过度换仓 → 交易摩擦吃收益
  BT（有hysteresis）→ 正常换仓 → 收益保留
  → #1 候选 VEC_HO +20% vs BT_HO +51%，gap +31pp（修复后 gap +26pp，部分为真实float/int差异）
```

**检查清单**：每个调用 `run_vec_backtest()` 的地方——
```
□ delta_rank 是否从 config.backtest.hysteresis 读取？
□ min_hold_days 是否从 config.backtest.hysteresis 读取？
□ cost_arr 是否与 BT 一致？
□ use_t1_open 是否与 BT 一致？
□ factor_signs / factor_icirs 是否从 WFO 输出传递？（Rule 24）
```

**验证方法**：
```bash
grep -n "run_vec_backtest(" scripts/*.py | grep -v "delta_rank"
# 返回 0 行才算全部对齐
```

## Rule 24: VEC 管线必须传播 factor_signs / factor_icirs 元数据

**教训 (2026-02-14)**：`run_full_space_vec_backtest.py` 只读取 WFO 输出的 `combo` 列，
丢弃 `factor_signs`/`factor_icirs`。下游 holdout/rolling 脚本从 VEC 输出读取，
也拿不到这些列 → 默认 all-positive + equal-weight → 5/6 非OHLCV因子反向使用。

```
故障链：
  WFO 输出（有 factor_signs/factor_icirs）
    → run_full_space_vec_backtest.py 只用 combo 列
      → full_space_results.parquet 无元数据列
        → holdout/rolling 默认 all-positive + equal-weight
          → VEC-BT holdout gap: mean |gap| = 20.3pp（正常应<2pp）
```

**修复**：
1. `run_full_space_vec_backtest.py`: 读取 factor_signs/factor_icirs，pre-multiply 后再调 VEC，
   输出 parquet 包含这两列
2. `run_holdout_validation.py` / `run_rolling_oos_consistency.py`: 已有 pre-multiply 逻辑，
   只要 VEC 输出带列就能自动生效

**修复后**：mean |gap| = 20.3pp → **1.47pp**（-93%），100% 策略 gap<5pp

**验证方法**：
```bash
uv run python -c "import pandas as pd; df=pd.read_parquet('results/vec_from_wfo_*/full_space_results.parquet'); print('factor_signs' in df.columns, 'factor_icirs' in df.columns)"
# 两个都应返回 True
```

## Rule 20: Numba 环境变量必须在 import 前设置

**教训 (2026-02-13)**：`run_robust_combo_wfo.py` 中 `_get_optimal_n_jobs()` 在 `ComboWFOConfig.__post_init__`
里设置 `NUMBA_NUM_THREADS=4`，但在它之前 `FactorCache.get_or_compute()` 已经调用了
`PreciseFactorLibrary` 的 `@njit(parallel=True)` 函数，Numba 线程池以 `os.cpu_count()=32` 启动。
后续 JIT 编译检测到 `NUMBA_NUM_THREADS` 被改 → `RuntimeError`。

```
故障链：
  1. factor_cache.get_or_compute() → cache 未命中
  2. PreciseFactorLibrary @njit(parallel=True) 首次执行 → 线程池以 32 启动
  3. ComboWFOConfig.__post_init__ → os.environ["NUMBA_NUM_THREADS"] = "4"
  4. compute_multiple_ics_numba JIT 编译 → config reload → 32≠4 → RuntimeError
```

**隐蔽性**：
- 直接跑 WFO 时 cache 命中 → 不触发步骤 2 → 永远不报错
- Pipeline 模式下 cache 可能未命中 → 触发 → 必现
- `make clean-numba` 清缓存后更容易复现

**原则**：
- `NUMBA_NUM_THREADS` 必须在 `import numba` 之前设置（`os.environ` 在文件顶部，import 语句之前）
- 一旦任何 `@njit(parallel=True)` 函数被调用，线程池就固化，不可更改
- 设置 Numba 环境变量的代码不能放在类的 `__init__`/`__post_init__` 里——太晚了

**修复模式**：
```python
# 文件最顶部，在任何 from xxx import 之前
import os
if not os.getenv("NUMBA_NUM_THREADS"):
    cpu_count = os.cpu_count() or 8
    n_jobs = min(cpu_count // 2, 8)
    os.environ["NUMBA_NUM_THREADS"] = str(max(1, cpu_count // n_jobs))

# 然后才 import 其他模块
import numpy as np
from etf_strategy.core.xxx import ...
```

## Rule 21: Cache hit/miss 改变代码路径 → 隐藏 bug

**教训 (2026-02-13)**：FactorCache 命中时跳过因子计算（含 Numba 并行函数），未命中时触发。
同一个脚本在 cache 命中时正常、cache 未命中时崩溃。

**原则**：
- 端到端测试必须覆盖 cold start（清缓存后至少跑一次 `make pipeline`）
- Cache 命中和未命中是**两条不同的代码路径**，都需要测试
- `make pipeline` ≠ 直接跑各步骤（环境变量传递、cache 状态、执行顺序都不同）

**检查方法**：
```bash
# 清缓存后跑 pipeline，确认不崩溃
make clean-numba
rm -rf .cache/factor_cache_*.pkl  # 可选：也清因子缓存
uv run python scripts/run_full_pipeline.py
```

## Rule 22: 路径依赖系统中信号态 ≠ 执行态，必须用执行态驱动状态逻辑

**教训 (2026-02-14)**：BT 引擎 Exp4.1 用 `_signal_portfolio`（滞后函数的输出）构建下次滞后的
`hmask`，形成自引用反馈环。VEC 用 `holdings[]`（执行态）。微小的 float/int 差异在信号态中
不被修正，链式放大到 +25.7pp gap（33 次多余交易）。切换为执行态后 gap 降到 -0.8pp。

**原则**：
- 滞后/动量/持有天数等状态依赖逻辑，**必须从执行态（actual holdings）读取**
- 信号态（signal portfolio）可用于日志/诊断，但**不能作为下一次信号生成的输入**
- 在路径依赖系统中，输出→输入的反馈环是 bug 的温床

**诊断捷径**：
- Hyst OFF gap < 5pp → 引擎基本对齐，问题不在引擎
- Hyst ON gap > 10pp → 大概率是状态追踪分歧，不是 float/int 差异
- 用消融法 > 逐行 debug（快 10x）

**详细分析**：`memory/2026-02_vec_bt_alignment_fix.md`

## Rule 25: 封板前 VEC-BT gap 是硬性前置检查

**教训 (2026-02-15)**：v7.0 封板时 VEC-BT gap +25.6pp，远超 10pp 红线，但没有系统性检查。
v8.0 之前修了三个管线 bug 后，155 候选全部 gap < 2pp。

**原则**：
- 封板前**必须**验证所有候选策略的 VEC-BT holdout gap
- Gap > 10pp = **不允许封板**，必须先排查管线问题
- Target gap < 5pp（正常的 float/int 差异范围）
- Gap 方向也要关注：BT > VEC（安全，实际执行不劣于预期）vs VEC > BT（需调查）

**检查方法**：
```python
# 封板前运行
gap = bt_holdout_return - vec_holdout_return
assert abs(gap) < 0.10, f"VEC-BT gap {gap:.1%} exceeds 10pp red flag"
```

## Rule 26: 修 bug 后不要"修一个封一版"

**教训 (2026-02-11~15)**：5天封了 v5.0/v6.0/v7.0/v8.0 四个版本，前三个全废弃。
每次只修一个 bug 就急于封板，结果下一个 bug 暴露后又要废弃。

**原则**：
- 发现 pipeline bug 后，**先全部修完**，再封板
- Bug 是乘法不是加法 — 3个小问题组合后致命
- 封板是昂贵操作（locked/ 快照 + CHECKSUMS + SEAL_SUMMARY），不应频繁执行
- 正确节奏：修所有已知 bug → full pipeline 端到端 → 全量 BT → VEC-BT gap 检查 → 封板

**反面案例**：
```
v5.0 (02-11) → 废弃 (02-12): ADX Winsorize bug
v6.0 (02-12) → 从未使用: C2 train gate fail
v7.0 (02-13) → 废弃 (02-15): IC-sign + metadata + exec-side bugs
v8.0 (02-15) → 首个 clean seal（三个 bug 全修后）
```

## Rule 23: shadow accounting 对交易决策的影响远小于状态追踪

**教训 (2026-02-14)**：`sizing_commission_rate = 50bp`（max rate）vs 实际 20bp（A股），
理论上每笔多估 30bp。实际影响：仓位估算差 ~0.3%，零交易决策变化。
因为 `broker.getcash()` 每次 rebalance 重置为真实现金，shadow 误差不累积。

**原则**：
- 排查 VEC-BT gap 时，先看**状态追踪**（hmask/hdays 来源），后看 shadow 层
- shadow commission 差异只影响单次仓位精度，不改变交易决策
- 真正改变交易决策的是：状态分歧 → 不同的 hmask → 不同的 swap 决策 → 链式分歧

---

## Rule 27: 因子窗口必须匹配执行频率

**教训 (2026-02-16 Phase 1)**：SHARE_CHG_5D (WFO rank #14) 远优于 SHARE_CHG_10D (rank #6873)，
-67pp 差距。本质原因：FREQ=5 → 每 5 天决策一次 → 5D 窗口恰好覆盖一个决策周期，
10D 混入上个周期的陈旧信息，信噪比下降。

**原则**：
- 因子回看窗口应与执行频率对齐：FREQ=5 → 优先 5D/10D 窗口，20D 次之，60D/120D 仅限慢变量
- 快变因子（动量、资金流）：窗口 ≈ FREQ → 1-2 个执行周期
- 慢变因子（位置、趋势质量）：窗口 = 多个 FREQ → 提供中长期背景
- 修改 FREQ 后需重新评估所有因子窗口的适配性

## Rule 28: 同源数学变换 ≠ 新信息

**教训 (2026-02-16 Phase 1)**：SHARE_ACCEL（份额变化的二阶导数）提升 Rolling 稳定性（83% vs 61%）
但降低 Sharpe（1.12 vs 1.38）。平滑不是新 alpha。

**原则**：
- 一阶导数 (SHARE_CHG) 和二阶导数 (SHARE_ACCEL) 编码的是**同一个数据源的不同尺度**，不是独立信息
- 数学变换（差分、加速度、EMA、Z-score）不创造信息正交性
- 真正的新 alpha 来自**不同数据源**，不是同一数据的不同处理方式
- 判断标准：新因子与现有 top-5 PC 的 correlation。同源变换通常 > 0.5

## Rule 29: 可信的负结论比可疑的正结论更有价值

**教训 (2026-02-16 Phase 1)**：Phase 1 "无提升"是项目首个在干净管线上的可信负结论。
此前 "Non-OHLCV all failed" 是假阴性（管线 bug），"10x 加速" 是假阳性（未 benchmark）。

**原则**：
- 负结论的前提：管线无已知 bug + 四关验证完整 + 搜索空间充分
- 在此前提下，"此路不通"比"可能有效但没验证"更有决策价值
- **停止在已饱和空间内优化（opportunity cost），转向新信息源**
- 每次得到负结论，先回溯检查 Rule 16（是管线问题还是真的没用？）

## Rule 30: Alpha = Price × Contrarian Flow

**来源 (2026-02-16)**：v8.0 两个策略家族的共同模式分析。

**模式**：
- composite_1: 价格突破 + 散户/杠杆流出 → 机构主导趋势
- core_4f: 价格上升 + 融资/份额双流出 → 信念驱动趋势
- **共性：价格信号 × 逆向资金流 = 有效 alpha**

**原则**：
- 只有价格 → 不知道趋势是谁驱动的（纯 OHLCV 占 Top200 仅 2%）
- 只有资金流 → 不知道流出是止损还是获利了结
- 交叉验证 → "价格涨 + 散户跑" = 高质量趋势信号
- **新因子的价值 = 它能否更精准地区分"谁在交易"**

## Rule 31: 不同管道观测同一现象 ≠ 新信息维度

**教训 (2026-02-17)**：Tushare moneyflow（按订单大小分档）IC=0.122, ICIR=0.448，
看似很强，但与 SHARE_CHG_5D 截面 rank 相关 rho=-0.577 → 同维度。

**Rule 28 vs Rule 31**：
```
Rule 28: 同一数据源的数学变换（SHARE_CHG → SHARE_ACCEL）
Rule 31: 不同数据源观测同一底层现象（moneyflow vs fund_share vs margin）
```

两者的共性：新"因子"看起来不同，但编码的是**同一种信息**。

**判断方法**：
- 计算新因子与现有因子的截面 rank 相关性
- |rho| < 0.3 → 正交，值得投入
- |rho| 0.3~0.5 → 部分正交，评估边际成本
- |rho| > 0.5 → 同维度，放弃

**本项目已覆盖的信息维度**：
```
1. Price/Trend:    SLOPE, BREAKOUT, PP120, ADX — OHLCV 完全覆盖
2. Volume:         OBV_SLOPE, UP_DOWN_VOL — OHLCV 完全覆盖
3. InvestorFlow:   SHARE_CHG_5D, MARGIN_BUY, MARGIN_CHG — fund_share+margin 已饱和
                   (moneyflow 也是此维度，rho=0.58，放弃)

未覆盖维度（真正的 alpha 机会）：
4. WHO:            北向资金（个股级别，非市场聚合）— Tushare 数据不支持
5. Forward-looking: 期权IV — 仅覆盖 2-3 标的
6. Cross-asset:    汇率 — A_SHARE_ONLY 下截面区分度存疑
7. Microstructure: IOPV折溢价 — 需盘中采集
```

**详细分析**：`memory/2026-02_moneyflow_validation.md`

---

## Rule 32: POS_SIZE 不是自由参数

> **策略在 POS_SIZE=N 下优化，降至 POS_SIZE=M<N 时性能崩塌。Ensemble via Capital Split 必须验证各子策略在目标 PS 下独立可行。**

**发现过程** (2026-02-17):
- composite_1 (PS=2): Sharpe 1.589, MDD 10.8%
- composite_1 (PS=1): Sharpe 0.402, MDD 20.2% → **-75% Sharpe degradation**
- core_4f (PS=2): Sharpe 1.225 → (PS=1): Sharpe 0.179 → **-85% degradation**
- Capital Split blend: Sharpe 0.363, worse than both standalone PS=2

**原因**：
- 单持仓 → 集中度100%, 无法分散选股误差
- Hysteresis 动态改变（fewer swap options）
- top-2 多样化选股 alpha 丧失

**推论**：
- 若考虑 ensemble，每个子策略必须在目标 PS 下独立验证
- PS=2 → PS=1 的降级不是线性的，而是非线性崩塌
- 这适用于所有"分资金给多策略"方案

---

## Rule 33: 截面收益离散度 ⊂ 市场波动率

> **A股 ETF 宇宙中，截面收益离散度与市场波动率 rho=0.538，属于同一信息维度。高波动环境本质上产生高收益离散。不可作为独立 WHEN 信号。**

**验证数据** (2026-02-17):
- 20D 离散度 vs regime vol: Pearson=0.538, Spearman=0.428
- 5D 离散度 vs regime vol: Pearson=0.369, Spearman=0.392
- 离散度对策略下期收益的 Pearson rho: 全部 <0.05, p>0.3
- Quartile 无单调关系（5D/20D × Train/HO 全部 NONE）

**原因**：
- 高波动=价格大幅变动=ETF间回报差异自然放大
- 这是数学恒等式而非经济信号
- Regime gate 已捕获波动率维度的全部可用信息

**详细报告**：`docs/research/when_how_dimension_research_20260217.md`
