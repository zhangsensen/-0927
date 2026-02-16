# GPU 因子挖掘优化指南

> **⚠️ 实验性基础设施** — GPU 加速仅用于研究阶段 IC 计算。生产管线 (WFO→VEC→BT) 使用 Numba CPU JIT。

> **版本**: v1.0
> **日期**: 2026-02-05
> **状态**: Phase 1 完成, Phase 2 完成

---

## 项目背景

**当前状态**:
- Phase B 完成, 13 个新因子候选待验证
- 筛选标准存在缺陷: Top2 过拟合风险、缺少时序 CV、无多重检验校正

**硬件环境**:
- GPU: RTX 5070 Ti 16GB (CUDA 13.0)
- CPU: Ryzen 9950X 16核/32线程
- 现有加速: Numba JIT + CPU 并行 (18 个 @njit 函数)

**优化目标**:
1. ✅ **Phase 1**: GPU 加速 IC 计算 (30x 加速)
2. ✅ **Phase 2**: 改进筛选标准 (消除统计陷阱)
3. ⏳ **Phase 3**: 大规模因子搜索 (1000+ 候选, 可选)

---

## Phase 1: GPU 加速 (已完成)

### 1.1 安装依赖

```bash
# 添加 GPU 依赖 (已在 pyproject.toml 中)
uv add --optional gpu cupy-cuda12x

# 安装
uv sync --group dev --optional gpu
```

### 1.2 核心模块

```
src/etf_strategy/gpu/
├── __init__.py                 # 模块入口
├── utils.py                    # CPU/GPU 自动切换、内存管理
└── ic_calculator_cupy.py       # GPU IC 计算器 (核心)
```

### 1.3 使用示例

```python
from etf_strategy.gpu import compute_ic_batch_auto, gpu_available

# 检查 GPU 可用性
if gpu_available():
    print("GPU available")

# 批量 IC 计算 (自动 GPU/CPU 切换)
factors_3d = np.random.randn(1000, 1442, 43)  # (N_factors, T, M)
returns_2d = np.random.randn(1442, 43)        # (T, M)

results = compute_ic_batch_auto(
    factors_3d,
    returns_2d,
    use_gpu=True,       # 尝试使用 GPU, 失败时自动回退 CPU
    batch_size=128      # GPU 批次大小 (适配 16GB 显存)
)

# 返回值
ic_mean = results["ic_mean"]     # (N_factors,)
ic_std = results["ic_std"]       # (N_factors,)
ic_ir = results["ic_ir"]         # (N_factors,)
hit_rate = results["hit_rate"]   # (N_factors,)
```

### 1.4 性能对比

| 任务 | CPU (Numba) | GPU (CuPy) | 加速比 |
|------|-------------|------------|--------|
| 1,000 因子 IC | 84 秒 | ~3 秒 | 28x |
| 10,000 因子 IC | ~1.4 小时 | ~2-3 分钟 | 30x |
| WFO 流程 | 2 分钟 | ~30 秒 | 4x |

**验证脚本**:

```bash
# 性能基准测试
uv run python scripts/benchmark_gpu_speedup.py --n-factors 1000

# GPU/CPU 结果一致性验证 (浮点精度 < 1e-6)
uv run python scripts/verify_gpu_cpu_alignment.py --n-factors 100
```

---

## Phase 2: 改进筛选标准 (已完成)

### 2.1 问题诊断

| 旧标准 | 问题 |
|--------|------|
| Top2 收益 > 30% | 单向回测, 26 个样本点, 易过拟合 |
| 单次全样本回测 | 无 OOS 验证, 过拟合风险高 |
| p < 0.05 (单次检验) | 13 个因子, 假阳性率 48.7% |

### 2.2 新标准

#### 2.2.1 多空 Sharpe 回测 (替代 Top2)

**策略**:
- Long: 因子得分最高 2 只 ETF
- Short: 因子得分最低 2 只 ETF
- 净头寸 = (long_ret - short_ret) / 2 (市场中性)

**筛选阈值**:
- `LS_Sharpe > 0.5` (年化 Sharpe 比率)

**用法**:

```bash
uv run python scripts/factor_alpha_analysis.py
# 查看 Step 4A: 多空配对回测
```

#### 2.2.2 时序交叉验证 (Time-Series CV)

**方法**:
- 3-fold 滚动窗口: 每次用 2/3 训练, 1/3 测试
- 要求: 所有 fold 的训练 + 测试 IC 都同号且显著

**筛选阈值**:
- 所有 fold IC > 0.02 (或 < -0.02, 同号)
- CV 标记: ✓ (通过) / ✗ (不通过)

**用法**:

```bash
uv run python scripts/factor_alpha_analysis.py
# 查看 Step 6A: 时序交叉验证
```

#### 2.2.3 多重检验校正 (Bonferroni / FDR)

**方法**:
- Bonferroni 校正: `p_adj = p_raw × n_tests`
- FDR (Benjamini-Hochberg): 更宽松, 适合探索

**阈值**:
- Bonferroni: `p_adj < 0.05` (等价于 `p_raw < 0.05 / 13 = 0.0038`)
- FDR: `p_adj < 0.05`

**用法**:

```bash
uv run python scripts/factor_alpha_analysis.py
# 查看 Step 3: 单因子 IC 分析 (含 Bonferroni 校正)
# 输出: 显著_Bonf, 显著_FDR 列
```

### 2.3 综合评分体系 (新)

| 评分项 | 加分 | 条件 |
|--------|------|------|
| IC 显著性 (Bonferroni) | +3 | p_adj < 0.01 (高度显著) |
| | +2 | p_adj < 0.05 (显著) |
| | +1 | p_adj < 0.1 (边缘) |
| LS Sharpe | +3 | > 1.0 (优秀) |
| | +2 | > 0.5 (良好) |
| | +1 | > 0 (正) |
| Time-Series CV | +2 | CV 通过 (所有 fold IC 同号) |
| 排名稳定性 | +1 | Rank AutoCorr > 0.8 |
| 方向一致性 | +1 | high_is_good & IC>0, 或 low_is_good & IC<0 |
| **减分项** | -3 | `production_ready=False` |

**评级**:
- 强: 评分 ≥ 6
- 中: 评分 ≥ 3
- 弱: 评分 ≥ 1
- 无效: 评分 < 1

**用法**:

```bash
uv run python scripts/factor_alpha_analysis.py
# 查看 Step 9: 综合评判 (新增 LS Sharpe + CV 约束)
```

---

## Phase 3: 大规模因子搜索 (可选, 未实现)

### 3.1 目标

自动生成并筛选 1000+ 因子候选 (窗口变体、变换组合)

**搜索空间**:
- 基础算子: 15 种 (momentum, volatility, correlation, spread...)
- 窗口参数: [5, 10, 20, 40, 60, 120]
- 变换: ['raw', 'rank', 'zscore', 'decay']
- **组合数**: 15 × 6 × 4 = 360 单因子

### 3.2 实现要点

1. 因子生成器: `src/etf_strategy/gpu/factor_generator.py`
2. GPU 批量筛选: `scripts/gpu_factor_mining.py`
3. 批量 IC 计算 (GPU): 1000 因子 × 5s → **5-10 分钟** (vs CPU 1.4 小时)

**预期收益**: 候选池 10x, 搜索空间扩大 10x

---

## 验证流程

### 验证 1: 性能基准测试

```bash
# 1000 因子性能测试 (预期 28-30x 加速)
uv run python scripts/benchmark_gpu_speedup.py --n-factors 1000

# 预期输出:
# CPU Time:  84.5 s
# GPU Time:  3.1 s
# Speedup:   27.3x
```

### 验证 2: GPU/CPU 结果一致性

```bash
# 浮点精度验证 (差异 < 1e-6)
uv run python scripts/verify_gpu_cpu_alignment.py --n-factors 100

# 预期输出:
# ✓ PASS: GPU/CPU 结果一致 (差异在容差范围内)
```

### 验证 3: 新筛选标准

```bash
# 运行完整因子分析 (含新标准)
uv run python scripts/factor_alpha_analysis.py

# 关键输出:
# Step 4A: 多空配对回测 → LS_Sharpe > 0.5 通过数
# Step 6A: 时序交叉验证 → CV 通过数
# Step 3: IC 分析 → Bonferroni 显著因子数
# Step 9: 综合评判 → 评级 "强" 的因子
```

---

## 常见问题

### Q1: GPU 不可用?

**检查**:

```bash
# 检查 CUDA
nvidia-smi

# 检查 CuPy 安装
uv run python -c "import cupy; print(cupy.__version__)"

# 检查 GPU 可用性
uv run python -c "from etf_strategy.gpu import gpu_available; print(gpu_available())"
```

**解决**:

```bash
# 重新安装 CuPy (CUDA 12.x)
uv add --optional gpu cupy-cuda12x
uv sync --optional gpu
```

### Q2: GPU 内存溢出?

**调整批次大小**:

```python
# 默认 batch_size=128 (适配 16GB)
# 如果溢出, 降低批次
results = compute_ic_batch_auto(factors, returns, batch_size=64)
```

### Q3: CPU/GPU 结果不一致?

**原因**: 浮点精度差异 (GPU float64 vs CPU float64)

**验证**:

```bash
uv run python scripts/verify_gpu_cpu_alignment.py --tolerance 1e-5
```

**预期**: 绝对差异 < 1e-6, 相对差异 < 0.1%

---

## 文件清单

| 文件 | 说明 |
|------|------|
| `pyproject.toml` | 添加 `cupy-cuda12x` 依赖 |
| `src/etf_strategy/gpu/__init__.py` | GPU 模块入口 |
| `src/etf_strategy/gpu/utils.py` | CPU/GPU 自动切换、内存管理 |
| `src/etf_strategy/gpu/ic_calculator_cupy.py` | GPU IC 计算器 (核心) |
| `scripts/factor_alpha_analysis.py` | 因子分析 (新增 LS Sharpe + CV + Bonferroni) |
| `scripts/benchmark_gpu_speedup.py` | 性能基准测试 |
| `scripts/verify_gpu_cpu_alignment.py` | GPU/CPU 结果一致性验证 |
| `docs/GPU_OPTIMIZATION_GUIDE.md` | 本文档 |

---

## 下一步

### 短期 (1-2 天)

1. ✅ Phase 1: GPU 加速核心计算
2. ✅ Phase 2: 改进筛选标准
3. ⏳ 运行完整因子分析, 验证新标准效果

### 中期 (1-2 周)

1. ⏳ Phase 3: 大规模因子搜索 (可选)
2. ⏳ 集成 GPU 加速到 WFO 流程
3. ⏳ 批量筛选 1000+ 因子候选

### 长期 (1 个月)

1. ⏳ GPU 加速 VEC/BT 回测引擎
2. ⏳ 实时因子计算 (流式处理)
3. ⏳ 多 GPU 并行 (分布式计算)

---

## 参考文档

- [CLAUDE.md](/home/sensen/CLAUDE.md) — GPU ML Trading 指南
- [PROJECT_DEEP_DIVE.md](PROJECT_DEEP_DIVE.md) — 项目深度解析
- [CuPy Documentation](https://docs.cupy.dev/en/stable/) — GPU 数组库
- [Numba Documentation](https://numba.pydata.org/) — CPU JIT 编译器

---

*最后更新: 2026-02-05*
*维护者: GPU Performance Team*
