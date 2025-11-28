<!-- ALLOW-MD -->
# 机器配置文档

**文档创建时间**: 2025-11-26  
**目的**: 记录开发环境硬件配置，确保项目充分利用机器性能

---

## 🖥️ 硬件配置

### CPU
- **型号**: AMD Ryzen 9 9950X 16-Core Processor
- **物理核心数**: 16 cores
- **逻辑核心数**: 32 threads (支持 SMT)
- **架构**: Zen 5 (2024 旗舰级)
- **性能特点**: 
  - 高主频 + 高核心数，适合多线程计算
  - 支持 AVX-512 指令集，NumPy/Pandas 性能优秀

### GPU
- **型号**: NVIDIA GeForce RTX 5070 Ti
- **显存**: 16 GB GDDR7
- **驱动版本**: 580.95.05
- **CUDA**: 支持 (需安装 PyTorch with CUDA)
- **性能特点**:
  - Blackwell 架构 (2025 最新一代)
  - 适合深度学习训练和推理
  - 16GB 显存足够处理大规模因子计算

### 内存
- **总容量**: 48 GB DDR5
- **可用内存**: ~31 GB (扣除系统占用)
- **Swap**: 8 GB (位于 /swap.img)
- **建议**: 
  - 当前配置对大规模回测已足够
  - 如需处理更大数据集，可考虑升级到 64GB

### 存储
- **设备**: NVMe SSD (nvme0n1p2)
- **总容量**: 1.9 TB
- **已用**: 271 GB (16%)
- **可用**: 1.5 TB
- **建议**: 空间充足，无需扩容

---

## 💻 软件环境

### 操作系统
- **发行版**: Ubuntu 24.04.3 LTS (Noble Numbat)
- **内核版本**: 6.14.0-36-generic (最新稳定内核)
- **支持**: LTS 版本，至 2029 年

### Python 环境
- **Python 版本**: 3.12.3
- **包管理器**: `uv` (高性能 Rust 实现)
- **虚拟环境**: 使用 `uv` 管理项目依赖

### 核心依赖版本
```
numpy            2.3.4     # 数值计算核心
pandas           2.3.3     # 数据分析主力
scikit-learn     1.7.2     # 机器学习框架
polars           1.33.1+   # 高性能 DataFrame
vectorbt         0.28.1+   # 量化回测加速
ta-lib           0.6.7+    # 技术指标库
```

### 开发工具
- **Git**: 已安装
- **UV**: 已安装 (快速依赖管理)
- **VS Code Server**: 配置代理 (setup_vscode_proxy.sh)

---

## ⚡ 性能优化建议

### 1. CPU 并行优化
当前 32 线程配置，建议设置以下环境变量：

```bash
# NumPy/OpenBLAS 线程数 (建议: 物理核心数 = 16)
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

# Pandas 并行处理
export PANDAS_COMPUTE_THREADS=16

# Polars 并行 (默认自动检测，但可手动设置)
export POLARS_MAX_THREADS=32

# Python multiprocessing
# 在代码中使用: os.cpu_count() 返回 32
```

**实践指导**:
- **I/O 密集型任务**: 使用全部 32 线程 (如数据下载、文件读写)
- **CPU 密集型任务**: 使用 16 物理核心 (避免 SMT 竞争)
- **混合任务**: 根据实际测试调整 (建议 20-24 线程)

### 2. 内存优化
48GB 配置建议：

```python
# 回测时分批处理
BATCH_SIZE = 1000  # 每批处理股票数
CHUNK_SIZE = 10000  # DataFrame 分块大小

# 使用 Polars 替代 Pandas (内存效率高 2-5x)
import polars as pl
df = pl.read_csv("large_file.csv", low_memory=True)

# 及时释放不用的变量
import gc
del large_dataframe
gc.collect()
```

### 3. GPU 加速 (可选)
RTX 5070 Ti 配置建议：

```bash
# 安装 PyTorch with CUDA 12.x
uv pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu124

# 验证 CUDA 可用性
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

**适用场景**:
- ✅ 深度学习因子训练 (NN, LSTM, Transformer)
- ✅ 大规模矩阵运算 (协方差计算、PCA)
- ❌ 传统量价因子计算 (CPU 已足够高效)

### 4. 存储 I/O 优化
NVMe SSD 配置：

```python
# 使用高效数据格式
df.to_parquet("data.parquet")  # 比 CSV 快 10x，体积小 80%
df.to_feather("data.feather")  # 零序列化开销

# 异步 I/O (配合 asyncio)
import aiofiles

# 批量读取避免磁盘碎片
files = glob.glob("data/*.parquet")
dfs = [pl.read_parquet(f) for f in files]
combined = pl.concat(dfs)
```

---

## 🎯 项目性能基准

基于当前硬件配置，预期性能指标：

| 任务类型 | 数据规模 | 预期耗时 | 备注 |
|---------|---------|---------|------|
| 因子计算 (154 factors) | 1000 股票 × 2000 天 | ~30 秒 | 使用 Polars + 多线程 |
| WFO 回测 (12 窗口) | 50 股票 × 10 年 | ~5 分钟 | 全 Vectorized |
| 完整生产运行 | All ETFs | ~15 分钟 | 包含因子 + 回测 + 优化 |
| 深度学习训练 | 10M 样本 | ~1 小时 | GPU 加速 (RTX 5070 Ti) |

**瓶颈分析**:
- ✅ CPU: 32 线程充足，暂无瓶颈
- ✅ 内存: 48GB 足够常规任务
- ⚠️ I/O: 1.9TB NVMe 已优秀，但大数据集建议用 Parquet 格式
- ⚠️ GPU: 当前未充分利用，可探索深度学习策略

---

## 🔧 环境配置脚本

### 快速设置性能参数

创建 `.env` 文件（项目根目录）:

```bash
# CPU 并行优化
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export OMP_NUM_THREADS=16

# Polars 并行
export POLARS_MAX_THREADS=32

# NumPy/Pandas 行为
export NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=1
export PANDAS_COMPUTE_THREADS=16

# 回测稳定性标志 (参考 .github/copilot-instructions.md)
export RB_STABLE_RANK=1
export RB_DAILY_IC_PRECOMP=1

# GPU 相关 (如已安装 CUDA)
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 加载配置

```bash
# 在 shell 启动时自动加载
echo 'source /home/sensen/dev/projects/-0927/.env' >> ~/.bashrc

# 或在项目中手动加载
source .env
```

---

## 📊 性能监控

### 实时监控命令

```bash
# CPU/内存监控
htop

# GPU 监控
watch -n 1 nvidia-smi

# 磁盘 I/O 监控
iotop

# 网络监控 (如使用远程数据)
iftop
```

### 性能分析

```bash
# Python 代码性能分析
export RB_PROFILE_BACKTEST=1
python etf_rotation_optimized/run_combo_wfo.py --quick

# 查看性能报告
ls -lh *.prof
python -m pstats output.prof
```

---

## 🚀 升级建议 (未来)

根据项目发展，可考虑以下升级路径：

1. **短期 (6 个月内)**:
   - ✅ 当前配置已足够，无需升级
   - 可选: 安装 PyTorch GPU 版本 (如需深度学习)

2. **中期 (1-2 年)**:
   - 内存升级到 64GB (如处理全市场 5000+ 股票)
   - 增加第二块 GPU (如需分布式训练)

3. **长期 (2+ 年)**:
   - 考虑 Threadripper PRO (64 核心) 或服务器平台
   - 组建计算集群 (多机分布式回测)

---

## 📝 维护日志

| 日期 | 变更内容 | 备注 |
|------|---------|------|
| 2025-11-26 | 初始配置记录 | AMD 9950X + RTX 5070 Ti |

---

**最后更新**: 2025-11-26  
**维护者**: Quant Team  
**参考文档**: `.github/copilot-instructions.md`, `docs/OUTPUT_SCHEMA.md`
