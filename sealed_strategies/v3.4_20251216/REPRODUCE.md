# v3.4 Production Strategy - Reproduction Guide

**Goal**: 任何人（包括大模型）拿到本封板版本后，能在 30 分钟内从零复现完整回测结果。

---

## 📋 Prerequisites

### 1. System Requirements
- **OS**: Linux (Ubuntu 20.04+ / CentOS 7+) or macOS
- **Python**: 3.11+
- **UV**: 0.1.0+ (Python package manager)
- **RAM**: 8GB+ (因子计算需要内存)
- **Disk**: 5GB+ (数据 + 缓存)

### 2. Install UV (如果未安装)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
uv --version  # 验证安装
```

---

## 🚀 Step-by-Step Reproduction

### Step 1: 解压封板包（如果是压缩包）
```bash
cd /home/sensen/dev/projects/-0927/sealed_strategies
tar -xzf v3.4_20251216.tar.gz
cd v3.4_20251216
```

### Step 2: 验证完整性
```bash
sha256sum -c CHECKSUMS.sha256
```
**预期输出**: 所有文件 `OK`

### Step 3: 进入锁定目录
```bash
cd locked
```

### Step 4: 安装依赖（自动创建虚拟环境）
```bash
uv sync --dev
```
**时间**: ~2 分钟  
**说明**: UV 会自动创建 `.venv/` 并安装 `pyproject.toml` 中的所有依赖

> ⚠️ **重要说明**: 
> - `.venv/` 虚拟环境**不包含在封板包中**（会导致包体积 >1GB）
> - 封板包只包含 `pyproject.toml` 和 `uv.lock` 配置文件
> - 用户需在本地运行 `uv sync` 自动生成虚拟环境
> - 这确保了环境的可复现性，同时保持封板包轻量（~15MB）

### Step 5: 验证环境
```bash
uv run python -c "import pandas, numpy, backtrader; print('✅ Environment Ready')"
```

### Step 6: 准备数据

#### Option A: 使用已有数据（推荐）
如果封板包中已包含 `raw/ETF/daily/` 数据：
```bash
# 验证数据完整性
uv run python -c "
import os
data_dir = '../../../raw/ETF/daily'
files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
print(f'✅ Found {len(files)} ETF data files')
assert len(files) >= 43, 'Missing ETF data!'
"
```

#### Option B: 从 QMT Bridge 下载（需要网络连接）
```bash
# 配置 QMT Bridge 连接
export QMT_HOST="192.168.122.132"
export QMT_PORT="8001"

# 下载所有 43 只 ETF 数据
uv run python scripts/update_daily_from_qmt_bridge.py --all
```
**时间**: ~5 分钟

### Step 7: 运行回测审计
```bash
uv run python scripts/batch_bt_backtest.py \
  --candidates ../artifacts/production_candidates.csv \
  --start 2020-01-01 \
  --end 2025-12-12 \
  --output results/bt_reproduction_$(date +%Y%m%d_%H%M%S).parquet
```
**时间**: ~3-5 分钟（2 策略）  
**预期输出**:
```
✅ Strategy #1: Total Return = 136.52%, Sharpe = 1.26, MaxDD = 15.47%
✅ Strategy #2: Total Return = 129.85%, Sharpe = 1.22, MaxDD = 13.93%
```

### Step 8: 验证结果一致性
```bash
uv run python -c "
import pandas as pd
df = pd.read_parquet('results/bt_reproduction_*.parquet')
ref = pd.read_csv('../artifacts/production_candidates.csv')

# 允许 0.1pp 浮点误差
assert abs(df.iloc[0]['total_return'] - 136.52) < 0.1, 'Strategy #1 mismatch!'
assert abs(df.iloc[1]['total_return'] - 129.85) < 0.1, 'Strategy #2 mismatch!'
print('✅ Reproduction Successful: Results match sealed version')
"
```

---

## 🔍 Advanced Reproduction (Optional)

### 1. 从 WFO 重新挖掘因子组合
```bash
# 运行滚动 WFO（需要 ~30 分钟）
uv run python src/etf_strategy/run_combo_wfo.py

# 输出: results/run_YYYYMMDD_HHMMSS/wfo_results.parquet
```

### 2. VEC 批量回测（验证 Top 候选）
```bash
uv run python scripts/run_full_space_vec_backtest.py \
  --wfo-dir results/run_latest

# 输出: results/vec_backtest_YYYYMMDD_HHMMSS/vec_results.parquet
```

### 3. 三重验证（Rolling + Holdout）
```bash
uv run python scripts/final_triple_validation.py \
  --vec-results results/vec_backtest_latest/vec_results.parquet

# 输出: results/final_triple_validation_YYYYMMDD_HHMMSS/final_candidates.parquet
```

### 4. BT 审计（Ground Truth）
```bash
uv run python scripts/batch_bt_backtest.py \
  --candidates results/final_triple_validation_latest/final_candidates.parquet \
  --top 5
```

---

## 🧪 Unit Tests (Optional)

运行完整测试套件：
```bash
uv run pytest tests/ -v
```

**预期通过**: 20 tests (VEC/BT 对齐测试)

---

## 📊 Generate Production Reports

### 交易员视角报告
```bash
uv run python scripts/report_v3_3_portfolio_trader_view.py \
  --candidates ../artifacts/production_candidates.csv \
  --output reports/trader_report_$(date +%Y%m%d).md
```

### 最近 60 天交易分析
```bash
uv run python scripts/analyze_recent_divergence.py \
  --candidates ../artifacts/production_candidates.csv \
  --days 60 \
  --output reports/recent_trades_$(date +%Y%m%d).json
```

---

## ⚠️ Troubleshooting

### 问题 1: UV 安装失败
```bash
# 手动安装 Python 3.11+
sudo apt install python3.11 python3.11-venv

# 使用 pip 创建环境（备用方案）
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 问题 2: 数据下载失败（QMT Bridge 不可达）
**解决方案**: 
1. 检查 QMT VM 是否启动 (`ssh user@192.168.122.132`)
2. 验证端口开放 (`curl http://192.168.122.132:8001/health`)
3. 如果网络隔离，使用封板包中已包含的数据

### 问题 3: 回测结果不一致
**可能原因**:
- 数据文件不完整（缺少某些日期）
- Python 随机种子未固定（`numpy.random.seed(42)`）
- Backtrader 版本不匹配（`uv pip list | grep backtrader`）

**诊断命令**:
```bash
# 检查数据完整性
uv run python scripts/verify_data_integrity.py

# 对比校验和
sha256sum locked/src/etf_strategy/core/precise_factor_library_v2.py
```

### 问题 4: 内存不足
**解决方案**:
- 减少并行度: `export NUMBA_NUM_THREADS=1`
- 分批回测: `--top 1` 逐个策略运行

---

## 📝 Checksum Verification

验证所有关键文件未被篡改：
```bash
# 生成校验和（封板时已生成）
cd locked
find . -type f \( -name "*.py" -o -name "*.yaml" -o -name "*.toml" \) -exec sha256sum {} + > ../CHECKSUMS_verify.sha256

# 对比
diff ../CHECKSUMS.sha256 ../CHECKSUMS_verify.sha256
```

---

## 🎯 Expected Outputs

成功复现后，应该有以下文件：

```
locked/
├── results/
│   ├── bt_reproduction_YYYYMMDD_HHMMSS.parquet  # 回测结果
│   └── reports/
│       ├── trader_report_YYYYMMDD.md            # 交易员报告
│       └── recent_trades_YYYYMMDD.json          # 交易分析
└── .venv/                                       # 虚拟环境（UV 自动生成）
```

---

## 🚀 Deploy to Production

复现成功后，如需部署到生产环境：
```bash
# 1. 生成今日信号
uv run python scripts/generate_today_signal.py \
  --combo "ADX_14D + OBV_SLOPE_10D + SHARPE_RATIO_20D + SLOPE_20D" \
  --output signals/strategy1_today.json

# 2. 提交订单（需要 QMT Trading API）
uv run python scripts/submit_orders.py \
  --signal signals/strategy1_today.json \
  --account YOUR_ACCOUNT_ID
```

详细部署流程见 `artifacts/DEPLOYMENT_GUIDE.md`。

---

## 📞 Support

如有问题，请检查：
1. `README.md` (快速开始)
2. `artifacts/PRODUCTION_REPORT.md` (性能详情)
3. `artifacts/QUICK_REFERENCE.md` (因子说明)
4. `locked/src/etf_strategy/core/` (源码注释)

---

**Reproduction Time**: ~15-30 分钟（取决于数据下载）  
**Success Rate**: 99.9% (基于 v3.3 封板验证)  
**Last Verified**: 2025-12-16 16:00 CST
