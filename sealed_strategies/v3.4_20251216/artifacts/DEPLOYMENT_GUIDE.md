# v3.4 Production Deployment Guide

**Version**: v3.4_20251216  
**Target**: Production Trading Environment  
**Strategies**: 2 (震荡市精选双策略)

---

## 📋 Pre-Deployment Checklist

### 1. Environment Verification
- [ ] UV 已安装且版本 ≥ 0.1.0
- [ ] Python 版本 = 3.11+
- [ ] 系统内存 ≥ 8GB
- [ ] 磁盘空间 ≥ 5GB（数据 + 缓存）
- [ ] QMT Trading Terminal 可达（`192.168.122.132:8001`）

### 2. Data Integrity
```bash
cd sealed_strategies/v3.4_20251216/locked

# 验证校验和
sha256sum -c ../CHECKSUMS.sha256

# 验证数据完整性
uv run python scripts/verify_data_integrity.py

# 检查最新数据
uv run python -c "
import pandas as pd
import os
data_dir = '../../../raw/ETF/daily'
latest_dates = []
for f in os.listdir(data_dir):
    if f.endswith('.csv'):
        df = pd.read_csv(os.path.join(data_dir, f))
        latest_dates.append(df['date'].max())
print(f'Latest data date: {max(latest_dates)}')
assert max(latest_dates) >= '2025-12-12', 'Data is outdated!'
"
```

### 3. Backtest Validation
```bash
# 运行完整回测（验证环境）
uv run python scripts/batch_bt_backtest.py \
  --candidates ../artifacts/production_candidates.csv \
  --start 2020-01-01 \
  --end 2025-12-12 \
  --output results/pre_deploy_validation.parquet

# 验证结果一致性
uv run python -c "
import pandas as pd
df = pd.read_parquet('results/pre_deploy_validation.parquet')
assert abs(df.iloc[0]['total_return'] - 136.52) < 0.1, 'Strategy #1 mismatch!'
assert abs(df.iloc[1]['total_return'] - 129.85) < 0.1, 'Strategy #2 mismatch!'
print('✅ Pre-deployment validation passed')
"
```

### 4. Risk Parameters Configuration
```bash
# 编辑风控配置（可选）
vi configs/risk_controls.yaml
```

**推荐配置**:
```yaml
risk_controls:
  # 单策略止损
  strategy_stop_loss: -0.20  # -20%
  
  # 组合止损
  portfolio_stop_loss: -0.15  # -15%
  
  # 单日熔断
  daily_circuit_breaker: -0.03  # -3%
  
  # QDII 持仓上限
  qdii_max_position: 0.50  # 50%
  
  # 同步大跌阈值
  sync_drop_threshold: -0.02  # 两策略单日同跌 > 2%
  sync_drop_action: "reduce_30"  # 次日减仓 30%
```

---

## 🚀 Deployment Steps

### Step 1: 环境初始化
```bash
cd sealed_strategies/v3.4_20251216/locked
uv sync --dev
uv pip install -e .
```

### Step 2: 数据更新（首次部署）
```bash
# 全量下载（首次）
uv run python scripts/update_daily_from_qmt_bridge.py --all --start 2020-01-01

# 验证数据完整性
uv run python scripts/verify_data_integrity.py
```

### Step 3: 创建信号生成脚本
```bash
cat > scripts/daily_production_signal.sh << 'EOF'
#!/bin/bash
set -e

# 1. 更新数据
uv run python scripts/update_daily_from_qmt_bridge.py --all

# 2. 生成 Strategy #1 信号
uv run python scripts/generate_today_signal.py \
  --combo "ADX_14D + OBV_SLOPE_10D + SHARPE_RATIO_20D + SLOPE_20D" \
  --output signals/strategy1_$(date +%Y%m%d).json

# 3. 生成 Strategy #2 信号
uv run python scripts/generate_today_signal.py \
  --combo "ADX_14D + OBV_SLOPE_10D + PRICE_POSITION_120D + SHARPE_RATIO_20D + SLOPE_20D" \
  --output signals/strategy2_$(date +%Y%m%d).json

# 4. 风控检查
uv run python scripts/risk_check.py \
  --signals signals/strategy*_$(date +%Y%m%d).json \
  --portfolio-state data/portfolio_state.json

# 5. 如果通过风控，输出信号摘要
uv run python scripts/summarize_signals.py \
  --signals signals/strategy*_$(date +%Y%m%d).json
EOF

chmod +x scripts/daily_production_signal.sh
```

### Step 4: 设置定时任务（Cron）
```bash
# 编辑 crontab
crontab -e

# 添加以下行（每个交易日 15:00 执行）
0 15 * * 1-5 cd /home/sensen/dev/projects/-0927/sealed_strategies/v3.4_20251216/locked && ./scripts/daily_production_signal.sh >> logs/signal_$(date +\%Y\%m\%d).log 2>&1
```

### Step 5: 手动执行首次信号（验证）
```bash
./scripts/daily_production_signal.sh
```

**预期输出**:
```
✅ Data updated: 43 ETFs, latest date 2025-12-16
✅ Strategy #1 signal generated: BUY [159949, 159915]
✅ Strategy #2 signal generated: BUY [159949, 159915]
✅ Risk check passed: QDII < 50%, no sync drop
📊 Signal Summary:
   - Strategy #1: 2 positions, total 100% allocated
   - Strategy #2: 2 positions, total 100% allocated
   - Portfolio: 4 positions (2 unique), QDII 0%
```

### Step 6: 集成交易终端（QMT API）
```bash
# 创建订单提交脚本
cat > scripts/submit_orders_to_qmt.py << 'EOF'
#!/usr/bin/env python
import asyncio
import json
from qmt_bridge import QMTClient, QMTClientConfig

async def submit_orders(signal_file: str):
    # 加载信号
    with open(signal_file) as f:
        signals = json.load(f)
    
    # 初始化 QMT 客户端
    config = QMTClientConfig(host="192.168.122.132", port=8001)
    client = QMTClient(config)
    
    # 提交订单
    for signal in signals['orders']:
        if signal['action'] == 'BUY':
            await client.place_order(
                code=signal['code'],
                direction='BUY',
                price=0,  # 市价单
                volume=signal['quantity']
            )
        elif signal['action'] == 'SELL':
            await client.place_order(
                code=signal['code'],
                direction='SELL',
                price=0,
                volume=signal['quantity']
            )
    
    print(f"✅ Submitted {len(signals['orders'])} orders")

if __name__ == '__main__':
    import sys
    asyncio.run(submit_orders(sys.argv[1]))
EOF

chmod +x scripts/submit_orders_to_qmt.py
```

### Step 7: 首次实盘执行（小资金测试）
```bash
# 生成信号
./scripts/daily_production_signal.sh

# 提交订单（先用小资金测试）
uv run python scripts/submit_orders_to_qmt.py signals/strategy1_$(date +%Y%m%d).json
uv run python scripts/submit_orders_to_qmt.py signals/strategy2_$(date +%Y%m%d).json
```

---

## 📊 Monitoring & Operations

### 日频监控（每日收盘后）

#### 1. 检查组合日收益
```bash
uv run python scripts/monitor_daily_pnl.py
```

**阈值**:
- 连续 3 日 < -1% → 暂停开新仓
- 单日 < -3% → 触发熔断，次日减仓 30%

#### 2. 检查 QDII 持仓占比
```bash
uv run python -c "
import pandas as pd
import json

# 加载当前持仓
with open('data/portfolio_state.json') as f:
    portfolio = json.load(f)

# 统计 QDII
qdii_codes = ['513100', '513500', '159920', '513050', '513130']
qdii_value = sum([p['value'] for p in portfolio['positions'] if p['code'] in qdii_codes])
total_value = portfolio['total_value']

qdii_ratio = qdii_value / total_value
print(f'QDII Ratio: {qdii_ratio:.2%}')

if qdii_ratio > 0.50:
    print('⚠️ WARNING: QDII > 50%, consider reducing 20%')
"
```

#### 3. 检查同步大跌
```bash
uv run python -c "
import pandas as pd
import json

# 加载今日收益
with open('data/daily_returns.json') as f:
    returns = json.load(f)

strategy1_ret = returns['strategy1'][-1]
strategy2_ret = returns['strategy2'][-1]

if strategy1_ret < -0.02 and strategy2_ret < -0.02:
    print('⚠️ WARNING: Both strategies dropped > 2% today!')
    print('🚨 ACTION: Reduce positions by 30% tomorrow')
"
```

### 周频审计（每周五）

#### 1. 持仓重合度
```bash
uv run python scripts/audit_portfolio_overlap.py
```

**阈值**: > 90% 说明分散失效，考虑停用一个策略

#### 2. 胜率统计
```bash
uv run python scripts/audit_win_rate.py --window 30
```

**阈值**: < 45% 持续 1 个月 → 暂停策略

#### 3. 回撤监控
```bash
uv run python scripts/audit_drawdown.py
```

**阈值**: > 20% → 全部清仓，等待信号

---

## 🛡️ Risk Control Mechanisms

### 1. 单策略止损（-20%）
**触发条件**: 单策略累计亏损 > 20%  
**Action**:
```bash
# 清空该策略所有持仓
uv run python scripts/close_all_positions.py --strategy strategy1
```

### 2. 组合止损（-15%）
**触发条件**: 组合累计亏损 > 15%  
**Action**:
```bash
# 清空所有策略持仓
uv run python scripts/close_all_positions.py --all
```

### 3. 单日熔断（-3%）
**触发条件**: 组合单日亏损 > 3%  
**Action**:
```bash
# 次日减仓 30%
uv run python scripts/reduce_positions.py --ratio 0.30
```

### 4. QDII 上限（50%）
**触发条件**: QDII 持仓占比 > 50%  
**Action**:
```bash
# 手动减仓 QDII 20%
uv run python scripts/reduce_qdii_positions.py --ratio 0.20
```

### 5. 同步大跌熔断（两策略单日同跌 > 2%）
**触发条件**: 两策略单日同时跌 > 2%  
**Action**:
```bash
# 次日减仓 30%
uv run python scripts/reduce_positions.py --ratio 0.30 --reason "sync_drop"
```

---

## 🔄 Rebalancing Rules

### 自动再平衡（每 3 交易日）
**流程**:
1. 下载最新数据 (`update_daily_from_qmt_bridge.py`)
2. 计算因子值 (`factor_calculator.py`)
3. 生成新信号 (`generate_today_signal.py`)
4. 风控检查 (`risk_check.py`)
5. 提交订单 (`submit_orders_to_qmt.py`)

### 手动再平衡（每周五）
**检查项**:
- 持仓重合度 > 90% → 分散失效
- 胜率 < 45% 持续 1 月 → 暂停策略
- 回撤 > 20% → 全部清仓

---

## 📈 Performance Tracking

### 日频指标
```bash
# 生成日报
uv run python scripts/generate_daily_report.py --date $(date +%Y%m%d)
```

**输出**: `reports/daily_report_YYYYMMDD.json`
```json
{
  "date": "2025-12-16",
  "portfolio": {
    "total_value": 1050000,
    "daily_return": -0.003,
    "cumulative_return": 0.05,
    "max_drawdown": 0.08
  },
  "strategy1": {
    "positions": ["159949", "159915"],
    "daily_return": -0.002,
    "cumulative_return": 0.06
  },
  "strategy2": {
    "positions": ["159949", "159915"],
    "daily_return": -0.004,
    "cumulative_return": 0.04
  },
  "qdii_ratio": 0.0,
  "alerts": []
}
```

### 周频报告
```bash
# 生成周报
uv run python scripts/generate_weekly_report.py --week $(date +%Y%W)
```

**输出**: `reports/weekly_report_YYYYWW.pdf`（包含图表）

### 月频审计
```bash
# 生成月报
uv run python scripts/generate_monthly_audit.py --month $(date +%Y%m)
```

**输出**: `reports/monthly_audit_YYYYMM.pdf`

---

## 🚨 Emergency Procedures

### 场景 1: 系统宕机（QMT 不可达）
**Action**:
1. 检查 QMT VM 状态 (`ssh user@192.168.122.132`)
2. 重启 QMT Bridge (`systemctl restart qmt-bridge`)
3. 如果仍不可达，手动登录 QMT 客户端执行

### 场景 2: 数据延迟（最新数据 > 1 天）
**Action**:
1. 检查 QMT Bridge 连接 (`curl http://192.168.122.132:8001/health`)
2. 手动下载数据 (`update_daily_from_qmt_bridge.py --all`)
3. 如果仍失败，暂停交易直到数据恢复

### 场景 3: 订单失败（余额不足、涨跌停）
**Action**:
1. 检查账户余额 (`qmt_client.get_assets()`)
2. 检查标的涨跌停状态
3. 调整订单量或换标的

### 场景 4: 单日暴跌 > 5%
**Action**:
1. 立即清空所有持仓
2. 暂停自动交易
3. 等待人工审核后重启

---

## 📝 Logging & Auditing

### 日志配置
```python
# configs/logging.yaml
logging:
  level: INFO
  format: "%(asctime)s [%(levelname)s] %(message)s"
  handlers:
    - type: file
      filename: logs/production.log
      maxBytes: 10485760  # 10MB
      backupCount: 30
    - type: console
```

### 审计轨迹
所有操作记录到数据库：
```sql
CREATE TABLE audit_log (
    timestamp DATETIME,
    action VARCHAR(50),
    strategy VARCHAR(20),
    details JSON,
    result VARCHAR(20)
);
```

---

## 🔧 Configuration Files

### 1. 风控配置 (`configs/risk_controls.yaml`)
```yaml
strategy_stop_loss: -0.20
portfolio_stop_loss: -0.15
daily_circuit_breaker: -0.03
qdii_max_position: 0.50
sync_drop_threshold: -0.02
```

### 2. 交易配置 (`configs/trading.yaml`)
```yaml
rebalance_freq: 3  # 交易日
position_size: 2   # 持仓数量
initial_capital: 1000000
commission: 0.0002
slippage: 0.0005
```

### 3. 监控配置 (`configs/monitoring.yaml`)
```yaml
daily_report_time: "15:30"
weekly_report_day: "Friday"
monthly_audit_day: 1
alert_channels:
  - email: trader@example.com
  - webhook: https://hooks.slack.com/...
```

---

## 📞 Support & Escalation

### 一般问题
1. 检查日志 (`logs/production.log`)
2. 运行诊断 (`uv run python scripts/diagnose.py`)
3. 查看文档 (`docs/`)

### 紧急问题
1. 立即停止自动交易
2. 手动清仓（如果必要）
3. 联系技术支持

---

**Deployment Date**: 2025-12-16  
**Last Review**: 2025-12-16  
**Next Review**: 2026-01-16 (月度审计)
