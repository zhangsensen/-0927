# 📁 配置文件说明

> 更新时间: 2025-12-02 | 版本: v3.1

## 配置文件清单

| 文件 | 用途 | 状态 |
|------|------|------|
| `combo_wfo_config.yaml` | **生产主配置** - WFO/VEC/BT 核心参数 | ✅ 生产 |
| `etf_pools.yaml` | ETF 分池定义 (7 类 + QDII Alpha) | ✅ 生产 |
| `etf_config.yaml` | ETF 详细信息 (下载管理器用) | ✅ 生产 |
| `FACTOR_SELECTION_CONSTRAINTS.yaml` | 因子筛选约束 (家族配额、互斥对) | ✅ 生产 |
| `full_config.yaml` | Tushare 全量下载配置 (6年) | ✅ 数据 |
| `quick_config.yaml` | Tushare 快速下载配置 (1年) | ✅ 数据 |
| `cn_holidays.txt` | 中国节假日列表 | ✅ 数据 |
| `risk_control_rules.yaml` | 风控规则 (草案) | ⚠️ 草案 |
| `monitor_thresholds.yaml` | 监控阈值 | ⚠️ 草案 |

## 核心配置：combo_wfo_config.yaml

```yaml
# 锁定参数 (v3.1)
FREQ: 3          # 调仓频率 (交易日)
POS: 2           # 持仓数量
ETF数量: 43      # 含 5 只 QDII
初始资金: 100万
手续费: 2bp
```

## ⚠️ QDII 保护规则

以下 5 只 QDII ETF **禁止从任何配置中移除**：

| 代码 | 名称 | 收益贡献 |
|------|------|---------|
| 513500 | 标普500 ETF | +25.37% ⭐ |
| 513130 | 恒生科技ETF(港元) | +23.69% |
| 513100 | 纳指100 ETF | +22.03% |
| 159920 | 恒生ETF | +17.13% |
| 513050 | 中概互联ETF | +2.01% |

> 📖 详见 `docs/ETF_POOL_ARCHITECTURE.md`

## 已删除配置 (2025-12-02)

| 文件 | 原因 |
|------|------|
| `strategy_config.yaml` | 过期参数 (FREQ=8, POS=3)，与生产不一致 |
| `default.yaml` | 遗留配置，与主配置重复 |
| `combo_wfo_config.yaml.bak` | 无用备份文件 |
