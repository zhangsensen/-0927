"""
Top1000策略数据读取示例

新版本特性:
1. 单个Parquet文件存储Top1000策略信息 (strategies_ranked.parquet)
2. 单个Parquet文件存储Top1000收益序列 (top1000_returns.parquet)
3. rank列: 1-1000，方便索引
"""

from pathlib import Path

import pandas as pd

# 示例结果目录
result_dir = Path("results/wfo/20251103/20251103_xxxxxx")  # 替换为实际路径

print("=" * 80)
print("Top1000策略数据读取示例")
print("=" * 80)

# 1. 读取策略排行
print("\n【1. 读取策略排行】")
strategies = pd.read_parquet(result_dir / "strategies_ranked.parquet")
print(f"加载策略数: {len(strategies)}")
print(f"列名: {list(strategies.columns)}")
print(f"\nTop5策略:")
print(
    strategies.head(5)[
        ["rank", "n_factors", "tau", "top_n", "z_threshold", "score", "sharpe_ratio"]
    ]
)

# 2. 读取收益序列（宽表）
print("\n【2. 读取收益序列】")
returns_wide = pd.read_parquet(result_dir / "top1000_returns.parquet")
print(f"收益序列形状: {returns_wide.shape} (天数 × 策略数)")
print(f"策略列名: {list(returns_wide.columns[:5])} ...")  # 前5个

# 3. 按rank获取特定策略收益
print("\n【3. 按rank获取策略收益】")
rank = 1  # Top1策略
if f"rank_{rank}" in returns_wide.columns:
    top1_returns = returns_wide[f"rank_{rank}"]
    print(f"Rank {rank} 策略收益序列:")
    print(f"  - 平均日收益: {top1_returns.mean():.4f}")
    print(f"  - 累计收益: {(1 + top1_returns).prod() - 1:.4f}")

# 4. 批量读取Top10收益
print("\n【4. 批量读取Top10收益】")
top10_cols = [f"rank_{i}" for i in range(1, 11) if f"rank_{i}" in returns_wide.columns]
top10_returns = returns_wide[top10_cols]
print(f"Top10收益矩阵形状: {top10_returns.shape}")

# 5. 计算Top10等权组合
print("\n【5. 计算Top10等权组合】")
combo_returns = top10_returns.mean(axis=1)
combo_equity = (1 + combo_returns).cumprod()
print(f"Top10组合:")
print(f"  - 累计收益: {combo_equity.iloc[-1] - 1:.4f}")
print(f"  - Sharpe: {combo_returns.mean() / combo_returns.std() * (252**0.5):.4f}")

# 6. 根据因子过滤策略
print("\n【6. 根据条件过滤策略】")
filtered = strategies[
    (strategies["sharpe_ratio"] > 0.75) & (strategies["coverage"] > 0.95)
]
print(f"符合条件的策略数: {len(filtered)}")
print(f"这些策略的rank: {filtered['rank'].tolist()[:10]}")

# 7. 对比不同rank的策略
print("\n【7. 性能梯度分析】")
for rank_group in [
    range(1, 11),
    range(11, 51),
    range(51, 101),
    range(101, 501),
    range(501, 1001),
]:
    subset = strategies[strategies["rank"].isin(rank_group)]
    print(
        f"Rank {min(rank_group)}-{max(rank_group)}: "
        f"sharpe={subset['sharpe_ratio'].mean():.4f}, "
        f"score={subset['score'].mean():.4f}"
    )

print("\n" + "=" * 80)
print("优势总结")
print("=" * 80)
print(
    """
✅ 单文件存储，加载快速（1-2秒 vs 分钟级）
✅ rank列直观，易于索引（rank_1 = Top1）
✅ 支持批量操作（Top10, Top50等）
✅ 节省磁盘空间（1个文件 vs 12万个文件）
✅ 方便做性能分析和可视化
"""
)
