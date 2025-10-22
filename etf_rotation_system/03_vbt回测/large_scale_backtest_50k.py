#!/usr/bin/env python3
"""大规模回测脚本 - 5万组合完整搜索"""
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

from config_loader_parallel import ParallelBacktestConfig
from parallel_backtest_configurable import ConfigurableParallelBacktestEngine


def get_project_root() -> Path:
    """获取项目根目录"""
    current = Path(__file__).resolve().parent
    while current.parent != current:
        if (current / "raw").exists() or (current.parent / "raw").exists():
            return current if (current / "raw").exists() else current.parent
        current = current.parent
    return Path(os.getenv("PROJECT_ROOT", Path.cwd()))


def get_latest_panel_file() -> Path:
    """自动找到最新生成的 panel.parquet"""
    project_root = get_project_root()
    panels_dir = project_root / "etf_rotation_system" / "data" / "results" / "panels"

    if not panels_dir.exists():
        return project_root / "etf_rotation_system" / "data" / "factor_panel.parquet"

    panel_dirs = [
        d for d in panels_dir.iterdir() if d.is_dir() and (d / "panel.parquet").exists()
    ]

    if not panel_dirs:
        return project_root / "etf_rotation_system" / "data" / "factor_panel.parquet"

    latest_dir = max(panel_dirs, key=lambda d: (d / "panel.parquet").stat().st_mtime)
    return latest_dir / "panel.parquet"


def get_latest_screening_file() -> Path:
    """自动找到最新生成的 passed_factors.csv"""
    project_root = get_project_root()
    screening_dir = (
        project_root / "etf_rotation_system" / "data" / "results" / "screening"
    )

    if not screening_dir.exists():
        return (
            project_root
            / "etf_rotation_system"
            / "data"
            / "results"
            / "screening"
            / "passed_factors.csv"
        )

    screening_dirs = [
        d
        for d in screening_dir.iterdir()
        if d.is_dir() and (d / "passed_factors.csv").exists()
    ]

    if not screening_dirs:
        return (
            project_root
            / "etf_rotation_system"
            / "data"
            / "results"
            / "screening"
            / "passed_factors.csv"
        )

    latest_dir = max(
        screening_dirs, key=lambda d: (d / "passed_factors.csv").stat().st_mtime
    )
    return latest_dir / "passed_factors.csv"


# 自动检测最新路径
PROJECT_ROOT = get_project_root()
PANEL_FILE = str(get_latest_panel_file())
PRICE_DIR = str(PROJECT_ROOT / "raw" / "ETF" / "daily")
SCREENING_FILE = str(get_latest_screening_file())
OUTPUT_DIR = str(PROJECT_ROOT / "etf_rotation_system" / "data" / "results" / "backtest")

print("=" * 100)
print("🚀 大规模回测 - 5万组合完整搜索")
print("=" * 100)
print()

# 创建配置 - 5万组合
config = ParallelBacktestConfig(
    panel_file=PANEL_FILE,
    price_dir=PRICE_DIR,
    screening_file=SCREENING_FILE,
    output_dir=OUTPUT_DIR,
    # 并行配置 - 最大化性能
    n_workers=8,  # 使用8个核心
    chunk_size=50,  # 每块50个组合
    # 因子配置 - 🎯 使用筛选后的12个核心因子
    top_k=12,  # 修改为12个因子（优化后筛选结果）
    factors=[],  # 空则自动从筛选结果加载（screening_20251022_014652/passed_factors.csv）
    # 回测参数
    # 🔧 修正：聚焦持仓5-10只，避免持仓2只的过拟合板块押注
    top_n_list=[5, 8, 10],  # 修改为5/8/10只，剔除2只
    rebalance_freq=20,
    # === Phase 1 改进 ===
    # A3: A股精细成本模型
    fees=0.003,  # A股 ETF: 0.3% 往返
    # B1: 智能 Rebalance (在成本计算中自动应用，5% 阈值)
    # 权重网格 - 快速验证配置（1万组合）
    weight_grid_points=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # 6个点（减少搜索空间）
    weight_sum_range=[0.6, 1.4],  # 🔧 放宽到[0.6, 1.4]以提高采样命中率
    max_combinations=10000,  # 🔧 改为1万组合快速验证
    # 输出配置
    verbose=True,
    save_top_results=100,
    save_best_config=True,
    enable_progress_bar=True,
    log_level="INFO",
)

print(f"📂 数据来源:")
print(f"  • 因子面板: {Path(PANEL_FILE).name}")
print(f"  • 价格数据: {PRICE_DIR}")
print(f"  • 因子筛选: {Path(SCREENING_FILE).name}")
print()
print(f"⚙️  回测配置:")
print(f"  • 因子数量: {config.top_k} 个核心因子（优化筛选后）")
print(f"  • 权重网格: 6个点 → 理论组合 {6**config.top_k:,}")
print(f"  • 实际测试: {config.max_combinations:,} 个精选组合")
print(f"  • Top-N列表: {config.top_n_list}")
print(f"  • 并行进程: {config.n_workers}")
print(f"  • 块大小: {config.chunk_size}")
print()
print(f"💰 成本模型 (Phase 1 优化):")
print(f"  • A3: A股精细成本 (佣金0.2% + 印花税0.1%)")
print(f"  • B1: 智能Rebalance (权重变化>5%才交易)")
print(f"  • 总往返成本: 0.3%")
print()

# 创建引擎
print("🔧 初始化回测引擎...")
engine = ConfigurableParallelBacktestEngine(config)

# 运行回测
print("\n⚡ 开始5万组合并行回测...")
print("=" * 100)

start_time = time.time()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

try:
    # 使用 run_parallel_backtest 获得标准格式的输出
    results, metadata = engine.run_parallel_backtest()
    elapsed = time.time() - start_time

    print("=" * 100)
    print(f"\n✅ 回测完成！")
    print(f"  • 总耗时: {elapsed:.2f} 秒 ({elapsed/60:.1f} 分钟)")
    print(f"  • 处理速度: {len(results) / elapsed:.1f} 组合/秒")
    print(f"  • 测试组合: {len(results):,} 个")
    print()

    # 显示Top 10结果
    print("🏆 Top 10 策略（按Sharpe排序）:")
    print("-" * 100)
    top10 = results.head(10)
    for idx, row in top10.iterrows():
        print(
            f"  #{idx+1:2d} | Sharpe={row['sharpe_ratio']:.4f} | Return={row['total_return']:7.2f}% | "
            f"Drawdown={row['max_drawdown']:7.2f}% | Top_N={int(row['top_n']):2d}"
        )

    print()
    print("📊 性能统计:")
    print(f"  • 平均Sharpe: {results['sharpe_ratio'].mean():.4f}")
    print(f"  • 中位数Sharpe: {results['sharpe_ratio'].median():.4f}")
    print(f"  • 最高Sharpe: {results['sharpe_ratio'].max():.4f}")
    print(f"  • 最低Sharpe: {results['sharpe_ratio'].min():.4f}")
    print(
        f"  • 正期望(Sharpe>0): {len(results[results['sharpe_ratio'] > 0]):,} ({len(results[results['sharpe_ratio'] > 0])/len(results)*100:.1f}%)"
    )
    print(
        f"  • 优秀(Sharpe>0.5): {len(results[results['sharpe_ratio'] > 0.5]):,} ({len(results[results['sharpe_ratio'] > 0.5])/len(results)*100:.1f}%)"
    )

    print()
    print("📁 结果已保存至:")
    backtest_dir = Path(OUTPUT_DIR) / f"backtest_{timestamp}"
    if backtest_dir.exists():
        print(f"  {backtest_dir}/")
        for file in backtest_dir.iterdir():
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"    ✓ {file.name} ({size_mb:.2f}MB)")

    print()
    print("=" * 100)
    print("✨ 5万组合大规模回测成功完成！")
    print("=" * 100)

except Exception as e:
    print(f"\n❌ 回测失败: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
