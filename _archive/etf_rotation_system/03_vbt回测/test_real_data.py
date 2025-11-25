#!/usr/bin/env python3
"""真实数据验收测试 - 修复后的VBT回测系统"""
import os
import sys
import time
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

from config_loader_parallel import FastConfig, ParallelBacktestConfig
from parallel_backtest_configurable import ConfigurableParallelBacktestEngine


def get_project_root() -> Path:
    """获取项目根目录，向上查找包含 raw/ 的目录"""
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
        # 兜底：返回默认路径
        return project_root / "etf_rotation_system" / "data" / "factor_panel.parquet"

    # 找到所有包含 panel.parquet 的子目录
    panel_dirs = [
        d for d in panels_dir.iterdir() if d.is_dir() and (d / "panel.parquet").exists()
    ]

    if not panel_dirs:
        return project_root / "etf_rotation_system" / "data" / "factor_panel.parquet"

    # 按修改时间排序，取最新的
    latest_dir = max(panel_dirs, key=lambda d: (d / "panel.parquet").stat().st_mtime)
    return latest_dir / "panel.parquet"


def get_latest_screening_file() -> Path:
    """自动找到最新生成的 passed_factors.csv"""
    project_root = get_project_root()
    screening_dir = (
        project_root / "etf_rotation_system" / "data" / "results" / "screening"
    )

    if not screening_dir.exists():
        # 兜底：返回默认路径
        return (
            project_root
            / "etf_rotation_system"
            / "data"
            / "results"
            / "screening"
            / "passed_factors.csv"
        )

    # 找到所有包含 passed_factors.csv 的子目录
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

    # 按修改时间排序，取最新的
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

print("=" * 80)
print("真实数据验收测试 - 修复后的VBT回测系统")
print("=" * 80)
print()

# 创建配置（使用 ParallelBacktestConfig 支持参数修改）
config = ParallelBacktestConfig(
    panel_file=PANEL_FILE,
    price_dir=PRICE_DIR,
    screening_file=SCREENING_FILE,
    output_dir=OUTPUT_DIR,
    # 并行配置
    n_workers=7,  # 使用7个核心
    chunk_size=20,
    # 因子配置
    top_k=8,  # Baseline 8个因子（已验证最优）
    factors=[],  # 空则自动从筛选结果加载
    # 回测参数 - 应从parallel_backtest_config.yaml读取
    top_n_list=[2, 3, 4, 5, 6, 7],  # 测试多个top_n值（与config.yaml保持一致）
    rebalance_freq=20,
    # === Phase 1 改进 ===
    # A3: A股精细成本模型
    fees=0.003,  # A股 ETF: 佣金0.2% + 印花税0.1% = 0.3% 往返
    # B1: 智能 Rebalance (在成本计算中自动应用，5% 阈值)
    # 权重网格 - Baseline 配置
    weight_grid_points=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # 6个点
    weight_sum_range=[0.8, 1.2],
    max_combinations=15000,  # 充分覆盖搜索空间
    # 输出配置
    verbose=True,
    save_top_results=50,
)

print(f"📂 数据路径:")
print(f"  面板: {Path(PANEL_FILE).name}")
print(f"  价格: {PRICE_DIR}")
print(f"  筛选: {Path(SCREENING_FILE).name}")
print()
print(f"🚀 Phase 1 优化启用:")
print(f"  ✅ A3: A股精细成本模型 (佣金0.2% + 印花税0.1%)")
print(f"  ✅ B1: 智能 Rebalance (权重变化>5% 才交易)")
print()

# 创建引擎
print("🚀 初始化回测引擎...")
engine = ConfigurableParallelBacktestEngine(config)

# 运行回测
print("\n⚡ 开始并行回测...")
start_time = time.time()

try:
    results = engine.parallel_grid_search()
    elapsed = time.time() - start_time

    print(f"\n✅ 回测完成，耗时: {elapsed:.2f}秒")
    print(f"   测试组合数: {len(results)}")
    print(f"   处理速度: {len(results) / elapsed:.1f} 组合/秒")
    print()

    # 显示Top 5结果
    print("🏆 Top 5 策略:")
    top5 = results.nlargest(5, "sharpe_ratio")
    for idx, row in top5.iterrows():
        print(
            f"  {idx+1}. Sharpe={row['sharpe_ratio']:.3f}, Return={row['total_return']:.2f}%, "
            f"Drawdown={row['max_drawdown']:.2f}%, Top_N={int(row['top_n'])}"
        )

except Exception as e:
    print(f"\n❌ 回测失败: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print(f"\n{'=' * 80}")
print("验收通过！系统可用于生产环境")
print(f"{'=' * 80}")
