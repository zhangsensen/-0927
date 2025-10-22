#!/usr/bin/env python3
"""
回测结果统一管理脚本 - 标准格式化保存和快速查询
"""
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd


def list_all_backtest_results(results_dir: str = None) -> None:
    """列出所有回测结果"""

    if not results_dir:
        project_root = Path(__file__).resolve().parent
        while project_root.parent != project_root:
            if (project_root / "raw").exists():
                break
            project_root = project_root.parent
        results_dir = (
            project_root / "etf_rotation_system" / "data" / "results" / "backtest"
        )
    else:
        results_dir = Path(results_dir)

    if not results_dir.exists():
        print(f"❌ 回测结果目录不存在: {results_dir}")
        return

    # 找到所有 backtest_YYYYMMDD_HHMMSS 文件夹
    backtest_dirs = sorted(
        [
            d
            for d in results_dir.iterdir()
            if d.is_dir() and d.name.startswith("backtest_")
        ],
        reverse=True,
    )

    if not backtest_dirs:
        print(f"⚠️  未找到任何回测结果")
        return

    print("=" * 120)
    print("📊 所有回测结果（按时间降序）")
    print("=" * 120)
    print()

    for idx, backtest_dir in enumerate(backtest_dirs[:20], 1):  # 显示最新20个
        timestamp = backtest_dir.name.replace("backtest_", "")

        # 读取配置
        config_file = backtest_dir / "best_config.json"
        log_file = backtest_dir / "backtest.log"
        results_file = backtest_dir / "results.csv"

        info_parts = []

        if config_file.exists():
            try:
                with open(config_file) as f:
                    config = json.load(f)
                    perf = config.get("performance", {})
                    sharpe = perf.get("sharpe_ratio", 0)
                    ret = perf.get("total_return", 0)
                    dd = perf.get("max_drawdown", 0)
                    info_parts.append(
                        f"Sharpe={sharpe:.4f} | Return={ret:.2f}% | DD={dd:.2f}%"
                    )
            except:
                pass

        if log_file.exists():
            try:
                with open(log_file) as f:
                    content = f.read()
                    if "处理策略:" in content:
                        for line in content.split("\n"):
                            if "处理策略:" in line:
                                num = line.split("个")[0].strip().split()[-1]
                                info_parts.append(f"Strategies={num}")
                                break
                    if "处理速度:" in content:
                        for line in content.split("\n"):
                            if "处理速度:" in line:
                                speed = line.split("速度:")[1].strip()
                                info_parts.append(f"Speed={speed}")
                                break
            except:
                pass

        size_mb = (
            results_file.stat().st_size / (1024 * 1024) if results_file.exists() else 0
        )

        print(f"{idx:2d}. [{timestamp}]")
        print(f"    {' | '.join(info_parts)}")
        print(
            f"    📁 {size_mb:.1f}MB  ✓ results.csv  ✓ best_config.json  ✓ backtest.log"
        )
        print()

    print(f"共找到 {len(backtest_dirs)} 个回测结果")
    print("=" * 120)


def show_best_config(timestamp: str, results_dir: str = None) -> None:
    """显示特定时间戳的最优配置"""

    if not results_dir:
        project_root = Path(__file__).resolve().parent
        while project_root.parent != project_root:
            if (project_root / "raw").exists():
                break
            project_root = project_root.parent
        results_dir = (
            project_root / "etf_rotation_system" / "data" / "results" / "backtest"
        )

    backtest_dir = Path(results_dir) / f"backtest_{timestamp}"

    if not backtest_dir.exists():
        print(f"❌ 找不到时间戳 {timestamp} 的结果")
        return

    config_file = backtest_dir / "best_config.json"

    if not config_file.exists():
        print(f"❌ 配置文件不存在: {config_file}")
        return

    with open(config_file) as f:
        config = json.load(f)

    print("=" * 100)
    print(f"📋 最优策略配置 [{timestamp}]")
    print("=" * 100)
    print()

    # 性能指标
    perf = config.get("performance", {})
    print("🎯 性能指标:")
    print(f"  • Sharpe比率: {perf.get('sharpe_ratio', 0):.4f}")
    print(f"  • 总收益率: {perf.get('total_return', 0):.2f}%")
    print(f"  • 最大回撤: {perf.get('max_drawdown', 0):.2f}%")
    print()

    # 权重配置
    weights = config.get("weights", {})
    print("⚖️  权重配置:")

    # 处理权重字符串或字典
    if isinstance(weights, str):
        import ast

        weights = ast.literal_eval(weights)

    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    for factor, weight in sorted_weights:
        if weight > 0:
            print(f"  • {factor}: {weight:.2f}")
    print()

    # 其他配置
    cfg = config.get("config", {})
    print("⚙️  回测参数:")
    print(f"  • Top-N: {config.get('top_n', 0)}")
    print(f"  • 工作进程: {cfg.get('n_workers', 0)}")
    print(f"  • 最大组合: {cfg.get('max_combinations', 0):,}")
    print(f"  • Top-N列表: {cfg.get('top_n_list', [])}")
    print()

    # 时间统计
    timing = config.get("timing", {})
    print("⏱️  执行统计:")
    print(f"  • 总耗时: {timing.get('total_time', 0):.2f}秒")
    print(f"  • 策略数: {timing.get('strategies_tested', 0):,}")
    print(f"  • 速度: {timing.get('speed_per_second', 0):.1f}策略/秒")
    print()

    # 数据源
    data = config.get("data_source", {})
    print("📂 数据源:")
    print(f"  • Panel: {Path(data.get('panel', '')).name}")
    print(f"  • Screening: {Path(data.get('screening', '')).name}")
    print()

    print("=" * 100)


def show_top_results(timestamp: str, top_n: int = 10, results_dir: str = None) -> None:
    """显示Top N策略"""

    if not results_dir:
        project_root = Path(__file__).resolve().parent
        while project_root.parent != project_root:
            if (project_root / "raw").exists():
                break
            project_root = project_root.parent
        results_dir = (
            project_root / "etf_rotation_system" / "data" / "results" / "backtest"
        )

    backtest_dir = Path(results_dir) / f"backtest_{timestamp}"
    results_file = backtest_dir / "results.csv"

    if not results_file.exists():
        print(f"❌ 找不到结果文件: {results_file}")
        return

    df = pd.read_csv(results_file, nrows=top_n)

    print("=" * 140)
    print(f"🏆 Top {min(top_n, len(df))} 策略 [{timestamp}]")
    print("=" * 140)
    print()

    for idx, row in df.iterrows():
        print(
            f"#{idx+1} | Sharpe={row['sharpe_ratio']:.4f} | Return={row['total_return']:7.2f}% | DD={row['max_drawdown']:7.2f}% | Top_N={int(row['top_n']):2d}"
        )

    print()
    print("=" * 140)


def compare_backtests(
    timestamp1: str, timestamp2: str, results_dir: str = None
) -> None:
    """比较两个回测结果"""

    if not results_dir:
        project_root = Path(__file__).resolve().parent
        while project_root.parent != project_root:
            if (project_root / "raw").exists():
                break
            project_root = project_root.parent
        results_dir = (
            project_root / "etf_rotation_system" / "data" / "results" / "backtest"
        )

    def get_best(ts):
        config_file = Path(results_dir) / f"backtest_{ts}" / "best_config.json"
        if not config_file.exists():
            return None
        with open(config_file) as f:
            return json.load(f)

    config1 = get_best(timestamp1)
    config2 = get_best(timestamp2)

    if not config1 or not config2:
        print("❌ 找不到对比的结果")
        return

    print("=" * 100)
    print(f"📊 回测对比 [{timestamp1}] vs [{timestamp2}]")
    print("=" * 100)
    print()

    perf1 = config1.get("performance", {})
    perf2 = config2.get("performance", {})

    metrics = [
        ("Sharpe比率", "sharpe_ratio"),
        ("总收益率%", "total_return"),
        ("最大回撤%", "max_drawdown"),
    ]

    print(f"{'指标':<15} | {'结果1':<15} | {'结果2':<15} | {'改进':<15}")
    print("-" * 65)

    for name, key in metrics:
        v1 = perf1.get(key, 0)
        v2 = perf2.get(key, 0)

        if key == "max_drawdown":
            # 回撤越小越好
            delta = v1 - v2  # 如果v2更小（负数更小），delta为正（改进）
            pct = (delta / abs(v1) * 100) if v1 != 0 else 0
            symbol = "↑" if delta > 0 else "↓"
        else:
            # Sharpe和收益越大越好
            delta = v2 - v1
            pct = (delta / abs(v1) * 100) if v1 != 0 else 0
            symbol = "↑" if delta > 0 else "↓"

        print(f"{name:<15} | {v1:>14.2f} | {v2:>14.2f} | {symbol} {abs(pct):>12.1f}%")

    print()
    print("=" * 100)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        # 列出所有结果
        list_all_backtest_results()
    elif sys.argv[1] == "list":
        list_all_backtest_results()
    elif sys.argv[1] == "config" and len(sys.argv) >= 3:
        show_best_config(sys.argv[2])
    elif sys.argv[1] == "top" and len(sys.argv) >= 3:
        top_n = int(sys.argv[3]) if len(sys.argv) >= 4 else 10
        show_top_results(sys.argv[2], top_n)
    elif sys.argv[1] == "compare" and len(sys.argv) >= 4:
        compare_backtests(sys.argv[2], sys.argv[3])
    else:
        print("用法:")
        print("  python backtest_manager.py list                    # 列出所有回测")
        print("  python backtest_manager.py config <timestamp>      # 显示最优配置")
        print("  python backtest_manager.py top <timestamp> [N]     # 显示Top N")
        print("  python backtest_manager.py compare <ts1> <ts2>     # 对比两个回测")
