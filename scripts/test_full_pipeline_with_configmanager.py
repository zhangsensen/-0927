#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试完整ETF流程 - ConfigManager版本（真实环境测试）"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
print(f"📁 项目根目录: {PROJECT_ROOT}")


def run_command(cmd, desc, cwd=None):
    """执行命令并检查结果"""
    print(f"\n{'='*60}")
    print(f"🚀 {desc}")
    print(f"{'='*60}")
    print(f"💻 命令: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=300,  # 5分钟超时
        )
        print(result.stdout)
        if result.stderr:
            print("⚠️ 警告:")
            print(result.stderr)
        print(f"✅ {desc} - 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {desc} - 失败")
        print(f"错误代码: {e.returncode}")
        print(f"标准输出:\n{e.stdout}")
        print(f"错误输出:\n{e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        print(f"❌ {desc} - 超时（>5分钟）")
        return False
    except Exception as e:
        print(f"❌ {desc} - 异常: {e}")
        return False


def check_output_files():
    """检查输出文件是否生成"""
    print(f"\n{'='*60}")
    print("📊 检查输出文件")
    print(f"{'='*60}")

    # 检查面板文件
    panel_dir = PROJECT_ROOT / "etf_rotation_system" / "data" / "results" / "panels"
    if panel_dir.exists():
        panel_files = list(panel_dir.glob("*.parquet"))
        print(f"✓ 面板文件: {len(panel_files)} 个")
        if panel_files:
            latest = max(panel_files, key=lambda p: p.stat().st_mtime)
            print(f"  最新: {latest.name}")

    # 检查筛选结果
    screening_dir = (
        PROJECT_ROOT / "etf_rotation_system" / "data" / "results" / "screening"
    )
    if screening_dir.exists():
        ic_files = list(screening_dir.glob("**/ic_statistics*.csv"))
        print(f"✓ IC统计文件: {len(ic_files)} 个")

    # 检查回测结果
    backtest_dir = (
        PROJECT_ROOT / "etf_rotation_system" / "data" / "results" / "backtests"
    )
    if backtest_dir.exists():
        result_files = list(backtest_dir.glob("**/*.csv"))
        print(f"✓ 回测结果文件: {len(result_files)} 个")

    print()


def main():
    """执行完整测试流程"""
    print("=" * 60)
    print("🧪 ETF完整流程测试 - ConfigManager统一配置")
    print("=" * 60)
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    results = {}

    # Step 1: 生成因子面板（使用少量ETF测试）
    step1 = run_command(
        ["python3", "generate_panel_refactored.py", "--workers", "2"],
        "Step 1: 生成因子面板",
        cwd=PROJECT_ROOT / "etf_rotation_system" / "01_横截面建设",
    )
    results["面板生成"] = step1

    if not step1:
        print("\n❌ 面板生成失败，终止测试")
        return False

    # Step 2: 因子筛选（使用生成的面板）
    step2 = run_command(
        ["python3", "run_etf_cross_section_configurable.py"],
        "Step 2: 因子筛选（IC/IR计算）",
        cwd=PROJECT_ROOT / "etf_rotation_system" / "02_因子筛选",
    )
    results["因子筛选"] = step2

    if not step2:
        print("\n⚠️ 因子筛选失败，但继续测试回测模块")

    # Step 3: 回测计算
    step3 = run_command(
        ["python3", "parallel_backtest_configurable.py"],
        "Step 3: 回测计算",
        cwd=PROJECT_ROOT / "etf_rotation_system" / "03_vbt回测",
    )
    results["回测计算"] = step3

    # 检查输出文件
    check_output_files()

    # 总结
    print("\n" + "=" * 60)
    print("📊 测试结果总结")
    print("=" * 60)

    for step, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{step}: {status}")

    all_passed = all(results.values())
    print()
    if all_passed:
        print("🎉 完整流程测试通过！ConfigManager 迁移成功！")
        print("✓ 面板生成正常")
        print("✓ 因子筛选正常")
        print("✓ 回测计算正常")
    else:
        print("⚠️ 部分测试失败，请检查日志")

    print(f"\n⏰ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
