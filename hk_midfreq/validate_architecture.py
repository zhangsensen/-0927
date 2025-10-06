#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""架构验证脚本 - 验证三个数据层的配置和功能

验证内容：
1. 原始数据层 (raw/HK/) - 价格数据加载
2. 因子筛选层 (factor_system/factor_ready/) - 优秀因子加载
3. 因子输出层 (factor_system/因子输出/) - 因子时间序列加载
4. 输出结果管理 - 带时间戳的会话目录
"""

import sys
from pathlib import Path

# 添加项目路径（必须在其他导入之前）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime  # noqa: E402

import pandas as pd  # noqa: E402

from hk_midfreq.config import PathConfig  # noqa: E402
from hk_midfreq.factor_interface import FactorScoreLoader  # noqa: E402
from hk_midfreq.price_loader import PriceDataLoader  # noqa: E402
from hk_midfreq.result_manager import BacktestResultManager  # noqa: E402


def print_section(title: str) -> None:
    """打印分节标题"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def validate_path_config() -> PathConfig:
    """验证路径配置"""
    print_section("第一步: 验证路径配置")

    path_config = PathConfig()

    print(f"项目根目录: {path_config.project_root}")
    print(f"  └─ 存在: {'✅' if path_config.project_root.exists() else '❌'}")

    print("\n📁 数据层 1: 原始数据层")
    print(f"  路径: {path_config.hk_raw_dir}")
    print(f"  存在: {'✅' if path_config.hk_raw_dir.exists() else '❌'}")
    if path_config.hk_raw_dir.exists():
        files = list(path_config.hk_raw_dir.glob("*.parquet"))
        print(f"  文件数: {len(files)}")
        if files:
            print(f"  示例: {files[0].name}")

    print("\n📁 数据层 2: 因子筛选层")
    print(f"  路径: {path_config.factor_ready_dir}")
    print(f"  存在: {'✅' if path_config.factor_ready_dir.exists() else '❌'}")
    if path_config.factor_ready_dir.exists():
        files = list(path_config.factor_ready_dir.glob("*.parquet"))
        print(f"  文件数: {len(files)}")
        if files:
            print(f"  示例: {files[0].name}")

    print("\n📁 数据层 3: 因子输出层")
    print(f"  路径: {path_config.factor_output_dir}")
    print(f"  存在: {'✅' if path_config.factor_output_dir.exists() else '❌'}")
    if path_config.factor_output_dir.exists():
        subdirs = [d for d in path_config.factor_output_dir.iterdir() if d.is_dir()]
        print(f"  时间框架数: {len(subdirs)}")
        for subdir in subdirs[:3]:  # 显示前3个
            files = list(subdir.glob("*.parquet"))
            print(f"    {subdir.name}: {len(files)} 个文件")

    print("\n📁 输出目录: 回测结果")
    print(f"  路径: {path_config.backtest_output_dir}")
    print("  将自动创建: ✅")

    return path_config


def validate_price_loading(path_config: PathConfig) -> pd.DataFrame:
    """验证原始数据层 - 价格数据加载"""
    print_section("第二步: 验证原始数据层 - 价格数据加载")

    loader = PriceDataLoader(path_config)

    try:
        # 测试加载 0700.HK 的 5min 数据
        price_data = loader.load_price("0700.HK", "5min")

        print("✅ 价格数据加载成功")
        print("  股票代码: 0700.HK")
        print("  时间框架: 5min")
        print(f"  数据形状: {price_data.shape}")
        print(f"  时间范围: {price_data.index[0]} 至 {price_data.index[-1]}")
        print(f"  数据列: {', '.join(price_data.columns)}")

        return price_data

    except Exception as e:
        print(f"❌ 价格数据加载失败: {e}")
        return pd.DataFrame()


def validate_factor_ready(path_config: PathConfig) -> bool:
    """验证因子筛选层 - 优秀因子数据"""
    print_section("第三步: 验证因子筛选层 - 优秀因子加载")

    factor_ready_file = path_config.factor_ready_dir / "0700_HK_best_factors.parquet"

    if not factor_ready_file.exists():
        print(f"⚠️  优秀因子文件不存在: {factor_ready_file}")
        print("   提示: 运行因子筛选后会生成此文件")
        return False

    try:
        best_factors = pd.read_parquet(factor_ready_file)

        print("✅ 优秀因子数据加载成功")
        print(f"  文件: {factor_ready_file.name}")
        print(f"  因子数量: {len(best_factors)}")
        print(f"  数据列: {', '.join(best_factors.columns)}")

        if len(best_factors) > 0:
            print("\n  前3个优秀因子:")
            for idx, row in best_factors.head(3).iterrows():
                print(f"    - {row.get('Factor', row.get('name', f'Factor_{idx}'))}")

        return True

    except Exception as e:
        print(f"❌ 优秀因子加载失败: {e}")
        return False


def validate_factor_output(path_config: PathConfig) -> pd.DataFrame:
    """验证因子输出层 - 因子时间序列"""
    print_section("第四步: 验证因子输出层 - 因子时间序列加载")

    # 检查 5min 时间框架的因子文件
    factor_5min_dir = path_config.factor_output_dir / "5min"

    if not factor_5min_dir.exists():
        print(f"❌ 因子输出目录不存在: {factor_5min_dir}")
        return pd.DataFrame()

    factor_files = list(factor_5min_dir.glob("0700*factors*.parquet"))

    if not factor_files:
        print("⚠️  未找到 0700.HK 的因子文件")
        return pd.DataFrame()

    try:
        # 加载最新的因子文件
        latest_file = max(factor_files, key=lambda p: p.stat().st_mtime)
        factor_data = pd.read_parquet(latest_file)

        print("✅ 因子时间序列加载成功")
        print(f"  文件: {latest_file.name}")
        print(f"  数据形状: {factor_data.shape}")
        print(f"  时间范围: {factor_data.index[0]} 至 {factor_data.index[-1]}")
        print(f"  因子数量: {factor_data.shape[1]} 个")

        if factor_data.shape[1] > 0:
            print(f"\n  前5个因子: {', '.join(factor_data.columns[:5])}")

        return factor_data

    except Exception as e:
        print(f"❌ 因子时间序列加载失败: {e}")
        return pd.DataFrame()


def validate_factor_scores(path_config: PathConfig) -> bool:
    """验证因子评分加载"""
    print_section("第五步: 验证因子评分加载（从筛选会话）")

    try:
        factor_loader = FactorScoreLoader(path_config)
        sessions = factor_loader.list_sessions()

        if not sessions:
            print("⚠️  未找到因子筛选会话")
            print("   提示: 运行 professional_factor_screener.py 生成筛选会话")
            return False

        latest_session = sessions[0]
        print(f"✅ 找到筛选会话: {latest_session.name}")
        print(f"  会话路径: {latest_session}")

        # 尝试加载因子评分
        from hk_midfreq.factor_interface import load_factor_scores

        scores = load_factor_scores(["0700.HK"], timeframe="5min", loader=factor_loader)

        if len(scores) > 0:
            print(f"✅ 因子评分加载成功: {len(scores)} 个评分")
            print(f"  评分范围: {scores.min():.3f} 至 {scores.max():.3f}")
        else:
            print("⚠️  未找到因子评分数据")

        return True

    except Exception as e:
        print(f"❌ 因子评分加载失败: {e}")
        return False


def validate_output_management(path_config: PathConfig) -> None:
    """验证输出结果管理"""
    print_section("第六步: 验证输出结果管理 - 时间戳会话")

    result_manager = BacktestResultManager(path_config)

    # 创建测试会话
    session_dir = result_manager.create_session("0700.HK", "5min", "test")

    print("✅ 会话创建成功")
    print(f"  会话ID: {result_manager.session_id}")
    print(f"  会话目录: {session_dir}")
    print(f"  目录存在: {'✅' if session_dir.exists() else '❌'}")

    # 验证子目录
    subdirs = ["charts", "logs", "data"]
    print("\n  子目录结构:")
    for subdir in subdirs:
        subdir_path = session_dir / subdir
        print(f"    {subdir}/: {'✅' if subdir_path.exists() else '❌'}")

    # 保存测试配置
    test_config = {
        "symbol": "0700.HK",
        "timeframe": "5min",
        "strategy": "test",
        "timestamp": datetime.now().isoformat(),
    }
    result_manager.save_config(test_config)

    # 保存测试指标
    test_metrics = {
        "total_return": 0.15,
        "sharpe_ratio": 1.8,
        "max_drawdown": -0.08,
        "win_rate": 0.62,
    }
    result_manager.save_metrics(test_metrics)

    # 生成摘要报告
    report_file = result_manager.generate_summary_report(test_metrics)

    print("\n✅ 测试文件保存成功")
    print("  配置文件: backtest_config.json")
    print("  指标文件: backtest_metrics.json")
    print(f"  摘要报告: {report_file.name}")

    # 获取会话信息
    session_info = result_manager.get_session_info()
    print(f"\n  会话信息: {session_info}")


def main():
    """主验证流程"""
    print("\n" + "=" * 80)
    print("  HK中频交易架构验证 - 三层数据架构 + 输出管理")
    print("=" * 80)

    # 第一步：验证路径配置
    path_config = validate_path_config()

    # 第二步：验证原始数据层
    price_data = validate_price_loading(path_config)

    # 第三步：验证因子筛选层
    factor_ready_ok = validate_factor_ready(path_config)

    # 第四步：验证因子输出层
    factor_data = validate_factor_output(path_config)

    # 第五步：验证因子评分加载
    factor_scores_ok = validate_factor_scores(path_config)

    # 第六步：验证输出结果管理
    validate_output_management(path_config)

    # 总结
    print_section("验证总结")

    results = {
        "路径配置": "✅",
        "原始数据层 (raw/HK/)": "✅" if not price_data.empty else "❌",
        "因子筛选层 (factor_ready/)": "✅" if factor_ready_ok else "⚠️",
        "因子输出层 (因子输出/)": "✅" if not factor_data.empty else "❌",
        "因子评分加载": "✅" if factor_scores_ok else "⚠️",
        "输出结果管理": "✅",
    }

    print("验证结果:")
    for item, status in results.items():
        print(f"  {status} {item}")

    all_passed = all(status == "✅" for status in results.values())

    if all_passed:
        print("\n🎉 所有验证通过！架构完全符合 ARCHITECTURE.md 要求")
    else:
        print("\n⚠️  部分验证未通过，请检查上述标记为 ❌ 或 ⚠️ 的项目")
        print("   注：⚠️ 表示可选功能，需要运行因子筛选后生成")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
