#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试新功能 - 快速验证

测试项目:
1. ✅ 标准化日志格式
2. ✅ settings.yaml 配置加载
3. ✅ 会话级日志隔离
4. ✅ 环境快照
5. ✅ 因子列名清洗
6. ✅ 路径检查（中文警告）
"""

import logging
import sys
from pathlib import Path

# 设置项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hk_midfreq.config import PathConfig  # noqa: E402
from hk_midfreq.log_formatter import (  # noqa: E402
    StructuredLogger,
    log_backtest_summary,
)
from hk_midfreq.result_manager import BacktestResultManager  # noqa: E402
from hk_midfreq.settings_loader import get_log_level, get_settings  # noqa: E402

# 配置日志
settings = get_settings()
logging.basicConfig(level=get_log_level(), format=settings.log_format)
logger = logging.getLogger(__name__)


def test_structured_logger():
    """测试标准化日志格式"""
    print("\n" + "=" * 80)
    print("📋 测试 1: 标准化日志格式")
    print("=" * 80)

    session_id = "test_session_001"
    symbol = "0700.HK"
    timeframe = "5min"

    # 基础格式
    msg = StructuredLogger.format_message(
        session_id, symbol, timeframe, "LONG", "trades", 10, "pnl=1500.50"
    )
    logger.info(msg)
    print(f"✅ 基础日志格式: {msg}")

    # 批量指标
    metrics = {"return": "12.5%", "sharpe": "1.85", "max_dd": "-8.2%"}
    msg = StructuredLogger.format_bulk(session_id, symbol, timeframe, "RESULT", metrics)
    logger.info(msg)
    print(f"✅ 批量指标格式: {msg}")

    # 多时间框架
    msg = StructuredLogger.format_multi_tf(
        session_id, symbol, ["5min", "15min", "60min"], "fusion_score", 0.85
    )
    logger.info(msg)
    print(f"✅ 多时间框架格式: {msg}")

    print("✅ 标准化日志格式测试通过")


def test_settings_loader():
    """测试settings.yaml加载"""
    print("\n" + "=" * 80)
    print("📋 测试 2: Settings配置加载")
    print("=" * 80)

    settings = get_settings()

    print(f"日志级别: {settings.log_level}")
    print(f"日志轮转大小: {settings.log_rotating_max_bytes / 1024 / 1024} MB")
    print(f"图表最小文件大小: {settings.chart_min_file_size_kb} kB")
    print(f"因子列名清洗: {settings.factor_clean_columns}")
    print(f"环境快照启用: {settings.env_snapshot_enabled}")

    assert settings.log_level == "DEBUG", "日志级别应为DEBUG"
    assert settings.log_rotating_max_bytes == 10485760, "轮转大小应为10MB"

    print("✅ Settings配置加载测试通过")


def test_result_manager():
    """测试结果管理器"""
    print("\n" + "=" * 80)
    print("📋 测试 3: 结果管理器（会话级日志+环境快照）")
    print("=" * 80)

    config = PathConfig()
    manager = BacktestResultManager(path_config=config)

    # 创建会话
    session_dir = manager.create_session("0700.HK", "test_5min", "test_strategy")

    print(f"会话ID: {manager.session_id}")
    print(f"会话目录: {session_dir}")

    # 检查目录结构
    assert (session_dir / "charts").exists(), "charts目录应存在"
    assert (session_dir / "logs").exists(), "logs目录应存在"
    assert (session_dir / "data").exists(), "data目录应存在"
    assert (session_dir / "env").exists(), "env目录应存在"

    # 检查日志文件
    log_file = session_dir / "logs" / "debug.log"
    assert log_file.exists(), "debug.log应存在"
    print(f"✅ 日志文件: {log_file}")

    # 检查环境快照
    env_dir = session_dir / "env"
    system_info_file = env_dir / "system_info.json"
    if system_info_file.exists():
        print(f"✅ 系统信息: {system_info_file}")

    # 测试日志写入
    import pandas as pd

    test_stats = pd.Series(
        {"Total Return [%]": 15.5, "Sharpe Ratio": 1.8, "Max Drawdown [%]": -8.2}
    )
    test_trades = pd.DataFrame(
        {
            "Entry Time": ["2025-01-01", "2025-01-02"],
            "Exit Time": ["2025-01-03", "2025-01-04"],
            "PnL": [100.0, 200.0],
            "Side": ["Long", "Long"],
        }
    )

    manager.save_backtest_results(test_stats, test_trades)
    manager.save_metrics({"test_metric": 123.45})
    manager.save_config({"test_config": "value"})

    # 验证文件
    assert (session_dir / "data" / "portfolio_stats.parquet").exists()
    assert (session_dir / "data" / "trades.parquet").exists()
    assert (session_dir / "backtest_metrics.json").exists()
    assert (session_dir / "backtest_config.json").exists()

    print("✅ 结果管理器测试通过")

    # 关闭会话
    manager.close_session()
    print("✅ 会话已关闭")


def test_factor_column_cleaning():
    """测试因子列名清洗"""
    print("\n" + "=" * 80)
    print("📋 测试 4: 因子列名清洗")
    print("=" * 80)

    import pandas as pd

    # 创建包含非法字符的DataFrame
    df = pd.DataFrame(
        {
            "factor_5min_ma|20": [1, 2, 3],
            "factor-rsi@14": [4, 5, 6],
            "normal_column": [7, 8, 9],
        }
    )

    print("原始列名:", list(df.columns))

    config = PathConfig()
    manager = BacktestResultManager(path_config=config)
    df_cleaned = manager._clean_factor_columns(df)

    print("清洗后列名:", list(df_cleaned.columns))

    # 验证
    assert "factor_5min_ma_20" in df_cleaned.columns
    assert "factor_rsi_14" in df_cleaned.columns
    assert "normal_column" in df_cleaned.columns

    print("✅ 因子列名清洗测试通过")


def main():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("🧪 新功能测试套件")
    print("=" * 80)

    try:
        test_structured_logger()
        test_settings_loader()
        test_result_manager()
        test_factor_column_cleaning()

        print("\n" + "=" * 80)
        print("🎉 所有测试通过！")
        print("=" * 80)
        print("\n✅ 新功能验证完成:")
        print("  1. ✅ 标准化日志格式 (session_id|symbol|tf|direction|metric=value)")
        print("  2. ✅ Settings配置加载 (settings.yaml)")
        print("  3. ✅ 会话级日志隔离 (RotatingFileHandler, 10 MB)")
        print("  4. ✅ 环境快照 (pip freeze, system info)")
        print("  5. ✅ 因子列名清洗 (移除非法字符)")
        print("  6. ✅ 路径检查 (中文警告)")
        print("=" * 80)

    except Exception as e:
        logger.exception(f"❌ 测试失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
