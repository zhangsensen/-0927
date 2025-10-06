#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多时间框架组合回测 - 严格工程化规范

核心约束:
1. 标准化日志: {session_id}|{symbol}|{tf}|{direction}|{metric}={value}
2. 从 settings.yaml 读取配置，禁止硬编码
3. 会话级日志隔离 (10 MB RotatingFileHandler)
4. 图表文件大小检查 (< 3 kB 抛异常)
5. 环境快照 (pip freeze)
6. 真实数据 + 真实信号
"""

import sys
from pathlib import Path

# 设置项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 从settings.yaml读取日志配置
import logging  # noqa: E402

from hk_midfreq.backtest_engine import run_portfolio_backtest  # noqa: E402
from hk_midfreq.config import PathConfig  # noqa: E402
from hk_midfreq.log_formatter import StructuredLogger  # noqa: E402
from hk_midfreq.price_loader import PriceDataLoader  # noqa: E402
from hk_midfreq.result_manager import BacktestResultManager  # noqa: E402
from hk_midfreq.settings_loader import get_log_level, get_settings  # noqa: E402
from hk_midfreq.strategy_core import StrategyCore  # noqa: E402

settings = get_settings()
logging.basicConfig(level=get_log_level(), format=settings.log_format)
logger = logging.getLogger(__name__)


def main():
    """执行多时间框架回测 - 严格规范"""
    symbol = "0700.HK"
    timeframe_composite = "multi_tf"

    # 提前生成session_id，避免使用temp_session
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_symbol = symbol.replace(".", "_")
    session_id = f"{clean_symbol}_HK_midfreq_{timeframe_composite}_{timestamp}"

    print(f"🚀 启动多时间框架回测 - 会话ID: {session_id}")
    print(f"📊 标的: {symbol} | 时间框架: {timeframe_composite}")

    msg = StructuredLogger.format_message(
        session_id, symbol, timeframe_composite, "START", "backtest_started", "TRUE"
    )
    logger.info(msg)

    # 1. 初始化配置
    config = PathConfig()
    print(f"📁 项目根目录: {config.project_root}")

    msg = StructuredLogger.format_message(
        session_id,
        symbol,
        timeframe_composite,
        "INIT",
        "config_loaded",
        str(config.project_root),
    )
    logger.info(msg)

    # 2. 提前创建会话目录和输出重定向
    result_manager = BacktestResultManager(path_config=config)

    # 手动设置session信息，避免重复生成
    result_manager.session_id = session_id
    result_manager.symbol = symbol
    result_manager.timeframe = timeframe_composite

    # 创建会话目录结构
    output_dir = result_manager.path_config.backtest_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    session_dir = output_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    result_manager.session_dir = session_dir

    # 创建子目录
    (session_dir / "charts").mkdir(exist_ok=True)
    (session_dir / "logs").mkdir(exist_ok=True)
    (session_dir / "data").mkdir(exist_ok=True)
    (session_dir / "env").mkdir(exist_ok=True)

    print(f"📂 会话目录: {session_dir}")

    # 立即启动输出重定向
    from hk_midfreq.result_manager import OutputRedirector

    output_redirector = OutputRedirector(session_dir / "logs", result_manager.settings)
    output_redirector.start_redirect()

    print(f"📝 输出重定向已启动 - 所有输出将保存到会话日志")

    # 继续完成result_manager初始化
    result_manager._setup_session_logging()
    result_manager.output_redirector = output_redirector

    # 统一日志handler - 将session handler添加到主logger和相关模块
    session_handler = result_manager.log_handler
    if session_handler:
        # 添加到主程序logger
        logger.addHandler(session_handler)

        # 添加到相关模块的logger
        import logging

        for module_name in [
            "hk_midfreq.price_loader",
            "hk_midfreq.strategy_core",
            "hk_midfreq.factor_interface",
            "hk_midfreq.fusion",
        ]:
            module_logger = logging.getLogger(module_name)
            module_logger.addHandler(session_handler)

    # 从此处开始使用真实session_id
    msg = StructuredLogger.format_message(
        session_id, symbol, timeframe_composite, "INIT", "session_dir", str(session_dir)
    )
    logger.info(msg)

    # 3. 加载多时间框架价格数据
    print(f"📊 开始加载多时间框架价格数据...")

    price_loader = PriceDataLoader(root=config.hk_raw_dir)
    timeframes = ["5min", "15min", "30min", "60min", "daily"]

    print(f"📈 目标时间框架: {timeframes}")
    logger.info(f"初始化价格加载器 - 数据根目录: {config.hk_raw_dir}")

    price_data_multi_tf = {}
    successful_loads = 0

    for i, tf in enumerate(timeframes, 1):
        print(f"📊 [{i}/{len(timeframes)}] 加载 {symbol} - {tf} 数据...")

        try:
            logger.debug(f"开始加载时间框架: {tf}")
            price = price_loader.load_price(symbol, tf)
            price_data_multi_tf[tf] = price
            successful_loads += 1

            # 数据质量统计
            data_start = price.index[0] if len(price) > 0 else "N/A"
            data_end = price.index[-1] if len(price) > 0 else "N/A"

            print(f"  ✅ {tf}: {len(price)} 条记录 ({data_start} 到 {data_end})")

            msg = StructuredLogger.format_message(
                session_id, symbol, tf, "LOAD", "price_data", f"{len(price)}_bars"
            )
            logger.info(msg)

            # 记录数据质量信息
            logger.debug(
                f"{tf} 数据质量 - 列: {list(price.columns)}, 时间范围: {data_start} 到 {data_end}"
            )

        except Exception as e:
            print(f"  ❌ {tf}: 加载失败 - {e}")

            msg = StructuredLogger.format_message(
                session_id, symbol, tf, "ERROR", "price_load_failed", str(e)
            )
            logger.error(msg)
            raise

    print(
        f"📊 价格数据加载完成 - 成功: {successful_loads}/{len(timeframes)} 个时间框架"
    )
    logger.info(f"多时间框架价格数据加载完成 - 总计 {successful_loads} 个时间框架")

    # 4. 初始化策略核心
    strategy = StrategyCore()
    msg = StructuredLogger.format_message(
        session_id, symbol, timeframe_composite, "INIT", "strategy_core", "SUCCESS"
    )
    logger.info(msg)

    # 4.1 尝试加载因子数据（如果可用）
    try:
        from hk_midfreq.factor_interface import FactorScoreLoader

        factor_loader = FactorScoreLoader()
        sessions = factor_loader.list_sessions()
        if sessions:
            msg = StructuredLogger.format_message(
                session_id,
                symbol,
                timeframe_composite,
                "FACTOR",
                "sessions_found",
                len(sessions),
            )
            logger.info(msg)

            # 尝试加载因子面板
            try:
                panel = factor_loader.load_factor_panels(
                    symbols=[symbol], timeframes=timeframes, max_factors=20
                )
                if not panel.empty:
                    msg = StructuredLogger.format_message(
                        session_id,
                        symbol,
                        timeframe_composite,
                        "FACTOR",
                        "panel_loaded",
                        len(panel),
                    )
                    logger.info(msg)
            except Exception as e:
                msg = StructuredLogger.format_message(
                    session_id,
                    symbol,
                    timeframe_composite,
                    "FACTOR",
                    "panel_load_failed",
                    str(e),
                )
                logger.warning(msg)
        else:
            msg = StructuredLogger.format_message(
                session_id,
                symbol,
                timeframe_composite,
                "FACTOR",
                "no_sessions",
                "using_price_only",
            )
            logger.info(msg)
    except Exception as e:
        msg = StructuredLogger.format_message(
            session_id,
            symbol,
            timeframe_composite,
            "FACTOR",
            "loader_init_failed",
            str(e),
        )
        logger.warning(msg)

    # 5. 生成多时间框架组合信号
    msg = StructuredLogger.format_message(
        session_id,
        symbol,
        timeframe_composite,
        "SIGNAL",
        "generation_started",
        "TRUE",
    )
    logger.info(msg)

    # 构建价格字典 {symbol: {timeframe: price}}
    price_dict = {symbol: price_data_multi_tf}

    # 生成信号 (不指定timeframe，触发多时间框架融合)
    msg = StructuredLogger.format_message(
        session_id,
        symbol,
        timeframe_composite,
        "SIGNAL",
        "build_universe_started",
        "TRUE",
    )
    logger.debug(msg)

    signals = strategy.build_signal_universe(price_dict)

    msg = StructuredLogger.format_message(
        session_id,
        symbol,
        timeframe_composite,
        "SIGNAL",
        "build_universe_completed",
        len(signals),
    )
    logger.info(msg)

    if symbol not in signals:
        msg = StructuredLogger.format_message(
            session_id, symbol, timeframe_composite, "ERROR", "signals_empty", "TRUE"
        )
        logger.error(msg)
        raise RuntimeError(f"未能为 {symbol} 生成有效信号")

    # signals[symbol] 是 StrategySignals 对象
    signal_obj = signals[symbol]
    if signal_obj.entries.empty or signal_obj.entries.sum() == 0:
        msg = StructuredLogger.format_message(
            session_id,
            symbol,
            timeframe_composite,
            "ERROR",
            "no_entry_signals",
            "TRUE",
        )
        logger.error(msg)
        raise RuntimeError(f"未能为 {symbol} 生成有效入场信号")

    signal_count = signal_obj.entries.sum()
    msg = StructuredLogger.format_message(
        session_id, symbol, timeframe_composite, "SIGNAL", "count", signal_count
    )
    logger.info(msg)

    # 6. 向量化回测
    msg = StructuredLogger.format_message(
        session_id,
        symbol,
        timeframe_composite,
        "BACKTEST",
        "execution_started",
        "TRUE",
    )
    logger.info(msg)

    # 构建回测输入：使用 run_portfolio_backtest 的标准接口
    artifacts = run_portfolio_backtest(
        price_data={symbol: price_data_multi_tf},
        signals=signals,
    )

    if artifacts is None:
        msg = StructuredLogger.format_message(
            session_id,
            symbol,
            timeframe_composite,
            "ERROR",
            "backtest_failed",
            "NO_ARTIFACTS",
        )
        logger.error(msg)
        raise RuntimeError(f"回测未生成有效结果")

    portfolio = artifacts.portfolio

    msg = StructuredLogger.format_message(
        session_id,
        symbol,
        timeframe_composite,
        "BACKTEST",
        "execution_completed",
        "SUCCESS",
    )
    logger.info(msg)

    # 7. 提取回测结果
    try:
        stats = portfolio.stats()
        trades = portfolio.trades.records_readable
        positions = portfolio.positions.records_readable

        # 记录stats的所有可用指标
        msg = StructuredLogger.format_message(
            session_id,
            symbol,
            timeframe_composite,
            "RESULT",
            "stats_keys",
            str(stats.index.tolist()),
        )
        logger.debug(msg)

        # 安全提取关键指标（保持数值类型）
        key_metrics = {}
        key_metrics_display = {}  # 用于日志显示的格式化版本

        # 尝试多种可能的列名
        total_return_keys = ["Total Return [%]", "Total Return", "total_return"]
        for key in total_return_keys:
            if key in stats.index:
                value = float(stats.loc[key])
                key_metrics["total_return"] = value
                key_metrics_display["total_return"] = f"{value:.2f}"
                break
        else:
            key_metrics["total_return"] = None
            key_metrics_display["total_return"] = "N/A"

        sharpe_keys = ["Sharpe Ratio", "Sharpe", "sharpe_ratio"]
        for key in sharpe_keys:
            if key in stats.index:
                value = float(stats.loc[key])
                key_metrics["sharpe"] = value
                key_metrics_display["sharpe"] = f"{value:.2f}"
                break
        else:
            key_metrics["sharpe"] = None
            key_metrics_display["sharpe"] = "N/A"

        max_dd_keys = ["Max Drawdown [%]", "Max Drawdown", "max_drawdown"]
        for key in max_dd_keys:
            if key in stats.index:
                value = float(stats.loc[key])
                key_metrics["max_dd"] = value
                key_metrics_display["max_dd"] = f"{value:.2f}"
                break
        else:
            key_metrics["max_dd"] = None
            key_metrics_display["max_dd"] = "N/A"

        total_trades_keys = ["Total Trades", "Trades", "total_trades"]
        for key in total_trades_keys:
            if key in stats.index:
                value = int(stats.loc[key])
                key_metrics["total_trades"] = value
                key_metrics_display["total_trades"] = str(value)
                break
        else:
            key_metrics["total_trades"] = 0
            key_metrics_display["total_trades"] = "0"

        msg = StructuredLogger.format_bulk(
            session_id, symbol, timeframe_composite, "RESULT", key_metrics_display
        )
        logger.info(msg)

    except Exception as e:
        msg = StructuredLogger.format_message(
            session_id,
            symbol,
            timeframe_composite,
            "ERROR",
            "result_extraction",
            str(e),
        )
        logger.error(msg)
        raise

    # 8. 保存结果
    try:
        msg = StructuredLogger.format_message(
            session_id, symbol, timeframe_composite, "SAVE", "data_started", "TRUE"
        )
        logger.debug(msg)

        result_manager.save_backtest_results(stats, trades, positions)

        msg = StructuredLogger.format_message(
            session_id, symbol, timeframe_composite, "SAVE", "data_completed", "SUCCESS"
        )
        logger.info(msg)

        result_manager.save_metrics(key_metrics)  # 使用数值类型版本
        result_manager.save_config(
            {
                "symbol": symbol,
                "timeframes": timeframes,
                "initial_cash": settings.get("backtest.initial_cash", 1000000.0),
                "commission": settings.get("backtest.commission", 0.002),
                "strategy": "multi_timeframe_fusion",
            }
        )

        msg = StructuredLogger.format_message(
            session_id, symbol, timeframe_composite, "SAVE", "all_results", "SUCCESS"
        )
        logger.info(msg)

        # 检查数据目录
        data_dir = session_dir / "data"
        data_files = list(data_dir.glob("*.csv")) + list(data_dir.glob("*.parquet"))
        msg = StructuredLogger.format_message(
            session_id,
            symbol,
            timeframe_composite,
            "SAVE",
            "data_files_count",
            len(data_files),
        )
        logger.info(msg)

    except Exception as e:
        msg = StructuredLogger.format_message(
            session_id, symbol, timeframe_composite, "ERROR", "save_failed", str(e)
        )
        logger.error(msg)
        raise

    # 9. 生成图表
    try:
        msg = StructuredLogger.format_message(
            session_id,
            symbol,
            timeframe_composite,
            "CHART",
            "generation_started",
            "TRUE",
        )
        logger.debug(msg)

        # 传递已处理的trades数据（包含Return [%]列）
        result_manager.generate_charts(stats, trades)

        # 检查图表目录
        charts_dir = session_dir / "charts"
        chart_files = list(charts_dir.glob("*.png")) + list(charts_dir.glob("*.jpg"))
        msg = StructuredLogger.format_message(
            session_id,
            symbol,
            timeframe_composite,
            "CHART",
            "files_count",
            len(chart_files),
        )
        logger.info(msg)

        if len(chart_files) == 0:
            msg = StructuredLogger.format_message(
                session_id,
                symbol,
                timeframe_composite,
                "CHART",
                "no_charts_generated",
                "WARNING",
            )
            logger.warning(msg)
        else:
            msg = StructuredLogger.format_message(
                session_id,
                symbol,
                timeframe_composite,
                "CHART",
                "all_charts",
                "SUCCESS",
            )
            logger.info(msg)

    except Exception as e:
        msg = StructuredLogger.format_message(
            session_id, symbol, timeframe_composite, "ERROR", "chart_failed", str(e)
        )
        logger.error(msg)
        # 图表失败不中断流程

        # 记录图表目录为空
        msg = StructuredLogger.format_message(
            session_id,
            symbol,
            timeframe_composite,
            "CHART",
            "directory_empty",
            "due_to_error",
        )
        logger.warning(msg)

    # 10. 生成摘要报告
    try:
        report_path = result_manager.generate_summary_report(key_metrics)
        msg = StructuredLogger.format_message(
            session_id, symbol, timeframe_composite, "SAVE", "summary_report", "SUCCESS"
        )
        logger.info(msg)

    except Exception as e:
        msg = StructuredLogger.format_message(
            session_id, symbol, timeframe_composite, "ERROR", "report_failed", str(e)
        )
        logger.error(msg)

    # 11. 打印最终结果 - 在关闭重定向前
    print("\n" + "=" * 80)
    print(f"✅ 多时间框架回测完成: {symbol}")
    print("=" * 80)
    print(f"📁 会话目录: {session_dir}")
    print(f"🆔 会话ID:   {session_id}")
    print("\n📊 关键指标:")
    for metric, value in key_metrics.items():
        print(f"  - {metric}: {value}")
    print("\n📁 输出文件:")
    print(f"  - 日志:   {session_dir / 'logs' / 'debug.log'}")
    print(f"  - 数据:   {session_dir / 'data'}")
    print(f"  - 图表:   {session_dir / 'charts'}")
    print(f"  - 报告:   {session_dir / 'summary_report.md'}")
    print(f"  - 环境:   {session_dir / 'env' / 'pip_freeze.txt'}")
    print("=" * 80)

    # 12. 关闭会话和清理
    # 移除添加的session handler
    if session_handler:
        logger.removeHandler(session_handler)
        for module_name in [
            "hk_midfreq.price_loader",
            "hk_midfreq.strategy_core",
            "hk_midfreq.factor_interface",
            "hk_midfreq.fusion",
        ]:
            module_logger = logging.getLogger(module_name)
            module_logger.removeHandler(session_handler)

    # 关闭会话（但保持输出重定向）
    result_manager.close_session()
    msg = StructuredLogger.format_message(
        session_id, symbol, timeframe_composite, "CLOSE", "session_closed", "SUCCESS"
    )
    logger.info(msg)

    # 最后停止输出重定向
    if "output_redirector" in locals():
        output_redirector.stop_redirect()
        print("📝 输出重定向已停止 - 所有日志已保存")


if __name__ == "__main__":
    session_id = None
    symbol = None
    output_redirector = None

    try:
        main()
    except Exception as e:
        # 尝试获取session_id和symbol用于日志
        import traceback

        error_detail = traceback.format_exc()

        # 如果能访问到全局变量，记录到会话日志
        try:
            from hk_midfreq.log_formatter import StructuredLogger

            if "result_manager" in dir() and hasattr(result_manager, "session_id"):
                session_id = result_manager.session_id
                symbol = result_manager.symbol or "UNKNOWN"
                msg = StructuredLogger.format_message(
                    session_id, symbol, "multi_tf", "FATAL", "exception", str(e)
                )
                logger.error(msg)
                logger.error(f"完整堆栈:\n{error_detail}")
        except:
            pass

        logger.exception(f"❌ 回测失败: {e}")
        print(f"\n{'='*80}")
        print(f"❌ 回测执行失败")
        print(f"{'='*80}")
        print(f"错误: {e}")
        print(f"\n完整堆栈:\n{error_detail}")
        print(f"{'='*80}")

        # 确保在异常情况下也停止输出重定向
        try:
            if "output_redirector" in locals() and output_redirector:
                output_redirector.stop_redirect()
                print("📝 输出重定向已停止（异常退出）")
        except:
            pass

        sys.exit(1)
    finally:
        # 最终清理：确保输出重定向被停止
        try:
            if "output_redirector" in locals() and output_redirector:
                output_redirector.stop_redirect()
        except:
            pass
