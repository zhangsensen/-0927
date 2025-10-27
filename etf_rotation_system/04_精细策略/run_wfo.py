#!/usr/bin/env python3
"""
WFO主入口脚本
运行完整的Walk-Forward Optimization流程
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from utils.config_loader import get_config_loader
from wfo import WFOAnalyzer, WFOBacktestRunner, WFOConfig, WFOEngine, WFOOptimizer

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_wfo_config(config_loader) -> WFOConfig:
    """从配置文件设置WFO配置"""
    wfo_section = config_loader.get("wfo_config", {})

    # 时间窗口
    time_windows = wfo_section.get("time_windows", {})
    train_window_months = time_windows.get("train_window_months", 12)
    test_window_months = time_windows.get("test_window_months", 3)
    step_months = time_windows.get("step_months", 3)

    # 数据范围
    data_range = wfo_section.get("data_range", {})
    start_str = data_range.get("start_date", "2022-01-01")
    end_str = data_range.get("end_date", "2024-12-31")

    data_start_date = datetime.strptime(start_str, "%Y-%m-%d")
    data_end_date = datetime.strptime(end_str, "%Y-%m-%d")

    # IS/OOS配置
    in_sample = wfo_section.get("in_sample", {})
    out_of_sample = wfo_section.get("out_of_sample", {})

    top_n_strategies = in_sample.get("top_n_strategies", 10)
    min_is_sharpe = in_sample.get("min_sharpe", 0.4)
    min_oos_sharpe = out_of_sample.get("min_sharpe", 0.3)

    # 过拟合检测
    overfit_detection = wfo_section.get("overfit_detection", {})
    max_overfit_ratio = overfit_detection.get("max_overfit_ratio", 1.5)
    max_decay_rate = overfit_detection.get("max_decay_rate", 0.25)

    # 并行配置
    parallel = wfo_section.get("parallel", {})
    n_workers = parallel.get("n_workers", 6)
    chunk_size = parallel.get("chunk_size", 10)

    # 随机种子
    random_seed = wfo_section.get("random_seed", 42)

    wfo_config = WFOConfig(
        train_window_months=train_window_months,
        test_window_months=test_window_months,
        step_months=step_months,
        data_start_date=data_start_date,
        data_end_date=data_end_date,
        top_n_strategies=top_n_strategies,
        min_is_sharpe=min_is_sharpe,
        max_overfit_ratio=max_overfit_ratio,
        max_decay_rate=max_decay_rate,
        min_oos_sharpe=min_oos_sharpe,
        n_workers=n_workers,
        chunk_size=chunk_size,
        random_seed=random_seed,
    )

    logger.info("WFO配置加载完成:")
    logger.info(f"  训练窗口: {train_window_months}月")
    logger.info(f"  测试窗口: {test_window_months}月")
    logger.info(f"  步进: {step_months}月")
    logger.info(f"  数据范围: {start_str} ~ {end_str}")
    logger.info(f"  预计周期数: {wfo_config.estimate_periods()}")

    return wfo_config


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="WFO (Walk-Forward Optimization) 主程序"
    )
    parser.add_argument(
        "--vbt-results", type=str, required=True, help="VBT回测结果目录路径"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config/fine_strategy_config.yaml",
        help="配置文件路径（默认: ./config/fine_strategy_config.yaml）",
    )
    parser.add_argument("--output", type=str, default="../output/wfo", help="输出目录")
    parser.add_argument("--top-k", type=int, default=200, help="从VBT结果提取Top-K策略")

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("WFO (Walk-Forward Optimization) 启动")
    logger.info("=" * 80)
    logger.info(f"VBT结果路径: {args.vbt_results}")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"输出目录: {args.output}")
    logger.info("")

    try:
        # 1. 加载配置
        logger.info("Step 1/6: 加载配置...")
        config_loader = get_config_loader(args.config)
        wfo_config = setup_wfo_config(config_loader)

        # 2. 初始化组件
        logger.info("\nStep 2/6: 初始化组件...")
        backtest_runner = WFOBacktestRunner(args.vbt_results)
        optimizer = WFOOptimizer()
        analyzer = WFOAnalyzer()
        engine = WFOEngine(wfo_config)

        # 3. 准备基础策略
        logger.info(f"\nStep 3/6: 准备基础策略 (Top {args.top_k})...")
        base_strategies = backtest_runner.prepare_strategies_from_results(
            top_k=args.top_k
        )
        logger.info(f"准备了 {len(base_strategies)} 个基础策略")

        # 4. 生成WFO周期
        logger.info("\nStep 4/6: 生成WFO周期...")
        periods = engine.generate_periods()
        logger.info(f"生成了 {len(periods)} 个WFO周期")

        # 5. 运行WFO流程
        logger.info("\nStep 5/6: 运行WFO流程...")
        results = engine.run_wfo_pipeline(
            backtest_runner=backtest_runner,
            optimizer=optimizer,
            analyzer=analyzer,
            base_strategies=base_strategies,
        )

        # 6. 保存结果
        logger.info("\nStep 6/6: 保存结果...")
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存完整结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"wfo_results_{timestamp}.json"
        engine.save_results(results_file)

        # 保存分析报告
        analysis_file = output_dir / f"wfo_analysis_{timestamp}.json"
        analyzer.save_analysis(results["overfit_analysis"], analysis_file)

        # 生成Markdown报告
        report_file = output_dir / f"WFO_REPORT_{timestamp}.md"
        analyzer.generate_report(results["overfit_analysis"], report_file)

        logger.info("\n" + "=" * 80)
        logger.info("✅ WFO流程完成")
        logger.info("=" * 80)
        logger.info(f"完整结果: {results_file}")
        logger.info(f"分析报告: {analysis_file}")
        logger.info(f"Markdown报告: {report_file}")
        logger.info("")

        # 打印核心结论
        summary = results["summary"]
        recommendations = results["overfit_analysis"]["recommendations"]

        logger.info("【核心结论】")
        logger.info(f"  总周期数: {summary.get('total_periods', 0)}")
        logger.info(f"  OOS平均Sharpe: {summary.get('oos_sharpe_mean', 0):.3f}")
        logger.info(f"  过拟合比: {summary.get('avg_overfit_ratio', 0):.3f}")
        logger.info(f"  性能衰减: {summary.get('performance_decay', 0)*100:.1f}%")
        logger.info(f"  综合评级: {recommendations.get('overall_grade', 'N/A')}")
        logger.info(
            f"  可部署: {'✅ 是' if recommendations.get('deployment_ready', False) else '❌ 否'}"
        )
        logger.info("")

        return 0

    except Exception as e:
        logger.error(f"WFO流程执行失败: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
