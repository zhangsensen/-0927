#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版WFO生产环境运行器
修复嵌套并行和数据重复加载问题

关键修改:
1. 移除Period级并行 - 顺序处理Period
2. 数据加载一次 - 所有Period共享
3. 保留策略级并行 - 8个worker处理策略

预期性能: 2000策略/秒 (恢复到旧版本水平)
"""

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# 添加上级目录到路径（包含config_loader_parallel.py和parallel_backtest_configurable.py）
PARENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PARENT_DIR))

from config_loader_parallel import load_fast_config_from_args
from parallel_backtest_configurable import ConfigurableParallelBacktestEngine

logger = logging.getLogger(__name__)


class OptimizedProductionRunner:
    """
    优化版生产环境WFO Runner

    性能优化:
    - 数据加载1次 (vs 19次)
    - 顺序处理Period (vs 并行)
    - 策略级并行保留 (8 workers)
    """

    def __init__(self, config_path: str):
        """初始化Runner"""
        self.config = load_fast_config_from_args(["-c", config_path])

        # 创建时间戳子目录
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path(self.config.output_dir) / f"wfo_{self.timestamp}"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # 创建日志子目录
        self.log_dir = self.results_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)

        logger.info(f"OptimizedProductionRunner初始化完成")
        logger.info(f"结果目录: {self.results_dir}")

    def _create_wfo_periods(self) -> List[Dict[str, Any]]:
        """
        创建WFO时间窗口

        Returns:
            periods: 时间窗口列表
        """
        # 使用配置中的WFO参数
        # simple_config.yaml中定义: train_months, test_months, step_months
        train_months = getattr(self.config, "train_months", 12)
        test_months = getattr(self.config, "test_months", 3)
        step_months = getattr(self.config, "step_months", 3)

        # 从实际数据推断日期范围（避免硬编码）
        start_date = pd.Timestamp("2020-01-02")  # 数据起始日期
        end_date = pd.Timestamp("2025-10-14")  # 数据截止日期

        periods = []
        current_start = start_date

        while True:
            is_start = current_start
            is_end = is_start + pd.DateOffset(months=train_months)
            oos_start = is_end + pd.DateOffset(days=1)
            oos_end = oos_start + pd.DateOffset(months=test_months)

            if oos_end > end_date:
                break

            periods.append(
                {
                    "period_id": len(periods) + 1,
                    "is_start": is_start,
                    "is_end": is_end,
                    "oos_start": oos_start,
                    "oos_end": oos_end,
                }
            )

            current_start += pd.DateOffset(months=step_months)

        logger.info(f"创建了 {len(periods)} 个WFO Period")
        return periods

    def run_production(self) -> Dict[str, Any]:
        """
        运行优化版生产环境回测

        核心优化:
        1. 主进程加载数据一次
        2. 顺序处理每个Period
        3. 每个Period内部策略级并行

        Returns:
            results: 完整结果字典
        """
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("开始运行优化版WFO生产环境")
        logger.info("=" * 60)

        # ========================================
        # 步骤1: 加载数据 (一次性)
        # ========================================
        logger.info("\n[1/4] 加载数据 (全局加载一次)")
        data_start = datetime.now()

        # 创建engine实例
        vbt_engine = ConfigurableParallelBacktestEngine(self.config)

        # 全局加载数据 (只加载1次!)
        logger.info("  - 加载因子Panel...")
        panel = vbt_engine._load_factor_panel()
        logger.info(f"    Panel shape: {panel.shape}")

        logger.info("  - 加载价格数据...")
        prices = vbt_engine._load_price_data()
        logger.info(f"    Prices shape: {prices.shape}")

        logger.info("  - 加载Top因子...")
        factors = vbt_engine._load_top_factors()
        logger.info(f"    Top factors: {len(factors)}")

        data_time = (datetime.now() - data_start).total_seconds()
        logger.info(f"  数据加载完成，耗时: {data_time:.2f}秒")

        # ========================================
        # 步骤2: 创建WFO Period
        # ========================================
        logger.info("\n[2/4] 创建WFO时间窗口")
        periods = self._create_wfo_periods()
        logger.info(f"  共 {len(periods)} 个Period")

        # ========================================
        # 步骤3: 顺序处理每个Period
        # ========================================
        logger.info("\n[3/4] 顺序处理每个Period (内部策略级并行)")

        # 从配置读取 IS/OOS 开关
        run_is = getattr(self.config, "run_is", True)
        run_oos = getattr(self.config, "run_oos", True)
        logger.info(f"🔧 配置: run_is={run_is}, run_oos={run_oos}")
        if not run_is:
            logger.warning("⚠️  IS回测已禁用！将仅运行OOS回测")
        if not run_oos:
            logger.warning("⚠️  OOS回测已禁用！将仅运行IS回测")
        logger.info("-" * 60)

        all_results = []

        for idx, period in enumerate(periods, 1):
            period_start = datetime.now()

            logger.info(f"\n处理 Period {idx}/{len(periods)}")
            logger.info(
                f"  IS:  {period['is_start'].date()} → {period['is_end'].date()}"
            )
            logger.info(
                f"  OOS: {period['oos_start'].date()} → {period['oos_end'].date()}"
            )

            # --------------------------------
            # 3.1: 切分数据 (快速切片, 0.6ms)
            # --------------------------------
            # Panel: MultiIndex (symbol, date)
            panel_dates = panel.index.get_level_values(1)

            # Prices: DateIndex
            price_dates = prices.index

            # IS数据
            is_panel_mask = (panel_dates >= period["is_start"]) & (
                panel_dates <= period["is_end"]
            )
            is_price_mask = (price_dates >= period["is_start"]) & (
                price_dates <= period["is_end"]
            )
            is_panel = panel.loc[is_panel_mask]
            is_prices = prices.loc[is_price_mask]
            logger.info(f"  IS数据: Panel{is_panel.shape}, Prices{is_prices.shape}")

            # OOS数据
            oos_panel_mask = (panel_dates >= period["oos_start"]) & (
                panel_dates <= period["oos_end"]
            )
            oos_price_mask = (price_dates >= period["oos_start"]) & (
                price_dates <= period["oos_end"]
            )
            oos_panel = panel.loc[oos_panel_mask]
            oos_prices = prices.loc[oos_price_mask]
            logger.info(f"  OOS数据: Panel{oos_panel.shape}, Prices{oos_prices.shape}")

            # --------------------------------
            # 3.2: IS回测 - 测试所有策略组合
            # --------------------------------
            run_is = getattr(self.config, "run_is", True)

            if run_is:
                logger.info("  运行IS回测 (测试所有策略)...")
                is_backtest_start = datetime.now()

                # 创建IS期间的临时引擎
                is_engine = ConfigurableParallelBacktestEngine(self.config)

                # ✅ 传入切分后的IS数据 (修复数据泄露)
                is_results = is_engine.parallel_grid_search(
                    panel=is_panel, prices=is_prices, factors=factors
                )

                is_time = (datetime.now() - is_backtest_start).total_seconds()
                logger.info(
                    f"  IS回测完成: {len(is_results)}个结果, 耗时{is_time:.1f}秒"
                )

                # 选择Top N策略用于OOS验证
                save_top_n = getattr(self.config, "save_top_n", 300)
                top_strategies = is_results.nlargest(save_top_n, "sharpe_ratio")
                logger.info(
                    f"  IS选出Top {save_top_n}策略 (Sharpe范围: {top_strategies['sharpe_ratio'].min():.3f} - {top_strategies['sharpe_ratio'].max():.3f})"
                )
            else:
                logger.info("  跳过IS回测 (配置禁用)")
                is_results = None
                top_strategies = None

            # --------------------------------
            # 3.3: OOS回测 - 只测试IS选出的Top N策略
            # --------------------------------
            run_oos = getattr(self.config, "run_oos", True)

            if run_oos and top_strategies is not None:
                logger.info(
                    f"  运行OOS回测 (验证IS选出的{len(top_strategies)}个策略)..."
                )
                oos_backtest_start = datetime.now()

                # 创建OOS期间的临时引擎
                oos_engine = ConfigurableParallelBacktestEngine(self.config)

                # 提取策略参数 (weights是字符串格式需要转回dict)
                import ast

                strategy_params = []
                for _, row in top_strategies.iterrows():
                    # 将字符串格式的weights转回字典
                    weights_dict = (
                        ast.literal_eval(row["weights"])
                        if isinstance(row["weights"], str)
                        else row["weights"]
                    )
                    strategy_params.append(
                        {
                            "weights": weights_dict,
                            "top_n": row["top_n"],
                            "rebalance_freq": row["rebalance_freq"],
                        }
                    )

                # ✅ 回测指定策略 + 传入OOS数据 (修复WFO逻辑)
                oos_results = oos_engine.backtest_specific_strategies(
                    strategy_params=strategy_params, panel=oos_panel, prices=oos_prices
                )

                oos_time = (datetime.now() - oos_backtest_start).total_seconds()
                logger.info(
                    f"  OOS回测完成: {len(oos_results)}个结果, 耗时{oos_time:.1f}秒"
                )

                # 统计OOS验证通过率 (只有在有结果时才统计)
                if len(oos_results) > 0:
                    oos_pass = oos_results[oos_results["sharpe_ratio"] > 0.3]
                    logger.info(
                        f"  OOS验证: {len(oos_pass)}/{len(oos_results)}策略通过 (Sharpe > 0.3)"
                    )
                else:
                    logger.warning("  ⚠️  OOS回测返回空结果！")
            elif run_oos and top_strategies is None:
                logger.warning("  跳过OOS回测 (IS未运行)")
                oos_results = None
            else:
                logger.info("  跳过OOS回测 (配置禁用)")
                oos_results = None

            # --------------------------------
            # 3.4: 合并结果
            # --------------------------------
            period_results = {
                "period_id": period["period_id"],
                "is_start": period["is_start"],
                "is_end": period["is_end"],
                "oos_start": period["oos_start"],
                "oos_end": period["oos_end"],
                "is_results": is_results,
                "oos_results": oos_results,
                "is_count": len(is_results) if is_results is not None else 0,
                "oos_count": len(oos_results) if oos_results is not None else 0,
                "is_time": is_time if run_is else 0,
                "oos_time": oos_time if run_oos else 0,
            }

            all_results.append(period_results)

            period_total_time = (datetime.now() - period_start).total_seconds()
            total_strategies = period_results["is_count"] + period_results["oos_count"]
            speed = total_strategies / period_total_time if period_total_time > 0 else 0

            logger.info(
                f"  Period完成: 总耗时{period_total_time:.1f}秒, 速度{speed:.0f}策略/秒"
            )

        logger.info("\n[4/4] 保存结果")

        total_is = sum(r["is_count"] for r in all_results)
        total_oos = sum(r["oos_count"] for r in all_results)
        total_strategies = total_is + total_oos
        total_time = (datetime.now() - start_time).total_seconds()
        overall_speed = total_strategies / total_time if total_time > 0 else 0

        summary = {
            "timestamp": self.timestamp,
            "run_time": start_time.isoformat(),
            "total_periods": len(periods),
            "total_strategies": total_strategies,
            "total_is": total_is,
            "total_oos": total_oos,
            "total_time_seconds": total_time,
            "data_load_time_seconds": data_time,
            "backtest_time_seconds": total_time - data_time,
            "overall_speed_strategies_per_sec": overall_speed,
            "config": {
                "run_is": run_is,
                "run_oos": run_oos,
                "rebalance_freq": self.config.rebalance_freq_list,
                "top_n": self.config.top_n_list,
                "n_workers": self.config.n_workers,
            },
        }

        # 保存到时间戳目录
        summary_file = self.results_dir / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"  摘要已保存: {summary_file}")

        # 将所有Period结果合并为DataFrame并保存为Parquet
        # 按照新的存储格式政策：禁止使用Pickle
        results_file = self.results_dir / "results.parquet"
        results_dfs = []
        for r in all_results:
            # 合并IS和OOS结果
            period_data = {
                "period_id": r["period_id"],
                "is_start": r["is_start"],
                "is_end": r["is_end"],
                "oos_start": r["oos_start"],
                "oos_end": r["oos_end"],
            }

            if r["is_results"] is not None:
                is_df = r["is_results"].copy()
                is_df["period_id"] = r["period_id"]
                is_df["phase"] = "IS"
                for k, v in period_data.items():
                    is_df[k] = v
                results_dfs.append(is_df)

            if r["oos_results"] is not None:
                oos_df = r["oos_results"].copy()
                oos_df["period_id"] = r["period_id"]
                oos_df["phase"] = "OOS"
                for k, v in period_data.items():
                    oos_df[k] = v
                results_dfs.append(oos_df)

        if results_dfs:
            combined_df = pd.concat(results_dfs, ignore_index=True)
            combined_df.to_parquet(results_file, compression="zstd", engine="pyarrow")
            logger.info(
                f"  详细结果已保存: {results_file} (Parquet格式, {len(combined_df)}条记录)"
            )
        else:
            logger.warning("  无结果可保存")

        # ========================================
        # 最终报告
        # ========================================
        logger.info("\n" + "=" * 60)
        logger.info("WFO生产环境回测完成")
        logger.info("=" * 60)
        logger.info(f"总Period数:     {len(periods)}")
        logger.info(f"总策略数:       {total_strategies:,}")
        logger.info(f"  - IS策略:     {total_is:,}")
        logger.info(f"  - OOS策略:    {total_oos:,}")
        logger.info(f"总耗时:         {total_time/60:.1f}分钟")
        logger.info(
            f"数据加载:       {data_time:.1f}秒 ({data_time/total_time*100:.1f}%)"
        )
        logger.info(f"回测计算:       {(total_time-data_time)/60:.1f}分钟")
        logger.info(f"整体速度:       {overall_speed:.0f} 策略/秒")
        logger.info("=" * 60)

        return {"summary": summary, "results": all_results}


def main():
    """主函数"""
    # 先创建 runner 获取时间戳目录
    config_path = Path(__file__).parent / "simple_config.yaml"
    runner = OptimizedProductionRunner(str(config_path))

    # 在时间戳目录下创建日志
    log_file = runner.log_dir / "wfo.log"

    # 强制重新配置root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []  # 清空已有handlers
    root_logger.addHandler(logging.FileHandler(log_file, encoding="utf-8"))
    root_logger.addHandler(logging.StreamHandler(sys.stdout))

    # 设置格式
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)

    # 现在logger应该工作了
    logger.info(f"🚀 WFO生产环境回测启动")
    logger.info(f"📁 结果目录: {runner.results_dir}")
    logger.info(f"📝 日志文件: {log_file}")

    results = runner.run_production()

    logger.info(f"\n✅ 完成 | 所有结果保存在: {runner.results_dir}")


if __name__ == "__main__":
    main()
