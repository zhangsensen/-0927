#!/usr/bin/env python3
"""性能基准 - 耗时与内存监控

核心功能：
1. 记录面板计算耗时
2. 记录筛选耗时
3. 记录回测耗时
4. 记录内存峰值
5. 生成基准表

Linus式原则：可观测、可预测、可优化
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """性能基准监控器"""

    def __init__(self, output_dir="factor_output/etf_rotation_production"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.benchmark_file = self.output_dir / "benchmark.csv"
        self.process = psutil.Process()

        # 初始化基准表
        if not self.benchmark_file.exists():
            with open(self.benchmark_file, "w") as f:
                f.write(
                    "timestamp,stage,factors,etfs,samples,duration_sec,memory_mb,memory_peak_mb\n"
                )

    def record_benchmark(
        self,
        stage,
        factors=0,
        etfs=0,
        samples=0,
        duration=0,
        memory_mb=0,
        memory_peak_mb=0,
    ):
        """记录性能基准"""
        timestamp = datetime.now().isoformat()

        with open(self.benchmark_file, "a") as f:
            f.write(
                f"{timestamp},{stage},{factors},{etfs},{samples},{duration:.2f},{memory_mb:.2f},{memory_peak_mb:.2f}\n"
            )

        logger.info(f"✅ 性能基准已记录: {stage}")
        logger.info(f"   耗时: {duration:.2f}秒")
        logger.info(f"   内存: {memory_mb:.2f}MB (峰值: {memory_peak_mb:.2f}MB)")

    def benchmark_panel_production(self):
        """基准测试：面板生产"""
        logger.info("=" * 80)
        logger.info("性能基准 - 面板生产")
        logger.info("=" * 80)

        import subprocess

        # 记录初始内存
        mem_before = self.process.memory_info().rss / 1024 / 1024

        # 运行面板生产（使用集成测试的小数据集）
        logger.info("\n运行面板生产（金丝雀数据）...")
        start_time = time.time()

        result = subprocess.run(
            [
                "python3",
                "scripts/produce_full_etf_panel.py",
                "--output-dir",
                "factor_output/benchmark_test",
            ],
            capture_output=True,
            text=True,
        )

        duration = time.time() - start_time

        # 记录峰值内存
        mem_after = self.process.memory_info().rss / 1024 / 1024
        mem_peak = max(mem_before, mem_after)

        if result.returncode == 0:
            # 读取生成的面板
            import pandas as pd

            panel_files = list(
                Path("factor_output/benchmark_test").glob("panel_*.parquet")
            )
            if panel_files:
                panel = pd.read_parquet(panel_files[0])
                factors = len(panel.columns)
                etfs = panel.index.get_level_values("symbol").nunique()
                samples = len(panel)

                self.record_benchmark(
                    "panel_production",
                    factors=factors,
                    etfs=etfs,
                    samples=samples,
                    duration=duration,
                    memory_mb=mem_after,
                    memory_peak_mb=mem_peak,
                )
            else:
                logger.warning("⚠️  未找到生成的面板文件")
        else:
            logger.error("❌ 面板生产失败")

    def load_and_display_benchmarks(self):
        """加载并显示所有基准"""
        logger.info("=" * 80)
        logger.info("性能基准历史")
        logger.info("=" * 80)

        if not self.benchmark_file.exists():
            logger.info("\n暂无基准数据")
            return

        import pandas as pd

        df = pd.read_csv(self.benchmark_file)

        if len(df) == 0:
            logger.info("\n暂无基准数据")
            return

        logger.info(f"\n共{len(df)}条基准记录\n")

        # 按阶段分组统计
        for stage in df["stage"].unique():
            stage_df = df[df["stage"] == stage]

            logger.info(f"{stage}:")
            logger.info(f"  记录数: {len(stage_df)}")
            logger.info(f"  平均耗时: {stage_df['duration_sec'].mean():.2f}秒")
            logger.info(f"  平均内存: {stage_df['memory_mb'].mean():.2f}MB")
            logger.info(f"  峰值内存: {stage_df['memory_peak_mb'].max():.2f}MB")

            if "factors" in stage_df.columns and stage_df["factors"].sum() > 0:
                logger.info(f"  平均因子数: {stage_df['factors'].mean():.0f}")
            if "etfs" in stage_df.columns and stage_df["etfs"].sum() > 0:
                logger.info(f"  平均ETF数: {stage_df['etfs'].mean():.0f}")
            if "samples" in stage_df.columns and stage_df["samples"].sum() > 0:
                logger.info(f"  平均样本数: {stage_df['samples'].mean():.0f}")

            logger.info("")

        # 显示最近5条记录
        logger.info("最近5条记录:")
        logger.info(df.tail(5).to_string(index=False))


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="性能基准")
    parser.add_argument(
        "command",
        choices=["run", "show"],
        help="命令: run(运行基准测试), show(显示历史)",
    )

    args = parser.parse_args()

    benchmark = PerformanceBenchmark()

    try:
        if args.command == "run":
            benchmark.benchmark_panel_production()
            logger.info("\n✅ 基准测试完成")

        elif args.command == "show":
            benchmark.load_and_display_benchmarks()

        sys.exit(0)

    except Exception as e:
        logger.error(f"❌ 失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
