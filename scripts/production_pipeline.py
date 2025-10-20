#!/usr/bin/env python3
"""
生产流水线主调度
整合：分池生产 → 回测 → 容量 → CI → 指标汇总 → 通知
"""
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from notification_handler import NotificationHandler, SnapshotManager
from path_utils import get_paths

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class ProductionPipeline:
    """生产流水线"""

    def __init__(self, base_dir: str = "factor_output/etf_rotation_production"):
        self.base_dir = Path(base_dir)
        self.notifier = NotificationHandler()
        self.snapshot_mgr = SnapshotManager(max_snapshots=10)
        self.pools = ["A_SHARE", "QDII", "OTHER"]
        self.failed_tasks = []

    def run_command(self, cmd: list, task_name: str) -> bool:
        """运行命令并捕获错误"""
        logger.info(f"\n{'=' * 80}")
        logger.info(f"执行任务: {task_name}")
        logger.info(f"命令: {' '.join(cmd)}")
        logger.info(f"{'=' * 80}")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=3600,  # 1小时超时
            )

            logger.info(result.stdout)
            logger.info(f"✅ {task_name} 完成")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ {task_name} 失败")
            logger.error(f"返回码: {e.returncode}")
            logger.error(f"错误输出:\n{e.stderr}")

            self.failed_tasks.append(
                {
                    "task": task_name,
                    "error": e.stderr[:500],  # 截断
                    "returncode": e.returncode,
                }
            )

            # 发送失败通知
            self.notifier.notify_failure(task_name, e.stderr[:200])

            return False

        except subprocess.TimeoutExpired:
            logger.error(f"❌ {task_name} 超时")
            self.failed_tasks.append({"task": task_name, "error": "Timeout"})
            self.notifier.notify_failure(task_name, "任务超时（>1小时）")
            return False

    def run_pool_management(self) -> bool:
        """运行分池管理（生产面板）"""
        return self.run_command(
            ["python3", "etf_factor_engine_production/scripts/produce_full_etf_panel.py"],
            "分池面板生产"
        )

    def run_aggregate_metrics(self) -> bool:
        """运行指标汇总"""
        # 注意：aggregate_pool_metrics.py 已废弃，功能已集成到新架构
        logger.info("⚠️  指标汇总功能已集成到新架构，此步骤已跳过")
        return True

    def run_ci_checks(self) -> bool:
        """运行 CI 检查（所有池）"""
        all_passed = True

        for pool in self.pools:
            pool_dir = self.base_dir / f"panel_{pool}"
            if not pool_dir.exists():
                logger.warning(f"⚠️  {pool} 目录不存在，跳过 CI")
                continue

            passed = self.run_command(
                ["python3", "scripts/ci_checks.py", "--output-dir", str(pool_dir)],
                f"CI检查 ({pool})",
            )

            if not passed:
                all_passed = False

        return all_passed

    def create_snapshot(self):
        """创建快照"""
        if self.snapshot_mgr is None:
            logger.info("⏭️  跳过快照创建（--skip-snapshot）")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.snapshot_mgr.create_snapshot(
            source_dir=str(self.base_dir), tag=f"production_{timestamp}"
        )

    def _dry_run_validation(self) -> bool:
        """Dry-run 模式：仅校验现有文件存在性"""
        logger.info("\n🔍 Dry-run 模式：校验现有文件...")

        all_valid = True

        # 检查各池目录和关键文件
        for pool in self.pools:
            pool_dir = self.base_dir / f"panel_{pool}"
            logger.info(f"\n检查池: {pool}")

            # 检查池目录
            if not pool_dir.exists():
                logger.error(f"  ❌ 池目录不存在: {pool_dir}")
                all_valid = False
                continue
            else:
                logger.info(f"  ✅ 池目录存在: {pool_dir}")

            # 检查回测指标文件
            metrics_file = pool_dir / "backtest_metrics.json"
            if not metrics_file.exists():
                logger.warning(f"  ⚠️  回测指标文件不存在: {metrics_file}")
            else:
                logger.info(f"  ✅ 回测指标文件存在: {metrics_file}")

            # 检查面板文件（至少有一个）
            panel_files = list(pool_dir.glob("panel_FULL_*.parquet"))
            if not panel_files:
                logger.warning(f"  ⚠️  未找到面板文件: {pool_dir}/panel_FULL_*.parquet")
            else:
                logger.info(f"  ✅ 找到 {len(panel_files)} 个面板文件")

        # 检查汇总报告
        summary_file = self.base_dir / "pool_metrics_summary.csv"
        if not summary_file.exists():
            logger.warning(f"\n⚠️  汇总报告不存在: {summary_file}")
        else:
            logger.info(f"\n✅ 汇总报告存在: {summary_file}")

        logger.info("\n" + "=" * 80)
        if all_valid:
            logger.info("✅ Dry-run 校验通过")
        else:
            logger.error("❌ Dry-run 校验失败（部分池目录不存在）")
        logger.info("=" * 80)

        return all_valid

    def run_full_pipeline(self, dry_run: bool = False) -> bool:
        """运行完整流水线

        Args:
            dry_run: 仅校验现有文件，不触发重新计算
        """
        logger.info("=" * 80)
        logger.info(f"生产流水线启动 {'(DRY-RUN 模式)' if dry_run else ''}")
        logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)

        start_time = datetime.now()

        # Dry-run 模式：仅校验文件存在性
        if dry_run:
            return self._dry_run_validation()

        # Step 1: 分池生产
        if not self.run_pool_management():
            logger.error("❌ 分池生产失败，终止流水线")
            return False

        # Step 2: 指标汇总
        if not self.run_aggregate_metrics():
            logger.warning("⚠️  指标汇总失败，继续执行")

        # Step 3: CI 检查
        ci_passed = self.run_ci_checks()

        # Step 4: 创建快照
        self.create_snapshot()

        # 汇总结果
        duration = (datetime.now() - start_time).total_seconds()

        logger.info("\n" + "=" * 80)
        logger.info("流水线执行完成")
        logger.info("=" * 80)
        logger.info(f"耗时: {duration:.1f}秒")
        logger.info(f"失败任务数: {len(self.failed_tasks)}")

        if self.failed_tasks:
            logger.error("\n失败任务列表:")
            for task in self.failed_tasks:
                logger.error(f"  - {task['task']}: {task['error'][:100]}")

        # 发送成功通知
        if not self.failed_tasks and ci_passed:
            summary = f"**耗时**: {duration:.1f}秒\n**状态**: 全部通过"
            self.notifier.notify_success("生产流水线", summary)
            logger.info("\n✅ 流水线全部通过")
            return True
        else:
            logger.error("\n❌ 流水线存在失败任务或 CI 未通过")
            return False


def main():
    """主函数"""
    import argparse

    # 从配置文件读取默认值
    paths = get_paths()

    parser = argparse.ArgumentParser(description="生产流水线主调度")
    parser.add_argument(
        "--base-dir",
        default=str(paths["output_root"]),
        help="基础输出目录（默认从配置文件读取）",
    )
    parser.add_argument("--skip-snapshot", action="store_true", help="跳过快照创建")
    parser.add_argument(
        "--dry-run", action="store_true", help="仅校验现有文件，不触发重新计算"
    )
    args = parser.parse_args()

    pipeline = ProductionPipeline(base_dir=args.base_dir)

    if args.skip_snapshot:
        pipeline.snapshot_mgr = None

    success = pipeline.run_full_pipeline(dry_run=args.dry_run)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
