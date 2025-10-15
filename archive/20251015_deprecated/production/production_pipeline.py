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
            ["python3", "scripts/pool_management.py"], "分池面板生产"
        )

    def run_aggregate_metrics(self) -> bool:
        """运行指标汇总"""
        return self.run_command(
            [
                "python3",
                "scripts/aggregate_pool_metrics.py",
                "--base-dir",
                str(self.base_dir),
            ],
            "分池指标汇总",
        )

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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.snapshot_mgr.create_snapshot(
            source_dir=str(self.base_dir), tag=f"production_{timestamp}"
        )

    def run_full_pipeline(self) -> bool:
        """运行完整流水线"""
        logger.info("=" * 80)
        logger.info("生产流水线启动")
        logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)

        start_time = datetime.now()

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

    parser = argparse.ArgumentParser(description="生产流水线主调度")
    parser.add_argument(
        "--base-dir",
        default="factor_output/etf_rotation_production",
        help="基础输出目录",
    )
    parser.add_argument("--skip-snapshot", action="store_true", help="跳过快照创建")
    args = parser.parse_args()

    pipeline = ProductionPipeline(base_dir=args.base_dir)

    if args.skip_snapshot:
        pipeline.snapshot_mgr = None

    success = pipeline.run_full_pipeline()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
