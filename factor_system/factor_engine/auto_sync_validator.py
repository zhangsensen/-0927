#!/usr/bin/env python3
"""
自动同步验证器
确保FactorEngine与factor_generation保持自动同步
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Set, Optional
import logging

from .factor_consistency_guard import FactorConsistencyGuard

logger = logging.getLogger(__name__)


class AutoSyncValidator:
    """自动同步验证器"""

    def __init__(self, root_dir: Path = None):
        self.root_dir = root_dir or Path(__file__).parent.parent.parent
        self.guard = FactorConsistencyGuard(root_dir)
        self.sync_log = self.root_dir / ".factor_sync_log.json"

    def validate_and_sync(self) -> bool:
        """验证并自动同步"""
        logger.info("🔄 开始自动同步验证...")

        # 1. 检查是否有基准快照
        if not self.guard.snapshot_file.exists():
            logger.warning("⚠️  未找到基准快照，自动创建...")
            if not self.guard.create_baseline_snapshot():
                logger.error("❌ 创建基准快照失败")
                return False

        # 2. 验证一致性
        is_consistent = self.guard.validate_consistency()

        if is_consistent:
            logger.info("✅ FactorEngine与factor_generation保持一致")
            self._log_sync_result("success", "一致性验证通过")
            return True

        # 3. 如果不一致，生成修复建议
        logger.error("❌ 发现不一致，生成修复方案...")
        repair_plan = self._generate_repair_plan()

        # 4. 记录同步日志
        self._log_sync_result("error", f"发现不一致: {repair_plan}")

        # 5. 输出修复建议
        self._print_repair_suggestions(repair_plan)

        return False

    def _generate_repair_plan(self) -> Dict:
        """生成修复方案"""
        # 获取当前状态
        gen_factors = self.guard.scan_factor_generation_factors()
        engine_factors = self.guard.scan_factor_engine_factors()

        gen_factor_names = set(gen_factors.keys())
        engine_factor_names = set(engine_factors.keys())

        missing_factors = gen_factor_names - engine_factor_names
        extra_factors = engine_factor_names - gen_factor_names

        return {
            "missing_factors": sorted(list(missing_factors)),
            "extra_factors": sorted(list(extra_factors)),
            "recommended_actions": self._generate_actions(missing_factors, extra_factors)
        }

    def _generate_actions(self, missing: Set[str], extra: Set[str]) -> List[str]:
        """生成修复动作"""
        actions = []

        if missing:
            actions.append(f"🔧 在FactorEngine中实现缺失的因子: {', '.join(missing)}")

        if extra:
            actions.append(f"🗑️  从FactorEngine中删除多余的因子: {', '.join(extra)}")
            actions.append("⚠️  注意：删除前请确认这些因子不会被其他代码引用")

        actions.append("📸 运行: python factor_consistency_guard.py create-baseline")
        actions.append("✅ 运行: python factor_consistency_guard.py validate")

        return actions

    def _print_repair_suggestions(self, repair_plan: Dict):
        """打印修复建议"""
        print("\n" + "=" * 60)
        print("🔧 FACTOR ENGINE 修复建议")
        print("=" * 60)

        if repair_plan["missing_factors"]:
            print(f"\n❌ 缺失因子 ({len(repair_plan['missing_factors'])}个):")
            for factor in repair_plan["missing_factors"]:
                print(f"   - {factor}")

        if repair_plan["extra_factors"]:
            print(f"\n⚠️  多余因子 ({len(repair_plan['extra_factors'])}个):")
            for factor in repair_plan["extra_factors"]:
                print(f"   - {factor}")

        print(f"\n🎯 推荐修复步骤:")
        for i, action in enumerate(repair_plan["recommended_actions"], 1):
            print(f"   {i}. {action}")

        print("\n" + "=" * 60)

    def _log_sync_result(self, status: str, message: str):
        """记录同步日志"""
        log_entry = {
            "timestamp": time.time(),
            "status": status,
            "message": message
        }

        logs = []
        if self.sync_log.exists():
            try:
                with open(self.sync_log, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            except Exception:
                logs = []

        logs.append(log_entry)

        # 保留最近100条记录
        if len(logs) > 100:
            logs = logs[-100:]

        try:
            with open(self.sync_log, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            logger.error(f"❌ 写入同步日志失败: {e}")

    def get_sync_history(self) -> List[Dict]:
        """获取同步历史"""
        if not self.sync_log.exists():
            return []

        try:
            with open(self.sync_log, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []

    def create_sync_report(self) -> Dict:
        """创建同步报告"""
        history = self.get_sync_history()
        recent_logs = history[-10:] if history else []

        # 统计成功率
        total = len(history)
        successful = len([log for log in history if log["status"] == "success"])
        success_rate = (successful / total * 100) if total > 0 else 0

        return {
            "total_syncs": total,
            "successful_syncs": successful,
            "success_rate": f"{success_rate:.1f}%",
            "recent_logs": recent_logs,
            "last_sync": history[-1] if history else None
        }


class FactorSyncMonitor:
    """因子同步监控器"""

    def __init__(self, root_dir: Path = None):
        self.root_dir = root_dir or Path(__file__).parent.parent.parent
        self.validator = AutoSyncValidator(root_dir)
        self.monitor_log = self.root_dir / ".factor_monitor.log"

    def start_monitoring(self, interval: int = 300) -> bool:
        """开始监控（后台进程）"""
        logger.info(f"🔍 开始监控FactorEngine一致性，间隔: {interval}秒")

        try:
            while True:
                # 验证一致性
                is_consistent = self.validator.validate_and_sync()

                if not is_consistent:
                    self._log_monitor_event("ALERT", "检测到FactorEngine不一致")

                # 等待下次检查
                time.sleep(interval)

        except KeyboardInterrupt:
            logger.info("⏹️  监控已停止")
            return True
        except Exception as e:
            logger.error(f"❌ 监控过程出错: {e}")
            return False

    def _log_monitor_event(self, level: str, message: str):
        """记录监控事件"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}\n"

        try:
            with open(self.monitor_log, 'a', encoding='utf-8') as f:
                f.write(log_entry)
        except Exception as e:
            logger.error(f"❌ 写入监控日志失败: {e}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="FactorEngine自动同步验证器")
    parser.add_argument("command", choices=["validate", "monitor", "report"], help="命令")
    parser.add_argument("--interval", type=int, default=300, help="监控间隔（秒）")

    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    validator = AutoSyncValidator()

    if args.command == "validate":
        print("🔍 开始验证FactorEngine一致性...")
        success = validator.validate_and_sync()
        sys.exit(0 if success else 1)

    elif args.command == "monitor":
        print("🔍 开始监控FactorEngine一致性...")
        monitor = FactorSyncMonitor()
        success = monitor.start_monitoring(args.interval)
        sys.exit(0 if success else 1)

    elif args.command == "report":
        print("📊 生成同步报告...")
        report = validator.create_sync_report()
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()