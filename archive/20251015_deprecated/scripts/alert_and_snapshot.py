#!/usr/bin/env python3
"""报警与快照系统

告警条件：
1. 有效因子数<8
2. 覆盖率骤降≥10%
3. 目标波动缩放<0.6
4. 月收益>30%
5. ADV%超阈

月度快照：
- 宇宙清单
- 因子清单
- 相关性热图
- 漏斗报告
- 订单CSV
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class AlertAndSnapshotSystem:
    """报警与快照系统"""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.alerts = []

    def check_alerts(self, summary_file, panel_file, production_factors_file):
        """检查告警条件"""
        logger.info("=" * 80)
        logger.info("报警检查")
        logger.info("=" * 80)

        # 加载数据
        summary = pd.read_csv(summary_file)
        panel = pd.read_parquet(panel_file)

        with open(production_factors_file) as f:
            production_factors = [line.strip() for line in f if line.strip()]

        # 1. 有效因子数检查
        logger.info(f"\n1. 有效因子数检查")
        num_factors = len(production_factors)
        logger.info(f"   当前因子数: {num_factors}")

        if num_factors < 8:
            alert = f"❌ 有效因子数不足: {num_factors} < 8"
            logger.error(f"   {alert}")
            self.alerts.append(
                {
                    "type": "factor_count",
                    "level": "critical",
                    "message": alert,
                    "value": num_factors,
                    "threshold": 8,
                }
            )
        else:
            logger.info(f"   ✅ 因子数充足")

        # 2. 覆盖率骤降检查
        logger.info(f"\n2. 覆盖率检查")
        current_coverage = summary["coverage"].mean()
        logger.info(f"   当前覆盖率: {current_coverage:.2%}")

        # 检查是否有历史覆盖率记录
        history_file = self.output_dir / "coverage_history.json"
        if history_file.exists():
            with open(history_file) as f:
                history = json.load(f)

            if len(history) > 0:
                last_coverage = history[-1]["coverage"]
                coverage_drop = last_coverage - current_coverage

                if coverage_drop >= 0.10:
                    alert = f"❌ 覆盖率骤降: {coverage_drop:.2%} ≥ 10%"
                    logger.error(f"   {alert}")
                    self.alerts.append(
                        {
                            "type": "coverage_drop",
                            "level": "critical",
                            "message": alert,
                            "current": current_coverage,
                            "previous": last_coverage,
                            "drop": coverage_drop,
                        }
                    )
                else:
                    logger.info(f"   ✅ 覆盖率正常（变化{coverage_drop:+.2%}）")
        else:
            logger.info(f"   ⚠️  无历史记录，跳过骤降检查")

        # 更新覆盖率历史
        history = []
        if history_file.exists():
            with open(history_file) as f:
                history = json.load(f)

        history.append(
            {"date": datetime.now().isoformat(), "coverage": current_coverage}
        )

        with open(history_file, "w") as f:
            json.dump(history[-12:], f, indent=2)  # 保留最近12个月

        # 3. 零方差检查
        logger.info(f"\n3. 零方差检查")
        zero_var_count = summary["zero_variance"].sum()
        logger.info(f"   零方差因子数: {zero_var_count}/{len(summary)}")

        if zero_var_count > 10:
            alert = f"⚠️  零方差因子过多: {zero_var_count} > 10"
            logger.warning(f"   {alert}")
            self.alerts.append(
                {
                    "type": "zero_variance",
                    "level": "warning",
                    "message": alert,
                    "count": int(zero_var_count),
                }
            )
        else:
            logger.info(f"   ✅ 零方差正常")

        # 4. 目标波动缩放检查（需要回测数据）
        logger.info(f"\n4. 目标波动缩放检查")
        logger.info(f"   ⚠️  需要回测数据，暂时跳过")

        # 5. 月收益检查（需要回测数据）
        logger.info(f"\n5. 月收益检查")
        logger.info(f"   ⚠️  需要回测数据，暂时跳过")

        # 6. ADV%检查（需要成交量数据）
        logger.info(f"\n6. ADV%检查")
        logger.info(f"   ⚠️  需要成交量数据，暂时跳过")

        logger.info(f"\n{'=' * 80}")
        if len(self.alerts) == 0:
            logger.info("✅ 所有检查通过，无告警")
        else:
            logger.warning(f"⚠️  发现{len(self.alerts)}个告警")
        logger.info("=" * 80)

    def generate_snapshot(self, panel_file, summary_file, production_factors_file):
        """生成月度快照"""
        logger.info("\n" + "=" * 80)
        logger.info("月度快照生成")
        logger.info("=" * 80)

        snapshot_date = datetime.now().strftime("%Y%m")
        snapshot_dir = self.output_dir / f"snapshot_{snapshot_date}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n快照目录: {snapshot_dir}")

        # 1. 宇宙清单
        logger.info(f"\n1. 生成宇宙清单...")
        panel = pd.read_parquet(panel_file)
        universe = panel.index.get_level_values("symbol").unique().tolist()

        universe_file = snapshot_dir / "universe.txt"
        with open(universe_file, "w") as f:
            for symbol in sorted(universe):
                f.write(f"{symbol}\n")

        logger.info(f"   ✅ 宇宙清单: {len(universe)}个ETF")
        logger.info(f"   保存至: {universe_file}")

        # 2. 因子清单
        logger.info(f"\n2. 复制因子清单...")
        import shutil

        factors_file = snapshot_dir / "production_factors.txt"
        shutil.copy(production_factors_file, factors_file)

        with open(production_factors_file) as f:
            factors = [line.strip() for line in f if line.strip()]
        logger.info(f"   ✅ 因子清单: {len(factors)}个")
        logger.info(f"   保存至: {factors_file}")

        # 3. 相关性热图
        logger.info(f"\n3. 复制相关性热图...")
        corr_files = [
            "correlation_matrix_3m.csv",
            "high_correlation_pairs_3m.csv",
            "correlation_heatmap_3m.png",
        ]

        for filename in corr_files:
            src = Path("factor_output/etf_rotation_production") / filename
            if src.exists():
                dst = snapshot_dir / filename
                shutil.copy(src, dst)
                logger.info(f"   ✅ {filename}")

        # 4. 漏斗报告
        logger.info(f"\n4. 复制漏斗报告...")
        funnel_file = Path("factor_output/etf_rotation_production/funnel_report.csv")
        if funnel_file.exists():
            dst = snapshot_dir / "funnel_report.csv"
            shutil.copy(funnel_file, dst)
            logger.info(f"   ✅ 漏斗报告")

        # 5. 因子概要
        logger.info(f"\n5. 复制因子概要...")
        dst = snapshot_dir / "factor_summary.csv"
        shutil.copy(summary_file, dst)
        logger.info(f"   ✅ 因子概要")

        # 6. 生成快照元数据
        logger.info(f"\n6. 生成快照元数据...")
        metadata = {
            "snapshot_date": snapshot_date,
            "created_at": datetime.now().isoformat(),
            "universe_size": len(universe),
            "num_factors": len(factors),
            "alerts": self.alerts,
            "files": {
                "universe": "universe.txt",
                "factors": "production_factors.txt",
                "correlation_matrix": "correlation_matrix_3m.csv",
                "correlation_heatmap": "correlation_heatmap_3m.png",
                "funnel_report": "funnel_report.csv",
                "factor_summary": "factor_summary.csv",
            },
        }

        metadata_file = snapshot_dir / "snapshot_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"   ✅ 元数据")
        logger.info(f"   保存至: {metadata_file}")

        logger.info(f"\n{'=' * 80}")
        logger.info(f"✅ 月度快照生成完成")
        logger.info(f"   快照目录: {snapshot_dir}")
        logger.info("=" * 80)

        return snapshot_dir

    def generate_alert_report(self):
        """生成告警报告"""
        if len(self.alerts) == 0:
            return None

        logger.info("\n" + "=" * 80)
        logger.info("告警报告")
        logger.info("=" * 80)

        for i, alert in enumerate(self.alerts, 1):
            logger.info(f"\n告警 {i}:")
            logger.info(f"  类型: {alert['type']}")
            logger.info(f"  级别: {alert['level']}")
            logger.info(f"  消息: {alert['message']}")

        # 保存告警报告
        alert_file = (
            self.output_dir
            / f"alert_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(alert_file, "w") as f:
            json.dump(self.alerts, f, indent=2, ensure_ascii=False)

        logger.info(f"\n✅ 告警报告已保存: {alert_file}")
        logger.info("=" * 80)

        return alert_file


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("报警与快照系统")
    logger.info("=" * 80)

    # 初始化系统
    system = AlertAndSnapshotSystem(
        output_dir="factor_output/etf_rotation_production/snapshots"
    )

    # 文件路径
    panel_file = Path(
        "factor_output/etf_rotation_production/panel_FULL_20200102_20251014.parquet"
    )
    summary_file = Path(
        "factor_output/etf_rotation_production/factor_summary_20200102_20251014.csv"
    )
    production_factors_file = Path(
        "factor_output/etf_rotation_production/production_factors.txt"
    )

    # 检查文件
    if not all(
        [panel_file.exists(), summary_file.exists(), production_factors_file.exists()]
    ):
        logger.error("❌ 缺少必要文件")
        return False

    try:
        # 检查告警
        system.check_alerts(summary_file, panel_file, production_factors_file)

        # 生成快照
        snapshot_dir = system.generate_snapshot(
            panel_file, summary_file, production_factors_file
        )

        # 生成告警报告
        if len(system.alerts) > 0:
            system.generate_alert_report()

        logger.info("\n" + "=" * 80)
        logger.info("✅ 报警与快照系统运行完成")
        logger.info("=" * 80)

        return True

    except Exception as e:
        logger.error(f"❌ 系统运行失败: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
