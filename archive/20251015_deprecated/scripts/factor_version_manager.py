#!/usr/bin/env python3
"""因子版本管理 - 可追溯性与回滚

核心功能：
1. 记录factor_id+params+engine_version+price_field
2. 生成factors_selected快照
3. 固化运行参数
4. 支持版本对比与回滚

Linus式原则：可追溯、可复现、可回滚
"""

import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class FactorVersionManager:
    """因子版本管理器"""

    def __init__(self, output_dir="factor_output/etf_rotation_production"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 版本历史目录
        self.versions_dir = self.output_dir / "versions"
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    def create_version_snapshot(self, panel_file, production_factors_file=None):
        """创建版本快照"""
        logger.info("=" * 80)
        logger.info("因子版本管理 - 创建快照")
        logger.info("=" * 80)

        panel_path = Path(panel_file)
        if not panel_path.exists():
            logger.error(f"❌ 面板文件不存在: {panel_file}")
            return False

        # 加载面板
        logger.info(f"\n加载面板: {panel_path.name}")
        panel = pd.read_parquet(panel_path)

        # 加载元数据
        meta_file = panel_path.parent / "panel_meta.json"
        if meta_file.exists():
            with open(meta_file) as f:
                panel_meta = json.load(f)
        else:
            panel_meta = {}

        # 加载生产因子清单
        if production_factors_file:
            prod_factors_path = Path(production_factors_file)
        else:
            prod_factors_path = self.output_dir / "production_factors.txt"

        if prod_factors_path.exists():
            with open(prod_factors_path) as f:
                production_factors = [line.strip() for line in f if line.strip()]
        else:
            production_factors = []

        # 计算因子哈希
        factors_hash = self._compute_factors_hash(panel.columns.tolist())
        production_hash = self._compute_factors_hash(production_factors)

        # 生成版本信息
        version_info = {
            "version_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "timestamp": datetime.now().isoformat(),
            "panel_file": str(panel_path),
            "engine_version": panel_meta.get("engine_version", "unknown"),
            "price_field": panel_meta.get("price_field", "close"),
            "price_field_priority": panel_meta.get(
                "price_field_priority", ["adj_close", "close"]
            ),
            "data_range": panel_meta.get("data_range", {}),
            "run_params": panel_meta.get("run_params", {}),
            "pools": panel_meta.get("pools", {}),
            "factors": {
                "total": len(panel.columns),
                "production": len(production_factors),
                "all_factors_hash": factors_hash,
                "production_factors_hash": production_hash,
                "all_factors": panel.columns.tolist(),
                "production_factors": production_factors,
            },
            "metrics": {
                "samples": len(panel),
                "etfs": panel.index.get_level_values("symbol").nunique(),
                "coverage": (panel.notna().sum().sum() / panel.size),
            },
        }

        # 保存版本快照
        version_file = self.versions_dir / f"version_{version_info['version_id']}.json"
        with open(version_file, "w") as f:
            json.dump(version_info, f, indent=2, ensure_ascii=False)

        logger.info(f"\n✅ 版本快照已创建:")
        logger.info(f"   版本ID: {version_info['version_id']}")
        logger.info(f"   引擎版本: {version_info['engine_version']}")
        logger.info(f"   价格字段: {version_info['price_field']}")
        logger.info(f"   因子总数: {version_info['factors']['total']}")
        logger.info(f"   生产因子数: {version_info['factors']['production']}")
        logger.info(f"   因子哈希: {factors_hash[:16]}...")
        logger.info(f"   生产哈希: {production_hash[:16]}...")
        logger.info(f"   快照文件: {version_file}")

        # 创建最新版本链接
        latest_file = self.versions_dir / "latest_version.json"
        with open(latest_file, "w") as f:
            json.dump(version_info, f, indent=2, ensure_ascii=False)
        logger.info(f"   最新版本: {latest_file}")

        # 保存生产因子快照（YAML格式）
        if production_factors:
            prod_snapshot_file = (
                self.versions_dir
                / f"production_factors_{version_info['version_id']}.yaml"
            )
            with open(prod_snapshot_file, "w") as f:
                f.write("# 生产因子快照\n")
                f.write(f"# 版本ID: {version_info['version_id']}\n")
                f.write(f"# 生成时间: {version_info['timestamp']}\n")
                f.write(f"# 引擎版本: {version_info['engine_version']}\n")
                f.write(f"# 价格字段: {version_info['price_field']}\n\n")
                f.write("production_factors:\n")
                for factor in production_factors:
                    f.write(f"  - {factor}\n")
            logger.info(f"   生产快照: {prod_snapshot_file}")

        return True

    def _compute_factors_hash(self, factors):
        """计算因子列表哈希"""
        factors_str = ",".join(sorted(factors))
        return hashlib.sha256(factors_str.encode()).hexdigest()

    def list_versions(self):
        """列出所有版本"""
        logger.info("=" * 80)
        logger.info("因子版本历史")
        logger.info("=" * 80)

        version_files = sorted(self.versions_dir.glob("version_*.json"))

        if len(version_files) == 0:
            logger.info("\n暂无版本历史")
            return

        logger.info(f"\n共{len(version_files)}个版本:\n")

        for version_file in version_files:
            with open(version_file) as f:
                version_info = json.load(f)

            logger.info(f"版本ID: {version_info['version_id']}")
            logger.info(f"  时间: {version_info['timestamp']}")
            logger.info(f"  引擎: {version_info['engine_version']}")
            logger.info(f"  价格字段: {version_info['price_field']}")
            logger.info(f"  因子数: {version_info['factors']['total']}")
            logger.info(f"  生产因子: {version_info['factors']['production']}")
            logger.info(
                f"  哈希: {version_info['factors']['all_factors_hash'][:16]}..."
            )
            logger.info("")

    def compare_versions(self, version1_id, version2_id):
        """对比两个版本"""
        logger.info("=" * 80)
        logger.info(f"版本对比: {version1_id} vs {version2_id}")
        logger.info("=" * 80)

        v1_file = self.versions_dir / f"version_{version1_id}.json"
        v2_file = self.versions_dir / f"version_{version2_id}.json"

        if not v1_file.exists():
            logger.error(f"❌ 版本1不存在: {version1_id}")
            return False

        if not v2_file.exists():
            logger.error(f"❌ 版本2不存在: {version2_id}")
            return False

        with open(v1_file) as f:
            v1 = json.load(f)

        with open(v2_file) as f:
            v2 = json.load(f)

        # 对比因子
        factors1 = set(v1["factors"]["all_factors"])
        factors2 = set(v2["factors"]["all_factors"])

        added = factors2 - factors1
        removed = factors1 - factors2
        common = factors1 & factors2

        logger.info(f"\n因子变化:")
        logger.info(f"  版本1因子数: {len(factors1)}")
        logger.info(f"  版本2因子数: {len(factors2)}")
        logger.info(f"  共同因子: {len(common)}")
        logger.info(f"  新增因子: {len(added)}")
        logger.info(f"  删除因子: {len(removed)}")

        if added:
            logger.info(f"\n新增因子:")
            for factor in sorted(added)[:10]:
                logger.info(f"  + {factor}")
            if len(added) > 10:
                logger.info(f"  ... 还有{len(added)-10}个")

        if removed:
            logger.info(f"\n删除因子:")
            for factor in sorted(removed)[:10]:
                logger.info(f"  - {factor}")
            if len(removed) > 10:
                logger.info(f"  ... 还有{len(removed)-10}个")

        # 对比生产因子
        prod1 = set(v1["factors"]["production_factors"])
        prod2 = set(v2["factors"]["production_factors"])

        prod_added = prod2 - prod1
        prod_removed = prod1 - prod2

        logger.info(f"\n生产因子变化:")
        logger.info(f"  版本1: {len(prod1)}个")
        logger.info(f"  版本2: {len(prod2)}个")
        logger.info(f"  新增: {len(prod_added)}个")
        logger.info(f"  删除: {len(prod_removed)}个")

        if prod_added:
            logger.info(f"\n新增生产因子:")
            for factor in sorted(prod_added):
                logger.info(f"  + {factor}")

        if prod_removed:
            logger.info(f"\n删除生产因子:")
            for factor in sorted(prod_removed):
                logger.info(f"  - {factor}")

        return True


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="因子版本管理")
    parser.add_argument(
        "command",
        choices=["snapshot", "list", "compare"],
        help="命令: snapshot(创建快照), list(列出版本), compare(对比版本)",
    )
    parser.add_argument("--panel-file", help="面板文件路径")
    parser.add_argument("--production-factors", help="生产因子文件路径")
    parser.add_argument("--version1", help="版本1 ID（用于对比）")
    parser.add_argument("--version2", help="版本2 ID（用于对比）")

    args = parser.parse_args()

    manager = FactorVersionManager()

    try:
        if args.command == "snapshot":
            if not args.panel_file:
                # 查找最新面板
                panel_dir = Path("factor_output/etf_rotation_production")
                panel_files = list(panel_dir.glob("panel_*.parquet"))
                if len(panel_files) == 0:
                    logger.error("❌ 未找到面板文件")
                    sys.exit(1)
                args.panel_file = sorted(panel_files)[-1]

            success = manager.create_version_snapshot(
                args.panel_file, args.production_factors
            )
            sys.exit(0 if success else 1)

        elif args.command == "list":
            manager.list_versions()
            sys.exit(0)

        elif args.command == "compare":
            if not args.version1 or not args.version2:
                logger.error("❌ 需要指定--version1和--version2")
                sys.exit(1)

            success = manager.compare_versions(args.version1, args.version2)
            sys.exit(0 if success else 1)

    except Exception as e:
        logger.error(f"❌ 失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
