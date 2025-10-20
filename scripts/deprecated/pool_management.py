#!/usr/bin/env python3
"""ETF分池管理 - A股与QDII分池生产与回测

核心功能：
1. 加载分池配置
2. 分池生产因子面板
3. 分池回测
4. 顶层权重整合

Linus式原则：
- 简洁：单一职责，配置驱动
- 实用：解决时区/节假日错窗问题
- 可证：所有操作可追溯
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class PoolManager:
    """ETF分池管理器"""

    def __init__(self, config_file="configs/etf_pools.yaml"):
        self.config_file = Path(config_file)
        self.config = self._load_config()

    def _load_config(self):
        """加载配置"""
        if not self.config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_file}")

        with open(self.config_file) as f:
            config = yaml.safe_load(f)

        logger.info(f"✅ 加载配置: {self.config_file}")
        return config

    def list_pools(self):
        """列出所有池"""
        logger.info("=" * 80)
        logger.info("ETF分池列表")
        logger.info("=" * 80)

        for pool_name, pool_config in self.config["pools"].items():
            logger.info(f"\n{pool_name}:")
            logger.info(f"  名称: {pool_config['name']}")
            logger.info(f"  描述: {pool_config['description']}")
            logger.info(f"  日历: {pool_config['calendar']}")
            logger.info(f"  时区: {pool_config['timezone']}")
            logger.info(f"  ETF数: {len(pool_config['symbols'])}")
            logger.info(f"  ETF列表: {', '.join(pool_config['symbols'][:5])}...")

    def get_pool_symbols(self, pool_name):
        """获取指定池的ETF列表"""
        if pool_name not in self.config["pools"]:
            raise ValueError(f"池不存在: {pool_name}")

        return self.config["pools"][pool_name]["symbols"]

    def produce_pool_panel(
        self,
        pool_name,
        output_dir=None,
        execute=True,
        run_backtest=True,
        run_capacity=True,
    ):
        """生产指定池的因子面板 + 回测 + 容量

        Args:
            pool_name: 池名称
            output_dir: 输出目录
            execute: 是否实际执行（True=执行，False=仅显示命令）
            run_backtest: 是否运行回测
            run_capacity: 是否运行容量检查
        """
        logger.info("=" * 80)
        logger.info(f"生产{pool_name}因子面板")
        logger.info("=" * 80)

        # 获取池配置
        pool_config = self.config["pools"][pool_name]
        symbols = pool_config["symbols"]

        # 确定输出目录（统一到 etf_rotation_production 下的子目录）
        if output_dir is None:
            output_dir = Path(
                f"factor_output/etf_rotation_production/panel_{pool_name}"
            )
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n池配置:")
        logger.info(f"  名称: {pool_config['name']}")
        logger.info(f"  ETF数: {len(symbols)}")
        logger.info(f"  输出目录: {output_dir}")

        if execute:
            logger.info(f"\n🚀 开始生产面板...")

            # 写入symbols白名单
            symbols_file = output_dir / f"{pool_name}_symbols.txt"
            with open(symbols_file, "w") as f:
                for symbol in symbols:
                    f.write(f"{symbol}\n")
            logger.info(f"  ✅ symbols白名单: {symbols_file}")

            # 实际执行面板生产
            import subprocess

            cmd = [
                "python3",
                "scripts/produce_full_etf_panel.py",
                "--output-dir",
                str(output_dir),
                "--symbols-file",
                str(symbols_file),
                "--pool-name",
                pool_name,
            ]

            logger.info(f"  执行命令: {' '.join(cmd)}")

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")

                if result.returncode == 0:
                    logger.info(f"  ✅ 面板生产成功")

                    # 检查输出文件
                    panel_files = list(output_dir.glob("panel_*.parquet"))
                    if panel_files:
                        panel_file = panel_files[0]
                        logger.info(f"  ✅ 生成面板文件: {panel_file.name}")

                        # 运行回测
                        if run_backtest:
                            self._run_backtest(pool_name, panel_file, output_dir)

                        # 运行容量检查
                        if run_capacity:
                            self._run_capacity_check(pool_name, output_dir)
                    else:
                        logger.warning(f"  ⚠️  未找到面板文件")
                else:
                    logger.error(f"  ❌ 面板生产失败")
                    logger.error(f"  错误信息: {result.stderr}")

            except Exception as e:
                logger.error(f"  ❌ 执行失败: {e}")
        else:
            # 仅显示命令
            logger.info(f"\n⚠️  需要调用produce_full_etf_panel.py生产面板")
            logger.info(f"   命令示例:")
            logger.info(f"   python3 scripts/produce_full_etf_panel.py \\")
            logger.info(f"       --output-dir {output_dir} \\")
            logger.info(f"       --symbols-file {pool_name}_symbols.txt \\")
            logger.info(f"       --pool-name {pool_name}")

        return output_dir

    def _run_backtest(self, pool_name, panel_file, output_dir):
        """运行回测"""
        logger.info(f"\n🚀 运行{pool_name}回测...")

        import subprocess

        # 检查是否存在生产因子列表
        production_factors_file = output_dir / "production_factors.txt"
        if not production_factors_file.exists():
            # 从 factor_summary 提取覆盖率>50%的因子
            summary_files = list(output_dir.glob("factor_summary_*.csv"))
            if summary_files:
                summary = pd.read_csv(summary_files[0])
                production_factors = summary[summary["coverage"] > 0.5][
                    "factor_id"
                ].tolist()
                with open(production_factors_file, "w") as f:
                    for factor in production_factors:
                        f.write(f"{factor}\n")
                logger.info(f"  ✅ 生成生产因子列表: {len(production_factors)}个")

        cmd = [
            "python3",
            "scripts/etf_rotation_backtest.py",
            "--panel-file",
            str(panel_file),
            "--price-dir",
            "raw/ETF/daily",
            "--production-factors",
            str(production_factors_file),
            "--output-dir",
            str(output_dir),
        ]

        logger.info(f"  执行命令: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")

            if result.returncode == 0:
                logger.info(f"  ✅ 回测完成")
            else:
                logger.error(f"  ❌ 回测失败: {result.stderr}")
        except Exception as e:
            logger.error(f"  ❌ 回测执行失败: {e}")

    def _run_capacity_check(self, pool_name, output_dir):
        """运行容量检查"""
        logger.info(f"\n🚀 运行{pool_name}容量检查...")

        import subprocess

        # 检查是否存在回测结果（统一为 backtest_metrics.json）
        metrics_file = output_dir / "backtest_metrics.json"
        if not metrics_file.exists():
            logger.warning(f"  ⚠️  未找到回测结果，跳过容量检查")
            return

        # 从配置读取资金与阈值
        cc = (self.config.get("capital_constraints") or {}).get(pool_name, {})
        target_capital = (
            str(cc.get("target_capital")) if "target_capital" in cc else None
        )
        adv_threshold = str(cc.get("max_adv_pct")) if "max_adv_pct" in cc else None

        cmd = [
            "python3",
            "scripts/capacity_constraints.py",
            "--backtest-dir",
            str(output_dir),
            "--price-dir",
            "raw/ETF/daily",
            "--output-dir",
            str(output_dir),
            "--pool-name",
            pool_name,
            "--config-file",
            "configs/etf_pools.yaml",
        ]
        if target_capital:
            cmd += ["--target-capital", target_capital]
        if adv_threshold:
            cmd += ["--adv-threshold", adv_threshold]

        logger.info(f"  执行命令: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")

            if result.returncode == 0:
                logger.info(f"  ✅ 容量检查完成")
            else:
                logger.error(f"  ❌ 容量检查失败: {result.stderr}")
        except Exception as e:
            logger.error(f"  ❌ 容量检查执行失败: {e}")

    def validate_pool(self, pool_name):
        """验证池配置"""
        logger.info(f"\n验证{pool_name}配置...")

        pool_config = self.config["pools"][pool_name]
        symbols = pool_config["symbols"]

        # 检查数据文件是否存在
        data_dir = Path("raw/ETF/daily")
        missing_symbols = []

        for symbol in symbols:
            data_files = list(data_dir.glob(f"{symbol}_*.parquet"))
            if len(data_files) == 0:
                missing_symbols.append(symbol)

        if len(missing_symbols) > 0:
            logger.warning(f"  ⚠️  缺少{len(missing_symbols)}个ETF数据:")
            for symbol in missing_symbols[:5]:
                logger.warning(f"    - {symbol}")
        else:
            logger.info(f"  ✅ 所有ETF数据完整")

        return len(missing_symbols) == 0

    def combine_pools(self, weights=None):
        """顶层整合多个池"""
        logger.info("=" * 80)
        logger.info("顶层池整合")
        logger.info("=" * 80)

        if weights is None:
            weights = self.config["strategy"]["portfolio"]["weights"]

        logger.info(f"\n权重配置:")
        for pool_name, weight in weights.items():
            logger.info(f"  {pool_name}: {weight:.1%}")

        logger.info(f"\n⚠️  池整合需要:")
        logger.info(f"  1. 各池独立回测结果")
        logger.info(f"  2. 对齐交易日历")
        logger.info(f"  3. 按权重合并持仓")

        return weights


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("ETF分池管理系统")
    logger.info("=" * 80)

    try:
        # 初始化管理器
        manager = PoolManager()

        # 列出所有池
        manager.list_pools()

        # 验证各池
        logger.info("\n" + "=" * 80)
        logger.info("池验证")
        logger.info("=" * 80)

        for pool_name in manager.config["pools"].keys():
            manager.validate_pool(pool_name)

        # 生产池面板（示例）
        logger.info("\n" + "=" * 80)
        logger.info("分池生产示例")
        logger.info("=" * 80)

        for pool_name in manager.config["pools"].keys():
            output_dir = manager.produce_pool_panel(pool_name)
            logger.info(f"\n✅ {pool_name}面板输出目录: {output_dir}")

        # 顶层整合
        logger.info("\n" + "=" * 80)
        logger.info("顶层整合")
        logger.info("=" * 80)

        weights = manager.combine_pools()

        logger.info("\n" + "=" * 80)
        logger.info("✅ 分池管理系统就绪")
        logger.info("=" * 80)

        logger.info(f"\n下一步:")
        logger.info(f"  1. 分池生产因子面板")
        logger.info(f"  2. 分池回测")
        logger.info(f"  3. 顶层权重整合")

        return True

    except Exception as e:
        logger.error(f"❌ 失败: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
