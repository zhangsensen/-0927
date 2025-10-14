#!/usr/bin/env python3
"""
资金流因子生产脚本 - 全量计算
Linus准则：配置驱动、可追溯、向量化、无冗余
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from factor_system.factor_engine.core.enhanced_engine import EnhancedFactorEngine
from factor_system.factor_engine.core.registry import get_global_registry
from factor_system.factor_engine.factors.money_flow.registry import (
    get_money_flow_factor_sets,
)
from factor_system.factor_engine.providers.money_flow_provider_engine import (
    MoneyFlowDataProvider,
)


class MoneyFlowFactorProducer:
    """资金流因子生产器"""

    def __init__(self, config_path: str):
        """
        初始化生产器

        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.project_root = project_root
        self._setup_logging()
        self.logger = logging.getLogger(__name__)

        # 初始化引擎
        self._init_engine()

    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _setup_logging(self):
        """配置日志系统"""
        log_config = self.config["logging"]
        log_level = getattr(logging, log_config["level"])

        # 创建日志目录
        if log_config["file_logging"]:
            log_dir = self.project_root / self.config["data_paths"]["log_dir"]
            log_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"production_{timestamp}.log"

            logging.basicConfig(
                level=log_level,
                format=log_config["format"],
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler(sys.stdout),
                ],
            )
        else:
            logging.basicConfig(
                level=log_level,
                format=log_config["format"],
                handlers=[logging.StreamHandler(sys.stdout)],
            )

    def _init_engine(self):
        """初始化因子引擎"""
        self.logger.info("初始化资金流因子引擎...")

        # 获取注册表
        registry = get_global_registry()

        # 创建数据提供者
        money_flow_dir = self.project_root / self.config["data_paths"]["money_flow_dir"]
        self.money_flow_provider = MoneyFlowDataProvider(
            money_flow_dir=money_flow_dir,
            enforce_t_plus_1=self.config["time_config"]["enforce_t_plus_1"],
        )

        # 创建增强引擎（简化版，不需要价格提供者）
        class DummyPriceProvider:
            def load_price_data(self, *args, **kwargs):
                return pd.DataFrame()

            def load_fundamental_data(self, *args, **kwargs):
                return pd.DataFrame()

            def get_trading_calendar(self, *args, **kwargs):
                return []

        self.engine = EnhancedFactorEngine(
            data_provider=DummyPriceProvider(), registry=registry
        )

        self.logger.info("✅ 引擎初始化完成")

    def _get_symbols(self) -> List[str]:
        """获取标的列表"""
        symbol_config = self.config["symbols"]

        if symbol_config["mode"] == "auto_scan":
            # 自动扫描数据目录
            money_flow_dir = self.project_root / self.config["data_paths"]["money_flow_dir"]
            symbols = []

            for file in money_flow_dir.glob("*.parquet"):
                # 提取股票代码（去除_moneyflow.parquet后缀）
                symbol = file.stem.replace("_moneyflow", "").replace("_money_flow", "")
                symbols.append(symbol)

            self.logger.info(f"📊 自动扫描到 {len(symbols)} 个标的")
            return sorted(symbols)
        else:
            # 手动指定
            symbols = symbol_config["manual_list"]
            self.logger.info(f"📊 使用手动指定的 {len(symbols)} 个标的")
            return symbols

    def _get_factors(self) -> List[str]:
        """获取因子列表"""
        factor_config = self.config["factors"]
        mode = factor_config["mode"]

        if mode == "all":
            # 全部因子
            factor_sets = get_money_flow_factor_sets()
            factors = factor_sets["money_flow_all"]["factors"]
            self.logger.info(f"🔧 使用全部 {len(factors)} 个资金流因子")
        elif mode == "core":
            # 核心因子
            factors = factor_config["core_factors"]
            self.logger.info(f"🔧 使用核心 {len(factors)} 个因子")
        elif mode == "custom":
            # 自定义因子
            factors = factor_config["custom_factors"]
            self.logger.info(f"🔧 使用自定义 {len(factors)} 个因子")
        else:
            raise ValueError(f"未知的因子模式: {mode}")

        return factors

    def produce(self):
        """执行因子生产"""
        self.logger.info("=" * 60)
        self.logger.info("🚀 开始资金流因子生产")
        self.logger.info("=" * 60)

        # 获取配置
        symbols = self._get_symbols()
        factors = self._get_factors()
        time_config = self.config["time_config"]

        start_date = datetime.strptime(time_config["start_date"], "%Y-%m-%d")
        end_date = datetime.strptime(time_config["end_date"], "%Y-%m-%d")

        self.logger.info(f"📅 时间范围: {start_date.date()} ~ {end_date.date()}")
        self.logger.info(f"📊 标的数量: {len(symbols)}")
        self.logger.info(f"🔧 因子数量: {len(factors)}")

        # 创建输出目录
        output_dir = self.project_root / self.config["data_paths"]["money_flow_output"]
        output_dir.mkdir(parents=True, exist_ok=True)

        # 批量处理
        execution_config = self.config["execution"]
        batch_size = execution_config["batch_size"]
        total_batches = (len(symbols) + batch_size - 1) // batch_size

        all_results = []
        success_count = 0
        error_count = 0

        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(symbols))
            batch_symbols = symbols[batch_start:batch_end]

            self.logger.info(
                f"\n📦 处理批次 {batch_idx + 1}/{total_batches}: "
                f"{batch_symbols[0]} ~ {batch_symbols[-1]}"
            )

            try:
                # 计算因子
                result = self.engine.calculate_money_flow_factors(
                    factors, batch_symbols, start_date, end_date
                )

                if not result.empty:
                    all_results.append(result)
                    success_count += len(batch_symbols)
                    self.logger.info(
                        f"✅ 批次完成: {result.shape[0]} 条记录, {result.shape[1]} 个因子"
                    )
                else:
                    error_count += len(batch_symbols)
                    self.logger.warning(f"⚠️ 批次结果为空")

            except Exception as e:
                error_count += len(batch_symbols)
                self.logger.error(f"❌ 批次处理失败: {e}")
                if not execution_config["continue_on_error"]:
                    raise

        # 合并结果
        if all_results:
            self.logger.info("\n📊 合并结果...")
            final_result = pd.concat(all_results)

            # 保存结果
            self._save_results(final_result, symbols, start_date, end_date)

            # 生成报告
            if self.config["output"]["generate_report"]:
                self._generate_report(final_result, success_count, error_count)

        # 总结
        self.logger.info("\n" + "=" * 60)
        self.logger.info("✅ 资金流因子生产完成")
        self.logger.info(f"成功: {success_count} 个标的")
        self.logger.info(f"失败: {error_count} 个标的")
        self.logger.info("=" * 60)

    def _save_results(
        self, result: pd.DataFrame, symbols: List[str], start_date: datetime, end_date: datetime
    ):
        """保存结果"""
        output_config = self.config["output"]
        output_dir = self.project_root / self.config["data_paths"]["money_flow_output"]

        # 生成文件名
        filename = output_config["filename_template"].format(
            symbol="ALL",
            start_date=start_date.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d"),
        )

        output_path = output_dir / filename

        # 保存
        if output_config["format"] == "parquet":
            result.to_parquet(
                output_path, compression=output_config["compression"], index=True
            )
        elif output_config["format"] == "csv":
            result.to_csv(output_path, index=True)
        else:
            raise ValueError(f"不支持的输出格式: {output_config['format']}")

        self.logger.info(f"💾 结果已保存: {output_path}")
        self.logger.info(f"   数据形状: {result.shape}")
        self.logger.info(f"   文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    def _generate_report(self, result: pd.DataFrame, success_count: int, error_count: int):
        """生成生产报告"""
        output_dir = self.project_root / self.config["data_paths"]["money_flow_output"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"production_report_{timestamp}.md"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# 资金流因子生产报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## 📊 生产统计\n\n")
            f.write(f"- **成功标的**: {success_count}\n")
            f.write(f"- **失败标的**: {error_count}\n")
            f.write(f"- **总记录数**: {result.shape[0]}\n")
            f.write(f"- **因子数量**: {result.shape[1]}\n\n")

            f.write("## 🔧 因子列表\n\n")
            for col in result.columns:
                f.write(f"- {col}\n")

            f.write("\n## 📈 数据质量\n\n")
            f.write("### 缺失值统计\n\n")
            missing = result.isnull().sum()
            missing_pct = (missing / len(result) * 100).round(2)
            for col in result.columns:
                f.write(f"- **{col}**: {missing[col]} ({missing_pct[col]}%)\n")

            f.write("\n### 基本统计\n\n")
            f.write("```\n")
            f.write(result.describe().to_string())
            f.write("\n```\n")

        self.logger.info(f"📄 报告已生成: {report_path}")


def main():
    """主函数"""
    # 配置文件路径
    config_path = project_root / "factor_system/config/money_flow_config.yaml"

    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        sys.exit(1)

    # 创建生产器并执行
    producer = MoneyFlowFactorProducer(str(config_path))
    producer.produce()


if __name__ == "__main__":
    main()
