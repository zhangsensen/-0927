#!/usr/bin/env python3
"""
资金流因子生产脚本 V2 - 直接调用因子类
Linus准则：消除中间层，直接计算
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from factor_system.factor_engine.factors.money_flow.core import (
    Flow_Price_Divergence,
    LargeOrder_Ratio,
    MainFlow_Momentum,
    MainNetInflow_Rate,
    MoneyFlow_Consensus,
    MoneyFlow_Hierarchy,
    OrderConcentration,
    SuperLargeOrder_Ratio,
)
from factor_system.factor_engine.factors.money_flow.enhanced import (
    Flow_Reversal_Ratio,
    Flow_Tier_Ratio_Delta,
    Institutional_Absorption,
)
from factor_system.factor_engine.providers.money_flow_provider_engine import (
    MoneyFlowDataProvider,
)


class DirectMoneyFlowProducer:
    """直接计算资金流因子"""

    # 因子类映射
    FACTOR_CLASSES = {
        "MainNetInflow_Rate": MainNetInflow_Rate,
        "LargeOrder_Ratio": LargeOrder_Ratio,
        "SuperLargeOrder_Ratio": SuperLargeOrder_Ratio,
        "OrderConcentration": OrderConcentration,
        "MoneyFlow_Hierarchy": MoneyFlow_Hierarchy,
        "MoneyFlow_Consensus": MoneyFlow_Consensus,
        "MainFlow_Momentum": MainFlow_Momentum,
        "Flow_Price_Divergence": Flow_Price_Divergence,
        "Institutional_Absorption": Institutional_Absorption,
        "Flow_Tier_Ratio_Delta": Flow_Tier_Ratio_Delta,
        "Flow_Reversal_Ratio": Flow_Reversal_Ratio,
    }

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.project_root = project_root
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        self._init_provider()

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _setup_logging(self):
        log_config = self.config["logging"]
        log_level = getattr(logging, log_config["level"])

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

    def _init_provider(self):
        money_flow_dir = self.project_root / self.config["data_paths"]["money_flow_dir"]
        self.provider = MoneyFlowDataProvider(
            money_flow_dir=money_flow_dir,
            enforce_t_plus_1=self.config["time_config"]["enforce_t_plus_1"],
        )
        self.logger.info("✅ 数据提供者初始化完成")

    def _get_symbols(self):
        symbol_config = self.config["symbols"]
        if symbol_config["mode"] == "auto_scan":
            money_flow_dir = (
                self.project_root / self.config["data_paths"]["money_flow_dir"]
            )
            symbols = []
            for file in money_flow_dir.glob("*.parquet"):
                symbol = file.stem.replace("_moneyflow", "").replace("_money_flow", "")
                symbols.append(symbol)
            self.logger.info(f"📊 自动扫描到 {len(symbols)} 个标的")
            return sorted(symbols)
        else:
            symbols = symbol_config["manual_list"]
            self.logger.info(f"📊 使用手动指定的 {len(symbols)} 个标的")
            return symbols

    def _get_factors(self):
        factor_config = self.config["factors"]
        mode = factor_config["mode"]

        if mode == "all":
            factors = list(self.FACTOR_CLASSES.keys())
            self.logger.info(f"🔧 计算全部 {len(factors)} 个因子")
        elif mode == "core":
            factors = factor_config["core_factors"]
            self.logger.info(f"🔧 计算核心 {len(factors)} 个因子")
        else:
            factors = factor_config["custom_factors"]

        return factors

    def produce(self):
        self.logger.info("=" * 60)
        self.logger.info("🚀 开始资金流因子生产（直接计算模式）")
        self.logger.info("=" * 60)

        symbols = self._get_symbols()
        factors = self._get_factors()
        time_config = self.config["time_config"]

        start_date = time_config["start_date"]
        end_date = time_config["end_date"]

        self.logger.info(f"📅 时间范围: {start_date} ~ {end_date}")

        # 加载所有数据
        self.logger.info("📥 加载资金流数据...")
        money_flow_data = self.provider.load_money_flow_data(
            symbols, "1day", start_date, end_date
        )

        if money_flow_data.empty:
            self.logger.error("❌ 未加载到任何数据")
            return

        self.logger.info(f"✅ 数据加载完成: {money_flow_data.shape}")

        # 计算因子
        results = {}
        for factor_name in factors:
            self.logger.info(f"🔧 计算因子: {factor_name}")

            try:
                factor_class = self.FACTOR_CLASSES[factor_name]
                factor_instance = factor_class()

                # 逐个股票计算
                factor_values = []
                for symbol in symbols:
                    try:
                        symbol_data = money_flow_data.xs(symbol, level="symbol")
                        if symbol_data.empty:
                            continue

                        # 调用因子计算方法（返回Series）
                        values = factor_instance.calculate(symbol_data)

                        # 构建MultiIndex DataFrame
                        if values is not None and isinstance(values, pd.Series):
                            # 创建MultiIndex
                            multi_idx = pd.MultiIndex.from_product(
                                [[symbol], values.index],
                                names=["symbol", "trade_date"],
                            )
                            factor_df = pd.DataFrame(
                                {factor_name: values.values}, index=multi_idx
                            )
                            factor_values.append(factor_df)

                    except Exception as e:
                        self.logger.warning(f"  ⚠️ {symbol} 计算失败: {e}")
                        continue

                if factor_values:
                    results[factor_name] = pd.concat(factor_values)
                    self.logger.info(
                        f"  ✅ {factor_name}: {results[factor_name].shape[0]} 条记录"
                    )
                else:
                    self.logger.warning(f"  ⚠️ {factor_name}: 无有效数据")

            except Exception as e:
                self.logger.error(f"  ❌ {factor_name} 失败: {e}")
                continue

        # 合并结果
        if results:
            self.logger.info("\n📊 合并因子结果...")
            final_result = pd.concat(results.values(), axis=1)
            self.logger.info(f"✅ 合并完成: {final_result.shape}")

            # 保存
            self._save_results(final_result, symbols, start_date, end_date)
            self._generate_report(final_result, len(symbols), 0)
        else:
            self.logger.error("❌ 无任何因子计算成功")

        self.logger.info("\n" + "=" * 60)
        self.logger.info("✅ 资金流因子生产完成")
        self.logger.info("=" * 60)

    def _save_results(self, result, symbols, start_date, end_date):
        output_config = self.config["output"]
        output_dir = self.project_root / self.config["data_paths"]["money_flow_output"]
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = output_config["filename_template"].format(
            symbol="ALL",
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
        )

        output_path = output_dir / filename

        if output_config["format"] == "parquet":
            result.to_parquet(
                output_path, compression=output_config["compression"], index=True
            )
        else:
            result.to_csv(output_path, index=True)

        self.logger.info(f"💾 结果已保存: {output_path}")
        self.logger.info(f"   数据形状: {result.shape}")
        self.logger.info(
            f"   文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB"
        )

    def _generate_report(self, result, success_count, error_count):
        output_dir = self.project_root / self.config["data_paths"]["money_flow_output"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"production_report_{timestamp}.md"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# 资金流因子生产报告（直接计算模式）\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## 📊 生产统计\n\n")
            f.write(f"- **成功标的**: {success_count}\n")
            f.write(f"- **失败标的**: {error_count}\n")
            f.write(f"- **总记录数**: {result.shape[0]}\n")
            f.write(f"- **因子数量**: {result.shape[1]}\n\n")

            f.write("## 🔧 因子列表\n\n")
            for col in result.columns:
                non_null = result[col].notna().sum()
                pct = non_null / len(result) * 100
                f.write(f"- **{col}**: {non_null} 条有效数据 ({pct:.1f}%)\n")

            f.write("\n## 📈 数据质量\n\n")
            f.write("### 缺失值统计\n\n")
            missing = result.isnull().sum()
            missing_pct = (missing / len(result) * 100).round(2)
            for col in result.columns:
                f.write(f"- **{col}**: {missing[col]} ({missing_pct[col]}%)\n")

            f.write("\n### 基本统计\n\n")
            f.write("```")
            f.write(result.describe().to_string())
            f.write("\n```\n")

        self.logger.info(f"📄 报告已生成: {report_path}")


def main():
    config_path = project_root / "factor_system/config/money_flow_config.yaml"

    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        sys.exit(1)

    producer = DirectMoneyFlowProducer(str(config_path))
    producer.produce()


if __name__ == "__main__":
    main()
