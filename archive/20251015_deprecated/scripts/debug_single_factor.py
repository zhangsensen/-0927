#!/usr/bin/env python3
"""单因子诊断脚本 - 快速定位全NaN根因

用法：
    python scripts/debug_single_factor.py --factor-id TA_SMA_20 --symbol 510300.SH
    python scripts/debug_single_factor.py --factor-id MACD_SIGNAL --symbol 159915.SZ --start 20200101 --end 20251014
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FactorDebugger:
    """因子诊断器"""

    def __init__(self, data_dir: str = "raw/ETF/daily"):
        self.data_dir = Path(data_dir)

    def load_symbol_data(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """加载单个标的数据"""
        logger.info(f"加载数据: {symbol}")

        # 查找文件
        pattern = f"{symbol}*daily*.parquet"
        files = list(self.data_dir.glob(pattern))

        if not files:
            logger.error(f"未找到文件: {pattern}")
            return pd.DataFrame()

        file = files[0]
        logger.info(f"文件: {file}")

        # 加载
        df = pd.read_parquet(file)
        logger.info(f"原始形状: {df.shape}")
        logger.info(f"列名: {df.columns.tolist()}")

        # 标准化列名
        if "trade_date" in df.columns:
            df["date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        else:
            logger.error("未找到日期列")
            return pd.DataFrame()

        # 日期过滤
        df = df[
            (df["date"] >= pd.to_datetime(start_date))
            & (df["date"] <= pd.to_datetime(end_date))
        ].copy()

        # 排序
        df = df.sort_values("date").reset_index(drop=True)

        logger.info(f"过滤后形状: {df.shape}")
        logger.info(f"日期范围: {df['date'].min()} ~ {df['date'].max()}")

        return df

    def diagnose_factor(
        self,
        factor_id: str,
        symbol: str,
        start_date: str = "20200101",
        end_date: str = "20251014",
    ) -> dict:
        """诊断单个因子"""
        logger.info("=" * 60)
        logger.info(f"诊断因子: {factor_id}")
        logger.info("=" * 60)

        result = {
            "factor_id": factor_id,
            "symbol": symbol,
            "success": False,
            "error": None,
            "stats": {},
        }

        try:
            # 1. 加载数据
            data = self.load_symbol_data(symbol, start_date, end_date)
            if data.empty:
                result["error"] = "数据加载失败"
                return result

            # 2. 检查价格字段
            logger.info("\n检查价格字段:")
            price_fields = ["close", "adj_close", "open", "high", "low", "volume"]
            for field in price_fields:
                if field in data.columns:
                    logger.info(
                        f"  ✅ {field}: {data[field].notna().sum()}/{len(data)} 非NaN"
                    )
                else:
                    logger.warning(f"  ❌ {field}: 不存在")

            # 3. 获取因子实例
            logger.info(f"\n加载因子: {factor_id}")
            from factor_system.factor_engine.core.registry import FactorRegistry

            registry = FactorRegistry()
            factor_class = registry.get_factor(factor_id)
            factor = factor_class()

            logger.info(f"因子类: {factor_class.__name__}")
            logger.info(f"min_history: {getattr(factor, 'min_history', 0)}")

            # 4. 准备输入数据
            logger.info("\n准备输入数据:")

            # 统一列名
            if "vol" in data.columns and "volume" not in data.columns:
                data["volume"] = data["vol"]
                logger.info("✅ 列名标准化: vol -> volume")

            # 确定价格字段并统一为close
            if "adj_close" in data.columns:
                price_field = "adj_close"
                data["close"] = data["adj_close"]
                logger.info("✅ 价格字段: adj_close -> close")
            elif "close" in data.columns:
                price_field = "close"
            else:
                result["error"] = "无可用价格字段"
                return result

            logger.info(f"使用价格字段: {price_field}")

            # 构造输入DataFrame（标准OHLCV格式）
            input_data = data[["date", "open", "high", "low", "close", "volume"]].copy()
            input_data = input_data.set_index("date")

            logger.info(f"输入数据形状: {input_data.shape}")
            logger.info(f"输入数据前5行:\n{input_data.head()}")

            # 5. 计算因子
            logger.info("\n计算因子:")
            factor_series = factor.calculate(input_data)

            logger.info(f"输出类型: {type(factor_series)}")
            logger.info(f"输出长度: {len(factor_series)}")

            # 6. 统计分析
            logger.info("\n统计分析:")

            total = len(factor_series)
            nan_count = factor_series.isna().sum()
            valid_count = total - nan_count
            coverage = valid_count / total if total > 0 else 0

            logger.info(f"总样本: {total}")
            logger.info(f"NaN数量: {nan_count}")
            logger.info(f"有效数量: {valid_count}")
            logger.info(f"覆盖率: {coverage:.2%}")

            if valid_count > 0:
                logger.info(f"均值: {factor_series.mean():.6f}")
                logger.info(f"标准差: {factor_series.std():.6f}")
                logger.info(f"最小值: {factor_series.min():.6f}")
                logger.info(f"最大值: {factor_series.max():.6f}")

                # 检查零方差
                if factor_series.std() == 0:
                    logger.warning("⚠️  零方差！")

            # 7. 显示样例
            logger.info("\n前10个值:")
            logger.info(factor_series.head(10))

            logger.info("\n后10个值:")
            logger.info(factor_series.tail(10))

            # 8. 检查冷启动期
            logger.info("\n冷启动期分析:")
            first_valid_idx = factor_series.first_valid_index()
            if first_valid_idx is not None:
                first_valid_pos = factor_series.index.get_loc(first_valid_idx)
                logger.info(f"首个有效值位置: {first_valid_pos}")
                logger.info(f"首个有效值日期: {first_valid_idx}")
                logger.info(f"冷启动期长度: {first_valid_pos} 天")
            else:
                logger.error("❌ 整列全NaN！")

            # 9. 保存结果
            result["success"] = True
            result["stats"] = {
                "total": int(total),
                "nan_count": int(nan_count),
                "valid_count": int(valid_count),
                "coverage": float(coverage),
                "mean": float(factor_series.mean()) if valid_count > 0 else None,
                "std": float(factor_series.std()) if valid_count > 0 else None,
                "min": float(factor_series.min()) if valid_count > 0 else None,
                "max": float(factor_series.max()) if valid_count > 0 else None,
                "first_valid_pos": (
                    int(first_valid_pos) if first_valid_idx is not None else None
                ),
                "zero_variance": (
                    bool(factor_series.std() == 0) if valid_count > 0 else None
                ),
            }

            # 10. 保存详细结果
            output_dir = Path("factor_output/debug")
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = (
                output_dir / f"{factor_id}_{symbol}_{start_date}_{end_date}.csv"
            )
            factor_df = pd.DataFrame(
                {"date": factor_series.index, "value": factor_series.values}
            )
            factor_df.to_csv(output_file, index=False)
            logger.info(f"\n✅ 详细结果已保存: {output_file}")

        except Exception as e:
            logger.error(f"❌ 计算失败: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            result["error"] = str(e)

        return result

    def batch_diagnose(self, factor_ids: list, symbol: str) -> pd.DataFrame:
        """批量诊断多个因子"""
        logger.info("=" * 60)
        logger.info(f"批量诊断 {len(factor_ids)} 个因子")
        logger.info("=" * 60)

        results = []
        for i, factor_id in enumerate(factor_ids, 1):
            logger.info(f"\n[{i}/{len(factor_ids)}] {factor_id}")
            result = self.diagnose_factor(factor_id, symbol)
            results.append(result)

        # 汇总
        summary = pd.DataFrame(
            [
                {
                    "factor_id": r["factor_id"],
                    "success": r["success"],
                    "coverage": r["stats"].get("coverage", 0) if r["success"] else 0,
                    "zero_variance": (
                        r["stats"].get("zero_variance", False)
                        if r["success"]
                        else False
                    ),
                    "first_valid_pos": (
                        r["stats"].get("first_valid_pos", None)
                        if r["success"]
                        else None
                    ),
                    "error": r["error"],
                }
                for r in results
            ]
        )

        # 保存汇总
        output_file = Path("factor_output/debug/batch_summary.csv")
        summary.to_csv(output_file, index=False)
        logger.info(f"\n✅ 批量诊断汇总已保存: {output_file}")

        return summary


def main():
    parser = argparse.ArgumentParser(description="单因子诊断脚本")
    parser.add_argument("--factor-id", required=True, help="因子ID")
    parser.add_argument("--symbol", default="510300.SH", help="标的代码")
    parser.add_argument("--start", default="20200101", help="开始日期")
    parser.add_argument("--end", default="20251014", help="结束日期")
    parser.add_argument("--data-dir", default="raw/ETF/daily", help="数据目录")
    parser.add_argument("--batch", nargs="+", help="批量诊断多个因子")

    args = parser.parse_args()

    debugger = FactorDebugger(args.data_dir)

    if args.batch:
        # 批量模式
        summary = debugger.batch_diagnose(args.batch, args.symbol)
        print("\n" + "=" * 60)
        print("批量诊断汇总")
        print("=" * 60)
        print(summary.to_string(index=False))
    else:
        # 单因子模式
        result = debugger.diagnose_factor(
            args.factor_id, args.symbol, args.start, args.end
        )

        print("\n" + "=" * 60)
        print("诊断结果")
        print("=" * 60)
        print(f"因子: {result['factor_id']}")
        print(f"成功: {result['success']}")
        if result["success"]:
            for key, value in result["stats"].items():
                print(f"{key}: {value}")
        else:
            print(f"错误: {result['error']}")


if __name__ == "__main__":
    main()
