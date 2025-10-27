"""
标准真实数据加载器 - 符合 PROJECT_GUIDELINES Step1 规范

核心原则:
1. 保留所有 NaN，禁止 ffill/bfill
2. 满窗原则：由因子计算层负责验证
3. 加载完整 OHLCV 数据
4. 前复权处理（adj_factor）
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StandardRealDataLoader:
    """
    标准真实数据加载器

    遵循 PROJECT_GUIDELINES.md Step1 规范：
    - 不填充缺失值
    - 加载完整 OHLCV 数据
    - 前复权处理
    - 只输出原始矩阵
    """

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / "raw" / "ETF" / "daily"
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")

    def load_ohlcv(
        self,
        etf_codes: Optional[list] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        加载 OHLCV 数据（前复权）

        Args:
            etf_codes: ETF 代码列表（如 ['510300', '510500']），None = 加载所有
            start_date: 开始日期 YYYY-MM-DD
            end_date: 结束日期 YYYY-MM-DD

        Returns:
            {
                'close': pd.DataFrame(index=date, columns=symbols),
                'high': pd.DataFrame,
                'low': pd.DataFrame,
                'open': pd.DataFrame,
                'volume': pd.DataFrame
            }

        注意：
        - 所有 NaN 保持原位，不填充
        - 使用 adj_close, adj_high, adj_low, adj_open（前复权）
        """
        # 1. 扫描可用文件（匹配格式：XXXXXX.SH_daily_*.parquet 或 XXXXXX.SZ_daily_*.parquet）
        parquet_files = list(self.data_dir.glob("*_daily_*.parquet"))

        # 2. 过滤指定的 ETF
        if etf_codes:
            filtered_files = []
            for code in etf_codes:
                # 尝试匹配 SH 或 SZ
                matched = [f for f in parquet_files if f.name.startswith(f"{code}.")]
                filtered_files.extend(matched)
            parquet_files = filtered_files

        logger.info(f"找到 {len(parquet_files)} 个 ETF 数据文件")

        # 3. 加载每个 ETF
        data_dict = {col: {} for col in ["close", "high", "low", "open", "volume"]}

        for file_path in parquet_files:
            # 提取代码（从文件名 XXXXXX.SH_daily_*.parquet）
            code = file_path.stem.split("_")[0].split(".")[0]

            try:
                df = pd.read_parquet(file_path)

                # 设置时间索引
                if "trade_date" in df.columns:
                    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
                    df.set_index("trade_date", inplace=True)

                # 时间范围过滤
                if start_date:
                    df = df[df.index >= start_date]
                if end_date:
                    df = df[df.index <= end_date]

                # 提取 OHLCV（使用前复权数据，保留 NaN）
                if "adj_close" in df.columns:
                    data_dict["close"][code] = df["adj_close"]
                if "adj_high" in df.columns:
                    data_dict["high"][code] = df["adj_high"]
                if "adj_low" in df.columns:
                    data_dict["low"][code] = df["adj_low"]
                if "adj_open" in df.columns:
                    data_dict["open"][code] = df["adj_open"]
                if "vol" in df.columns:
                    data_dict["volume"][code] = df["vol"]

            except Exception as e:
                logger.warning(f"加载 {code} 失败: {e}")
                continue

        # 4. 转换为 DataFrame（保留所有 NaN）
        result = {}
        for col in ["close", "high", "low", "open", "volume"]:
            if data_dict[col]:
                result[col] = pd.DataFrame(data_dict[col])

        # 5. 对齐所有 DataFrame 的索引
        if result and "close" in result:
            all_dates = result["close"].index
            for col in result:
                result[col] = result[col].reindex(all_dates)

            logger.info(
                f"加载完成: {len(result['close'].columns)} ETFs × {len(result['close'])} 日期"
            )
            logger.info(
                f"日期范围: {result['close'].index[0]} 至 {result['close'].index[-1]}"
            )
        else:
            logger.error("没有加载到任何数据")

        return result

    def compute_returns(self, close_df: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
        """
        计算收益率（保留 NaN）

        Args:
            close_df: 收盘价 DataFrame
            periods: 周期数

        Returns:
            收益率 DataFrame（NaN 传播）
        """
        returns = close_df.pct_change(periods)
        return returns

    def get_summary(self, ohlcv_data: Dict[str, pd.DataFrame]) -> dict:
        """
        获取数据质量摘要

        Args:
            ohlcv_data: OHLCV 数据字典

        Returns:
            {
                'total_dates': int,
                'total_symbols': int,
                'missing_ratio': {symbol: ratio},
                'coverage_ratio': {symbol: ratio}
            }
        """
        close_df = ohlcv_data["close"]

        summary = {
            "total_dates": len(close_df),
            "total_symbols": len(close_df.columns),
            "date_range": (
                str(close_df.index[0].date()),
                str(close_df.index[-1].date()),
            ),
            "missing_ratio": {},
            "coverage_ratio": {},
        }

        for col in close_df.columns:
            total = len(close_df)
            missing = close_df[col].isna().sum()
            summary["missing_ratio"][col] = missing / total
            summary["coverage_ratio"][col] = 1 - (missing / total)

        return summary


def test_loader():
    """测试加载器"""
    loader = StandardRealDataLoader()

    # 加载少量 ETF 测试
    test_codes = ["510300", "510500", "159915"]
    ohlcv = loader.load_ohlcv(
        etf_codes=test_codes, start_date="2020-01-01", end_date="2024-12-31"
    )

    print("\n" + "=" * 80)
    print("数据加载测试")
    print("=" * 80)

    for col, df in ohlcv.items():
        print(f"\n{col.upper()}:")
        print(f"  形状: {df.shape}")
        print(f"  缺失率: {df.isna().sum().sum() / (df.shape[0] * df.shape[1]):.2%}")
        print(f"  示例:\n{df.head(3)}")

    # 计算收益率
    returns = loader.compute_returns(ohlcv["close"])
    print(f"\n收益率:")
    print(f"  形状: {returns.shape}")
    print(
        f"  缺失率: {returns.isna().sum().sum() / (returns.shape[0] * returns.shape[1]):.2%}"
    )

    # 摘要
    summary = loader.get_summary(ohlcv)
    print(f"\n数据摘要:")
    print(f"  日期数: {summary['total_dates']}")
    print(f"  标的数: {summary['total_symbols']}")
    print(f"  日期范围: {summary['date_range']}")
    print(f"\n覆盖率:")
    for code, ratio in summary["coverage_ratio"].items():
        print(f"    {code}: {ratio:.2%}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_loader()
