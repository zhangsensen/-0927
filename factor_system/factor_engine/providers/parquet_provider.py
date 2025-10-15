"""
Parquet数据提供者 - 多市场支持 + 分钟转日线
支持：HK/US/SH/SZ市场，自动resample分钟数据为日线
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

from factor_system.factor_engine.providers.base import DataProvider

logger = logging.getLogger(__name__)


class ParquetDataProvider(DataProvider):
    """
    Parquet数据提供者 - 多市场统一接口

    特性：
    1. 支持HK/US/SH/SZ多市场
    2. 自动分钟转日线（timeframe="daily"时）
    3. 智能文件查找（优先日线，回退分钟）
    """

    def __init__(self, raw_data_dir: Path):
        """
        初始化数据提供者

        Args:
            raw_data_dir: 数据根目录，包含HK/US/SH/SZ子目录
        """
        self.raw_data_dir = raw_data_dir

        # 市场目录映射
        self.market_dirs = {
            "HK": raw_data_dir / "HK",
            "US": raw_data_dir / "US",
            "SH": raw_data_dir / "SH",
            "SZ": raw_data_dir / "SZ",
            "ETF": raw_data_dir / "ETF" / "daily",
        }

        # 检查至少一个市场存在
        existing_markets = [m for m, d in self.market_dirs.items() if d.exists()]
        if not existing_markets:
            raise ValueError(
                f"未找到任何市场数据目录: {list(self.market_dirs.values())}"
            )

        logger.info("=" * 60)
        logger.info("ParquetDataProvider 初始化")
        logger.info("=" * 60)
        logger.info(f"✅ 支持市场: {existing_markets}")
        logger.info(f"📁 已启用目录:")
        for market in existing_markets:
            logger.info(f"   - {market}: {self.market_dirs[market]}")
        logger.info(f"🔄 minute→daily 聚合: 已启用（当日线文件不存在时自动触发）")
        logger.info("=" * 60)

        # 记录不存在的市场
        for market, dir_path in self.market_dirs.items():
            if not dir_path.exists():
                logger.debug(f"{market} 市场目录不存在: {dir_path}")

    def load_price_data(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        加载价格数据

        Args:
            symbols: 股票代码列表（如 ["600036.SH", "0700.HK"]）
            timeframe: 时间框架（支持 "daily", "15min", "60min" 等）
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            MultiIndex DataFrame (symbol, timestamp) × OHLCV
        """
        # 输入验证
        self._validate_inputs(symbols, timeframe)

        all_data = []

        for symbol in symbols:
            try:
                # 加载单个股票数据
                symbol_data = self._load_single_symbol(
                    symbol, timeframe, start_date, end_date
                )

                if not symbol_data.empty:
                    # 添加symbol列并设置MultiIndex
                    symbol_data["symbol"] = symbol
                    symbol_data = symbol_data.set_index("symbol", append=True)
                    symbol_data = symbol_data.swaplevel(0, 1)  # (symbol, timestamp)
                    all_data.append(symbol_data)

                    logger.info(f"✅ {symbol}: {len(symbol_data)} 条记录")
                else:
                    logger.warning(f"⚠️ {symbol}: 无数据")

            except Exception as e:
                logger.error(f"❌ {symbol} 加载失败: {e}")
                continue

        if not all_data:
            logger.warning("未加载到任何数据")
            return pd.DataFrame()

        # 合并所有股票数据
        result = pd.concat(all_data)
        logger.info(f"数据加载完成: {result.shape}")

        return result

    def _load_single_symbol(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """加载单个股票数据"""
        # 检测市场
        market = self._detect_market(symbol)
        market_dir = self.market_dirs[market]

        if not market_dir.exists():
            raise ValueError(f"{market} 市场目录不存在: {market_dir}")

        # 查找文件
        if timeframe == "daily":
            # 优先查找日线文件
            data = self._try_load_daily_file(symbol, market_dir, start_date, end_date)

            if data.empty:
                # 回退：从分钟数据resample
                logger.info(f"{symbol}: 未找到日线文件，尝试从分钟数据转换")
                data = self._load_and_resample_minute(
                    symbol, market_dir, start_date, end_date
                )
        else:
            # 加载指定时间框架数据
            data = self._load_timeframe_file(
                symbol, market_dir, timeframe, start_date, end_date
            )

        return data

    def _try_load_daily_file(
        self,
        symbol: str,
        market_dir: Path,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """尝试加载日线文件"""
        # 可能的文件名模式
        patterns = [
            f"{symbol}_daily_*.parquet",
            f"{symbol}_1day_*.parquet",
            f"{symbol.replace('.', '')}_daily_*.parquet",
        ]

        for pattern in patterns:
            files = list(market_dir.glob(pattern))
            if files:
                logger.info(f"✅ {symbol}: 已优先日线文件，未触发分钟聚合")
                logger.debug(f"   文件: {files[0].name}")
                return self._read_and_filter(files[0], start_date, end_date)

        return pd.DataFrame()

    def _load_and_resample_minute(
        self,
        symbol: str,
        market_dir: Path,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """从分钟数据resample为日线"""
        # 查找分钟级文件
        minute_patterns = [
            f"{symbol}_1min_*.parquet",
            f"{symbol}.parquet",  # 无后缀的默认文件
            f"{symbol.replace('.', '')}.parquet",
        ]

        minute_file = None
        for pattern in minute_patterns:
            files = list(market_dir.glob(pattern))
            if files:
                minute_file = files[0]
                break

        if not minute_file:
            logger.warning(f"{symbol}: 未找到分钟数据文件")
            return pd.DataFrame()

        logger.info(f"{symbol}: 从分钟数据转日线 ({minute_file.name})")

        # 读取分钟数据
        minute_data = pd.read_parquet(minute_file)

        # 确保有datetime列
        if "datetime" not in minute_data.columns:
            if minute_data.index.name == "datetime" or isinstance(
                minute_data.index, pd.DatetimeIndex
            ):
                minute_data = minute_data.reset_index()
            else:
                raise ValueError(f"{symbol}: 缺少datetime列")

        # 转换为datetime类型
        minute_data["datetime"] = pd.to_datetime(minute_data["datetime"])

        # 过滤时间范围
        mask = (minute_data["datetime"] >= start_date) & (
            minute_data["datetime"] <= end_date
        )
        minute_data = minute_data[mask]

        if minute_data.empty:
            return pd.DataFrame()

        # Resample到日线
        daily_data = self._resample_to_daily(minute_data)

        return daily_data

    def _resample_to_daily(self, minute_data: pd.DataFrame) -> pd.DataFrame:
        """
        分钟数据重采样为日线

        聚合规则：
        - open: first
        - high: max
        - low: min
        - close: last
        - volume: sum
        """
        # 设置datetime为索引
        minute_data = minute_data.set_index("datetime")

        # 定义聚合规则
        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }

        # 只聚合存在的列
        agg_dict = {k: v for k, v in agg_dict.items() if k in minute_data.columns}

        # Resample（从数据起始点对齐）
        daily = minute_data.resample(
            "D", origin="start", label="left", closed="left"
        ).agg(agg_dict)

        # 过滤非交易日（OHLC全为NaN）
        daily = daily.dropna(subset=["open", "high", "low", "close"], how="all")

        # 保持datetime索引（不要reset_index，以便与资金流数据对齐）
        logger.debug(f"Resample完成: {len(minute_data)} 分钟 -> {len(daily)} 日线")

        return daily

    def _resample_to_timeframe(
        self, minute_data: pd.DataFrame, timeframe: str, symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """
        分钟数据重采样为指定时间框架

        A股交易时间：9:30-11:30（上午120分钟），13:00-15:00（下午120分钟）
        为避免跨午休重采样，使用简单的resample + 过滤非交易时段

        Args:
            minute_data: 分钟级数据
            timeframe: 目标时间框架 ("1min", "5min", "15min", "30min", "60min", "2h", "4h")

        Returns:
            重采样后的数据
        """
        # 时间框架映射
        resample_map = {
            "1min": "1min",
            "5min": "5min",
            "15min": "15min",
            "30min": "30min",
            "60min": "60min",
            "120min": "120min",
            "240min": "240min",
            "2h": "120min",
            "4h": "240min",
        }

        resample_freq = resample_map.get(timeframe, "1min")

        # 定义聚合规则
        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }

        # 只聚合存在的列
        agg_dict = {k: v for k, v in agg_dict.items() if k in minute_data.columns}

        # 确保datetime是索引
        if "datetime" in minute_data.columns:
            minute_data = minute_data.set_index("datetime")

        # 如果是A股(SH/SZ)且需要会话感知重采样，则走专用路径
        try:
            is_ashare = symbol is not None and (
                symbol.endswith(".SH") or symbol.endswith(".SZ")
            )
        except Exception:
            is_ashare = False

        # 240min=4小时，跨越整个交易日，不适合会话感知重采样
        if is_ashare and resample_freq in {"5min", "15min", "30min", "60min", "120min"}:
            from factor_system.utils.session_resample import resample_ashare_intraday

            res = resample_ashare_intraday(minute_data, resample_freq)
            if not res.empty:
                res["session_aware"] = True
            logger.debug(
                f"会话重采样: {symbol} {len(minute_data)} -> {len(res)} {timeframe}"
            )
            return res

        # 非A股或其他频率，退回常规resample（不跨午休的严格性不保证）
        resampled_data = minute_data.resample(resample_freq).agg(agg_dict)
        resampled_data = resampled_data.dropna(
            subset=["open", "high", "low", "close"], how="all"
        )

        # A股240min特殊处理：过滤非交易时段的K线
        if is_ashare and resample_freq == "240min":
            # 240min每天会产生2根（08:00, 12:00），但只有包含交易时段的才有效
            # 保留12:00-16:00的K线（包含下午交易时段13:00-15:00）
            def is_valid_240min(ts):
                h = ts.hour
                # 保留12:00或16:00的K线
                return h in [12, 16]

            resampled_data = resampled_data[resampled_data.index.map(is_valid_240min)]

        logger.debug(
            f"通用重采样: {len(minute_data)} -> {len(resampled_data)} {timeframe}"
        )
        return resampled_data

    def _load_timeframe_file(
        self,
        symbol: str,
        market_dir: Path,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """加载指定时间框架文件"""
        # 时间框架映射
        tf_map = {
            "1min": "1min",
            "5min": "5min",
            "15min": "15min",
            "30min": "30min",
            "60min": "60min",
            "2h": "2h",
            "4h": "4h",
        }

        tf_str = tf_map.get(timeframe, timeframe)

        # 查找文件
        patterns = [
            f"{symbol}_{tf_str}_*.parquet",
            f"{symbol.replace('.', '')}_{tf_str}_*.parquet",
            f"{symbol}.parquet",  # 默认文件（1分钟数据）
            f"{symbol.replace('.', '')}.parquet",
        ]

        for i, pattern in enumerate(patterns):
            files = list(market_dir.glob(pattern))
            if files:
                df = self._read_and_filter(files[0], start_date, end_date)

                # 如果加载的是默认文件（1分钟数据），且需要转换为其他时间框架
                if i >= 2 and timeframe != "1min":  # 后两个pattern是默认文件
                    logger.info(f"{symbol}: 从1分钟数据转换为{timeframe}")
                    df = self._resample_to_timeframe(df, timeframe, symbol)

                return df

        logger.warning(f"{symbol}: 未找到 {timeframe} 数据文件")
        return pd.DataFrame()

    def _read_and_filter(
        self,
        file_path: Path,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """读取并过滤数据"""
        df = pd.read_parquet(file_path)

        # ETF数据列名映射
        if "trade_date" in df.columns:
            df["datetime"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
            df = df.drop(columns=["trade_date"])
        if "vol" in df.columns and "volume" not in df.columns:
            df["volume"] = df["vol"]

        # 确保有datetime列
        if "datetime" not in df.columns:
            if df.index.name == "datetime" or isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
            else:
                raise ValueError(f"文件缺少datetime列: {file_path}")

        # 转换为datetime
        df["datetime"] = pd.to_datetime(df["datetime"])

        # 过滤时间范围（end_date扩展到当天结束23:59:59）
        from datetime import timedelta

        end_date_inclusive = end_date + timedelta(days=1) - timedelta(seconds=1)
        mask = (df["datetime"] >= start_date) & (df["datetime"] <= end_date_inclusive)
        df = df[mask]

        # 设置索引
        df = df.set_index("datetime")

        return df

    def _detect_market(self, symbol: str) -> str:
        """检测股票所属市场"""
        if symbol.endswith(".HK"):
            return "HK"
        elif symbol.endswith(".US"):
            return "US"
        elif symbol.endswith(".SH"):
            # ETF优先判断（检查ETF目录是否有对应文件）
            etf_dir = self.market_dirs.get("ETF")
            if etf_dir and etf_dir.exists():
                etf_files = list(etf_dir.glob(f"{symbol}_daily_*.parquet"))
                if etf_files:
                    return "ETF"
            return "SH"
        elif symbol.endswith(".SZ"):
            # ETF优先判断
            etf_dir = self.market_dirs.get("ETF")
            if etf_dir and etf_dir.exists():
                etf_files = list(etf_dir.glob(f"{symbol}_daily_*.parquet"))
                if etf_files:
                    return "ETF"
            return "SZ"
        else:
            raise ValueError(
                f"无法识别市场: {symbol}，支持格式: '0700.HK', 'BABA.US', '600036.SH', '000001.SZ'"
            )

    def _validate_inputs(self, symbols: List[str], timeframe: str):
        """验证输入参数"""
        if not symbols:
            raise ValueError("symbols列表不能为空")

        # 验证symbol格式
        valid_suffixes = (".HK", ".US", ".SH", ".SZ")
        for symbol in symbols:
            if not any(symbol.endswith(suffix) for suffix in valid_suffixes):
                raise ValueError(
                    f"symbol格式错误: {symbol}，支持格式: '0700.HK', 'BABA.US', '600036.SH', '000001.SZ'"
                )

        # 验证timeframe
        valid_timeframes = {
            "1min",
            "5min",
            "15min",
            "30min",
            "60min",
            "120min",
            "240min",
            "2h",
            "4h",
            "daily",
        }
        if timeframe not in valid_timeframes:
            raise ValueError(
                f"不支持的timeframe: {timeframe}，支持: {valid_timeframes}"
            )

    def load_fundamental_data(self, *args, **kwargs) -> pd.DataFrame:
        """加载基本面数据（暂未实现）"""
        logger.warning("load_fundamental_data 未实现")
        return pd.DataFrame()

    def get_trading_calendar(self, *args, **kwargs) -> List:
        """获取交易日历（暂未实现）"""
        logger.warning("get_trading_calendar 未实现")
        return []
