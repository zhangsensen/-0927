"""
数据加载器 | Data Loader

统一数据加载接口，合并自 scripts/standard_real_data_loader.py

作者: Linus
日期: 2025-10-XX
"""

import hashlib
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """
    统一数据加载器

    遵循 PROJECT_GUIDELINES.md Step1 规范：
    - 不填充缺失值
    - 加载完整 OHLCV 数据
    - 前复权处理
    - 只输出原始矩阵
    """

    def __init__(self, data_dir: Optional[str] = None, cache_dir: Optional[str] = None):
        if data_dir is None:
            # 优先从环境变量读取，然后默认路径
            import os

            data_dir = os.getenv("ETF_DATA_DIR")
            if data_dir is None:
                project_root = Path(__file__).parent.parent.parent
                data_dir = project_root / "raw" / "ETF" / "daily"
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")

        # 缓存目录
        if cache_dir is None:
            import os

            cache_dir = os.getenv("ETF_CACHE_DIR")
            if cache_dir is None:
                cache_dir = Path(__file__).parent.parent / ".cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _generate_cache_key(self, etf_codes, start_date, end_date):
        """生成缓存键（包含数据文件修改时间，确保数据更新后缓存失效）"""
        codes_str = "-".join(sorted(etf_codes)) if etf_codes else "all"

        # ✅ FIX: 加入数据目录最新修改时间，避免数据更新后返回旧缓存
        # 遍历所有 parquet 文件，取最新的 mtime
        try:
            parquet_files = list(self.data_dir.glob("*.parquet"))
            if parquet_files:
                latest_mtime = max(f.stat().st_mtime for f in parquet_files)
            else:
                # 兜底：如果没有parquet文件，用数据目录本身的mtime
                latest_mtime = self.data_dir.stat().st_mtime
        except (OSError, ValueError):
            # 异常时使用0，此时缓存会每次miss（安全降级）
            logger.warning(f"无法获取数据目录 {self.data_dir} 的修改时间，缓存将失效")
            latest_mtime = 0

        # 缓存键格式: codes_startdate_enddate_mtime
        key_str = f"{codes_str}_{start_date}_{end_date}_{int(latest_mtime)}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def load_ohlcv(
        self,
        etf_codes: Optional[list] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        加载 OHLCV 数据（前复权）+ 缓存加速

        Args:
            etf_codes: ETF 代码列表（如 ['510300', '510500']），None = 加载所有
            start_date: 开始日期 YYYY-MM-DD
            end_date: 结束日期 YYYY-MM-DD
            use_cache: 是否使用缓存（默认True）

        Returns:
            {
                'close': pd.DataFrame(index=date, columns=symbols),
                'high': pd.DataFrame,
                'low': pd.DataFrame,
                'open': pd.DataFrame,
                'volume': pd.DataFrame,
                'amount': pd.DataFrame (交易额, 可选 - 仅当原始数据包含amount列时)
            }

        注意：
        - 所有 NaN 保持原位，不填充
        - 使用 adj_close, adj_high, adj_low, adj_open（前复权）
        - 首次加载会建立缓存，后续加载从缓存读取（快43倍）
        """
        # 检查缓存
        if use_cache:
            cache_key = self._generate_cache_key(etf_codes, start_date, end_date)
            cache_file = self.cache_dir / f"ohlcv_{cache_key}.pkl"

            if cache_file.exists():
                logger.info(f"从缓存加载数据: {cache_file.name}")
                try:
                    with open(cache_file, "rb") as f:
                        return pickle.load(f)
                except Exception as e:
                    # 兼容不同numpy版本/路径变化导致的pickle反序列化失败
                    logger.warning(f"缓存读取失败，忽略并重建缓存: {e}")
        # 1. 扫描可用文件
        parquet_files = list(self.data_dir.glob("*_daily_*.parquet"))

        # 2. 过滤指定的 ETF（兼容有/无交易所后缀的code，例如 510300 vs 510300.SH）
        if etf_codes:
            filtered_files = []
            for code in etf_codes:
                code = str(code).strip()
                for f in parquet_files:
                    stem_prefix = f.stem.split("_")[0]  # e.g., 510300.SH
                    base = stem_prefix.split(".")[0]  # 510300
                    if (
                        base == code
                        or stem_prefix == code
                        or stem_prefix.startswith(f"{code}.")
                    ):
                        filtered_files.append(f)
            parquet_files = filtered_files

        logger.info(f"找到 {len(parquet_files)} 个 ETF 数据文件")

        # 3. 加载每个 ETF
        data_dict = {col: {} for col in ["close", "high", "low", "open", "volume", "amount"]}

        for file_path in parquet_files:
            code = file_path.stem.split("_")[0].split(".")[0]

            try:
                df = pd.read_parquet(file_path)
            except FileNotFoundError as e:
                raise FileNotFoundError(f"数据文件不存在: {file_path}") from e
            except Exception as e:
                raise RuntimeError(f"读取{code}失败: {file_path}, 错误: {e}") from e

            # 设置时间索引
            if "trade_date" not in df.columns:
                raise ValueError(f"{code} 缺少trade_date列")

            df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
            df.set_index("trade_date", inplace=True)

            # 时间范围过滤
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]

            # 验证必需列（volume列兼容vol/volume双写法）
            required_mapping = {
                "close": "adj_close",
                "high": "adj_high",
                "low": "adj_low",
                "open": "adj_open",
            }

            # 验证OHLC列
            for target_col, source_col in required_mapping.items():
                if source_col not in df.columns:
                    raise ValueError(f"{code} 缺少{source_col}列")
                data_dict[target_col][code] = df[source_col]

            # volume列特殊处理：优先vol，回退volume
            if "vol" in df.columns:
                data_dict["volume"][code] = df["vol"]
            elif "volume" in df.columns:
                data_dict["volume"][code] = df["volume"]
            else:
                raise ValueError(f"{code} 缺少vol或volume列")

            # amount列（交易额）：可选，用于 Amihud 等因子精度提升
            if "amount" in df.columns:
                data_dict["amount"][code] = df["amount"]

        # 4. 转换为 DataFrame（保留所有 NaN）
        result = {}
        for col in ["close", "high", "low", "open", "volume", "amount"]:
            if data_dict[col]:
                result[col] = pd.DataFrame(data_dict[col])

        # 5. 对齐所有 DataFrame 的索引
        if not result or "close" not in result:
            raise ValueError("没有加载到任何数据")

        all_dates = result["close"].index
        # 确保列按字母顺序排序，保证 VEC/BT 对齐
        sorted_cols = sorted(result["close"].columns)

        for col in result:
            result[col] = result[col].reindex(all_dates)
            # Some fields (e.g. amount) may have fewer ETFs than close;
            # only select columns that exist, then reindex to add missing as NaN
            available = [c for c in sorted_cols if c in result[col].columns]
            result[col] = result[col][available].reindex(columns=sorted_cols)

        logger.info(
            f"加载完成: {len(result['close'].columns)} ETFs × {len(result['close'])} 日期"
        )
        logger.info(
            f"日期范围: {result['close'].index[0]} 至 {result['close'].index[-1]}"
        )

        # 6. 数据契约验证
        from .data_contract import DataContract

        try:
            DataContract.validate_ohlcv(result)
            logger.info("✅ 数据契约验证通过")
        except ValueError as e:
            raise ValueError(f"数据质量不符合契约: {e}") from e

        # 7. 保存缓存
        if use_cache:
            logger.info(f"保存缓存: {cache_file.name}")
            with open(cache_file, "wb") as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

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
        return close_df.pct_change(periods)

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

    # ------------------------------------------------------------------
    # Non-OHLCV data loaders (fund_share, margin, fund_nav, FX)
    # ------------------------------------------------------------------

    @property
    def _etf_root(self) -> Path:
        """Return raw/ETF/ root dir (parent of daily/)."""
        return self.data_dir.parent

    @staticmethod
    def _ts_code_to_etf_code(ts_code: str) -> str:
        """Convert Tushare ts_code '159801.SZ' → bare code '159801'."""
        return ts_code.split(".")[0]

    def _get_trading_dates(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DatetimeIndex:
        """Load OHLCV trading calendar as reference DatetimeIndex."""
        ohlcv = self.load_ohlcv(start_date=start_date, end_date=end_date)
        return ohlcv["close"].index

    def load_fund_share(
        self,
        etf_codes: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        trading_dates: Optional[pd.DatetimeIndex] = None,
    ) -> pd.DataFrame:
        """Load fund share (份额) data for ETFs.

        Args:
            etf_codes: ETF codes to load (None = all available).
            start_date: Start date 'YYYY-MM-DD'.
            end_date: End date 'YYYY-MM-DD'.
            trading_dates: Reference trading calendar. If None, derived from
                OHLCV data.

        Returns:
            DataFrame with DatetimeIndex (trading dates), columns = ETF codes,
            values = fd_share (float). Forward-filled for gaps.
        """
        share_dir = self._etf_root / "fund_share"
        if not share_dir.exists():
            raise FileNotFoundError(f"fund_share 目录不存在: {share_dir}")

        parquet_files = sorted(share_dir.glob("fund_share_*.parquet"))
        series_dict: Dict[str, pd.Series] = {}

        for fp in parquet_files:
            # Extract code from filename: fund_share_159801.parquet → 159801
            code = fp.stem.replace("fund_share_", "").split(".")[0]
            if etf_codes and code not in etf_codes:
                continue

            df = pd.read_parquet(fp)
            if "trade_date" not in df.columns or "fd_share" not in df.columns:
                logger.warning(f"fund_share {code}: 缺少必需列, 跳过")
                continue

            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.set_index("trade_date").sort_index()
            df = df[~df.index.duplicated(keep="last")]

            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]

            series_dict[code] = df["fd_share"]

        if not series_dict:
            raise ValueError("没有加载到任何 fund_share 数据")

        result = pd.DataFrame(series_dict)

        # Align to trading calendar and forward fill
        if trading_dates is None:
            trading_dates = self._get_trading_dates(start_date, end_date)
        result = result.reindex(trading_dates).ffill()

        # Sort columns for reproducibility
        result = result[sorted(result.columns)]
        logger.info(
            f"fund_share 加载完成: {len(result.columns)} ETFs × {len(result)} 日期, "
            f"NaN率 {result.isna().mean().mean():.1%}"
        )
        return result

    def load_margin(
        self,
        field: str = "rzye",
        etf_codes: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        trading_dates: Optional[pd.DatetimeIndex] = None,
    ) -> pd.DataFrame:
        """Load margin trading data (融资融券).

        Args:
            field: Margin field to load. One of: rzye (融资余额), rqye (融券余额),
                rzmre (融资买入额), rqyl (融券余量), rzche (融资偿还额),
                rqchl (融券偿还量), rqmcl (融券卖出量), rzrqye (融资融券余额).
            etf_codes: ETF codes to filter (None = all available).
            start_date: Start date 'YYYY-MM-DD'.
            end_date: End date 'YYYY-MM-DD'.
            trading_dates: Reference trading calendar.

        Returns:
            DataFrame with DatetimeIndex, columns = ETF codes, values = field.
            Forward-filled. ETFs without margin data have NaN columns.
        """
        valid_fields = [
            "rzye", "rqye", "rzmre", "rqyl", "rzche", "rqchl", "rqmcl", "rzrqye",
        ]
        if field not in valid_fields:
            raise ValueError(f"无效字段 '{field}', 可选: {valid_fields}")

        margin_file = self._etf_root / "margin" / "margin_pool43_2020_now.parquet"
        if not margin_file.exists():
            raise FileNotFoundError(f"margin 文件不存在: {margin_file}")

        df = pd.read_parquet(margin_file)

        # trade_date is string 'YYYYMMDD'
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        df["etf_code"] = df["ts_code"].apply(self._ts_code_to_etf_code)

        if start_date:
            df = df[df["trade_date"] >= start_date]
        if end_date:
            df = df[df["trade_date"] <= end_date]

        # Pivot to wide format: date x etf_code
        pivot = df.pivot_table(
            index="trade_date", columns="etf_code", values=field, aggfunc="last"
        )
        pivot = pivot.sort_index()

        if etf_codes:
            # Keep only requested codes; missing ones become NaN columns
            pivot = pivot.reindex(columns=sorted(etf_codes))
        else:
            pivot = pivot[sorted(pivot.columns)]

        # Align to trading calendar and forward fill
        if trading_dates is None:
            trading_dates = self._get_trading_dates(start_date, end_date)
        pivot = pivot.reindex(trading_dates).ffill()

        logger.info(
            f"margin({field}) 加载完成: {len(pivot.columns)} ETFs × {len(pivot)} 日期, "
            f"NaN率 {pivot.isna().mean().mean():.1%}"
        )
        return pivot

    def load_fund_nav(
        self,
        field: str = "unit_nav",
        etf_codes: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        trading_dates: Optional[pd.DatetimeIndex] = None,
    ) -> pd.DataFrame:
        """Load fund NAV data aligned by ann_date (publication date).

        CRITICAL: Uses ann_date (T+1 publication) as index, NOT trade_date,
        to prevent lookahead bias. The NAV for trade_date T is published on
        ann_date T+1, so using ann_date ensures we only see data that was
        actually available at each point in time.

        Args:
            field: NAV field to load. One of: unit_nav, adj_nav, accum_nav,
                total_netasset, net_asset.
            etf_codes: ETF codes to load (None = all available).
            start_date: Start date 'YYYY-MM-DD'.
            end_date: End date 'YYYY-MM-DD'.
            trading_dates: Reference trading calendar.

        Returns:
            DataFrame with DatetimeIndex (aligned to ann_date), columns = ETF
            codes, values = requested NAV field. Forward-filled.
        """
        valid_fields = ["unit_nav", "adj_nav", "accum_nav", "total_netasset", "net_asset"]
        if field not in valid_fields:
            raise ValueError(f"无效字段 '{field}', 可选: {valid_fields}")

        nav_dir = self._etf_root / "fund_nav"
        if not nav_dir.exists():
            raise FileNotFoundError(f"fund_nav 目录不存在: {nav_dir}")

        parquet_files = sorted(nav_dir.glob("fund_nav_*.parquet"))
        series_dict: Dict[str, pd.Series] = {}

        for fp in parquet_files:
            code = fp.stem.replace("fund_nav_", "").split(".")[0]
            if etf_codes and code not in etf_codes:
                continue

            df = pd.read_parquet(fp)
            if "ann_date" not in df.columns:
                logger.warning(f"fund_nav {code}: 缺少 ann_date 列, 跳过")
                continue
            if field not in df.columns:
                logger.warning(f"fund_nav {code}: 缺少 {field} 列, 跳过")
                continue

            # Use ann_date as index to prevent lookahead
            df["ann_date"] = pd.to_datetime(df["ann_date"], format="%Y%m%d")
            df = df.set_index("ann_date").sort_index()
            df = df[~df.index.duplicated(keep="last")]

            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]

            series_dict[code] = df[field]

        if not series_dict:
            raise ValueError("没有加载到任何 fund_nav 数据")

        result = pd.DataFrame(series_dict)

        # Align to trading calendar and forward fill
        if trading_dates is None:
            trading_dates = self._get_trading_dates(start_date, end_date)
        result = result.reindex(trading_dates).ffill()

        result = result[sorted(result.columns)]
        logger.info(
            f"fund_nav({field}) 加载完成: {len(result.columns)} ETFs × {len(result)} 日期, "
            f"NaN率 {result.isna().mean().mean():.1%}"
        )
        return result

    def load_fx(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        trading_dates: Optional[pd.DatetimeIndex] = None,
    ) -> pd.DataFrame:
        """Load FX rate data (USDCNH, USDCNY, HKDCNY).

        Sources:
        - USDCNH: Tushare fx_daily bid_close (offshore CNH rate)
        - USDCNY: BOC中间价 (onshore CNY rate, 美元 column / 100)
        - HKDCNY: BOC中间价 (港元 column / 100)

        Args:
            start_date: Start date 'YYYY-MM-DD'.
            end_date: End date 'YYYY-MM-DD'.
            trading_dates: Reference trading calendar.

        Returns:
            DataFrame with DatetimeIndex, columns = ['USDCNH', 'USDCNY', 'HKDCNY'].
            Forward-filled for non-trading days (FX and A-share calendars differ).
        """
        fx_dir = self._etf_root / "fx"
        if not fx_dir.exists():
            raise FileNotFoundError(f"FX 目录不存在: {fx_dir}")

        series_dict: Dict[str, pd.Series] = {}

        # 1. USDCNH from Tushare fx_daily
        usdcnh_file = fx_dir / "usdcnh_daily.parquet"
        if usdcnh_file.exists():
            df = pd.read_parquet(usdcnh_file)
            df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
            df = df.set_index("trade_date").sort_index()
            df = df[~df.index.duplicated(keep="last")]
            if "bid_close" in df.columns:
                series_dict["USDCNH"] = df["bid_close"]
            else:
                logger.warning("usdcnh_daily.parquet 缺少 bid_close 列")
        else:
            logger.warning(f"USDCNH 文件不存在: {usdcnh_file}")

        # 2. BOC FX (USDCNY, HKDCNY)
        boc_file = fx_dir / "boc_fx_daily.parquet"
        if boc_file.exists():
            df = pd.read_parquet(boc_file)
            df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
            df = df.set_index("trade_date").sort_index()
            df = df[~df.index.duplicated(keep="last")]
            # BOC rates are per 100 foreign currency → divide by 100
            if "美元" in df.columns:
                series_dict["USDCNY"] = df["美元"] / 100.0
            if "港元" in df.columns:
                series_dict["HKDCNY"] = df["港元"] / 100.0
        else:
            logger.warning(f"BOC FX 文件不存在: {boc_file}")

        if not series_dict:
            raise ValueError("没有加载到任何 FX 数据")

        result = pd.DataFrame(series_dict)

        # Date filter
        if start_date:
            result = result[result.index >= start_date]
        if end_date:
            result = result[result.index <= end_date]

        # Align to trading calendar and forward fill
        if trading_dates is None:
            trading_dates = self._get_trading_dates(start_date, end_date)
        result = result.reindex(trading_dates).ffill()

        logger.info(
            f"FX 加载完成: {list(result.columns)}, {len(result)} 日期, "
            f"NaN率 {result.isna().mean().mean():.1%}"
        )
        return result
