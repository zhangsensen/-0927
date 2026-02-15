"""
融资融券数据爬虫

数据来源：AkShare → 上交所/深交所融资融券明细
- 日频，按个券粒度（含ETF）
- 历史深度：2010年至今
- 可直接用于因子回测

关键函数:
  - stock_margin_detail_sse(date): 上交所融资融券明细
  - stock_margin_detail_szse(date): 深交所融资融券明细
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class MarginDataCrawler:
    """融资融券数据爬虫 - 基于AkShare"""

    def __init__(self, output_dir: str = "raw/ETF/margin"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_trading_dates(
        self, start_date: str, end_date: str
    ) -> List[str]:
        """
        生成交易日列表（粗略版：跳过周末）
        实际运行中如果某天无数据，AkShare会返回空/报错，我们直接跳过
        """
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
        dates = []
        current = start
        while current <= end:
            # 跳过周末
            if current.weekday() < 5:
                dates.append(current.strftime("%Y%m%d"))
            current += timedelta(days=1)
        return dates

    def fetch_sse_margin_detail(self, date: str) -> pd.DataFrame:
        """
        获取上交所某日的融资融券明细

        Args:
            date: 日期，格式 YYYYMMDD

        Returns:
            DataFrame 包含各标的证券的融资融券数据
        """
        from akshare import stock_margin_detail_sse

        try:
            df = stock_margin_detail_sse(date=date)
            if df is not None and not df.empty:
                df["日期"] = date
                df["交易所"] = "SSE"
                return df
        except Exception as e:
            # 非交易日/节假日返回空，不算错误
            error_msg = str(e)
            if "holiday" in error_msg.lower() or "不是交易日" in error_msg:
                pass  # 静默跳过
            else:
                logger.debug(f"SSE {date}: {e}")
        return pd.DataFrame()

    def fetch_szse_margin_detail(self, date: str) -> pd.DataFrame:
        """
        获取深交所某日的融资融券明细

        Args:
            date: 日期，格式 YYYYMMDD

        Returns:
            DataFrame 包含各标的证券的融资融券数据
        """
        from akshare import stock_margin_detail_szse

        try:
            df = stock_margin_detail_szse(date=date)
            if df is not None and not df.empty:
                df["日期"] = date
                df["交易所"] = "SZSE"
                return df
        except Exception as e:
            error_msg = str(e)
            if "holiday" in error_msg.lower() or "不是交易日" in error_msg:
                pass
            else:
                logger.debug(f"SZSE {date}: {e}")
        return pd.DataFrame()

    def fetch_date_range(
        self,
        start_date: str = "20200101",
        end_date: Optional[str] = None,
        exchange: str = "both",
        delay: float = 1.0,
        etf_codes: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        批量获取日期范围内的融资融券明细

        Args:
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD，默认今天
            exchange: 'sse', 'szse', 或 'both'
            delay: 请求间隔秒数
            etf_codes: 过滤ETF代码列表，None表示返回全部

        Returns:
            合并后的DataFrame
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")

        dates = self._get_trading_dates(start_date, end_date)
        total = len(dates)
        logger.info(
            f"准备抓取 {total} 个交易日的融资融券数据 "
            f"({start_date} ~ {end_date}, exchange={exchange})"
        )

        all_dfs = []
        success_count = 0

        for i, date in enumerate(dates):
            if i % 50 == 0:
                logger.info(f"Progress: {i}/{total} ({success_count} days with data)")

            dfs = []

            if exchange in ("sse", "both"):
                df_sse = self.fetch_sse_margin_detail(date)
                if not df_sse.empty:
                    dfs.append(df_sse)

            if exchange in ("szse", "both"):
                df_szse = self.fetch_szse_margin_detail(date)
                if not df_szse.empty:
                    dfs.append(df_szse)

            if dfs:
                day_df = pd.concat(dfs, ignore_index=True)

                # 如果指定了ETF代码，过滤
                if etf_codes:
                    code_col = "标的证券代码" if "标的证券代码" in day_df.columns else None
                    if code_col is None:
                        # 尝试其他可能的列名
                        for col in day_df.columns:
                            if "代码" in col:
                                code_col = col
                                break

                    if code_col:
                        day_df = day_df[day_df[code_col].astype(str).isin(etf_codes)]

                if not day_df.empty:
                    all_dfs.append(day_df)
                    success_count += 1

            # 控制请求频率
            time.sleep(delay)

        if not all_dfs:
            logger.warning("未获取到任何融资融券数据")
            return pd.DataFrame()

        result = pd.concat(all_dfs, ignore_index=True)
        logger.info(
            f"抓取完成: {success_count}/{total} 天有数据, 共 {len(result)} 条记录"
        )
        return result

    def save(self, df: pd.DataFrame, filename: str = "margin_detail.parquet"):
        """保存数据为Parquet"""
        if df.empty:
            logger.warning("数据为空，跳过保存")
            return

        output_path = self.output_dir / filename
        df.to_parquet(output_path, index=False)
        logger.info(f"数据已保存: {output_path} ({len(df)} 条)")

    def incremental_update(
        self,
        exchange: str = "both",
        etf_codes: Optional[List[str]] = None,
    ):
        """
        增量更新：检查已有数据的最新日期，从那天开始拉取
        """
        filename = "margin_detail.parquet"
        filepath = self.output_dir / filename

        if filepath.exists():
            existing = pd.read_parquet(filepath)
            if "日期" in existing.columns and not existing.empty:
                last_date = str(existing["日期"].max())
                # 从最后日期的下一天开始
                start = datetime.strptime(last_date, "%Y%m%d") + timedelta(days=1)
                start_date = start.strftime("%Y%m%d")
                logger.info(f"增量更新: 从 {start_date} 开始")
            else:
                start_date = "20200101"
                existing = pd.DataFrame()
        else:
            start_date = "20200101"
            existing = pd.DataFrame()

        new_data = self.fetch_date_range(
            start_date=start_date,
            exchange=exchange,
            etf_codes=etf_codes,
        )

        if not new_data.empty:
            if not existing.empty:
                combined = pd.concat([existing, new_data], ignore_index=True)
                # 去重
                combined = combined.drop_duplicates()
            else:
                combined = new_data

            self.save(combined, filename)
        else:
            logger.info("没有新数据需要更新")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    crawler = MarginDataCrawler()

    # === 测试: 先拉一天数据验证接口可用性 ===
    print("\n=== 测试: 获取单日融资融券明细 ===")

    # 测试上交所
    test_date = "20260211"  # 最近交易日
    print(f"\n--- SSE {test_date} ---")
    df_sse = crawler.fetch_sse_margin_detail(test_date)
    if not df_sse.empty:
        print(f"获取到 {len(df_sse)} 条记录")
        print(f"列名: {list(df_sse.columns)}")

        # 筛选ETF (5开头的是上交所ETF)
        code_col = [c for c in df_sse.columns if "代码" in c][0]
        etf_mask = df_sse[code_col].astype(str).str.startswith("5")
        df_etf = df_sse[etf_mask]
        print(f"\n其中ETF: {len(df_etf)} 条")
        if not df_etf.empty:
            print(df_etf.head(10).to_string())
    else:
        print("无数据 (可能非交易日)")

    # 测试深交所
    print(f"\n--- SZSE {test_date} ---")
    df_szse = crawler.fetch_szse_margin_detail(test_date)
    if not df_szse.empty:
        print(f"获取到 {len(df_szse)} 条记录")
        code_col = [c for c in df_szse.columns if "代码" in c][0]
        etf_mask = df_szse[code_col].astype(str).str.startswith("1")
        df_etf = df_szse[etf_mask]
        print(f"\n其中ETF: {len(df_etf)} 条")
        if not df_etf.empty:
            print(df_etf.head(10).to_string())
    else:
        print("无数据 (可能非交易日)")
