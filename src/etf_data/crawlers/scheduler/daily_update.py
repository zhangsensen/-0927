"""
爬虫调度器 - 每日定时更新ETF数据
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from etf_data.crawlers.sources.eastmoney_crawler import EastmoneyETFCrawler
from etf_data.crawlers.sources.eastmoney_detail_crawler import EastmoneyDetailCrawler
from etf_data.crawlers.sources.shares_crawler import TushareSharesCrawler
from etf_data.crawlers.sources.tushare_flow_crawler import TushareFlowCrawler

logger = logging.getLogger(__name__)


class DailyDataUpdater:
    """每日数据更新器"""

    def __init__(self, output_dir: str = "raw/ETF"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化爬虫
        self.eastmoney_crawler = EastmoneyETFCrawler()
        self.detail_crawler = EastmoneyDetailCrawler()
        self.shares_crawler = TushareSharesCrawler()
        self.flow_crawler = TushareFlowCrawler()

    def update_etf_list(self) -> bool:
        """更新ETF列表"""
        return self._update_etf_list()


    def _update_etf_list(self) -> bool:
        logger.info("开始更新ETF列表...")

        df = self.eastmoney_crawler.get_etf_list()
        if df.empty:
            logger.error("ETF列表更新失败")
            return False

        output_path = self.output_dir / "etf_list.parquet"
        df.to_parquet(output_path, index=False)
        logger.info(f"ETF列表已更新: {len(df)} 只，保存至 {output_path}")
        return True

    def update_realtime_quotes(self, etf_codes: list = None) -> bool:
        """
        更新ETF实时行情

        Args:
            etf_codes: 要更新的ETF代码列表，为None时更新全部
        """
        logger.info("开始更新ETF实时行情...")

        if etf_codes is None:
            # 从已有的列表文件读取
            list_file = self.output_dir / "etf_list.parquet"
            if not list_file.exists():
                logger.error("ETF列表文件不存在，请先运行update_etf_list")
                return False

            import pandas as pd

            etf_list = pd.read_parquet(list_file)
            etf_codes = etf_list["code"].tolist()[:100]  # 先更新前100只

        # 分批获取（每次最多100只）
        batch_size = 100
        all_quotes = []

        for i in range(0, len(etf_codes), batch_size):
            batch = etf_codes[i : i + batch_size]
            logger.info(f"获取第 {i // batch_size + 1} 批 ({len(batch)} 只)...")

            df = self.eastmoney_crawler.get_etf_realtime_quote(batch)
            if not df.empty:
                all_quotes.append(df)

        if all_quotes:
            import pandas as pd

            combined = pd.concat(all_quotes, ignore_index=True)

            # 按日期保存
            today = datetime.now().strftime("%Y%m%d")
            output_path = self.output_dir / "realtime" / f"quotes_{today}.parquet"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            combined.to_parquet(output_path, index=False)
            logger.info(f"实时行情已更新: {len(combined)} 只，保存至 {output_path}")
            return True

        return False

    def update_etf_snapshot(self) -> bool:
        """
        日频ETF快照采集 (高价值)

        通过 AkShare fund_etf_spot_em() 获取全量ETF快照，包含:
        - IOPV实时估值 / 基金折价率
        - 主力/超大单/大单/中单/小单 净流入
        - 最新份额 (日频)
        - 委比/外盘/内盘
        - 量比/换手率

        每日盘后运行，按日期存储为快照文件
        """
        logger.info("开始采集ETF日频快照...")

        try:
            from akshare import fund_etf_spot_em

            df = fund_etf_spot_em()
            if df is None or df.empty:
                logger.error("ETF快照获取失败: 返回空数据")
                return False

            # 按日期保存
            today = datetime.now().strftime("%Y%m%d")
            save_dir = self.output_dir / "snapshots"
            save_dir.mkdir(parents=True, exist_ok=True)

            output_path = save_dir / f"snapshot_{today}.parquet"
            df.to_parquet(output_path, index=False)
            logger.info(
                f"ETF快照已保存: {len(df)} 只, "
                f"包含字段: {list(df.columns[:8])}... "
                f"→ {output_path}"
            )
            return True

        except Exception as e:
            logger.error(f"ETF快照采集失败: {e}")
            return False

    def update_margin_data(self, date: str = None) -> bool:
        """
        日频融资融券数据采集 (高价值)

        通过 AkShare stock_margin_detail_sse/szse 获取当日融资融券明细，包含:
        - 按个券粒度 (覆盖ETF)
        - 融资余额/买入额/偿还额
        - 融券余量/卖出量/偿还量

        Args:
            date: 日期YYYYMMDD, 默认今天
        """
        logger.info("开始采集融资融券数据...")

        if date is None:
            date = datetime.now().strftime("%Y%m%d")

        try:
            from etf_data.crawlers.sources.margin_crawler import MarginDataCrawler

            margin_crawler = MarginDataCrawler(
                output_dir=str(self.output_dir / "margin")
            )

            # 获取上交所
            df_sse = margin_crawler.fetch_sse_margin_detail(date)
            # 获取深交所
            df_szse = margin_crawler.fetch_szse_margin_detail(date)

            dfs = []
            if not df_sse.empty:
                dfs.append(df_sse)
            if not df_szse.empty:
                dfs.append(df_szse)

            if not dfs:
                logger.warning(f"{date}: 未获取到融资融券数据 (可能非交易日)")
                return False

            combined = pd.concat(dfs, ignore_index=True)

            # 过滤出ETF (5开头=上交所ETF, 1开头=深交所ETF)
            code_col = None
            for col in combined.columns:
                if "代码" in col:
                    code_col = col
                    break

            if code_col:
                etf_mask = (
                    combined[code_col].astype(str).str.startswith("5")
                    | combined[code_col].astype(str).str.startswith("1")
                )
                etf_df = combined[etf_mask].copy()
            else:
                etf_df = combined

            # 保存
            save_dir = self.output_dir / "margin"
            save_dir.mkdir(parents=True, exist_ok=True)

            output_path = save_dir / f"margin_{date}.parquet"
            etf_df.to_parquet(output_path, index=False)
            logger.info(
                f"融资融券数据已保存: {len(etf_df)} 只ETF → {output_path}"
            )
            return True

        except Exception as e:
            logger.error(f"融资融券数据采集失败: {e}")
            return False

    def update_shares_history(self, etf_codes: list = None) -> bool:
        """
        更新ETF历史份额数据 (Tushare)
        """
        logger.info("开始更新ETF历史份额 (Tushare)...")
        if not self.shares_crawler.pro:
            logger.warning("Tushare未配置，跳过份额更新")
            return False

        if etf_codes is None:
            # 读取列表
            list_file = self.output_dir / "etf_list.parquet"
            if not list_file.exists():
                return False
            etf_list = pd.read_parquet(list_file)
            etf_codes = etf_list["code"].tolist()

        total = len(etf_codes)
        for i, code in enumerate(etf_codes, 1):
            if i % 10 == 0:
                logger.info(f"份额更新进度: {i}/{total}")
            self.shares_crawler.update_shares(str(code))
        
        return True

    def update_flow_backfill(self, etf_codes: list = None) -> bool:
        """
        补全ETF历史资金流向 (Tushare)
        """
        logger.info("开始补全ETF历史资金流向 (Tushare)...")
        if not self.flow_crawler.pro:
            logger.warning("Tushare未配置，跳过资金流补全")
            return False

        if etf_codes is None:
            list_file = self.output_dir / "etf_list.parquet"
            if not list_file.exists():
                return False
            etf_list = pd.read_parquet(list_file)
            etf_codes = etf_list["code"].tolist()

        total = len(etf_codes)
        for i, code in enumerate(etf_codes, 1):
            if i % 10 == 0:
                logger.info(f"资金流补全进度: {i}/{total}")
            self.flow_crawler.update_flow_history(str(code))
        
        return True

    def run_daily_update(self):
        """执行每日完整更新"""
        logger.info("=" * 60)
        logger.info("开始每日数据更新")
        logger.info("=" * 60)

        # 1. 更新ETF列表
        self.update_etf_list()

        # 2. 更新历史份额 (P0)
        self.update_shares_history()
        
        # 3. 补全历史资金流 (P0)
        self.update_flow_backfill()

        # 4. 更新实时行情
        self.update_realtime_quotes()

        # 5. [高价值] ETF日频快照 (IOPV/折溢价/资金流/份额)
        self.update_etf_snapshot()

        # 6. [高价值] 融资融券数据 (日频杠杆情绪)
        self.update_margin_data()

        logger.info("=" * 60)
        logger.info("每日数据更新完成")
        logger.info("=" * 60)


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # 运行更新
    updater = DailyDataUpdater()
    updater.run_daily_update()

