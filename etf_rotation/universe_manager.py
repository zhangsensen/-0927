"""ETF宇宙月度锁定管理"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class ETFUniverseManager:
    """ETF宇宙管理器：月度锁定+流动性过滤"""

    def __init__(self, config_path: str):
        """
        初始化宇宙管理器

        Args:
            config_path: ETF配置文件路径（etf_config.yaml）
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.etf_data_path = Path("raw/ETF/daily")

    def get_monthly_universe(
        self, trade_date: str, min_amount_20d: float = 20000000
    ) -> list[str]:
        """
        月度宇宙锁定：流动性+数据完整性

        Args:
            trade_date: 交易日期（YYYYMMDD格式）
            min_amount_20d: 20日均成交额阈值（元），默认2000万

        Returns:
            符合条件的ETF代码列表
        """
        eligible = []

        # 遍历所有ETF池
        for pool_name, pool_config in self.config.get("etf_list", {}).items():
            if "etfs" not in pool_config:
                continue

            for etf in pool_config["etfs"]:
                code = etf["code"]
                if self._check_eligibility(code, trade_date, min_amount_20d):
                    eligible.append(code)
                    logger.debug(f"✅ {code} 通过准入检查")
                else:
                    logger.debug(f"❌ {code} 未通过准入检查")

        logger.info(
            f"月度宇宙锁定完成：{len(eligible)}/{self._count_total_etfs()} 只ETF"
        )
        return eligible

    def _check_eligibility(self, code: str, date: str, min_amount_20d: float) -> bool:
        """
        准入检查：20日均额>阈值 + 数据完整性

        Args:
            code: ETF代码
            date: 检查日期（YYYYMMDD）
            min_amount_20d: 20日均成交额阈值（元）

        Returns:
            是否通过准入检查
        """
        # 查找ETF文件
        etf_files = list(self.etf_data_path.glob(f"{code}_daily_*.parquet"))
        if not etf_files:
            logger.debug(f"{code}: 未找到数据文件")
            return False

        try:
            # 加载数据
            df = pd.read_parquet(etf_files[0])
            df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")

            # 过滤到检查日期
            check_date = pd.to_datetime(date, format="%Y%m%d")
            df = df[df["trade_date"] <= check_date]

            if len(df) < 20:
                logger.debug(f"{code}: 数据不足20天")
                return False

            # 取最近20天
            recent = df.tail(20)

            # 1. 流动性检查：20日均额（amount单位：千元）
            avg_amount = recent["amount"].mean() * 1000  # 转为元
            if avg_amount < min_amount_20d:
                logger.debug(
                    f"{code}: 流动性不足 {avg_amount/1e6:.2f}M < {min_amount_20d/1e6:.2f}M"
                )
                return False

            # 2. 数据完整性检查：无缺失值
            if recent[["close", "vol", "amount"]].isnull().any().any():
                logger.debug(f"{code}: 数据存在缺失值")
                return False

            # 3. 非停牌检查：成交量>0
            if (recent["vol"] == 0).any():
                logger.debug(f"{code}: 存在停牌日")
                return False

            return True

        except Exception as e:
            logger.warning(f"{code}: 准入检查异常 - {e}")
            return False

    def _count_total_etfs(self) -> int:
        """统计配置中ETF总数"""
        total = 0
        for pool_config in self.config.get("etf_list", {}).values():
            if "etfs" in pool_config:
                total += len(pool_config["etfs"])
        return total
