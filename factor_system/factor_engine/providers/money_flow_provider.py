"""
资金流数据提供者 - 统一口径、标准化、生成约束掩码

Linus准则：
- 固定schema，统一时区
- 向量化处理，禁止循环
- 可mock、可回放
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import zscore


class MoneyFlowProvider:
    """资金流数据提供者"""

    def __init__(self, data_dir: Path, snapshot_id: Optional[str] = None,
                 input_unit: str = "wan_yuan", output_unit: str = "yuan",
                 data_source: str = "tushare_pro",
                 enforce_t_plus_1: bool = True):
        self.data_dir = Path(data_dir)
        self.snapshot_id = snapshot_id or self._generate_snapshot_id()
        # 原始数据默认来自 TuShare，金额口径为万元；输出统一为元
        self.input_unit = input_unit
        self.output_unit = output_unit
        self.data_source = data_source
        # 【时序安全】资金流数据T+1发布，强制滞后1天
        self.enforce_t_plus_1 = enforce_t_plus_1

    def _generate_snapshot_id(self) -> str:
        """生成数据快照ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"snapshot_{timestamp}"

    def load_money_flow(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        加载资金流数据并执行标准化

        Returns:
            DataFrame with columns:
            - trade_date (index)
            - buy_small_amount, sell_small_amount, ...
            - turnover_amount (分母)
            - main_net, retail_net, total_net
            - tradability_mask (0/1)
            - snapshot_id
        """
        # 1. 加载原始数据 - 支持多种文件名格式
        possible_names = [
            f"{symbol}_money_flow.parquet",  # 标准格式
            f"{symbol}_moneyflow.parquet",  # TuShare实际格式
            f"{symbol}.parquet",            # 简化格式
        ]

        file_path = None
        for name in possible_names:
            path = self.data_dir / name
            if path.exists():
                file_path = path
                break

        if not file_path:
            raise FileNotFoundError(f"Money flow data not found for {symbol}. Tried: {possible_names}")

        df = pd.read_parquet(file_path)

        # 处理日期格式兼容性：trade_date可能是字符串或datetime格式
        if df["trade_date"].dtype == 'object':
            # 字符串格式，需要转换为datetime进行比较
            start_date_str = start_date.replace('-', '')
            end_date_str = end_date.replace('-', '')
            df = df[
                (df["trade_date"] >= start_date_str) & (df["trade_date"] <= end_date_str)
            ].copy()
        else:
            # datetime格式，直接比较
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df[
                (df["trade_date"] >= start_dt) & (df["trade_date"] <= end_dt)
            ].copy()

        if df.empty:
            raise ValueError(f"No data found for {symbol} in date range")

        # 【修复】确保索引是DatetimeIndex并按时间升序排列
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df.set_index("trade_date", inplace=True)
        df = df.sort_index()  # 确保时间序列升序

        # 2. 字段映射（向量化）
        df = self._map_fields(df)

        # 3. 单位标准化（万元 -> 元）
        df = self._normalize_units(df)

        # 4. 计算衍生字段（向量化）
        df = self._calculate_derived_fields(df)

        # 5. 生成tradability_mask（向量化）
        df = self._generate_tradability_mask(df)

        # 6. 标准化处理（向量化）
        df = self._standardize_factors(df)

        # 7. 【时序安全】强制T+1滞后处理
        if self.enforce_t_plus_1:
            df = self._apply_t_plus_1_lag(df)

        # 8. 加载并合并价格数据（用于Flow_Price_Divergence等因子）
        df = self._load_and_merge_price_data(df, symbol, start_date, end_date)

        # 9. 添加元信息
        df["snapshot_id"] = self.snapshot_id
        df["data_source"] = self.data_source
        df["value_unit"] = self.output_unit
        df["temporal_safe"] = self.enforce_t_plus_1

        return df

    def _map_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """字段映射 - 向量化"""
        rename_map = {
            "buy_sm_amount": "buy_small_amount",
            "sell_sm_amount": "sell_small_amount",
            "buy_md_amount": "buy_medium_amount",
            "sell_md_amount": "sell_medium_amount",
            "buy_lg_amount": "buy_large_amount",
            "sell_lg_amount": "sell_large_amount",
            "buy_elg_amount": "buy_super_large_amount",
            "sell_elg_amount": "sell_super_large_amount",
        }
        return df.rename(columns=rename_map)

    def _calculate_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算衍生字段 - 向量化"""
        # 成交额（分母，单位：与output_unit一致，默认元）
        df["turnover_amount"] = (
            df["buy_small_amount"]
            + df["sell_small_amount"]
            + df["buy_medium_amount"]
            + df["sell_medium_amount"]
            + df["buy_large_amount"]
            + df["sell_large_amount"]
            + df["buy_super_large_amount"]
            + df["sell_super_large_amount"]
        )

        # 主力净额
        df["main_net"] = (
            df["buy_large_amount"]
            + df["buy_super_large_amount"]
            - df["sell_large_amount"]
            - df["sell_super_large_amount"]
        )

        # 散户净额
        df["retail_net"] = df["buy_small_amount"] - df["sell_small_amount"]

        # 总净额
        df["total_net"] = df["net_mf_amount"]  # 已有字段

        return df

    def _normalize_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化金额单位到指定输出单位（默认元）。
        输入默认来自 TuShare（金金额单位=万元），统一乘以10000。

        仅对金额相关列进行转换。
        """
        amount_cols = [
            "buy_small_amount",
            "sell_small_amount",
            "buy_medium_amount",
            "sell_medium_amount",
            "buy_large_amount",
            "sell_large_amount",
            "buy_super_large_amount",
            "sell_super_large_amount",
            "net_mf_amount",
        ]

        if self.input_unit == self.output_unit:
            return df

        # 仅支持从万元->元的标准路径，其他路径按需扩展
        if self.input_unit == "wan_yuan" and self.output_unit == "yuan":
            for col in amount_cols:
                if col in df.columns:
                    df[col] = df[col] * 10000.0
            return df

        # 未识别的单位转换，直接抛错避免悄然错误
        raise ValueError(f"Unsupported unit normalization: {self.input_unit} -> {self.output_unit}")

    def _generate_tradability_mask(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成可交易性掩码 - 向量化

        mask=0 条件：
        - 成交额极低（<1%分位）
        """
        mask = np.ones(len(df), dtype=int)

        # 成交额极低（<1%分位）
        if len(df) > 0:
            turnover_threshold = df["turnover_amount"].quantile(0.01)
            mask &= (df["turnover_amount"] >= turnover_threshold).astype(int)

        df["tradability_mask"] = mask
        return df

    def _standardize_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化处理 - 向量化

        流程：winsorize(1%,99%) -> zscore
        """
        factor_cols = [
            "main_net",
            "retail_net",
            "total_net",
            "buy_large_amount",
            "sell_large_amount",
            "buy_super_large_amount",
            "sell_super_large_amount",
        ]

        for col in factor_cols:
            if col in df.columns and len(df) > 0:
                # Winsorize
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)
                df[f"{col}_winsorized"] = df[col].clip(lower, upper)

                # Z-score
                try:
                    df[f"{col}_zscore"] = zscore(
                        df[f"{col}_winsorized"], nan_policy="omit"
                    )
                except Exception:
                    df[f"{col}_zscore"] = 0.0

        return df

    def _apply_t_plus_1_lag(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【时序安全】T+1滞后处理 - 向量化
        
        资金流数据在T日收盘后才发布，因此T日数据只能在T+1日使用
        对所有资金流字段执行shift(1)
        """
        # 需要滞后的资金流字段
        money_flow_cols = [
            # 原始字段
            "buy_small_amount", "sell_small_amount",
            "buy_medium_amount", "sell_medium_amount",
            "buy_large_amount", "sell_large_amount",
            "buy_super_large_amount", "sell_super_large_amount",
            "turnover_amount", "net_mf_amount",
            # 衍生字段
            "main_net", "retail_net", "total_net",
            # 标准化字段
        ]
        
        # 添加所有winsorized和zscore字段
        money_flow_cols.extend([col for col in df.columns if "_winsorized" in col])
        money_flow_cols.extend([col for col in df.columns if "_zscore" in col])
        
        # 向量化滞后处理
        for col in money_flow_cols:
            if col in df.columns:
                df[col] = df[col].shift(1)
        
        return df

    def _load_and_merge_price_data(self, df: pd.DataFrame, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        加载价格数据并与资金流数据合并

        用于Flow_Price_Divergence等需要价格数据的因子计算
        """
        try:
            # 1. 查找价格数据文件
            price_file = self.data_dir.parent / f"{symbol}.parquet"

            if not price_file.exists():
                # 如果在SH目录没找到，尝试其他目录
                possible_paths = [
                    self.data_dir.parent.parent / "raw" / "SH" / f"{symbol}.parquet",
                    self.data_dir.parent.parent / "raw" / "SZ" / f"{symbol}.parquet",
                    self.data_dir.parent.parent / "raw" / "A股" / symbol / f"{symbol}_1day_*.parquet",
                ]

                for path in possible_paths:
                    if path.exists():
                        price_file = path
                        break
                else:
                    # 未找到价格数据，返回原始资金流数据
                    print(f"⚠️ 未找到{symbol}的价格数据，跳过价格数据合并")
                    return df

            # 2. 加载价格数据
            price_data = pd.read_parquet(price_file)

            # 3. 标准化价格数据格式
            if 'datetime' in price_data.columns:
                price_data['datetime'] = pd.to_datetime(price_data['datetime'])
                price_data.set_index('datetime', inplace=True)
            elif 'trade_date' in price_data.columns:
                price_data['trade_date'] = pd.to_datetime(price_data['trade_date'])
                price_data.set_index('trade_date', inplace=True)

            # 4. 重采样为日线（如果是分钟级数据）
            if len(price_data) > 500:  # 可能是分钟级数据
                price_daily = price_data.resample('D').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum',
                    'turnover': 'sum'
                }).dropna()
            else:
                price_daily = price_data.copy()

            # 5. 过滤时间范围
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            price_daily = price_daily[(price_daily.index >= start_dt) & (price_daily.index <= end_dt)]

            # 6. 合并数据
            merged_df = df.join(price_daily[['open', 'high', 'low', 'close', 'volume', 'turnover']], how='left')

            print(f"✅ 成功合并{symbol}价格数据: 价格数据{len(price_daily)}天, 合并后{len(merged_df)}天")

            return merged_df

        except Exception as e:
            print(f"⚠️ 加载{symbol}价格数据失败: {e}")
            return df

    def freeze_signal_at_1430(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        14:30信号冻结 - 向量化

        实盘执行：T+1开盘价或开盘加权5min
        """
        # 信号列统一后缀_signal_1430
        signal_cols = [col for col in df.columns if "_zscore" in col]
        for col in signal_cols:
            df[f"{col}_signal_1430"] = df[col].shift(1)  # T+1执行

        return df
