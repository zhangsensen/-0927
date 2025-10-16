#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF横截面因子计算模块
提供ETF横截面分析所需的各种因子
"""

import numpy as np
import pandas as pd
import talib
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

from factor_system.factor_engine.providers.etf_cross_section_provider import ETFCrossSectionDataManager
from factor_system.factor_engine.providers.etf_cross_section_storage import ETFCrossSectionStorage

logger = logging.getLogger(__name__)


class ETFCrossSectionFactors:
    """ETF横截面因子计算器"""

    def __init__(self, data_manager: Optional[ETFCrossSectionDataManager] = None,
                 enable_storage: bool = True):
        """
        初始化ETF横截面因子计算器

        Args:
            data_manager: ETF数据管理器，None时创建新实例
            enable_storage: 是否启用数据存储功能
        """
        self.data_manager = data_manager or ETFCrossSectionDataManager()
        self.factor_cache = {}

        # 存储管理器
        if enable_storage:
            self.storage = ETFCrossSectionStorage()
        else:
            self.storage = None

    # ========== 动量因子 ==========

    def calculate_momentum_factors(self,
                                 price_df: pd.DataFrame,
                                 periods: List[int] = [21, 63, 126, 252]) -> pd.DataFrame:
        """
        计算动量因子

        Args:
            price_df: 价格数据DataFrame，包含 etf_code, trade_date, close
            periods: 动量周期，默认为1M、3M、6M、12M

        Returns:
            动量因子DataFrame
        """
        momentum_factors = []

        for etf_code in price_df['etf_code'].unique():
            etf_data = price_df[price_df['etf_code'] == etf_code].copy()
            etf_data = etf_data.sort_values('trade_date').reset_index(drop=True)

            if len(etf_data) < max(periods) + 21:  # 需要足够的历史数据
                continue

            close_prices = etf_data['close'].values

            for period in periods:
                # 动量 = (当前价格 / N天前价格) - 1
                momentum = np.zeros(len(close_prices))
                momentum[period:] = (close_prices[period:] / close_prices[:-period]) - 1

                # 计算动量强度（避免短期噪音）
                if period >= 63:
                    momentum_strength = talib.STOCH(
                        close_prices, close_prices, close_prices,
                        fastk_period=period//3, slowk_period=3, slowd_period=3
                    )[0]
                else:
                    momentum_strength = momentum

                for i, (date, mom, strength) in enumerate(zip(etf_data['trade_date'], momentum, momentum_strength)):
                    if i >= period and not np.isnan(mom) and not np.isnan(strength):
                        momentum_factors.append({
                            'etf_code': etf_code,
                            'date': date,
                            f'momentum_{period}d': mom,
                            f'momentum_strength_{period}d': strength
                        })

        momentum_df = pd.DataFrame(momentum_factors)
        logger.info(f"动量因子计算完成: {len(momentum_df)} 条记录")

        return momentum_df

    # ========== 质量因子 ==========

    def calculate_quality_factors(self,
                                price_df: pd.DataFrame,
                                window: int = 252) -> pd.DataFrame:
        """
        计算质量因子

        Args:
            price_df: 价格数据DataFrame
            window: 计算窗口，默认为1年

        Returns:
            质量因子DataFrame
        """
        quality_factors = []

        for etf_code in price_df['etf_code'].unique():
            etf_data = price_df[price_df['etf_code'] == etf_code].copy()
            etf_data = etf_data.sort_values('trade_date').reset_index(drop=True)

            if len(etf_data) < window + 21:
                continue

            close_prices = etf_data['close'].values
            volumes = etf_data['vol'].values

            # 波动率（年化）
            returns = np.diff(np.log(close_prices))
            volatility = np.zeros(len(close_prices))
            for i in range(window, len(returns)):
                vol = np.std(returns[i-window:i]) * np.sqrt(252)
                volatility[i+1] = vol

            # 最大回撤
            max_drawdown = np.zeros(len(close_prices))
            peak = close_prices[0]
            for i in range(1, len(close_prices)):
                if close_prices[i] > peak:
                    peak = close_prices[i]
                dd = (peak - close_prices[i]) / peak
                max_drawdown[i] = dd

            # 夏普比率（简化版）
            sharpe_ratio = np.zeros(len(close_prices))
            for i in range(window, len(returns)):
                mean_return = np.mean(returns[i-window:i]) * 252
                vol = volatility[i]
                sharpe_ratio[i] = mean_return / vol if vol > 0 else 0

            # 胜率
            win_rate = np.zeros(len(close_prices))
            for i in range(window, len(returns)):
                wins = np.sum(returns[i-window:i] > 0)
                win_rate[i] = wins / window

            for i, date in enumerate(etf_data['trade_date']):
                if i >= window:
                    quality_factors.append({
                        'etf_code': etf_code,
                        'date': date,
                        'volatility_1y': volatility[i],
                        'max_drawdown_1y': max_drawdown[i],
                        'sharpe_ratio_1y': sharpe_ratio[i],
                        'win_rate_1y': win_rate[i],
                        'quality_score': sharpe_ratio[i] - volatility[i] - max_drawdown[i]  # 综合质量得分
                    })

        quality_df = pd.DataFrame(quality_factors)
        logger.info(f"质量因子计算完成: {len(quality_df)} 条记录")

        return quality_df

    # ========== 流动性因子 ==========

    def calculate_liquidity_factors(self,
                                  price_df: pd.DataFrame,
                                  window: int = 21) -> pd.DataFrame:
        """
        计算流动性因子

        Args:
            price_df: 价格数据DataFrame，包含volume和amount
            window: 计算窗口，默认为1个月

        Returns:
            流动性因子DataFrame
        """
        liquidity_factors = []

        for etf_code in price_df['etf_code'].unique():
            etf_data = price_df[price_df['etf_code'] == etf_code].copy()
            etf_data = etf_data.sort_values('trade_date').reset_index(drop=True)

            if len(etf_data) < window:
                continue

            volumes = etf_data['vol'].values
            amounts = etf_data['amount'].values
            closes = etf_data['close'].values

            # 平均成交量（ADV）
            avg_volume = np.zeros(len(volumes))
            for i in range(window, len(volumes)):
                avg_volume[i] = np.mean(volumes[i-window:i])

            # 平均成交额
            avg_amount = np.zeros(len(amounts))
            for i in range(window, len(amounts)):
                avg_amount[i] = np.mean(amounts[i-window:i])

            # 成交量标准差（衡量稳定性）
            volume_std = np.zeros(len(volumes))
            for i in range(window, len(volumes)):
                volume_std[i] = np.std(volumes[i-window:i])

            # 换手率（简化估算）
            turnover_rate = np.zeros(len(volumes))
            for i in range(window, len(volumes)):
                if closes[i] > 0:
                    turnover_rate[i] = amounts[i] / (closes[i] * 1000000000)  # 假设规模为10亿

            # 流动性得分
            liquidity_score = np.zeros(len(volumes))
            for i in range(window, len(volumes)):
                # 综合考虑成交额、稳定性和换手率
                score = (avg_amount[i] / 1000000) * (1 - volume_std[i] / avg_volume[i]) * turnover_rate[i]
                liquidity_score[i] = score

            for i, date in enumerate(etf_data['trade_date']):
                if i >= window:
                    liquidity_factors.append({
                        'etf_code': etf_code,
                        'date': date,
                        'avg_volume_21d': avg_volume[i],
                        'avg_amount_21d': avg_amount[i],
                        'volume_stability': 1 - volume_std[i] / avg_volume[i] if avg_volume[i] > 0 else 0,
                        'turnover_rate': turnover_rate[i],
                        'liquidity_score': liquidity_score[i]
                    })

        liquidity_df = pd.DataFrame(liquidity_factors)
        logger.info(f"流动性因子计算完成: {len(liquidity_df)} 条记录")

        return liquidity_df

    # ========== 技术因子 ==========

    def calculate_technical_factors(self,
                                  price_df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术因子

        Args:
            price_df: 价格数据DataFrame

        Returns:
            技术因子DataFrame
        """
        technical_factors = []

        for etf_code in price_df['etf_code'].unique():
            etf_data = price_df[price_df['etf_code'] == etf_code].copy()
            etf_data = etf_data.sort_values('trade_date').reset_index(drop=True)

            if len(etf_data) < 50:
                continue

            close_prices = etf_data['close'].values
            high_prices = etf_data['high'].values if 'high' in etf_data.columns else close_prices
            low_prices = etf_data['low'].values if 'low' in etf_data.columns else close_prices
            volumes = etf_data['vol'].values

            # RSI
            rsi = talib.RSI(close_prices, timeperiod=14)

            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close_prices)

            # 布林带
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices, timeperiod=20)

            # 威廉指标
            williams_r = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=14)

            # CCI
            cci = talib.CCI(high_prices, low_prices, close_prices, timeperiod=14)

            # 成交量价格趋势
            vpt = np.zeros(len(close_prices))
            for i in range(1, len(close_prices)):
                if close_prices[i-1] > 0:
                    vpt[i] = vpt[i-1] + volumes[i] * (close_prices[i] - close_prices[i-1]) / close_prices[i-1]

            for i, date in enumerate(etf_data['trade_date']):
                if i >= 20 and not (np.isnan(rsi[i]) or np.isnan(macd[i]) or np.isnan(williams_r[i])):
                    technical_factors.append({
                        'etf_code': etf_code,
                        'date': date,
                        'rsi_14': rsi[i],
                        'macd': macd[i],
                        'macd_signal': macd_signal[i],
                        'macd_histogram': macd_hist[i],
                        'bb_position': (close_prices[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i]) if bb_upper[i] != bb_lower[i] else 0.5,
                        'williams_r': williams_r[i],
                        'cci_14': cci[i],
                        'vpt': vpt[i],
                        'technical_score': (rsi[i]/50 - 1) + (macd_hist[i] * 1000) + (williams_r[i]/50 + 1)  # 综合技术得分
                    })

        technical_df = pd.DataFrame(technical_factors)
        logger.info(f"技术因子计算完成: {len(technical_df)} 条记录")

        return technical_df

    # ========== 因子融合 ==========

    def calculate_composite_factors(self,
                                  momentum_df: pd.DataFrame,
                                  quality_df: pd.DataFrame,
                                  liquidity_df: pd.DataFrame,
                                  technical_df: pd.DataFrame) -> pd.DataFrame:
        """
        计算综合因子

        Args:
            momentum_df: 动量因子DataFrame
            quality_df: 质量因子DataFrame
            liquidity_df: 流动性因子DataFrame
            technical_df: 技术因子DataFrame

        Returns:
            综合因子DataFrame
        """
        # 合并所有因子
        dfs = [momentum_df, quality_df, liquidity_df, technical_df]

        # 过滤空的DataFrame
        dfs = [df for df in dfs if not df.empty]

        if not dfs:
            return pd.DataFrame()

        # 使用merge合并
        composite_df = dfs[0].copy()

        for df in dfs[1:]:
            composite_df = pd.merge(
                composite_df,
                df,
                on=['etf_code', 'date'],
                how='outer'
            )

        # 计算综合得分
        if composite_df.empty:
            return composite_df

        # 动量得分（使用不同周期的动量）
        momentum_cols = [col for col in composite_df.columns if 'momentum_' in col and 'strength' not in col]
        if momentum_cols:
            composite_df['momentum_score'] = composite_df[momentum_cols].mean(axis=1, skipna=True)

        # 技术得分
        technical_cols = ['rsi_14', 'macd_histogram', 'williams_r', 'cci_14']
        available_tech_cols = [col for col in technical_cols if col in composite_df.columns]
        if available_tech_cols:
            # 标准化技术指标
            tech_score = 0
            for col in available_tech_cols:
                if col == 'rsi_14':
                    tech_score += (composite_df[col] - 50) / 50
                elif col == 'macd_histogram':
                    tech_score += np.tanh(composite_df[col] * 1000)  # 压缩极值
                elif col == 'williams_r':
                    tech_score += (composite_df[col] + 50) / 50
                elif col == 'cci_14':
                    tech_score += np.tanh(composite_df[col] / 100)

            composite_df['technical_score_normalized'] = tech_score / len(available_tech_cols)

        # 综合得分（加权平均）
        score_components = []
        weights = []

        if 'momentum_score' in composite_df.columns:
            score_components.append(composite_df['momentum_score'])
            weights.append(0.4)  # 动量权重40%

        if 'quality_score' in composite_df.columns:
            score_components.append(composite_df['quality_score'])
            weights.append(0.3)  # 质量权重30%

        if 'liquidity_score' in composite_df.columns:
            # 流动性得分需要标准化
            liquidity_norm = (composite_df['liquidity_score'] - composite_df['liquidity_score'].min()) / \
                           (composite_df['liquidity_score'].max() - composite_df['liquidity_score'].min())
            score_components.append(liquidity_norm)
            weights.append(0.2)  # 流动性权重20%

        if 'technical_score_normalized' in composite_df.columns:
            score_components.append(composite_df['technical_score_normalized'])
            weights.append(0.1)  # 技术权重10%

        if score_components:
            composite_df['composite_score'] = sum(score * weight for score, weight in zip(score_components, weights))

        logger.info(f"综合因子计算完成: {len(composite_df)} 条记录")
        return composite_df

    # ========== 主要接口方法 ==========

    def calculate_all_factors(self,
                            start_date: str,
                            end_date: str,
                            etf_codes: Optional[List[str]] = None,
                            use_cache: bool = True,
                            save_to_storage: bool = True) -> pd.DataFrame:
        """
        计算所有因子

        Args:
            start_date: 开始日期
            end_date: 结束日期
            etf_codes: ETF代码列表，None表示所有ETF
            use_cache: 是否使用缓存
            save_to_storage: 是否保存到存储

        Returns:
            完整的因子DataFrame
        """
        logger.info(f"开始计算ETF横截面因子: {start_date} ~ {end_date}")

        # 生成缓存键
        cache_key = f"all_factors_{start_date}_{end_date}_{'_'.join(etf_codes or [])}"

        # 尝试从缓存加载
        if use_cache and self.storage:
            cached_data = self.storage.load_cache(cache_key)
            if cached_data is not None:
                logger.info(f"从缓存加载因子数据: {len(cached_data)} 条记录")
                return cached_data

        # 获取价格数据
        if etf_codes is None:
            etf_codes = self.data_manager.get_etf_universe()

        price_data = self.data_manager.get_time_series_data(start_date, end_date, etf_codes)
        if price_data.empty:
            logger.error("未获取到价格数据")
            return pd.DataFrame()

        # 计算各类因子
        momentum_df = self.calculate_momentum_factors(price_data)
        quality_df = self.calculate_quality_factors(price_data)
        liquidity_df = self.calculate_liquidity_factors(price_data)
        technical_df = self.calculate_technical_factors(price_data)

        # 融合因子
        composite_df = self.calculate_composite_factors(
            momentum_df, quality_df, liquidity_df, technical_df
        )

        # 保存到存储
        if save_to_storage and self.storage and not composite_df.empty:
            # 保存综合因子数据
            self.storage.save_composite_factors(
                composite_df, etf_codes, start_date, end_date
            )

            # 保存到缓存
            self.storage.save_cache(cache_key, composite_df, ttl_hours=24)

        logger.info(f"ETF横截面因子计算完成: {len(composite_df)} 条记录，{composite_df['etf_code'].nunique()} 只ETF")
        return composite_df

    def get_factor_ranking(self,
                          date: str,
                          top_n: int = 10,
                          factor_col: str = 'composite_score') -> pd.DataFrame:
        """
        获取指定日期的因子排名

        Args:
            date: 查询日期
            top_n: 返回前N只ETF
            factor_col: 排序因子列

        Returns:
            排名后的DataFrame
        """
        # 尝试从存储中加载指定日期的横截面数据
        if self.storage:
            cross_section_data = self.storage.load_cross_section_data(date, "daily")
            if cross_section_data is not None and factor_col in cross_section_data.columns:
                # 按因子排序
                ranked_data = cross_section_data.sort_values(factor_col, ascending=False)
                return ranked_data.head(top_n)

        # 如果存储中没有，尝试重新计算
        logger.warning(f"存储中未找到日期 {date} 的数据，尝试重新计算")
        try:
            # 计算当天的因子
            end_date = pd.to_datetime(date)
            start_date = end_date - timedelta(days=30)  # 使用30天数据计算因子

            factors_df = self.calculate_all_factors(
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                use_cache=True,
                save_to_storage=True
            )

            if not factors_df.empty:
                # 获取指定日期的数据
                target_date = pd.to_datetime(date)
                date_factors = factors_df[factors_df['date'] == target_date]

                if not date_factors.empty and factor_col in date_factors.columns:
                    ranked_data = date_factors.sort_values(factor_col, ascending=False)
                    return ranked_data.head(top_n)

        except Exception as e:
            logger.error(f"重新计算因子失败: {e}")

        return pd.DataFrame()

    def load_stored_factors(self,
                           start_date: str,
                           end_date: str,
                           etf_codes: Optional[List[str]] = None) -> pd.DataFrame:
        """
        从存储中加载已计算的因子数据

        Args:
            start_date: 开始日期
            end_date: 结束日期
            etf_codes: ETF代码列表

        Returns:
            因子DataFrame
        """
        if not self.storage:
            logger.warning("存储功能未启用")
            return pd.DataFrame()

        # 尝试从存储加载
        stored_factors = self.storage.load_composite_factors(start_date, end_date)

        if stored_factors is not None:
            # 如果指定了ETF列表，进行过滤
            if etf_codes is not None:
                stored_factors = stored_factors[stored_factors['etf_code'].isin(etf_codes)]

            logger.info(f"从存储加载因子数据: {len(stored_factors)} 条记录")
            return stored_factors

        logger.info(f"存储中未找到 {start_date} ~ {end_date} 的因子数据")
        return pd.DataFrame()

    def get_storage_info(self) -> Dict:
        """
        获取存储信息

        Returns:
            存储信息字典
        """
        if not self.storage:
            return {"error": "存储功能未启用"}

        return self.storage.get_storage_info()

    def clear_cache(self) -> int:
        """
        清理过期缓存

        Returns:
            清理的文件数量
        """
        if not self.storage:
            logger.warning("存储功能未启用")
            return 0

        return self.storage.cleanup_expired_cache()


# 便捷函数
def calculate_etf_cross_section_factors(start_date: str,
                                      end_date: str,
                                      etf_codes: Optional[List[str]] = None) -> pd.DataFrame:
    """
    计算ETF横截面因子的便捷函数

    Args:
        start_date: 开始日期
        end_date: 结束日期
        etf_codes: ETF代码列表

    Returns:
        因子DataFrame
    """
    calculator = ETFCrossSectionFactors()
    return calculator.calculate_all_factors(start_date, end_date, etf_codes)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    # 测试因子计算
    start_date = "2024-01-01"
    end_date = "2025-10-14"
    test_etfs = ['510300.SH', '159915.SZ', '515030.SH', '518880.SH', '513100.SH']

    calculator = ETFCrossSectionFactors()
    factors_df = calculator.calculate_all_factors(start_date, end_date, test_etfs)

    if not factors_df.empty:
        print(f"因子数据示例:")
        print(factors_df.head())

        # 显示因子列
        print(f"\n可用因子列: {factors_df.columns.tolist()}")

        # 显示综合得分最高的ETF
        if 'composite_score' in factors_df.columns:
            latest_date = factors_df['date'].max()
            latest_factors = factors_df[factors_df['date'] == latest_date]
            top_etfs = latest_factors.nlargest(5, 'composite_score')

            print(f"\n{latest_date} 综合得分最高的ETF:")
            for _, row in top_etfs.iterrows():
                print(f"{row['etf_code']}: {row['composite_score']:.4f}")
    else:
        print("因子计算失败或无数据")