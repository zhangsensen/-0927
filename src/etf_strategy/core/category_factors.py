"""
分类因子库 | Category-Specific Factor Library
================================================================================
针对不同资产类别（债券、商品、QDII）的专用因子

设计原则：
1. 债券因子：利率敏感性、久期代理、收益率曲线斜率
2. 商品因子：美元逆相关、波动率代理、期货曲线结构
3. QDII因子：跨市场价差、汇率动量、时区套利

核心创新：
- 每个资产类别使用最适合其定价逻辑的因子
- 避免用权益因子（如RSI、ADX）去选债券或黄金
- 因子与资产定价机制匹配

================================================================================
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numba import njit

logger = logging.getLogger(__name__)


# =============================================================================
# 因子元数据定义
# =============================================================================


@dataclass
class CategoryFactorMetadata:
    """分类因子元数据"""

    name: str
    description: str
    category: str  # 'BOND', 'COMMODITY', 'QDII', 'UNIVERSAL'
    required_data: List[str]  # 所需数据列
    window: int
    bounded: bool
    direction: str  # 'high_is_good', 'low_is_good', 'neutral'


# =============================================================================
# Numba加速函数
# =============================================================================


@njit
def _rolling_yield_slope_numba(
    short_term: np.ndarray, long_term: np.ndarray, window: int
) -> np.ndarray:
    """
    计算滚动收益率曲线斜率（长端 - 短端）

    斜率为正 → 正常曲线（经济扩张预期）
    斜率为负 → 倒挂曲线（衰退预警）
    """
    n = len(short_term)
    result = np.full(n, np.nan)

    for i in range(window - 1, n):
        # 窗口内均值差
        short_mean = np.nanmean(short_term[i - window + 1 : i + 1])
        long_mean = np.nanmean(long_term[i - window + 1 : i + 1])

        if np.isnan(short_mean) or np.isnan(long_mean):
            result[i] = np.nan
        else:
            result[i] = long_mean - short_mean

    return result


@njit
def _rolling_duration_proxy_numba(
    prices: np.ndarray, rates: np.ndarray, window: int
) -> np.ndarray:
    """
    久期代理：价格对利率变化的敏感性

    公式：ΔP/P ÷ Δr（百分比变化 / 利率变化）
    负值表示正常的反向关系（利率↑ → 债券价格↓）
    """
    n = len(prices)
    result = np.full(n, np.nan)
    eps = 1e-10

    for i in range(window, n):
        price_ret = (prices[i] - prices[i - window]) / (prices[i - window] + eps)
        rate_change = rates[i] - rates[i - window]

        if abs(rate_change) < eps:
            result[i] = np.nan
        else:
            result[i] = -price_ret / rate_change  # 负号使正值表示正久期

    return result


# =============================================================================
# 债券因子类
# =============================================================================


class BondFactors:
    """
    债券专用因子

    核心逻辑：
    - 债券定价由利率水平、久期、信用利差决定
    - 动量因子对债券效果较差，应使用利率敏感性因子

    因子清单：
    1. YIELD_MOMENTUM_20D: 收益率动量（债券价格反向）
    2. DURATION_PROXY_60D: 久期代理（利率敏感性）
    3. CREDIT_SPREAD_PROXY: 信用利差代理（相对国债表现）
    4. BOND_VOL_20D: 债券波动率（反映利率不确定性）
    """

    def __init__(self):
        self.metadata = self._build_metadata()

    def _build_metadata(self) -> Dict[str, CategoryFactorMetadata]:
        return {
            "YIELD_MOMENTUM_20D": CategoryFactorMetadata(
                name="YIELD_MOMENTUM_20D",
                description="20日收益率动量（对应债券价格反向）",
                category="BOND",
                required_data=["close"],
                window=20,
                bounded=False,
                direction="high_is_good",  # 收益率上升 → 债券价格下跌，对于做空有利
            ),
            "DURATION_PROXY_60D": CategoryFactorMetadata(
                name="DURATION_PROXY_60D",
                description="60日久期代理（价格对利率敏感性）",
                category="BOND",
                required_data=["close", "benchmark_rate"],
                window=60,
                bounded=False,
                direction="low_is_good",  # 低久期 = 低风险
            ),
            "BOND_VOL_REGIME": CategoryFactorMetadata(
                name="BOND_VOL_REGIME",
                description="债券波动率体制（高/低波动）",
                category="BOND",
                required_data=["close"],
                window=60,
                bounded=True,  # [0, 1]
                direction="low_is_good",  # 低波动 = 稳定
            ),
            "BOND_MOMENTUM_SCORE": CategoryFactorMetadata(
                name="BOND_MOMENTUM_SCORE",
                description="债券综合动量评分",
                category="BOND",
                required_data=["close"],
                window=20,
                bounded=False,
                direction="high_is_good",
            ),
        }

    def yield_momentum_20d(self, close: pd.Series) -> pd.Series:
        """
        收益率动量 | YIELD_MOMENTUM_20D

        对于债券ETF，直接用价格动量的负值作为收益率动量代理
        （价格下跌 → 收益率上升）

        Returns:
            pd.Series: 收益率动量（百分比）
        """
        # 价格动量（反向表示收益率动量）
        price_mom = (close / close.shift(20) - 1) * 100
        return -price_mom  # 取反：价格下跌 = 收益率上升

    def duration_proxy_60d(
        self, close: pd.Series, benchmark_rate: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        久期代理 | DURATION_PROXY_60D

        如果没有基准利率数据，使用自身波动率的逆作为代理
        （低波动债券通常是短久期）

        Returns:
            pd.Series: 久期代理值
        """
        if benchmark_rate is not None and len(benchmark_rate) == len(close):
            # 有基准利率：计算真实久期代理
            result = _rolling_duration_proxy_numba(
                close.values, benchmark_rate.values, window=60
            )
            return pd.Series(result, index=close.index)
        else:
            # 无基准利率：用波动率作为久期代理
            # 高波动 = 高久期（利率敏感性高）
            returns = close.pct_change()
            vol = returns.rolling(window=60, min_periods=60).std() * np.sqrt(252)
            return vol * 100  # 转为百分比

    def bond_vol_regime(self, close: pd.Series) -> pd.Series:
        """
        债券波动率体制 | BOND_VOL_REGIME

        将当前波动率与历史分布比较，输出 [0, 1] 的分位数
        0 = 极低波动，1 = 极高波动

        Returns:
            pd.Series: 波动率分位数 [0, 1]
        """
        returns = close.pct_change()
        vol_20 = returns.rolling(window=20, min_periods=20).std()

        # 60日滚动分位数
        def rolling_percentile(x):
            if len(x) < 60:
                return np.nan
            return (x.iloc[-1:].values[0] <= x).mean()

        vol_regime = vol_20.rolling(window=60, min_periods=60).apply(
            rolling_percentile, raw=False
        )
        return vol_regime

    def bond_momentum_score(self, close: pd.Series) -> pd.Series:
        """
        债券综合动量评分 | BOND_MOMENTUM_SCORE

        结合短期(5日)和中期(20日)动量，针对债券的低波动特性优化

        Returns:
            pd.Series: 综合动量评分
        """
        mom_5 = (close / close.shift(5) - 1) * 100
        mom_20 = (close / close.shift(20) - 1) * 100

        # 债券动量权重：短期0.4，中期0.6（债券趋势更稳定）
        score = 0.4 * mom_5 + 0.6 * mom_20
        return score

    def compute_all(
        self, prices: Dict[str, pd.DataFrame], symbols: List[str]
    ) -> pd.DataFrame:
        """
        批量计算所有债券因子

        Args:
            prices: {'close': DataFrame, ...}
            symbols: 债券ETF代码列表

        Returns:
            DataFrame with MultiIndex columns (factor, symbol)
        """
        close_df = prices["close"][symbols]

        results = {}

        for symbol in symbols:
            close = close_df[symbol]

            results[(symbol, "YIELD_MOMENTUM_20D")] = self.yield_momentum_20d(close)
            results[(symbol, "DURATION_PROXY_60D")] = self.duration_proxy_60d(close)
            results[(symbol, "BOND_VOL_REGIME")] = self.bond_vol_regime(close)
            results[(symbol, "BOND_MOMENTUM_SCORE")] = self.bond_momentum_score(close)

        # 转换为 (factor, symbol) 格式
        df = pd.DataFrame(results)
        df.columns = pd.MultiIndex.from_tuples(
            [(col[1], col[0]) for col in df.columns], names=["factor", "symbol"]
        )

        return df.sort_index(axis=1)


# =============================================================================
# 商品因子类
# =============================================================================


class CommodityFactors:
    """
    商品专用因子

    核心逻辑：
    - 黄金/白银主要受美元、实际利率、避险情绪驱动
    - 传统技术因子效果有限，应使用宏观相关因子

    因子清单：
    1. USD_INVERSE_MOM_20D: 美元逆相关动量（美元弱→商品强）
    2. GOLD_SAFE_HAVEN_SCORE: 避险情绪评分（股市跌→黄金涨）
    3. COMMODITY_TREND_20D: 商品趋势强度
    4. SILVER_GOLD_RATIO: 金银比（风险偏好指标）
    """

    def __init__(self):
        self.metadata = self._build_metadata()

    def _build_metadata(self) -> Dict[str, CategoryFactorMetadata]:
        return {
            "USD_INVERSE_MOM_20D": CategoryFactorMetadata(
                name="USD_INVERSE_MOM_20D",
                description="20日美元逆相关动量",
                category="COMMODITY",
                required_data=["close"],
                window=20,
                bounded=False,
                direction="high_is_good",
            ),
            "COMMODITY_TREND_20D": CategoryFactorMetadata(
                name="COMMODITY_TREND_20D",
                description="20日商品趋势强度",
                category="COMMODITY",
                required_data=["close"],
                window=20,
                bounded=False,
                direction="high_is_good",
            ),
            "GOLD_SAFE_HAVEN_SCORE": CategoryFactorMetadata(
                name="GOLD_SAFE_HAVEN_SCORE",
                description="避险情绪评分",
                category="COMMODITY",
                required_data=["close", "market_close"],
                window=20,
                bounded=True,
                direction="high_is_good",
            ),
            "COMMODITY_VOL_ADJ_MOM": CategoryFactorMetadata(
                name="COMMODITY_VOL_ADJ_MOM",
                description="波动率调整动量",
                category="COMMODITY",
                required_data=["close"],
                window=20,
                bounded=False,
                direction="high_is_good",
            ),
        }

    def usd_inverse_mom_20d(
        self, close: pd.Series, usd_proxy: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        美元逆相关动量 | USD_INVERSE_MOM_20D

        黄金与美元负相关，当美元走弱时黄金倾向上涨
        如果没有美元数据，使用商品自身动量作为代理

        Returns:
            pd.Series: 动量（百分比）
        """
        if usd_proxy is not None and len(usd_proxy) == len(close):
            # 美元走弱 = 商品利好
            usd_mom = (usd_proxy / usd_proxy.shift(20) - 1) * 100
            return -usd_mom  # 取反
        else:
            # 使用自身动量
            return (close / close.shift(20) - 1) * 100

    def commodity_trend_20d(self, close: pd.Series) -> pd.Series:
        """
        商品趋势强度 | COMMODITY_TREND_20D

        结合价格位置和动量，评估趋势强度

        Returns:
            pd.Series: 趋势强度评分
        """
        # 20日动量
        mom = (close / close.shift(20) - 1) * 100

        # 价格相对20日高低点的位置
        high_20 = close.rolling(window=20, min_periods=20).max()
        low_20 = close.rolling(window=20, min_periods=20).min()
        pos = (close - low_20) / (high_20 - low_20 + 1e-10)

        # 趋势评分 = 动量 × 位置权重
        # 位置高 + 动量正 = 强上升趋势
        trend_score = mom * (0.5 + pos)

        return trend_score

    def gold_safe_haven_score(
        self, gold_close: pd.Series, market_close: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        避险情绪评分 | GOLD_SAFE_HAVEN_SCORE

        当股市下跌时黄金上涨的程度，反映避险属性强度

        Returns:
            pd.Series: 避险评分 [0, 1]
        """
        gold_ret = gold_close.pct_change()

        if market_close is not None and len(market_close) == len(gold_close):
            market_ret = market_close.pct_change()

            # 负相关程度
            corr = gold_ret.rolling(window=20, min_periods=20).corr(market_ret)

            # 转换为 [0, 1]：-1 → 1（完美避险），+1 → 0（完全正相关）
            safe_haven = (1 - corr) / 2
        else:
            # 无市场数据：使用波动率逆指标
            vol = gold_ret.rolling(window=20, min_periods=20).std()
            safe_haven = 1 - vol.rank(pct=True)

        return safe_haven.clip(0, 1)

    def commodity_vol_adj_mom(self, close: pd.Series) -> pd.Series:
        """
        波动率调整动量 | COMMODITY_VOL_ADJ_MOM

        动量 / 波动率，衡量单位风险的收益

        Returns:
            pd.Series: 波动率调整后的动量
        """
        returns = close.pct_change()

        mom = returns.rolling(window=20, min_periods=20).sum() * 100
        vol = returns.rolling(window=20, min_periods=20).std() * np.sqrt(252) * 100

        adj_mom = mom / (vol + 1e-10)

        return adj_mom

    def compute_all(
        self,
        prices: Dict[str, pd.DataFrame],
        symbols: List[str],
        market_proxy: Optional[str] = "510300.SH",
    ) -> pd.DataFrame:
        """
        批量计算所有商品因子

        Args:
            prices: {'close': DataFrame, ...}
            symbols: 商品ETF代码列表
            market_proxy: 市场基准代码（用于避险评分）

        Returns:
            DataFrame with MultiIndex columns (factor, symbol)
        """
        close_df = prices["close"]

        # 获取市场基准
        market_close = None
        if market_proxy in close_df.columns:
            market_close = close_df[market_proxy]

        results = {}

        for symbol in symbols:
            if symbol not in close_df.columns:
                logger.warning(f"商品因子计算跳过: {symbol} 不在价格数据中")
                continue

            close = close_df[symbol]

            results[(symbol, "USD_INVERSE_MOM_20D")] = self.usd_inverse_mom_20d(close)
            results[(symbol, "COMMODITY_TREND_20D")] = self.commodity_trend_20d(close)
            results[(symbol, "GOLD_SAFE_HAVEN_SCORE")] = self.gold_safe_haven_score(
                close, market_close
            )
            results[(symbol, "COMMODITY_VOL_ADJ_MOM")] = self.commodity_vol_adj_mom(
                close
            )

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df.columns = pd.MultiIndex.from_tuples(
            [(col[1], col[0]) for col in df.columns], names=["factor", "symbol"]
        )

        return df.sort_index(axis=1)


# =============================================================================
# QDII因子类
# =============================================================================


class QDIIFactors:
    """
    QDII专用因子

    核心逻辑：
    - QDII涉及跨市场投资，需考虑汇率、时差、跨市场套利
    - 美股ETF受纳斯达克/标普走势影响
    - 港股ETF受恒生指数和南向资金影响

    因子清单：
    1. CROSS_MARKET_SPREAD: 跨市场价差（溢价/折价）
    2. FX_MOMENTUM_20D: 汇率动量
    3. QDII_RELATIVE_STRENGTH: 相对海外基准强度
    4. OVERNIGHT_GAP: 隔夜跳空（时区差异机会）
    """

    def __init__(self):
        self.metadata = self._build_metadata()

    def _build_metadata(self) -> Dict[str, CategoryFactorMetadata]:
        return {
            "QDII_MOMENTUM_20D": CategoryFactorMetadata(
                name="QDII_MOMENTUM_20D",
                description="20日QDII动量",
                category="QDII",
                required_data=["close"],
                window=20,
                bounded=False,
                direction="high_is_good",
            ),
            "FX_ADJUSTED_MOM": CategoryFactorMetadata(
                name="FX_ADJUSTED_MOM",
                description="汇率调整动量",
                category="QDII",
                required_data=["close"],
                window=20,
                bounded=False,
                direction="high_is_good",
            ),
            "QDII_VOL_RATIO": CategoryFactorMetadata(
                name="QDII_VOL_RATIO",
                description="QDII波动率比率",
                category="QDII",
                required_data=["close"],
                window=20,
                bounded=False,
                direction="low_is_good",
            ),
            "OVERNIGHT_GAP_SCORE": CategoryFactorMetadata(
                name="OVERNIGHT_GAP_SCORE",
                description="隔夜跳空评分",
                category="QDII",
                required_data=["open", "close"],
                window=20,
                bounded=False,
                direction="neutral",
            ),
        }

    def qdii_momentum_20d(self, close: pd.Series) -> pd.Series:
        """
        QDII动量 | QDII_MOMENTUM_20D

        标准20日动量，适用于跨境ETF

        Returns:
            pd.Series: 动量（百分比）
        """
        return (close / close.shift(20) - 1) * 100

    def fx_adjusted_mom(
        self, close: pd.Series, fx_rate: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        汇率调整动量 | FX_ADJUSTED_MOM

        扣除汇率变动后的真实动量
        如果无汇率数据，返回原始动量

        Returns:
            pd.Series: 汇率调整后的动量（百分比）
        """
        raw_mom = (close / close.shift(20) - 1) * 100

        if fx_rate is not None and len(fx_rate) == len(close):
            fx_mom = (fx_rate / fx_rate.shift(20) - 1) * 100
            return raw_mom - fx_mom  # 扣除汇率影响

        return raw_mom

    def qdii_vol_ratio(self, close: pd.Series) -> pd.Series:
        """
        QDII波动率比率 | QDII_VOL_RATIO

        近期波动率 / 历史波动率，识别波动率突变

        Returns:
            pd.Series: 波动率比率
        """
        returns = close.pct_change()

        vol_10 = returns.rolling(window=10, min_periods=10).std()
        vol_60 = returns.rolling(window=60, min_periods=60).std()

        ratio = vol_10 / (vol_60 + 1e-10)

        return ratio

    def overnight_gap_score(self, open_price: pd.Series, close: pd.Series) -> pd.Series:
        """
        隔夜跳空评分 | OVERNIGHT_GAP_SCORE

        开盘跳空的累计方向，反映隔夜消息的影响
        正值 = 持续跳空高开（利好消息频繁）
        负值 = 持续跳空低开（利空消息频繁）

        Returns:
            pd.Series: 跳空评分
        """
        prev_close = close.shift(1)
        gap = (open_price - prev_close) / (prev_close + 1e-10) * 100

        # 20日累计跳空
        gap_score = gap.rolling(window=20, min_periods=20).sum()

        return gap_score

    def compute_all(
        self, prices: Dict[str, pd.DataFrame], symbols: List[str]
    ) -> pd.DataFrame:
        """
        批量计算所有QDII因子

        Args:
            prices: {'close': DataFrame, 'open': DataFrame, ...}
            symbols: QDII ETF代码列表

        Returns:
            DataFrame with MultiIndex columns (factor, symbol)
        """
        close_df = prices["close"]
        open_df = prices.get("open")

        results = {}

        for symbol in symbols:
            if symbol not in close_df.columns:
                logger.warning(f"QDII因子计算跳过: {symbol} 不在价格数据中")
                continue

            close = close_df[symbol]

            results[(symbol, "QDII_MOMENTUM_20D")] = self.qdii_momentum_20d(close)
            results[(symbol, "FX_ADJUSTED_MOM")] = self.fx_adjusted_mom(close)
            results[(symbol, "QDII_VOL_RATIO")] = self.qdii_vol_ratio(close)

            # 隔夜跳空需要开盘价
            if open_df is not None and symbol in open_df.columns:
                results[(symbol, "OVERNIGHT_GAP_SCORE")] = self.overnight_gap_score(
                    open_df[symbol], close
                )

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df.columns = pd.MultiIndex.from_tuples(
            [(col[1], col[0]) for col in df.columns], names=["factor", "symbol"]
        )

        return df.sort_index(axis=1)


# =============================================================================
# 统一的分类因子管理器
# =============================================================================


class CategoryFactorManager:
    """
    分类因子管理器

    统一管理不同资产类别的因子计算：
    - EQUITY: 使用 PreciseFactorLibrary（原有18个因子）
    - BOND: 使用 BondFactors
    - COMMODITY: 使用 CommodityFactors
    - QDII: 使用 QDIIFactors
    """

    def __init__(self):
        self.bond_factors = BondFactors()
        self.commodity_factors = CommodityFactors()
        self.qdii_factors = QDIIFactors()

        # 因子-类别映射
        self.factor_category_map = self._build_category_map()

    def _build_category_map(self) -> Dict[str, str]:
        """构建因子名到类别的映射"""
        mapping = {}

        for name in self.bond_factors.metadata:
            mapping[name] = "BOND"
        for name in self.commodity_factors.metadata:
            mapping[name] = "COMMODITY"
        for name in self.qdii_factors.metadata:
            mapping[name] = "QDII"

        return mapping

    def get_factors_for_category(self, category: str) -> List[str]:
        """获取指定类别的因子列表"""
        return [
            name for name, cat in self.factor_category_map.items() if cat == category
        ]

    def compute_factors_for_pool(
        self,
        pool_name: str,
        prices: Dict[str, pd.DataFrame],
        symbols: List[str],
        market_proxy: str = "510300.SH",
    ) -> pd.DataFrame:
        """
        为指定池计算因子

        Args:
            pool_name: 池名称（如 'BOND', 'COMMODITY', 'QDII', 'EQUITY_*'）
            prices: 价格数据字典
            symbols: ETF代码列表
            market_proxy: 市场基准代码

        Returns:
            DataFrame with factors
        """
        pool_upper = pool_name.upper()

        if pool_upper == "BOND":
            return self.bond_factors.compute_all(prices, symbols)
        elif pool_upper == "COMMODITY":
            return self.commodity_factors.compute_all(prices, symbols, market_proxy)
        elif pool_upper == "QDII":
            return self.qdii_factors.compute_all(prices, symbols)
        else:
            # 权益类池使用通用因子库
            # 这里返回空，由调用方使用 PreciseFactorLibrary
            logger.info(f"池 {pool_name} 使用通用因子库（PreciseFactorLibrary）")
            return pd.DataFrame()

    def list_all_factors(self) -> Dict[str, List[str]]:
        """列出所有分类因子"""
        return {
            "BOND": list(self.bond_factors.metadata.keys()),
            "COMMODITY": list(self.commodity_factors.metadata.keys()),
            "QDII": list(self.qdii_factors.metadata.keys()),
        }


# =============================================================================
# 测试入口
# =============================================================================

if __name__ == "__main__":
    print("CategoryFactorManager 示例")
    print("=" * 70)

    manager = CategoryFactorManager()

    print("\n【分类因子清单】")
    for category, factors in manager.list_all_factors().items():
        print(f"\n{category}:")
        for factor in factors:
            print(f"  - {factor}")

    print("\n✅ 分类因子管理器已准备就绪")
