"""
统一因子计算API - 研究与回测的唯一入口

单例模式，确保全局只有一个FactorEngine实例，
研究、回测、批量生成全部通过这里调用。
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from factor_system.factor_engine.core.cache import CacheConfig
from factor_system.factor_engine.core.engine import FactorEngine
from factor_system.factor_engine.core.registry import FactorRegistry
from factor_system.factor_engine.providers.parquet_provider import ParquetDataProvider
from factor_system.factor_engine.settings import FactorEngineSettings, get_settings

logger = logging.getLogger(__name__)


# 全局单例
_global_engine: Optional[FactorEngine] = None
_global_config: Optional[Dict] = None


def clear_global_engine():
    """清理全局引擎实例（用于测试）"""
    global _global_engine, _global_config
    _global_engine = None
    _global_config = None


def get_engine(
    raw_data_dir: Optional[Path] = None,
    registry_file: Optional[Path] = None,
    cache_config: Optional[CacheConfig] = None,
    force_reinit: bool = False,
) -> FactorEngine:
    """
    获取全局FactorEngine单例

    Args:
        raw_data_dir: 原始数据目录（默认从settings获取）
        registry_file: 因子注册表路径（默认从settings获取）
        cache_config: 缓存配置（默认从settings获取）
        force_reinit: 是否强制重新初始化

    Returns:
        全局FactorEngine实例

    Examples:
        >>> # 使用默认配置
        >>> engine = get_engine()

        >>> # 使用自定义数据目录
        >>> engine = get_engine(raw_data_dir=Path("/data/hk"))

        >>> # 强制重新初始化
        >>> engine = get_engine(force_reinit=True)
    """
    global _global_engine, _global_config

    # 获取配置
    settings = get_settings()

    # 使用配置，允许参数覆盖
    if raw_data_dir is None:
        raw_data_dir = settings.data_paths.raw_data_dir
    if registry_file is None:
        registry_file = settings.data_paths.registry_file
    if cache_config is None:
        cache_config = CacheConfig(
            memory_size_mb=settings.cache.memory_size_mb,
            disk_cache_dir=settings.cache.disk_cache_dir,
            ttl_hours=settings.cache.ttl_hours,
            enable_disk=settings.cache.enable_disk,
            enable_memory=settings.cache.enable_memory,
            copy_mode=settings.cache.copy_mode,
        )

    # 确保目录存在
    settings.ensure_directories()

    # 检查配置是否变更（规范化路径）
    current_config = {
        "raw_data_dir": str(raw_data_dir.resolve()),
        "registry_file": (
            str(registry_file.resolve())
            if registry_file.exists()
            else str(registry_file)
        ),
        "cache_memory_mb": cache_config.memory_size_mb,
        "cache_ttl_hours": cache_config.ttl_hours,
        "cache_enable_disk": cache_config.enable_disk,
        "cache_enable_memory": cache_config.enable_memory,
        "cache_copy_mode": cache_config.copy_mode,
        "cache_disk_cache_dir": str(cache_config.disk_cache_dir),
        "engine_n_jobs": settings.engine.n_jobs,
    }

    # 只在真正需要时重新初始化
    if _global_engine is None or force_reinit:
        should_reinit = True
    elif _global_config is None:
        should_reinit = True
    else:
        # 比较配置
        should_reinit = (
            current_config["raw_data_dir"] != _global_config.get("raw_data_dir")
            or current_config["registry_file"] != _global_config.get("registry_file")
            or current_config["cache_memory_mb"]
            != _global_config.get("cache_memory_mb")
            or current_config["cache_ttl_hours"]
            != _global_config.get("cache_ttl_hours")
            or current_config["cache_enable_disk"]
            != _global_config.get("cache_enable_disk")
            or current_config["cache_enable_memory"]
            != _global_config.get("cache_enable_memory")
            or current_config["cache_copy_mode"]
            != _global_config.get("cache_copy_mode")
            or current_config["cache_disk_cache_dir"]
            != _global_config.get("cache_disk_cache_dir")
            or current_config["engine_n_jobs"] != _global_config.get("engine_n_jobs")
        )

    if should_reinit:
        logger.info("初始化全局FactorEngine...")

        # 初始化数据提供者
        data_provider = ParquetDataProvider(raw_data_dir)

        # 初始化注册表
        registry = FactorRegistry(registry_file)

        # 注册核心因子
        _register_core_factors(registry)

        # 创建引擎
        _global_engine = FactorEngine(
            data_provider=data_provider,
            registry=registry,
            cache_config=cache_config,
        )

        _global_config = current_config

        logger.info(f"✅ FactorEngine已初始化: {len(registry.factors)}个因子已注册")

    return _global_engine


def _register_core_factors(registry: FactorRegistry):
    """注册所有可用因子到注册表"""
    try:
        # 技术指标 (36个)
        # 移动平均 (15个)
        from factor_system.factor_engine.factors.overlap import (
            BBANDS,
            DEMA,
            EMA,
            KAMA,
            MAMA,
            MIDPOINT,
            MIDPRICE,
            SAR,
            SAREXT,
            SMA,
            T3,
            TEMA,
            TRIMA,
            WMA,
        )

        # K线形态 (选择10个常用形态)
        from factor_system.factor_engine.factors.pattern import (
            CDL2CROWS,
            CDL3BLACKCROWS,
            CDL3WHITESOLDIERS,
            CDLDOJI,
            CDLDRAGONFLYDOJI,
            CDLENGULFING,
            CDLEVENINGDOJISTAR,
            CDLHAMMER,
            CDLHANGINGMAN,
            CDLHARAMI,
        )

        # 统计指标 (21个)
        from factor_system.factor_engine.factors.statistic import (
            AVGPRICE,
            BETA,
            CORREL,
            HT_DCPERIOD,
            HT_DCPHASE,
            HT_PHASOR,
            HT_SINE,
            HT_TRENDLINE,
            HT_TRENDMODE,
            LINEARREG,
            LINEARREG_ANGLE,
            LINEARREG_INTERCEPT,
            LINEARREG_SLOPE,
            MEDPRICE,
            STDDEV,
            TSF,
            TYPPRICE,
            VAR,
            WCLPRICE,
        )
        from factor_system.factor_engine.factors.technical import (
            AD,
            ADOSC,
            ADX,
            ADXR,
            APO,
            AROON,
            AROONOSC,
            ATR,
            BOP,
            CCI,
            CMO,
            DX,
            MACD,
            MFI,
            MINUS_DI,
            MINUS_DM,
            MOM,
            NATR,
            OBV,
            PLUS_DI,
            PLUS_DM,
            PPO,
            ROC,
            ROCP,
            ROCR,
            ROCR100,
            RSI,
            STOCH,
            STOCHF,
            STOCHRSI,
            TRANGE,
            TRIX,
            ULTOSC,
            WILLR,
        )

        # 批量注册所有因子
        factor_classes = [
            # Technical Indicators (36)
            AD,
            ADOSC,
            ADX,
            ADXR,
            APO,
            AROON,
            AROONOSC,
            ATR,
            BOP,
            CCI,
            CMO,
            DX,
            MACD,
            MFI,
            MINUS_DI,
            MINUS_DM,
            MOM,
            NATR,
            OBV,
            PLUS_DI,
            PLUS_DM,
            PPO,
            ROC,
            ROCP,
            ROCR,
            ROCR100,
            RSI,
            STOCH,
            STOCHF,
            STOCHRSI,
            TRANGE,
            TRIX,
            ULTOSC,
            WILLR,
            # Overlap Studies (15)
            BBANDS,
            DEMA,
            EMA,
            KAMA,
            MAMA,
            MIDPOINT,
            MIDPRICE,
            SAR,
            SAREXT,
            SMA,
            T3,
            TEMA,
            TRIMA,
            WMA,
            # Statistic Functions (21)
            AVGPRICE,
            BETA,
            CORREL,
            HT_DCPERIOD,
            HT_DCPHASE,
            HT_PHASOR,
            HT_SINE,
            HT_TRENDLINE,
            HT_TRENDMODE,
            LINEARREG,
            LINEARREG_ANGLE,
            LINEARREG_INTERCEPT,
            LINEARREG_SLOPE,
            MEDPRICE,
            STDDEV,
            TSF,
            TYPPRICE,
            VAR,
            WCLPRICE,
            # Pattern Recognition (10 selected)
            CDL2CROWS,
            CDL3BLACKCROWS,
            CDLDOJI,
            CDLDRAGONFLYDOJI,
            CDLENGULFING,
            CDLEVENINGDOJISTAR,
            CDLHAMMER,
            CDLHANGINGMAN,
            CDLHARAMI,
            CDL3WHITESOLDIERS,
        ]

        for factor_class in factor_classes:
            try:
                registry.register(factor_class)
            except Exception as e:
                logger.warning(f"注册因子{factor_class.factor_id}失败: {e}")

        logger.debug(f"已注册{len(registry.factors)}个核心因子")

    except Exception as e:
        logger.error(f"注册核心因子失败: {e}")


def calculate_factors(
    factor_ids: List[str],
    symbols: List[str],
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
    use_cache: bool = True,
    n_jobs: Optional[int] = None,
) -> pd.DataFrame:
    """
    计算因子（统一入口）

    Args:
        factor_ids: 因子ID列表
        symbols: 股票代码列表
        timeframe: 时间框架
        start_date: 开始日期
        end_date: 结束日期
        use_cache: 是否使用缓存
        n_jobs: 并行任务数（symbol维度，默认从settings获取）

    Returns:
        因子DataFrame，MultiIndex(timestamp, symbol)

    Examples:
        >>> from datetime import datetime
        >>> # 计算RSI和MACD因子
        >>> factors = calculate_factors(
        ...     factor_ids=["RSI", "MACD"],
        ...     symbols=["0700.HK", "0005.HK"],
        ...     timeframe="15min",
        ...     start_date=datetime(2025, 9, 1),
        ...     end_date=datetime(2025, 9, 30),
        ... )
        >>> print(factors.shape)  # (timestamp_count * symbol_count, factor_count)
    """
    settings = get_settings()
    if n_jobs is None:
        n_jobs = settings.engine.n_jobs

    engine = get_engine()
    return engine.calculate_factors(
        factor_ids=factor_ids,
        symbols=symbols,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        use_cache=use_cache,
        n_jobs=n_jobs,
    )


def calculate_factor_set(
    set_id: str,
    symbols: List[str],
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    计算预定义的因子集

    Args:
        set_id: 因子集ID（如 "hk_midfreq_core"）
        symbols: 股票代码列表
        timeframe: 时间框架
        start_date: 开始日期
        end_date: 结束日期
        use_cache: 是否使用缓存

    Returns:
        因子DataFrame
    """
    engine = get_engine()

    # 从注册表获取因子集
    factor_set = engine.registry.get_factor_set(set_id)
    if not factor_set:
        available_sets = engine.registry.list_factor_sets()
        raise ValueError(
            f"因子集 '{set_id}' 不存在。\n" f"可用的因子集: {available_sets}"
        )

    factor_ids = factor_set.get("factors", [])
    logger.info(f"加载因子集 '{set_id}': {len(factor_ids)}个因子")

    return engine.calculate_factors(
        factor_ids=factor_ids,
        symbols=symbols,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        use_cache=use_cache,
    )


def list_available_factors() -> List[str]:
    """
    列出所有可用因子

    Returns:
        因子ID列表
    """
    engine = get_engine()
    return list(engine.registry.factors.keys())


def get_factor_metadata(factor_id: str) -> Optional[Dict]:
    """
    获取因子元数据

    Args:
        factor_id: 因子ID

    Returns:
        元数据字典，不存在时返回None
    """
    engine = get_engine()
    return engine.registry.get_metadata(factor_id)


def prewarm_cache(
    factor_ids: List[str],
    symbols: List[str],
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
):
    """
    预热缓存

    Args:
        factor_ids: 因子ID列表
        symbols: 股票代码列表
        timeframe: 时间框架
        start_date: 开始日期
        end_date: 结束日期
    """
    engine = get_engine()
    logger.info(f"开始预热缓存: {len(factor_ids)}个因子, {len(symbols)}个标的")
    engine.prewarm_cache(factor_ids, symbols, timeframe, start_date, end_date)
    logger.info("✅ 缓存预热完成")


def clear_cache():
    """清空所有缓存"""
    engine = get_engine()
    engine.clear_cache()
    logger.info("✅ 缓存已清空")


def get_cache_stats() -> Dict:
    """
    获取缓存统计

    Returns:
        缓存统计字典
    """
    engine = get_engine()
    return engine.get_cache_stats()


# 便捷函数
def calculate_single_factor(
    factor_id: str,
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
) -> pd.Series:
    """
    计算单个因子（便捷函数）

    Args:
        factor_id: 因子ID
        symbol: 股票代码
        timeframe: 时间框架
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        因子值Series

    Examples:
        >>> from datetime import datetime
        >>> # 计算单只股票的RSI
        >>> rsi = calculate_single_factor(
        ...     factor_id="RSI",
        ...     symbol="0700.HK",
        ...     timeframe="15min",
        ...     start_date=datetime(2025, 9, 1),
        ...     end_date=datetime(2025, 9, 30),
        ... )
        >>> print(f"RSI值: {rsi.tail()}")
    """
    result = calculate_factors(
        factor_ids=[factor_id],
        symbols=[symbol],
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
    )

    if result.empty:
        return pd.Series(dtype=float)

    # 如果是MultiIndex，提取单symbol
    if isinstance(result.index, pd.MultiIndex):
        result = result.xs(symbol, level="symbol")

    return result[factor_id] if factor_id in result.columns else pd.Series(dtype=float)


def calculate_core_factors(
    symbols: List[str],
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    计算核心因子集（常用技术指标）

    Args:
        symbols: 股票代码列表
        timeframe: 时间框架
        start_date: 开始日期
        end_date: 结束日期
        use_cache: 是否使用缓存

    Returns:
        因子DataFrame，包含RSI、MACD、STOCH、WILLR、CCI、ATR等核心因子

    Examples:
        >>> from datetime import datetime
        >>> # 计算腾讯的核心技术指标
        >>> core_factors = calculate_core_factors(
        ...     symbols=["0700.HK"],
        ...     timeframe="15min",
        ...     start_date=datetime(2025, 9, 1),
        ...     end_date=datetime(2025, 9, 30),
        ... )
        >>> print(f"核心因子: {core_factors.columns.tolist()}")
    """
    core_factor_ids = [
        # 动量指标
        "RSI",
        "STOCH",
        "WILLR",
        "CCI",
        "CMO",
        "MOM",
        "ROC",
        # 趋势指标
        "ADX",
        "AROON",
        "DX",
        "PLUS_DI",
        "MINUS_DI",
        # 波动率指标
        "ATR",
        "NATR",
        "TRANGE",
        # 成交量指标
        "OBV",
        "AD",
        "ADOSC",
        "MFI",
        # 移动平均
        "SMA",
        "EMA",
        "WMA",
        "DEMA",
        "TEMA",
        "BBANDS",
    ]

    return calculate_factors(
        factor_ids=core_factor_ids,
        symbols=symbols,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        use_cache=use_cache,
    )


def calculate_momentum_factors(
    symbols: List[str],
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    计算动量因子集

    Args:
        symbols: 股票代码列表
        timeframe: 时间框架
        start_date: 开始日期
        end_date: 结束日期
        use_cache: 是否使用缓存

    Returns:
        因子DataFrame，包含所有动量相关因子
    """
    momentum_factor_ids = [
        "RSI",
        "STOCH",
        "WILLR",
        "CCI",
        "CMO",
        "MOM",
        "ROC",
        "MACD",
        "ADX",
        "ADXR",
        "AROON",
        "DX",
        "PLUS_DI",
        "MINUS_DI",
    ]

    return calculate_factors(
        factor_ids=momentum_factor_ids,
        symbols=symbols,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        use_cache=use_cache,
    )


def list_factor_categories() -> Dict[str, List[str]]:
    """
    列出所有因子类别

    Returns:
        因子类别字典，key为类别名，value为因子ID列表

    Examples:
        >>> categories = list_factor_categories()
        >>> print("技术指标:", categories['technical'])
        >>> print("移动平均:", categories['overlap'])
    """
    engine = get_engine()
    categories = {}

    # 按类别分组
    for factor_id, factor_class in engine.registry.factors.items():
        category = getattr(factor_class, "category", "unknown")
        if category not in categories:
            categories[category] = []
        categories[category].append(factor_id)

    # 排序
    for category in categories:
        categories[category].sort()

    return categories


def list_factors_by_category(category: str) -> List[str]:
    """
    列出指定类别的因子

    Args:
        category: 因子类别（如 'technical', 'overlap', 'pattern'）

    Returns:
        该类别下的因子ID列表

    Examples:
        >>> # 列出所有技术指标
        >>> technical_factors = list_factors_by_category('technical')
        >>> print(f"技术指标: {technical_factors}")
    """
    engine = get_engine()
    factors = []

    for factor_id, factor_class in engine.registry.factors.items():
        if getattr(factor_class, "category", None) == category:
            factors.append(factor_id)

    return sorted(factors)


# 异常类
class UnknownFactorError(ValueError):
    """未知因子错误"""

    def __init__(self, factor_id: str, available_factors: List[str]):
        self.factor_id = factor_id
        self.available_factors = available_factors

        similar = [f for f in available_factors if factor_id.upper() in f.upper()]

        message = (
            f"❌ 未知因子: '{factor_id}'\n\n"
            f"可用因子列表 ({len(available_factors)}个):\n"
        )

        if similar:
            message += f"  相似因子: {', '.join(similar[:5])}\n\n"

        message += f"  全部因子: {', '.join(sorted(available_factors))}"

        super().__init__(message)
