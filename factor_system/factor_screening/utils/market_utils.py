#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
市场解析工具
支持HK/US市场的股票代码解析和路径构建
"""

import logging
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)


def parse_market(symbol: str) -> str:
    """
    从股票代码解析市场
    
    Args:
        symbol: 股票代码，支持多种格式
        
    Returns:
        市场代码: 'HK' 或 'US'
        
    Examples:
        >>> parse_market('0700.HK')
        'HK'
        >>> parse_market('0700HK')
        'HK'
        >>> parse_market('AAPL.US')
        'US'
        >>> parse_market('AAPLUS')
        'US'
    """
    # 优先检查点分隔格式
    if '.' in symbol:
        market = symbol.split('.')[-1].upper()
        if market in ['HK', 'US']:
            return market
    
    # 检查后缀
    if symbol.endswith('HK'):
        return 'HK'
    elif symbol.endswith('US'):
        return 'US'
    
    # 无法解析时抛出异常
    raise ValueError(
        f"无法解析市场代码: {symbol}. "
        f"支持格式: '0700.HK', '0700HK', 'AAPL.US', 'AAPLUS'"
    )


def normalize_symbol(symbol: str) -> Tuple[str, str]:
    """
    标准化股票代码，分离代码和市场
    
    Args:
        symbol: 原始股票代码
        
    Returns:
        (clean_symbol, market)
        
    Examples:
        >>> normalize_symbol('0700.HK')
        ('0700', 'HK')
        >>> normalize_symbol('0700HK')
        ('0700', 'HK')
        >>> normalize_symbol('AAPL.US')
        ('AAPL', 'US')
        >>> normalize_symbol('AAPLUS')
        ('AAPL', 'US')
    """
    market = parse_market(symbol)
    
    # 移除市场后缀
    clean_symbol = symbol.replace(f'.{market}', '').replace(market, '')
    
    return clean_symbol, market


def construct_factor_file_path(
    data_root: Path,
    symbol: str,
    timeframe: str,
    file_suffix: str = 'factors'
) -> Path:
    """
    🔧 修复版：构建因子文件路径，支持扁平目录结构

    Args:
        data_root: 因子数据根目录
        symbol: 股票代码 (支持任意格式)
        timeframe: 时间框架 (如: 5min, 15min, 1day)
        file_suffix: 文件后缀 (默认: 'factors')

    Returns:
        完整的因子文件路径

    Examples:
        >>> construct_factor_file_path(
        ...     Path('/factor_output'),
        ...     '0700.HK',
        ...     '15min'
        ... )
        PosixPath('/factor_output/HK/0700HK_15min_factors_20251008_224251.parquet')
    """
    from .timeframe_utils import map_timeframe

    clean_symbol, market = normalize_symbol(symbol)
    market_dir = data_root / market

    if not market_dir.exists():
        raise FileNotFoundError(f"市场因子目录不存在: {market_dir}")

    # 🔧 修复：支持分层目录结构 (market/timeframe/) 和扁平结构 (market/)
    search_patterns = []
    search_dirs = []
    
    # 优先级 1：分层目录 market/timeframe/
    timeframe_dir = market_dir / timeframe
    if timeframe_dir.exists():
        search_dirs.append(timeframe_dir)
    
    # 优先级 2：扁平目录 market/
    search_dirs.append(market_dir)

    # 1. 最优先：带时间戳的标准化格式
    # 格式：0005HK_15min_factors_20251008_224251.parquet
    search_patterns.append(f"{clean_symbol}{market}_{timeframe}_{file_suffix}_*.parquet")

    # 2. 次优先：原始符号格式（保留点分隔）
    # 格式：0700.HK_15min_factors_20251008_224251.parquet
    search_patterns.append(f"{symbol}_{timeframe}_{file_suffix}_*.parquet")

    # 3. 第三优先：时间框架映射格式（15min -> 15m）
    mapped_timeframe = map_timeframe(timeframe, 'factor')
    if mapped_timeframe != timeframe:
        search_patterns.append(f"{clean_symbol}{market}_{mapped_timeframe}_{file_suffix}_*.parquet")
        search_patterns.append(f"{symbol}_{mapped_timeframe}_{file_suffix}_*.parquet")

    # 4. 回退选项：无时间戳格式
    search_patterns.append(f"{clean_symbol}{market}_{timeframe}_{file_suffix}.parquet")
    search_patterns.append(f"{symbol}_{timeframe}_{file_suffix}.parquet")

    # 在所有搜索目录中匹配文件
    for search_dir in search_dirs:
        for pattern in search_patterns:
            matching_files = list(search_dir.glob(pattern))
            if matching_files:
                # 如果有多个文件，选择最新的
                if len(matching_files) > 1:
                    matching_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    logger.debug(f"找到多个匹配文件，选择最新的: {matching_files[0].name}")
                return matching_files[0]

    # 如果都找不到，抛出详细的错误信息
    available_files = list(market_dir.glob("**/*.parquet"))  # 递归搜索所有parquet
    available_sample = [f.relative_to(market_dir) for f in available_files[:5]]  # 显示相对路径

    raise FileNotFoundError(
        f"未找到匹配的因子文件: {symbol} {timeframe}\n"
        f"搜索目录: {market_dir}\n"
        f"搜索模式: {search_patterns}\n"
        f"可用文件示例: {available_sample}"
    )


def construct_price_file_path(
    data_root: Path,
    symbol: str,
    timeframe: str
) -> Path:
    """
    构建价格数据文件路径
    
    Args:
        data_root: 价格数据根目录 (如: ../raw)
        symbol: 股票代码
        timeframe: 时间框架
        
    Returns:
        完整的价格文件路径
        
    Examples:
        >>> construct_price_file_path(
        ...     Path('/raw'), 
        ...     '0700.HK', 
        ...     '5min'
        ... )
        PosixPath('/raw/HK/0700HK_5min_2025-03-06_2025-09-02.parquet')
    """
    clean_symbol, market = normalize_symbol(symbol)
    
    # 价格文件在市场目录下，不在时间框架子目录
    market_dir = data_root / market
    
    if not market_dir.exists():
        raise FileNotFoundError(f"市场数据目录不存在: {market_dir}")
    
    # 查找匹配的价格文件 (文件名包含时间范围)
    # 格式: {symbol}{market}_{timeframe}_YYYY-MM-DD_YYYY-MM-DD.parquet
    pattern = f"{clean_symbol}{market}_{timeframe}_*.parquet"
    matching_files = list(market_dir.glob(pattern))
    
    if not matching_files:
        raise FileNotFoundError(
            f"未找到价格文件: {market_dir / pattern}"
        )
    
    # 如果有多个文件，选择最新的
    if len(matching_files) > 1:
        logger.warning(
            f"找到多个价格文件: {len(matching_files)}个，使用最新的"
        )
        matching_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    return matching_files[0]


def discover_stocks(data_root: Path, market: str = None) -> dict:
    """
    🔧 修复版：自动发现所有可用股票，支持扁平目录结构

    Args:
        data_root: 因子数据根目录
        market: 指定市场 ('HK', 'US') 或 None (全部市场)

    Returns:
        {
            'HK': ['0005HK', '0700HK', ...],
            'US': ['AAPL', 'COIN', ...]
        }
    """
    data_root = Path(data_root)
    stocks = {}

    markets = ['HK', 'US'] if market is None else [market]

    for mkt in markets:
        stocks[mkt] = set()

        # 🔧 修复：支持实际的扁平目录结构
        market_dir = data_root / mkt

        if not market_dir.exists():
            logger.warning(f"市场目录不存在: {market_dir}")
            continue

        # 🔧 修复：扫描扁平目录结构中的因子文件
        # 实际文件格式：0005HK_1min_factors_20251008_224251.parquet
        pattern_files = list(market_dir.glob('*_factors_*.parquet'))

        if not pattern_files:
            # 如果没有找到带时间戳的文件，尝试旧格式
            pattern_files = list(market_dir.glob('*_factors.parquet'))

        if not pattern_files:
            logger.warning(f"在 {market_dir} 中未找到任何因子文件")
            continue

        for file_path in pattern_files:
            try:
                # 解析股票代码：0005HK_1min_factors_20251008_224251.parquet -> 0005HK
                filename_parts = file_path.stem.split('_')
                if len(filename_parts) >= 2:
                    symbol = filename_parts[0]
                    # 验证symbol是否包含市场后缀
                    if symbol.endswith(mkt):
                        stocks[mkt].add(symbol)
                    else:
                        # 如果没有市场后缀，添加上去
                        symbol_with_market = f"{symbol}{mkt}"
                        stocks[mkt].add(symbol_with_market)
            except Exception as e:
                logger.warning(f"解析文件名失败 {file_path.name}: {e}")
                continue

        logger.info(f"🔍 发现 {mkt} 市场股票: {len(stocks[mkt])} 只")

    # 转换为排序列表
    result = {k: sorted(list(v)) for k, v in stocks.items()}

    # 记录总体发现情况
    total_stocks = sum(len(v) for v in result.values())
    logger.info(f"📊 总计发现股票: {total_stocks} 只")

    return result


def validate_symbol_format(symbol: str) -> bool:
    """
    验证股票代码格式
    
    Args:
        symbol: 股票代码
        
    Returns:
        是否为有效格式
    """
    try:
        parse_market(symbol)
        return True
    except ValueError:
        return False


def format_symbol_for_display(symbol: str) -> str:
    """
    格式化股票代码用于显示
    
    Args:
        symbol: 原始股票代码
        
    Returns:
        格式化后的代码 (统一为点分隔格式)
        
    Examples:
        >>> format_symbol_for_display('0700HK')
        '0700.HK'
        >>> format_symbol_for_display('AAPLUS')
        'AAPL.US'
    """
    try:
        clean_symbol, market = normalize_symbol(symbol)
        return f"{clean_symbol}.{market}"
    except ValueError:
        return symbol  # 无法解析时返回原值

