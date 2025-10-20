#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ETF轮动因子面板生成 - 重构版本（使用配置驱动）"""
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# 导入配置类
from config.config_classes import FactorPanelConfig, OutputConfig

# 配置日志（可从配置文件读取）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def validate_path(path: str, must_exist: bool = False) -> Path:
    """验证并规范化路径"""
    try:
        p = Path(path).resolve()
        if must_exist and not p.exists():
            raise FileNotFoundError(f"路径不存在: {path}")
        # 防止路径穿越
        if '..' in str(p):
            raise ValueError(f"非法路径: {path}")
        return p
    except Exception as e:
        logger.error(f"路径验证失败: {path} - {e}")
        raise


def load_config(config_path: str = None) -> FactorPanelConfig:
    """加载配置文件"""
    if config_path is None:
        # 尝试默认配置路径
        default_paths = [
            "config/factor_panel_config.yaml",
            "config/etf_config.yaml"
        ]

        for path in default_paths:
            if Path(path).exists():
                config_path = path
                break

    if config_path and Path(config_path).exists():
        logger.info(f"加载配置文件: {config_path}")
        config = FactorPanelConfig.from_yaml(config_path)
    else:
        logger.warning("未找到配置文件，使用默认配置")
        config = FactorPanelConfig()

    # 验证配置
    if not config.validate():
        raise ValueError("配置验证失败")

    return config


def load_price_data(data_dir: Path, config: FactorPanelConfig) -> pd.DataFrame:
    """加载价格数据（完整OHLCV）"""
    logger.info(f"加载价格数据: {data_dir}")
    files = sorted(data_dir.glob('*.parquet'))
    if not files:
        raise FileNotFoundError(f"未找到数据文件: {data_dir}")

    prices = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            symbol = f.stem.split('_')[0]
            df['symbol'] = symbol
            df['date'] = pd.to_datetime(df['trade_date'])

            # 处理成交量列别名
            if config.data_processing.volume_column_alias in df.columns:
                if 'volume' not in df.columns:
                    df['volume'] = df[config.data_processing.volume_column_alias]

            # 检查必需列
            required_cols = set(config.data_processing.required_columns)
            available_cols = set(df.columns)

            missing_required = required_cols - available_cols
            if missing_required:
                logger.warning(f"文件 {f.name} 缺少必需列: {missing_required}")
                continue

            # 选择列
            selected_cols = []
            for col in config.data_processing.required_columns:
                if col in df.columns:
                    selected_cols.append(col)

            for col in config.data_processing.optional_columns:
                if col in df.columns:
                    selected_cols.append(col)

            prices.append(df[selected_cols])

        except Exception as e:
            logger.error(f"加载失败 {f.name}: {e}")
            continue

    if not prices:
        raise ValueError("无有效数据")

    price_df = pd.concat(prices, ignore_index=True)
    logger.info(f"加载完成: {len(prices)} 个标的, {len(price_df)} 条记录")
    return price_df


def calculate_factors_single(args: Tuple[str, pd.DataFrame, FactorPanelConfig]) -> pd.DataFrame:
    """计算单个标的的因子（并行化单元）- 配置驱动版本"""
    symbol, symbol_data, config = args

    try:
        # 提取数据
        open_p = symbol_data['open'].values
        high = symbol_data['high'].values
        low = symbol_data['low'].values
        close = symbol_data['close'].values
        volume = symbol_data['volume'].values
        dates = symbol_data['date'].values

        # 转为Series便于向量化
        s_open = pd.Series(open_p, index=symbol_data.index)
        s_high = pd.Series(high, index=symbol_data.index)
        s_low = pd.Series(low, index=symbol_data.index)
        s_close = pd.Series(close, index=symbol_data.index)
        s_vol = pd.Series(volume, index=symbol_data.index)

        factors = pd.DataFrame(index=symbol_data.index)
        factors['date'] = dates
        factors['symbol'] = symbol

        # ========== 配置驱动的因子计算 ==========

        # 1. 动量因子
        if config.factor_enable.momentum:
            for period in config.factor_windows.momentum:
                factors[f'MOMENTUM_{period}D'] = (s_close / s_close.shift(period) - 1).values

        # 2. 波动率因子
        if config.factor_enable.volatility:
            ret = s_close.pct_change()
            for window in config.factor_windows.volatility:
                factors[f'VOLATILITY_{window}D'] = (
                    ret.rolling(window, min_periods=config.trading.min_periods).std() *
                    np.sqrt(config.trading.days_per_year)
                ).values

        # 3. 回撤因子
        if config.factor_enable.drawdown:
            for window in config.factor_windows.drawdown:
                rolling_max = s_close.rolling(window, min_periods=config.trading.min_periods).max()
                dd = (s_close - rolling_max) / rolling_max
                factors[f'DRAWDOWN_{window}D'] = dd.values

        # 4. 动量加速
        if config.factor_enable.momentum_acceleration:
            if len(config.factor_windows.momentum) >= 2:
                # 使用最短和最长的动量周期
                short_period = min(config.factor_windows.momentum)
                long_period = max(config.factor_windows.momentum)
                mom_short = s_close / s_close.shift(short_period) - 1
                mom_long = s_close / s_close.shift(long_period) - 1
                factors['MOM_ACCEL'] = (mom_short - mom_long).values

        # 5. RSI因子
        if config.factor_enable.rsi:
            for window in config.factor_windows.rsi:
                delta = s_close.diff()
                gain = delta.where(delta > 0, 0).rolling(window, min_periods=config.trading.min_periods).mean()
                loss = -delta.where(delta < 0, 0).rolling(window, min_periods=config.trading.min_periods).mean()
                rs = gain / (loss + config.trading.epsilon_small)
                rsi = 100 - (100 / (1 + rs))
                factors[f'RSI_{window}'] = rsi.values

        # 6. 价格位置因子
        if config.factor_enable.price_position:
            for window in config.factor_windows.price_position:
                roll_high = s_high.rolling(window, min_periods=config.trading.min_periods).max()
                roll_low = s_low.rolling(window, min_periods=config.trading.min_periods).min()
                pos = (s_close - roll_low) / (roll_high - roll_low + config.trading.epsilon_small)
                factors[f'PRICE_POSITION_{window}D'] = pos.values

        # 7. 成交量比率因子
        if config.factor_enable.volume_ratio:
            for window in config.factor_windows.volume_ratio:
                vol_ma = s_vol.rolling(window, min_periods=config.trading.min_periods).mean()
                factors[f'VOLUME_RATIO_{window}D'] = (s_vol / (vol_ma + config.trading.epsilon_small)).values

        # 8. 隔夜跳空动量
        if config.factor_enable.overnight_return:
            prev_close = s_close.shift(1)
            factors['OVERNIGHT_RETURN'] = ((s_open - prev_close) / prev_close).values

        # 9. ATR真实波动幅度
        if config.factor_enable.atr:
            tr1 = s_high - s_low
            tr2 = (s_high - s_close.shift(1)).abs()
            tr3 = (s_low - s_close.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            factors['ATR_14'] = tr.rolling(config.factor_windows.atr_period, min_periods=config.trading.min_periods).mean().values

        # 10. 十字星形态
        if config.factor_enable.doji_pattern:
            body = (s_close - s_open).abs()
            range_hl = s_high - s_low
            threshold = config.thresholds.doji_body_threshold or config.trading.epsilon_small
            factors['DOJI_PATTERN'] = (body / (range_hl + threshold)).values

        # 11. 日内波动率
        if config.factor_enable.intraday_range:
            factors['INTRA_DAY_RANGE'] = ((s_high - s_low) / s_close).values

        # 12. 看涨吞没形态
        if config.factor_enable.bullish_engulfing:
            prev_open = s_open.shift(1)
            prev_body = (s_close.shift(1) - prev_open).abs()
            curr_body = (s_close - s_open).abs()
            is_bullish = (s_close > s_open) & (s_close.shift(1) < prev_open)
            is_engulfing = (curr_body > prev_body) & (s_close > prev_open) & (s_open < s_close.shift(1))
            factors['BULLISH_ENGULFING'] = (is_bullish & is_engulfing).astype(float).values

        # 13. 锤子线反转信号
        if config.factor_enable.hammer_pattern:
            body = (s_close - s_open).abs()
            lower_shadow = s_close - s_low
            upper_shadow = s_high - s_close
            is_hammer = (lower_shadow > config.thresholds.hammer_lower_shadow_ratio * body) & \
                       (upper_shadow < config.thresholds.hammer_upper_shadow_ratio * body)
            factors['HAMMER_PATTERN'] = is_hammer.astype(float).values

        # 14. 价格冲击
        if config.factor_enable.price_impact:
            price_change = s_close.pct_change().abs()
            vol_change = s_vol.pct_change().abs()
            factors['PRICE_IMPACT'] = (price_change / (vol_change + config.trading.epsilon_small)).values

        # 15. 量价趋势一致性
        if config.factor_enable.volume_price_trend:
            price_dir = (s_close > s_close.shift(1)).astype(float)
            vol_dir = (s_vol > s_vol.shift(1)).astype(float)
            vpt = (price_dir == vol_dir).astype(float).rolling(
                config.factor_windows.vpt_trend_window,
                min_periods=config.trading.min_periods
            ).mean()
            factors['VOLUME_PRICE_TREND'] = vpt.values

        # 16. 短期成交量动态 (5日)
        if config.factor_enable.vol_ma_ratio_5:
            vol_ma5 = s_vol.rolling(config.factor_windows.amount_surge_short, min_periods=config.trading.min_periods).mean()
            factors['VOL_MA_RATIO_5'] = (s_vol / (vol_ma5 + config.trading.epsilon_small)).values

        # 17. 成交量稳定性
        if config.factor_enable.vol_volatility_20:
            vol_std = s_vol.rolling(config.factor_windows.vol_volatility_window, min_periods=config.trading.min_periods).std()
            vol_mean = s_vol.rolling(config.factor_windows.vol_volatility_window, min_periods=config.trading.min_periods).mean()
            factors['VOL_VOLATILITY_20'] = (vol_std / (vol_mean + config.trading.epsilon_small)).values

        # 18. 真实波动率
        if config.factor_enable.true_range:
            tr1 = s_high - s_low
            tr2 = (s_high - s_close.shift(1)).abs()
            tr3 = (s_low - s_close.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            factors['TRUE_RANGE'] = (tr / s_close).values

        # 19. 买入压力
        if config.factor_enable.buy_pressure:
            factors['BUY_PRESSURE'] = ((s_close - s_low) / (s_high - s_low + config.trading.epsilon_small)).values

        # ========== 资金流因子（需要amount数据）==========

        # 准备amount数据
        if 'amount' in symbol_data.columns:
            s_amount = pd.Series(symbol_data['amount'].values, index=symbol_data.index)
        elif config.data_processing.fallback_estimation:
            # 使用volume * close估算
            s_amount = s_vol * s_close
            logger.debug(f"{symbol}: 使用成交量*收盘价估算成交额")
        else:
            logger.debug(f"{symbol}: 无成交额数据，跳过资金流因子")
            s_amount = None

        if s_amount is not None:
            # 20. VWAP偏离度
            if config.factor_enable.vwap_deviation:
                vwap = s_amount / (s_vol + config.trading.epsilon_small)
                factors['VWAP_DEVIATION'] = ((s_close - vwap) / (vwap + config.trading.epsilon_small)).values

            # 21. 成交额突增
            if config.factor_enable.amount_surge_5d:
                amount_ma5 = s_amount.rolling(config.factor_windows.amount_surge_short, min_periods=config.trading.min_periods).mean()
                amount_ma20 = s_amount.rolling(config.factor_windows.amount_surge_long, min_periods=config.trading.min_periods).mean()
                factors['AMOUNT_SURGE_5D'] = (amount_ma5 / (amount_ma20 + config.trading.epsilon_small) - 1).values

            # 22. 量价背离
            if config.factor_enable.price_volume_div:
                price_change = s_close.pct_change()
                vol_change = s_vol.pct_change()
                pv_divergence = np.sign(price_change) * vol_change.rolling(
                    config.factor_windows.price_volume_div_window,
                    min_periods=config.trading.min_periods
                ).mean()
                factors['PRICE_VOLUME_DIV'] = pv_divergence.values

            # 23. 大单流入信号
            if config.factor_enable.large_order_signal:
                avg_price = s_amount / (s_vol + config.trading.epsilon_small)
                avg_price_change = avg_price.pct_change()
                vol_ratio = s_vol / s_vol.rolling(config.factor_windows.vol_ratio_window, min_periods=config.trading.min_periods).mean()
                large_order = ((avg_price_change > 0) &
                             (vol_ratio > config.thresholds.large_order_volume_ratio)).astype(float)
                factors['LARGE_ORDER_SIGNAL'] = large_order.values

        # 24. 日内价格位置
        if config.factor_enable.intraday_position:
            price_pos = (s_close - s_low) / (s_high - s_low + config.trading.epsilon_small)
            factors['INTRADAY_POSITION'] = price_pos.rolling(
                config.factor_windows.intraday_position_window,
                min_periods=config.trading.min_periods
            ).mean().values

        return factors

    except Exception as e:
        logger.error(f"因子计算失败 {symbol}: {e}")
        if config.processing.continue_on_symbol_error:
            return pd.DataFrame()
        else:
            raise


def calculate_factors_parallel(
    price_df: pd.DataFrame,
    config: FactorPanelConfig
) -> pd.DataFrame:
    """并行计算因子"""
    symbols = sorted(price_df['symbol'].unique())
    logger.info(f"并行计算因子: {len(symbols)} 个标的, {config.processing.max_workers} 个进程")

    # 准备任务
    tasks = []
    for symbol in symbols:
        symbol_data = price_df[price_df['symbol'] == symbol].sort_values('date')
        tasks.append((symbol, symbol_data, config))

    # 并行执行
    factors_list = []
    failed_symbols = []

    with ProcessPoolExecutor(max_workers=config.processing.max_workers) as executor:
        futures = {executor.submit(calculate_factors_single, task): task[0]
                   for task in tasks}

        for future in tqdm(as_completed(futures), total=len(futures), desc="计算因子"):
            symbol = futures[future]
            try:
                result = future.result()
                if not result.empty:
                    factors_list.append(result)
                else:
                    failed_symbols.append(symbol)
            except Exception as e:
                logger.error(f"任务失败 {symbol}: {e}")
                failed_symbols.append(symbol)

    # 检查失败率
    if failed_symbols:
        failure_rate = len(failed_symbols) / len(symbols)
        logger.warning(f"失败的标的: {failed_symbols}, 失败率: {failure_rate:.2%}")

        if failure_rate > config.processing.max_failure_rate:
            raise ValueError(f"失败率 {failure_rate:.2%} 超过阈值 {config.processing.max_failure_rate:.2%}")

    if not factors_list:
        raise ValueError("无有效因子数据")

    panel = pd.concat(factors_list, ignore_index=True)
    panel = panel.set_index(['symbol', 'date']).sort_index()

    return panel


def save_results(panel: pd.DataFrame, output_dir: Path, config: OutputConfig) -> Tuple[str, str]:
    """保存结果 - 可配置的时间戳子目录"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 创建输出目录
    if config.timestamp_subdirectory:
        timestamp_dir = output_dir / f'panel_{timestamp}'
    else:
        timestamp_dir = output_dir

    timestamp_dir.mkdir(parents=True, exist_ok=True)

    # 保存面板
    panel_file = timestamp_dir / 'panel.parquet'
    panel.to_parquet(panel_file)
    logger.info(f"面板已保存: {panel_file}")

    # 保存元数据
    if config.save_metadata:
        meta = {
            'timestamp': timestamp,
            'etf_count': panel.index.get_level_values('symbol').nunique(),
            'factor_count': len(panel.columns),
            'data_points': len(panel),
            'coverage_rate': float(panel.notna().mean().mean()),
            'factors': panel.columns.tolist(),
            'date_range': {
                'start': str(panel.index.get_level_values('date').min().date()),
                'end': str(panel.index.get_level_values('date').max().date())
            },
            'files': {
                'panel': str(panel_file),
                'directory': str(timestamp_dir)
            }
        }

        meta_file = timestamp_dir / 'metadata.json'
        with open(meta_file, 'w') as f:
            json.dump(meta, f, indent=2)
        logger.info(f"元数据已保存: {meta_file}")

        # 输出统计
        logger.info(f"面板统计:")
        logger.info(f"  标的数: {meta['etf_count']}")
        logger.info(f"  因子数: {meta['factor_count']}")
        logger.info(f"  数据点: {meta['data_points']}")
        logger.info(f"  覆盖率: {meta['coverage_rate']:.2%}")
        logger.info(f"  保存目录: {timestamp_dir}")

        # 创建执行日志文件
        if config.save_execution_log:
            log_file = timestamp_dir / 'execution_log.txt'
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"ETF横截面因子面板生成执行日志\\n")
                f.write(f"执行时间: {timestamp}\\n")
                f.write(f"标的数: {meta['etf_count']}\\n")
                f.write(f"因子数: {meta['factor_count']}\\n")
                f.write(f"数据点: {meta['data_points']}\\n")
                f.write(f"覆盖率: {meta['coverage_rate']:.2%}\\n")
                f.write(f"时间范围: {meta['date_range']['start']} 至 {meta['date_range']['end']}\\n")
                f.write(f"\\n因子列表:\\n")
                for i, factor in enumerate(meta['factors'], 1):
                    f.write(f"  {i:2d}. {factor}\\n")

            logger.info(f"执行日志已保存: {log_file}")

        return str(panel_file), str(meta_file)

    return str(panel_file), ""


def generate_etf_panel(
    data_dir: str,
    output_dir: str,
    config_path: str = None,
    max_workers: int = None
) -> Tuple[str, str]:
    """生成ETF因子面板（主函数）- 配置驱动版本"""
    logger.info("="*80)
    logger.info("ETF轮动因子面板生成 - 配置驱动版本")
    logger.info("="*80)

    try:
        # 加载配置
        config = load_config(config_path)

        # 命令行参数覆盖配置
        if max_workers is not None:
            config.processing.max_workers = max_workers

        # 验证路径
        data_dir_path = validate_path(data_dir, must_exist=True)
        output_dir_path = validate_path(output_dir)

        # 加载价格数据
        price_df = load_price_data(data_dir_path, config)

        # 并行计算因子
        panel = calculate_factors_parallel(price_df, config)

        # 保存结果
        return save_results(panel, output_dir_path, config.output)

    except Exception as e:
        logger.error(f"面板生成失败: {e}")
        raise


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='ETF因子面板生成 - 配置驱动版本')
    parser.add_argument('--data-dir', help='数据目录')
    parser.add_argument('--output-dir', help='输出目录')
    parser.add_argument('--config', help='配置文件路径')
    parser.add_argument('--workers', type=int, help='并行进程数')

    args = parser.parse_args()

    try:
        # 使用配置中的默认值，如果命令行未提供
        config = load_config(args.config)

        data_dir = args.data_dir or config.paths.data_dir
        output_dir = args.output_dir or config.paths.output_dir
        max_workers = args.workers or config.processing.max_workers

        panel_file, meta_file = generate_etf_panel(
            data_dir,
            output_dir,
            args.config,
            max_workers
        )
        logger.info("✅ 完成")
    except Exception as e:
        logger.error(f"❌ 失败: {e}")
        exit(1)