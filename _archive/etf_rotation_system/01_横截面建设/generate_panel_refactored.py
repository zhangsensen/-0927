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

# 导入配置类
from config.config_classes import FactorPanelConfig, OutputConfig
from tqdm import tqdm

# 配置日志（可从配置文件读取）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def validate_path(path: str, must_exist: bool = False) -> Path:
    """验证并规范化路径"""
    try:
        p = Path(path).resolve()
        if must_exist and not p.exists():
            raise FileNotFoundError(f"路径不存在: {path}")
        # 防止路径穿越
        if ".." in str(p):
            raise ValueError(f"非法路径: {path}")
        return p
    except Exception as e:
        logger.error(f"路径验证失败: {path} - {e}")
        raise


def load_config(config_path: str = None) -> FactorPanelConfig:
    """加载配置文件"""
    if config_path is None:
        # 尝试默认配置路径
        default_paths = ["config/factor_panel_config.yaml", "config/etf_config.yaml"]

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
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"未找到数据文件: {data_dir}")

    prices = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            symbol = f.stem.split("_")[0]
            df["symbol"] = symbol
            df["date"] = pd.to_datetime(df["trade_date"])

            # 处理成交量列别名
            if config.data_processing.volume_column_alias in df.columns:
                if "volume" not in df.columns:
                    df["volume"] = df[config.data_processing.volume_column_alias]

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


def calculate_factors_single(
    args: Tuple[str, pd.DataFrame, FactorPanelConfig]
) -> pd.DataFrame:
    """计算单个标的的因子（并行化单元）- 配置驱动版本"""
    symbol, symbol_data, config = args

    try:
        # 提取数据
        open_p = symbol_data["open"].values
        high = symbol_data["high"].values
        low = symbol_data["low"].values
        close = symbol_data["close"].values
        volume = symbol_data["volume"].values
        dates = symbol_data["date"].values

        # 转为Series便于向量化
        s_open = pd.Series(open_p, index=symbol_data.index)
        s_high = pd.Series(high, index=symbol_data.index)
        s_low = pd.Series(low, index=symbol_data.index)
        s_close = pd.Series(close, index=symbol_data.index)
        s_vol = pd.Series(volume, index=symbol_data.index)

        factors = pd.DataFrame(index=symbol_data.index)
        factors["date"] = dates
        factors["symbol"] = symbol

        # ========== 配置驱动的因子计算 ==========

        # 1. 动量因子
        if config.factor_enable.momentum:
            for period in config.factor_windows.momentum:
                factors[f"MOMENTUM_{period}D"] = (
                    s_close / s_close.shift(period) - 1
                ).values

        # 2. 波动率因子
        if config.factor_enable.volatility:
            ret = s_close.pct_change()
            for window in config.factor_windows.volatility:
                factors[f"VOLATILITY_{window}D"] = (
                    ret.rolling(window, min_periods=config.trading.min_periods).std()
                    * np.sqrt(config.trading.days_per_year)
                ).values

        # 3. 回撤因子
        if config.factor_enable.drawdown:
            for window in config.factor_windows.drawdown:
                rolling_max = s_close.rolling(
                    window, min_periods=config.trading.min_periods
                ).max()
                dd = (s_close - rolling_max) / rolling_max
                factors[f"DRAWDOWN_{window}D"] = dd.values

        # 4. 动量加速
        if config.factor_enable.momentum_acceleration:
            if len(config.factor_windows.momentum) >= 2:
                # 使用最短和最长的动量周期
                short_period = min(config.factor_windows.momentum)
                long_period = max(config.factor_windows.momentum)
                mom_short = s_close / s_close.shift(short_period) - 1
                mom_long = s_close / s_close.shift(long_period) - 1
                factors["MOM_ACCEL"] = (mom_short - mom_long).values

        # 5. RSI因子
        if config.factor_enable.rsi:
            for window in config.factor_windows.rsi:
                delta = s_close.diff()
                gain = (
                    delta.where(delta > 0, 0)
                    .rolling(window, min_periods=config.trading.min_periods)
                    .mean()
                )
                loss = (
                    -delta.where(delta < 0, 0)
                    .rolling(window, min_periods=config.trading.min_periods)
                    .mean()
                )
                rs = gain / (loss + config.trading.epsilon_small)
                rsi = 100 - (100 / (1 + rs))
                factors[f"RSI_{window}"] = rsi.values

        # 6. 价格位置因子
        if config.factor_enable.price_position:
            for window in config.factor_windows.price_position:
                roll_high = s_high.rolling(
                    window, min_periods=config.trading.min_periods
                ).max()
                roll_low = s_low.rolling(
                    window, min_periods=config.trading.min_periods
                ).min()
                pos = (s_close - roll_low) / (
                    roll_high - roll_low + config.trading.epsilon_small
                )
                factors[f"PRICE_POSITION_{window}D"] = pos.values

        # 7. 成交量比率因子
        if config.factor_enable.volume_ratio:
            for window in config.factor_windows.volume_ratio:
                vol_ma = s_vol.rolling(
                    window, min_periods=config.trading.min_periods
                ).mean()
                factors[f"VOLUME_RATIO_{window}D"] = (
                    s_vol / (vol_ma + config.trading.epsilon_small)
                ).values

        # 8. 隔夜跳空动量
        if config.factor_enable.overnight_return:
            prev_close = s_close.shift(1)
            factors["OVERNIGHT_RETURN"] = ((s_open - prev_close) / prev_close).values

        # 9. ATR真实波动幅度
        if config.factor_enable.atr:
            tr1 = s_high - s_low
            tr2 = (s_high - s_close.shift(1)).abs()
            tr3 = (s_low - s_close.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            factors["ATR_14"] = (
                tr.rolling(
                    config.factor_windows.atr_period,
                    min_periods=config.trading.min_periods,
                )
                .mean()
                .values
            )

        # 10. 十字星形态
        if config.factor_enable.doji_pattern:
            body = (s_close - s_open).abs()
            range_hl = s_high - s_low
            threshold = (
                config.thresholds.doji_body_threshold or config.trading.epsilon_small
            )
            factors["DOJI_PATTERN"] = (body / (range_hl + threshold)).values

        # 11. 日内波动率
        if config.factor_enable.intraday_range:
            factors["INTRA_DAY_RANGE"] = ((s_high - s_low) / s_close).values

        # 12. 看涨吞没形态
        if config.factor_enable.bullish_engulfing:
            prev_open = s_open.shift(1)
            prev_body = (s_close.shift(1) - prev_open).abs()
            curr_body = (s_close - s_open).abs()
            is_bullish = (s_close > s_open) & (s_close.shift(1) < prev_open)
            is_engulfing = (
                (curr_body > prev_body)
                & (s_close > prev_open)
                & (s_open < s_close.shift(1))
            )
            factors["BULLISH_ENGULFING"] = (
                (is_bullish & is_engulfing).astype(float).values
            )

        # 13. 锤子线反转信号
        if config.factor_enable.hammer_pattern:
            body = (s_close - s_open).abs()
            lower_shadow = s_close - s_low
            upper_shadow = s_high - s_close
            is_hammer = (
                lower_shadow > config.thresholds.hammer_lower_shadow_ratio * body
            ) & (upper_shadow < config.thresholds.hammer_upper_shadow_ratio * body)
            factors["HAMMER_PATTERN"] = is_hammer.astype(float).values

        # 14. 价格冲击
        if config.factor_enable.price_impact:
            price_change = s_close.pct_change().abs()
            vol_change = s_vol.pct_change().abs()
            factors["PRICE_IMPACT"] = (
                price_change / (vol_change + config.trading.epsilon_small)
            ).values

        # 15. 量价趋势一致性
        if config.factor_enable.volume_price_trend:
            price_dir = (s_close > s_close.shift(1)).astype(float)
            vol_dir = (s_vol > s_vol.shift(1)).astype(float)
            vpt = (
                (price_dir == vol_dir)
                .astype(float)
                .rolling(
                    config.factor_windows.vpt_trend_window,
                    min_periods=config.trading.min_periods,
                )
                .mean()
            )
            factors["VOLUME_PRICE_TREND"] = vpt.values

        # 16. 短期成交量动态 (5日)
        if config.factor_enable.vol_ma_ratio_5:
            vol_ma5 = s_vol.rolling(
                config.factor_windows.amount_surge_short,
                min_periods=config.trading.min_periods,
            ).mean()
            factors["VOL_MA_RATIO_5"] = (
                s_vol / (vol_ma5 + config.trading.epsilon_small)
            ).values

        # 17. 成交量稳定性
        if config.factor_enable.vol_volatility_20:
            vol_std = s_vol.rolling(
                config.factor_windows.vol_volatility_window,
                min_periods=config.trading.min_periods,
            ).std()
            vol_mean = s_vol.rolling(
                config.factor_windows.vol_volatility_window,
                min_periods=config.trading.min_periods,
            ).mean()
            factors["VOL_VOLATILITY_20"] = (
                vol_std / (vol_mean + config.trading.epsilon_small)
            ).values

        # 18. 真实波动率
        if config.factor_enable.true_range:
            tr1 = s_high - s_low
            tr2 = (s_high - s_close.shift(1)).abs()
            tr3 = (s_low - s_close.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            factors["TRUE_RANGE"] = (tr / s_close).values

        # 19. 买入压力
        if config.factor_enable.buy_pressure:
            factors["BUY_PRESSURE"] = (
                (s_close - s_low) / (s_high - s_low + config.trading.epsilon_small)
            ).values

        # ========== 资金流因子（需要amount数据）==========

        # 准备amount数据
        if "amount" in symbol_data.columns:
            s_amount = pd.Series(symbol_data["amount"].values, index=symbol_data.index)
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
                factors["VWAP_DEVIATION"] = (
                    (s_close - vwap) / (vwap + config.trading.epsilon_small)
                ).values

            # 21. 成交额突增
            if config.factor_enable.amount_surge_5d:
                amount_ma5 = s_amount.rolling(
                    config.factor_windows.amount_surge_short,
                    min_periods=config.trading.min_periods,
                ).mean()
                amount_ma20 = s_amount.rolling(
                    config.factor_windows.amount_surge_long,
                    min_periods=config.trading.min_periods,
                ).mean()
                factors["AMOUNT_SURGE_5D"] = (
                    amount_ma5 / (amount_ma20 + config.trading.epsilon_small) - 1
                ).values

            # 22. 量价背离
            if config.factor_enable.price_volume_div:
                price_change = s_close.pct_change()
                vol_change = s_vol.pct_change()
                pv_divergence = (
                    np.sign(price_change)
                    * vol_change.rolling(
                        config.factor_windows.price_volume_div_window,
                        min_periods=config.trading.min_periods,
                    ).mean()
                )
                factors["PRICE_VOLUME_DIV"] = pv_divergence.values

            # 23. 大单流入信号
            if config.factor_enable.large_order_signal:
                avg_price = s_amount / (s_vol + config.trading.epsilon_small)
                avg_price_change = avg_price.pct_change()
                vol_ratio = (
                    s_vol
                    / s_vol.rolling(
                        config.factor_windows.vol_ratio_window,
                        min_periods=config.trading.min_periods,
                    ).mean()
                )
                large_order = (
                    (avg_price_change > 0)
                    & (vol_ratio > config.thresholds.large_order_volume_ratio)
                ).astype(float)
                factors["LARGE_ORDER_SIGNAL"] = large_order.values

        # 24. 日内价格位置
        if config.factor_enable.intraday_position:
            price_pos = (s_close - s_low) / (
                s_high - s_low + config.trading.epsilon_small
            )
            factors["INTRADAY_POSITION"] = (
                price_pos.rolling(
                    config.factor_windows.intraday_position_window,
                    min_periods=config.trading.min_periods,
                )
                .mean()
                .values
            )

        # ========== 新增：流动性因子 ==========

        # 25. Amihud非流动性指标
        if config.factor_enable.illiquidity and s_amount is not None:
            ret_abs = s_close.pct_change().abs()
            illiq = ret_abs / (s_amount + config.trading.epsilon_small)
            factors["ILLIQUIDITY_20D"] = (
                illiq.rolling(
                    config.factor_windows.illiquidity_window,
                    min_periods=config.trading.min_periods,
                )
                .mean()
                .values
            )

        # 26. 相对换手率
        if config.factor_enable.turnover_ratio:
            turnover = s_vol / (
                s_vol.rolling(252, min_periods=60).mean() + config.trading.epsilon_small
            )
            turnover_ma = turnover.rolling(
                config.factor_windows.turnover_ma_window,
                min_periods=config.trading.min_periods,
            ).mean()
            factors["TURNOVER_MA_RATIO"] = (
                turnover / (turnover_ma + config.trading.epsilon_small)
            ).values

        # 27. 成交额变化率
        if config.factor_enable.amount_change_rate and s_amount is not None:
            amount_change = (
                s_amount - s_amount.shift(config.factor_windows.amount_change_window)
            ) / (
                s_amount.shift(config.factor_windows.amount_change_window)
                + config.trading.epsilon_small
            )
            factors["AMOUNT_CHANGE_RATE"] = amount_change.values

        # ========== 新增：微观结构因子 ==========

        # 28. 振幅因子
        if config.factor_enable.amplitude:
            amplitude = (s_high - s_low) / (s_close + config.trading.epsilon_small)
            factors["AMPLITUDE_20D"] = (
                amplitude.rolling(
                    config.factor_windows.amplitude_window,
                    min_periods=config.trading.min_periods,
                )
                .mean()
                .values
            )

        # 29. 上下影线比率
        if config.factor_enable.shadow_ratio:
            upper_shadow = s_high - pd.concat([s_open, s_close], axis=1).max(axis=1)
            lower_shadow = pd.concat([s_open, s_close], axis=1).min(axis=1) - s_low
            shadow_ratio = upper_shadow / (lower_shadow + config.trading.epsilon_small)
            factors["SHADOW_RATIO"] = shadow_ratio.values

        # 30. 涨跌天数比率
        if config.factor_enable.up_down_days_ratio:
            up_days = (s_close > s_close.shift(1)).astype(float)
            up_ratio = up_days.rolling(
                config.factor_windows.up_down_days_window,
                min_periods=config.trading.min_periods,
            ).mean()
            factors["UP_DOWN_DAYS_RATIO"] = up_ratio.values

        # ========== 新增：趋势强度因子 ==========

        # 31. 线性回归斜率
        if config.factor_enable.linear_slope:

            def calc_slope(window_data):
                if len(window_data) < 2:
                    return np.nan
                x = np.arange(len(window_data))
                try:
                    slope = np.polyfit(x, window_data, 1)[0]
                    return slope / (window_data[-1] + config.trading.epsilon_small)
                except:
                    return np.nan

            slope = s_close.rolling(
                config.factor_windows.linear_slope_window,
                min_periods=config.trading.min_periods,
            ).apply(calc_slope, raw=True)
            factors["LINEAR_SLOPE_20D"] = slope.values

        # 32. 距离年度新高
        if config.factor_enable.distance_to_high:
            roll_max = s_close.rolling(
                config.factor_windows.distance_to_high_window, min_periods=60
            ).max()
            distance = s_close / (roll_max + config.trading.epsilon_small) - 1
            factors["DISTANCE_TO_52W_HIGH"] = distance.values

        # ========== 新增：相对强弱因子 ==========

        # 33. 相对强度（这里用自身历史作为基准）
        if config.factor_enable.relative_strength_vs_index:
            ret_20d = (
                s_close / s_close.shift(config.factor_windows.relative_strength_window)
                - 1
            )
            ret_ma = ret_20d.rolling(60, min_periods=20).mean()
            factors["RELATIVE_STRENGTH_20D"] = (ret_20d - ret_ma).values

        # 34. 相对振幅
        if config.factor_enable.relative_amplitude:
            amplitude = (s_high - s_low) / (s_close + config.trading.epsilon_small)
            amp_ma = amplitude.rolling(60, min_periods=20).mean()
            factors["RELATIVE_AMPLITUDE"] = (
                amplitude / (amp_ma + config.trading.epsilon_small)
            ).values

        # ========== 新增：质量因子 ==========

        # 35. 收益质量
        if config.factor_enable.return_quality:
            ret = s_close.pct_change()
            ret_mean = ret.rolling(
                config.factor_windows.return_quality_window,
                min_periods=config.trading.min_periods,
            ).mean()
            ret_std = ret.rolling(
                config.factor_windows.return_quality_window,
                min_periods=config.trading.min_periods,
            ).std()
            factors["RETURN_QUALITY"] = (
                ret_mean / (ret_std + config.trading.epsilon_small)
            ).values

        # 36. 夏普比率
        if config.factor_enable.sharpe_ratio:
            ret = s_close.pct_change()
            ret_mean = ret.rolling(
                config.factor_windows.sharpe_ratio_window,
                min_periods=config.trading.min_periods,
            ).mean()
            ret_std = ret.rolling(
                config.factor_windows.sharpe_ratio_window,
                min_periods=config.trading.min_periods,
            ).std()
            sharpe = (
                ret_mean
                / (ret_std + config.trading.epsilon_small)
                * np.sqrt(config.trading.days_per_year)
            )
            factors["SHARPE_RATIO_60D"] = sharpe.values

        # 37. 回撤恢复速度
        if config.factor_enable.drawdown_recovery_speed:
            rolling_max = s_close.rolling(
                config.factor_windows.drawdown_recovery_window,
                min_periods=config.trading.min_periods,
            ).max()
            dd = (s_close - rolling_max) / (rolling_max + config.trading.epsilon_small)
            # 计算距离最高点的天数（简化版）
            is_new_high = (s_close >= rolling_max).astype(float)
            days_since_high = (
                (~is_new_high.astype(bool)).groupby(is_new_high.cumsum()).cumsum()
            )
            recovery_speed = 1.0 / (days_since_high + 1.0)
            factors["DRAWDOWN_RECOVERY_SPEED"] = recovery_speed.values

        # ========== 新增：经典技术指标（学术验证有效）==========

        # 38. MACD指标 (12,26,9) - 简化版：只保留HIST柱状图
        if config.factor_enable.macd:
            ema_fast = s_close.ewm(
                span=config.factor_windows.macd_fast, adjust=False
            ).mean()
            ema_slow = s_close.ewm(
                span=config.factor_windows.macd_slow, adjust=False
            ).mean()
            macd_diff = ema_fast - ema_slow
            macd_signal = macd_diff.ewm(
                span=config.factor_windows.macd_signal, adjust=False
            ).mean()
            macd_hist = macd_diff - macd_signal

            # 简化版：只输出HIST，删除DIFF和SIGNAL
            factors["MACD_HIST"] = macd_hist.values

        # 39. KDJ指标 (9,3,3)
        if config.factor_enable.kdj:
            low_n = s_low.rolling(
                config.factor_windows.kdj_n, min_periods=config.trading.min_periods
            ).min()
            high_n = s_high.rolling(
                config.factor_windows.kdj_n, min_periods=config.trading.min_periods
            ).max()
            rsv = (
                (s_close - low_n)
                / (high_n - low_n + config.trading.epsilon_small)
                * 100
            )

            # K值：RSV的M1日移动平均
            k_values = rsv.ewm(span=config.factor_windows.kdj_m1, adjust=False).mean()
            # D值：K值的M2日移动平均
            d_values = k_values.ewm(
                span=config.factor_windows.kdj_m2, adjust=False
            ).mean()
            # J值：3K - 2D
            j_values = 3 * k_values - 2 * d_values

            factors["KDJ_K"] = k_values.values
            factors["KDJ_D"] = d_values.values
            factors["KDJ_J"] = j_values.values

        # 40. 布林带指标 (20,2) - 简化版：只保留WIDTH宽度
        if config.factor_enable.bollinger_bands:
            boll_mid = s_close.rolling(
                config.factor_windows.boll_window,
                min_periods=config.trading.min_periods,
            ).mean()
            boll_std = s_close.rolling(
                config.factor_windows.boll_window,
                min_periods=config.trading.min_periods,
            ).std()
            boll_upper = boll_mid + config.factor_windows.boll_std * boll_std
            boll_lower = boll_mid - config.factor_windows.boll_std * boll_std

            # 简化版：只输出WIDTH宽度，删除POSITION位置
            boll_width = (boll_upper - boll_lower) / (
                boll_mid + config.trading.epsilon_small
            )
            factors["BOLL_WIDTH"] = boll_width.values

        # 41. 乖离率 (5,20,60)
        if config.factor_enable.bias:
            for window in config.factor_windows.bias_windows:
                ma = s_close.rolling(
                    window, min_periods=config.trading.min_periods
                ).mean()
                bias = (s_close - ma) / (ma + config.trading.epsilon_small) * 100
                factors[f"BIAS_{window}D"] = bias.values

        # 42. 威廉指标 WR (14)
        if config.factor_enable.williams_r:
            high_n = s_high.rolling(
                config.factor_windows.wr_window, min_periods=config.trading.min_periods
            ).max()
            low_n = s_low.rolling(
                config.factor_windows.wr_window, min_periods=config.trading.min_periods
            ).min()
            wr = (
                (high_n - s_close)
                / (high_n - low_n + config.trading.epsilon_small)
                * (-100)
            )
            factors["WR_14"] = wr.values

            # 43. OBV能量潮
        if config.factor_enable.obv and s_vol is not None:
            # OBV计算：价格上涨时累加成交量,下跌时累减
            price_change = s_close.diff()
            obv = (s_vol * np.sign(price_change)).cumsum()
            # OBV的移动平均变化率
            obv_ma = obv.rolling(
                config.factor_windows.obv_ma_window,
                min_periods=config.trading.min_periods,
            ).mean()
            obv_change = obv / (obv_ma + config.trading.epsilon_small) - 1
            factors["OBV_CHANGE"] = obv_change.values

        # ========== 新增：5个简单ETF因子 ==========

        # 44. 趋势一致性
        if config.factor_enable.trend_consistency:
            # 价格高于均线的比例，反映趋势稳定性
            ma = s_close.rolling(
                config.factor_windows.trend_consistency_window,
                min_periods=config.trading.min_periods,
            ).mean()
            above_ma = (s_close > ma).astype(float)
            trend_consistency = above_ma.rolling(
                config.factor_windows.trend_consistency_window,
                min_periods=config.trading.min_periods,
            ).mean()
            factors["TREND_CONSISTENCY"] = trend_consistency.values

        # 45. 极端收益频率
        if config.factor_enable.extreme_return_freq:
            # 统计超过N倍标准差的收益率出现次数
            returns = s_close.pct_change()
            ret_std = returns.rolling(
                config.factor_windows.extreme_return_window,
                min_periods=config.trading.min_periods,
            ).std()
            threshold = config.factor_windows.extreme_return_threshold * ret_std
            extreme_returns = (returns.abs() > threshold).astype(float)
            extreme_freq = extreme_returns.rolling(
                config.factor_windows.extreme_return_window,
                min_periods=config.trading.min_periods,
            ).sum()
            factors["EXTREME_RETURN_FREQ"] = extreme_freq.values

        # 46. 连续上涨天数
        if config.factor_enable.consecutive_up_days:
            # 计算连续上涨天数（正值为上涨，负值为下跌）
            returns = s_close.pct_change()
            up_days = (returns > 0).astype(int)
            down_days = (returns < 0).astype(int)

            # 计算连续序列
            consecutive = pd.Series(0, index=s_close.index)
            streak = 0
            for i in range(len(returns)):
                if pd.isna(returns.iloc[i]):
                    consecutive.iloc[i] = 0
                elif returns.iloc[i] > 0:
                    streak = streak + 1 if streak > 0 else 1
                    consecutive.iloc[i] = streak
                elif returns.iloc[i] < 0:
                    streak = streak - 1 if streak < 0 else -1
                    consecutive.iloc[i] = streak
                else:
                    streak = 0
                    consecutive.iloc[i] = 0

            factors["CONSECUTIVE_UP_DAYS"] = consecutive.values

        # 47. 量价背离强度
        if config.factor_enable.volume_price_divergence and s_vol is not None:
            # 计算价格变化与成交量变化的相关性（负相关表示背离）
            price_change = s_close.pct_change()
            vol_change = s_vol.pct_change()

            # 滚动相关性
            corr = price_change.rolling(
                config.factor_windows.volume_price_corr_window,
                min_periods=config.trading.min_periods,
            ).corr(vol_change)

            # 负相关表示背离（乘以-1使背离为正值）
            divergence = -corr
            factors["VOLUME_PRICE_DIVERGENCE"] = divergence.values

        # 48. 波动率突变
        if config.factor_enable.volatility_regime_shift:
            # 短期波动率与长期波动率之比，反映波动率状态突变
            returns = s_close.pct_change()

            vol_short = returns.rolling(
                config.factor_windows.volatility_short_window,
                min_periods=config.trading.min_periods,
            ).std()

            vol_long = returns.rolling(
                config.factor_windows.volatility_long_window,
                min_periods=config.trading.min_periods,
            ).std()

            regime_shift = vol_short / (vol_long + config.trading.epsilon_small)
            factors["VOLATILITY_REGIME_SHIFT"] = regime_shift.values

        return factors

    except Exception as e:
        logger.error(f"因子计算失败 {symbol}: {e}")
        if config.processing.continue_on_symbol_error:
            return pd.DataFrame()
        else:
            raise


def calculate_relative_rotation_factors(
    panel: pd.DataFrame, price_df: pd.DataFrame, benchmark_symbol: str = "510300.SH"
) -> pd.DataFrame:
    """
    计算横截面相对轮动因子 - ETF轮动策略的核心

    相对轮动因子关注ETF之间的相对强弱,而非绝对表现
    这才是横截面策略的本质:识别相对优异的资产并动态轮动

    Args:
        panel: 已计算的因子面板 (symbol, date) MultiIndex
        price_df: 原始价格数据
        benchmark_symbol: 基准标的(默认沪深300)

    Returns:
        包含相对轮动因子的面板
    """
    logger.info("计算横截面相对轮动因子...")

    # 重置索引以便操作
    panel_reset = panel.reset_index()

    # 获取所有日期和标的
    all_dates = sorted(panel_reset["date"].unique())
    all_symbols = sorted(panel_reset["symbol"].unique())

    if len(all_dates) < 60:
        logger.warning("数据不足60天,跳过相对轮动因子")
        return panel

    # 准备基准收益率
    benchmark_data = price_df[price_df["symbol"] == benchmark_symbol].sort_values(
        "date"
    )
    if benchmark_data.empty:
        # 使用等权平均作为基准
        logger.info(f"基准{benchmark_symbol}不存在,使用等权平均")
        benchmark_returns = {}
        for date in all_dates:
            date_data = price_df[price_df["date"] == date]
            if len(date_data) > 1:
                mean_ret = (
                    date_data.groupby("symbol")["close"].last().pct_change().mean()
                )
                benchmark_returns[date] = mean_ret if not pd.isna(mean_ret) else 0
    else:
        benchmark_rets = benchmark_data["close"].pct_change()
        benchmark_returns = dict(zip(benchmark_data["date"], benchmark_rets))

    # 为每个标的计算相对轮动因子
    rotation_factors = []

    for symbol in all_symbols:
        symbol_data = price_df[price_df["symbol"] == symbol].sort_values("date")
        if len(symbol_data) < 60:
            continue

        symbol_data = symbol_data.reset_index(drop=True)
        closes = symbol_data["close"].values
        dates = symbol_data["date"].values
        returns = np.zeros(len(closes))
        returns[1:] = (closes[1:] / closes[:-1]) - 1

        # 1. 相对动量20日/60日
        rel_mom_20 = np.zeros(len(closes))
        rel_mom_60 = np.zeros(len(closes))

        for i in range(20, len(closes)):
            etf_ret_20 = (closes[i] / closes[i - 20]) - 1
            bench_ret_20 = sum(
                [
                    benchmark_returns.get(d, 0)
                    for d in dates[i - 20 : i + 1]
                    if d in benchmark_returns
                ]
            )
            rel_mom_20[i] = etf_ret_20 - bench_ret_20

        for i in range(60, len(closes)):
            etf_ret_60 = (closes[i] / closes[i - 60]) - 1
            bench_ret_60 = sum(
                [
                    benchmark_returns.get(d, 0)
                    for d in dates[i - 60 : i + 1]
                    if d in benchmark_returns
                ]
            )
            rel_mom_60[i] = etf_ret_60 - bench_ret_60

        # 2. 横截面排名 (简化版 - 使用20日动量直接计算)
        # 注：完整横截面排名需要所有ETF数据，此处简化为相对动量的归一化
        cs_rank = np.zeros(len(closes))
        cs_rank_change = np.zeros(len(closes))

        # 使用相对动量作为排名代理（避免嵌套循环）
        for i in range(20, len(closes)):
            # 使用20日相对动量作为排名指标
            if i >= 60:
                # 计算60日窗口内的排名百分位（相对自己的历史）
                recent_mom = rel_mom_20[i - 60 : i + 1]
                if len(recent_mom) > 0:
                    sorted_mom = np.sort(recent_mom)
                    rank = np.searchsorted(sorted_mom, rel_mom_20[i])
                    cs_rank[i] = rank / len(sorted_mom) if len(sorted_mom) > 0 else 0.5

            # 排名变化
            if i >= 25:
                cs_rank_change[i] = cs_rank[i] - cs_rank[i - 5]

        # 3. 波动率调整超额收益
        vol_adj_excess = np.zeros(len(closes))
        for i in range(60, len(closes)):
            excess = rel_mom_60[i]
            vol = np.std(returns[i - 60 : i]) * np.sqrt(252)
            vol_adj_excess[i] = excess / vol if vol > 0 else 0

        # 4. 相对强度偏离(均值回归信号)
        rs_deviation = np.zeros(len(closes))
        epsilon = 1e-9  # 防止除零的最小值
        for i in range(60, len(closes)):
            # 相对强度 = ETF收益 / 基准收益
            recent_rs = []
            for j in range(max(0, i - 60), i):
                etf_r = returns[j]
                bench_r = benchmark_returns.get(dates[j], epsilon)
                # 使用epsilon保护，避免除以过小的值
                if abs(bench_r) > epsilon:
                    recent_rs.append(etf_r / bench_r)

            if len(recent_rs) > 10:
                mean_rs = np.mean(recent_rs)
                std_rs = np.std(recent_rs)
                current_rs = returns[i] / max(
                    abs(benchmark_returns.get(dates[i], epsilon)), epsilon
                )
                rs_deviation[i] = (
                    (current_rs - mean_rs) / std_rs if std_rs > epsilon else 0
                )

        # 保存因子
        for i, date in enumerate(dates):
            if i >= 60:
                rotation_factors.append(
                    {
                        "symbol": symbol,
                        "date": date,
                        "RELATIVE_MOMENTUM_20D": rel_mom_20[i],
                        "RELATIVE_MOMENTUM_60D": rel_mom_60[i],
                        "CS_RANK_PERCENTILE": cs_rank[i],
                        "CS_RANK_CHANGE_5D": cs_rank_change[i],
                        "VOL_ADJUSTED_EXCESS": vol_adj_excess[i],
                        "RS_DEVIATION": rs_deviation[i],
                    }
                )

    if not rotation_factors:
        logger.warning("相对轮动因子计算失败,返回原面板")
        return panel

    # 转为DataFrame并计算综合轮动得分
    rotation_df = pd.DataFrame(rotation_factors)

    # Z-score标准化
    for col in [
        "RELATIVE_MOMENTUM_20D",
        "RELATIVE_MOMENTUM_60D",
        "CS_RANK_CHANGE_5D",
        "VOL_ADJUSTED_EXCESS",
        "RS_DEVIATION",
    ]:
        if col in rotation_df.columns:
            mean_val = rotation_df[col].mean()
            std_val = rotation_df[col].std()
            if std_val > 1e-9:
                rotation_df[f"{col}_ZSCORE"] = (rotation_df[col] - mean_val) / std_val
            else:
                rotation_df[f"{col}_ZSCORE"] = 0.0

    # 综合轮动得分 = 加权Z-score
    # 相对动量60% + 排名变化20% + 波动率调整10% + RS偏离10%
    rotation_df["ROTATION_SCORE"] = (
        0.30 * rotation_df["RELATIVE_MOMENTUM_20D_ZSCORE"]
        + 0.30 * rotation_df["RELATIVE_MOMENTUM_60D_ZSCORE"]
        + 0.20 * rotation_df["CS_RANK_CHANGE_5D_ZSCORE"]
        + 0.10 * rotation_df["VOL_ADJUSTED_EXCESS_ZSCORE"]
        + 0.10 * rotation_df["RS_DEVIATION_ZSCORE"]
    )

    # 合并到原面板
    rotation_df = rotation_df.set_index(["symbol", "date"])
    panel_merged = panel.join(rotation_df, how="left")

    logger.info(f"✅ 相对轮动因子计算完成: 新增 {len(rotation_df.columns)} 个因子")
    return panel_merged


def calculate_factors_parallel(
    price_df: pd.DataFrame, config: FactorPanelConfig
) -> pd.DataFrame:
    """并行计算因子"""
    symbols = sorted(price_df["symbol"].unique())
    logger.info(
        f"并行计算因子: {len(symbols)} 个标的, {config.processing.max_workers} 个进程"
    )

    # 准备任务
    tasks = []
    for symbol in symbols:
        symbol_data = price_df[price_df["symbol"] == symbol].sort_values("date")
        tasks.append((symbol, symbol_data, config))

    # 并行执行
    factors_list = []
    failed_symbols = []

    with ProcessPoolExecutor(max_workers=config.processing.max_workers) as executor:
        futures = {
            executor.submit(calculate_factors_single, task): task[0] for task in tasks
        }

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
            raise ValueError(
                f"失败率 {failure_rate:.2%} 超过阈值 {config.processing.max_failure_rate:.2%}"
            )

    if not factors_list:
        raise ValueError("无有效因子数据")

    panel = pd.concat(factors_list, ignore_index=True)
    panel = panel.set_index(["symbol", "date"]).sort_index()

    # 🔥 新增：计算横截面相对轮动因子（ETF轮动策略的核心）
    panel = calculate_relative_rotation_factors(panel, price_df)

    return panel


def save_results(
    panel: pd.DataFrame, output_dir: Path, config: OutputConfig
) -> Tuple[str, str]:
    """保存结果 - 可配置的时间戳子目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 创建输出目录
    if config.timestamp_subdirectory:
        timestamp_dir = output_dir / f"panel_{timestamp}"
    else:
        timestamp_dir = output_dir

    timestamp_dir.mkdir(parents=True, exist_ok=True)

    # 保存面板
    panel_file = timestamp_dir / "panel.parquet"
    panel.to_parquet(panel_file)
    logger.info(f"面板已保存: {panel_file}")

    # 保存元数据
    if config.save_metadata:
        meta = {
            "timestamp": timestamp,
            "etf_count": panel.index.get_level_values("symbol").nunique(),
            "factor_count": len(panel.columns),
            "data_points": len(panel),
            "coverage_rate": float(panel.notna().mean().mean()),
            "factors": panel.columns.tolist(),
            "date_range": {
                "start": str(panel.index.get_level_values("date").min().date()),
                "end": str(panel.index.get_level_values("date").max().date()),
            },
            "files": {"panel": str(panel_file), "directory": str(timestamp_dir)},
        }

        meta_file = timestamp_dir / "metadata.json"
        with open(meta_file, "w") as f:
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
            log_file = timestamp_dir / "execution_log.txt"
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(f"ETF横截面因子面板生成执行日志\\n")
                f.write(f"执行时间: {timestamp}\\n")
                f.write(f"标的数: {meta['etf_count']}\\n")
                f.write(f"因子数: {meta['factor_count']}\\n")
                f.write(f"数据点: {meta['data_points']}\\n")
                f.write(f"覆盖率: {meta['coverage_rate']:.2%}\\n")
                f.write(
                    f"时间范围: {meta['date_range']['start']} 至 {meta['date_range']['end']}\\n"
                )
                f.write(f"\\n因子列表:\\n")
                for i, factor in enumerate(meta["factors"], 1):
                    f.write(f"  {i:2d}. {factor}\\n")

            logger.info(f"执行日志已保存: {log_file}")

        return str(panel_file), str(meta_file)

    return str(panel_file), ""


def generate_etf_panel(
    data_dir: str, output_dir: str, config_path: str = None, max_workers: int = None
) -> Tuple[str, str]:
    """生成ETF因子面板（主函数）- 配置驱动版本"""
    logger.info("=" * 80)
    logger.info("ETF轮动因子面板生成 - 配置驱动版本")
    logger.info("=" * 80)

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ETF因子面板生成 - 配置驱动版本")
    parser.add_argument("--data-dir", help="数据目录")
    parser.add_argument("--output-dir", help="输出目录")
    parser.add_argument("--config", help="配置文件路径")
    parser.add_argument("--workers", type=int, help="并行进程数")

    args = parser.parse_args()

    try:
        # 使用配置中的默认值，如果命令行未提供
        config = load_config(args.config)

        data_dir = args.data_dir or config.paths.data_dir
        output_dir = args.output_dir or config.paths.output_dir
        max_workers = args.workers or config.processing.max_workers

        panel_file, meta_file = generate_etf_panel(
            data_dir, output_dir, args.config, max_workers
        )
        logger.info("✅ 完成")
    except Exception as e:
        logger.error(f"❌ 失败: {e}")
        exit(1)
