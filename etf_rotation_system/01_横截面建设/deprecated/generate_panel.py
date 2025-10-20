#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ETF轮动因子面板生成 - 生产级"""
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

# 配置日志
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


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"配置加载失败，使用默认配置: {e}")
        return {
            "factor_generation": {
                "momentum_periods": [20, 63, 126, 252],
                "volatility_windows": [20, 60, 120],
                "rsi_windows": [6, 14, 24],
                "price_position_windows": [20, 60, 120],
                "volume_ratio_windows": [20, 60],
            }
        }


def load_price_data(data_dir: Path) -> pd.DataFrame:
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

            if "vol" in df.columns and "volume" not in df.columns:
                df["volume"] = df["vol"]

            # 完整OHLCV + amount数据
            required_cols = ["date", "open", "high", "low", "close", "volume", "symbol"]
            if "amount" in df.columns:
                required_cols.append("amount")
            prices.append(df[required_cols])
        except Exception as e:
            logger.error(f"加载失败 {f.name}: {e}")
            continue

    if not prices:
        raise ValueError("无有效数据")

    price_df = pd.concat(prices, ignore_index=True)
    logger.info(f"加载完成: {len(prices)} 个标的, {len(price_df)} 条记录")
    return price_df


def calculate_factors_single(args: Tuple[str, pd.DataFrame, Dict]) -> pd.DataFrame:
    """计算单个标的的因子（并行化单元）- 35个因子"""
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

        # ========== 原有18个因子 ==========

        # 动量因子 (4个)
        for period in config["momentum_periods"]:
            factors[f"MOMENTUM_{period}D"] = (
                s_close / s_close.shift(period) - 1
            ).values

        # 波动率因子 (3个)
        ret = s_close.pct_change()
        for window in config["volatility_windows"]:
            factors[f"VOLATILITY_{window}D"] = (
                ret.rolling(window, min_periods=1).std() * np.sqrt(252)
            ).values

        # 🔧 修复回撤因子 (2个) - 添加min_periods=1
        for window in [63, 126]:
            rolling_max = s_close.rolling(window, min_periods=1).max()
            dd = (s_close - rolling_max) / rolling_max
            factors[f"DRAWDOWN_{window}D"] = dd.values

        # 动量加速 (1个)
        mom_short = s_close / s_close.shift(63) - 1
        mom_long = s_close / s_close.shift(252) - 1
        factors["MOM_ACCEL"] = (mom_short - mom_long).values

        # RSI (3个)
        for window in config["rsi_windows"]:
            delta = s_close.diff()
            gain = delta.where(delta > 0, 0).rolling(window, min_periods=1).mean()
            loss = -delta.where(delta < 0, 0).rolling(window, min_periods=1).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            factors[f"RSI_{window}"] = rsi.values

        # 价格位置 (3个)
        for window in config["price_position_windows"]:
            roll_high = s_high.rolling(window, min_periods=1).max()
            roll_low = s_low.rolling(window, min_periods=1).min()
            pos = (s_close - roll_low) / (roll_high - roll_low + 1e-10)
            factors[f"PRICE_POSITION_{window}D"] = pos.values

        # 成交量比率 (2个)
        for window in config["volume_ratio_windows"]:
            vol_ma = s_vol.rolling(window, min_periods=1).mean()
            factors[f"VOLUME_RATIO_{window}D"] = (s_vol / (vol_ma + 1e-10)).values

        # ========== 新增12个因子 ==========

        # 1. OVERNIGHT_RETURN - 隔夜跳空动量
        prev_close = s_close.shift(1)
        factors["OVERNIGHT_RETURN"] = ((s_open - prev_close) / prev_close).values

        # 2. ATR_14 - 真实波动幅度
        tr1 = s_high - s_low
        tr2 = (s_high - s_close.shift(1)).abs()
        tr3 = (s_low - s_close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        factors["ATR_14"] = tr.rolling(14, min_periods=1).mean().values

        # 3. DOJI_PATTERN - 十字星形态
        body = (s_close - s_open).abs()
        range_hl = s_high - s_low
        factors["DOJI_PATTERN"] = (body / (range_hl + 1e-10)).values

        # 4. INTRA_DAY_RANGE - 日内波动率
        factors["INTRA_DAY_RANGE"] = ((s_high - s_low) / s_close).values

        # 5. BULLISH_ENGULFING - 看涨吞没形态
        prev_open = s_open.shift(1)
        prev_body = (s_close.shift(1) - prev_open).abs()
        curr_body = (s_close - s_open).abs()
        is_bullish = (s_close > s_open) & (s_close.shift(1) < prev_open)
        is_engulfing = (
            (curr_body > prev_body)
            & (s_close > prev_open)
            & (s_open < s_close.shift(1))
        )
        factors["BULLISH_ENGULFING"] = (is_bullish & is_engulfing).astype(float).values

        # 6. HAMMER_PATTERN - 锤子线反转信号
        lower_shadow = s_close - s_low
        upper_shadow = s_high - s_close
        is_hammer = (lower_shadow > 2 * body) & (upper_shadow < body)
        factors["HAMMER_PATTERN"] = is_hammer.astype(float).values

        # 7. PRICE_IMPACT - 价格冲击（市场微观结构）
        price_change = s_close.pct_change().abs()
        vol_change = s_vol.pct_change().abs()
        factors["PRICE_IMPACT"] = (price_change / (vol_change + 1e-10)).values

        # 8. VOLUME_PRICE_TREND - 量价趋势一致性
        price_dir = (s_close > s_close.shift(1)).astype(float)
        vol_dir = (s_vol > s_vol.shift(1)).astype(float)
        vpt = (price_dir == vol_dir).astype(float).rolling(20, min_periods=1).mean()
        factors["VOLUME_PRICE_TREND"] = vpt.values

        # 9. VOL_MA_RATIO_5 - 短期成交量动态
        vol_ma5 = s_vol.rolling(5, min_periods=1).mean()
        factors["VOL_MA_RATIO_5"] = (s_vol / (vol_ma5 + 1e-10)).values

        # 10. VOL_VOLATILITY_20 - 成交量稳定性
        vol_std = s_vol.rolling(20, min_periods=1).std()
        vol_mean = s_vol.rolling(20, min_periods=1).mean()
        factors["VOL_VOLATILITY_20"] = (vol_std / (vol_mean + 1e-10)).values

        # 11. TRUE_RANGE - 波动率结构
        factors["TRUE_RANGE"] = (tr / s_close).values

        # 12. BUY_PRESSURE - 日内价格位置
        factors["BUY_PRESSURE"] = ((s_close - s_low) / (s_high - s_low + 1e-10)).values

        # ========== 新增5个资金流因子 ==========

        # 需要amount数据，从原始数据中提取
        if "amount" in symbol_data.columns:
            s_amount = pd.Series(symbol_data["amount"].values, index=symbol_data.index)
        else:
            # 如果没有amount数据，用volume * close估算
            s_amount = s_vol * s_close

        # 13. VWAP_DEVIATION - VWAP偏离度 (资金流强弱)
        vwap = s_amount / (s_vol + 1e-10)  # 成交额/成交量 = 均价
        factors["VWAP_DEVIATION"] = ((s_close - vwap) / (vwap + 1e-10)).values

        # 14. AMOUNT_SURGE_5D - 成交额突增 (资金流入信号)
        amount_ma5 = s_amount.rolling(5, min_periods=1).mean()
        amount_ma20 = s_amount.rolling(20, min_periods=1).mean()
        factors["AMOUNT_SURGE_5D"] = (amount_ma5 / (amount_ma20 + 1e-10) - 1).values

        # 15. PRICE_VOLUME_DIV - 量价背离 (资金流向信号)
        price_change = s_close.pct_change()
        vol_change = s_vol.pct_change()
        pv_divergence = (
            np.sign(price_change) * vol_change.rolling(5, min_periods=1).mean()
        )
        factors["PRICE_VOLUME_DIV"] = pv_divergence.values

        # 16. INTRADAY_POSITION - 日内价格位置 (Williams %R类型，非资金流)
        # 修复：重命名为准确名称，避免误导
        price_pos = (s_close - s_low) / (s_high - s_low + 1e-10)
        factors["INTRADAY_POSITION"] = price_pos.rolling(5, min_periods=1).mean().values

        # 17. LARGE_ORDER_SIGNAL - 大单流入 (机构资金活动)
        # 用均价变化 + 成交量异常代理大单活动
        avg_price = s_amount / (s_vol + 1e-10)
        avg_price_change = avg_price.pct_change()
        vol_ratio = s_vol / s_vol.rolling(20, min_periods=1).mean()
        large_order = ((avg_price_change > 0) & (vol_ratio > 1.2)).astype(float)
        factors["LARGE_ORDER_SIGNAL"] = large_order.values

        return factors

    except Exception as e:
        logger.error(f"因子计算失败 {symbol}: {e}")
        return pd.DataFrame()


def calculate_factors_parallel(
    price_df: pd.DataFrame, config: Dict, max_workers: int = 4
) -> pd.DataFrame:
    """并行计算因子"""
    symbols = sorted(price_df["symbol"].unique())
    logger.info(f"并行计算因子: {len(symbols)} 个标的, {max_workers} 个进程")

    # 准备任务
    tasks = []
    for symbol in symbols:
        symbol_data = price_df[price_df["symbol"] == symbol].sort_values("date")
        tasks.append((symbol, symbol_data, config["factor_generation"]))

    # 并行执行
    factors_list = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(calculate_factors_single, task): task[0] for task in tasks
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="计算因子"):
            symbol = futures[future]
            try:
                result = future.result()
                if not result.empty:
                    factors_list.append(result)
            except Exception as e:
                logger.error(f"任务失败 {symbol}: {e}")

    if not factors_list:
        raise ValueError("无有效因子数据")

    panel = pd.concat(factors_list, ignore_index=True)
    panel = panel.set_index(["symbol", "date"]).sort_index()

    return panel


def save_results(panel: pd.DataFrame, output_dir: Path) -> Tuple[str, str]:
    """保存结果 - 使用时间戳子目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 创建时间戳子目录
    timestamp_dir = output_dir / f"panel_{timestamp}"
    timestamp_dir.mkdir(parents=True, exist_ok=True)

    # 保存面板
    panel_file = timestamp_dir / "panel.parquet"
    panel.to_parquet(panel_file)
    logger.info(f"面板已保存: {panel_file}")

    # 保存元数据
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
    log_file = timestamp_dir / "execution_log.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"ETF横截面因子面板生成执行日志\n")
        f.write(f"执行时间: {timestamp}\n")
        f.write(f"标的数: {meta['etf_count']}\n")
        f.write(f"因子数: {meta['factor_count']}\n")
        f.write(f"数据点: {meta['data_points']}\n")
        f.write(f"覆盖率: {meta['coverage_rate']:.2%}\n")
        f.write(
            f"时间范围: {meta['date_range']['start']} 至 {meta['date_range']['end']}\n"
        )
        f.write(f"\n因子列表:\n")
        for i, factor in enumerate(meta["factors"], 1):
            f.write(f"  {i:2d}. {factor}\n")

    logger.info(f"执行日志已保存: {log_file}")

    return str(panel_file), str(meta_file)


def generate_etf_panel(
    data_dir: str, output_dir: str, config_path: str = None, max_workers: int = 4
) -> Tuple[str, str]:
    """生成ETF因子面板（主函数）"""
    logger.info("=" * 80)
    logger.info("ETF轮动因子面板生成")
    logger.info("=" * 80)

    try:
        # 验证路径
        data_dir_path = validate_path(data_dir, must_exist=True)
        output_dir_path = validate_path(output_dir)

        # 加载配置
        config = (
            load_config(config_path)
            if config_path
            else load_config("config/etf_config.yaml")
        )

        # 加载价格数据
        price_df = load_price_data(data_dir_path)

        # 并行计算因子
        panel = calculate_factors_parallel(price_df, config, max_workers)

        # 保存结果
        return save_results(panel, output_dir_path)

    except Exception as e:
        logger.error(f"面板生成失败: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ETF因子面板生成")
    parser.add_argument("--data-dir", default="raw/ETF/daily", help="数据目录")
    parser.add_argument(
        "--output-dir",
        default="etf_rotation_system/data/results/panels",
        help="输出目录",
    )
    parser.add_argument("--config", help="配置文件路径")
    parser.add_argument("--workers", type=int, default=4, help="并行进程数")

    args = parser.parse_args()

    try:
        panel_file, meta_file = generate_etf_panel(
            args.data_dir, args.output_dir, args.config, args.workers
        )
        logger.info("✅ 完成")
    except Exception as e:
        logger.error(f"❌ 失败: {e}")
        exit(1)
