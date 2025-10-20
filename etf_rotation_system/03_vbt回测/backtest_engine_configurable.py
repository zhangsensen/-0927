#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ETF轮动回测引擎 - 配置化版本

支持外部配置文件，无需修改代码即可调整所有参数
"""

import json
import glob
import itertools
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import vectorbt as vbt
    HAS_VBT = True
except ImportError:
    HAS_VBT = False
    print("警告: 未安装 vectorbt，请运行: pip install vectorbt")

from config_loader import ConfigLoader, load_config_from_args


def load_factor_panel(panel_path: str) -> pd.DataFrame:
    """加载因子面板"""
    panel = pd.read_parquet(panel_path)
    if not isinstance(panel.index, pd.MultiIndex):
        raise ValueError("面板必须是 (symbol, date) MultiIndex")
    return panel


def load_price_data(price_dir: str) -> pd.DataFrame:
    """加载价格数据"""
    prices = []
    for f in sorted(glob.glob(f'{price_dir}/*.parquet')):
        df = pd.read_parquet(f)
        symbol = f.split('/')[-1].split('_')[0]
        df['symbol'] = symbol
        df['date'] = pd.to_datetime(df['trade_date'])
        prices.append(df[['date', 'close', 'symbol']])

    price_df = pd.concat(prices, ignore_index=True)
    pivot = price_df.pivot(index='date', columns='symbol', values='close')
    return pivot.sort_index().ffill()


def load_top_factors(screening_csv: str, top_k: int = 10, factor_list: List[str] = None) -> List[str]:
    """从筛选结果加载Top K因子或使用指定的因子列表"""
    if factor_list and len(factor_list) > 0:
        # 使用指定的因子列表
        return factor_list

    # 从筛选结果加载
    df = pd.read_csv(screening_csv)
    col_name = 'factor' if 'factor' in df.columns else 'panel_factor'
    available_factors = df.head(top_k)[col_name].tolist()

    if len(available_factors) < top_k:
        print(f"警告: 筛选结果只有 {len(available_factors)} 个因子，少于请求的 {top_k} 个")

    return available_factors


def calculate_composite_score(
    panel: pd.DataFrame,
    factors: List[str],
    weights: Dict[str, float],
    method: str = 'zscore'
) -> pd.DataFrame:
    """计算复合因子得分 - 完全向量化实现"""
    # 重塑为 (date, symbol) 结构
    factor_data = panel[factors].unstack(level='symbol')

    # 向量化标准化
    if method == 'zscore':
        normalized = (factor_data - factor_data.mean(axis=1, skipna=True).values[:, np.newaxis]) / (factor_data.std(axis=1, skipna=True).values[:, np.newaxis] + 1e-8)
    else:  # rank
        normalized = factor_data.rank(axis=1, pct=True) * 2 - 1  # [-1, 1]

    # 获取维度信息
    n_dates, n_total = normalized.shape
    n_factors = len(factors)
    n_symbols = n_total // n_factors

    # 重塑为 (dates, symbols, factors) 用于矩阵乘法
    reshaped = normalized.values.reshape(n_dates, n_symbols, n_factors)

    # 向量化加权求和 - 无循环
    weight_array = np.array([weights.get(f, 0) for f in factors])
    scores_array = np.sum(reshaped * weight_array[np.newaxis, np.newaxis, :], axis=2)

    # 创建结果DataFrame
    symbols = [col[1] for col in normalized.columns[::n_factors]]  # 提取symbol名称
    scores = pd.DataFrame(scores_array, index=normalized.index, columns=symbols)

    return scores


def build_target_weights(scores: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """构建Top-N目标权重"""
    ranks = scores.rank(axis=1, ascending=False, method='first')
    selection = ranks <= top_n
    weights = selection.astype(float)
    weights = weights.div(weights.sum(axis=1), axis=0).fillna(0.0)
    return weights


def backtest_topn_rotation(
    prices: pd.DataFrame,
    scores: pd.DataFrame,
    top_n: int = 5,
    rebalance_freq: int = 20,
    fees: float = 0.001,
    init_cash: float = 1_000_000
) -> Dict:
    """Top-N轮动回测 - 向量化实现"""
    # 对齐日期
    common_dates = prices.index.intersection(scores.index)
    prices = prices.loc[common_dates]
    scores = scores.loc[common_dates]

    # 构建目标权重
    weights = build_target_weights(scores, top_n)

    # 向量化调仓日权重更新 - 无循环
    rebalance_mask = pd.Series(np.arange(len(weights)) % rebalance_freq == 0, index=weights.index)
    rebalance_mask.iloc[0] = True  # 第一天调仓

    # 使用 ffill 向前填充权重
    weights_ffill = weights.where(rebalance_mask, np.nan).ffill().fillna(0.0)

    # 计算收益 - 确保列对齐
    asset_returns = prices.pct_change().fillna(0.0)
    prev_weights = weights_ffill.shift().fillna(0.0)

    # 对齐列名
    common_symbols = asset_returns.columns.intersection(prev_weights.columns)
    asset_returns_aligned = asset_returns[common_symbols]
    prev_weights_aligned = prev_weights[common_symbols]

    gross_returns = (prev_weights_aligned * asset_returns_aligned).sum(axis=1)

    # 交易成本
    weight_diff = weights_ffill.diff().abs().sum(axis=1).fillna(0.0)
    turnover = 0.5 * weight_diff
    net_returns = gross_returns - fees * turnover

    # 净值曲线
    equity = (1 + net_returns).cumprod() * init_cash

    # 统计指标
    total_return = (equity.iloc[-1] / init_cash - 1) * 100
    periods_per_year = 252
    sharpe = net_returns.mean() / net_returns.std() * np.sqrt(periods_per_year) if net_returns.std() > 0 else 0

    running_max = equity.cummax()
    drawdown = (equity / running_max - 1) * 100
    max_dd = drawdown.min()

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'final_value': equity.iloc[-1],
        'turnover': turnover.sum()
    }


def grid_search_weights(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    factors: List[str],
    top_n_list: List[int] = [3, 5, 8],
    weight_grid: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    max_combos: int = 10000,
    rebalance_freq: int = 20,
    weight_sum_range: List[float] = [0.7, 1.3],
    enable_cache: bool = True,
    primary_metric: str = 'sharpe_ratio'
) -> pd.DataFrame:
    """网格搜索因子权重组合 - 向量化优化"""
    # 获取日志器
    logger = logging.getLogger('backtest_engine')
    # 生成权重组合
    weight_combos = list(itertools.product(weight_grid, repeat=len(factors)))
    logger.info(f"生成权重组合: {len(weight_combos):,} 个理论组合")

    # 向量化优化：使用numpy进行高效计算
    weight_array = np.array(weight_combos)
    weight_sums = np.sum(weight_array, axis=1)  # 向量化求和

    # 向量化过滤有效组合
    valid_mask = (weight_sums >= weight_sum_range[0]) & (weight_sums <= weight_sum_range[1])

    # 修复：正确的逻辑 - 先过滤，再限制组合数
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) > max_combos:
        # 有效组合太多，先取前N个
        valid_indices = valid_indices[:max_combos]
    valid_combos = [weight_combos[i] for i in valid_indices]

    # 记录搜索配置
    logger.info(f"权重搜索配置: {len(valid_combos)} 组合 × {len(top_n_list)} Top-N")
    logger.info(f"权重网格: {weight_grid}")
    logger.info(f"权重和范围: {weight_sum_range}")
    print(f"开始网格搜索: {len(valid_combos)} 组合 × {len(top_n_list)} Top-N")

    # 预计算所有得分矩阵以避免重复计算
    score_cache = {}

    results = []
    logger.info("开始权重组合测试...")
    tested_count = 0

    for weights in tqdm(valid_combos, desc="权重组合", disable=not True):
        weight_dict = dict(zip(factors, weights))
        weights_key = tuple(weights)

        # 缓存得分矩阵
        if enable_cache and weights_key not in score_cache:
            score_cache[weights_key] = calculate_composite_score(panel, factors, weight_dict)

        scores = score_cache[weights_key] if enable_cache else calculate_composite_score(panel, factors, weight_dict)

        # 批量测试所有 top_n 值
        for top_n in top_n_list:
            try:
                result = backtest_topn_rotation(
                    prices=prices,
                    scores=scores,
                    top_n=top_n,
                    rebalance_freq=rebalance_freq
                )

                results.append({
                    'weights': str(weight_dict),
                    'top_n': top_n,
                    'total_return': result['total_return'],
                    'sharpe_ratio': result['sharpe_ratio'],
                    'max_drawdown': result['max_drawdown'],
                    'final_value': result['final_value'],
                    'turnover': result['turnover']
                })
                tested_count += 1

                # 每100个组合记录一次进度
                if tested_count % 100 == 0:
                    logger.info(f"已测试 {tested_count} 个策略组合...")

            except Exception as e:
                logger.error(f"组合失败: {weight_dict}, top_n={top_n}, 错误: {e}")
                print(f"组合失败: {weight_dict}, top_n={top_n}, 错误: {e}")

    # 向量化结果处理
    logger.info(f"开始处理搜索结果，共收集到 {len(results)} 个策略结果")
    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df.sort_values(primary_metric, ascending=False)
        logger.info(f"结果已按 {primary_metric} 排序")

        # 记录最优策略信息
        best_strategy = df.iloc[0]
        logger.info(f"最优策略: {best_strategy['weights']}, top_n={best_strategy['top_n']}, sharpe={best_strategy['sharpe_ratio']:.3f}")
    else:
        logger.warning("未找到任何有效的策略组合")

    # 记录搜索完成
    logger.info(f"权重网格搜索完成: 共测试 {len(df)} 个策略组合")

    return df


def run_backtest_with_config(config) -> tuple[pd.DataFrame, dict]:
    """使用配置对象运行回测"""
    # 获取时间戳（从配置中的output_dir路径提取）
    output_path = Path(config.output_dir)
    timestamp = output_path.name.replace('backtest_', '')

    # 获取日志器
    logger = logging.getLogger('backtest_engine')

    print("ETF轮动回测引擎 - 配置化版本")
    print("="*80)
    print(f"时间戳: {timestamp}")
    print(f"面板: {config.panel_file}")
    print(f"筛选: {config.screening_file}")
    print(f"预设: {getattr(config, '_preset_name', 'default')}")

    # 记录到日志
    logger.info(f"开始回测 - 时间戳: {timestamp}")
    logger.info(f"面板文件: {config.panel_file}")
    logger.info(f"筛选文件: {config.screening_file}")

    # 加载数据
    print("\\n加载数据...")
    logger.info("开始加载数据...")
    panel = load_factor_panel(config.panel_file)
    prices = load_price_data(config.price_dir)
    factors = load_top_factors(config.screening_file, config.top_k, config.factors if config.factors else None)

    print(f"  因子数: {len(factors)}")
    print(f"  因子: {factors[:5]}...")
    print(f"  ETF数: {len(prices.columns)}")
    print(f"  日期: {prices.index.min().date()} ~ {prices.index.max().date()}")
    print(f"  权重网格: {config.weight_grid_points}")
    print(f"  最大组合: {config.max_combinations}")

    # 记录数据统计信息到日志
    logger.info(f"数据加载完成 - 因子数: {len(factors)}, ETF数: {len(prices.columns)}")
    logger.info(f"时间范围: {prices.index.min().date()} ~ {prices.index.max().date()}")
    logger.info(f"权重网格: {config.weight_grid_points}, 最大组合: {config.max_combinations}")

    # 网格搜索
    print("\\n开始回测...")
    logger.info("开始权重网格搜索...")
    results = grid_search_weights(
        panel=panel,
        prices=prices,
        factors=factors,
        top_n_list=config.top_n_list,
        weight_grid=config.weight_grid_points,
        max_combos=config.max_combinations,
        rebalance_freq=config.rebalance_freq,
        weight_sum_range=config.weight_sum_range,
        enable_cache=config.enable_score_cache,
        primary_metric=config.primary_metric
    )

    # 保存结果
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 简化文件名，因为已经有时间戳文件夹
    csv_file = output_path / 'results.csv'
    results.to_csv(csv_file, index=False)
    logger.info(f"回测结果已保存: {csv_file}")
    logger.info(f"输出文件夹: {output_path}")
    print(f"\\n结果: {csv_file}")
    print(f"文件夹: {output_path}")

    # 输出Top N
    top_n = min(config.save_top_results, len(results))
    print(f"\\nTop {top_n} 策略:")
    print(results.head(top_n).to_string(index=False))

    # 保存最优策略配置
    if len(results) > 0 and config.save_best_config:
        best = results.iloc[0]
        best_config = {
            'timestamp': timestamp,
            'preset_name': getattr(config, '_preset_name', 'default'),
            'weights': best['weights'],
            'top_n': int(best['top_n']),
            'rebalance_freq': config.rebalance_freq,
            'performance': {
                'total_return': float(best['total_return']),
                'sharpe_ratio': float(best['sharpe_ratio']),
                'max_drawdown': float(best['max_drawdown']),
                'final_value': float(best['final_value']),
                'turnover': float(best['turnover'])
            },
            'factors': factors,
            'config_used': {
                'weight_grid_points': config.weight_grid_points,
                'max_combinations': config.max_combinations,
                'standardization_method': config.standardization_method,
                'fees': config.fees,
                'init_cash': config.init_cash
            },
            'data_source': {
                'panel': config.panel_file,
                'screening': config.screening_file,
                'price_dir': config.price_dir
            }
        }

        config_file = output_path / 'best_config.json'
        with open(config_file, 'w') as f:
            json.dump(best_config, f, indent=2, ensure_ascii=False)

        logger.info(f"最优策略配置已保存: {config_file}")
        logger.info("=== 回测完成 ===")
        print(f"最优配置: {config_file}")

        return results, best_config

    return results, {}


def setup_logging(config):
    """设置日志配置"""
    logger = logging.getLogger('backtest_engine')
    logger.setLevel(logging.INFO)

    # 清除现有的处理器
    logger.handlers.clear()

    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 检查是否需要文件日志 - 从根级别属性读取
    log_to_file = getattr(config, 'log_to_file', False)

    if log_to_file:
        log_dir = Path(getattr(config, 'log_dir', '/tmp'))
        log_prefix = getattr(config, 'log_prefix', 'backtest_log')
        console_output = getattr(config, 'console_output', True)

        try:
            log_dir.mkdir(parents=True, exist_ok=True)

            # 简化日志文件名，因为文件夹已经有时间戳
            log_file = log_dir / "backtest.log"

            # 文件处理器
            file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            logger.info(f"日志文件: {log_file}")

        except Exception as e:
            print(f"❌ 创建日志文件失败: {e}")
            logger.error(f"创建日志文件失败: {e}")

    return logger


def main():
    """主函数 - 支持命令行参数和配置文件"""
    parser = argparse.ArgumentParser(description='ETF轮动回测引擎 - 配置化版本')

    # 基本参数
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--preset', type=str, help='使用预设配置')
    parser.add_argument('--output-dir', type=str, help='指定输出文件夹（用于重用现有文件夹）')

    # 数据路径参数（覆盖配置文件）
    parser.add_argument('--panel', type=str, help='因子面板文件路径')
    parser.add_argument('--price-dir', type=str, help='价格数据目录')
    parser.add_argument('--screening', type=str, help='因子筛选结果文件')

    # 回测参数（覆盖配置文件）
    parser.add_argument('--max-combos', type=int, help='最大组合数')
    parser.add_argument('--top-k', type=int, help='使用的前K个因子')

    # 工具参数
    parser.add_argument('--list-presets', action='store_true', help='列出所有可用预设')
    parser.add_argument('--show-config', action='store_true', help='显示当前配置')
    parser.add_argument('--verbose', action='store_true', help='详细输出模式')

    args = parser.parse_args()

    # 初始化配置加载器
    if args.config:
        loader = ConfigLoader(args.config)
    else:
        loader = ConfigLoader()  # 使用默认配置文件

    # 列出预设
    if args.list_presets:
        presets = loader.list_presets()
        print("可用预设:")
        for preset in presets:
            print(f"  - {preset}")
        return

    # 加载配置
    try:
        config = load_config_from_args(args)

        # 处理输出文件夹
        if args.output_dir:
            # 用户指定了输出文件夹，直接使用
            timestamp_dir = Path(args.output_dir)
            timestamp_dir.mkdir(parents=True, exist_ok=True)
            print(f"使用指定文件夹: {timestamp_dir}")
        else:
            # 创建新的时间戳文件夹
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_output_path = Path(config.output_dir)
            timestamp_dir = base_output_path / f"backtest_{timestamp}"
            timestamp_dir.mkdir(parents=True, exist_ok=True)
            print(f"创建新文件夹: {timestamp_dir}")

        # 更新配置中的输出路径
        config.output_dir = str(timestamp_dir)
        config.log_dir = str(timestamp_dir)  # 同时更新日志目录

        # 初始化日志系统（在创建时间戳文件夹之后）
        logger = setup_logging(config)
        logger.info("=== ETF轮动回测引擎启动 ===")
        logger.info(f"配置文件: {args.config or '默认配置'}")

        # 保存预设名称
        if args.preset:
            config._preset_name = args.preset
            logger.info(f"使用预设: {args.preset}")

        # 显示配置
        if args.show_config:
            print("当前配置:")
            print(f"  权重网格: {config.weight_grid_points}")
            print(f"  最大组合数: {config.max_combinations}")
            print(f"  Top-N列表: {config.top_n_list}")
            print(f"  标准化方法: {config.standardization_method}")
            print(f"  主要指标: {config.primary_metric}")
            print()

        # 运行回测
        results, best_config = run_backtest_with_config(config)

        if args.verbose:
            print(f"\\n回测完成! 共测试 {len(results)} 个组合")
            if best_config:
                print(f"最优策略夏普比率: {best_config['performance']['sharpe_ratio']:.3f}")
                print(f"最优策略总收益: {best_config['performance']['total_return']:.2f}%")

        logger.info("程序执行完成")

    except Exception as e:
        error_msg = f"程序执行错误: {e}"
        print(error_msg)
        logger.error(error_msg)
        if args.verbose:
            import traceback
            traceback.print_exc()
            logger.error("详细错误信息已输出到控制台")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())