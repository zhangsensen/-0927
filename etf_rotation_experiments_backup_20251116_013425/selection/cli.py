"""
CLI 入口模块

处理命令行参数解析、配置加载、数据 IO 等。
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd

from .core import select_top200, DEFAULT_CONFIG, deep_merge_config

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logging.warning("未安装 pyyaml，配置文件功能不可用（pip install pyyaml）")


def load_config_file(config_path: str) -> dict:
    """
    加载 YAML 配置文件
    
    参数:
        config_path: 配置文件路径
    
    返回:
        配置字典（失败时返回空字典）
    """
    if not YAML_AVAILABLE:
        logging.error("无法加载配置文件：pyyaml 未安装")
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logging.info(f"成功加载配置文件: {config_path}")
        return config or {}
    except FileNotFoundError:
        logging.error(f"配置文件不存在: {config_path}")
        return {}
    except yaml.YAMLError as e:
        logging.error(f"配置文件解析失败: {e}")
        return {}
    except Exception as e:
        logging.error(f"加载配置文件时发生错误: {e}")
        return {}


def parse_args():
    """
    解析命令行参数
    
    返回:
        argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description='从 Top-2000 回测结果中筛选 Top-200 组合',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 最简单（使用默认配置）
  python scripts/run_top200_selection.py \\
      --input data/top2000.csv \\
      --output data/top200_selected.csv

  # 使用自定义配置文件
  python scripts/run_top200_selection.py \\
      --input data/top2000.csv \\
      --output data/top200_selected.csv \\
      --config my_config.yaml

  # 通过 CLI 覆盖关键参数
  python scripts/run_top200_selection.py \\
      --input data/top2000.csv \\
      --output data/top200_selected.csv \\
      --min-sharpe-net 1.0 \\
      --max-turnover 1.4 \\
      --verbose
        """
    )
    
    # 必需参数
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='输入 CSV 文件路径（Top-2000 回测结果）'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='输出 CSV 文件路径（Top-200 筛选结果）'
    )
    
    # 可选参数：配置文件
    parser.add_argument(
        '--config', '-c',
        default=None,
        help='YAML 配置文件路径（可选）'
    )
    
    # 可选参数：质量过滤阈值覆盖
    parser.add_argument(
        '--min-sharpe-net',
        type=float,
        default=None,
        help='覆盖最低 Sharpe 比率（默认: 0.95）'
    )
    parser.add_argument(
        '--max-dd-net',
        type=float,
        default=None,
        help='覆盖最大回撤（默认: -0.28）'
    )
    parser.add_argument(
        '--min-annual-ret-net',
        type=float,
        default=None,
        help='覆盖最低年化收益（默认: 0.12）'
    )
    parser.add_argument(
        '--max-turnover',
        type=float,
        default=None,
        help='覆盖最大换手率（默认: 1.6）'
    )
    
    # 可选参数：高换手控制
    parser.add_argument(
        '--max-high-turnover-ratio',
        type=float,
        default=None,
        help='覆盖高换手最大比例（默认: 0.3）'
    )
    
    # 其他选项
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='显示详细日志'
    )
    
    return parser.parse_args()


def setup_logging(verbose: bool):
    """
    配置日志输出
    
    参数:
        verbose: 是否详细模式
    """
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """
    应用命令行参数覆盖到配置
    
    参数:
        config: 基础配置字典
        args: 命令行参数
    
    返回:
        覆盖后的配置字典
    """
    config = config.copy()
    
    # 质量过滤阈值覆盖
    if args.min_sharpe_net is not None:
        if 'quality_filter' not in config:
            config['quality_filter'] = {}
        if 'standard' not in config['quality_filter']:
            config['quality_filter']['standard'] = {}
        config['quality_filter']['standard']['min_sharpe_net'] = args.min_sharpe_net
        logging.info(f"CLI 覆盖: min_sharpe_net = {args.min_sharpe_net}")
    
    if args.max_dd_net is not None:
        if 'quality_filter' not in config:
            config['quality_filter'] = {}
        if 'standard' not in config['quality_filter']:
            config['quality_filter']['standard'] = {}
        config['quality_filter']['standard']['max_dd_net'] = args.max_dd_net
        logging.info(f"CLI 覆盖: max_dd_net = {args.max_dd_net}")
    
    if args.min_annual_ret_net is not None:
        if 'quality_filter' not in config:
            config['quality_filter'] = {}
        if 'standard' not in config['quality_filter']:
            config['quality_filter']['standard'] = {}
        config['quality_filter']['standard']['min_annual_ret_net'] = args.min_annual_ret_net
        logging.info(f"CLI 覆盖: min_annual_ret_net = {args.min_annual_ret_net}")
    
    if args.max_turnover is not None:
        if 'quality_filter' not in config:
            config['quality_filter'] = {}
        if 'standard' not in config['quality_filter']:
            config['quality_filter']['standard'] = {}
        config['quality_filter']['standard']['max_turnover'] = args.max_turnover
        logging.info(f"CLI 覆盖: max_turnover = {args.max_turnover}")
    
    # 高换手控制覆盖
    if args.max_high_turnover_ratio is not None:
        if 'turnover_control' not in config:
            config['turnover_control'] = {}
        config['turnover_control']['max_ratio'] = args.max_high_turnover_ratio
        logging.info(f"CLI 覆盖: max_high_turnover_ratio = {args.max_high_turnover_ratio}")
    
    return config


def main():
    """CLI 主入口"""
    # 解析参数
    args = parse_args()
    
    # 配置日志
    setup_logging(args.verbose)
    
    logging.info("Top-200 筛选工具启动")
    
    # 第一层：默认配置
    config = DEFAULT_CONFIG.copy()
    
    # 第二层：配置文件（如果提供）
    if args.config:
        file_config = load_config_file(args.config)
        if file_config:
            config = deep_merge_config(config, file_config)
            logging.info("已合并配置文件")
    
    # 第三层：命令行参数覆盖
    config = apply_cli_overrides(config, args)
    
    # 加载数据
    logging.info(f"加载数据: {args.input}")
    try:
        df = pd.read_csv(args.input)
        logging.info(f"成功加载 {len(df)} 行数据")
    except FileNotFoundError:
        logging.error(f"输入文件不存在: {args.input}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"加载数据失败: {e}")
        sys.exit(1)
    
    # 执行筛选
    try:
        result = select_top200(df, config, verbose=args.verbose)
    except Exception as e:
        logging.error(f"筛选过程中发生错误: {e}", exc_info=True)
        sys.exit(1)
    
    # 保存结果
    logging.info(f"保存结果到: {args.output}")
    try:
        # 确保输出目录存在
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        result.to_csv(args.output, index=False)
        logging.info(f"成功保存 {len(result)} 行数据")
    except Exception as e:
        logging.error(f"保存结果失败: {e}")
        sys.exit(1)
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("筛选完成！摘要统计:")
    print("=" * 60)
    
    print(f"\n总数: {len(result)}")
    
    print(f"\ncombo_size 分布:")
    size_dist = result['combo_size'].value_counts().sort_index()
    for size, count in size_dist.items():
        pct = count / len(result) * 100
        print(f"  size={size}: {count} ({pct:.1f}%)")
    
    high_turnover_threshold = config['turnover_control']['threshold']
    high_count = (result['avg_turnover'] > high_turnover_threshold).sum()
    high_pct = high_count / len(result) * 100
    print(f"\n高换手 (>{high_turnover_threshold}): {high_count} ({high_pct:.1f}%)")
    
    print(f"\ndominant_factor 分布:")
    factor_dist = result['dominant_factor'].value_counts()
    for factor, count in factor_dist.items():
        pct = count / len(result) * 100
        print(f"  {factor}: {count} ({pct:.1f}%)")
    
    print(f"\n平均性能指标:")
    print(f"  annual_ret_net: {result['annual_ret_net'].mean():.2%}")
    print(f"  sharpe_net: {result['sharpe_net'].mean():.3f}")
    print(f"  max_dd_net: {result['max_dd_net'].mean():.2%}")
    print(f"  avg_turnover: {result['avg_turnover'].mean():.3f}")
    
    print(f"\n输出文件: {args.output}")
    print("=" * 60)


if __name__ == '__main__':
    main()
