#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速启动命令行接口
作者：量化首席工程师
版本：1.0.0
日期：2025-09-30

功能：
1. 简化的命令行接口
2. 快速配置和启动
3. 预设模板选择
4. 实时进度显示
"""

import argparse
import sys
from pathlib import Path
import logging
from typing import List, Optional
import json

from config_manager import ConfigManager, ScreeningConfig, BatchConfig
from batch_screener import BatchScreener
from professional_factor_screener import ProfessionalFactorScreener

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FactorScreeningCLI:
    """因子筛选命令行接口"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.batch_screener = BatchScreener(self.config_manager)
    
    def create_parser(self) -> argparse.ArgumentParser:
        """创建命令行解析器"""
        parser = argparse.ArgumentParser(
            description="专业级因子筛选系统 - 快速启动工具",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
使用示例:

1. 单股票快速筛选:
   python cli.py single 0700.HK 60min

2. 批量筛选多时间框架:
   python cli.py batch 0700.HK,0005.HK 30min,60min

3. 使用配置文件:
   python cli.py config batch_config.yaml

4. 列出预设配置:
   python cli.py presets

5. 创建配置模板:
   python cli.py templates
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='可用命令')
        
        # 单个筛选命令
        single_parser = subparsers.add_parser('single', help='单个股票筛选')
        single_parser.add_argument('symbol', help='股票代码 (例: 0700.HK)')
        single_parser.add_argument('timeframe', help='时间框架 (例: 60min)')
        single_parser.add_argument('--preset', default='default', 
                                 help='预设配置 (default/quick/deep/high_freq)')
        single_parser.add_argument('--output', default='./output', 
                                 help='输出目录')
        single_parser.add_argument('--data-root', default='./output',
                                 help='因子数据根目录')
        single_parser.add_argument('--raw-data-root', default='../raw',
                                 help='原始数据根目录')
        
        # 批量筛选命令
        batch_parser = subparsers.add_parser('batch', help='批量筛选')
        batch_parser.add_argument('symbols', help='股票代码列表 (逗号分隔)')
        batch_parser.add_argument('timeframes', help='时间框架列表 (逗号分隔)')
        batch_parser.add_argument('--preset', default='default',
                                help='预设配置')
        batch_parser.add_argument('--task-name', default='batch_screening',
                                help='任务名称')
        batch_parser.add_argument('--output', default='./output',
                                help='输出目录')
        batch_parser.add_argument('--data-root', default='./output',
                                help='因子数据根目录')
        batch_parser.add_argument('--raw-data-root', default='../raw',
                                help='原始数据根目录')
        batch_parser.add_argument('--max-workers', type=int, default=2,
                                help='最大并发任务数')
        batch_parser.add_argument('--continue-on-error', action='store_true',
                                help='遇到错误时继续执行')
        
        # 配置文件命令
        config_parser = subparsers.add_parser('config', help='使用配置文件')
        config_parser.add_argument('config_file', help='配置文件路径')
        config_parser.add_argument('--type', choices=['screening', 'batch'], 
                                 default='batch', help='配置类型')
        
        # 预设配置列表
        presets_parser = subparsers.add_parser('presets', help='列出预设配置')
        
        # 创建模板
        templates_parser = subparsers.add_parser('templates', help='创建配置模板')
        templates_parser.add_argument('--output-dir', default='./configs',
                                    help='模板输出目录')
        
        # 状态查询
        status_parser = subparsers.add_parser('status', help='查询系统状态')
        
        return parser
    
    def run_single_screening(self, args) -> None:
        """运行单个筛选"""
        logger.info(f"开始单个筛选: {args.symbol} {args.timeframe}")
        
        try:
            # 获取预设配置
            config = self.config_manager.get_preset(args.preset)
            
            # 更新配置
            config.symbols = [args.symbol]
            config.timeframes = [args.timeframe]
            config.data_root = args.data_root
            config.raw_data_root = args.raw_data_root
            config.output_dir = args.output
            # 确保因子数据根目录也更新
            config.factor_data_root = args.data_root
            
            # 验证配置
            errors = self.config_manager.validate_config(config)
            if errors:
                logger.error(f"配置验证失败: {errors}")
                return
            
            # 创建筛选器
            screener = ProfessionalFactorScreener(
                data_root=config.data_root,
                config=config
            )
            
            # 执行筛选
            results = screener.screen_factors_comprehensive(
                symbol=args.symbol,
                timeframe=args.timeframe
            )
            
            # 获取顶级因子
            top_factors = screener.get_top_factors(
                results, top_n=10, min_score=0.0, require_significant=False
            )
            
            # 显示结果
            logger.info(f"筛选完成!")
            logger.info(f"总因子数: {len(results)}")
            logger.info(f"顶级因子 (前5个):")
            for i, factor in enumerate(top_factors[:5], 1):
                logger.info(f"  {i}. {factor.name}: {factor.comprehensive_score:.3f}")
            
            # 生成报告
            if config.save_reports:
                report_df = screener.generate_screening_report(results, symbol=args.symbol, timeframe=args.timeframe)
                logger.info(f"报告已生成，包含 {len(report_df)} 个因子")
            
        except Exception as e:
            logger.error(f"单个筛选失败: {str(e)}")
            raise
    
    def run_batch_screening(self, args) -> None:
        """运行批量筛选"""
        # 解析参数
        symbols = [s.strip() for s in args.symbols.split(',')]
        timeframes = [t.strip() for t in args.timeframes.split(',')]
        
        logger.info(f"开始批量筛选:")
        logger.info(f"股票: {symbols}")
        logger.info(f"时间框架: {timeframes}")
        
        try:
            # 创建批量配置
            batch_config = self.config_manager.create_batch_config(
                task_name=args.task_name,
                symbols=symbols,
                timeframes=timeframes,
                preset=args.preset
            )
            
            # 更新配置
            batch_config.max_concurrent_tasks = args.max_workers
            batch_config.continue_on_error = args.continue_on_error

            # 更新数据目录和输出目录
            for config in batch_config.screening_configs:
                config.data_root = args.data_root
                config.raw_data_root = args.raw_data_root
                config.output_dir = args.output
                config.factor_data_root = args.data_root
            
            # 显示配置摘要
            logger.info("批量配置摘要:")
            print(self.config_manager.get_config_summary(batch_config))
            
            # 运行批量筛选
            batch_result = self.batch_screener.run_batch(batch_config)
            
            # 保存结果
            saved_files = self.batch_screener.save_results(batch_result, args.output)
            
            # 显示结果摘要
            logger.info("批量筛选完成!")
            logger.info(f"总任务数: {batch_result.total_tasks}")
            logger.info(f"成功任务: {batch_result.completed_tasks}")
            logger.info(f"失败任务: {batch_result.failed_tasks}")
            logger.info(f"成功率: {batch_result.completed_tasks/batch_result.total_tasks*100:.1f}%")
            
            if batch_result.summary_stats:
                stats = batch_result.summary_stats
                logger.info(f"总因子数: {stats.get('total_factors', 0)}")
                logger.info(f"显著因子数: {stats.get('total_significant_factors', 0)}")
                
                if stats.get('most_common_top_factors'):
                    logger.info("最常见的顶级因子:")
                    for factor, count in stats['most_common_top_factors'][:5]:
                        logger.info(f"  {factor}: {count}次")
            
            logger.info(f"详细结果已保存到: {saved_files.get('summary', 'N/A')}")
            
        except Exception as e:
            logger.error(f"批量筛选失败: {str(e)}")
            raise
    
    def run_config_file(self, args) -> None:
        """使用配置文件运行"""
        config_file = Path(args.config_file)
        
        if not config_file.exists():
            logger.error(f"配置文件不存在: {config_file}")
            return
        
        logger.info(f"加载配置文件: {config_file}")
        
        try:
            if args.type == 'screening':
                # 单个筛选配置
                config = self.config_manager.load_config(config_file, 'screening')
                
                # 验证配置
                errors = self.config_manager.validate_config(config)
                if errors:
                    logger.error(f"配置验证失败: {errors}")
                    return
                
                # 显示配置摘要
                print(self.config_manager.get_config_summary(config))
                
                # 执行筛选
                screener = ProfessionalFactorScreener(
                    data_root=config.data_root,
                    config=config
                )
                
                for symbol in config.symbols:
                    for timeframe in config.timeframes:
                        logger.info(f"筛选 {symbol} {timeframe}...")
                        results = screener.screen_factors_comprehensive(symbol, timeframe)
                        logger.info(f"完成，因子数: {len(results)}")
                        
            elif args.type == 'batch':
                # 批量筛选配置
                batch_config = self.config_manager.load_config(config_file, 'batch')
                
                # 显示配置摘要
                print(self.config_manager.get_config_summary(batch_config))
                
                # 运行批量筛选
                batch_result = self.batch_screener.run_batch(batch_config)
                
                # 保存结果
                saved_files = self.batch_screener.save_results(batch_result)
                logger.info(f"结果已保存: {saved_files.get('summary', 'N/A')}")
                
        except Exception as e:
            logger.error(f"配置文件执行失败: {str(e)}")
            raise
    
    def list_presets(self) -> None:
        """列出预设配置"""
        presets = self.config_manager.list_presets()
        
        print("\n可用的预设配置:")
        print("=" * 50)
        for name, description in presets.items():
            print(f"{name:15} - {description}")
        
        print("\n使用方法:")
        print("  python cli.py single 0700.HK 60min --preset quick")
        print("  python cli.py batch 0700.HK 60min --preset deep")
    
    def create_templates(self, args) -> None:
        """创建配置模板"""
        output_dir = args.output_dir
        logger.info(f"创建配置模板到: {output_dir}")
        
        # 设置配置管理器的输出目录
        self.config_manager.config_dir = Path(output_dir)
        
        # 创建模板
        self.config_manager.create_config_templates()
        
        logger.info("配置模板创建完成!")
        logger.info(f"模板位置: {Path(output_dir) / 'templates'}")
        
        print("\n创建的模板文件:")
        templates_dir = Path(output_dir) / "templates"
        if templates_dir.exists():
            for template_file in templates_dir.glob("*.yaml"):
                print(f"  {template_file.name}")
    
    def show_status(self) -> None:
        """显示系统状态"""
        import psutil
        import platform
        
        print("\n系统状态:")
        print("=" * 40)
        print(f"操作系统: {platform.system()} {platform.release()}")
        print(f"Python版本: {platform.python_version()}")
        print(f"CPU核心数: {psutil.cpu_count()}")
        print(f"内存总量: {psutil.virtual_memory().total / 1024**3:.1f} GB")
        print(f"内存可用: {psutil.virtual_memory().available / 1024**3:.1f} GB")
        print(f"磁盘可用: {psutil.disk_usage('.').free / 1024**3:.1f} GB")
        
        print("\n配置管理器状态:")
        print(f"配置目录: {self.config_manager.config_dir}")
        print(f"可用预设: {len(self.config_manager.presets)}")
        
        # 检查数据目录
        data_dirs = ["./output", "../raw", "./configs"]
        print("\n数据目录状态:")
        for data_dir in data_dirs:
            path = Path(data_dir)
            if path.exists():
                print(f"  {data_dir}: ✓ 存在")
            else:
                print(f"  {data_dir}: ✗ 不存在")
    
    def run(self, args: Optional[List[str]] = None) -> None:
        """运行CLI"""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        if not parsed_args.command:
            parser.print_help()
            return
        
        try:
            if parsed_args.command == 'single':
                self.run_single_screening(parsed_args)
            elif parsed_args.command == 'batch':
                self.run_batch_screening(parsed_args)
            elif parsed_args.command == 'config':
                self.run_config_file(parsed_args)
            elif parsed_args.command == 'presets':
                self.list_presets()
            elif parsed_args.command == 'templates':
                self.create_templates(parsed_args)
            elif parsed_args.command == 'status':
                self.show_status()
            else:
                parser.print_help()
                
        except KeyboardInterrupt:
            logger.info("用户中断操作")
        except Exception as e:
            logger.error(f"执行失败: {str(e)}")
            if logger.level <= logging.DEBUG:
                import traceback
                traceback.print_exc()
            sys.exit(1)


def main():
    """主函数"""
    cli = FactorScreeningCLI()
    cli.run()


if __name__ == "__main__":
    main()
