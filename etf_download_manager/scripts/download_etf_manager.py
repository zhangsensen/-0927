#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF下载管理器主脚本
统一的ETF数据下载入口
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from etf_download_manager import (
    ETFDownloadManager,
    ETFConfig,
    ETFDataSource,
    ETFDownloadType,
    ETFListManager,
    ETFPriority,
    ETFExchange
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('etf_download.log')
    ]
)
logger = logging.getLogger(__name__)


def create_default_config() -> ETFConfig:
    """创建默认配置"""
    try:
        from etf_download_manager.config import load_config
        return load_config('etf_config')
    except:
        return ETFConfig(
            source=ETFDataSource.TUSHARE,
            tushare_token="4a24bcfff16f7593632e6c46976a83e6a26f8f565daa156cb9ea9c1f",  # 使用配置的Token
            base_dir="raw/ETF",
            years_back=2,
            max_retries=3,
            retry_delay=1.0,
            request_delay=0.2,
            download_types=[ETFDownloadType.DAILY],
            save_format="parquet",
            verbose=True
        )


def create_quick_download_config() -> ETFConfig:
    """创建快速下载配置（只下载核心ETF）"""
    try:
        from etf_download_manager.config import load_config
        config = load_config('etf_config')
        config.years_back = 1  # 只下载1年数据
        config.max_retries = 2
        config.retry_delay = 0.5
        config.request_delay = 0.1
        return config
    except:
        return ETFConfig(
            source=ETFDataSource.TUSHARE,
            tushare_token="4a24bcfff16f7593632e6c46976a83e6a26f8f565daa156cb9ea9c1f",
            base_dir="raw/ETF",
            years_back=1,
            max_retries=2,
            retry_delay=0.5,
            request_delay=0.1,
            download_types=[ETFDownloadType.DAILY],
            save_format="parquet",
            verbose=True
        )


def create_full_download_config() -> ETFConfig:
    """创建完整下载配置"""
    return ETFConfig(
        source=ETFDataSource.TUSHARE,
        tushare_token="",  # 将从环境变量获取
        base_dir="raw/ETF",
        years_back=3,  # 下载3年数据
        max_retries=5,
        retry_delay=2.0,
        request_delay=0.3,
        download_types=[ETFDownloadType.DAILY, ETFDownloadType.MONEYFLOW],
        save_format="parquet",
        create_summary=True,
        verbose=True,
        batch_size=20
    )


def print_banner():
    """打印横幅"""
    print("=" * 80)
    print("ETF下载管理器 v1.0.0")
    print("统一管理ETF数据下载，消除重复代码")
    print("=" * 80)


def print_etf_categories(list_manager: ETFListManager):
    """打印ETF分类信息"""
    summary = list_manager.get_etf_summary()

    print("\n=== ETF分类概览 ===")
    print(f"总数量: {summary['total_count']}")
    print(f"必配ETF: {summary['must_have_count']} ⭐")
    print(f"高优先级: {summary['high_priority_count']}")
    print(f"下载状态: 已完成{summary['completed_downloads']}, 待下载{summary['pending_downloads']}")

    print("\n=== 分类统计 ===")
    for category, count in sorted(summary['categories'].items()):
        print(f"  {category}: {count}个")


def download_core_etfs(config: ETFConfig, list_manager: ETFListManager) -> bool:
    """下载核心ETF"""
    print("\n=== 下载核心ETF ===")

    # 获取核心ETF
    core_etfs = list_manager.get_must_have_etfs()
    if not core_etfs:
        print("未找到核心ETF")
        return False

    print(f"找到 {len(core_etfs)} 只核心ETF")
    list_manager.print_etf_list(core_etfs[:10])  # 显示前10只

    # 创建下载器
    try:
        downloader = ETFDownloadManager(config)
    except Exception as e:
        print(f"❌ 初始化下载器失败: {e}")
        return False

    # 开始下载
    print(f"\n开始下载核心ETF数据...")
    stats = downloader.download_multiple_etfs(core_etfs)

    # 显示结果
    print(f"\n=== 下载结果 ===")
    print(f"总数量: {stats.total_etfs}")
    print(f"成功: {stats.success_count}")
    print(f"失败: {stats.failed_count}")
    print(f"成功率: {stats.success_rate:.1f}%")
    print(f"耗时: {stats.duration}")

    if stats.failed_etfs:
        print(f"\n失败的ETF: {', '.join(stats.failed_etfs[:5])}")
        if len(stats.failed_etfs) > 5:
            print(f"...还有 {len(stats.failed_etfs) - 5} 只ETF失败")

    return stats.success_count > 0


def download_by_priority(config: ETFConfig, list_manager: ETFListManager, min_priority: str) -> bool:
    """按优先级下载ETF"""
    print(f"\n=== 按优先级下载ETF (>= {min_priority}) ===")

    try:
        priority_enum = ETFPriority(min_priority)
    except ValueError:
        print(f"❌ 无效的优先级: {min_priority}")
        print(f"可用优先级: {[p.value for p in ETFPriority]}")
        return False

    # 筛选ETF
    filtered_etfs = list_manager.filter_etfs(priorities=[p for p in ETFPriority if p.value >= min_priority])

    if not filtered_etfs:
        print(f"未找到优先级 >= {min_priority} 的ETF")
        return False

    print(f"找到 {len(filtered_etfs)} 只ETF")
    list_manager.print_etf_list(filtered_etfs[:15])  # 显示前15只

    # 创建下载器
    try:
        downloader = ETFDownloadManager(config)
    except Exception as e:
        print(f"❌ 初始化下载器失败: {e}")
        return False

    # 开始下载
    print(f"\n开始下载ETF数据...")
    stats = downloader.download_multiple_etfs(filtered_etfs)

    # 显示结果
    print(f"\n=== 下载结果 ===")
    print(f"总数量: {stats.total_etfs}")
    print(f"成功: {stats.success_count}")
    print(f"失败: {stats.failed_count}")
    print(f"成功率: {stats.success_rate:.1f}%")
    print(f"耗时: {stats.duration}")
    print(f"总记录数: {stats.total_daily_records + stats.total_moneyflow_records:,}")

    return stats.success_count > 0


def download_specific_etfs(config: ETFConfig, list_manager: ETFListManager, etf_codes: List[str]) -> bool:
    """下载指定的ETF"""
    print(f"\n=== 下载指定ETF ===")
    print(f"指定ETF: {', '.join(etf_codes)}")

    # 查找ETF
    etfs = []
    not_found = []

    for code in etf_codes:
        etf = list_manager.get_etf_by_code(code)
        if etf:
            etfs.append(etf)
        else:
            not_found.append(code)

    if not_found:
        print(f"❌ 未找到ETF: {', '.join(not_found)}")

    if not etfs:
        print("没有找到可下载的ETF")
        return False

    print(f"找到 {len(etfs)} 只ETF:")
    list_manager.print_etf_list(etfs)

    # 创建下载器
    try:
        downloader = ETFDownloadManager(config)
    except Exception as e:
        print(f"❌ 初始化下载器失败: {e}")
        return False

    # 开始下载
    print(f"\n开始下载ETF数据...")
    stats = downloader.download_multiple_etfs(etfs)

    # 显示结果
    print(f"\n=== 下载结果 ===")
    print(f"总数量: {stats.total_etfs}")
    print(f"成功: {stats.success_count}")
    print(f"失败: {stats.failed_count}")
    print(f"成功率: {stats.success_rate:.1f}%")
    print(f"耗时: {stats.duration}")

    return stats.success_count > 0


def list_etfs(list_manager: ETFListManager, category: Optional[str] = None, max_items: int = 50):
    """列出ETF"""
    print("\n=== ETF列表 ===")

    if category:
        etfs = list_manager.get_etfs_by_category(category)
        print(f"分类: {category}")
    else:
        etfs = list_manager.get_all_etfs()
        print("所有ETF")

    if not etfs:
        print("未找到ETF")
        return

    print(f"共 {len(etfs)} 只ETF:")
    list_manager.print_etf_list(etfs, max_items=max_items)


def update_etf_data(config: ETFConfig, list_manager: ETFListManager, etf_code: str, days_back: int = 30):
    """更新ETF数据"""
    print(f"\n=== 更新ETF数据 ===")
    print(f"ETF代码: {etf_code}")
    print(f"更新最近 {days_back} 天的数据")

    # 查找ETF
    etf = list_manager.get_etf_by_code(etf_code)
    if not etf:
        print(f"❌ 未找到ETF: {etf_code}")
        return False

    print(f"找到ETF: {etf.name} ({etf.ts_code})")

    # 创建下载器
    try:
        downloader = ETFDownloadManager(config)
    except Exception as e:
        print(f"❌ 初始化下载器失败: {e}")
        return False

    # 更新数据
    print(f"\n开始更新数据...")
    result = downloader.update_etf_data(etf, days_back)

    # 显示结果
    print(f"\n=== 更新结果 ===")
    if result.success:
        print(f"✅ 更新成功")
        print(f"日线数据: {result.daily_records} 条记录")
        print(f"资金流向数据: {result.moneyflow_records} 条记录")
        if result.file_paths:
            print(f"保存路径: {result.file_paths}")
    else:
        print(f"❌ 更新失败: {result.error_message}")

    return result.success


def validate_data(config: ETFConfig, list_manager: ETFListManager):
    """验证数据完整性"""
    print("\n=== 验证数据完整性 ===")

    # 获取所有ETF
    all_etfs = list_manager.get_all_etfs()

    if not all_etfs:
        print("未找到ETF")
        return

    # 创建下载器和数据管理器
    try:
        downloader = ETFDownloadManager(config)
        validation_results = downloader.validate_downloaded_data(all_etfs)
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        return

    # 统计结果
    total_count = len(validation_results)
    valid_count = sum(1 for r in validation_results.values() if r['overall_valid'])
    daily_valid_count = sum(1 for r in validation_results.values() if r['daily_valid'])
    moneyflow_valid_count = sum(1 for r in validation_results.values() if r['moneyflow_valid'])

    print(f"总ETF数量: {total_count}")
    print(f"数据完整: {valid_count} ({valid_count/total_count*100:.1f}%)")
    print(f"日线数据完整: {daily_valid_count}")
    print(f"资金流向数据完整: {moneyflow_valid_count}")

    # 显示有问题的ETF
    problematic_etfs = [code for code, result in validation_results.items() if not result['overall_valid']]
    if problematic_etfs:
        print(f"\n有问题的ETF ({len(problematic_etfs)}只):")
        for code in problematic_etfs[:10]:
            result = validation_results[code]
            etf = list_manager.get_etf_by_ts_code(code)
            etf_name = etf.name if etf else "未知"
            issues = []
            if result['daily_exists'] and not result['daily_valid']:
                issues.append("日线数据异常")
            if result['moneyflow_exists'] and not result['moneyflow_valid']:
                issues.append("资金流向数据异常")
            print(f"  ❌ {code} - {etf_name}: {', '.join(issues)}")

        if len(problematic_etfs) > 10:
            print(f"  ...还有 {len(problematic_etfs) - 10} 只ETF有问题")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ETF下载管理器')
    parser.add_argument('--action', '-a', choices=[
        'list', 'download-core', 'download-priority', 'download-specific',
        'update', 'validate', 'summary'
    ], default='summary', help='执行的动作')

    parser.add_argument('--config', '-c', choices=['default', 'quick', 'full', 'custom'],
                       default='default', help='配置类型')
    parser.add_argument('--config-file', help='自定义配置文件路径')

    parser.add_argument('--priority', '-p', choices=[p.value for p in ETFPriority],
                       default='must_have', help='最小优先级 (用于download-priority)')

    parser.add_argument('--etf-codes', '-e', nargs='+', help='指定ETF代码 (用于download-specific)')
    parser.add_argument('--etf-code', help='单个ETF代码 (用于update)')

    parser.add_argument('--category', help='ETF分类 (用于list)')
    parser.add_argument('--max-items', type=int, default=50, help='最大显示数量 (用于list)')

    parser.add_argument('--days-back', type=int, default=30, help='更新天数 (用于update)')

    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')

    args = parser.parse_args()

    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 打印横幅
    print_banner()

    # 加载配置
    if args.config == 'custom' and args.config_file:
        try:
            config = ETFConfig.from_yaml(args.config_file)
            print(f"✅ 已加载自定义配置: {args.config_file}")
        except Exception as e:
            print(f"❌ 加载配置文件失败: {e}")
            return
    else:
        config_map = {
            'default': create_default_config,
            'quick': create_quick_download_config,
            'full': create_full_download_config
        }
        config = config_map[args.config]()
        print(f"✅ 使用{args.config}配置")

    # 检查Token
    if config.source == ETFDataSource.TUSHARE and not config.tushare_token:
        print("❌ Tushare Token未设置")
        print("请设置环境变量 TUSHARE_TOKEN 或在配置文件中指定")
        return

    # 创建ETF清单管理器
    list_manager = ETFListManager()

    # 显示配置信息
    print(f"数据源: {config.source.value}")
    print(f"下载类型: {[dt.value for dt in config.download_types]}")
    print(f"时间范围: {config.start_date} ~ {config.end_date}")
    print(f"数据目录: {config.base_dir}")
    print(f"保存格式: {config.save_format}")

    # 执行动作
    success = False

    if args.action == 'summary':
        list_manager.print_summary()
        print_etf_categories(list_manager)
        success = True

    elif args.action == 'list':
        list_etfs(list_manager, args.category, args.max_items)
        success = True

    elif args.action == 'download-core':
        success = download_core_etfs(config, list_manager)

    elif args.action == 'download-priority':
        success = download_by_priority(config, list_manager, args.priority)

    elif args.action == 'download-specific':
        if not args.etf_codes:
            print("❌ 请指定ETF代码 (--etf-codes)")
            return
        success = download_specific_etfs(config, list_manager, args.etf_codes)

    elif args.action == 'update':
        if not args.etf_code:
            print("❌ 请指定ETF代码 (--etf-code)")
            return
        success = update_etf_data(config, list_manager, args.etf_code, args.days_back)

    elif args.action == 'validate':
        validate_data(config, list_manager)
        success = True

    # 显示结果
    if success:
        print("\n✅ 操作完成")
    else:
        print("\n❌ 操作失败")
        sys.exit(1)


if __name__ == "__main__":
    main()