#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF批量下载脚本
支持批量下载所有ETF或按分类下载
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from etf_data import (
    ETFConfig,
    ETFDownloadManager,
    ETFDownloadType,
    ETFListManager,
    ETFPriority,
)


def download_all_etfs():
    """下载所有ETF"""
    print("=" * 80)
    print("ETF全量下载器")
    print("=" * 80)

    # 检查Token
    tushare_token = os.getenv("TUSHARE_TOKEN")
    if not tushare_token:
        print("❌ 错误: 请设置环境变量 TUSHARE_TOKEN")
        print("例如: export TUSHARE_TOKEN='your_actual_token'")
        return

    # 创建配置
    config = ETFConfig(
        tushare_token=tushare_token,
        base_dir="raw/ETF",
        years_back=2,
        download_types=[ETFDownloadType.DAILY],
        save_format="parquet",
        request_delay=0.3,
        batch_size=20,
        verbose=True,
    )

    # 获取所有ETF
    list_manager = ETFListManager()
    all_etfs = list_manager.get_all_etfs()

    if not all_etfs:
        print("❌ 未找到ETF")
        return

    print(f"找到 {len(all_etfs)} 只ETF")

    # 按优先级分组显示
    priority_groups = {}
    for etf in all_etfs:
        priority = etf.priority.value
        if priority not in priority_groups:
            priority_groups[priority] = []
        priority_groups[priority].append(etf)

    print("\n=== 按优先级分组 ===")
    priority_order = [
        "core",
        "must_have",
        "high",
        "medium",
        "recommended",
        "hedge",
        "low",
        "optional",
    ]
    for priority in priority_order:
        if priority in priority_groups:
            count = len(priority_groups[priority])
            icon = "⭐" if priority in ["core", "must_have", "high"] else ""
            print(f"{priority}: {count}只 {icon}")

    print(f"\n开始下载所有ETF数据...")
    print("注意: 这可能需要较长时间...")

    # 创建下载器并开始下载
    try:
        downloader = ETFDownloadManager(config)
        stats = downloader.download_multiple_etfs(all_etfs)

        # 显示结果
        print("\n" + "=" * 80)
        print("全量下载完成！")
        print("=" * 80)
        print(f"总ETF数量: {stats.total_etfs}")
        print(f"成功下载: {stats.success_count}")
        print(f"失败数量: {stats.failed_count}")
        print(f"成功率: {stats.success_rate:.1f}%")
        print(f"总记录数: {stats.total_daily_records:,}")
        print(f"耗时: {stats.duration}")

        if stats.failed_etfs:
            print(f"\n失败的ETF ({len(stats.failed_etfs)}只):")
            for i, etf_code in enumerate(stats.failed_etfs[:10]):
                print(f"  ❌ {etf_code}")
            if len(stats.failed_etfs) > 10:
                print(f"  ...还有 {len(stats.failed_etfs) - 10} 只ETF失败")

        print(f"\n数据已保存至: {config.base_dir}/")
        print(f"✅ ETF数据下载完成！")

    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return


def download_by_categories():
    """按分类下载ETF"""
    print("=" * 80)
    print("ETF分类下载器")
    print("=" * 80)

    # 检查Token
    tushare_token = os.getenv("TUSHARE_TOKEN")
    if not tushare_token:
        print("❌ 错误: 请设置环境变量 TUSHARE_TOKEN")
        return

    # 创建配置
    config = ETFConfig(
        tushare_token=tushare_token,
        base_dir="raw/ETF",
        years_back=2,
        download_types=[ETFDownloadType.DAILY],
        save_format="parquet",
        request_delay=0.2,
        batch_size=15,
        verbose=True,
    )

    # 获取ETF清单
    list_manager = ETFListManager()
    summary = list_manager.get_etf_summary()

    print("可用分类:")
    for category, count in sorted(summary["categories"].items()):
        print(f"  {category}: {count}只")

    # 交互选择分类
    print("\n请选择要下载的分类 (用逗号分隔，输入'all'下载所有分类):")
    user_input = input("分类: ").strip()

    if user_input.lower() == "all":
        # 下载所有分类
        download_all_etfs()
        return

    # 解析用户输入
    selected_categories = [cat.strip() for cat in user_input.split(",") if cat.strip()]
    if not selected_categories:
        print("❌ 未选择有效分类")
        return

    # 筛选ETF
    filtered_etfs = []
    for category in selected_categories:
        if category in summary["categories"]:
            etfs = list_manager.get_etfs_by_category(category)
            filtered_etfs.extend(etfs)
            print(f"✅ 分类 '{category}': {len(etfs)}只ETF")
        else:
            print(f"❌ 分类 '{category}' 不存在")

    if not filtered_etfs:
        print("❌ 未找到匹配的ETF")
        return

    # 去重
    unique_etfs = list({etf.code: etf for etf in filtered_etfs}.values())
    print(f"\n共找到 {len(unique_etfs)} 只ETF")

    # 创建下载器并开始下载
    try:
        downloader = ETFDownloadManager(config)
        stats = downloader.download_multiple_etfs(unique_etfs)

        # 显示结果
        print("\n" + "=" * 80)
        print("分类下载完成！")
        print("=" * 80)
        print(f"总ETF数量: {stats.total_etfs}")
        print(f"成功下载: {stats.success_count}")
        print(f"失败数量: {stats.failed_count}")
        print(f"成功率: {stats.success_rate:.1f}%")
        print(f"总记录数: {stats.total_daily_records:,}")
        print(f"耗时: {stats.duration}")

        if stats.failed_etfs:
            print(f"\n失败的ETF: {', '.join(stats.failed_etfs)}")

        print(f"\n数据已保存至: {config.base_dir}/daily/")
        print(f"✅ ETF数据下载完成！")

    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return


def download_high_priority():
    """下载高优先级ETF"""
    print("=" * 80)
    print("高优先级ETF下载器")
    print("=" * 80)

    # 检查Token
    tushare_token = os.getenv("TUSHARE_TOKEN")
    if not tushare_token:
        print("❌ 错误: 请设置环境变量 TUSHARE_TOKEN")
        return

    # 创建配置
    config = ETFConfig(
        tushare_token=tushare_token,
        base_dir="raw/ETF",
        years_back=2,
        download_types=[ETFDownloadType.DAILY],
        save_format="parquet",
        request_delay=0.2,
        verbose=True,
    )

    # 获取高优先级ETF
    list_manager = ETFListManager()
    high_priority_etfs = list_manager.filter_etfs(
        priorities=[ETFPriority.CORE, ETFPriority.MUST_HAVE, ETFPriority.HIGH]
    )

    if not high_priority_etfs:
        print("❌ 未找到高优先级ETF")
        return

    print(f"找到 {len(high_priority_etfs)} 只高优先级ETF:")
    for etf in high_priority_etfs[:10]:
        priority_icon = "⭐⭐⭐" if etf.priority == ETFPriority.CORE else "⭐⭐"
        print(f"  {etf.code} - {etf.name} {priority_icon}")
    if len(high_priority_etfs) > 10:
        print(f"  ...还有 {len(high_priority_etfs) - 10} 只ETF")

    print(f"\n开始下载高优先级ETF数据...")

    # 创建下载器并开始下载
    try:
        downloader = ETFDownloadManager(config)
        stats = downloader.download_multiple_etfs(high_priority_etfs)

        # 显示结果
        print("\n" + "=" * 80)
        print("高优先级ETF下载完成！")
        print("=" * 80)
        print(f"总ETF数量: {stats.total_etfs}")
        print(f"成功下载: {stats.success_count}")
        print(f"失败数量: {stats.failed_count}")
        print(f"成功率: {stats.success_rate:.1f}%")
        print(f"总记录数: {stats.total_daily_records:,}")
        print(f"耗时: {stats.duration}")

        if stats.failed_etfs:
            print(f"\n失败的ETF: {', '.join(stats.failed_etfs)}")

        print(f"\n数据已保存至: {config.base_dir}/daily/")
        print(f"✅ 高优先级ETF数据下载完成！")

    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return


def main():
    """主函数"""
    print("ETF批量下载器")
    print("1. 下载所有ETF")
    print("2. 按分类下载")
    print("3. 下载高优先级ETF")

    try:
        choice = input("\n请选择下载方式 (1/2/3): ").strip()
        if choice == "1":
            download_all_etfs()
        elif choice == "2":
            download_by_categories()
        elif choice == "3":
            download_high_priority()
        else:
            print("❌ 无效选择")
    except KeyboardInterrupt:
        print("\n\n下载已取消")
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")


if __name__ == "__main__":
    main()
