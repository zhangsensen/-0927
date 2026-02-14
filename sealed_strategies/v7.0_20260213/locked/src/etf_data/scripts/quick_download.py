#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF快速下载脚本
最简单的ETF数据下载方式
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
)


def main():
    """主函数 - 快速下载核心ETF"""
    print("=" * 60)
    print("ETF快速下载器")
    print("=" * 60)

    # 检查Token
    tushare_token = os.getenv("TUSHARE_TOKEN")
    if not tushare_token:
        print("❌ 错误: 请设置环境变量 TUSHARE_TOKEN")
        print("例如: export TUSHARE_TOKEN='your_actual_token'")
        return

    print(f"✅ Token已设置")
    print(f"开始下载核心ETF数据...")
    print()

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

    # 获取核心ETF
    list_manager = ETFListManager()
    core_etfs = list_manager.get_must_have_etfs()

    if not core_etfs:
        print("❌ 未找到核心ETF")
        return

    print(f"找到 {len(core_etfs)} 只核心ETF:")
    for etf in core_etfs[:5]:  # 只显示前5只
        print(f"  {etf.code} - {etf.name}")
    if len(core_etfs) > 5:
        print(f"  ...还有 {len(core_etfs) - 5} 只ETF")
    print()

    # 创建下载器并开始下载
    try:
        downloader = ETFDownloadManager(config)
        stats = downloader.download_multiple_etfs(core_etfs)

        # 显示结果
        print("\n" + "=" * 60)
        print("下载完成！")
        print("=" * 60)
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


if __name__ == "__main__":
    main()
