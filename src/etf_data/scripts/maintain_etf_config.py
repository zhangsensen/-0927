#!/usr/bin/env python3
"""
ETF配置维护工具
提供ETF配置的日常维护功能：
1. 查看配置摘要
2. 添加新ETF
3. 删除ETF
4. 生成下载列表
5. 验证配置完整性
6. 导出配置报告
"""

import argparse
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from config import ETFConfigManager, ETFInfo


def view_summary(config_manager: ETFConfigManager) -> None:
    """查看配置摘要"""
    print("=" * 50)
    print("ETF配置摘要")
    print("=" * 50)
    config_manager.print_summary()


def add_new_etf(config_manager: ETFConfigManager, args) -> None:
    """添加新ETF"""
    try:
        etf_info = ETFInfo(
            code=args.code,
            name=args.name,
            category=args.category,
            priority=args.priority,
            description=args.description or "",
        )

        config_manager.add_etf(args.group, etf_info)
        config_manager.save_config()
        print(f"✅ 成功添加ETF: {args.code}")

    except ValueError as e:
        print(f"❌ 添加失败: {e}")


def remove_etf(config_manager: ETFConfigManager, args) -> None:
    """删除ETF"""
    if config_manager.remove_etf(args.code):
        config_manager.save_config()
        print(f"✅ 成功删除ETF: {args.code}")
    else:
        print(f"❌ 删除失败: 未找到ETF {args.code}")


def list_etfs(config_manager: ETFConfigManager, args) -> None:
    """列出ETF"""
    if args.priority:
        etfs = config_manager.get_etfs_by_priority(args.priority)
        title = f"优先级 {args.priority} 的ETF"
    elif args.category:
        etfs = config_manager.get_etfs_by_category(args.category)
        title = f"类别 '{args.category}' 的ETF"
    elif args.group:
        etfs = config_manager.get_etfs_by_category_group(args.group)
        group_info = config_manager.config["etf_list"].get(args.group, {})
        title = f"分组 '{group_info.get('name', args.group)}' 的ETF"
    else:
        etfs = config_manager.get_all_etfs()
        title = "所有ETF"

    print(f"\n=== {title} ({len(etfs)}只) ===")
    print(f"{'代码':<12} {'名称':<20} {'类别':<12} {'优先级':<6} {'描述'}")
    print("-" * 80)

    for etf in etfs:
        print(
            f"{etf.code:<12} {etf.name:<20} {etf.category:<12} {etf.priority:<6} {etf.description[:30]}"
        )


def generate_download_list(config_manager: ETFConfigManager, args) -> None:
    """生成下载列表"""
    priorities = args.priorities if args.priorities else None
    download_list = config_manager.get_download_list(priorities=priorities)

    output_file = args.output or "etf_download_list.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# ETF下载列表\n")
        f.write(f"# 生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# 总数量: {len(download_list)}\n")

        if priorities:
            f.write(f"# 优先级筛选: {priorities}\n")

        f.write("\n")

        for i, etf_code in enumerate(download_list, 1):
            # 查找ETF详细信息
            all_etfs = config_manager.get_all_etfs()
            etf_info = next((etf for etf in all_etfs if etf.code == etf_code), None)

            if etf_info:
                f.write(f"# {i}. {etf_info.name} - {etf_info.category}\n")
            f.write(f"{etf_code}\n")

    print(f"✅ 下载列表已生成: {output_file}")
    print(f"包含 {len(download_list)} 只ETF")


def validate_config(config_manager: ETFConfigManager, args) -> None:
    """验证配置"""
    print("正在验证配置...")
    issues = config_manager.validate_config()

    if not issues:
        print("✅ 配置验证通过，无发现问题")
        return

    print(f"⚠️  发现 {len(issues)} 个配置问题:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")

    if args.fix:
        print("\n尝试自动修复...")
        # 这里可以添加自动修复逻辑
        print("自动修复功能待实现")


def export_report(config_manager: ETFConfigManager, args) -> None:
    """导出配置报告"""
    output_file = args.output or "etf_config_report.md"

    stats = config_manager.get_statistics()
    all_etfs = config_manager.get_all_etfs()

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# ETF配置报告\n\n")
        f.write(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 摘要信息
        f.write("## 摘要信息\n\n")
        f.write(f"- 总ETF数量: {stats['total_etfs']}\n")
        f.write(f"- 分组数量: {len(stats['groups'])}\n")
        f.write(f"- 类别数量: {len(stats['categories'])}\n\n")

        # 优先级分布
        f.write("## 优先级分布\n\n")
        f.write("| 优先级 | 数量 | 说明 |\n")
        f.write("|--------|------|------|\n")
        priority_names = {1: "最高优先级", 2: "高优先级", 3: "一般优先级"}
        for priority, count in stats["priorities"].items():
            f.write(f"| {priority} | {count} | {priority_names[priority]} |\n")
        f.write("\n")

        # 类别分布
        f.write("## 类别分布\n\n")
        f.write("| 类别 | 数量 |\n")
        f.write("|------|------|\n")
        for category, count in sorted(stats["categories"].items()):
            f.write(f"| {category} | {count} |\n")
        f.write("\n")

        # 分组详情
        f.write("## 分组详情\n\n")
        for group_key, group_info in stats["groups"].items():
            f.write(f"### {group_info['name']}\n\n")
            f.write(f"**描述**: {group_info['description']}\n\n")
            f.write(f"**数量**: {group_info['count']}只\n\n")

            # 列出该分组的ETF
            group_etfs = config_manager.get_etfs_by_category_group(group_key)
            if group_etfs:
                f.write("| 代码 | 名称 | 类别 | 优先级 | 描述 |\n")
                f.write("|------|------|------|--------|------|\n")
                for etf in group_etfs:
                    f.write(
                        f"| {etf.code} | {etf.name} | {etf.category} | {etf.priority} | {etf.description[:50]} |\n"
                    )
                f.write("\n")

        # 完整ETF列表
        f.write("## 完整ETF列表\n\n")
        f.write("| 代码 | 名称 | 类别 | 优先级 | 描述 |\n")
        f.write("|------|------|------|--------|------|\n")
        for etf in all_etfs:
            f.write(
                f"| {etf.code} | {etf.name} | {etf.category} | {etf.priority} | {etf.description[:50]} |\n"
            )

    print(f"✅ 配置报告已导出: {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ETF配置维护工具")
    parser.add_argument("--config", help="配置文件路径", default=None)

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # 查看摘要
    subparsers.add_parser("summary", help="查看配置摘要")

    # 添加ETF
    add_parser = subparsers.add_parser("add", help="添加新ETF")
    add_parser.add_argument("--code", required=True, help="ETF代码（如：510300.SH）")
    add_parser.add_argument("--name", required=True, help="ETF名称")
    add_parser.add_argument("--category", required=True, help="ETF类别")
    add_parser.add_argument(
        "--priority", type=int, choices=[1, 2, 3], required=True, help="优先级（1-3）"
    )
    add_parser.add_argument("--description", help="ETF描述")
    add_parser.add_argument(
        "--group", required=True, help="分组名称（如：sector_etfs）"
    )

    # 删除ETF
    remove_parser = subparsers.add_parser("remove", help="删除ETF")
    remove_parser.add_argument("--code", required=True, help="ETF代码")

    # 列出ETF
    list_parser = subparsers.add_parser("list", help="列出ETF")
    list_group = list_parser.add_mutually_exclusive_group()
    list_group.add_argument(
        "--priority", type=int, choices=[1, 2, 3], help="按优先级筛选"
    )
    list_group.add_argument("--category", help="按类别筛选")
    list_group.add_argument("--group", help="按分组筛选")

    # 生成下载列表
    download_parser = subparsers.add_parser("download-list", help="生成下载列表")
    download_parser.add_argument(
        "--priorities", nargs="+", type=int, choices=[1, 2, 3], help="优先级筛选"
    )
    download_parser.add_argument("--output", help="输出文件路径")

    # 验证配置
    validate_parser = subparsers.add_parser("validate", help="验证配置")
    validate_parser.add_argument("--fix", action="store_true", help="尝试自动修复")

    # 导出报告
    export_parser = subparsers.add_parser("export", help="导出配置报告")
    export_parser.add_argument("--output", help="输出文件路径")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        # 加载配置管理器
        config_manager = ETFConfigManager(args.config)

        # 执行命令
        if args.command == "summary":
            view_summary(config_manager)
        elif args.command == "add":
            add_new_etf(config_manager, args)
        elif args.command == "remove":
            remove_etf(config_manager, args)
        elif args.command == "list":
            list_etfs(config_manager, args)
        elif args.command == "download-list":
            generate_download_list(config_manager, args)
        elif args.command == "validate":
            validate_config(config_manager, args)
        elif args.command == "export":
            export_report(config_manager, args)

    except Exception as e:
        print(f"❌ 执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import pandas as pd  # 导入pandas用于时间戳

    main()
