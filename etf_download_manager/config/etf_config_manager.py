#!/usr/bin/env python3
"""
ETF配置管理器
提供模块化的ETF配置管理功能，支持：
1. 从配置文件加载ETF清单
2. 按优先级和类别筛选ETF
3. 添加新ETF到配置
4. 生成下载任务列表
5. 配置验证和维护
"""

import yaml
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ETFInfo:
    """ETF信息数据类"""
    code: str
    name: str
    category: str
    priority: int
    description: str = ""

    def __post_init__(self):
        """数据验证"""
        if not self.code:
            raise ValueError("ETF代码不能为空")
        if self.priority not in [1, 2, 3]:
            raise ValueError("优先级必须是1、2或3")


class ETFConfigManager:
    """ETF配置管理器"""

    def __init__(self, config_path: str = None):
        """
        初始化配置管理器

        Args:
            config_path: 配置文件路径，默认使用当前目录下的etf_config.yaml
        """
        if config_path is None:
            config_path = Path(__file__).parent / "etf_config.yaml"

        self.config_path = Path(config_path)
        self.config = None
        self.load_config()

    def load_config(self) -> None:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            print(f"✅ 成功加载配置文件: {self.config_path}")
        except Exception as e:
            raise RuntimeError(f"加载配置文件失败: {e}")

    def save_config(self) -> None:
        """保存配置文件"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False,
                         allow_unicode=True, indent=2)
            print(f"✅ 成功保存配置文件: {self.config_path}")
        except Exception as e:
            raise RuntimeError(f"保存配置文件失败: {e}")

    def get_all_etfs(self) -> List[ETFInfo]:
        """获取所有ETF列表"""
        etfs = []

        if 'etf_list' not in self.config:
            return etfs

        for category_name, category_data in self.config['etf_list'].items():
            if 'etfs' in category_data:
                for etf_data in category_data['etfs']:
                    etf = ETFInfo(
                        code=etf_data['code'],
                        name=etf_data['name'],
                        category=etf_data['category'],
                        priority=etf_data['priority'],
                        description=etf_data.get('description', '')
                    )
                    etfs.append(etf)

        return etfs

    def get_etfs_by_priority(self, priority: int) -> List[ETFInfo]:
        """按优先级获取ETF列表"""
        all_etfs = self.get_all_etfs()
        return [etf for etf in all_etfs if etf.priority == priority]

    def get_etfs_by_category(self, category: str) -> List[ETFInfo]:
        """按类别获取ETF列表"""
        all_etfs = self.get_all_etfs()
        return [etf for etf in all_etfs if etf.category == category]

    def get_etfs_by_category_group(self, group_name: str) -> List[ETFInfo]:
        """按分类组获取ETF列表（如：core_etfs, sector_etfs等）"""
        etfs = []

        if 'etf_list' not in self.config or group_name not in self.config['etf_list']:
            return etfs

        group_data = self.config['etf_list'][group_name]
        if 'etfs' in group_data:
            for etf_data in group_data['etfs']:
                etf = ETFInfo(
                    code=etf_data['code'],
                    name=etf_data['name'],
                    category=etf_data['category'],
                    priority=etf_data['priority'],
                    description=etf_data.get('description', '')
                )
                etfs.append(etf)

        return etfs

    def get_download_list(self, priorities: List[int] = None,
                         categories: List[str] = None,
                         exclude_categories: List[str] = None) -> List[str]:
        """
        获取下载列表

        Args:
            priorities: 优先级列表，如[1, 2]
            categories: 包含的类别列表
            exclude_categories: 排除的类别列表

        Returns:
            ETF代码列表
        """
        all_etfs = self.get_all_etfs()

        # 按优先级筛选
        if priorities:
            all_etfs = [etf for etf in all_etfs if etf.priority in priorities]

        # 按类别筛选
        if categories:
            all_etfs = [etf for etf in all_etfs if etf.category in categories]

        # 排除特定类别
        if exclude_categories:
            all_etfs = [etf for etf in all_etfs if etf.category not in exclude_categories]

        # 按优先级和代码排序
        all_etfs.sort(key=lambda x: (x.priority, x.code))

        return [etf.code for etf in all_etfs]

    def add_etf(self, group_name: str, etf_info: ETFInfo) -> None:
        """
        添加新ETF到指定分类组

        Args:
            group_name: 分类组名称（如：sector_etfs）
            etf_info: ETF信息
        """
        if 'etf_list' not in self.config:
            self.config['etf_list'] = {}

        if group_name not in self.config['etf_list']:
            # 创建新的分类组
            self.config['etf_list'][group_name] = {
                'name': group_name.replace('_', ' ').title(),
                'description': f"{group_name}相关的ETF产品",
                'etfs': []
            }

        # 检查是否已存在
        existing_codes = [etf['code'] for etf in self.config['etf_list'][group_name]['etfs']]
        if etf_info.code in existing_codes:
            print(f"⚠️  ETF {etf_info.code} 已存在于 {group_name} 中")
            return

        # 添加新ETF
        new_etf = {
            'code': etf_info.code,
            'name': etf_info.name,
            'category': etf_info.category,
            'priority': etf_info.priority,
            'description': etf_info.description
        }

        self.config['etf_list'][group_name]['etfs'].append(new_etf)
        print(f"✅ 成功添加ETF {etf_info.code} 到 {group_name}")

    def remove_etf(self, etf_code: str) -> bool:
        """
        删除指定ETF

        Args:
            etf_code: ETF代码

        Returns:
            是否成功删除
        """
        if 'etf_list' not in self.config:
            return False

        for group_name, group_data in self.config['etf_list'].items():
            if 'etfs' in group_data:
                original_count = len(group_data['etfs'])
                group_data['etfs'] = [etf for etf in group_data['etfs'] if etf['code'] != etf_code]

                if len(group_data['etfs']) < original_count:
                    print(f"✅ 成功删除ETF {etf_code} 从 {group_name}")
                    return True

        print(f"⚠️  未找到ETF {etf_code}")
        return False

    def get_statistics(self) -> Dict[str, Any]:
        """获取配置统计信息"""
        stats = {
            'total_etfs': 0,
            'categories': {},
            'priorities': {1: 0, 2: 0, 3: 0},
            'groups': {}
        }

        all_etfs = self.get_all_etfs()
        stats['total_etfs'] = len(all_etfs)

        # 统计类别
        for etf in all_etfs:
            # 类别统计
            if etf.category not in stats['categories']:
                stats['categories'][etf.category] = 0
            stats['categories'][etf.category] += 1

            # 优先级统计
            stats['priorities'][etf.priority] += 1

        # 统计分组
        if 'etf_list' in self.config:
            for group_name, group_data in self.config['etf_list'].items():
                if 'etfs' in group_data:
                    stats['groups'][group_name] = {
                        'name': group_data.get('name', group_name),
                        'count': len(group_data['etfs']),
                        'description': group_data.get('description', '')
                    }

        return stats

    def validate_config(self) -> List[str]:
        """
        验证配置文件

        Returns:
            问题列表，空列表表示无问题
        """
        issues = []

        # 检查基本结构
        if 'etf_list' not in self.config:
            issues.append("缺少etf_list配置节点")
            return issues

        # 检查重复代码
        all_codes = []
        for group_name, group_data in self.config['etf_list'].items():
            if 'etfs' not in group_data:
                issues.append(f"分组 {group_name} 缺少etfs列表")
                continue

            for i, etf in enumerate(group_data['etfs']):
                # 检查必需字段
                required_fields = ['code', 'name', 'category', 'priority']
                for field in required_fields:
                    if field not in etf:
                        issues.append(f"{group_name}.etfs[{i}] 缺少必需字段: {field}")

                # 检查代码重复
                if 'code' in etf:
                    if etf['code'] in all_codes:
                        issues.append(f"发现重复ETF代码: {etf['code']}")
                    else:
                        all_codes.append(etf['code'])

                    # 检查代码格式
                    code = etf['code']
                    if not ('.SH' in code or '.SZ' in code):
                        issues.append(f"ETF代码格式不正确: {code}")

                # 检查优先级
                if 'priority' in etf and etf['priority'] not in [1, 2, 3]:
                    issues.append(f"{group_name}.etfs[{i}] 优先级必须是1、2或3，当前为: {etf['priority']}")

        return issues

    def print_summary(self) -> None:
        """打印配置摘要"""
        stats = self.get_statistics()

        print("=== ETF配置摘要 ===")
        print(f"总ETF数量: {stats['total_etfs']}")
        print(f"分组数量: {len(stats['groups'])}")
        print(f"类别数量: {len(stats['categories'])}")

        print("\n=== 按优先级分布 ===")
        for priority, count in stats['priorities'].items():
            priority_name = {1: "最高优先级", 2: "高优先级", 3: "一般优先级"}
            print(f"{priority_name[priority]}: {count}只")

        print("\n=== 按类别分布 ===")
        for category, count in sorted(stats['categories'].items()):
            print(f"{category}: {count}只")

        print("\n=== 分组详情 ===")
        for group_key, group_info in stats['groups'].items():
            print(f"{group_info['name']}: {group_info['count']}只 - {group_info['description']}")

    def export_download_list(self, output_file: str = None,
                           priorities: List[int] = None) -> None:
        """
        导出下载列表

        Args:
            output_file: 输出文件路径，默认输出到当前目录
            priorities: 优先级筛选
        """
        if output_file is None:
            output_file = "etf_download_list.txt"

        download_list = self.get_download_list(priorities=priorities)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# ETF下载列表\n")
            f.write(f"# 生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# 总数量: {len(download_list)}\n\n")

            for i, etf_code in enumerate(download_list, 1):
                f.write(f"{etf_code}\n")

        print(f"✅ 下载列表已导出到: {output_file}")


# 便捷函数
def load_etf_config(config_path: str = None) -> ETFConfigManager:
    """加载ETF配置管理器"""
    return ETFConfigManager(config_path)


if __name__ == "__main__":
    # 示例用法
    import pandas as pd

    print("=== ETF配置管理器示例 ===")

    # 加载配置
    config_manager = ETFConfigManager()

    # 打印摘要
    config_manager.print_summary()

    # 验证配置
    issues = config_manager.validate_config()
    if issues:
        print(f"\n⚠️  发现 {len(issues)} 个配置问题:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ 配置验证通过，无发现问题")

    # 示例：获取最高优先级ETF
    priority_1_etfs = config_manager.get_etfs_by_priority(1)
    print(f"\n最高优先级ETF ({len(priority_1_etfs)}只):")
    for etf in priority_1_etfs:
        print(f"  {etf.code} - {etf.name}")

    # 示例：导出下载列表
    config_manager.export_download_list("etf_download_priority_1_2.txt", priorities=[1, 2])

    print("\n=== 示例完成 ===")