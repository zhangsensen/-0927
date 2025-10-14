#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF清单管理模块
"""

from typing import List, Dict, Optional

from .models import ETFInfo, ETFPriority, ETFExchange, ETFStatus
from .config import ETFListConfig


class ETFListManager:
    """ETF清单管理器"""

    def __init__(self, list_config: Optional[ETFListConfig] = None):
        """
        初始化ETF清单管理器

        Args:
            list_config: ETF清单配置
        """
        self.list_config = list_config or self._load_default_list()
        self._etf_cache = {}

    def _load_default_list(self) -> ETFListConfig:
        """加载默认ETF清单"""
        try:
            # 尝试从现有的etf_download_list.py加载
            return ETFListConfig.from_python_file("/Users/zhangshenshen/深度量化0927/etf_download_list.py")
        except:
            # 如果加载失败，返回内置的默认清单
            return self._get_builtin_list()

    def _get_builtin_list(self) -> ETFListConfig:
        """获取内置的默认ETF清单"""
        existing_etfs = [
            {
                'code': '510300',
                'name': '沪深300ETF',
                'category': '宽基指数',
                'subcategory': '大盘蓝筹',
                'priority': 'core',
                'daily_volume': '超百亿',
                'file_exists': True
            },
            {
                'code': '159915',
                'name': '创业板ETF',
                'category': '宽基指数',
                'subcategory': '成长风格',
                'priority': 'high',
                'daily_volume': '69.82亿',
                'file_exists': True
            },
            {
                'code': '510500',
                'name': '中证500ETF',
                'category': '宽基指数',
                'subcategory': '中盘代表',
                'priority': 'high',
                'daily_volume': '超百亿',
                'file_exists': True
            },
            {
                'code': '518880',
                'name': '黄金ETF',
                'category': '商品避险',
                'subcategory': '避险资产',
                'priority': 'hedge',
                'daily_volume': '较高',
                'file_exists': True
            }
        ]

        new_etfs = [
            {
                'code': '588000',
                'name': '科创50ETF',
                'category': '科技成长',
                'subcategory': '科技创新核心',
                'priority': 'must_have',
                'daily_volume': '61.23亿',
                'file_exists': False,
                'download_status': 'pending',
                'note': '必配 ⭐'
            },
            {
                'code': '512480',
                'name': '半导体ETF',
                'category': '科技栈',
                'subcategory': '硬科技龙头',
                'priority': 'must_have',
                'daily_volume': '20.07亿',
                'file_exists': False,
                'download_status': 'pending',
                'note': '必配 ⭐'
            },
            {
                'code': '515790',
                'name': '光伏ETF',
                'category': '主题赛道',
                'subcategory': '清洁能源',
                'priority': 'must_have',
                'daily_volume': '6-12亿',
                'file_exists': False,
                'download_status': 'pending',
                'note': '必配 ⭐'
            }
        ]

        return ETFListConfig(existing_etfs=existing_etfs, new_etfs=new_etfs)

    def dict_to_etf_info(self, etf_dict: Dict) -> ETFInfo:
        """将字典转换为ETFInfo对象"""
        return ETFInfo(
            code=etf_dict['code'],
            name=etf_dict['name'],
            ts_code=etf_dict.get('ts_code', ''),
            category=etf_dict.get('category', ''),
            subcategory=etf_dict.get('subcategory', ''),
            priority=ETFPriority(etf_dict.get('priority', 'optional')),
            exchange=ETFExchange.SH if etf_dict['code'].startswith('5') else ETFExchange.SZ,
            daily_volume=etf_dict.get('daily_volume', ''),
            description=etf_dict.get('description', ''),
            file_exists=etf_dict.get('file_exists', False),
            download_status=ETFStatus(etf_dict.get('download_status', 'pending')),
            note=etf_dict.get('note', '')
        )

    def get_all_etfs(self) -> List[ETFInfo]:
        """获取所有ETF"""
        cache_key = "all_etfs"
        if cache_key not in self._etf_cache:
            all_etfs_dict = self.list_config.get_all_etfs()
            self._etf_cache[cache_key] = [self.dict_to_etf_info(etf_dict) for etf_dict in all_etfs_dict]
        return self._etf_cache[cache_key]

    def get_existing_etfs(self) -> List[ETFInfo]:
        """获取已有ETF"""
        cache_key = "existing_etfs"
        if cache_key not in self._etf_cache:
            existing_etfs_dict = self.list_config.existing_etfs
            self._etf_cache[cache_key] = [self.dict_to_etf_info(etf_dict) for etf_dict in existing_etfs_dict]
        return self._etf_cache[cache_key]

    def get_new_etfs(self) -> List[ETFInfo]:
        """获取新增ETF"""
        cache_key = "new_etfs"
        if cache_key not in self._etf_cache:
            new_etfs_dict = self.list_config.new_etfs
            self._etf_cache[cache_key] = [self.dict_to_etf_info(etf_dict) for etf_dict in new_etfs_dict]
        return self._etf_cache[cache_key]

    def get_optional_etfs(self) -> List[ETFInfo]:
        """获取可选ETF"""
        cache_key = "optional_etfs"
        if cache_key not in self._etf_cache:
            optional_etfs_dict = self.list_config.optional_etfs
            self._etf_cache[cache_key] = [self.dict_to_etf_info(etf_dict) for etf_dict in optional_etfs_dict]
        return self._etf_cache[cache_key]

    def get_must_have_etfs(self) -> List[ETFInfo]:
        """获取必须拥有的ETF"""
        all_etfs = self.get_all_etfs()
        return [etf for etf in all_etfs if etf.priority in [ETFPriority.CORE, ETFPriority.MUST_HAVE]]

    def get_high_priority_etfs(self) -> List[ETFInfo]:
        """获取高优先级ETF"""
        all_etfs = self.get_all_etfs()
        return [etf for etf in all_etfs if etf.priority in [ETFPriority.CORE, ETFPriority.MUST_HAVE, ETFPriority.HIGH]]

    def get_etf_by_code(self, code: str) -> Optional[ETFInfo]:
        """根据代码获取ETF信息"""
        all_etfs = self.get_all_etfs()
        for etf in all_etfs:
            if etf.code == code:
                return etf
        return None

    def get_etf_by_ts_code(self, ts_code: str) -> Optional[ETFInfo]:
        """根据TS代码获取ETF信息"""
        all_etfs = self.get_all_etfs()
        for etf in all_etfs:
            if etf.ts_code == ts_code:
                return etf
        return None

    def get_etfs_by_priority(self, priority: ETFPriority) -> List[ETFInfo]:
        """根据优先级获取ETF列表"""
        all_etfs = self.get_all_etfs()
        return [etf for etf in all_etfs if etf.priority == priority]

    def get_etfs_by_category(self, category: str) -> List[ETFInfo]:
        """根据分类获取ETF列表"""
        all_etfs = self.get_all_etfs()
        return [etf for etf in all_etfs if etf.category == category]

    def get_etfs_by_exchange(self, exchange: ETFExchange) -> List[ETFInfo]:
        """根据交易所获取ETF列表"""
        all_etfs = self.get_all_etfs()
        return [etf for etf in all_etfs if etf.exchange == exchange]

    def filter_etfs(self,
                    priorities: Optional[List[ETFPriority]] = None,
                    categories: Optional[List[str]] = None,
                    exchanges: Optional[List[ETFExchange]] = None,
                    include_codes: Optional[List[str]] = None,
                    exclude_codes: Optional[List[str]] = None) -> List[ETFInfo]:
        """
        筛选ETF

        Args:
            priorities: 优先级列表
            categories: 分类列表
            exchanges: 交易所列表
            include_codes: 包含的代码列表
            exclude_codes: 排除的代码列表

        Returns:
            筛选后的ETF列表
        """
        all_etfs = self.get_all_etfs()

        filtered_etfs = []

        for etf in all_etfs:
            # 优先级筛选
            if priorities and etf.priority not in priorities:
                continue

            # 分类筛选
            if categories and etf.category not in categories:
                continue

            # 交易所筛选
            if exchanges and etf.exchange not in exchanges:
                continue

            # 包含代码筛选
            if include_codes and etf.code not in include_codes:
                continue

            # 排除代码筛选
            if exclude_codes and etf.code in exclude_codes:
                continue

            filtered_etfs.append(etf)

        return filtered_etfs

    def get_etf_summary(self) -> Dict:
        """获取ETF清单汇总"""
        all_etfs = self.get_all_etfs()

        summary = {
            'total_count': len(all_etfs),
            'existing_count': len(self.get_existing_etfs()),
            'new_count': len(self.get_new_etfs()),
            'optional_count': len(self.get_optional_etfs()),
            'must_have_count': len(self.get_must_have_etfs()),
            'high_priority_count': len(self.get_high_priority_etfs()),
            'completed_downloads': len([etf for etf in all_etfs if etf.download_status == ETFStatus.COMPLETED]),
            'failed_downloads': len([etf for etf in all_etfs if etf.download_status == ETFStatus.FAILED]),
            'pending_downloads': len([etf for etf in all_etfs if etf.download_status == ETFStatus.PENDING]),
            'categories': {},
            'exchanges': {},
            'priorities': {}
        }

        # 统计各类别数量
        for etf in all_etfs:
            category = etf.category
            if category not in summary['categories']:
                summary['categories'][category] = 0
            summary['categories'][category] += 1

        # 统计各交易所数量
        for etf in all_etfs:
            exchange = etf.exchange.value
            if exchange not in summary['exchanges']:
                summary['exchanges'][exchange] = 0
            summary['exchanges'][exchange] += 1

        # 统计各优先级数量
        for etf in all_etfs:
            priority = etf.priority.value
            if priority not in summary['priorities']:
                summary['priorities'][priority] = 0
            summary['priorities'][priority] += 1

        return summary

    def print_etf_list(self, etfs: Optional[List[ETFInfo]] = None, max_items: int = 50):
        """打印ETF列表"""
        if etfs is None:
            etfs = self.get_all_etfs()

        print(f"{'代码':<8} {'名称':<20} {'交易所':<6} {'分类':<15} {'优先级':<10} {'状态':<10}")
        print("-" * 80)

        displayed_count = 0
        for etf in etfs:
            if displayed_count >= max_items:
                break

            status_icon = {
                ETFStatus.PENDING: "⏳",
                ETFStatus.DOWNLOADING: "⬇️",
                ETFStatus.COMPLETED: "✅",
                ETFStatus.FAILED: "❌"
            }.get(etf.download_status, "❓")

            priority_icon = ""
            if etf.priority in [ETFPriority.CORE, ETFPriority.MUST_HAVE]:
                priority_icon = "⭐"

            print(f"{etf.code:<8} {etf.name:<20} {etf.exchange.value:<6} "
                  f"{etf.category:<15} {etf.priority.value:<8}{priority_icon:<2} "
                  f"{status_icon} {etf.download_status.value}")

            displayed_count += 1

        if len(etfs) > max_items:
            print(f"... 还有 {len(etfs) - max_items} 只ETF未显示")

    def print_summary(self):
        """打印ETF清单汇总"""
        summary = self.get_etf_summary()

        print("=== ETF清单汇总 ===")
        print(f"总数量: {summary['total_count']}")
        print(f"已有ETF: {summary['existing_count']}")
        print(f"新增ETF: {summary['new_count']}")
        print(f"可选ETF: {summary['optional_count']}")
        print(f"必配ETF: {summary['must_have_count']}")
        print(f"高优先级ETF: {summary['high_priority_count']}")
        print()
        print("=== 下载状态 ===")
        print(f"已完成: {summary['completed_downloads']}")
        print(f"失败: {summary['failed_downloads']}")
        print(f"待下载: {summary['pending_downloads']}")
        print()

        print("=== 分类统计 ===")
        for category, count in sorted(summary['categories'].items()):
            print(f"{category}: {count}个")
        print()

        print("=== 交易所统计 ===")
        for exchange, count in sorted(summary['exchanges'].items()):
            print(f"{exchange}: {count}个")
        print()

        print("=== 优先级统计 ===")
        for priority, count in sorted(summary['priorities'].items()):
            icon = "⭐" if priority in ['core', 'must_have', 'high'] else ""
            print(f"{priority}: {count}个 {icon}")