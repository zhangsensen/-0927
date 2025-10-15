#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF下载清单和配置文件
包含所有需要下载的ETF代码、名称和分类信息
按用户建议优先级排序
"""

# 现有的ETF（已有数据）
EXISTING_ETFS = [
    {
        "code": "159915",
        "name": "创业板ETF",
        "category": "宽基指数",
        "subcategory": "成长风格",
        "priority": "high",
        "daily_volume": "69.82亿",
        "file_exists": True,
    },
    {
        "code": "510300",
        "name": "沪深300ETF",
        "category": "宽基指数",
        "subcategory": "大盘蓝筹",
        "priority": "core",
        "daily_volume": "超百亿",
        "file_exists": True,
    },
    {
        "code": "510500",
        "name": "中证500ETF",
        "category": "宽基指数",
        "subcategory": "中盘代表",
        "priority": "high",
        "daily_volume": "超百亿",
        "file_exists": True,
    },
    {
        "code": "512100",
        "name": "中证1000ETF",
        "category": "宽基指数",
        "subcategory": "小盘代表",
        "priority": "high",
        "daily_volume": "较高",
        "file_exists": True,
    },
    {
        "code": "518880",
        "name": "黄金ETF",
        "category": "商品避险",
        "subcategory": "避险资产",
        "priority": "must_have",
        "daily_volume": "较高",
        "file_exists": True,
    },
    {
        "code": "512660",
        "name": "军工ETF",
        "category": "行业主题",
        "subcategory": "国防军工",
        "priority": "medium",
        "daily_volume": "中等",
        "file_exists": True,
    },
    {
        "code": "512690",
        "name": "酒ETF",
        "category": "行业主题",
        "subcategory": "消费细分",
        "priority": "medium",
        "daily_volume": "中等",
        "file_exists": True,
    },
    {
        "code": "512880",
        "name": "证券ETF",
        "category": "金融板块",
        "subcategory": "周期金融",
        "priority": "medium",
        "daily_volume": "较高",
        "file_exists": True,
    },
    {
        "code": "159992",
        "name": "动漫ETF",
        "category": "行业主题",
        "subcategory": "文化主题",
        "priority": "low",
        "daily_volume": "较低",
        "file_exists": True,
    },
]

# 新增需要下载的ETF（按用户建议优先级排序）
NEW_ETFS = [
    # 宽基/风格（最高优先级）
    {
        "code": "510050",
        "name": "上证50ETF",
        "category": "宽基指数",
        "subcategory": "大盘蓝筹",
        "priority": "core",
        "daily_volume": "超百亿",
        "file_exists": False,
        "download_status": "pending",
        "note": "核心宽基 ⭐⭐⭐",
    },
    {
        "code": "159949",
        "name": "创业板50ETF",
        "category": "宽基指数",
        "subcategory": "成长风格",
        "priority": "core",
        "daily_volume": "较高",
        "file_exists": False,
        "download_status": "pending",
        "note": "核心宽基 ⭐⭐⭐",
    },
    {
        "code": "515180",
        "name": "中证红利ETF",
        "category": "宽基指数",
        "subcategory": "价值风格",
        "priority": "must_have",
        "daily_volume": "中等",
        "file_exists": False,
        "download_status": "pending",
        "note": "红利策略 ⭐⭐",
    },
    # 科技TMT与先进制造（高优先级）
    {
        "code": "159995",
        "name": "芯片ETF",
        "category": "科技半导体",
        "subcategory": "硬科技龙头",
        "priority": "must_have",
        "daily_volume": "15-25亿",
        "file_exists": False,
        "download_status": "pending",
        "note": "科技核心 ⭐⭐",
    },
    {
        "code": "512720",
        "name": "计算机ETF",
        "category": "科技TMT",
        "subcategory": "IT软件",
        "priority": "must_have",
        "daily_volume": "8-15亿",
        "file_exists": False,
        "download_status": "pending",
        "note": "科技核心 ⭐⭐",
    },
    {
        "code": "515650",
        "name": "通信ETF",
        "category": "科技TMT",
        "subcategory": "5G通信",
        "priority": "must_have",
        "daily_volume": "5-10亿",
        "file_exists": False,
        "download_status": "pending",
        "note": "5G主题 ⭐⭐",
    },
    {
        "code": "159801",
        "name": "机器人ETF",
        "category": "先进制造",
        "subcategory": "智能制造",
        "priority": "must_have",
        "daily_volume": "3-8亿",
        "file_exists": False,
        "download_status": "pending",
        "note": "智能制造 ⭐⭐",
    },
    {
        "code": "516090",
        "name": "AIGC人工智能ETF",
        "category": "科技AI",
        "subcategory": "人工智能",
        "priority": "must_have",
        "daily_volume": "5-12亿",
        "file_exists": False,
        "download_status": "pending",
        "note": "AI主题 ⭐⭐",
    },
    {
        "code": "588000",
        "name": "科创50ETF",
        "category": "科技成长",
        "subcategory": "科技创新核心",
        "priority": "must_have",
        "daily_volume": "61.23亿",
        "file_exists": False,
        "download_status": "pending",
        "note": "必配 ⭐⭐",
    },
    # 新能源链条（高优先级）
    {
        "code": "516160",
        "name": "新能源ETF",
        "category": "新能源",
        "subcategory": "新能源综合",
        "priority": "must_have",
        "daily_volume": "8-15亿",
        "file_exists": False,
        "download_status": "pending",
        "note": "新能源核心 ⭐⭐",
    },
    {
        "code": "515790",
        "name": "光伏ETF",
        "category": "新能源",
        "subcategory": "太阳能",
        "priority": "must_have",
        "daily_volume": "6-12亿",
        "file_exists": False,
        "download_status": "pending",
        "note": "光伏龙头 ⭐⭐",
    },
    {
        "code": "515030",
        "name": "新能源车ETF",
        "category": "新能源",
        "subcategory": "电动车产业链",
        "priority": "must_have",
        "daily_volume": "10-20亿",
        "file_exists": False,
        "download_status": "pending",
        "note": "新能源车 ⭐⭐",
    },
    {
        "code": "516520",
        "name": "储能ETF",
        "category": "新能源",
        "subcategory": "储能电池",
        "priority": "must_have",
        "daily_volume": "3-8亿",
        "file_exists": False,
        "download_status": "pending",
        "note": "储能主题 ⭐⭐",
    },
    # 消费+金融细分（中高优先级）
    {
        "code": "159928",
        "name": "消费ETF",
        "category": "消费",
        "subcategory": "大消费",
        "priority": "must_have",
        "daily_volume": "8-15亿",
        "file_exists": False,
        "download_status": "pending",
        "note": "消费龙头 ⭐⭐",
    },
    {
        "code": "512800",
        "name": "银行ETF",
        "category": "金融",
        "subcategory": "银行板块",
        "priority": "must_have",
        "daily_volume": "15-25亿",
        "file_exists": False,
        "download_status": "pending",
        "note": "银行板块 ⭐⭐",
    },
    {
        "code": "512010",
        "name": "医药ETF",
        "category": "医药健康",
        "subcategory": "医药综合",
        "priority": "must_have",
        "daily_volume": "10-20亿",
        "file_exists": False,
        "download_status": "pending",
        "note": "医药核心 ⭐⭐",
    },
]

# 推荐配置ETF（中优先级）
RECOMMENDED_ETFS = [
    # 医药健康细分
    {
        "code": "159859",
        "name": "生物医药ETF",
        "category": "医药健康",
        "subcategory": "生物医药",
        "priority": "recommended",
        "daily_volume": "3-8亿",
        "file_exists": False,
        "download_status": "pending",
        "note": "生物医药 ⭐",
    },
    # 消费细分
    {
        "code": "512980",
        "name": "家电ETF",
        "category": "消费",
        "subcategory": "家用电器",
        "priority": "recommended",
        "daily_volume": "2-5亿",
        "file_exists": False,
        "download_status": "pending",
        "note": "家电板块 ⭐",
    },
    # 资源与周期
    {
        "code": "512400",
        "name": "有色金属ETF",
        "category": "资源周期",
        "subcategory": "有色金属",
        "priority": "recommended",
        "daily_volume": "5-10亿",
        "file_exists": False,
        "download_status": "pending",
        "note": "有色龙头 ⭐",
    },
    {
        "code": "510170",
        "name": "石油石化ETF",
        "category": "资源周期",
        "subcategory": "油气石化",
        "priority": "recommended",
        "daily_volume": "2-6亿",
        "file_exists": False,
        "download_status": "pending",
        "note": "油气板块 ⭐",
    },
    {
        "code": "512000",
        "name": "煤炭ETF",
        "category": "资源周期",
        "subcategory": "煤炭板块",
        "priority": "recommended",
        "daily_volume": "3-8亿",
        "file_exists": False,
        "download_status": "pending",
        "note": "煤炭能源 ⭐",
    },
    {
        "code": "512200",
        "name": "钢铁ETF",
        "category": "资源周期",
        "subcategory": "钢铁板块",
        "priority": "recommended",
        "daily_volume": "3-6亿",
        "file_exists": False,
        "download_status": "pending",
        "note": "钢铁板块 ⭐",
    },
    # 商品贵金属
    {
        "code": "518850",
        "name": "白银ETF",
        "category": "商品避险",
        "subcategory": "贵金属",
        "priority": "recommended",
        "daily_volume": "1-3亿",
        "file_exists": False,
        "download_status": "pending",
        "note": "白银贵金属 ⭐",
    },
]

# 可选扩展ETF（债券、海外指数等）
OPTIONAL_ETFS = [
    # 债券/转债（防守资产）
    {
        "code": "511010",
        "name": "5年国债ETF",
        "category": "债券防守",
        "subcategory": "国债",
        "priority": "recommended",
        "daily_volume": "5-10亿",
        "file_exists": False,
        "download_status": "pending",
        "note": "国债防守 ⭐",
    },
    {
        "code": "511260",
        "name": "10年国债ETF",
        "category": "债券防守",
        "subcategory": "国债",
        "priority": "recommended",
        "daily_volume": "3-8亿",
        "file_exists": False,
        "download_status": "pending",
        "note": "长期国债 ⭐",
    },
    {
        "code": "511380",
        "name": "可转债ETF",
        "category": "债券防守",
        "subcategory": "可转债",
        "priority": "recommended",
        "daily_volume": "2-5亿",
        "file_exists": False,
        "download_status": "pending",
        "note": "可转债 ⭐",
    },
    # 海外指数（QDII）
    {
        "code": "513100",
        "name": "纳指100ETF",
        "category": "海外指数",
        "subcategory": "美股科技",
        "priority": "recommended",
        "daily_volume": "10-20亿",
        "file_exists": False,
        "download_status": "pending",
        "note": "美股科技 ⭐",
    },
    {
        "code": "513500",
        "name": "标普500ETF",
        "category": "海外指数",
        "subcategory": "美股大盘",
        "priority": "recommended",
        "daily_volume": "5-10亿",
        "file_exists": False,
        "download_status": "pending",
        "note": "美股大盘 ⭐",
    },
    {
        "code": "513130",
        "name": "恒生科技ETF",
        "category": "海外指数",
        "subcategory": "港股科技",
        "priority": "recommended",
        "daily_volume": "3-8亿",
        "file_exists": False,
        "download_status": "pending",
        "note": "港股科技 ⭐",
    },
    {
        "code": "159920",
        "name": "恒生ETF",
        "category": "海外指数",
        "subcategory": "港股大盘",
        "priority": "recommended",
        "daily_volume": "5-15亿",
        "file_exists": False,
        "download_status": "pending",
        "note": "港股大盘 ⭐",
    },
    {
        "code": "513050",
        "name": "中概互联网ETF",
        "category": "海外指数",
        "subcategory": "中概股",
        "priority": "recommended",
        "daily_volume": "8-15亿",
        "file_exists": False,
        "download_status": "pending",
        "note": "中概互联 ⭐",
    },
    # 其他可选ETF
    {
        "code": "159819",
        "name": "人工智能ETF",
        "category": "科技AI",
        "subcategory": "AI算力",
        "priority": "optional",
        "daily_volume": "3-8亿",
        "file_exists": False,
        "download_status": "pending",
        "note": "AI算力主题",
    },
    {
        "code": "159883",
        "name": "医疗器械ETF",
        "category": "医药健康",
        "subcategory": "医疗器械",
        "priority": "optional",
        "daily_volume": "2-5亿",
        "file_exists": False,
        "download_status": "pending",
        "note": "医疗器械细分",
    },
    {
        "code": "588200",
        "name": "科创芯片ETF",
        "category": "科技半导体",
        "subcategory": "芯片细分",
        "priority": "optional",
        "daily_volume": "2-6亿",
        "file_exists": False,
        "download_status": "pending",
        "note": "科创板芯片",
    },
]


def get_all_etfs():
    """获取所有ETF列表"""
    return EXISTING_ETFS + NEW_ETFS + RECOMMENDED_ETFS + OPTIONAL_ETFS


def get_existing_etfs():
    """获取已有ETF列表"""
    return EXISTING_ETFS


def get_new_etfs():
    """获取新增ETF列表"""
    return NEW_ETFS


def get_recommended_etfs():
    """获取推荐ETF列表"""
    return RECOMMENDED_ETFS


def get_optional_etfs():
    """获取可选ETF列表"""
    return OPTIONAL_ETFS


def get_must_have_etfs():
    """获取必须拥有的ETF列表"""
    return [etf for etf in get_all_etfs() if etf["priority"] in ["must_have", "core"]]


def get_high_priority_etfs():
    """获取高优先级ETF列表"""
    return [
        etf
        for etf in get_all_etfs()
        if etf["priority"] in ["must_have", "core", "high"]
    ]


def get_etf_by_code(code):
    """根据代码获取ETF信息"""
    for etf in get_all_etfs():
        if etf["code"] == code:
            return etf
    return None


def get_etf_by_category(category):
    """根据分类获取ETF列表"""
    return [etf for etf in get_all_etfs() if etf["category"] == category]


def get_etf_summary():
    """获取ETF清单汇总"""
    all_etfs = get_all_etfs()
    summary = {
        "total_count": len(all_etfs),
        "existing_count": len(EXISTING_ETFS),
        "new_count": len(NEW_ETFS),
        "recommended_count": len(RECOMMENDED_ETFS),
        "optional_count": len(OPTIONAL_ETFS),
        "completed_downloads": len(
            [
                etf
                for etf in NEW_ETFS + RECOMMENDED_ETFS + OPTIONAL_ETFS
                if etf.get("download_status") == "completed"
            ]
        ),
        "pending_downloads": len(
            [
                etf
                for etf in NEW_ETFS + RECOMMENDED_ETFS + OPTIONAL_ETFS
                if etf.get("download_status") != "completed"
            ]
        ),
        "categories": {},
    }

    # 统计各类别数量
    for etf in all_etfs:
        category = etf["category"]
        if category not in summary["categories"]:
            summary["categories"][category] = 0
        summary["categories"][category] += 1

    return summary


def print_etf_list_by_priority():
    """按优先级打印ETF列表"""
    print("=== 完整ETF清单（按优先级排序） ===")
    print(f"总计: {len(get_all_etfs())} 只ETF\n")

    # 核心必配
    core_etfs = [etf for etf in get_all_etfs() if etf["priority"] == "core"]
    print("🔥 核心必配ETF:")
    for etf in core_etfs:
        status = "✅" if etf.get("file_exists", False) else "❌"
        print(
            f"  {status} {etf['code']} - {etf['name']} ({etf['category']}) - {etf.get('note', '')}"
        )

    # 必须配置
    must_have_etfs = [etf for etf in get_all_etfs() if etf["priority"] == "must_have"]
    print(f"\n⭐ 必须配置ETF ({len(must_have_etfs)}只):")
    for etf in must_have_etfs:
        status = "✅" if etf.get("file_exists", False) else "❌"
        print(
            f"  {status} {etf['code']} - {etf['name']} ({etf['category']}) - {etf.get('note', '')}"
        )

    # 高优先级
    high_etfs = [etf for etf in get_all_etfs() if etf["priority"] == "high"]
    print(f"\n📈 高优先级ETF ({len(high_etfs)}只):")
    for etf in high_etfs:
        status = "✅" if etf.get("file_exists", False) else "❌"
        print(f"  {status} {etf['code']} - {etf['name']} ({etf['category']})")

    # 推荐配置
    recommended_etfs = [
        etf for etf in get_all_etfs() if etf["priority"] == "recommended"
    ]
    print(f"\n💡 推荐配置ETF ({len(recommended_etfs)}只):")
    for etf in recommended_etfs:
        status = "✅" if etf.get("file_exists", False) else "❌"
        print(
            f"  {status} {etf['code']} - {etf['name']} ({etf['category']}) - {etf.get('note', '')}"
        )

    # 可选扩展
    optional_etfs = [etf for etf in get_all_etfs() if etf["priority"] == "optional"]
    print(f"\n📊 可选扩展ETF ({len(optional_etfs)}只):")
    for etf in optional_etfs:
        status = "✅" if etf.get("file_exists", False) else "❌"
        print(
            f"  {status} {etf['code']} - {etf['name']} ({etf['category']}) - {etf.get('note', '')}"
        )


if __name__ == "__main__":
    # 打印ETF清单汇总
    summary = get_etf_summary()
    print("=== ETF清单汇总 ===")
    print(f"总数量: {summary['total_count']}")
    print(f"已有ETF: {summary['existing_count']}")
    print(f"新增ETF: {summary['new_count']}")
    print(f"推荐ETF: {summary['recommended_count']}")
    print(f"可选ETF: {summary['optional_count']}")
    print(f"已完成下载: {summary['completed_downloads']}")
    print(f"待下载: {summary['pending_downloads']}")

    print("\n=== 分类统计 ===")
    for category, count in summary["categories"].items():
        print(f"{category}: {count}个")

    print("\n=== 必配ETF清单 ===")
    must_have = get_must_have_etfs()
    for etf in must_have:
        status = (
            "✅"
            if etf.get("file_exists", False)
            or etf.get("download_status") == "completed"
            else "❌"
        )
        print(f"{status} {etf['code']} - {etf['name']} ({etf['daily_volume']})")

    print("\n")
    print_etf_list_by_priority()
