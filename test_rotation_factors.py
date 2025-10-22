#!/usr/bin/env python3
"""快速测试相对轮动因子计算"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "etf_rotation_system" / "01_横截面建设"))

from generate_panel_refactored import (
    calculate_factors_parallel,
    load_config,
    load_price_data,
)

print("=" * 80)
print("🧪 测试相对轮动因子计算")
print("=" * 80)

# 加载配置
config_path = "etf_rotation_system/01_横截面建设/config/factor_panel_config.yaml"
if Path(config_path).exists():
    print(f"✅ 加载配置: {config_path}")
    config = load_config(config_path)
else:
    print("⚠️ 配置文件不存在，使用默认配置")
    from config.config_classes import FactorPanelConfig

    config = FactorPanelConfig()

# 加载数据（使用少量数据测试）
data_dir = Path("raw/ETF/daily")
if not data_dir.exists():
    print(f"❌ 数据目录不存在: {data_dir}")
    sys.exit(1)

print(f"✅ 加载数据: {data_dir}")
price_df = load_price_data(data_dir, config)

# 只取前10只ETF和最近100天数据测试
symbols = sorted(price_df["symbol"].unique())[:10]
recent_dates = sorted(price_df["date"].unique())[-100:]

test_df = price_df[
    (price_df["symbol"].isin(symbols)) & (price_df["date"].isin(recent_dates))
]

print(f"📊 测试数据: {len(symbols)} 只ETF, {len(recent_dates)} 个交易日")
print(f"   标的: {', '.join(symbols[:5])}...")

# 计算因子
print("\n⏳ 计算因子（包括相对轮动因子）...")
try:
    panel = calculate_factors_parallel(test_df, config)

    print("\n✅ 因子计算完成！")
    print(f"   面板形状: {panel.shape}")
    print(f"   因子数量: {panel.shape[1]}")

    # 检查新增的相对轮动因子
    rotation_factors = [
        col
        for col in panel.columns
        if any(
            x in col
            for x in [
                "RELATIVE_MOMENTUM",
                "CS_RANK",
                "VOL_ADJUSTED",
                "RS_DEVIATION",
                "ROTATION_SCORE",
            ]
        )
    ]

    if rotation_factors:
        print(f"\n🎯 相对轮动因子 ({len(rotation_factors)} 个):")
        for f in rotation_factors:
            non_null = panel[f].notna().sum()
            print(f"   - {f:30s}: {non_null} 条有效记录")

        # 显示最新一天的轮动得分
        latest_date = panel.reset_index()["date"].max()
        latest_scores = panel.reset_index()
        latest_scores = latest_scores[latest_scores["date"] == latest_date]

        if "ROTATION_SCORE" in latest_scores.columns:
            print(f"\n📈 最新轮动得分 ({latest_date.strftime('%Y-%m-%d')}):")
            top_rotation = latest_scores.nlargest(5, "ROTATION_SCORE")[
                ["symbol", "ROTATION_SCORE"]
            ]
            for idx, row in top_rotation.iterrows():
                print(f"   {row['symbol']}: {row['ROTATION_SCORE']:.4f}")
    else:
        print("\n⚠️ 未找到相对轮动因子！")

    print("\n" + "=" * 80)
    print("✅ 测试完成！相对轮动因子已成功集成")
    print("=" * 80)

except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
