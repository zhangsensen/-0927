"""
分钟级数据功能演示

展示：
1. 分钟数据加载
2. 尾盘抢筹比率计算
3. 四段小时K生成
4. 日内因子计算
5. 配置管理使用
"""

from pathlib import Path

import pandas as pd

from factor_system.config.factor_config import FactorConfig
from factor_system.factor_engine.factors.money_flow.constraints import (
    calculate_tail30_ratio,
    generate_hourly_4seg,
)
from factor_system.factor_engine.factors.money_flow.intraday import (
    IntradayMomentum,
    IntradayVolumeSurge,
)
from factor_system.factor_engine.providers.minute_data_provider import (
    MinuteDataProvider,
)
from factor_system.utils.data_continuity import DataContinuityChecker


def main():
    """主函数"""
    print("=" * 70)
    print("分钟级数据功能演示")
    print("=" * 70)

    # 1. 加载配置
    print("\n1. 加载因子配置...")
    config = FactorConfig()
    tail30_config = config.get_factor_params("tail30_ratio", "constraints")
    print(f"   尾盘抢筹配置: {tail30_config}")

    # 2. 创建分钟数据提供者
    print("\n2. 创建分钟数据提供者...")
    data_dir = Path("raw/SH")
    provider = MinuteDataProvider(data_dir=data_dir)

    try:
        # 3. 加载分钟数据
        print("\n3. 加载分钟数据...")
        symbol = "600036.SH"
        start_date = "2024-08-23"
        end_date = "2024-09-23"

        minute_data = provider.load_date_range(symbol, start_date, end_date)
        print(f"   ✓ 加载成功: {len(minute_data)}条分钟数据")
        print(f"   ✓ 日期范围: {minute_data.index.min()} ~ {minute_data.index.max()}")

        # 4. 计算尾盘抢筹比率
        print("\n4. 计算尾盘抢筹比率...")
        tail30_ratio = calculate_tail30_ratio(
            minute_data,
            include_auction=tail30_config.get("include_auction", False),
            zscore_window=tail30_config.get("zscore_window", 60),
        )
        print(f"   ✓ 计算完成: {len(tail30_ratio)}个交易日")
        print(f"   ✓ 平均比率: {tail30_ratio.mean():.4f}")
        print(f"   ✓ 标准差: {tail30_ratio.std():.4f}")
        print(f"\n   最近5日尾盘抢筹比率:")
        print(tail30_ratio.tail().to_string())

        # 5. 生成四段小时K
        print("\n5. 生成四段小时K...")
        hourly_4seg = generate_hourly_4seg(minute_data)
        print(f"   ✓ 生成完成: {len(hourly_4seg)}个时段")

        # 统计每日段数
        daily_seg_counts = hourly_4seg.groupby(hourly_4seg["date"]).size()
        print(f"   ✓ 交易日数: {len(daily_seg_counts)}")
        print(f"   ✓ 每日段数: {daily_seg_counts.unique()}")

        # 显示最近一日的四段数据
        if not hourly_4seg.empty:
            last_date = hourly_4seg["date"].max()
            last_day_segs = hourly_4seg[hourly_4seg["date"] == last_date]
            print(f"\n   {last_date.date()} 四段小时K:")
            print(
                last_day_segs[
                    ["segment_name", "open", "high", "low", "close", "volume"]
                ].to_string()
            )

        # 6. 计算日内因子
        print("\n6. 计算日内因子...")

        # 成交量爆发
        volume_surge = IntradayVolumeSurge(lookback_days=20)
        surge_values = volume_surge.calculate(minute_data)
        print(f"   ✓ 成交量爆发因子: {len(surge_values)}个值")
        print(f"     平均爆发倍数: {surge_values.mean():.2f}x")
        print(f"     最大爆发倍数: {surge_values.max():.2f}x")

        # 日内动量
        intraday_momentum = IntradayMomentum()
        momentum_values = intraday_momentum.calculate(minute_data)
        print(f"   ✓ 日内动量因子: {len(momentum_values)}个值")
        print(f"     平均动量: {momentum_values.mean():.4f}")
        print(f"     动量标准差: {momentum_values.std():.4f}")

        # 7. 数据质量检查
        print("\n7. 数据质量检查...")
        checker = DataContinuityChecker()

        # 聚合为日线进行连续性检查
        daily_data = provider.aggregate_to_daily(minute_data)
        continuity_result = checker.validate_gap_calculation(daily_data["close"])

        print(f"   ✓ 数据连续性: {continuity_result['is_continuous']}")
        print(f"   ✓ 总交易日: {continuity_result['total_days']}")
        print(f"   ✓ 连续天数: {continuity_result['continuous_days']}")
        print(f"   ✓ 连续性比率: {continuity_result['continuity_ratio']:.2%}")

        if not continuity_result["is_continuous"]:
            print(f"   ⚠ {continuity_result.get('warning', 'Data has gaps')}")

        # 8. 保存结果
        print("\n8. 保存结果...")
        output_dir = Path("output/minute_data_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存尾盘抢筹比率
        tail30_ratio.to_csv(output_dir / f"{symbol}_tail30_ratio.csv")
        print(f"   ✓ 尾盘抢筹比率已保存")

        # 保存四段小时K
        hourly_4seg.to_csv(output_dir / f"{symbol}_hourly_4seg.csv")
        print(f"   ✓ 四段小时K已保存")

        # 保存日内因子
        intraday_factors = pd.DataFrame(
            {"volume_surge": surge_values, "intraday_momentum": momentum_values}
        )
        intraday_factors.to_csv(output_dir / f"{symbol}_intraday_factors.csv")
        print(f"   ✓ 日内因子已保存")

        print("\n" + "=" * 70)
        print("✅ 分钟级数据功能演示完成！")
        print("=" * 70)

    except FileNotFoundError as e:
        print(f"\n❌ 错误: 数据文件未找到")
        print(f"   请确保分钟数据存在于: {data_dir}")
        print(f"   错误详情: {e}")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
