#!/usr/bin/env python3
"""
对齐多时间框架因子筛选启动脚本
直接运行，无需命令行参数
"""
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from professional_factor_screener import ProfessionalFactorScreener
from config_manager import ConfigManager

def main():
    """主函数"""
    print("🚀 启动对齐多时间框架因子筛选...")

    # 加载配置
    config_manager = ConfigManager()
    config_file = Path("configs/complete_multi_tf_screening.yaml")

    if not config_file.exists():
        print(f"❌ 配置文件不存在: {config_file}")
        return

    print(f"📋 加载配置文件: {config_file}")

    try:
        # 加载配置
        config = config_manager.load_config(config_file, 'screening')

        # 显示配置摘要
        print(f"📊 配置摘要:")
        print(f"   股票: {', '.join(config.symbols)}")
        print(f"   时间框架: {', '.join(config.timeframes)}")
        print(f"   数据根目录: {config.data_root}")
        print(f"   输出目录: {config.output_dir}")

        # 创建筛选器
        screener = ProfessionalFactorScreener(
            data_root=config.data_root,
            config=config
        )

        # 执行多时间框架筛选
        symbol = config.symbols[0]
        timeframes = config.timeframes

        print(f"\n🎯 开始筛选: {symbol}")
        print(f"   时间框架: {', '.join(timeframes)}")

        results = screener.screen_multiple_timeframes(symbol, timeframes)

        # 显示结果摘要
        print(f"\n✅ 筛选完成!")
        print(f"   成功时间框架: {len(results)}/{len(timeframes)}")

        total_factors = sum(len(tf_results) for tf_results in results.values())
        total_top_factors = sum(
            sum(1 for m in tf_results.values() if m.comprehensive_score >= 0.8)
            for tf_results in results.values()
        )

        print(f"   总因子数: {total_factors}")
        print(f"   顶级因子数: {total_top_factors}")

        # 显示各时间框架结果
        for tf, tf_results in results.items():
            tf_top_factors = sum(1 for m in tf_results.values() if m.comprehensive_score >= 0.8)
            print(f"   {tf}: {len(tf_results)} 因子, {tf_top_factors} 顶级因子")

        print(f"\n📁 结果已保存到: {config.output_dir}")

    except Exception as e:
        print(f"❌ 筛选失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()