#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生产环境完整横截面构建
使用真实ETF数据，计算所有因子
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from factor_system.factor_engine.factors.etf_cross_section import (
    create_etf_cross_section_manager,
    ETFCrossSectionConfig
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('production_full_cross_section.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_etf_symbols():
    """加载所有有效的ETF代码"""
    logger.info("加载ETF代码...")
    
    data_dir = project_root / "raw" / "ETF" / "daily"
    etf_files = list(data_dir.glob("*.parquet"))
    
    symbols = []
    for f in etf_files:
        # 提取ETF代码: 515030.SH_daily_20200102_20251014.parquet -> 515030.SH
        symbol = f.stem.split('_')[0]
        symbols.append(symbol)
    
    logger.info(f"✅ 找到 {len(symbols)} 只ETF")
    return sorted(list(set(symbols)))  # 去重并排序


def get_latest_common_date(symbols):
    """获取所有ETF都有数据的最新日期"""
    logger.info("查找最新共同日期...")
    
    data_dir = project_root / "raw" / "ETF" / "daily"
    latest_dates = []
    
    for symbol in symbols[:10]:  # 只检查前10只，加快速度
        files = list(data_dir.glob(f"{symbol}_*.parquet"))
        if files:
            df = pd.read_parquet(files[0])
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            latest_dates.append(df['trade_date'].max())
    
    if latest_dates:
        common_date = min(latest_dates)
        logger.info(f"✅ 最新共同日期: {common_date.date()}")
        return common_date
    
    return datetime(2025, 10, 14)


def calculate_factors_batch(manager, symbols, factor_ids, start_date, end_date):
    """批量计算因子 - 使用manager统一接口"""
    logger.info(f"计算因子: {len(symbols)}只ETF × {len(factor_ids)}个因子")
    
    try:
        # 🔥 关键修复：使用manager.calculate_factors统一接口
        result = manager.calculate_factors(
            symbols=symbols,
            timeframe='daily',
            start_date=start_date,
            end_date=end_date,
            factor_ids=factor_ids  # 一次性计算所有因子
        )
        
        if result.factors_df is not None and not result.factors_df.empty:
            logger.info(f"✅ 因子计算完成: {result.factors_df.shape}")
            logger.info(f"   成功: {len(result.successful_factors)} 个")
            logger.info(f"   失败: {len(result.failed_factors)} 个")
            return result.factors_df
        else:
            logger.error("❌ 因子计算返回空结果")
            return None
            
    except Exception as e:
        logger.error(f"❌ 因子计算失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def build_cross_section(combined_df, target_date):
    """从时间序列数据构建横截面"""
    logger.info(f"构建横截面: {target_date.date()}")
    
    if combined_df is None or combined_df.empty:
        logger.error("❌ 输入数据为空")
        return None
    
    # 提取指定日期的数据
    if isinstance(combined_df.index, pd.MultiIndex):
        try:
            # 尝试精确匹配
            cross_section = combined_df.xs(target_date, level=0)
        except KeyError:
            # 查找最近日期
            dates = combined_df.index.get_level_values(0).unique()
            closest_date = min(dates, key=lambda d: abs((d - target_date).total_seconds()))
            cross_section = combined_df.xs(closest_date, level=0)
            logger.warning(f"使用最近日期: {closest_date.date()}")
    else:
        cross_section = combined_df
    
    logger.info(f"✅ 横截面构建完成: {cross_section.shape}")
    return cross_section


def analyze_factors(cross_section):
    """分析因子有效性"""
    logger.info("\n" + "="*80)
    logger.info("因子有效性分析")
    logger.info("="*80)
    
    total_etfs = len(cross_section)
    total_factors = len(cross_section.columns)
    
    logger.info(f"横截面维度: {total_etfs} 只ETF × {total_factors} 个因子")
    
    # 统计每个因子
    factor_stats = []
    for factor_id in cross_section.columns:
        values = cross_section[factor_id]
        valid_count = values.notna().sum()
        valid_rate = valid_count / len(values) * 100
        
        factor_stats.append({
            'factor_id': factor_id,
            'valid_count': valid_count,
            'valid_rate': valid_rate,
            'mean': values.mean() if valid_count > 0 else np.nan,
            'std': values.std() if valid_count > 0 else np.nan
        })
    
    stats_df = pd.DataFrame(factor_stats)
    
    # 分类
    effective = stats_df[stats_df['valid_rate'] >= 50]
    partial = stats_df[(stats_df['valid_rate'] > 0) & (stats_df['valid_rate'] < 50)]
    invalid = stats_df[stats_df['valid_rate'] == 0]
    
    logger.info("\n因子生效情况:")
    logger.info(f"  ✅ 完全生效 (≥50%): {len(effective)} 个 ({len(effective)/total_factors*100:.1f}%)")
    logger.info(f"  ⚠️ 部分生效 (<50%): {len(partial)} 个 ({len(partial)/total_factors*100:.1f}%)")
    logger.info(f"  ❌ 未生效 (0%):    {len(invalid)} 个 ({len(invalid)/total_factors*100:.1f}%)")
    
    # 显示生效因子
    if len(effective) > 0:
        logger.info("\n完全生效的因子 (前30个):")
        for _, row in effective.head(30).iterrows():
            logger.info(f"  {row['factor_id']}: {row['valid_rate']:.1f}% ({row['valid_count']}/{total_etfs})")
    
    # 保存统计
    output_dir = project_root / "output" / "cross_sections"
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_file = output_dir / "factor_effectiveness_stats.csv"
    stats_df.to_csv(stats_file, index=False)
    logger.info(f"\n✅ 详细统计已保存: {stats_file}")
    
    return stats_df


def main():
    """主函数"""
    logger.info("="*80)
    logger.info("生产环境完整横截面构建")
    logger.info("="*80)
    
    try:
        # 1. 加载ETF列表
        symbols = load_etf_symbols()
        logger.info(f"ETF列表: {symbols[:10]}...")
        
        # 2. 确定日期 - 使用完整数据窗口
        target_date = get_latest_common_date(symbols)
        # 🔥 修复：使用250个交易日（约1年）窗口，足够计算MA120等长周期指标
        start_date = target_date - timedelta(days=365)  # 1年数据
        
        logger.info(f"时间范围: {start_date.date()} ~ {target_date.date()}")
        logger.info(f"数据窗口: ~250个交易日，支持MA120等长周期指标")
        
        # 3. 获取可用因子列表
        logger.info("\n初始化因子管理器...")
        config = ETFCrossSectionConfig()
        config.enable_dynamic_factors = True
        config.max_dynamic_factors = 1000  # 移除因子数量限制

        manager = create_etf_cross_section_manager(config)

        # 强制注册所有动态因子
        manager._register_all_dynamic_factors()

        available_factors = manager.get_available_factors()

        logger.info(f"✅ 可用因子: {len(available_factors)} 个")
        logger.info(f"   动态因子: {len(manager.factor_registry.list_factors(is_dynamic=True))} 个")
        logger.info(f"   传统因子: {len(available_factors) - len(manager.factor_registry.list_factors(is_dynamic=True))} 个")

        # 4. 批量计算因子
        logger.info("\n" + "="*80)
        logger.info("开始计算因子")
        logger.info("="*80)

        combined_df = calculate_factors_batch(
            manager=manager,  # 🔥 传入manager
            symbols=symbols,
            factor_ids=available_factors,  # 计算所有因子！
            start_date=start_date,
            end_date=target_date
        )
        
        if combined_df is None:
            logger.error("❌ 因子计算失败")
            return
        
        # 5. 构建横截面
        cross_section = build_cross_section(combined_df, target_date)
        
        if cross_section is None:
            logger.error("❌ 横截面构建失败")
            return
        
        # 6. 分析因子有效性
        analyze_factors(cross_section)
        
        # 7. 保存横截面数据
        output_dir = project_root / "output" / "cross_sections"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"cross_section_{target_date.strftime('%Y%m%d')}.parquet"
        cross_section.to_parquet(output_file)
        
        logger.info(f"\n✅ 横截面数据已保存: {output_file}")
        logger.info(f"   大小: {output_file.stat().st_size / 1024:.2f} KB")
        
        logger.info("\n" + "="*80)
        logger.info("✅ 全部完成！")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"\n❌ 执行失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
