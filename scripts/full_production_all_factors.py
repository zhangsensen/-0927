#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全面生产环境 - 所有ETF × 所有因子
使用5年完整历史数据
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from factor_system.factor_engine import api
from factor_system.factor_engine.factors.etf_cross_section import (
    create_etf_cross_section_manager,
    ETFCrossSectionConfig
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('full_production_all_factors.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_all_etf_symbols():
    """加载所有ETF代码"""
    logger.info("="*80)
    logger.info("加载ETF代码列表")
    logger.info("="*80)
    
    data_dir = project_root / "raw" / "ETF" / "daily"
    etf_files = list(data_dir.glob("*.parquet"))
    
    symbols = []
    for f in etf_files:
        symbol = f.stem.split('_')[0]
        symbols.append(symbol)
    
    symbols = sorted(list(set(symbols)))
    logger.info(f"✅ 找到 {len(symbols)} 只ETF")
    logger.info(f"ETF列表: {', '.join(symbols[:10])}...")
    
    return symbols


def get_data_date_range(symbols):
    """获取数据的日期范围"""
    logger.info("\n" + "="*80)
    logger.info("分析数据日期范围")
    logger.info("="*80)
    
    data_dir = project_root / "raw" / "ETF" / "daily"
    
    all_start_dates = []
    all_end_dates = []
    
    for symbol in symbols[:5]:  # 采样5只ETF
        files = list(data_dir.glob(f"{symbol}_*.parquet"))
        if files:
            df = pd.read_parquet(files[0])
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            all_start_dates.append(df['trade_date'].min())
            all_end_dates.append(df['trade_date'].max())
    
    if all_start_dates and all_end_dates:
        # 使用所有ETF都有数据的日期范围
        common_start = max(all_start_dates)
        common_end = min(all_end_dates)
        
        logger.info(f"数据日期范围:")
        logger.info(f"  最早日期: {common_start.date()}")
        logger.info(f"  最新日期: {common_end.date()}")
        logger.info(f"  总天数: {(common_end - common_start).days} 天")
        logger.info(f"  约 {(common_end - common_start).days / 365:.1f} 年")
        
        return common_start, common_end
    
    # 默认值
    return datetime(2020, 1, 1), datetime(2025, 10, 14)


def get_all_available_factors(manager):
    """获取所有可用因子"""
    logger.info("\n" + "="*80)
    logger.info("获取所有可用因子")
    logger.info("="*80)

    # 🔥 关键修复：先强制注册所有动态因子
    logger.info("🔧 注册动态因子...")
    manager._register_all_dynamic_factors()

    available_factors = manager.get_available_factors()

    # 分类统计
    legacy_factors = [f for f in available_factors if not f.startswith('VBT_') and not f.startswith('TALIB_') and not f.startswith('TA_')]
    vbt_factors = [f for f in available_factors if f.startswith('VBT_')]
    talib_factors = [f for f in available_factors if f.startswith('TALIB_') or f.startswith('TA_')]

    logger.info(f"因子总数: {len(available_factors)}")
    logger.info(f"  - 传统因子: {len(legacy_factors)}")
    logger.info(f"  - VBT因子: {len(vbt_factors)}")
    logger.info(f"  - TA-Lib因子: {len(talib_factors)}")

    return available_factors, legacy_factors, vbt_factors, talib_factors


def calculate_factors_in_batches(symbols, factor_ids, start_date, end_date, batch_size=20):
    """分批计算因子"""
    logger.info("\n" + "="*80)
    logger.info(f"开始批量计算因子")
    logger.info("="*80)
    logger.info(f"ETF数量: {len(symbols)}")
    logger.info(f"因子数量: {len(factor_ids)}")
    logger.info(f"批次大小: {batch_size}")
    logger.info(f"时间范围: {start_date.date()} ~ {end_date.date()}")
    
    total_batches = (len(factor_ids) + batch_size - 1) // batch_size
    all_results = []
    successful_factors = []
    failed_factors = []
    
    start_time = time.time()
    
    for batch_idx in range(total_batches):
        batch_start_idx = batch_idx * batch_size
        batch_end_idx = min(batch_start_idx + batch_size, len(factor_ids))
        batch_factors = factor_ids[batch_start_idx:batch_end_idx]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"批次 {batch_idx + 1}/{total_batches}: 计算 {len(batch_factors)} 个因子")
        logger.info(f"{'='*60}")
        
        batch_results = []
        
        for i, factor_id in enumerate(batch_factors, 1):
            try:
                factor_start = time.time()
                
                result = api.calculate_factors(
                    factor_ids=[factor_id],
                    symbols=symbols,
                    timeframe='daily',
                    start_date=start_date,
                    end_date=end_date
                )
                
                factor_time = time.time() - factor_start
                
                if result is not None and not result.empty:
                    batch_results.append(result)
                    successful_factors.append(factor_id)
                    logger.info(f"  [{batch_idx*batch_size + i}/{len(factor_ids)}] ✅ {factor_id}: "
                               f"{result.shape} ({factor_time:.2f}s)")
                else:
                    failed_factors.append((factor_id, "空结果"))
                    logger.warning(f"  [{batch_idx*batch_size + i}/{len(factor_ids)}] ⚠️ {factor_id}: 空结果")
                    
            except Exception as e:
                failed_factors.append((factor_id, str(e)))
                logger.error(f"  [{batch_idx*batch_size + i}/{len(factor_ids)}] ❌ {factor_id}: {str(e)[:100]}")
                continue
        
        # 合并批次结果
        if batch_results:
            batch_combined = pd.concat(batch_results, axis=1)
            all_results.append(batch_combined)
            logger.info(f"✅ 批次 {batch_idx + 1} 完成: {len(batch_results)}/{len(batch_factors)} 个因子成功")
        else:
            logger.warning(f"⚠️ 批次 {batch_idx + 1} 无有效结果")
    
    total_time = time.time() - start_time
    
    # 合并所有结果
    if all_results:
        combined_df = pd.concat(all_results, axis=1)
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ 因子计算完成")
        logger.info(f"{'='*80}")
        logger.info(f"成功: {len(successful_factors)}/{len(factor_ids)} ({len(successful_factors)/len(factor_ids)*100:.1f}%)")
        logger.info(f"失败: {len(failed_factors)}/{len(factor_ids)} ({len(failed_factors)/len(factor_ids)*100:.1f}%)")
        logger.info(f"总耗时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
        logger.info(f"平均每因子: {total_time/len(factor_ids):.2f}秒")
        logger.info(f"结果维度: {combined_df.shape}")
        
        return combined_df, successful_factors, failed_factors
    
    logger.error("❌ 所有批次均失败")
    return None, successful_factors, failed_factors


def build_cross_sections_for_dates(combined_df, dates):
    """为多个日期构建横截面"""
    logger.info("\n" + "="*80)
    logger.info(f"构建多日期横截面")
    logger.info("="*80)
    logger.info(f"日期数量: {len(dates)}")
    
    cross_sections = {}
    
    for date in dates:
        try:
            if isinstance(combined_df.index, pd.MultiIndex):
                try:
                    cross_section = combined_df.xs(date, level=0)
                except KeyError:
                    # 查找最近日期
                    available_dates = combined_df.index.get_level_values(0).unique()
                    closest_date = min(available_dates, key=lambda d: abs((d - date).total_seconds()))
                    cross_section = combined_df.xs(closest_date, level=0)
                    logger.info(f"  {date.date()}: 使用最近日期 {closest_date.date()}")
            else:
                cross_section = combined_df
            
            cross_sections[date] = cross_section
            logger.info(f"  ✅ {date.date()}: {cross_section.shape}")
            
        except Exception as e:
            logger.error(f"  ❌ {date.date()}: {str(e)}")
            continue
    
    return cross_sections


def analyze_factor_effectiveness(cross_section, date):
    """分析因子有效性"""
    total_etfs = len(cross_section)
    total_factors = len(cross_section.columns)
    
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
            'std': values.std() if valid_count > 0 else np.nan,
            'min': values.min() if valid_count > 0 else np.nan,
            'max': values.max() if valid_count > 0 else np.nan
        })
    
    stats_df = pd.DataFrame(factor_stats)
    
    # 分类
    effective = stats_df[stats_df['valid_rate'] >= 80]
    partial = stats_df[(stats_df['valid_rate'] >= 50) & (stats_df['valid_rate'] < 80)]
    weak = stats_df[(stats_df['valid_rate'] > 0) & (stats_df['valid_rate'] < 50)]
    invalid = stats_df[stats_df['valid_rate'] == 0]
    
    logger.info(f"\n{date.date()} 因子有效性:")
    logger.info(f"  ✅ 优秀 (≥80%): {len(effective)} 个 ({len(effective)/total_factors*100:.1f}%)")
    logger.info(f"  🟡 良好 (50-80%): {len(partial)} 个 ({len(partial)/total_factors*100:.1f}%)")
    logger.info(f"  ⚠️ 较弱 (<50%): {len(weak)} 个 ({len(weak)/total_factors*100:.1f}%)")
    logger.info(f"  ❌ 无效 (0%): {len(invalid)} 个 ({len(invalid)/total_factors*100:.1f}%)")
    
    return stats_df


def save_results(cross_sections, stats_dfs, successful_factors, failed_factors):
    """保存所有结果"""
    logger.info("\n" + "="*80)
    logger.info("保存结果")
    logger.info("="*80)
    
    output_dir = project_root / "output" / "full_production"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 保存横截面数据
    for date, cross_section in cross_sections.items():
        filename = f"cross_section_{date.strftime('%Y%m%d')}.parquet"
        filepath = output_dir / filename
        cross_section.to_parquet(filepath)
        logger.info(f"  ✅ {filename}: {cross_section.shape}, {filepath.stat().st_size/1024:.1f} KB")
    
    # 2. 保存因子统计
    for date, stats_df in stats_dfs.items():
        filename = f"factor_stats_{date.strftime('%Y%m%d')}.csv"
        filepath = output_dir / filename
        stats_df.to_csv(filepath, index=False)
        logger.info(f"  ✅ {filename}: {len(stats_df)} 个因子")
    
    # 3. 保存因子列表
    factor_list_file = output_dir / "factor_list.txt"
    with open(factor_list_file, 'w') as f:
        f.write("# 成功计算的因子\n")
        for factor in successful_factors:
            f.write(f"{factor}\n")
        f.write(f"\n# 失败的因子 ({len(failed_factors)}个)\n")
        for factor, reason in failed_factors:
            f.write(f"{factor}: {reason}\n")
    
    logger.info(f"  ✅ factor_list.txt: {len(successful_factors)} 成功, {len(failed_factors)} 失败")
    
    # 4. 生成汇总报告
    summary_file = output_dir / "summary_report.txt"
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("全面生产环境 - 汇总报告\n")
        f.write("="*80 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"横截面数量: {len(cross_sections)}\n")
        f.write(f"成功因子数: {len(successful_factors)}\n")
        f.write(f"失败因子数: {len(failed_factors)}\n")
        f.write(f"因子成功率: {len(successful_factors)/(len(successful_factors)+len(failed_factors))*100:.1f}%\n\n")
        
        for date, cross_section in cross_sections.items():
            f.write(f"\n{date.date()}:\n")
            f.write(f"  ETF数量: {len(cross_section)}\n")
            f.write(f"  因子数量: {len(cross_section.columns)}\n")
    
    logger.info(f"  ✅ summary_report.txt")
    logger.info(f"\n所有结果已保存到: {output_dir}")


def main():
    """主函数"""
    logger.info("="*80)
    logger.info("全面生产环境 - 所有ETF × 所有因子")
    logger.info("="*80)
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 1. 加载ETF列表
        symbols = load_all_etf_symbols()
        
        # 2. 获取数据日期范围
        start_date, end_date = get_data_date_range(symbols)
        
        # 使用最近1年的数据（更快，且足够支持大部分因子）
        start_date = end_date - timedelta(days=365)
        logger.info(f"\n使用数据范围: {start_date.date()} ~ {end_date.date()} (约1年)")
        
        # 3. 初始化管理器并获取所有因子
        logger.info("\n初始化因子管理器...")
        config = ETFCrossSectionConfig()
        config.enable_legacy_factors = True
        config.enable_dynamic_factors = True
        config.max_dynamic_factors = 500  # 不限制，使用所有因子
        
        manager = create_etf_cross_section_manager(config)
        available_factors, legacy_factors, vbt_factors, talib_factors = get_all_available_factors(manager)
        
        # 4. 批量计算所有因子
        combined_df, successful_factors, failed_factors = calculate_factors_in_batches(
            symbols=symbols,
            factor_ids=available_factors,
            start_date=start_date,
            end_date=end_date,
            batch_size=30  # 每批30个因子
        )
        
        if combined_df is None:
            logger.error("❌ 因子计算失败")
            return
        
        # 5. 构建多个日期的横截面（最近5个交易日）
        target_dates = [
            end_date,
            end_date - timedelta(days=1),
            end_date - timedelta(days=2),
            end_date - timedelta(days=7),
            end_date - timedelta(days=30)
        ]
        
        cross_sections = build_cross_sections_for_dates(combined_df, target_dates)
        
        # 6. 分析每个横截面的因子有效性
        stats_dfs = {}
        for date, cross_section in cross_sections.items():
            stats_df = analyze_factor_effectiveness(cross_section, date)
            stats_dfs[date] = stats_df
        
        # 7. 保存所有结果
        save_results(cross_sections, stats_dfs, successful_factors, failed_factors)
        
        # 8. 最终总结
        logger.info("\n" + "="*80)
        logger.info("✅ 全面生产完成！")
        logger.info("="*80)
        logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"ETF数量: {len(symbols)}")
        logger.info(f"因子总数: {len(available_factors)}")
        logger.info(f"成功因子: {len(successful_factors)} ({len(successful_factors)/len(available_factors)*100:.1f}%)")
        logger.info(f"横截面数: {len(cross_sections)}")
        
    except Exception as e:
        logger.error(f"\n❌ 执行失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
