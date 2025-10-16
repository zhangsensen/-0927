#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生产环境横截面验证
使用真实ETF数据，验证所有因子生效情况
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from factor_system.factor_engine.factors.etf_cross_section import (
    create_etf_cross_section_manager,
    ETFCrossSectionConfig
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('production_cross_section_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_all_etf_symbols():
    """加载所有ETF代码"""
    logger.info("加载ETF代码列表...")
    
    # 从数据目录读取所有ETF文件
    data_dir = project_root / "raw" / "ETF" / "daily"
    
    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        return []
    
    etf_files = list(data_dir.glob("*.parquet"))
    # 提取真实的ETF代码（去掉_daily_日期后缀）
    symbols = []
    for f in etf_files:
        # 文件名格式: 515030.SH_daily_20200102_20251014.parquet
        # 提取: 515030.SH
        symbol = f.stem.split('_')[0]
        symbols.append(symbol)
    
    logger.info(f"✅ 找到 {len(symbols)} 只ETF")
    logger.info(f"ETF列表: {symbols[:10]}..." if len(symbols) > 10 else f"ETF列表: {symbols}")
    
    return symbols


def validate_etf_data(symbols):
    """验证ETF数据质量"""
    logger.info("\n" + "="*80)
    logger.info("数据质量验证")
    logger.info("="*80)
    
    data_dir = project_root / "raw" / "ETF" / "daily"
    
    valid_symbols = []
    data_stats = []
    
    for symbol in symbols:
        # 查找匹配的文件（文件名格式: 515030.SH_daily_20200102_20251014.parquet）
        matching_files = list(data_dir.glob(f"{symbol}_*.parquet"))
        
        if not matching_files:
            logger.warning(f"❌ {symbol}: 未找到数据文件")
            continue
        
        file_path = matching_files[0]  # 使用第一个匹配的文件
        
        try:
            df = pd.read_parquet(file_path)
            
            if df.empty:
                logger.warning(f"❌ {symbol}: 数据为空")
                continue
            
            # 检查必要列（使用实际列名）
            required_cols = ['trade_date', 'open', 'high', 'low', 'close', 'vol']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.warning(f"❌ {symbol}: 缺少列 {missing_cols}")
                continue
            
            # 统一列名
            df = df.rename(columns={'trade_date': 'date', 'vol': 'volume', 'ts_code': 'symbol'})
            
            # 统计信息
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            date_range = f"{df['date'].min().date()} ~ {df['date'].max().date()}"
            
            data_stats.append({
                'symbol': symbol,
                'records': len(df),
                'date_range': date_range,
                'latest_date': df['date'].max()
            })
            
            valid_symbols.append(symbol)
            logger.info(f"✅ {symbol}: {len(df)}条记录, {date_range}")
            
        except Exception as e:
            logger.error(f"❌ {symbol}: 读取失败 - {str(e)}")
            continue
    
    logger.info(f"\n数据验证完成: {len(valid_symbols)}/{len(symbols)} 只ETF数据有效")
    
    return valid_symbols, data_stats


def build_full_cross_section(manager, symbols, date):
    """构建完整横截面"""
    logger.info("\n" + "="*80)
    logger.info(f"构建横截面: {date.strftime('%Y-%m-%d')}")
    logger.info("="*80)
    
    # 获取所有可用因子
    available_factors = manager.get_available_factors()
    logger.info(f"可用因子总数: {len(available_factors)}")
    
    # 分类统计
    legacy_factors = [f for f in available_factors if not f.startswith('VBT_') and not f.startswith('TALIB_')]
    dynamic_factors = [f for f in available_factors if f.startswith('VBT_') or f.startswith('TALIB_')]
    
    logger.info(f"  - 传统因子: {len(legacy_factors)}")
    logger.info(f"  - 动态因子: {len(dynamic_factors)}")
    
    # 分批计算（避免一次性计算太多因子）
    batch_size = 50
    all_results = []
    
    for i in range(0, len(available_factors), batch_size):
        batch_factors = available_factors[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(available_factors) + batch_size - 1) // batch_size
        
        logger.info(f"\n批次 {batch_num}/{total_batches}: 计算 {len(batch_factors)} 个因子")
        
        try:
            result = manager.build_cross_section(
                date=date,
                symbols=symbols,
                factor_ids=batch_factors
            )
            
            if result.cross_section_df is not None and not result.cross_section_df.empty:
                all_results.append(result.cross_section_df)
                logger.info(f"✅ 批次 {batch_num} 完成: {result.cross_section_df.shape}")
            else:
                logger.warning(f"⚠️ 批次 {batch_num} 返回空结果")
                
        except Exception as e:
            logger.error(f"❌ 批次 {batch_num} 失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    # 合并所有批次结果
    if all_results:
        full_cross_section = pd.concat(all_results, axis=1)
        logger.info(f"\n✅ 横截面构建完成: {full_cross_section.shape}")
        return full_cross_section
    else:
        logger.error("❌ 所有批次均失败")
        return None


def analyze_factor_effectiveness(cross_section_df):
    """分析因子有效性"""
    logger.info("\n" + "="*80)
    logger.info("因子有效性分析")
    logger.info("="*80)
    
    if cross_section_df is None or cross_section_df.empty:
        logger.error("❌ 横截面数据为空，无法分析")
        return
    
    total_factors = len(cross_section_df.columns)
    total_symbols = len(cross_section_df.index)
    
    logger.info(f"横截面维度: {total_symbols} 只ETF × {total_factors} 个因子")
    
    # 统计每个因子的有效性
    factor_stats = []
    
    for factor_id in cross_section_df.columns:
        factor_values = cross_section_df[factor_id]
        
        # 统计信息
        valid_count = factor_values.notna().sum()
        valid_rate = valid_count / len(factor_values) * 100
        
        if valid_count > 0:
            mean_val = factor_values.mean()
            std_val = factor_values.std()
            min_val = factor_values.min()
            max_val = factor_values.max()
        else:
            mean_val = std_val = min_val = max_val = np.nan
        
        factor_stats.append({
            'factor_id': factor_id,
            'valid_count': valid_count,
            'valid_rate': valid_rate,
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val
        })
    
    stats_df = pd.DataFrame(factor_stats)
    
    # 分类统计
    effective_factors = stats_df[stats_df['valid_rate'] >= 50.0]
    partial_factors = stats_df[(stats_df['valid_rate'] > 0) & (stats_df['valid_rate'] < 50.0)]
    invalid_factors = stats_df[stats_df['valid_rate'] == 0]
    
    logger.info(f"\n因子生效情况:")
    logger.info(f"  ✅ 完全生效 (≥50%): {len(effective_factors)} 个 ({len(effective_factors)/total_factors*100:.1f}%)")
    logger.info(f"  ⚠️ 部分生效 (<50%): {len(partial_factors)} 个 ({len(partial_factors)/total_factors*100:.1f}%)")
    logger.info(f"  ❌ 未生效 (0%):    {len(invalid_factors)} 个 ({len(invalid_factors)/total_factors*100:.1f}%)")
    
    # 显示完全生效的因子
    if len(effective_factors) > 0:
        logger.info(f"\n完全生效的因子 (前20个):")
        for idx, row in effective_factors.head(20).iterrows():
            logger.info(f"  {row['factor_id']}: {row['valid_rate']:.1f}% 有效, "
                       f"均值={row['mean']:.4f}, 标准差={row['std']:.4f}")
    
    # 显示未生效的因子
    if len(invalid_factors) > 0:
        logger.info(f"\n未生效的因子 (前20个):")
        for idx, row in invalid_factors.head(20).iterrows():
            logger.info(f"  ❌ {row['factor_id']}")
    
    # 保存详细统计
    stats_file = project_root / "scripts" / "factor_effectiveness_stats.csv"
    stats_df.to_csv(stats_file, index=False)
    logger.info(f"\n详细统计已保存: {stats_file}")
    
    return stats_df


def save_cross_section_data(cross_section_df, date):
    """保存横截面数据"""
    logger.info("\n" + "="*80)
    logger.info("保存横截面数据")
    logger.info("="*80)
    
    if cross_section_df is None or cross_section_df.empty:
        logger.warning("⚠️ 横截面数据为空，跳过保存")
        return
    
    # 保存为parquet格式
    output_dir = project_root / "output" / "cross_sections"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"cross_section_{date.strftime('%Y%m%d')}.parquet"
    cross_section_df.to_parquet(output_file)
    
    logger.info(f"✅ 横截面数据已保存: {output_file}")
    logger.info(f"   维度: {cross_section_df.shape}")
    logger.info(f"   大小: {output_file.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    """主函数"""
    logger.info("="*80)
    logger.info("生产环境横截面验证")
    logger.info("="*80)
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 1. 加载ETF代码
        symbols = load_all_etf_symbols()
        if not symbols:
            logger.error("❌ 未找到ETF数据")
            return
        
        # 2. 验证数据质量
        valid_symbols, data_stats = validate_etf_data(symbols)
        if not valid_symbols:
            logger.error("❌ 没有有效的ETF数据")
            return
        
        # 3. 确定横截面日期（使用最新的共同日期）
        latest_dates = [stat['latest_date'] for stat in data_stats]
        cross_section_date = min(latest_dates)  # 使用所有ETF都有数据的最新日期
        
        logger.info(f"\n横截面日期: {cross_section_date.strftime('%Y-%m-%d')}")
        
        # 4. 初始化管理器
        logger.info("\n初始化因子管理器...")
        config = ETFCrossSectionConfig()
        config.enable_legacy_factors = True
        config.enable_dynamic_factors = True
        config.max_dynamic_factors = 200  # 使用更多因子
        
        manager = create_etf_cross_section_manager(config)
        logger.info("✅ 管理器初始化完成")
        
        # 5. 构建横截面
        cross_section_df = build_full_cross_section(
            manager=manager,
            symbols=valid_symbols,
            date=cross_section_date
        )
        
        # 6. 分析因子有效性
        if cross_section_df is not None:
            stats_df = analyze_factor_effectiveness(cross_section_df)
            
            # 7. 保存结果
            save_cross_section_data(cross_section_df, cross_section_date)
        
        logger.info("\n" + "="*80)
        logger.info("验证完成")
        logger.info("="*80)
        logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        logger.error(f"\n❌ 验证失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
