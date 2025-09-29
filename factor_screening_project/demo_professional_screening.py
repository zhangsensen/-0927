#!/usr/bin/env python3
"""
专业级因子筛选系统 - 实际使用演示
作者：量化首席工程师
版本：2.0.0
日期：2025-09-29

本演示展示如何使用专业级因子筛选系统进行实际的因子筛选工作
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from professional_factor_screener import (
    ProfessionalFactorScreener, 
    ScreeningConfig, 
    FactorMetrics
)

def demo_professional_screening():
    """演示专业级因子筛选"""
    print("="*100)
    print("专业级量化交易因子筛选系统 - 实际使用演示")
    print("="*100)
    print("版本: 2.0.0")
    print("特性: 5维度筛选框架 + 多重比较校正 + 交易成本评估")
    print("="*100)
    
    # 1. 配置系统
    print("\n📋 步骤1: 配置筛选系统")
    print("-" * 50)
    
    config = ScreeningConfig(
        # 多周期IC配置
        ic_horizons=[1, 3, 5, 10, 20],          # 1日到20日的预测周期
        min_sample_size=100,                     # 最小样本量要求
        rolling_window=60,                       # 滚动IC窗口
        
        # 统计显著性配置
        alpha_level=0.05,                        # 5%显著性水平
        fdr_method="benjamini_hochberg",         # BH-FDR校正
        
        # 独立性分析配置
        vif_threshold=5.0,                       # VIF阈值
        correlation_threshold=0.8,               # 相关性阈值
        base_factors=["MA5", "MA10", "RSI14"],   # 基准因子
        
        # 交易成本配置
        commission_rate=0.002,                   # 0.2%佣金
        slippage_bps=5.0,                       # 5bp滑点
        market_impact_coeff=0.1,                # 市场冲击系数
        
        # 筛选阈值配置
        min_ic_threshold=0.02,                   # 最小IC阈值
        min_ir_threshold=0.5,                    # 最小IR阈值
        min_stability_threshold=0.6,             # 最小稳定性阈值
        
        # 评分权重配置 (5维度)
        weight_predictive=0.35,                  # 预测能力权重
        weight_stability=0.25,                   # 稳定性权重
        weight_independence=0.20,                # 独立性权重
        weight_practicality=0.15,                # 实用性权重
        weight_adaptability=0.05,                # 短周期适应性权重
        
        # 性能配置
        max_workers=4,                           # 并行工作线程
        cache_enabled=True,                      # 启用缓存
        memory_limit_mb=2048                     # 内存限制
    )
    
    print(f"✅ 配置完成:")
    print(f"   - IC分析周期: {config.ic_horizons}")
    print(f"   - 显著性水平: {config.alpha_level} ({config.fdr_method}校正)")
    print(f"   - 交易成本: {config.commission_rate*100:.1f}%佣金 + {config.slippage_bps}bp滑点")
    print(f"   - 5维度权重: 预测{config.weight_predictive:.0%} + 稳定{config.weight_stability:.0%} + 独立{config.weight_independence:.0%} + 实用{config.weight_practicality:.0%} + 适应{config.weight_adaptability:.0%}")
    
    # 2. 初始化筛选器
    print(f"\n🔧 步骤2: 初始化筛选器")
    print("-" * 50)
    
    data_root = "/Users/zhangshenshen/深度量化0927/factor_system/output"
    screener = ProfessionalFactorScreener(data_root, config=config)
    
    print(f"✅ 筛选器初始化完成")
    print(f"   - 数据根目录: {data_root}")
    print(f"   - 缓存目录: {screener.cache_dir}")
    
    # 3. 执行因子筛选
    print(f"\n🎯 步骤3: 执行5维度因子筛选")
    print("-" * 50)
    
    symbol = "0700.HK"
    timeframes = ["5min", "15min", "30min", "60min", "daily"]
    
    print(f"目标股票: {symbol}")
    print(f"分析时间框架: {timeframes}")
    print()
    
    all_results = {}
    
    for timeframe in timeframes:
        print(f"🔍 分析时间框架: {timeframe}")
        
        try:
            # 执行5维度综合筛选
            results = screener.screen_factors_comprehensive(symbol, timeframe)
            all_results[timeframe] = results
            
            # 统计结果
            total_factors = len(results)
            significant_factors = sum(1 for m in results.values() if m.is_significant)
            high_score_factors = sum(1 for m in results.values() if m.comprehensive_score > 0.7)
            
            print(f"   ✅ 筛选完成: 总因子={total_factors}, 显著={significant_factors}, 高分={high_score_factors}")
            
            # 获取顶级因子
            top_factors = screener.get_top_factors(
                results, top_n=5, min_score=0.6, require_significant=False
            )
            
            if top_factors:
                print(f"   🏆 顶级因子 (前3名):")
                for i, metrics in enumerate(top_factors[:3]):
                    print(f"      {i+1}. {metrics.name}: 综合得分={metrics.comprehensive_score:.3f}")
            else:
                print(f"   ⚠️  未找到高质量因子")
            
        except FileNotFoundError:
            print(f"   ❌ 数据文件不存在，跳过 {timeframe}")
            continue
        except Exception as e:
            print(f"   ❌ 筛选失败: {str(e)}")
            continue
        
        print()
    
    # 4. 生成综合分析报告
    print(f"📊 步骤4: 生成综合分析报告")
    print("-" * 50)
    
    if all_results:
        # 选择最佳时间框架进行详细分析
        best_timeframe = None
        best_score = 0
        
        for timeframe, results in all_results.items():
            if results:
                avg_score = np.mean([m.comprehensive_score for m in results.values()])
                if avg_score > best_score:
                    best_score = avg_score
                    best_timeframe = timeframe
        
        if best_timeframe:
            print(f"🎯 最佳时间框架: {best_timeframe} (平均得分: {best_score:.3f})")
            
            best_results = all_results[best_timeframe]
            
            # 生成详细报告
            report_df = screener.generate_screening_report(best_results)
            
            print(f"✅ 详细报告已生成: {len(report_df)} 个因子")
            
            # 获取各层级因子
            tier1_factors = [m for m in best_results.values() if m.comprehensive_score >= 0.8]
            tier2_factors = [m for m in best_results.values() if 0.6 <= m.comprehensive_score < 0.8]
            tier3_factors = [m for m in best_results.values() if 0.4 <= m.comprehensive_score < 0.6]
            
            print(f"\n📈 因子分级结果:")
            print(f"   🥇 Tier 1 (≥0.8): {len(tier1_factors)} 个因子")
            print(f"   🥈 Tier 2 (0.6-0.8): {len(tier2_factors)} 个因子")
            print(f"   🥉 Tier 3 (0.4-0.6): {len(tier3_factors)} 个因子")
            
            # 显示顶级因子详细信息
            top_factors = screener.get_top_factors(
                best_results, top_n=10, min_score=0.5, require_significant=False
            )
            
            if top_factors:
                print(f"\n🏆 顶级因子详细分析 (前10名):")
                print("="*120)
                print(f"{'排名':<4} {'因子名称':<20} {'综合':<6} {'预测':<6} {'稳定':<6} {'独立':<6} {'实用':<6} {'适应':<6} {'IC均值':<8} {'IR':<6} {'显著':<6}")
                print("="*120)
                
                for i, metrics in enumerate(top_factors):
                    significance = "***" if metrics.corrected_p_value < 0.001 else \
                                 "**" if metrics.corrected_p_value < 0.01 else \
                                 "*" if metrics.corrected_p_value < 0.05 else ""
                    
                    print(f"{i+1:<4} {metrics.name:<20} {metrics.comprehensive_score:.3f}  "
                          f"{metrics.predictive_score:.3f}  {metrics.stability_score:.3f}  "
                          f"{metrics.independence_score:.3f}  {metrics.practicality_score:.3f}  "
                          f"{metrics.adaptability_score:.3f}  {metrics.ic_mean:+.4f}   "
                          f"{metrics.ic_ir:.3f}  {significance:<6}")
                
                print("="*120)
                print("显著性标记: *** p<0.001, ** p<0.01, * p<0.05")
                
                # 5维度分析摘要
                print(f"\n📊 5维度分析摘要:")
                print("-" * 60)
                
                avg_predictive = np.mean([m.predictive_score for m in top_factors])
                avg_stability = np.mean([m.stability_score for m in top_factors])
                avg_independence = np.mean([m.independence_score for m in top_factors])
                avg_practicality = np.mean([m.practicality_score for m in top_factors])
                avg_adaptability = np.mean([m.adaptability_score for m in top_factors])
                
                print(f"1. 预测能力: {avg_predictive:.3f} {'🟢优秀' if avg_predictive > 0.7 else '🟡良好' if avg_predictive > 0.5 else '🔴需改进'}")
                print(f"2. 稳定性:   {avg_stability:.3f} {'🟢优秀' if avg_stability > 0.7 else '🟡良好' if avg_stability > 0.5 else '🔴需改进'}")
                print(f"3. 独立性:   {avg_independence:.3f} {'🟢优秀' if avg_independence > 0.7 else '🟡良好' if avg_independence > 0.5 else '🔴需改进'}")
                print(f"4. 实用性:   {avg_practicality:.3f} {'🟢优秀' if avg_practicality > 0.7 else '🟡良好' if avg_practicality > 0.5 else '🔴需改进'}")
                print(f"5. 适应性:   {avg_adaptability:.3f} {'🟢优秀' if avg_adaptability > 0.7 else '🟡良好' if avg_adaptability > 0.5 else '🔴需改进'}")
                
                # 投资建议
                print(f"\n💡 投资建议:")
                print("-" * 60)
                
                if len(tier1_factors) >= 3:
                    print("🎯 建议策略: 多因子组合策略")
                    print("   - 使用Tier 1因子构建核心组合")
                    print("   - Tier 2因子作为辅助信号")
                    print("   - 建议权重: 等权重或IC加权")
                elif len(tier2_factors) >= 5:
                    print("🎯 建议策略: 精选因子策略")
                    print("   - 使用Tier 2因子构建组合")
                    print("   - 加强风险管理和仓位控制")
                    print("   - 建议权重: IC加权或风险平价")
                else:
                    print("⚠️  建议策略: 谨慎观望")
                    print("   - 当前因子质量不足以支撑稳定策略")
                    print("   - 建议扩大因子库或优化计算方法")
                    print("   - 考虑使用更长的历史数据")
                
                # 风险提示
                print(f"\n⚠️  风险提示:")
                print("-" * 60)
                print("1. 因子有效性可能随市场环境变化")
                print("2. 建议定期重新筛选和验证因子")
                print("3. 实际交易中需考虑流动性和冲击成本")
                print("4. 多因子组合需要适当的风险管理")
                
            else:
                print("❌ 未找到符合标准的高质量因子")
        else:
            print("❌ 所有时间框架均未找到有效数据")
    else:
        print("❌ 未能加载任何有效数据")
    
    # 5. 系统性能统计
    print(f"\n⚡ 步骤5: 系统性能统计")
    print("-" * 50)
    
    import psutil
    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
    memory_used = current_memory - screener.start_memory
    
    print(f"✅ 性能统计:")
    print(f"   - 内存使用: {memory_used:.1f}MB")
    print(f"   - 缓存状态: {'启用' if config.cache_enabled else '禁用'}")
    print(f"   - 并行线程: {config.max_workers}")
    
    print(f"\n🎉 专业级因子筛选完成!")
    print("="*100)

def demo_custom_configuration():
    """演示自定义配置"""
    print("\n" + "="*100)
    print("自定义配置演示")
    print("="*100)
    
    # 保守型配置 - 适合稳健投资
    conservative_config = ScreeningConfig(
        ic_horizons=[5, 10, 20],                 # 更长周期
        min_sample_size=200,                     # 更大样本量
        alpha_level=0.01,                        # 更严格显著性
        fdr_method="bonferroni",                 # 更保守校正
        min_ic_threshold=0.03,                   # 更高IC要求
        min_ir_threshold=0.8,                    # 更高IR要求
        weight_stability=0.40,                   # 更重视稳定性
        weight_predictive=0.30,
        weight_independence=0.20,
        weight_practicality=0.10,
        weight_adaptability=0.00
    )
    
    # 激进型配置 - 适合高频交易
    aggressive_config = ScreeningConfig(
        ic_horizons=[1, 2, 3],                   # 短周期
        min_sample_size=50,                      # 较小样本量
        alpha_level=0.10,                        # 宽松显著性
        fdr_method="benjamini_hochberg",         # 标准校正
        min_ic_threshold=0.015,                  # 较低IC要求
        min_ir_threshold=0.3,                    # 较低IR要求
        weight_predictive=0.50,                  # 更重视预测能力
        weight_adaptability=0.20,               # 更重视适应性
        weight_stability=0.15,
        weight_independence=0.10,
        weight_practicality=0.05
    )
    
    print("📋 保守型配置 (稳健投资):")
    print(f"   - IC周期: {conservative_config.ic_horizons}")
    print(f"   - 显著性: {conservative_config.alpha_level} ({conservative_config.fdr_method})")
    print(f"   - IC阈值: {conservative_config.min_ic_threshold}")
    print(f"   - 权重分配: 稳定性{conservative_config.weight_stability:.0%}")
    
    print(f"\n📋 激进型配置 (高频交易):")
    print(f"   - IC周期: {aggressive_config.ic_horizons}")
    print(f"   - 显著性: {aggressive_config.alpha_level} ({aggressive_config.fdr_method})")
    print(f"   - IC阈值: {aggressive_config.min_ic_threshold}")
    print(f"   - 权重分配: 预测能力{aggressive_config.weight_predictive:.0%}")

if __name__ == "__main__":
    try:
        # 主演示
        demo_professional_screening()
        
        # 自定义配置演示
        demo_custom_configuration()
        
        print(f"\n✨ 演示完成! 系统已准备就绪。")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

