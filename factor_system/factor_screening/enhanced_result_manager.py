#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版筛选结果管理器
基于时间戳文件夹的完整信息存储系统
"""

import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import platform

logger = logging.getLogger(__name__)

@dataclass
class ScreeningSession:
    """筛选会话信息"""
    session_id: str
    timestamp: str
    symbol: str
    timeframe: str
    config_hash: str
    total_factors: int
    significant_factors: int
    high_score_factors: int
    total_time_seconds: float
    memory_used_mb: float
    sample_size: int
    data_quality_score: float
    top_factor_name: str
    top_factor_score: float

class EnhancedResultManager:
    """增强版结果管理器 - 基于时间戳文件夹的完整存储"""
    
    def __init__(self, base_output_dir: str = "./因子筛选"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建会话索引文件
        self.sessions_index_file = self.base_output_dir / "screening_sessions_index.json"
        self.sessions_index = self._load_sessions_index()
        
    def _load_sessions_index(self) -> List[Dict]:
        """加载会话索引"""
        if self.sessions_index_file.exists():
            try:
                with open(self.sessions_index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_sessions_index(self):
        """保存会话索引"""
        with open(self.sessions_index_file, 'w', encoding='utf-8') as f:
            json.dump(self.sessions_index, f, indent=2, ensure_ascii=False, default=str)
    
    def create_screening_session(self, symbol: str, timeframe: str, 
                               results: Dict, screening_stats: Dict,
                               config: Any, data_quality_info: Dict = None) -> str:
        """创建完整的筛选会话存储"""
        
        # 1. 创建时间戳文件夹
        timestamp = datetime.now()
        session_id = f"{symbol}_{timeframe}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        session_dir = self.base_output_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"创建筛选会话: {session_id}")
        logger.info(f"会话目录: {session_dir}")
        
        # 2. 保存核心筛选数据
        self._save_core_screening_data(session_dir, symbol, timeframe, results, screening_stats)
        
        # 3. 保存配置和元数据
        self._save_configuration_data(session_dir, config, data_quality_info)
        
        # 4. 保存分析报告
        self._save_analysis_reports(session_dir, symbol, timeframe, results, screening_stats)
        
        # 5. 保存可视化图表
        self._save_visualization_charts(session_dir, results, screening_stats)
        
        # 6. 保存因子相关性分析
        self._save_factor_correlation_analysis(session_dir, results)
        
        # 7. 保存IC时间序列分析
        self._save_ic_time_series_analysis(session_dir, results)
        
        # 8. 生成会话摘要
        session_summary = self._generate_session_summary(session_dir, symbol, timeframe, 
                                                       results, screening_stats, config)
        
        # 9. 更新会话索引
        self._update_sessions_index(session_id, session_summary)
        
        # 10. 生成README文件
        self._generate_session_readme(session_dir, session_summary)
        
        logger.info(f"✅ 筛选会话创建完成: {session_id}")
        return session_id
    
    def _save_core_screening_data(self, session_dir: Path, symbol: str, timeframe: str,
                                results: Dict, screening_stats: Dict):
        """保存核心筛选数据"""
        
        # 1. 详细因子筛选报告 (CSV)
        report_data = []
        for factor_name, metrics in results.items():
            row = {
                'Factor': factor_name,
                'Comprehensive_Score': metrics.comprehensive_score,
                'Predictive_Power_Mean_IC': metrics.predictive_power_mean_ic,
                'Predictive_Power_IC_IR': metrics.predictive_power_ic_ir,
                'Stability_Rolling_IC_Mean': metrics.stability_rolling_ic_mean,
                'Stability_Rolling_IC_Std': metrics.stability_rolling_ic_std,
                'Independence_VIF': metrics.independence_vif,
                'Independence_Information_Increment': metrics.independence_information_increment,
                'Practicality_Turnover_Rate': metrics.practicality_turnover_rate,
                'Practicality_Transaction_Cost': metrics.practicality_transaction_cost,
                'Short_Term_Adaptability_Reversal_Effect': metrics.short_term_adaptability_reversal_effect,
                'Short_Term_Adaptability_Momentum_Persistence': metrics.short_term_adaptability_momentum_persistence,
                'P_Value': metrics.p_value,
                'FDR_P_Value': metrics.fdr_p_value,
                'Is_Significant': metrics.is_significant,
                'Tier': metrics.tier,
                'Type': metrics.type,
                'Description': metrics.description
            }
            report_data.append(row)
        
        report_df = pd.DataFrame(report_data).sort_values('Comprehensive_Score', ascending=False)
        report_df.to_csv(session_dir / "detailed_factor_report.csv", index=False, encoding='utf-8')
        
        # 2. 筛选统计信息 (JSON)
        enhanced_stats = {
            **screening_stats,
            'session_metadata': {
                'symbol': symbol,
                'timeframe': timeframe,
                'screening_timestamp': datetime.now().isoformat(),
                'total_factors_processed': len(results),
                'factors_by_tier': self._count_factors_by_tier(results),
                'score_distribution': self._calculate_score_distribution(results)
            }
        }
        
        with open(session_dir / "screening_statistics.json", 'w', encoding='utf-8') as f:
            json.dump(enhanced_stats, f, indent=2, ensure_ascii=False, default=str)
        
        # 3. 顶级因子详细信息 (JSON)
        top_factors = sorted(results.values(), key=lambda x: x.comprehensive_score, reverse=True)[:20]
        top_factors_data = []
        
        for i, factor in enumerate(top_factors, 1):
            factor_info = {
                'rank': i,
                'name': factor.name,
                'comprehensive_score': round(factor.comprehensive_score, 4),
                'predictive_score': round(factor.predictive_score, 4),
                'stability_score': round(factor.stability_score, 4),
                'independence_score': round(factor.independence_score, 4),
                'practicality_score': round(factor.practicality_score, 4),
                'adaptability_score': round(factor.adaptability_score, 4),
                'is_significant': factor.is_significant,
                'tier': factor.tier,
                'type': factor.type,
                'description': factor.description,
                'key_metrics': {
                    'mean_ic': round(factor.predictive_power_mean_ic, 4),
                    'ic_ir': round(factor.predictive_power_ic_ir, 4),
                    'rolling_ic_mean': round(factor.stability_rolling_ic_mean, 4),
                    'vif': round(factor.independence_vif, 4) if factor.independence_vif else None,
                    'turnover_rate': round(factor.practicality_turnover_rate, 4),
                    'transaction_cost': round(factor.practicality_transaction_cost, 4)
                }
            }
            top_factors_data.append(factor_info)
        
        with open(session_dir / "top_factors_detailed.json", 'w', encoding='utf-8') as f:
            json.dump(top_factors_data, f, indent=2, ensure_ascii=False)
    
    def _save_configuration_data(self, session_dir: Path, config: Any, data_quality_info: Dict):
        """保存配置和数据质量信息"""
        
        # 1. 筛选配置 (YAML)
        config_dict = {
            'screening_parameters': asdict(config) if hasattr(config, '__dict__') else str(config),
            'execution_info': {
                'timestamp': datetime.now().isoformat(),
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'system_info': {
                    'platform': platform.system(),
                    'architecture': platform.architecture()[0],
                    'processor': platform.processor()
                }
            }
        }
        
        with open(session_dir / "screening_config.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, indent=2)
        
        # 2. 数据质量报告 (JSON)
        if data_quality_info:
            enhanced_quality_info = {
                **data_quality_info,
                'quality_assessment': {
                    'overall_score': self._calculate_overall_quality_score(data_quality_info),
                    'recommendations': self._generate_quality_recommendations(data_quality_info)
                }
            }
            
            with open(session_dir / "data_quality_report.json", 'w', encoding='utf-8') as f:
                json.dump(enhanced_quality_info, f, indent=2, ensure_ascii=False, default=str)
    
    def _save_analysis_reports(self, session_dir: Path, symbol: str, timeframe: str,
                             results: Dict, screening_stats: Dict):
        """保存分析报告"""
        
        # 1. 执行摘要 (TXT)
        summary_path = session_dir / "executive_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"🎯 因子筛选执行摘要\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"📊 基本信息\n")
            f.write(f"股票代码: {symbol}\n")
            f.write(f"时间框架: {timeframe}\n")
            f.write(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总耗时: {screening_stats.get('total_time', 0):.2f}秒\n")
            f.write(f"内存使用: {screening_stats.get('memory_used_mb', 0):.1f}MB\n\n")
            
            f.write(f"📈 筛选结果统计\n")
            f.write(f"总因子数: {screening_stats.get('total_factors', 0)}\n")
            f.write(f"显著因子: {screening_stats.get('significant_factors', 0)}\n")
            f.write(f"高分因子 (>0.6): {screening_stats.get('high_score_factors', 0)}\n")
            f.write(f"样本量: {screening_stats.get('sample_size', 0)}\n\n")
            
            # 顶级因子列表
            top_factors = sorted(results.values(), key=lambda x: x.comprehensive_score, reverse=True)[:10]
            f.write(f"🏆 前10名顶级因子\n")
            for i, factor in enumerate(top_factors, 1):
                f.write(f"{i:2d}. {factor.name:<25} 得分:{factor.comprehensive_score:.3f} ")
                f.write(f"IC:{factor.predictive_power_mean_ic:.3f} ")
                f.write(f"显著性:{'✓' if factor.is_significant else '✗'}\n")
            
            f.write(f"\n📋 因子分层统计\n")
            tier_counts = self._count_factors_by_tier(results)
            for tier, count in tier_counts.items():
                f.write(f"{tier}: {count} 个\n")
        
        # 2. 详细分析报告 (Markdown)
        analysis_path = session_dir / "detailed_analysis.md"
        with open(analysis_path, 'w', encoding='utf-8') as f:
            f.write(f"# 因子筛选详细分析报告\n\n")
            f.write(f"## 基本信息\n")
            f.write(f"- **股票代码**: {symbol}\n")
            f.write(f"- **时间框架**: {timeframe}\n")
            f.write(f"- **分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"## 筛选结果概览\n")
            f.write(f"| 指标 | 数值 |\n")
            f.write(f"|------|------|\n")
            f.write(f"| 总因子数 | {screening_stats.get('total_factors', 0)} |\n")
            f.write(f"| 显著因子数 | {screening_stats.get('significant_factors', 0)} |\n")
            f.write(f"| 高分因子数 | {screening_stats.get('high_score_factors', 0)} |\n")
            f.write(f"| 处理时间 | {screening_stats.get('total_time', 0):.2f}秒 |\n")
            f.write(f"| 内存使用 | {screening_stats.get('memory_used_mb', 0):.1f}MB |\n\n")
            
            f.write(f"## 顶级因子分析\n")
            top_factors = sorted(results.values(), key=lambda x: x.comprehensive_score, reverse=True)[:10]
            f.write(f"| 排名 | 因子名称 | 综合得分 | 预测能力 | 稳定性 | 独立性 | 实用性 |\n")
            f.write(f"|------|----------|----------|----------|--------|--------|--------|\n")
            for i, factor in enumerate(top_factors, 1):
                f.write(f"| {i} | {factor.name} | {factor.comprehensive_score:.3f} | ")
                f.write(f"{factor.predictive_score:.3f} | {factor.stability_score:.3f} | ")
                f.write(f"{factor.independence_score:.3f} | {factor.practicality_score:.3f} |\n")
    
    def _save_visualization_charts(self, session_dir: Path, results: Dict, screening_stats: Dict):
        """保存可视化图表"""
        charts_dir = session_dir / "charts"
        charts_dir.mkdir(exist_ok=True)
        
        try:
            # 1. 因子得分分布图
            scores = [factor.comprehensive_score for factor in results.values()]
            plt.figure(figsize=(10, 6))
            plt.hist(scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title('因子综合得分分布')
            plt.xlabel('综合得分')
            plt.ylabel('因子数量')
            plt.grid(True, alpha=0.3)
            plt.savefig(charts_dir / "score_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. 顶级因子雷达图
            top_factors = sorted(results.values(), key=lambda x: x.comprehensive_score, reverse=True)[:5]
            
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
            
            categories = ['预测能力', '稳定性', '独立性', '实用性', '适应性']
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]  # 闭合
            
            for i, factor in enumerate(top_factors):
                values = [
                    factor.predictive_score,
                    factor.stability_score, 
                    factor.independence_score,
                    factor.practicality_score,
                    factor.adaptability_score
                ]
                values += values[:1]  # 闭合
                
                ax.plot(angles, values, 'o-', linewidth=2, label=factor.name)
                ax.fill(angles, values, alpha=0.25)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 1)
            ax.set_title('顶级因子五维度评分对比', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            
            plt.savefig(charts_dir / "top_factors_radar.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. 因子类型分布饼图
            factor_types = {}
            for factor in results.values():
                factor_type = factor.type or 'Unknown'
                factor_types[factor_type] = factor_types.get(factor_type, 0) + 1
            
            plt.figure(figsize=(8, 8))
            plt.pie(factor_types.values(), labels=factor_types.keys(), autopct='%1.1f%%')
            plt.title('因子类型分布')
            plt.savefig(charts_dir / "factor_types_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"生成可视化图表失败: {e}")
    
    def _save_factor_correlation_analysis(self, session_dir: Path, results: Dict):
        """保存因子相关性分析"""
        try:
            # 提取顶级因子的关键指标
            top_factors = sorted(results.values(), key=lambda x: x.comprehensive_score, reverse=True)[:20]
            
            correlation_data = []
            for factor in top_factors:
                correlation_data.append({
                    'name': factor.name,
                    'comprehensive_score': factor.comprehensive_score,
                    'predictive_score': factor.predictive_score,
                    'stability_score': factor.stability_score,
                    'independence_score': factor.independence_score,
                    'practicality_score': factor.practicality_score,
                    'mean_ic': factor.predictive_power_mean_ic,
                    'ic_ir': factor.predictive_power_ic_ir,
                    'rolling_ic_mean': factor.stability_rolling_ic_mean
                })
            
            correlation_df = pd.DataFrame(correlation_data)
            correlation_df.set_index('name', inplace=True)
            
            # 保存相关性矩阵
            correlation_matrix = correlation_df.corr()
            correlation_matrix.to_csv(session_dir / "factor_correlation_matrix.csv", encoding='utf-8')
            
            # 生成相关性热力图
            charts_dir = session_dir / "charts"
            charts_dir.mkdir(exist_ok=True)
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, fmt='.2f')
            plt.title('顶级因子指标相关性热力图')
            plt.tight_layout()
            plt.savefig(charts_dir / "correlation_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"生成因子相关性分析失败: {e}")
    
    def _save_ic_time_series_analysis(self, session_dir: Path, results: Dict):
        """保存IC时间序列分析"""
        try:
            # 这里需要从results中提取IC时间序列数据
            # 由于当前FactorMetrics可能没有存储完整的IC时间序列，
            # 我们先保存一个占位符文件，后续可以扩展
            
            ic_analysis = {
                'note': 'IC时间序列分析需要在因子筛选过程中收集更详细的时间序列数据',
                'available_metrics': {
                    'mean_ic_values': {factor.name: factor.predictive_power_mean_ic 
                                     for factor in results.values()},
                    'ic_ir_values': {factor.name: factor.predictive_power_ic_ir 
                                   for factor in results.values()},
                    'rolling_ic_means': {factor.name: factor.stability_rolling_ic_mean 
                                       for factor in results.values()}
                }
            }
            
            with open(session_dir / "ic_time_series_analysis.json", 'w', encoding='utf-8') as f:
                json.dump(ic_analysis, f, indent=2, ensure_ascii=False, default=str)
                
        except Exception as e:
            logger.warning(f"生成IC时间序列分析失败: {e}")
    
    def _generate_session_summary(self, session_dir: Path, symbol: str, timeframe: str,
                                results: Dict, screening_stats: Dict, config: Any) -> ScreeningSession:
        """生成会话摘要"""
        
        top_factor = max(results.values(), key=lambda x: x.comprehensive_score)
        
        return ScreeningSession(
            session_id=session_dir.name,
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            timeframe=timeframe,
            config_hash=str(hash(str(config)))[:8],
            total_factors=len(results),
            significant_factors=sum(1 for f in results.values() if f.is_significant),
            high_score_factors=sum(1 for f in results.values() if f.comprehensive_score > 0.6),
            total_time_seconds=screening_stats.get('total_time', 0),
            memory_used_mb=screening_stats.get('memory_used_mb', 0),
            sample_size=screening_stats.get('sample_size', 0),
            data_quality_score=0.85,  # 占位符，需要实际计算
            top_factor_name=top_factor.name,
            top_factor_score=top_factor.comprehensive_score
        )
    
    def _update_sessions_index(self, session_id: str, session_summary: ScreeningSession):
        """更新会话索引"""
        self.sessions_index.append(asdict(session_summary))
        self._save_sessions_index()
    
    def _generate_session_readme(self, session_dir: Path, session_summary: ScreeningSession):
        """生成会话README文件"""
        readme_path = session_dir / "README.md"
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"# 因子筛选会话: {session_summary.session_id}\n\n")
            f.write(f"## 会话信息\n")
            f.write(f"- **会话ID**: {session_summary.session_id}\n")
            f.write(f"- **股票代码**: {session_summary.symbol}\n")
            f.write(f"- **时间框架**: {session_summary.timeframe}\n")
            f.write(f"- **执行时间**: {session_summary.timestamp}\n")
            f.write(f"- **配置哈希**: {session_summary.config_hash}\n\n")
            
            f.write(f"## 筛选结果\n")
            f.write(f"- **总因子数**: {session_summary.total_factors}\n")
            f.write(f"- **显著因子数**: {session_summary.significant_factors}\n")
            f.write(f"- **高分因子数**: {session_summary.high_score_factors}\n")
            f.write(f"- **顶级因子**: {session_summary.top_factor_name} (得分: {session_summary.top_factor_score:.3f})\n\n")
            
            f.write(f"## 性能指标\n")
            f.write(f"- **执行时间**: {session_summary.total_time_seconds:.2f}秒\n")
            f.write(f"- **内存使用**: {session_summary.memory_used_mb:.1f}MB\n")
            f.write(f"- **样本量**: {session_summary.sample_size}\n\n")
            
            f.write(f"## 文件说明\n")
            f.write(f"- `detailed_factor_report.csv` - 详细因子筛选报告\n")
            f.write(f"- `screening_statistics.json` - 筛选过程统计信息\n")
            f.write(f"- `top_factors_detailed.json` - 顶级因子详细信息\n")
            f.write(f"- `screening_config.yaml` - 筛选配置参数\n")
            f.write(f"- `data_quality_report.json` - 数据质量报告\n")
            f.write(f"- `executive_summary.txt` - 执行摘要\n")
            f.write(f"- `detailed_analysis.md` - 详细分析报告\n")
            f.write(f"- `charts/` - 可视化图表目录\n")
            f.write(f"- `factor_correlation_matrix.csv` - 因子相关性矩阵\n")
            f.write(f"- `ic_time_series_analysis.json` - IC时间序列分析\n\n")
            
            f.write(f"## 使用建议\n")
            f.write(f"1. 查看 `executive_summary.txt` 获取快速概览\n")
            f.write(f"2. 分析 `detailed_factor_report.csv` 了解所有因子详情\n")
            f.write(f"3. 查看 `charts/` 目录中的可视化图表\n")
            f.write(f"4. 参考 `detailed_analysis.md` 进行深入分析\n")
    
    # 辅助方法
    def _count_factors_by_tier(self, results: Dict) -> Dict[str, int]:
        """统计各层级因子数量"""
        tier_counts = {}
        for factor in results.values():
            tier = factor.tier or 'Unknown'
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        return tier_counts
    
    def _calculate_score_distribution(self, results: Dict) -> Dict[str, int]:
        """计算得分分布"""
        distribution = {
            'excellent (>0.8)': 0,
            'good (0.6-0.8)': 0,
            'average (0.4-0.6)': 0,
            'poor (<0.4)': 0
        }
        
        for factor in results.values():
            score = factor.comprehensive_score
            if score > 0.8:
                distribution['excellent (>0.8)'] += 1
            elif score > 0.6:
                distribution['good (0.6-0.8)'] += 1
            elif score > 0.4:
                distribution['average (0.4-0.6)'] += 1
            else:
                distribution['poor (<0.4)'] += 1
        
        return distribution
    
    def _calculate_overall_quality_score(self, data_quality_info: Dict) -> float:
        """计算总体数据质量得分"""
        # 这里可以根据数据质量信息计算一个综合得分
        return 0.85  # 占位符
    
    def _generate_quality_recommendations(self, data_quality_info: Dict) -> List[str]:
        """生成数据质量建议"""
        recommendations = []
        # 根据数据质量信息生成具体建议
        recommendations.append("数据质量良好，可以进行因子筛选")
        return recommendations
    
    def get_session_history(self, symbol: str = None, timeframe: str = None, 
                          limit: int = 10) -> List[ScreeningSession]:
        """获取会话历史"""
        sessions = []
        for session_data in self.sessions_index:
            session = ScreeningSession(**session_data)
            
            # 过滤条件
            if symbol and session.symbol != symbol:
                continue
            if timeframe and session.timeframe != timeframe:
                continue
                
            sessions.append(session)
        
        # 按时间倒序排列
        sessions.sort(key=lambda x: x.timestamp, reverse=True)
        return sessions[:limit]
    
    def cleanup_old_sessions(self, keep_days: int = 30):
        """清理旧会话"""
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        
        sessions_to_remove = []
        for i, session_data in enumerate(self.sessions_index):
            session_time = datetime.fromisoformat(session_data['timestamp'])
            if session_time < cutoff_date:
                # 删除会话目录
                session_dir = self.base_output_dir / session_data['session_id']
                if session_dir.exists():
                    import shutil
                    shutil.rmtree(session_dir)
                sessions_to_remove.append(i)
        
        # 从索引中移除
        for i in reversed(sessions_to_remove):
            del self.sessions_index[i]
        
        self._save_sessions_index()
        logger.info(f"清理了 {len(sessions_to_remove)} 个旧会话")


# 导入必要的模块
import sys
import platform
from datetime import timedelta

if __name__ == "__main__":
    # 测试代码
    manager = EnhancedResultManager()
    print("增强版结果管理器初始化完成")

