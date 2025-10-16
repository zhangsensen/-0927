"""元数据生成器 - 完整的数据概要和质量指标

Linus原则: 可追溯、可监控、可验证
"""
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class MetadataGenerator:
    """元数据生成器"""
    
    @staticmethod
    def generate_metadata(
        panel: pd.DataFrame,
        panel_file: Path,
        run_params: Dict[str, Any],
        optimization_report: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """生成完整元数据
        
        Args:
            panel: 因子面板DataFrame
            panel_file: 面板文件路径
            run_params: 运行参数
            optimization_report: 优化报告
            
        Returns:
            完整元数据字典
        """
        logger.info("生成元数据...")
        
        # 基础统计
        file_size_mb = panel_file.stat().st_size / 1024 / 1024 if panel_file.exists() else 0
        
        # 提取日期范围
        if isinstance(panel.index, pd.MultiIndex):
            dates = panel.index.get_level_values('date')
            symbols = panel.index.get_level_values('symbol')
            start_date = dates.min()
            end_date = dates.max()
            etf_count = symbols.nunique()
            date_count = dates.nunique()
        else:
            start_date = panel.index.min()
            end_date = panel.index.max()
            etf_count = 1
            date_count = len(panel)
        
        # 数据质量指标
        null_rate = panel.isnull().sum().sum() / (len(panel) * len(panel.columns))
        zero_rate = (panel == 0).sum().sum() / (len(panel) * len(panel.columns))
        
        # 因子统计
        factor_volatility = panel.std()
        high_vol_count = (factor_volatility > 100).sum()
        
        # 相关性统计
        if len(panel.columns) > 0:
            import numpy as np
            corr_matrix = panel.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            high_corr_pairs = (upper_triangle > 0.95).sum().sum()
        else:
            high_corr_pairs = 0
        
        # 内存使用
        memory_mb = panel.memory_usage(deep=True).sum() / 1024 / 1024
        
        # 构建元数据
        metadata = {
            "engine_version": run_params.get("engine_version", "1.0.0"),
            
            "data_summary": {
                "start_date": str(start_date.date()) if hasattr(start_date, 'date') else str(start_date),
                "end_date": str(end_date.date()) if hasattr(end_date, 'date') else str(end_date),
                "etf_count": int(etf_count),
                "date_count": int(date_count),
                "factor_count": len(panel.columns),
                "data_points": len(panel),
                "coverage_rate": float(1 - null_rate),
                "file_size_mb": float(file_size_mb),
                "memory_mb": float(memory_mb),
                "compression_ratio": float(file_size_mb / memory_mb) if memory_mb > 0 else 0
            },
            
            "quality_metrics": {
                "null_rate": float(null_rate),
                "zero_rate": float(zero_rate),
                "high_volatility_factors": int(high_vol_count),
                "high_correlation_pairs": int(high_corr_pairs),
                "avg_factor_volatility": float(factor_volatility.mean()),
                "max_factor_volatility": float(factor_volatility.max())
            },
            
            "run_params": run_params,
            
            "timestamp": datetime.now().isoformat()
        }
        
        # 添加优化报告
        if optimization_report:
            metadata["optimization"] = optimization_report
        
        logger.info("元数据生成完成")
        logger.info(f"  ETF数量: {metadata['data_summary']['etf_count']}")
        logger.info(f"  因子数量: {metadata['data_summary']['factor_count']}")
        logger.info(f"  数据点数: {metadata['data_summary']['data_points']}")
        logger.info(f"  覆盖率: {metadata['data_summary']['coverage_rate']:.2%}")
        logger.info(f"  文件大小: {metadata['data_summary']['file_size_mb']:.1f} MB")
        
        return metadata
    
    @staticmethod
    def save_metadata(metadata: Dict[str, Any], output_file: Path):
        """保存元数据到JSON文件
        
        Args:
            metadata: 元数据字典
            output_file: 输出文件路径
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"元数据已保存: {output_file}")
