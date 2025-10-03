#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
输入验证工具 - P3-1安全加固
提供参数范围检查、路径安全验证、异常边界处理
"""

import re
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class ValidationError(Exception):
    """输入验证异常"""
    pass


class InputValidator:
    """输入验证器 - 统一的参数和数据验证"""
    
    # 安全路径模式（防止路径遍历攻击）
    SAFE_PATH_PATTERN = re.compile(r'^[a-zA-Z0-9_\-./]+$')
    DANGEROUS_PATH_PATTERNS = ['..', '~', '$', '`', '|', ';', '&']
    
    # 股票代码模式
    SYMBOL_PATTERN = re.compile(r'^[A-Z0-9]{4,6}\.(HK|SZ|SH|US)$')
    
    # 时间框架模式
    VALID_TIMEFRAMES = ['1min', '2min', '3min', '5min', '15min', '30min', '60min', 'daily', '1d']
    
    @staticmethod
    def validate_symbol(symbol: str, strict: bool = True) -> Tuple[bool, str]:
        """
        验证股票代码格式
        
        Args:
            symbol: 股票代码
            strict: 严格模式（完全匹配模式），否则仅基础检查
            
        Returns:
            (是否有效, 错误消息)
        """
        if not isinstance(symbol, str):
            return False, f"symbol必须是字符串，实际类型: {type(symbol)}"
        
        if len(symbol) == 0:
            return False, "symbol不能为空"
        
        if len(symbol) > 20:
            return False, f"symbol长度过长({len(symbol)})，最大20字符"
        
        if strict:
            if not InputValidator.SYMBOL_PATTERN.match(symbol):
                return False, (
                    f"symbol格式不符合规范: {symbol}。"
                    f"应为: XXXX.HK | XXXX.SZ | XXXX.SH | XXXX.US"
                )
        
        return True, ""
    
    @staticmethod
    def validate_timeframe(timeframe: str) -> Tuple[bool, str]:
        """
        验证时间框架格式
        
        Args:
            timeframe: 时间框架字符串
            
        Returns:
            (是否有效, 错误消息)
        """
        if not isinstance(timeframe, str):
            return False, f"timeframe必须是字符串，实际类型: {type(timeframe)}"
        
        if timeframe not in InputValidator.VALID_TIMEFRAMES:
            return False, (
                f"timeframe无效: {timeframe}。"
                f"有效值: {InputValidator.VALID_TIMEFRAMES}"
            )
        
        return True, ""
    
    @staticmethod
    def validate_path_safety(path: Union[str, Path], must_exist: bool = False) -> Tuple[bool, str]:
        """
        验证路径安全性（防止路径遍历攻击）
        
        Args:
            path: 文件/目录路径
            must_exist: 是否必须存在
            
        Returns:
            (是否安全, 错误消息)
        """
        if not isinstance(path, (str, Path)):
            return False, f"path必须是str或Path，实际类型: {type(path)}"
        
        path_str = str(path)
        
        # 检查危险模式
        for pattern in InputValidator.DANGEROUS_PATH_PATTERNS:
            if pattern in path_str:
                return False, f"路径包含危险字符: {pattern}"
        
        # 检查绝对路径或相对路径规范性
        try:
            normalized_path = Path(path).resolve()
        except Exception as e:
            return False, f"路径解析失败: {e}"
        
        # 检查是否存在（可选）
        if must_exist and not normalized_path.exists():
            return False, f"路径不存在: {normalized_path}"
        
        return True, ""
    
    @staticmethod
    def validate_numeric_range(
        value: Union[int, float],
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        param_name: str = "value"
    ) -> Tuple[bool, str]:
        """
        验证数值范围
        
        Args:
            value: 待验证数值
            min_value: 最小值（可选）
            max_value: 最大值（可选）
            param_name: 参数名称（用于错误消息）
            
        Returns:
            (是否有效, 错误消息)
        """
        if not isinstance(value, (int, float, np.number)):
            return False, f"{param_name}必须是数值类型，实际类型: {type(value)}"
        
        if not np.isfinite(value):
            return False, f"{param_name}包含非法值: {value}"
        
        if min_value is not None and value < min_value:
            return False, f"{param_name}={value} 小于最小值 {min_value}"
        
        if max_value is not None and value > max_value:
            return False, f"{param_name}={value} 大于最大值 {max_value}"
        
        return True, ""
    
    @staticmethod
    def validate_dataframe(
        df: pd.DataFrame,
        min_rows: int = 1,
        required_columns: Optional[List[str]] = None,
        allow_nan: bool = True
    ) -> Tuple[bool, str]:
        """
        验证DataFrame格式和内容
        
        Args:
            df: 待验证DataFrame
            min_rows: 最小行数
            required_columns: 必需的列名（可选）
            allow_nan: 是否允许NaN值
            
        Returns:
            (是否有效, 错误消息)
        """
        if not isinstance(df, pd.DataFrame):
            return False, f"必须是DataFrame，实际类型: {type(df)}"
        
        if len(df) < min_rows:
            return False, f"数据行数({len(df)})不足最小要求({min_rows})"
        
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                return False, f"缺少必需列: {missing_cols}"
        
        if not allow_nan:
            nan_count = df.isnull().sum().sum()
            if nan_count > 0:
                return False, f"包含{nan_count}个NaN值，不允许"
        
        return True, ""
    
    @staticmethod
    def validate_screening_config(config: Any) -> Tuple[bool, str]:
        """
        验证筛选配置参数
        
        Args:
            config: ScreeningConfig对象
            
        Returns:
            (是否有效, 错误消息)
        """
        # IC周期验证
        if hasattr(config, 'ic_horizons'):
            if not isinstance(config.ic_horizons, list):
                return False, "ic_horizons必须是列表"
            
            if len(config.ic_horizons) == 0:
                return False, "ic_horizons不能为空"
            
            for horizon in config.ic_horizons:
                is_valid, msg = InputValidator.validate_numeric_range(
                    horizon, min_value=1, max_value=100, param_name="ic_horizon"
                )
                if not is_valid:
                    return False, msg
        
        # alpha水平验证
        if hasattr(config, 'alpha_level'):
            is_valid, msg = InputValidator.validate_numeric_range(
                config.alpha_level, min_value=0.001, max_value=0.2, param_name="alpha_level"
            )
            if not is_valid:
                return False, msg
        
        # 最小样本量验证
        if hasattr(config, 'min_sample_size'):
            is_valid, msg = InputValidator.validate_numeric_range(
                config.min_sample_size, min_value=50, max_value=10000, param_name="min_sample_size"
            )
            if not is_valid:
                return False, msg
        
        # VIF阈值验证
        if hasattr(config, 'vif_threshold'):
            is_valid, msg = InputValidator.validate_numeric_range(
                config.vif_threshold, min_value=1.0, max_value=100.0, param_name="vif_threshold"
            )
            if not is_valid:
                return False, msg
        
        # 权重验证
        weight_fields = [
            'weight_predictive_power',
            'weight_stability',
            'weight_independence',
            'weight_practicality',
            'weight_short_term_adaptability'
        ]
        
        total_weight = 0.0
        for field in weight_fields:
            if hasattr(config, field):
                weight = getattr(config, field)
                is_valid, msg = InputValidator.validate_numeric_range(
                    weight, min_value=0.0, max_value=1.0, param_name=field
                )
                if not is_valid:
                    return False, msg
                total_weight += weight
        
        if total_weight > 0 and abs(total_weight - 1.0) > 0.01:
            return False, f"权重总和({total_weight:.3f})必须接近1.0"
        
        return True, ""
    
    @staticmethod
    def sanitize_filename(filename: str, max_length: int = 200) -> str:
        """
        清理文件名，移除危险字符
        
        Args:
            filename: 原始文件名
            max_length: 最大长度
            
        Returns:
            清理后的文件名
        """
        # 移除危险字符
        safe_chars = re.sub(r'[^\w\-\.]', '_', filename)
        
        # 限制长度
        if len(safe_chars) > max_length:
            name, ext = safe_chars.rsplit('.', 1) if '.' in safe_chars else (safe_chars, '')
            name = name[:max_length - len(ext) - 1]
            safe_chars = f"{name}.{ext}" if ext else name
        
        return safe_chars


# 便捷函数

def validate_and_load_config(config_path: Union[str, Path]) -> Any:
    """
    验证并加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置对象
        
    Raises:
        ValidationError: 配置验证失败
    """
    # 路径安全检查
    is_safe, msg = InputValidator.validate_path_safety(config_path, must_exist=True)
    if not is_safe:
        raise ValidationError(f"配置文件路径不安全: {msg}")
    
    # 加载配置
    try:
        from config_manager import load_config  # type: ignore
        config = load_config(str(config_path))
    except Exception as e:
        raise ValidationError(f"加载配置失败: {e}")
    
    # 验证配置内容
    is_valid, msg = InputValidator.validate_screening_config(config)
    if not is_valid:
        raise ValidationError(f"配置参数无效: {msg}")
    
    return config


def validate_factor_data(
    factors: pd.DataFrame,
    returns: pd.Series,
    min_sample_size: int = 200
) -> None:
    """
    验证因子和收益数据
    
    Args:
        factors: 因子DataFrame
        returns: 收益Series
        min_sample_size: 最小样本量
        
    Raises:
        ValidationError: 数据验证失败
    """
    # 验证因子数据
    is_valid, msg = InputValidator.validate_dataframe(
        factors, min_rows=min_sample_size, allow_nan=True
    )
    if not is_valid:
        raise ValidationError(f"因子数据无效: {msg}")
    
    # 验证收益数据
    if not isinstance(returns, pd.Series):
        raise ValidationError(f"收益数据必须是Series，实际类型: {type(returns)}")
    
    if len(returns) < min_sample_size:
        raise ValidationError(f"收益数据样本量不足: {len(returns)} < {min_sample_size}")
    
    # 验证时间对齐
    common_idx = factors.index.intersection(returns.index)
    if len(common_idx) < min_sample_size:
        raise ValidationError(
            f"因子和收益时间对齐后样本量不足: {len(common_idx)} < {min_sample_size}"
        )

