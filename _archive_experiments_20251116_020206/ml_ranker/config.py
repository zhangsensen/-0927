"""
数据源配置模块: 支持多换仓周期的训练数据源配置
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import yaml


@dataclass
class DataSource:
    """
    单个数据源配置
    
    Attributes:
        wfo_dir: WFO结果目录路径
        real_dir: 真实回测结果目录路径
        rebalance_days: 换仓周期天数
        weight: 数据集权重(保留字段,暂未使用)
        label: 数据源标签(可选,用于日志显示)
    """
    wfo_dir: str
    real_dir: str
    rebalance_days: int
    weight: float = 1.0
    label: Optional[str] = None
    
    def __post_init__(self):
        """验证配置有效性"""
        if self.rebalance_days <= 0:
            raise ValueError(f"rebalance_days必须为正整数,当前值: {self.rebalance_days}")
        
        if self.weight <= 0:
            raise ValueError(f"weight必须为正数,当前值: {self.weight}")
        
        # 转换为Path对象以便后续使用
        self.wfo_dir = str(Path(self.wfo_dir))
        self.real_dir = str(Path(self.real_dir))
    
    @property
    def display_name(self) -> str:
        """显示名称"""
        if self.label:
            return f"{self.label} ({self.rebalance_days}天)"
        return f"{self.rebalance_days}天换仓"


@dataclass
class DatasetConfig:
    """
    完整数据集配置
    
    Attributes:
        datasets: 数据源列表
        target_col: 主目标列名(用于训练)
        secondary_target: 次要目标列名(用于验证)
        metadata: 额外元数据(可选)
    """
    datasets: List[DataSource]
    target_col: str = "annual_ret_net"
    secondary_target: Optional[str] = "sharpe_net"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """验证配置有效性"""
        if not self.datasets:
            raise ValueError("datasets列表不能为空")
        
        if not self.target_col:
            raise ValueError("target_col不能为空")
    
    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> DatasetConfig:
        """
        从YAML配置文件加载
        
        Args:
            yaml_path: YAML配置文件路径
            
        Returns:
            DatasetConfig对象
            
        Raises:
            FileNotFoundError: 配置文件不存在
            ValueError: YAML格式错误
        """
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"YAML格式错误: {e}")
        
        if not isinstance(data, dict):
            raise ValueError("YAML根节点必须是字典")
        
        if 'datasets' not in data:
            raise ValueError("YAML配置缺少'datasets'字段")
        
        # 解析数据源列表
        datasets = []
        for idx, ds_data in enumerate(data['datasets'], 1):
            try:
                datasets.append(DataSource(**ds_data))
            except TypeError as e:
                raise ValueError(f"数据源{idx}配置错误: {e}")
        
        # 构建配置对象
        return cls(
            datasets=datasets,
            target_col=data.get('target_col', 'annual_ret_net'),
            secondary_target=data.get('secondary_target', 'sharpe_net'),
            metadata=data.get('metadata', {})
        )
    
    @classmethod
    def from_single_source(
        cls,
        wfo_dir: str,
        real_dir: str,
        rebalance_days: int = 8,
        target_col: str = "annual_ret_net"
    ) -> DatasetConfig:
        """
        从单个数据源创建配置(用于向后兼容)
        
        Args:
            wfo_dir: WFO结果目录
            real_dir: 真实回测目录
            rebalance_days: 换仓周期
            target_col: 目标列名
            
        Returns:
            DatasetConfig对象
        """
        dataset = DataSource(
            wfo_dir=wfo_dir,
            real_dir=real_dir,
            rebalance_days=rebalance_days,
            label="单数据源"
        )
        
        return cls(
            datasets=[dataset],
            target_col=target_col
        )
    
    def get_total_weight(self) -> float:
        """计算总权重"""
        return sum(ds.weight for ds in self.datasets)
    
    def get_rebalance_days_list(self) -> List[int]:
        """获取所有换仓周期列表"""
        return [ds.rebalance_days for ds in self.datasets]
    
    def summary(self) -> str:
        """生成配置摘要"""
        lines = [
            "数据集配置摘要:",
            f"  数据源数量: {len(self.datasets)}",
            f"  目标列: {self.target_col}",
            f"  次要目标: {self.secondary_target or 'None'}",
            f"  换仓周期: {self.get_rebalance_days_list()}",
            "",
            "数据源详情:"
        ]
        
        for idx, ds in enumerate(self.datasets, 1):
            lines.append(f"  [{idx}] {ds.display_name}")
            lines.append(f"      WFO: {ds.wfo_dir}")
            lines.append(f"      回测: {ds.real_dir}")
            lines.append(f"      权重: {ds.weight}")
        
        return "\n".join(lines)
