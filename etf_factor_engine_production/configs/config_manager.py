"""配置管理器 - 统一配置访问接口

Linus原则: 配置驱动，消除硬编码
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    """ETF因子系统配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化配置管理器
        
        Args:
            config_path: 配置文件路径，默认使用etf_config.yaml
        """
        if config_path is None:
            config_path = Path(__file__).parent / "etf_config.yaml"
        
        self.config_path = Path(config_path)
        self._load_config()
    
    def _load_config(self):
        """加载配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
    
    @property
    def data_source_dir(self) -> str:
        """数据源目录"""
        return self.config['data']['source_dir']
    
    @property
    def output_dir(self) -> str:
        """输出目录"""
        return self.config['data']['output_dir']
    
    @property
    def start_date(self) -> str:
        """默认起始日期"""
        return self.config['data']['date_range']['start']
    
    @property
    def end_date(self) -> str:
        """默认结束日期"""
        return self.config['data']['date_range']['end']
    
    @property
    def engine_version(self) -> str:
        """引擎版本"""
        return self.config['engine']['version']
    
    @property
    def price_field(self) -> str:
        """价格字段"""
        return self.config['engine']['price_field']
    
    @property
    def compression(self) -> str:
        """压缩算法"""
        return self.config.get('performance', {}).get('compression', 'snappy')
    
    @property
    def dtype_optimization(self) -> bool:
        """是否启用数据类型优化"""
        return self.config.get('performance', {}).get('dtype_optimization', True)
    
    @property
    def correlation_threshold(self) -> float:
        """因子去重相关性阈值"""
        return self.config.get('quality', {}).get('correlation_threshold', 0.95)
    
    @property
    def null_threshold(self) -> float:
        """缺失率阈值"""
        return self.config.get('quality', {}).get('null_threshold', 0.1)
    
    @property
    def zero_threshold(self) -> float:
        """零值率阈值"""
        return self.config.get('quality', {}).get('zero_threshold', 0.5)
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项
        
        Args:
            key: 配置键（支持点号分隔的嵌套键）
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value


# 全局配置实例
_config_instance = None


def get_config() -> ConfigManager:
    """获取全局配置实例（单例模式）"""
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager()
    return _config_instance
