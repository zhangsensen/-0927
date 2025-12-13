"""
市场Regime检测器 v1.0
================================================================================
用于识别市场环境类型（牛市/熊市/震荡市），支持WFO按regime分组训练

设计思路：
1. 综合多个指标判断市场状态
2. 滑动窗口计算，避免前视偏差
3. 输出每日的regime标签

关键指标：
- 趋势：MA20 vs MA60, Slope
- 波动：Volatility
- 收益：Rolling Return
- Sharpe：Risk-adjusted return
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """市场regime类型"""
    BULL = "bull"      # 牛市：上涨+低波动
    BEAR = "bear"      # 熊市：下跌+高波动
    SIDEWAYS = "sideways"  # 震荡：横盘+中等波动


class RegimeDetector:
    """
    市场Regime检测器
    
    算法：
    1. 计算市场指数（43 ETF等权均值）
    2. 计算滑动窗口指标（MA, Vol, Sharpe）
    3. 基于规则分类regime
    
    规则（使用60日窗口）：
    - BULL: Return > 5% AND Volatility < 20% AND Sharpe > 0.5
    - BEAR: Return < -5% OR (Return < 0 AND Volatility > 25%)
    - SIDEWAYS: 其他
    """
    
    def __init__(
        self,
        window: int = 60,
        bull_return_threshold: float = 0.05,
        bear_return_threshold: float = -0.05,
        bull_vol_threshold: float = 0.20,
        bear_vol_threshold: float = 0.25,
        bull_sharpe_threshold: float = 0.5
    ):
        """
        参数:
            window: 滑动窗口大小（交易日）
            bull_return_threshold: 牛市收益阈值
            bear_return_threshold: 熊市收益阈值
            bull_vol_threshold: 牛市波动率阈值
            bear_vol_threshold: 熊市波动率阈值
            bull_sharpe_threshold: 牛市Sharpe阈值
        """
        self.window = window
        self.bull_return_threshold = bull_return_threshold
        self.bear_return_threshold = bear_return_threshold
        self.bull_vol_threshold = bull_vol_threshold
        self.bear_vol_threshold = bear_vol_threshold
        self.bull_sharpe_threshold = bull_sharpe_threshold
    
    def detect_regime(
        self,
        prices: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        检测市场regime
        
        参数:
            prices: OHLCV数据字典，必须包含'close'
        
        返回:
            regime_series: (T,) 每日的regime标签
            metrics_df: (T, n_metrics) 计算的中间指标
        """
        close = prices['close']
        
        # 1. 计算市场指数（等权平均）
        market_index = close.mean(axis=1)
        
        # 2. 计算滚动收益率
        rolling_return = market_index.pct_change(self.window)
        
        # 3. 计算滚动波动率（年化）
        daily_returns = market_index.pct_change()
        rolling_vol = daily_returns.rolling(self.window).std() * np.sqrt(252)
        
        # 4. 计算滚动Sharpe（假设无风险利率=0）
        rolling_mean_ret = daily_returns.rolling(self.window).mean() * 252
        rolling_sharpe = rolling_mean_ret / (rolling_vol + 1e-10)
        
        # 5. 计算MA20 vs MA60
        ma20 = market_index.rolling(20).mean()
        ma60 = market_index.rolling(60).mean()
        ma_diff_pct = (ma20 - ma60) / ma60
        
        # 6. 分类regime
        regime = pd.Series(index=close.index, dtype=str)
        
        for i, date in enumerate(close.index):
            if i < self.window:
                # 前window天无法判断
                regime.iloc[i] = MarketRegime.SIDEWAYS.value
                continue
            
            ret = rolling_return.iloc[i]
            vol = rolling_vol.iloc[i]
            sharpe = rolling_sharpe.iloc[i]
            ma_diff = ma_diff_pct.iloc[i]
            
            # 判断逻辑
            if pd.isna(ret) or pd.isna(vol):
                regime.iloc[i] = MarketRegime.SIDEWAYS.value
            elif (ret > self.bull_return_threshold and 
                  vol < self.bull_vol_threshold and 
                  sharpe > self.bull_sharpe_threshold):
                regime.iloc[i] = MarketRegime.BULL.value
            elif (ret < self.bear_return_threshold or 
                  (ret < 0 and vol > self.bear_vol_threshold)):
                regime.iloc[i] = MarketRegime.BEAR.value
            else:
                regime.iloc[i] = MarketRegime.SIDEWAYS.value
        
        # 7. 组装指标DataFrame
        metrics_df = pd.DataFrame({
            'market_index': market_index,
            'rolling_return': rolling_return,
            'rolling_vol': rolling_vol,
            'rolling_sharpe': rolling_sharpe,
            'ma20': ma20,
            'ma60': ma60,
            'ma_diff_pct': ma_diff_pct,
            'regime': regime
        }, index=close.index)
        
        logger.info(f"✅ Regime检测完成: {len(close)} 日期")
        self._log_regime_stats(regime)
        
        return regime, metrics_df
    
    def _log_regime_stats(self, regime: pd.Series):
        """记录regime统计信息"""
        regime_counts = regime.value_counts()
        total = len(regime)
        
        logger.info("📊 Regime分布:")
        for r, count in regime_counts.items():
            logger.info(f"  {r:10s}: {count:5d} ({count/total:6.1%})")
    
    def get_regime_periods(
        self,
        regime: pd.Series,
        target_regime: MarketRegime
    ) -> list[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        获取特定regime的连续时间段
        
        参数:
            regime: regime序列
            target_regime: 目标regime类型
        
        返回:
            [(start_date, end_date), ...] 连续时间段列表
        """
        is_target = regime == target_regime.value
        
        periods = []
        start = None
        
        for i, (date, is_match) in enumerate(is_target.items()):
            if is_match and start is None:
                start = date
            elif not is_match and start is not None:
                periods.append((start, regime.index[i-1]))
                start = None
        
        # 处理最后一段
        if start is not None:
            periods.append((start, regime.index[-1]))
        
        return periods


if __name__ == '__main__':
    """测试代码"""
    import yaml
    from pathlib import Path
    from etf_strategy.core.data_loader import DataLoader
    
    print("=" * 80)
    print("🔬 Regime检测器测试")
    print("=" * 80)
    
    # 加载配置
    ROOT = Path(__file__).parent.parent.parent.parent
    config_path = ROOT / "configs/combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 加载数据
    data_loader = DataLoader(
        data_dir=config["data"]["data_dir"],
        cache_dir=config["data"]["cache_dir"],
    )
    
    ohlcv_data = data_loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date="2020-01-01",
        end_date="2025-12-08",
    )
    
    print(f"✅ 数据加载: {len(ohlcv_data['close'])} 天 × {len(config['data']['symbols'])} ETF")
    
    # 检测regime
    detector = RegimeDetector(window=60)
    regime, metrics = detector.detect_regime(ohlcv_data)
    
    # 输出结果
    print("\n" + "=" * 80)
    print("📊 Regime分布")
    print("=" * 80)
    print(regime.value_counts())
    
    # 显示最近30天
    print("\n" + "=" * 80)
    print("📅 最近30天 Regime")
    print("=" * 80)
    recent = metrics.tail(30)[['market_index', 'rolling_return', 'rolling_vol', 'rolling_sharpe', 'regime']]
    print(recent.to_string())
    
    # 按regime分段
    print("\n" + "=" * 80)
    print("📈 牛市时间段（前10个）")
    print("=" * 80)
    bull_periods = detector.get_regime_periods(regime, MarketRegime.BULL)
    for start, end in bull_periods[:10]:
        days = (end - start).days
        print(f"  {start.date()} ~ {end.date()} ({days}天)")
    
    print("\n✅ 测试完成")
