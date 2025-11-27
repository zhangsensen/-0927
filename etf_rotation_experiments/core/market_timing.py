import pandas as pd
import numpy as np

class LightTimingModule:
    """
    Light Timing Module as described in 1126.md
    
    Logic:
    - MA Signal (HS300): Price > MA200 ? 1 : -1
    - Mom Signal (HS300): MOM_20D > 0 ? 1 : -1
    - Gold Signal (Gold ETF): Price > MA200 ? 1 : -1
    
    Composite = 0.4 * MA + 0.4 * Mom + 0.2 * Gold
    Position = 0.3 if Composite < -0.4 else 1.0
    """
    
    def __init__(self, extreme_threshold: float = -0.4, extreme_position: float = 0.3):
        self.extreme_threshold = extreme_threshold
        self.extreme_position = extreme_position
        
    def compute_position_ratios(self, close_df: pd.DataFrame, market_symbol: str = '510300', gold_symbol: str = '518880') -> pd.Series:
        """
        Compute daily position ratios (0.0 to 1.0).
        
        Args:
            close_df: DataFrame of close prices (Date x Symbol)
            market_symbol: Ticker for Market Index (default 510300 HS300)
            gold_symbol: Ticker for Gold ETF (default 518880)
            
        Returns:
            pd.Series of position ratios (index=Date)
        """
        # Check if symbols exist
        if market_symbol not in close_df.columns:
            # Fallback: try 510050 if 510300 missing
            if '510050' in close_df.columns:
                market_symbol = '510050'
            else:
                print(f"Warning: Market symbol {market_symbol} not found in prices. Timing disabled.")
                return pd.Series(1.0, index=close_df.index)
            
        market_close = close_df[market_symbol]
        
        # MA Signal: Price > MA200
        ma200 = market_close.rolling(window=200, min_periods=1).mean()
        # Use 0 for NaN to avoid issues, though min_periods=1 helps
        ma_signal = np.where(market_close > ma200, 1.0, -1.0)
        ma_signal = pd.Series(ma_signal, index=market_close.index)
        
        # Mom Signal: 20D Return > 0
        mom20 = market_close.pct_change(20)
        mom_signal = np.where(mom20 > 0, 1.0, -1.0)
        # Handle NaNs in mom20 (first 20 days) -> assume Bullish (1.0) to avoid defensive start?
        # Or neutral. Let's assume 1.0 to match "default full position".
        mom_signal[mom20.isna()] = 1.0
        mom_signal = pd.Series(mom_signal, index=market_close.index)
        
        # Gold Signal: Price > MA200
        if gold_symbol in close_df.columns:
            gold_close = close_df[gold_symbol]
            gold_ma200 = gold_close.rolling(window=200, min_periods=1).mean()
            gold_signal = np.where(gold_close > gold_ma200, 1.0, -1.0)
            gold_signal = pd.Series(gold_signal, index=gold_close.index)
        else:
            # If Gold missing, use Neutral (0.0)
            gold_signal = pd.Series(0.0, index=market_close.index)
            
        # Composite Score
        composite = 0.4 * ma_signal + 0.4 * mom_signal + 0.2 * gold_signal
        
        # Position Ratios
        # Default 1.0
        position_ratios = pd.Series(1.0, index=close_df.index)
        # Apply defensive mode
        position_ratios[composite < self.extreme_threshold] = self.extreme_position
        
        return position_ratios
