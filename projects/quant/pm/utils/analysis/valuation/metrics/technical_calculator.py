import pandas as pd
import numpy as np

class TechnicalCalculator:

    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> float:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if len(rsi) > 0 else np.nan
    
    def calculate_momentum(self, prices: pd.Series) -> dict:

        if len(prices) < 63:
            return {'1m': np.nan, '3m': np.nan}
        
        returns_1m = (prices.iloc[-1] / prices.iloc[-21] - 1) * 100 if len(prices) >= 21 else np.nan
        returns_3m = (prices.iloc[-1] / prices.iloc[-63] - 1) * 100 if len(prices) >= 63 else np.nan
        
        return {'1m': returns_1m, '3m': returns_3m}