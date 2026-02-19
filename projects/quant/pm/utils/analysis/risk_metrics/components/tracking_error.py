import numpy as np
import pandas as pd
from typing import Dict
from .helpers import calculate_portfolio_returns
from ....tools.config import ANNUAL_FACTOR, ROLLING_WINDOW, TRACKING_ERROR_DEFAULTS

_TE = TRACKING_ERROR_DEFAULTS

class TrackingErrorCalculator:
    def __init__(self, annual_factor: float = None):
        self.annual_factor = annual_factor if annual_factor is not None else ANNUAL_FACTOR
    
    def calculate(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        benchmark_returns: pd.Series,
        ddof: int = 1
    ) -> Dict[str, float]:

        portfolio_ret = calculate_portfolio_returns(returns, weights)
        df = pd.DataFrame({
            'portfolio': portfolio_ret,
            'benchmark': benchmark_returns
        }).dropna()
        
        if df.empty:
            return {
                'te_daily': np.nan,
                'te_annual': np.nan,
                'information_ratio': np.nan,
                'excess_return_annual': np.nan
            }

        active_returns = df['portfolio'] - df['benchmark']
        te_daily = float(active_returns.std(ddof=ddof))
        te_annual = float(te_daily * np.sqrt(self.annual_factor))
        excess_return_annual = float(active_returns.mean() * self.annual_factor)
        info_ratio = float(excess_return_annual / te_annual) if te_annual > 0 else np.nan
        
        return {
            'te_daily': te_daily,
            'te_annual': te_annual,
            'excess_return_annual': excess_return_annual,
            'information_ratio': info_ratio
        }
    
    def calculate_rolling(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        benchmark_returns: pd.Series,
        window: int = None, 
        ddof: int = 1
    ) -> pd.Series:

        if window is None:
            window = max(_TE['min_rolling_window'], ROLLING_WINDOW // _TE['rolling_window_divisor'])
        
        portfolio_ret = calculate_portfolio_returns(returns, weights)
        df = pd.DataFrame({
            'portfolio': portfolio_ret,
            'benchmark': benchmark_returns
        }).dropna()
        
        if df.empty or len(df) < window:
            return pd.Series(dtype=float)

        active_returns = df['portfolio'] - df['benchmark']
        te_rolling = active_returns.rolling(window=window).std(ddof=ddof) * np.sqrt(self.annual_factor)
        
        return te_rolling