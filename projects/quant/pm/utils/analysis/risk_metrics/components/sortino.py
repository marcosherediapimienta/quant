import numpy as np
import pandas as pd
from typing import Optional
from .helpers import calculate_portfolio_returns, annualize_return


class SortinoCalculator:

    def __init__(self, annual_factor: float = 252.0):
        self.annual_factor = annual_factor
    
    def calculate(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        risk_free_rate: float,
        target_return: Optional[float] = None,
        ddof: int = 0
    ) -> float:

        portfolio_ret = calculate_portfolio_returns(returns, weights)
        
        if target_return is None:
            target_return = 0.0
        
        annual_return = annualize_return(portfolio_ret, self.annual_factor)
        downside_returns = portfolio_ret[portfolio_ret < target_return]
        
        if len(downside_returns) == 0:
            return np.inf if annual_return > risk_free_rate else np.nan
        
        downside_vol = downside_returns.std(ddof=ddof) * np.sqrt(self.annual_factor)
        
        if downside_vol == 0:
            return np.nan
        
        sortino = (annual_return - risk_free_rate) / downside_vol
        return float(sortino) if np.isfinite(sortino) else None
    
    def calculate_rolling(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        risk_free_rate: float,
        window: int = 252,
        target_return: Optional[float] = None,
        ddof: int = 0
    ) -> pd.Series:

        portfolio_ret = calculate_portfolio_returns(returns, weights)
        
        if target_return is None:
            target_return = 0.0
        
        def rolling_sortino(window_data):
            if len(window_data) < 2:
                return np.nan
            ann_ret = window_data.mean() * self.annual_factor
            downside = window_data[window_data < target_return]
            if len(downside) == 0:
                return np.inf if ann_ret > risk_free_rate else np.nan
            d_vol = downside.std(ddof=ddof) * np.sqrt(self.annual_factor)
            if d_vol == 0:
                return np.nan
            return (ann_ret - risk_free_rate) / d_vol
        
        return portfolio_ret.rolling(window).apply(rolling_sortino, raw=False)