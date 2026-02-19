import numpy as np
import pandas as pd
from .helpers import calculate_portfolio_returns, annualize_return
from ....tools.config import ANNUAL_FACTOR, ROLLING_WINDOW

class SortinoCalculator:
    def __init__(self, annual_factor: float = None):
        self.annual_factor = annual_factor if annual_factor is not None else ANNUAL_FACTOR
    
    def calculate(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        risk_free_rate: float,
        mar: float = 0.0, 
        ddof: int = 0
    ) -> float:
  
        portfolio_ret = calculate_portfolio_returns(returns, weights)
        annual_return = annualize_return(portfolio_ret, self.annual_factor)
        daily_mar = mar / self.annual_factor
        downside_returns = portfolio_ret[portfolio_ret < daily_mar]
        
        if len(downside_returns) == 0:
            return np.inf
        
        downside_squared = (downside_returns - daily_mar) ** 2
        downside_vol = np.sqrt(downside_squared.mean()) * np.sqrt(self.annual_factor)
        
        if downside_vol == 0:
            return np.nan
        
        sortino = (annual_return - risk_free_rate) / downside_vol
        return float(sortino) if np.isfinite(sortino) else None
    
    def calculate_rolling(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        risk_free_rate: float,
        mar: float = 0.0,  
        window: int = None,
        ddof: int = 0
    ) -> pd.Series:

        window = window if window is not None else ROLLING_WINDOW
        
        portfolio_ret = calculate_portfolio_returns(returns, weights)
        daily_mar = mar / self.annual_factor
        
        def rolling_sortino(x):
            if len(x) < 2:
                return np.nan
            
            mu = x.mean() * self.annual_factor
            downside = x[x < daily_mar]
            
            if len(downside) == 0:
                return np.inf
    
            dd_squared = ((downside - daily_mar) ** 2).mean()
            dd = np.sqrt(dd_squared) * np.sqrt(self.annual_factor)
            
            return (mu - risk_free_rate) / dd if dd > 0 else np.nan
        
        return portfolio_ret.rolling(window).apply(rolling_sortino, raw=False)