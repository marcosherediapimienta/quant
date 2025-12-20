import numpy as np
import pandas as pd
from .helpers import calculate_portfolio_returns, annualize_return, annualize_volatility

class SharpeCalculator:

    def __init__(self, annual_factor: float = 252.0):
        self.annual_factor = annual_factor
    
    def calculate(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        risk_free_rate: float,
        ddof: int = 0
    ) -> float:

        portfolio_ret = calculate_portfolio_returns(returns, weights)
        annual_return = annualize_return(portfolio_ret, self.annual_factor)
        annual_vol = annualize_volatility(portfolio_ret, self.annual_factor, ddof)
        
        if annual_vol == 0:
            return np.nan
        
        sharpe = (annual_return - risk_free_rate) / annual_vol
        return float(sharpe) if np.isfinite(sharpe) else None
    
    def calculate_rolling(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        risk_free_rate: float,
        window: int = 252,
        ddof: int = 0
    ) -> pd.Series:
    
        portfolio_ret = calculate_portfolio_returns(returns, weights)
        mu_roll = portfolio_ret.rolling(window).mean() * self.annual_factor
        sigma_roll = portfolio_ret.rolling(window).std(ddof=ddof) * np.sqrt(self.annual_factor)
        sharpe_roll = (mu_roll - risk_free_rate) / (sigma_roll + 1e-12)
        return sharpe_roll