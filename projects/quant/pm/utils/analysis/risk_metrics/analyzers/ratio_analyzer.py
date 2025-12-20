import numpy as np
import pandas as pd
from typing import Dict
from ..components.sharpe import SharpeCalculator
from ..components.sortino import SortinoCalculator
from ..components.helpers import (
    calculate_portfolio_returns,
    annualize_return,
    annualize_volatility
)

class RatioAnalyzer:

    def __init__(self, annual_factor: float = 252.0):
        self.annual_factor = annual_factor
        self.sharpe_calc = SharpeCalculator(annual_factor)
        self.sortino_calc = SortinoCalculator(annual_factor)
    
    def calculate_all_ratios(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        risk_free_rate: float,
        ddof: int = 0
    ) -> Dict[str, float]:

        portfolio_ret = calculate_portfolio_returns(returns, weights)
        annual_return = annualize_return(portfolio_ret, self.annual_factor)
        annual_vol = annualize_volatility(portfolio_ret, self.annual_factor, ddof)
        excess_return = annual_return - risk_free_rate
        downside_returns = portfolio_ret[portfolio_ret < 0]
        if len(downside_returns) > 0:
            downside_vol = downside_returns.std(ddof=ddof) * np.sqrt(self.annual_factor)
        else:
            downside_vol = 0.0
        
        sharpe = self.sharpe_calc.calculate(returns, weights, risk_free_rate, ddof)
        sortino = self.sortino_calc.calculate(returns, weights, risk_free_rate, ddof=ddof)
        
        return {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'annual_return': float(annual_return),
            'annual_volatility': float(annual_vol),
            'downside_volatility': float(downside_vol),
            'excess_return': float(excess_return),
            'risk_free_rate': risk_free_rate
        }
    
    def calculate_rolling(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        risk_free_rate: float,
        window: int = 252,
        ddof: int = 0
    ) -> pd.DataFrame:

        sharpe_rolling = self.sharpe_calc.calculate_rolling(
            returns, weights, risk_free_rate, window, ddof
        )
        
        sortino_rolling = self.sortino_calc.calculate_rolling(
            returns, weights, risk_free_rate, window, ddof=ddof
        )
        
        return pd.DataFrame({
            'sharpe_rolling': sharpe_rolling,
            'sortino_rolling': sortino_rolling
        })