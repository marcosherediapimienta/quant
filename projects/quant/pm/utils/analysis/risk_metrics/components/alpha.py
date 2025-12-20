import numpy as np
import pandas as pd
from typing import Dict
from .helpers import calculate_portfolio_returns, annualize_return

class AlphaCalculator:

    def __init__(self, annual_factor: float = 252.0):
        self.annual_factor = annual_factor
    
    def calculate(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        benchmark_returns: pd.Series,
        risk_free_rate: float,
        beta: float = None,
        ddof: int = 1
    ) -> Dict[str, float]:

        portfolio_ret = calculate_portfolio_returns(returns, weights)
        df = pd.DataFrame({
            'portfolio': portfolio_ret,
            'benchmark': benchmark_returns
        }).dropna()
        
        if df.empty:
            return {
                'alpha': np.nan,
                'alpha_annual': np.nan,
                'portfolio_return_annual': np.nan,
                'benchmark_return_annual': np.nan,
                'beta_used': np.nan
            }

        portfolio_return_annual = annualize_return(df['portfolio'], self.annual_factor)
        benchmark_return_annual = annualize_return(df['benchmark'], self.annual_factor)

        if beta is None:
            cov_matrix = np.cov(df['portfolio'], df['benchmark'], ddof=ddof)
            beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else 1.0

        expected_return = risk_free_rate + beta * (benchmark_return_annual - risk_free_rate)
        alpha_annual = float(portfolio_return_annual - expected_return)
        alpha_daily = alpha_annual / self.annual_factor
        
        return {
            'alpha': alpha_daily,
            'alpha_annual': alpha_annual,
            'portfolio_return_annual': float(portfolio_return_annual),
            'benchmark_return_annual': float(benchmark_return_annual),
            'expected_return': float(expected_return),
            'beta_used': float(beta)
        }