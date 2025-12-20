import numpy as np
import pandas as pd
from scipy.stats import jarque_bera
from typing import Dict
from .helpers import calculate_portfolio_returns

class DistributionMoments:

    def __init__(self):
        pass
    
    def calculate_skewness(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray
    ) -> float:

        portfolio_ret = calculate_portfolio_returns(returns, weights)
        skew = portfolio_ret.skew()
        return float(skew)
    
    def calculate_kurtosis(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        excess: bool = True
    ) -> float:
        portfolio_ret = calculate_portfolio_returns(returns, weights)
        
        if excess:
            kurt = portfolio_ret.kurtosis()
        else:
            kurt = portfolio_ret.kurtosis() + 3.0
        
        return float(kurt)
    
    def calculate_jarque_bera(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray
    ) -> Dict[str, float]:
    
        portfolio_ret = calculate_portfolio_returns(returns, weights)
        jb_stat, p_value = jarque_bera(portfolio_ret)
        
        return {
            'jb_statistic': float(jb_stat),
            'p_value': float(p_value),
            'is_normal': bool(p_value > 0.05)
        }
    
    def calculate_all(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray
    ) -> Dict[str, float]:

        portfolio_ret = calculate_portfolio_returns(returns, weights)
        mean = float(portfolio_ret.mean())
        std = float(portfolio_ret.std(ddof=0))
        median = float(portfolio_ret.median())
        skew = self.calculate_skewness(returns, weights)
        excess_kurt = self.calculate_kurtosis(returns, weights, excess=True)
        jb_results = self.calculate_jarque_bera(returns, weights)
        p1 = float(portfolio_ret.quantile(0.01))
        p5 = float(portfolio_ret.quantile(0.05))
        p95 = float(portfolio_ret.quantile(0.95))
        p99 = float(portfolio_ret.quantile(0.99))
        
        return {
            'mean': mean,
            'median': median,
            'std': std,
            'skewness': skew,
            'excess_kurtosis': excess_kurt,
            'jb_statistic': jb_results['jb_statistic'],
            'jb_p_value': jb_results['p_value'],
            'is_normal': jb_results['is_normal'],
            'percentile_1': p1,
            'percentile_5': p5,
            'percentile_95': p95,
            'percentile_99': p99
        }