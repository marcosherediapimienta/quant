import numpy as np
import pandas as pd
from scipy import stats
from typing import Literal, Dict
from .helpers import calculate_portfolio_returns
from ....tools.config import DEFAULT_CONFIDENCE_LEVEL
from ....tools.config import DEFAULT_CONFIDENCE_LEVEL, MONTE_CARLO_SIMULATIONS, MONTE_CARLO_SEED

class VaRCalculator:

    def __init__(self, annual_factor: float = 252.0):
        self.annual_factor = annual_factor
    
    def calculate_historical(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        confidence_level: float = DEFAULT_CONFIDENCE_LEVEL
    ) -> Dict[str, float]:

        portfolio_ret = calculate_portfolio_returns(returns, weights)
        alpha = 1.0 - confidence_level
        
        var_daily = np.quantile(portfolio_ret, alpha)
        var_annual = var_daily * np.sqrt(self.annual_factor)
        
        return {
            'var_daily': float(var_daily),
            'var_annual': float(var_annual),
            'var_daily_pct': float(var_daily * 100),
            'var_annual_pct': float(var_annual * 100)
        }
    
    def calculate_parametric(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
 
        portfolio_ret = calculate_portfolio_returns(returns, weights)
        alpha = 1.0 - confidence_level
        
        mu = portfolio_ret.mean()
        sigma = portfolio_ret.std(ddof=0)
        z = stats.norm.ppf(alpha)

        var_daily = mu + sigma * z
        var_annual = var_daily * np.sqrt(self.annual_factor)
        
        return {
            'var_daily': float(var_daily),
            'var_annual': float(var_annual),
            'var_daily_pct': float(var_daily * 100),
            'var_annual_pct': float(var_annual * 100)
        }
    
    def calculate_monte_carlo(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
        n_simulations: int = MONTE_CARLO_SIMULATIONS,
        seed: int = MONTE_CARLO_SEED
    ) -> Dict[str, float]:

        portfolio_ret = calculate_portfolio_returns(returns, weights)
        alpha = 1.0 - confidence_level
        
        mu = portfolio_ret.mean()
        sigma = portfolio_ret.std(ddof=0)

        rng = np.random.default_rng(seed)
        simulations = rng.normal(mu, sigma, n_simulations)
        
        var_daily = np.quantile(simulations, alpha)
        var_annual = var_daily * np.sqrt(self.annual_factor)
        
        return {
            'var_daily': float(var_daily),
            'var_annual': float(var_annual),
            'var_daily_pct': float(var_daily * 100),
            'var_annual_pct': float(var_annual * 100)
        }
    
    def calculate(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        confidence_level: float = 0.95,
        method: Literal['historical', 'parametric', 'monte_carlo'] = 'historical',
        **kwargs
    ) -> Dict[str, float]:

        if method == 'historical':
            return self.calculate_historical(returns, weights, confidence_level)
        elif method == 'parametric':
            return self.calculate_parametric(returns, weights, confidence_level)
        elif method == 'monte_carlo':
            return self.calculate_monte_carlo(returns, weights, confidence_level, **kwargs)
        else:
            raise ValueError(
                f"Método '{method}' no válido. "
                f"Usa: 'historical', 'parametric' o 'monte_carlo'"
            )