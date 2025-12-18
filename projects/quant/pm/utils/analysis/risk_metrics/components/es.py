import numpy as np
import pandas as pd
from scipy import stats
from typing import Literal, Dict
from .portfolio_helpers import calculate_portfolio_returns


class ESCalculator:

    def __init__(self, annual_factor: float = 252.0):
        self.annual_factor = annual_factor
    
    def calculate_historical(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        confidence_level: float = 0.95,
        var_value: float = None
    ) -> Dict[str, float]:

        portfolio_ret = calculate_portfolio_returns(returns, weights)
        alpha = 1.0 - confidence_level
        
        if var_value is None:
            var_value = np.quantile(portfolio_ret, alpha)

        tail_losses = portfolio_ret[portfolio_ret <= var_value]
        
        if len(tail_losses) == 0:
            es_daily = var_value
        else:
            es_daily = tail_losses.mean()
        
        es_annual = es_daily * np.sqrt(self.annual_factor)
        
        return {
            'es_daily': float(es_daily),
            'es_annual': float(es_annual),
            'es_daily_pct': float(es_daily * 100),
            'es_annual_pct': float(es_annual * 100)
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
        es_daily = mu - sigma * stats.norm.pdf(z) / alpha
        es_annual = es_daily * np.sqrt(self.annual_factor)
        
        return {
            'es_daily': float(es_daily),
            'es_annual': float(es_annual),
            'es_daily_pct': float(es_daily * 100),
            'es_annual_pct': float(es_annual * 100)
        }
    
    def calculate_monte_carlo(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        confidence_level: float = 0.95,
        n_simulations: int = 10000,
        seed: int = 42,
        var_value: float = None
    ) -> Dict[str, float]:

        portfolio_ret = calculate_portfolio_returns(returns, weights)
        alpha = 1.0 - confidence_level
        
        mu = portfolio_ret.mean()
        sigma = portfolio_ret.std(ddof=0)
        
        # Simulaciones
        rng = np.random.default_rng(seed)
        simulations = rng.normal(mu, sigma, n_simulations)
        
        if var_value is None:
            var_value = np.quantile(simulations, alpha)
        
        tail_losses = simulations[simulations <= var_value]
        es_daily = tail_losses.mean()
        es_annual = es_daily * np.sqrt(self.annual_factor)
        
        return {
            'es_daily': float(es_daily),
            'es_annual': float(es_annual),
            'es_daily_pct': float(es_daily * 100),
            'es_annual_pct': float(es_annual * 100)
        }
    
    def calculate_all(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        confidence_level: float = 0.95,
        n_simulations: int = 10000,
        seed: int = 42
    ) -> Dict[str, Dict[str, float]]:

        results = {
            'historical': self.calculate_historical(returns, weights, confidence_level),
            'parametric': self.calculate_parametric(returns, weights, confidence_level),
            'monte_carlo': self.calculate_monte_carlo(
                returns, weights, confidence_level, n_simulations, seed
            )
        }
        
        return results