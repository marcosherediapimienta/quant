import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Literal
from .helpers import calculate_portfolio_returns
from ....tools.config import (
    ANNUAL_FACTOR,
    DEFAULT_CONFIDENCE_LEVEL,
    MONTE_CARLO_SIMULATIONS,
    MONTE_CARLO_SEED
)

class ESCalculator:
    def __init__(self, annual_factor: float = None):
        self.annual_factor = annual_factor if annual_factor is not None else ANNUAL_FACTOR
    
    def calculate_historical(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        confidence_level: float = None,
        var_value: float = None
    ) -> Dict[str, float]:

        confidence_level = confidence_level if confidence_level is not None else DEFAULT_CONFIDENCE_LEVEL
        
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
        confidence_level: float = None
    ) -> Dict[str, float]:

        confidence_level = confidence_level if confidence_level is not None else DEFAULT_CONFIDENCE_LEVEL
        
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
        confidence_level: float = None,
        n_simulations: int = None,
        seed: int = None,
        var_value: float = None
    ) -> Dict[str, float]:

        confidence_level = confidence_level if confidence_level is not None else DEFAULT_CONFIDENCE_LEVEL
        n_simulations = n_simulations if n_simulations is not None else MONTE_CARLO_SIMULATIONS
        seed = seed if seed is not None else MONTE_CARLO_SEED
        
        portfolio_ret = calculate_portfolio_returns(returns, weights)
        alpha = 1.0 - confidence_level
        mu = portfolio_ret.mean()
        sigma = portfolio_ret.std(ddof=0)
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
    
    def calculate(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        confidence_level: float = None,
        method: Literal['historical', 'parametric', 'monte_carlo'] = 'historical',
        **kwargs
    ) -> Dict[str, float]:

        confidence_level = confidence_level if confidence_level is not None else DEFAULT_CONFIDENCE_LEVEL

        strategies = {
            'historical': lambda: self.calculate_historical(
                returns, weights, confidence_level,
                var_value=kwargs.get('var_value'),
            ),
            'parametric': lambda: self.calculate_parametric(
                returns, weights, confidence_level,
            ),
            'monte_carlo': lambda: self.calculate_monte_carlo(
                returns, weights, confidence_level,
                n_simulations=kwargs.get('n_simulations'),
                seed=kwargs.get('seed'),
                var_value=kwargs.get('var_value'),
            ),
        }

        if method not in strategies:
            raise ValueError(
                f"Method '{method}' not valid. "
                f"Options: {list(strategies.keys())}"
            )

        return strategies[method]()
    
    def calculate_all_methods(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        confidence_level: float = None,
        n_simulations: int = None,
        seed: int = None
    ) -> Dict[str, Dict[str, float]]:

        confidence_level = confidence_level if confidence_level is not None else DEFAULT_CONFIDENCE_LEVEL
        n_simulations = n_simulations if n_simulations is not None else MONTE_CARLO_SIMULATIONS
        seed = seed if seed is not None else MONTE_CARLO_SEED
        
        return {
            'historical': self.calculate_historical(returns, weights, confidence_level),
            'parametric': self.calculate_parametric(returns, weights, confidence_level),
            'monte_carlo': self.calculate_monte_carlo(
                returns, weights, confidence_level, n_simulations, seed
            )
        }