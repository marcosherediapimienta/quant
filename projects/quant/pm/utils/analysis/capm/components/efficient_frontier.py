import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import List


@dataclass
class FrontierResult:
    returns: np.ndarray
    volatilities: np.ndarray
    weights: np.ndarray
    assets: List[str]


class EfficientFrontierCalculator:

    def __init__(self, annual_factor: float = 252.0):
        self.annual_factor = annual_factor
    
    def calculate(
        self,
        returns: pd.DataFrame,
        n_points: int = 60,
        allow_short: bool = False
    ) -> FrontierResult:

        if returns.empty or len(returns.columns) < 2:
            return FrontierResult(np.array([]), np.array([]), np.array([]), [])
        
        assets = list(returns.columns)
        mean_ret = returns.mean() * self.annual_factor
        cov_matrix = returns.cov() * self.annual_factor
        
        n = len(assets)
        bounds = tuple((-1, 1) if allow_short else (0, 1) for _ in range(n))
        
        def portfolio_variance(w):
            return float(w.T @ cov_matrix.values @ w)
        
        def portfolio_return(w):
            return float(np.sum(w * mean_ret.values))
        
        # Rango de retornos objetivo
        targets = np.linspace(mean_ret.min(), mean_ret.max(), n_points)
        
        eff_returns = []
        eff_volatilities = []
        eff_weights = []
        
        for target in targets:
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {"type": "eq", "fun": lambda w, t=target: portfolio_return(w) - t}
            ]
            
            x0 = np.full(n, 1.0 / n)
            result = minimize(
                portfolio_variance,
                x0=x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                w = result.x
                eff_weights.append(w)
                eff_returns.append(portfolio_return(w))
                eff_volatilities.append(np.sqrt(portfolio_variance(w)))
        
        return FrontierResult(
            np.array(eff_returns),
            np.array(eff_volatilities),
            np.array(eff_weights),
            assets
        )
    
    def minimum_variance_portfolio(
        self,
        returns: pd.DataFrame,
        allow_short: bool = False
    ) -> tuple:

        if returns.empty:
            return np.nan, np.nan, np.array([])
        
        mean_ret = returns.mean() * self.annual_factor
        cov_matrix = returns.cov() * self.annual_factor
        n = len(returns.columns)
        
        bounds = tuple((-1, 1) if allow_short else (0, 1) for _ in range(n))
        
        def portfolio_variance(w):
            return float(w.T @ cov_matrix.values @ w)
        
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        
        result = minimize(
            portfolio_variance,
            x0=np.full(n, 1.0 / n),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            w = result.x
            ret = float(np.sum(w * mean_ret.values))
            vol = np.sqrt(portfolio_variance(w))
            return ret, vol, w
        
        return np.nan, np.nan, np.array([])