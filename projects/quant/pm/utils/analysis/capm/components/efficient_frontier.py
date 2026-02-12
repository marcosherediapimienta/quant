import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import List, Optional
from ....tools.config import FRONTIER_POINTS, OPTIMIZATION_METHOD, ANNUAL_FACTOR

@dataclass
class FrontierResult:
    returns: np.ndarray
    volatilities: np.ndarray
    weights: np.ndarray
    assets: List[str]
    min_return: Optional[float] = None  
    max_return: Optional[float] = None 
    optimization_failures: int = 0      

class EfficientFrontierCalculator:
    def __init__(self, annual_factor: float = None):
        self.annual_factor = annual_factor if annual_factor else ANNUAL_FACTOR
    
    def calculate(
        self,
        returns: pd.DataFrame,
        n_points: int = None,
        allow_short: bool = False
    ) -> FrontierResult:

        n_points = n_points if n_points else FRONTIER_POINTS

        if returns.empty or len(returns.columns) < 2:
            return FrontierResult(
                np.array([]), np.array([]), np.array([]), [],
                None, None, 0
            )
        
        assets = list(returns.columns)
        returns_clean = returns.dropna()
        
        if returns_clean.empty:
            return FrontierResult(
                np.array([]), np.array([]), np.array([]), [],
                None, None, 0
            )
        
        mean_ret = returns_clean.mean() * self.annual_factor
        cov_matrix = returns_clean.cov() * self.annual_factor

        if mean_ret.isna().any():
            mean_ret = mean_ret.fillna(0)
        
        n = len(assets)
        bounds = tuple((-1, 1) if allow_short else (0, 1) for _ in range(n))
        
        def portfolio_variance(w):
            return float(w.T @ cov_matrix.values @ w)
        
        def portfolio_return(w):
            return float(np.sum(w * mean_ret.values))

        min_var_ret, min_var_vol, min_var_weights = self.minimum_variance_portfolio(
            returns, allow_short
        )
        
        max_ret_portfolio = self._maximum_return_portfolio(
            mean_ret, cov_matrix, bounds
        )

        if not np.isnan(min_var_ret) and max_ret_portfolio is not None:
            min_achievable = min_var_ret
            max_achievable = max_ret_portfolio['return']
        else:
            min_achievable = mean_ret.min()
            max_achievable = mean_ret.max()

        margin = (max_achievable - min_achievable) * 0.01
        targets = np.linspace(
            min_achievable + margin,
            max_achievable - margin,
            n_points
        )

        eff_returns = []
        eff_volatilities = []
        eff_weights = []
        failures = 0
        
        for target in targets:
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {"type": "eq", "fun": lambda w, t=target: portfolio_return(w) - t}
            ]

            x0 = np.full(n, 1.0 / n)
            
            try:
                result = minimize(
                    portfolio_variance,
                    x0=x0,
                    method=OPTIMIZATION_METHOD,
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 1000} 
                )

                if result.success and self._validate_portfolio(result.x, bounds):
                    w = result.x
                    eff_weights.append(w)
                    eff_returns.append(portfolio_return(w))
                    eff_volatilities.append(np.sqrt(portfolio_variance(w)))
                else:
                    failures += 1
                    
            except Exception as e:
                failures += 1
                continue
        
        return FrontierResult(
            np.array(eff_returns),
            np.array(eff_volatilities),
            np.array(eff_weights),
            assets,
            min_achievable,
            max_achievable,
            failures
        )
    
    def _maximum_return_portfolio(
        self,
        mean_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        bounds: tuple
    ) -> Optional[dict]:

        n = len(mean_returns)
        
        def neg_return(w):
            return -float(np.sum(w * mean_returns.values))
        
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        best_asset_idx = np.argmax(mean_returns.values)
        x0 = np.zeros(n)
        x0[best_asset_idx] = 1.0
        
        try:
            result = minimize(
                neg_return,
                x0=x0,
                method=OPTIMIZATION_METHOD,
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                w = result.x
                ret = float(np.sum(w * mean_returns.values))
                vol = float(np.sqrt(w.T @ cov_matrix.values @ w))
                
                return {
                    'return': ret,
                    'volatility': vol,
                    'weights': w
                }
        except Exception:
            pass
        
        return None
    
    def _validate_portfolio(self, weights: np.ndarray, bounds: tuple) -> bool:

        if not np.all(np.isfinite(weights)):
            return False

        if not np.isclose(np.sum(weights), 1.0, atol=0.01):
            return False

        for w, (lower, upper) in zip(weights, bounds):
            if w < lower - 1e-6 or w > upper + 1e-6:
                return False

        return True
    
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
            method=OPTIMIZATION_METHOD,
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            w = result.x
            ret = float(np.sum(w * mean_ret.values))
            vol = np.sqrt(portfolio_variance(w))
            return ret, vol, w
        
        return np.nan, np.nan, np.array([])