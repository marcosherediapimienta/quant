import numpy as np
import pandas as pd
from typing import List, Dict
from scipy.optimize import minimize
from ....tools.config import PORTFOLIO_CONFIG

class WeightOptimizer:
    def __init__(
        self,
        risk_free_rate: float = 0.0,
        annual_trading_days: int = 0,
        min_data_points: int = 0,
        scipy_method: str = '',
        max_weight: float = 1.0  
    ):
        opt_config = PORTFOLIO_CONFIG['optimization']
        self.risk_free_rate = risk_free_rate if risk_free_rate > 0 else opt_config['risk_free_rate']
        self.annual_trading_days = annual_trading_days if annual_trading_days > 0 else opt_config['annual_trading_days']
        self.min_data_points = min_data_points if min_data_points > 0 else opt_config['min_data_points']
        self.scipy_method = scipy_method if scipy_method else opt_config['scipy_method']
        self.max_weight = max_weight
    
    def optimize(
        self,
        tickers: List[str],
        method: str = '',
        returns_data: pd.DataFrame = None,
        analysis_results: Dict = None
    ) -> Dict[str, float]:

        if not tickers:
            raise ValueError("La lista de tickers no puede estar vacía")

        if not method:
            method = PORTFOLIO_CONFIG['optimization']['default_method']
        
        if method == 'equal':
            return self._equal_weights(tickers)
        elif method == 'score':
            if analysis_results is None:
                return self._equal_weights(tickers)
            return self._score_weights(tickers, analysis_results)
        elif method == 'score_risk_adjusted':
            if analysis_results is None or returns_data is None or returns_data.empty:
                return self._equal_weights(tickers)
            return self._score_risk_adjusted_weights(tickers, analysis_results, returns_data)
        elif method == 'markowitz':
            if returns_data is None or returns_data.empty:
                return self._equal_weights(tickers)
            return self._markowitz_weights(tickers, returns_data)
        else:
            return self._equal_weights(tickers)
    
    def _equal_weights(self, tickers: List[str]) -> Dict[str, float]:

        if not tickers:
            raise ValueError("La lista de tickers no puede estar vacía")
        w = 1.0 / len(tickers)
        return {t: w for t in tickers}
    
    def _score_weights(self, tickers: List[str], results: Dict) -> Dict[str, float]:
        raw = np.array([
            float(results.get(t, {}).get('scores', {}).get('total', 0.0))
            for t in tickers
        ], dtype=float)
        
        raw = np.maximum(raw, 0.0)

        s = raw.sum()
        if s <= 0:
            return self._equal_weights(tickers)

        w = raw / s
        return {t: float(wi) for t, wi in zip(tickers, w)}
    
    def _score_risk_adjusted_weights(
        self, 
        tickers: List[str], 
        results: Dict, 
        returns: pd.DataFrame
    ) -> Dict[str, float]:

        available_tickers = [t for t in tickers if t in returns.columns]
        if not available_tickers:
            return self._equal_weights(tickers)
        
        r = returns[available_tickers].dropna()
        if len(r) < self.min_data_points:
            return self._equal_weights(tickers)

        vol = r.std() * np.sqrt(self.annual_trading_days)
        vol = vol.replace(0, np.nan).fillna(vol.mean())

        raw_scores = np.array([
            float(results.get(t, {}).get('scores', {}).get('total', 0.0))
            for t in available_tickers
        ], dtype=float)
        raw_scores = np.maximum(raw_scores, 0.0)

        raw = raw_scores / vol.values
        s = raw.sum()
        
        if s <= 0:
            return self._equal_weights(available_tickers)
        
        w = raw / s
        weights = {t: float(wi) for t, wi in zip(available_tickers, w)}

        for t in tickers:
            if t not in weights:
                weights[t] = 0.0
        
        return weights
    
    def _markowitz_weights(self, tickers: List[str], returns: pd.DataFrame) -> Dict[str, float]:

        try:
            available_tickers = [t for t in tickers if t in returns.columns]
            if not available_tickers:
                return self._equal_weights(tickers)
            
            returns_subset = returns[available_tickers].dropna()

            if len(returns_subset) < self.min_data_points:
                return self._equal_weights(tickers)

            mean_ret = returns_subset.mean() * self.annual_trading_days
            cov_matrix = returns_subset.cov() * self.annual_trading_days
            
            n = len(available_tickers)

            cov_matrix = cov_matrix.values
            cov_matrix = cov_matrix + np.eye(n) * 1e-6

            def neg_sharpe(w):
                ret = np.dot(w, mean_ret.values)
                vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
                return -(ret - self.risk_free_rate) / vol if vol > 1e-10 else 1e10

            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = tuple((0, self.max_weight) for _ in range(n))

            result = minimize(
                neg_sharpe,
                x0=np.array([1/n] * n),
                method=self.scipy_method,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'disp': False}
            )
            
            if result.success:
                w = np.clip(result.x, 0.0, 1.0)
                w = w / w.sum() if w.sum() > 0 else np.array([1/n] * n)
                weights = {t: float(wi) for t, wi in zip(available_tickers, w)}

                for t in tickers:
                    if t not in weights:
                        weights[t] = 0.0
                
                return weights
            else:
                return self._equal_weights(tickers)
                
        except Exception as e:
            return self._equal_weights(tickers)