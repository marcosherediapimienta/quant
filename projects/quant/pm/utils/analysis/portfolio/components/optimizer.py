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
        scipy_method: str = ''
    ):
        opt_config = PORTFOLIO_CONFIG['optimization']
        self.risk_free_rate = risk_free_rate if risk_free_rate > 0 else opt_config['risk_free_rate']
        self.annual_trading_days = annual_trading_days if annual_trading_days > 0 else opt_config['annual_trading_days']
        self.min_data_points = min_data_points if min_data_points > 0 else opt_config['min_data_points']
        self.scipy_method = scipy_method if scipy_method else opt_config['scipy_method']
    
    def optimize(
        self,
        tickers: List[str],
        method: str = '',
        returns_data: pd.DataFrame = None,
        analysis_results: Dict = None
    ) -> Dict[str, float]:

        if not method:
            method = PORTFOLIO_CONFIG['optimization']['default_method']
        
        if method == 'equal':
            return self._equal_weights(tickers)
        elif method == 'score':
            if analysis_results is None:
                return self._equal_weights(tickers)
            return self._score_weights(tickers, analysis_results)
        elif method == 'markowitz':
            if returns_data is None or returns_data.empty:
                return self._equal_weights(tickers)
            return self._markowitz_weights(tickers, returns_data)
        else:
            return self._equal_weights(tickers)
    
    def _equal_weights(self, tickers: List[str]) -> Dict[str, float]:
        w = 1.0 / len(tickers)
        return {t: w for t in tickers}
    
    def _score_weights(self, tickers: List[str], results: Dict) -> Dict[str, float]:
        scores = {t: results[t].get('scores', {}).get('total', 0) for t in tickers}
        total = sum(scores.values())
        
        if total == 0:
            return self._equal_weights(tickers)
        
        return {t: s / total for t, s in scores.items()}
    
    def _markowitz_weights(self, tickers: List[str], returns: pd.DataFrame) -> Dict[str, float]:
        returns_subset = returns[tickers].dropna()

        if len(returns_subset) < self.min_data_points:
            return self._equal_weights(tickers)
        
        mean_ret = returns_subset.mean() * self.annual_trading_days
        cov_matrix = returns_subset.cov() * self.annual_trading_days
        n = len(tickers)

        def neg_sharpe(w):
            ret = np.dot(w, mean_ret)
            vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            return -(ret - self.risk_free_rate) / vol if vol > 0 else 0

        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = tuple((0, 1) for _ in range(n))

        result = minimize(
            neg_sharpe,
            x0=np.array([1/n] * n),
            method=self.scipy_method,
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            return {t: float(w) for t, w in zip(tickers, result.x)}
        else:
            return self._equal_weights(tickers)