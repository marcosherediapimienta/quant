import numpy as np
import pandas as pd
from scipy.stats import jarque_bera, anderson, norm
from typing import Dict, List
from .helpers import calculate_portfolio_returns
from ....tools.config import SIGNIFICANCE_LEVEL, ANDERSON_DARLING

_AD = ANDERSON_DARLING

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
        weights: np.ndarray,
        alpha: float = None
    ) -> Dict[str, float]:

        alpha = alpha if alpha is not None else SIGNIFICANCE_LEVEL
        portfolio_ret = calculate_portfolio_returns(returns, weights)
        jb_stat, p_value = jarque_bera(portfolio_ret)
        
        return {
            'jb_statistic': float(jb_stat),
            'p_value': float(p_value),
            'is_normal': bool(p_value > alpha)  
        }
    
    def calculate_anderson_darling(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        significance_level: float = None
    ) -> Dict[str, float]:

        significance_level = significance_level if significance_level is not None else SIGNIFICANCE_LEVEL
        
        portfolio_ret = calculate_portfolio_returns(returns, weights)
        
        try:
            result = anderson(portfolio_ret, dist='norm', method='interpolate')
            ad_statistic = float(result.statistic)
            p_value = float(result.pvalue)
            is_normal = p_value > significance_level
            critical_value = _AD['critical_value_5pct']

        except TypeError:
            result = anderson(portfolio_ret, dist='norm')
            ad_statistic = float(result.statistic)
            p_value = None
            critical_values = result.critical_values
            sig_levels = result.significance_level
            
            target_pct = significance_level * 100
            idx = min(range(len(sig_levels)), key=lambda i: abs(sig_levels[i] - target_pct))
            critical_value = float(critical_values[idx])
            is_normal = ad_statistic < critical_value

        severity_ratio = ad_statistic / critical_value if critical_value > 0 else float('inf')
        
        if severity_ratio < 1.0:
            tail_risk = "LOW"
        elif severity_ratio < _AD['severity_moderate']:
            tail_risk = "MODERATE"
        elif severity_ratio < _AD['severity_high']:
            tail_risk = "HIGH"
        else:
            tail_risk = "SEVERE"
        
        result_dict = {
            'ad_statistic': ad_statistic,
            'critical_value': float(critical_value),
            'significance_level': significance_level,
            'is_normal': is_normal,
            'severity_ratio': float(severity_ratio),
            'tail_risk': tail_risk,
        }
        
        if p_value is not None:
            result_dict['p_value'] = p_value
        
        return result_dict
    
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
        
        ad_results = self.calculate_anderson_darling(returns, weights)
        
        histogram = self._build_histogram(portfolio_ret, mean, std, returns)

        per_ticker = {}
        if len(returns.columns) > 1:
            for ticker in returns.columns:
                col = returns[ticker].dropna()
                per_ticker[ticker] = {
                    'mean': round(float(col.mean()), 6),
                    'std': round(float(col.std(ddof=0)), 6),
                    'skewness': round(float(col.skew()), 4),
                    'excess_kurtosis': round(float(col.kurtosis()), 4),
                }

        return {
            'mean': mean,
            'median': median,
            'std': std,
            'skewness': skew,
            'excess_kurtosis': excess_kurt,
            'jb_statistic': jb_results['jb_statistic'],
            'jb_p_value': jb_results['p_value'],
            'is_normal': jb_results['is_normal'],
            'ad_statistic': ad_results['ad_statistic'],
            'ad_critical_value': ad_results['critical_value'],
            'ad_is_normal': ad_results['is_normal'],
            'ad_tail_risk': ad_results['tail_risk'],
            'percentile_1': p1,
            'percentile_5': p5,
            'percentile_95': p95,
            'percentile_99': p99,
            'histogram': histogram,
            'per_ticker': per_ticker,
        }

    @staticmethod
    def _build_histogram(
        portfolio_ret: pd.Series,
        mu: float,
        sigma: float,
        returns_df: pd.DataFrame,
        bins: int = 50,
    ) -> List[Dict[str, float]]:
        counts, edges = np.histogram(portfolio_ret, bins=bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        normal_pdf = norm.pdf(centers, mu, sigma)

        result = []
        tickers = list(returns_df.columns) if len(returns_df.columns) > 1 else []

        ticker_densities = {}
        for ticker in tickers:
            t_counts, _ = np.histogram(returns_df[ticker].dropna(), bins=edges, density=True)
            ticker_densities[ticker] = t_counts

        for i, (c, d, n) in enumerate(zip(centers, counts, normal_pdf)):
            point = {
                'x': round(float(c) * 100, 4),
                'density': round(float(d), 4),
                'normal': round(float(n), 4),
            }
            for ticker in tickers:
                point[ticker] = round(float(ticker_densities[ticker][i]), 4)
            result.append(point)

        return result