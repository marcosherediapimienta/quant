import numpy as np
import pandas as pd
from typing import Dict
from .helpers import calculate_portfolio_returns

class BetaCalculator:

    def __init__(self):
        pass
    
    def calculate(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        benchmark_returns: pd.Series,
        ddof: int = 1
    ) -> Dict[str, float]:

        portfolio_ret = calculate_portfolio_returns(returns, weights)
        df = pd.DataFrame({
            'portfolio': portfolio_ret,
            'benchmark': benchmark_returns
        }).dropna()
        
        if df.empty or len(df) < 2:
            return {
                'beta': np.nan,
                'r_squared': np.nan,
                'correlation': np.nan
            }

        cov_matrix = np.cov(df['portfolio'], df['benchmark'], ddof=ddof)
        cov_pb = cov_matrix[0, 1]
        var_b = cov_matrix[1, 1]
        beta = float(cov_pb / var_b) if var_b > 0 else np.nan
        corr = float(df['portfolio'].corr(df['benchmark']))
        r_squared = float(corr ** 2) if not np.isnan(corr) else np.nan
        
        return {
            'beta': beta,
            'r_squared': r_squared,
            'correlation': corr
        }
    
    def calculate_rolling(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        benchmark_returns: pd.Series,
        window: int = 252,
        ddof: int = 1
    ) -> pd.Series:

        portfolio_ret = calculate_portfolio_returns(returns, weights)

        df = pd.DataFrame({
            'portfolio': portfolio_ret,
            'benchmark': benchmark_returns
        }).dropna()
        
        if df.empty or len(df) < window:
            return pd.Series(dtype=float)

        def calculate_beta_for_window(window_df):
            """Calcula beta para una ventana del DataFrame"""
            if len(window_df) < 2:
                return np.nan
            
            p = window_df['portfolio'].values
            b = window_df['benchmark'].values
            
            if len(p) < 2 or len(b) < 2:
                return np.nan
            
            try:
                cov_matrix = np.cov(p, b, ddof=ddof)
                cov_pb = cov_matrix[0, 1]
                var_b = cov_matrix[1, 1]
                return cov_pb / var_b if var_b > 0 else np.nan
            except:
                return np.nan

        beta_rolling = df.rolling(window=window).apply(
            lambda x: calculate_beta_for_window(df.loc[x.index]),
            raw=False
        )['portfolio']
        
        return beta_rolling