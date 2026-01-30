from typing import Dict
import numpy as np
import pandas as pd
import statsmodels.api as sm
from dataclasses import dataclass
from ..tools.config import (
    MAX_LAG,
    CORRELATION_MIN_OBS,
    HAC_MAXLAGS,
    CORRELATION_LAGS_DEFAULT,
)

@dataclass
class CorrelationResult:
    lag: int
    corr: float
    t_stat: float
    p_value: float
    n_obs: int

@dataclass
class BestLagResult:
    factor: str
    lag: int
    corr: float
    t_stat: float
    p_value: float
    n_obs: int


class MacroCorrelationCalculator:
    def __init__(
        self,
        max_lag: int = None,
        min_obs: int = None,
        hac_maxlags: int = None
    ):
        self.max_lag = max_lag if max_lag is not None else MAX_LAG
        self.min_obs = min_obs if min_obs is not None else CORRELATION_MIN_OBS
        self.hac_maxlags = hac_maxlags if hac_maxlags is not None else HAC_MAXLAGS
    
    def calculate_lagged(
        self,
        y: pd.Series,
        x: pd.Series
    ) -> pd.DataFrame:

        results = []
        
        for lag in range(-self.max_lag, self.max_lag + 1):
            y_aligned = y.copy()
            x_lagged = x.shift(lag)
            
            df = pd.concat([y_aligned, x_lagged], axis=1).dropna()
            
            if len(df) < self.min_obs:
                continue
            
            corr = df.iloc[:, 0].corr(df.iloc[:, 1])

            try:
                X = sm.add_constant(df.iloc[:, 1].values)
                model = sm.OLS(df.iloc[:, 0].values, X)
                n = len(df)
                hac_lags = self.hac_maxlags if self.hac_maxlags is not None else int(np.sqrt(n))
                fit = model.fit(cov_type='HAC', cov_kwds={'maxlags': hac_lags})
                t_stat = float(fit.tvalues[1])
                p_value = float(fit.pvalues[1])
            except Exception:
                t_stat = np.nan
                p_value = np.nan
            
            results.append({
                'lag': lag,
                'corr': float(corr),
                't': t_stat,
                'p': p_value,
                'n': len(df)
            })
        
        return pd.DataFrame(results)
    
    def find_best_lag(
        self,
        portfolio_returns: pd.Series,
        macro_factors: pd.DataFrame
    ) -> pd.DataFrame:

        results = []
        
        for factor_name in macro_factors.columns:
            try:
                lag_df = self.calculate_lagged(
                    portfolio_returns,
                    macro_factors[factor_name]
                )
                
                if lag_df.empty:
                    continue

                significant_lags = lag_df[lag_df['p'] < 0.05].copy()
                
                if not significant_lags.empty:
                    idx_best = significant_lags['corr'].abs().idxmax()
                    best_row = significant_lags.loc[idx_best]
                    is_significant = True
                else:
                    idx_best = lag_df['corr'].abs().idxmax()
                    best_row = lag_df.loc[idx_best]
                    is_significant = False
                    print(f"⚠️  [Macro] {factor_name}: Ninguna correlación significativa (mejor p-value: {best_row['p']:.4f})")
                
                results.append({
                    'factor': factor_name,
                    'lag': int(best_row['lag']),
                    'corr': float(best_row['corr']),
                    't': float(best_row['t']),
                    'p': float(best_row['p']),
                    'n': int(best_row['n']),
                    'is_significant': is_significant
                })
                
            except Exception as e:
                print(f"[Macro] Error con factor {factor_name}: {e}")
        
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('corr', key=abs, ascending=False)
        
        return df
    
    def calculate_matrix_with_lags(
        self,
        portfolio_returns: pd.Series,
        macro_factors: pd.DataFrame,
        lags: list = None
    ) -> Dict[int, pd.Series]:

        if lags is None:
            lags = CORRELATION_LAGS_DEFAULT
        
        results = {}
        
        for lag in lags:
            corrs = {}
            for factor_name in macro_factors.columns:
                try:
                    x_lagged = macro_factors[factor_name].shift(lag)
                    df = pd.concat([portfolio_returns, x_lagged], axis=1).dropna()
                    
                    if len(df) >= self.min_obs:
                        corrs[factor_name] = df.iloc[:, 0].corr(df.iloc[:, 1])
                        
                except Exception:
                    corrs[factor_name] = np.nan
            
            results[lag] = pd.Series(corrs)
        
        return results
    
    def calculate_rolling(
        self,
        portfolio_returns: pd.Series,
        macro_factor: pd.Series,
        window: int = 252,
        min_periods: int = None
    ) -> pd.Series:

        if min_periods is None:
            min_periods = window // 2

        df = pd.concat([portfolio_returns, macro_factor], axis=1).dropna()
        rolling_corr = df.iloc[:, 0].rolling(
            window=window,
            min_periods=min_periods
        ).corr(df.iloc[:, 1])
        
        return rolling_corr