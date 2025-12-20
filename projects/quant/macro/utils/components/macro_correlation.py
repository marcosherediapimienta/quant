from typing import Dict
import numpy as np
import pandas as pd
import statsmodels.api as sm
from ..tools.config import (
    MAX_LAG,
    CORRELATION_MIN_OBS,
    HAC_MAXLAGS,
)

def lagged_correlation(
    y: pd.Series,
    x: pd.Series,
    max_lag: int = None,
    min_obs: int = None
) -> pd.DataFrame:

    if max_lag is None:
        max_lag = MAX_LAG
    if min_obs is None:
        min_obs = CORRELATION_MIN_OBS
    
    results = []
    
    for lag in range(-max_lag, max_lag + 1):
        y_aligned = y.copy()
        x_lagged = x.shift(lag)
        
        df = pd.concat([y_aligned, x_lagged], axis=1).dropna()
        
        if len(df) < min_obs:
            continue
        
        corr = df.iloc[:, 0].corr(df.iloc[:, 1])
        
        try:
            X = sm.add_constant(df.iloc[:, 1].values)
            model = sm.OLS(df.iloc[:, 0].values, X)
            n = len(df)
            hac_lags = HAC_MAXLAGS if HAC_MAXLAGS is not None else int(np.sqrt(n))
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

def best_lagged_correlation(
    portfolio_returns: pd.Series,
    macro_factors: pd.DataFrame,
    max_lag: int = None,
    min_obs: int = None
) -> pd.DataFrame:

    results = []
    
    for factor_name in macro_factors.columns:
        try:
            lag_df = lagged_correlation(
                portfolio_returns,
                macro_factors[factor_name],
                max_lag,
                min_obs
            )
            
            if lag_df.empty:
                continue
            
            idx_best = lag_df['corr'].abs().idxmax()
            best_row = lag_df.loc[idx_best]
            
            results.append({
                'factor': factor_name,
                'lag': int(best_row['lag']),
                'corr': float(best_row['corr']),
                't': float(best_row['t']),
                'p': float(best_row['p']),
                'n': int(best_row['n'])
            })
            
        except Exception as e:
            print(f"[Macro] Error con factor {factor_name}: {e}")
    
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values('corr', key=abs, ascending=False)
    
    return df

def correlation_matrix_with_lags(
    portfolio_returns: pd.Series,
    macro_factors: pd.DataFrame,
    lags: list = None
) -> Dict[int, pd.Series]:

    if lags is None:
        lags = [0, 1, 5, 21, 63, 126]
    
    results = {}
    
    for lag in lags:
        corrs = {}
        for factor_name in macro_factors.columns:
            try:
                x_lagged = macro_factors[factor_name].shift(lag)
                df = pd.concat([portfolio_returns, x_lagged], axis=1).dropna()
                
                if len(df) >= CORRELATION_MIN_OBS:
                    corrs[factor_name] = df.iloc[:, 0].corr(df.iloc[:, 1])
                    
            except Exception:
                corrs[factor_name] = np.nan
        
        results[lag] = pd.Series(corrs)
    
    return results

def rolling_correlation(
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