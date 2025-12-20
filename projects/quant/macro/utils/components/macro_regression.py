from typing import Dict, List
import numpy as np
import pandas as pd
import statsmodels.api as sm
from dataclasses import dataclass
from ..tools.config import (
    REGRESSION_MIN_OBS,
    REGRESSION_SIGNIFICANCE,
)

@dataclass
class RegressionResult:
    alpha: float
    alpha_annual: float
    betas: Dict[str, float]
    t_stats: Dict[str, float]
    p_values: Dict[str, float]
    r_squared: float
    adj_r_squared: float
    residuals: pd.Series
    fitted_values: pd.Series
    n_obs: int
    factor_names: List[str]

def multifactor_regression(
    portfolio_returns: pd.Series,
    factors: pd.DataFrame,
    annual_factor: int = 252,
    use_hac: bool = True,
    hac_maxlags: int = None
) -> RegressionResult:

    df = pd.concat([portfolio_returns, factors], axis=1).dropna()
    
    if len(df) < REGRESSION_MIN_OBS:
        raise ValueError(
            f"Observaciones insuficientes: {len(df)} < {REGRESSION_MIN_OBS}"
        )

    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    factor_names = list(factors.columns)

    X_with_const = sm.add_constant(X, has_constant='add')
    model = sm.OLS(y, X_with_const, hasconst=True)

    if use_hac:
        n = len(y)
        maxlags = hac_maxlags if hac_maxlags is not None else int(np.sqrt(n))
        fit = model.fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
    else:
        fit = model.fit()

    alpha_daily = float(fit.params[0])
    alpha_annual = (1 + alpha_daily) ** annual_factor - 1
    betas = {name: float(fit.params[i+1]) for i, name in enumerate(factor_names)}
    t_stats = {name: float(fit.tvalues[i+1]) for i, name in enumerate(factor_names)}
    p_values = {name: float(fit.pvalues[i+1]) for i, name in enumerate(factor_names)}
    residuals = pd.Series(fit.resid, index=df.index, name='residuals')
    fitted = pd.Series(fit.fittedvalues, index=df.index, name='fitted')
    
    return RegressionResult(
        alpha=alpha_daily,
        alpha_annual=alpha_annual,
        betas=betas,
        t_stats=t_stats,
        p_values=p_values,
        r_squared=float(fit.rsquared),
        adj_r_squared=float(fit.rsquared_adj),
        residuals=residuals,
        fitted_values=fitted,
        n_obs=len(df),
        factor_names=factor_names
    )

def factor_decomposition(
    result: RegressionResult,
    factors: pd.DataFrame
) -> Dict[str, pd.Series]:

    decomposition = {}
    factors_aligned = factors.reindex(result.residuals.index)

    for factor_name in result.factor_names:
        beta = result.betas[factor_name]
        factor_series = factors_aligned[factor_name]
        contribution = beta * factor_series
        decomposition[factor_name] = contribution

    decomposition['alpha'] = pd.Series(
        result.alpha, 
        index=result.residuals.index
    )
    decomposition['residual'] = result.residuals
    
    return decomposition

def rolling_multifactor_regression(
    portfolio_returns: pd.Series,
    factors: pd.DataFrame,
    window: int = 252,
    min_periods: int = None,
    annual_factor: int = 252
) -> pd.DataFrame:

    if min_periods is None:
        min_periods = window

    df = pd.concat([portfolio_returns, factors], axis=1).dropna()
    factor_names = list(factors.columns)
    
    results = []
    
    for i in range(min_periods, len(df) + 1):
        window_data = df.iloc[max(0, i - window):i]
        
        if len(window_data) < min_periods:
            continue
        
        try:
            y = window_data.iloc[:, 0].values
            X = window_data.iloc[:, 1:].values
            X_const = sm.add_constant(X, has_constant='add')
        
            model = sm.OLS(y, X_const, hasconst=True)
            fit = model.fit()

            row = {
                'date': df.index[i-1],
                'alpha': float(fit.params[0]),
                'r_squared': float(fit.rsquared)
            }
            
            for j, name in enumerate(factor_names):
                row[f'beta_{name}'] = float(fit.params[j+1])
            
            results.append(row)
            
        except Exception:
            continue
    
    return pd.DataFrame(results).set_index('date')

def significant_factors(
    result: RegressionResult,
    significance_level: float = None
) -> List[str]:

    if significance_level is None:
        significance_level = REGRESSION_SIGNIFICANCE
    
    significant = []
    
    for factor_name in result.factor_names:
        if result.p_values[factor_name] < significance_level:
            significant.append(factor_name)
    
    return significant

def risk_decomposition(
    result: RegressionResult,
    factors: pd.DataFrame
) -> Dict[str, float]:

    factors_aligned = factors.reindex(result.residuals.index)
    total_var = result.residuals.index.to_series().map(
        lambda x: factors_aligned.loc[x] if x in factors_aligned.index else np.nan
    )

    systematic_returns = result.fitted_values
    idiosyncratic_returns = result.residuals
    
    var_total = systematic_returns.var() + idiosyncratic_returns.var()
    var_systematic = systematic_returns.var()
    var_idiosyncratic = idiosyncratic_returns.var()
    
    return {
        'total_variance': float(var_total),
        'systematic_variance': float(var_systematic),
        'idiosyncratic_variance': float(var_idiosyncratic),
        'systematic_pct': float(var_systematic / var_total * 100),
        'idiosyncratic_pct': float(var_idiosyncratic / var_total * 100),
        'r_squared': result.r_squared
    }