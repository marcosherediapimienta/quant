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


class MacroRegressionCalculator:
    def __init__(
        self,
        annual_factor: int = 252,
        use_hac: bool = True,
        hac_maxlags: int = None,
        min_obs: int = None,
        significance_level: float = None
    ):
        self.annual_factor = annual_factor
        self.use_hac = use_hac
        self.hac_maxlags = hac_maxlags
        self.min_obs = min_obs if min_obs is not None else REGRESSION_MIN_OBS
        self.significance_level = significance_level if significance_level is not None else REGRESSION_SIGNIFICANCE
    
    def calculate_multifactor(
        self,
        portfolio_returns: pd.Series,
        factors: pd.DataFrame
    ) -> RegressionResult:

        df = pd.concat([portfolio_returns, factors], axis=1).dropna()
        
        if len(df) < self.min_obs:
            raise ValueError(
                f"Observaciones insuficientes: {len(df)} < {self.min_obs}"
            )

        y = df.iloc[:, 0].values
        X = df.iloc[:, 1:].values
        factor_names = list(factors.columns)

        X_with_const = sm.add_constant(X, has_constant='add')
        model = sm.OLS(y, X_with_const, hasconst=True)

        if self.use_hac:
            n = len(y)
            maxlags = self.hac_maxlags if self.hac_maxlags is not None else int(np.sqrt(n))
            fit = model.fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
        else:
            fit = model.fit()

        alpha_daily = float(fit.params[0])

        if alpha_daily <= -0.95:
            import logging as _log
            _log.getLogger(__name__).warning(
                "Extreme daily alpha detected: %.4f — indicates ≥95%% daily loss, likely data error. Returning NaN.",
                alpha_daily
            )
            alpha_annual = np.nan
        elif alpha_daily > -0.99:
            alpha_annual = (1 + alpha_daily) ** self.annual_factor - 1
        else:
            import logging as _log
            _log.getLogger(__name__).warning(
                "Very negative daily alpha: %.4f — using linear approximation", alpha_daily
            )
            alpha_annual = alpha_daily * self.annual_factor
        
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
    
    def calculate_factor_decomposition(
        self,
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
    
    def calculate_rolling(
        self,
        portfolio_returns: pd.Series,
        factors: pd.DataFrame,
        window: int = 252,
        min_periods: int = None
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
    
    def get_significant_factors(
        self,
        result: RegressionResult
    ) -> List[str]:

        significant = []
        
        for factor_name in result.factor_names:
            if result.p_values[factor_name] < self.significance_level:
                significant.append(factor_name)
        
        return significant
    
    def calculate_risk_decomposition(
        self,
        result: RegressionResult,
        portfolio_returns: pd.Series = None 
    ) -> Dict[str, float]:

        common_index = result.residuals.index
        portfolio_original = result.fitted_values + result.residuals
        fitted_vals = result.fitted_values
        residual_vals = result.residuals
        
        if not isinstance(fitted_vals, pd.Series):
            fitted_vals = pd.Series(
                fitted_vals.flatten() if hasattr(fitted_vals, 'flatten') and fitted_vals.ndim > 1 else fitted_vals, 
                index=common_index
            )
        
        if not isinstance(residual_vals, pd.Series):
            residual_vals = pd.Series(
                residual_vals.flatten() if hasattr(residual_vals, 'flatten') and residual_vals.ndim > 1 else residual_vals, 
                index=common_index
            )

        df_clean = pd.DataFrame({
            'portfolio': portfolio_original,
            'fitted': fitted_vals,
            'residuals': residual_vals
        })

        if df_clean.isna().any().any():
            import logging as _log
            _log.getLogger(__name__).warning("NaN values found in risk decomposition, dropping rows...")
            df_clean = df_clean.dropna()
        
        if len(df_clean) == 0:
            raise ValueError("No hay datos válidos después de limpiar NaN")

        var_total = float(np.var(df_clean['portfolio'].values, ddof=0))
        var_systematic = float(np.var(df_clean['fitted'].values, ddof=0))
        var_idiosyncratic = float(np.var(df_clean['residuals'].values, ddof=0))
        sum_vars = var_systematic + var_idiosyncratic
        cov_fitted_residual = float(np.cov(df_clean['fitted'].values, df_clean['residuals'].values)[0, 1])
        
        if abs(var_total - sum_vars) > 1e-4:
            import logging as _log
            _log.getLogger(__name__).warning(
                "Variance decomposition mismatch: Var(Y)=%.6f, Var(Ŷ)+Var(ε)=%.6f, "
                "diff=%.6f (%.2f%%), Cov(Ŷ,ε)=%.6f — n=%d",
                var_total, sum_vars, abs(var_total - sum_vars),
                abs(var_total - sum_vars) / var_total * 100 if var_total else 0,
                cov_fitted_residual, len(df_clean)
            )
        
        return {
            'total_variance': var_total,
            'systematic_variance': var_systematic,
            'idiosyncratic_variance': var_idiosyncratic,
            'systematic_pct': float(var_systematic / var_total * 100) if var_total > 0 else 0.0,
            'idiosyncratic_pct': float(var_idiosyncratic / var_total * 100) if var_total > 0 else 0.0,
            'r_squared': result.r_squared
        }