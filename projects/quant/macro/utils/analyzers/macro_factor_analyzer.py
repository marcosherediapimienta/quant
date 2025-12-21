import pandas as pd
from typing import Dict
from ..components.macro_regression import (
    multifactor_regression,
    factor_decomposition,
    significant_factors,
    risk_decomposition,
    RegressionResult
)
from ..components.macro_correlation import (
    best_lagged_correlation
)
from ..tools.config import REGRESSION_MIN_OBS

class MacroFactorAnalyzer:

    def __init__(self, annual_factor: int = 252):
        self.annual_factor = annual_factor
    
    def analyze(
        self,
        portfolio_returns: pd.Series,
        macro_factors: pd.DataFrame,
        use_hac: bool = True
    ) -> Dict:

        if len(portfolio_returns) < REGRESSION_MIN_OBS:
            raise ValueError(
                f"Observaciones insuficientes: {len(portfolio_returns)}"
            )

        regression_result: RegressionResult = multifactor_regression(
            portfolio_returns,
            macro_factors,
            self.annual_factor,
            use_hac
        )

        significant = significant_factors(regression_result)
        risk_decomp = risk_decomposition(regression_result, macro_factors)
        factor_contrib = factor_decomposition(regression_result, macro_factors)
        
        return {
            'regression': regression_result,
            'significant_factors': significant,
            'risk_decomposition': risk_decomp,
            'factor_contributions': factor_contrib,
            'alpha': regression_result.alpha,
            'alpha_annual': regression_result.alpha_annual,
            'betas': regression_result.betas,
            't_stats': regression_result.t_stats,
            'p_values': regression_result.p_values,
            'r_squared': regression_result.r_squared,
            'adj_r_squared': regression_result.adj_r_squared,
            'n_obs': regression_result.n_obs
        }
    
    def analyze_with_lags(
        self,
        portfolio_returns: pd.Series,
        macro_factors: pd.DataFrame,
        max_lag: int = 126
    ) -> Dict:

        best_lags = best_lagged_correlation(
            portfolio_returns,
            macro_factors,
            max_lag
        )
    
        return {
            'best_lags': best_lags,
            'leading_factors': best_lags[best_lags['lag'] < 0],  
            'lagging_factors': best_lags[best_lags['lag'] > 0],  
            'concurrent_factors': best_lags[best_lags['lag'] == 0]
        }