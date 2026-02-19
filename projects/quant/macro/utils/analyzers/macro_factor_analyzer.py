import pandas as pd
from typing import Dict
from ..components.macro_regression import MacroRegressionCalculator, RegressionResult
from ..components.macro_correlation import MacroCorrelationCalculator
from ..tools.config import REGRESSION_MIN_OBS, MAX_LAG

class MacroFactorAnalyzer:
    def __init__(self, annual_factor: int = 252):
        self.annual_factor = annual_factor
        self.regression_calc = MacroRegressionCalculator(annual_factor=annual_factor)
        self.correlation_calc = MacroCorrelationCalculator()
    
    def analyze(
        self,
        portfolio_returns: pd.Series,
        macro_factors: pd.DataFrame,
        use_hac: bool = True
    ) -> Dict:

        if len(portfolio_returns) < REGRESSION_MIN_OBS:
            raise ValueError(
                f"Insufficient observations: {len(portfolio_returns)}"
            )

        self.regression_calc.use_hac = use_hac

        regression_result: RegressionResult = self.regression_calc.calculate_multifactor(
            portfolio_returns,
            macro_factors
        )

        significant = self.regression_calc.get_significant_factors(regression_result)
        risk_decomp = self.regression_calc.calculate_risk_decomposition(
            regression_result, 
            portfolio_returns
        )
        factor_contrib = self.regression_calc.calculate_factor_decomposition(
            regression_result, 
            macro_factors
        )
        
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
        max_lag: int = MAX_LAG
    ) -> Dict:

        self.correlation_calc.max_lag = max_lag

        best_lags = self.correlation_calc.find_best_lag(
            portfolio_returns,
            macro_factors
        )
    
        return {
            'best_lags': best_lags,
            'leading_factors': best_lags[best_lags['lag'] < 0],  
            'lagging_factors': best_lags[best_lags['lag'] > 0],  
            'concurrent_factors': best_lags[best_lags['lag'] == 0]
        }