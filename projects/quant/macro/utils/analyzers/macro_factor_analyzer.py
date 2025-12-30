import pandas as pd
from typing import Dict
from ..components.macro_regression import MacroRegressionCalculator, RegressionResult
from ..components.macro_correlation import MacroCorrelationCalculator
from ..tools.config import REGRESSION_MIN_OBS

class MacroFactorAnalyzer:
    """
    Analizador de alto nivel para factores macro.
    
    Responsabilidad: Orquestar análisis de regresión multifactor y generar insights.
    """

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
        """
        Análisis completo de factores macro.
        
        Args:
            portfolio_returns: Retornos del portfolio
            macro_factors: DataFrame con factores macro
            use_hac: Si usar HAC standard errors
            
        Returns:
            Dict con resultados de regresión y análisis
        """
        if len(portfolio_returns) < REGRESSION_MIN_OBS:
            raise ValueError(
                f"Observaciones insuficientes: {len(portfolio_returns)}"
            )

        # Actualizar configuración de HAC si es necesario
        self.regression_calc.use_hac = use_hac

        # Calcular regresión multifactor
        regression_result: RegressionResult = self.regression_calc.calculate_multifactor(
            portfolio_returns,
            macro_factors
        )

        # Obtener factores significativos
        significant = self.regression_calc.get_significant_factors(regression_result)
        
        # Descomposición de riesgo
        risk_decomp = self.regression_calc.calculate_risk_decomposition(
            regression_result, 
            macro_factors
        )
        
        # Contribución por factor
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
        max_lag: int = 126
    ) -> Dict:
        """
        Análisis de factores con lags óptimos.
        
        Args:
            portfolio_returns: Retornos del portfolio
            macro_factors: DataFrame con factores macro
            max_lag: Máximo lag a analizar
            
        Returns:
            Dict con factores leading, lagging y concurrent
        """
        # Actualizar max_lag del calculator
        self.correlation_calc.max_lag = max_lag

        # Encontrar mejores lags
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