import pandas as pd
from typing import Dict
from ..components.macro_correlation import MacroCorrelationCalculator
from ..tools.config import MAX_LAG


class MacroCorrelationAnalyzer:
    """
    Analizador de alto nivel para correlaciones macro.
    
    Responsabilidad: Orquestar análisis de correlaciones y generar insights.
    """

    def __init__(self, max_lag: int = None):
        """
        Args:
            max_lag: Máximo lag a analizar (None = usar config)
        """
        self.max_lag = max_lag if max_lag is not None else MAX_LAG
        self.calculator = MacroCorrelationCalculator(max_lag=self.max_lag)
    
    def analyze(
        self,
        portfolio_returns: pd.Series,
        macro_factors: pd.DataFrame
    ) -> Dict:
        """
        Análisis completo de correlaciones.
        
        Args:
            portfolio_returns: Retornos del portfolio
            macro_factors: DataFrame con factores macro
            
        Returns:
            Dict con análisis de lags y top factors
        """
        best_lags = self.calculator.find_best_lag(
            portfolio_returns,
            macro_factors
        )

        corr_matrix_lags = self.calculator.calculate_matrix_with_lags(
            portfolio_returns,
            macro_factors,
            lags=[0, 1, 5, 21, 63, 126]
        )
        
        return {
            'best_lagged_correlations': best_lags,
            'correlation_by_lag': corr_matrix_lags,
            'top_factors': best_lags.head(5),
            'bottom_factors': best_lags.tail(5)
        }
    
    def analyze_rolling(
        self,
        portfolio_returns: pd.Series,
        macro_factors: pd.DataFrame,
        window: int = 252
    ) -> pd.DataFrame:
        """
        Análisis de correlaciones móviles.
        
        Args:
            portfolio_returns: Retornos del portfolio
            macro_factors: DataFrame con factores macro
            window: Ventana móvil
            
        Returns:
            DataFrame con correlaciones móviles por factor
        """
        rolling_corrs = {}
        
        for factor_name in macro_factors.columns:
            rolling_corrs[factor_name] = self.calculator.calculate_rolling(
                portfolio_returns,
                macro_factors[factor_name],
                window
            )
        
        return pd.DataFrame(rolling_corrs)