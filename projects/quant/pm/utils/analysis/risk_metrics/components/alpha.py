"""
Calculadora de Alpha (Jensen's Alpha).

Alpha mide el retorno excedente ajustado por riesgo sistemático (beta).
"""

import numpy as np
import pandas as pd
from typing import Dict
from .helpers import calculate_portfolio_returns, annualize_return
from ....tools.config import ANNUAL_FACTOR

class AlphaCalculator:
    """
    Calcula Alpha de Jensen.
    
    Responsabilidad: Medir retorno excedente vs retorno esperado por CAPM.
    
    Fórmula: α = Rp - [Rf + β(Rm - Rf)]
    donde:
    - Rp = Retorno del portfolio
    - Rf = Tasa libre de riesgo
    - β = Beta del portfolio
    - Rm = Retorno del mercado
    """

    def __init__(self, annual_factor: float = None):
        """
        Args:
            annual_factor: Factor de anualización (None = usar config)
        """
        self.annual_factor = annual_factor if annual_factor else ANNUAL_FACTOR
    
    def calculate(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        benchmark_returns: pd.Series,
        risk_free_rate: float,
        beta: float = None,
        ddof: int = 1
    ) -> Dict[str, float]:
        """
        Calcula Alpha de Jensen.
        
        Args:
            returns: DataFrame de retornos del portfolio
            weights: Pesos del portfolio
            benchmark_returns: Retornos del benchmark
            risk_free_rate: Tasa libre de riesgo anualizada
            beta: Beta precalculado (None = calcular automáticamente)
            ddof: Grados de libertad para covarianza
            
        Returns:
            Dict con alpha diario, anual y componentes
        """
        portfolio_ret = calculate_portfolio_returns(returns, weights)
        df = pd.DataFrame({
            'portfolio': portfolio_ret,
            'benchmark': benchmark_returns
        }).dropna()
        
        if df.empty:
            return {
                'alpha': np.nan,
                'alpha_annual': np.nan,
                'portfolio_return_annual': np.nan,
                'benchmark_return_annual': np.nan,
                'beta_used': np.nan
            }

        # Anualizar retornos
        portfolio_return_annual = annualize_return(df['portfolio'], self.annual_factor)
        benchmark_return_annual = annualize_return(df['benchmark'], self.annual_factor)

        # Calcular beta si no se proporciona
        if beta is None:
            cov_matrix = np.cov(df['portfolio'], df['benchmark'], ddof=ddof)
            beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else 1.0

        # Alpha de Jensen
        expected_return = risk_free_rate + beta * (benchmark_return_annual - risk_free_rate)
        alpha_annual = float(portfolio_return_annual - expected_return)
        alpha_daily = alpha_annual / self.annual_factor
        
        return {
            'alpha': alpha_daily,
            'alpha_annual': alpha_annual,
            'portfolio_return_annual': float(portfolio_return_annual),
            'benchmark_return_annual': float(benchmark_return_annual),
            'expected_return': float(expected_return),
            'beta_used': float(beta)
        }