"""
Calculadora de Beta.

Beta mide la sensibilidad del portfolio a movimientos del mercado.
"""

import numpy as np
import pandas as pd
from typing import Dict
from .helpers import calculate_portfolio_returns
from ....tools.config import ROLLING_WINDOW 

class BetaCalculator:
    """
    Calcula Beta del portfolio vs benchmark.
    
    Responsabilidad: Medir riesgo sistemático (sensibilidad al mercado).
    
    Fórmula: β = Cov(Rp, Rm) / Var(Rm)
    
    Interpretación:
    - β = 1: Se mueve igual que el mercado
    - β > 1: Más volátil que el mercado (amplifica movimientos)
    - β < 1: Menos volátil que el mercado (amortigua movimientos)
    - β < 0: Se mueve inversamente al mercado (cobertura)
    
    Ejemplo:
        β = 1.5 significa que si el mercado sube 1%, el portfolio sube 1.5%
        β = 0.5 significa que si el mercado baja 1%, el portfolio baja 0.5%
    """

    def __init__(self):
        """
        Beta calculator sin parámetros de configuración.
        
        Nota: Beta es independiente del factor de anualización.
        """
        pass
    
    def calculate(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        benchmark_returns: pd.Series,
        ddof: int = 1
    ) -> Dict[str, float]:
        """
        Calcula Beta del portfolio.
        
        Args:
            returns: DataFrame de retornos diarios del portfolio
            weights: Pesos del portfolio (deben sumar 1)
            benchmark_returns: Serie de retornos del benchmark
            ddof: Grados de libertad para covarianza (1 = sample, 0 = population)
            
        Returns:
            Dict con:
            - beta: Sensibilidad al mercado
            - r_squared: Proporción de varianza explicada por el mercado
            - correlation: Correlación con el benchmark
            
        Ejemplo:
            >>> calculator = BetaCalculator()
            >>> result = calculator.calculate(returns, weights, benchmark_returns)
            >>> print(f"Beta: {result['beta']:.2f}")
            >>> print(f"R²: {result['r_squared']:.2%}")
        """
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

        # Calcular matriz de covarianza
        cov_matrix = np.cov(df['portfolio'], df['benchmark'], ddof=ddof)
        cov_pb = cov_matrix[0, 1]  # Covarianza portfolio-benchmark
        var_b = cov_matrix[1, 1]   # Varianza del benchmark
        
        # Beta = Cov(P,B) / Var(B)
        beta = float(cov_pb / var_b) if var_b > 0 else np.nan
        
        # Correlación y R²
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
        window: int = None, 
        ddof: int = 1
    ) -> pd.Series:
        """
        Calcula Beta móvil (rolling beta).
        
        Útil para observar cómo cambia la sensibilidad al mercado en el tiempo.
        
        Args:
            returns: DataFrame de retornos
            weights: Pesos del portfolio
            benchmark_returns: Retornos del benchmark
            window: Ventana móvil en días (None = usar ROLLING_WINDOW de config)
            ddof: Grados de libertad
            
        Returns:
            Series con beta móvil indexado por fecha
            
        Ejemplo:
            >>> rolling_beta = calculator.calculate_rolling(returns, weights, benchmark)
            >>> print(f"Beta actual: {rolling_beta.iloc[-1]:.2f}")
            >>> print(f"Beta promedio: {rolling_beta.mean():.2f}")
        """
        # ✅ Usar configuración si no se especifica
        window = window if window else ROLLING_WINDOW
        
        portfolio_ret = calculate_portfolio_returns(returns, weights)

        df = pd.DataFrame({
            'portfolio': portfolio_ret,
            'benchmark': benchmark_returns
        }).dropna()
        
        if df.empty or len(df) < window:
            return pd.Series(dtype=float)

        def _calculate_beta_window(window_df):
            """
            Calcula beta para una ventana específica.
            
            Responsabilidad: Aislar cálculo de beta de una ventana.
            """
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
            except Exception:
                return np.nan

        # Calcular beta móvil
        beta_rolling = df.rolling(window=window).apply(
            lambda x: _calculate_beta_window(df.loc[x.index]),
            raw=False
        )['portfolio']
        
        return beta_rolling