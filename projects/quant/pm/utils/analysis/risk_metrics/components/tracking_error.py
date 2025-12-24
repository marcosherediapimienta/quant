"""
Calculadora de Tracking Error e Information Ratio.

Tracking Error mide cuánto se desvía un portfolio de su benchmark.
"""

import numpy as np
import pandas as pd
from typing import Dict
from .helpers import calculate_portfolio_returns
from ....tools.config import ANNUAL_FACTOR, ROLLING_WINDOW

class TrackingErrorCalculator:
    """
    Calcula Tracking Error (TE) y métricas relacionadas.
    
    Responsabilidad: Medir desviación vs benchmark.
    
    TE = Desviación estándar de los retornos activos
    Information Ratio = Exceso de retorno / TE
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
        ddof: int = 1
    ) -> Dict[str, float]:
        """
        Calcula Tracking Error e Information Ratio.
        
        Args:
            returns: DataFrame de retornos del portfolio
            weights: Pesos del portfolio
            benchmark_returns: Retornos del benchmark
            ddof: Grados de libertad para std
            
        Returns:
            Dict con:
            - te_daily: Tracking error diario
            - te_annual: Tracking error anualizado
            - excess_return_annual: Retorno activo anualizado
            - information_ratio: IR = Exceso / TE
            
        Interpretación de TE:
        - TE bajo (<2%): Portfolio cercano al benchmark (index tracking)
        - TE medio (2-5%): Desviación moderada
        - TE alto (>5%): Portfolio muy activo
        """
        portfolio_ret = calculate_portfolio_returns(returns, weights)
        df = pd.DataFrame({
            'portfolio': portfolio_ret,
            'benchmark': benchmark_returns
        }).dropna()
        
        if df.empty:
            return {
                'te_daily': np.nan,
                'te_annual': np.nan,
                'information_ratio': np.nan,
                'excess_return_annual': np.nan
            }

        # Retornos activos = Portfolio - Benchmark
        active_returns = df['portfolio'] - df['benchmark']
        
        # Tracking Error = std(retornos activos)
        te_daily = float(active_returns.std(ddof=ddof))
        te_annual = float(te_daily * np.sqrt(self.annual_factor))
        
        # Exceso de retorno anualizado
        excess_return_annual = float(active_returns.mean() * self.annual_factor)
        
        # Information Ratio = Exceso / TE
        info_ratio = float(excess_return_annual / te_annual) if te_annual > 0 else np.nan
        
        return {
            'te_daily': te_daily,
            'te_annual': te_annual,
            'excess_return_annual': excess_return_annual,
            'information_ratio': info_ratio
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
        Calcula Tracking Error móvil.
        
        Args:
            returns: DataFrame de retornos
            weights: Pesos del portfolio
            benchmark_returns: Retornos del benchmark
            window: Ventana móvil (None = usar config, default sugerido: ~63 días / 3 meses)
            ddof: Grados de libertad
            
        Returns:
            Serie con TE anualizado móvil
        """
        #Usar config, con fallback razonable si es muy largo
        if window is None:
            # Para TE, 63 días (3 meses) es más común que 252 días
            # Pero usamos ROLLING_WINDOW / 4 para ser consistentes
            window = max(63, ROLLING_WINDOW // 4)
        
        portfolio_ret = calculate_portfolio_returns(returns, weights)
        df = pd.DataFrame({
            'portfolio': portfolio_ret,
            'benchmark': benchmark_returns
        }).dropna()
        
        if df.empty or len(df) < window:
            return pd.Series(dtype=float)
        
        # TE móvil anualizado
        active_returns = df['portfolio'] - df['benchmark']
        te_rolling = active_returns.rolling(window=window).std(ddof=ddof) * np.sqrt(self.annual_factor)
        
        return te_rolling