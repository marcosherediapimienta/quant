"""
Calculadora de momentos estadísticos de la distribución.

Analiza skewness, kurtosis y normalidad de los retornos.
"""

import numpy as np
import pandas as pd
from scipy.stats import jarque_bera
from typing import Dict
from .helpers import calculate_portfolio_returns
from ....tools.config import SIGNIFICANCE_LEVEL

class DistributionMoments:
    """
    Calcula momentos estadísticos de retornos.
    
    Responsabilidad: Analizar forma de la distribución de retornos.
    
    Momentos:
    1. Media (1er momento)
    2. Desviación estándar (2do momento)
    3. Skewness (3er momento) - asimetría
    4. Kurtosis (4to momento) - colas gordas
    """

    def __init__(self):
        """Distribution moments calculator sin parámetros."""
        pass
    
    def calculate_skewness(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray
    ) -> float:
        """
        Calcula skewness (asimetría).
        
        Interpretación:
        - Skew = 0: Simétrico (normal)
        - Skew < 0: Cola izquierda (más pérdidas extremas)
        - Skew > 0: Cola derecha (más ganancias extremas)
        """
        portfolio_ret = calculate_portfolio_returns(returns, weights)
        skew = portfolio_ret.skew()
        return float(skew)
    
    def calculate_kurtosis(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        excess: bool = True
    ) -> float:
        """
        Calcula kurtosis (colas gordas).
        
        Args:
            excess: Si True, retorna excess kurtosis (normal = 0)
                   Si False, retorna kurtosis total (normal = 3)
                   
        Interpretación (excess kurtosis):
        - Kurt = 0: Colas normales
        - Kurt > 0: Colas gordas (leptokurtic) - más eventos extremos
        - Kurt < 0: Colas delgadas (platykurtic) - menos eventos extremos
        """
        portfolio_ret = calculate_portfolio_returns(returns, weights)
        
        if excess:
            kurt = portfolio_ret.kurtosis()  # pandas retorna excess por default
        else:
            kurt = portfolio_ret.kurtosis() + 3.0
        
        return float(kurt)
    
    def calculate_jarque_bera(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        alpha: float = None
    ) -> Dict[str, float]:
        """
        Test de Jarque-Bera para normalidad.
        
        Args:
            alpha: Nivel de significancia (None = usar config)
            
        H0: Los datos siguen una distribución normal
        H1: Los datos NO siguen una distribución normal
        
        Returns:
            Dict con estadístico JB, p-value y conclusión
        """
        alpha = alpha if alpha else SIGNIFICANCE_LEVEL
        
        portfolio_ret = calculate_portfolio_returns(returns, weights)
        jb_stat, p_value = jarque_bera(portfolio_ret)
        
        return {
            'jb_statistic': float(jb_stat),
            'p_value': float(p_value),
            'is_normal': bool(p_value > alpha)  
        }
    
    def calculate_all(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray
    ) -> Dict[str, float]:
        """
        Calcula todos los momentos y percentiles.
        
        Returns:
            Dict completo con todos los estadísticos
        """
        portfolio_ret = calculate_portfolio_returns(returns, weights)
        
        # Momentos básicos
        mean = float(portfolio_ret.mean())
        std = float(portfolio_ret.std(ddof=0))
        median = float(portfolio_ret.median())
        
        # Momentos superiores
        skew = self.calculate_skewness(returns, weights)
        excess_kurt = self.calculate_kurtosis(returns, weights, excess=True)
        jb_results = self.calculate_jarque_bera(returns, weights)
        
        # Percentiles
        p1 = float(portfolio_ret.quantile(0.01))
        p5 = float(portfolio_ret.quantile(0.05))
        p95 = float(portfolio_ret.quantile(0.95))
        p99 = float(portfolio_ret.quantile(0.99))
        
        return {
            'mean': mean,
            'median': median,
            'std': std,
            'skewness': skew,
            'excess_kurtosis': excess_kurt,
            'jb_statistic': jb_results['jb_statistic'],
            'jb_p_value': jb_results['p_value'],
            'is_normal': jb_results['is_normal'],
            'percentile_1': p1,
            'percentile_5': p5,
            'percentile_95': p95,
            'percentile_99': p99
        }