"""
Calculadora de correlaciones entre activos.

Analiza relaciones lineales entre retornos de diferentes activos.
"""

import numpy as np
import pandas as pd
from typing import Dict
from ....tools.config import ROLLING_WINDOW

class CorrelationCalculator:
    """
    Calcula correlaciones entre activos del portfolio.
    
    Responsabilidad: Medir relaciones lineales entre retornos.
    
    Correlación útil para:
    - Diversificación (buscar correlaciones bajas)
    - Gestión de riesgo (identificar exposiciones similares)
    - Construcción de portfolio (evitar activos muy correlacionados)
    """

    def __init__(self):
        """Correlation calculator sin parámetros de configuración."""
        pass
    
    def calculate(
        self,
        returns: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Calcula matriz de correlación.
        
        Args:
            returns: DataFrame con retornos de múltiples activos
            
        Returns:
            Dict con:
            - correlation_matrix: Matriz de correlación completa
            - mean_correlation: Correlación promedio (excl. diagonal)
            
        Interpretación:
        - Corr = 1: Perfectamente correlacionados (se mueven igual)
        - Corr = 0: No correlacionados (independientes)
        - Corr = -1: Perfectamente anti-correlacionados (cobertura perfecta)
        """
        corr_matrix = returns.corr()
        
        # Correlación promedio (solo triángulo superior, sin diagonal)
        upper_triangle = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
        mean_corr = float(upper_triangle.mean()) if len(upper_triangle) > 0 else np.nan
        
        return {
            'correlation_matrix': corr_matrix,
            'mean_correlation': mean_corr
        }
    
    def calculate_rolling(
        self,
        returns: pd.DataFrame,
        window: int = None  
    ) -> pd.DataFrame:
        """
        Calcula correlaciones móviles entre pares de activos.
        
        Args:
            returns: DataFrame con retornos
            window: Ventana móvil (None = usar ROLLING_WINDOW de config)
            
        Returns:
            DataFrame con series de correlaciones para cada par
            Columnas: 'ASSET1_ASSET2' con correlaciones móviles
            
        Útil para:
        - Observar cómo cambian las correlaciones en el tiempo
        - Identificar períodos de crisis (correlaciones tienden a 1)
        - Validar supuestos de diversificación
        """
        window = window if window else ROLLING_WINDOW 
        
        n_assets = returns.shape[1]
        
        if n_assets < 2:
            return pd.DataFrame()
        
        # Caso especial: solo 2 activos
        if n_assets == 2:
            col1, col2 = returns.columns
            corr_rolling = returns[col1].rolling(window=window).corr(returns[col2])
            return pd.DataFrame({f'{col1}_{col2}': corr_rolling})
        
        # Caso general: múltiples activos (calcular todas las combinaciones)
        corr_dict = {}
        cols = returns.columns
        
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                pair_name = f'{cols[i]}_{cols[j]}'
                corr_dict[pair_name] = returns[cols[i]].rolling(window=window).corr(returns[cols[j]])
        
        return pd.DataFrame(corr_dict)
    
    def calculate_correlation_volatility(
        self,
        returns: pd.DataFrame,
        window: int = None 
    ) -> float:
        """
        Calcula volatilidad de las correlaciones.
        
        Mide cuánto varían las correlaciones en el tiempo.
        
        Args:
            returns: DataFrame con retornos
            window: Ventana móvil (None = usar config)
            
        Returns:
            Volatilidad promedio de correlaciones
            
        Interpretación:
        - Corr Vol baja: Relaciones estables
        - Corr Vol alta: Relaciones cambiantes (riesgo de modelo)
        """
        window = window if window else ROLLING_WINDOW  
        
        corr_rolling = self.calculate_rolling(returns, window)
        
        if corr_rolling.empty:
            return np.nan
        
        # Volatilidad promedio de todas las correlaciones
        corr_vol = corr_rolling.std().mean()
        
        return float(corr_vol)