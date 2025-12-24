"""
Calculadora de Drawdown y ratios relacionados.

Drawdown mide la caída desde un pico histórico.
"""

import numpy as np
import pandas as pd
from typing import Dict
from .helpers import calculate_portfolio_returns, annualize_return
from ....tools.config import ANNUAL_FACTOR

class DrawdownCalculator:
    """
    Calcula drawdown y métricas relacionadas.
    
    Responsabilidad: Medir pérdidas desde máximos históricos.
    
    Métricas calculadas:
    - Max Drawdown: Mayor caída desde un pico
    - Calmar Ratio: Retorno anual / Max Drawdown
    - Sterling Ratio: Retorno / Promedio de 3 peores drawdowns mensuales
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
        risk_free_rate: float = 0.0
    ) -> Dict:
        """
        Calcula drawdown y métricas relacionadas.
        
        Args:
            returns: DataFrame de retornos
            weights: Pesos del portfolio
            risk_free_rate: Tasa libre de riesgo (para Sterling Ratio)
            
        Returns:
            Dict con:
            - max_drawdown: Mayor caída (valor negativo)
            - max_drawdown_date: Fecha del mayor drawdown
            - max_underwater_duration: Días más largos bajo el agua
            - calmar_ratio: Retorno / Max Drawdown
            - sterling_ratio: Retorno / Avg(3 peores DD mensuales)
            - drawdown_series: Serie temporal de drawdowns
            - cumulative_returns: Serie acumulada
        """
        portfolio_ret = calculate_portfolio_returns(returns, weights)
        
        # Calcular retornos acumulados
        cumulative = (1 + portfolio_ret).cumprod()
        
        # Calcular drawdown
        running_max = cumulative.cummax()
        drawdown = (cumulative / running_max) - 1.0
        
        # Max drawdown y fecha
        max_dd = float(drawdown.min())
        max_dd_date = drawdown.idxmin()
        
        # Calcular duración máxima bajo el agua
        underwater = drawdown < 0
        current_duration = 0
        max_duration = 0
        in_drawdown = False

        for date, is_underwater in underwater.items():
            if is_underwater:
                if not in_drawdown:
                    in_drawdown = True
                    drawdown_start = date
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                in_drawdown = False
                current_duration = 0

        # Calmar Ratio = Retorno Anual / |Max Drawdown|
        annual_return = annualize_return(portfolio_ret, self.annual_factor)
        calmar = float(annual_return / abs(max_dd)) if max_dd < 0 else np.nan
        
        # Sterling Ratio = (Retorno - Rf) / Avg(3 peores DD mensuales)
        dd_monthly = drawdown.resample('ME').min()
        worst_3 = dd_monthly.nsmallest(3)
        
        if len(worst_3) >= 1 and worst_3.mean() < 0:
            sterling = float((annual_return - risk_free_rate) / abs(worst_3.mean()))
        else:
            sterling = np.nan
        
        return {
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd * 100,
            'max_drawdown_date': max_dd_date,
            'max_underwater_duration': int(max_duration),
            'calmar_ratio': calmar,
            'sterling_ratio': sterling,
            'drawdown_series': drawdown,
            'cumulative_returns': cumulative,
            'annual_return': float(annual_return)
        }