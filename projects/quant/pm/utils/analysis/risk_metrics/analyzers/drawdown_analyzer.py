import numpy as np
import pandas as pd
from typing import Dict
from ..components.drawdown import DrawdownCalculator
from ....tools.config import ANNUAL_FACTOR

class DrawdownAnalyzer:
    """
    Analyzer para analizar drawdowns y ratios basados en drawdown.
    
    Responsabilidad: Coordinar cálculo de drawdowns, Calmar y Sterling ratios.
    """

    def __init__(self, annual_factor: float = None):
        """
        Args:
            annual_factor: Factor de anualización. Por defecto usa config.ANNUAL_FACTOR
        """
        self.annual_factor = annual_factor or ANNUAL_FACTOR
        self.dd_calc = DrawdownCalculator(self.annual_factor)
    
    def analyze(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        risk_free_rate: float = None
    ) -> Dict:
        """
        Analiza drawdowns del portafolio.
        
        Args:
            returns: DataFrame de retornos diarios
            weights: Array de pesos del portafolio
            risk_free_rate: Tasa libre de riesgo. Por defecto 0.0
            
        Returns:
            Dict con max drawdown, duración, Calmar, Sterling y series temporales
        """
        if risk_free_rate is None:
            risk_free_rate = 0.0
        
        results = self.dd_calc.calculate(returns, weights, risk_free_rate)
        
        return {
            'max_drawdown': results['max_drawdown'],
            'max_drawdown_pct': results['max_drawdown_pct'],
            'max_drawdown_date': results['max_drawdown_date'],
            'max_underwater_duration': results['max_underwater_duration'],
            'calmar_ratio': results['calmar_ratio'],
            'sterling_ratio': results['sterling_ratio'],
            'annual_return': results['annual_return'],
            'drawdown_series': results['drawdown_series'],
            'cumulative_returns': results['cumulative_returns']
        }