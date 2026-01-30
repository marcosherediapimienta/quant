import numpy as np
import pandas as pd
from typing import Dict
from ..components.correlation import CorrelationCalculator
from ....tools.config import ROLLING_WINDOW

class CorrelationAnalyzer:
    def __init__(self):
        self.corr_calc = CorrelationCalculator()
    
    def analyze(
        self,
        returns: pd.DataFrame
    ) -> Dict:
    
        results = self.corr_calc.calculate(returns)
        corr_matrix = results['correlation_matrix']
        upper_triangle = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
        
        return {
            'correlation_matrix': corr_matrix,
            'mean_correlation': results['mean_correlation'],
            'min_correlation': float(upper_triangle.min()),
            'max_correlation': float(upper_triangle.max()),
            'std_correlation': float(upper_triangle.std())
        }
    
    def analyze_rolling(
        self,
        returns: pd.DataFrame,
        window: int = None
    ) -> Dict:

        if window is None:
            window = ROLLING_WINDOW
        
        corr_rolling = self.corr_calc.calculate_rolling(returns, window)
        corr_vol = self.corr_calc.calculate_correlation_volatility(returns, window)
        
        return {
            'correlation_rolling': corr_rolling,
            'correlation_volatility': corr_vol
        }