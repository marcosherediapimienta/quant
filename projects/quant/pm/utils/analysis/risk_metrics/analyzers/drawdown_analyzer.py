import numpy as np
import pandas as pd
from typing import Dict
from ..components.drawdown import DrawdownCalculator


class DrawdownAnalyzer:

    def __init__(self, annual_factor: float = 252.0):
        self.annual_factor = annual_factor
        self.dd_calc = DrawdownCalculator(annual_factor)
    
    def analyze(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        risk_free_rate: float = 0.0
    ) -> Dict:

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