import pandas as pd
from typing import Dict
from ..components.macro_regression import MacroRegressionCalculator
from ..tools.config import SENSITIVITY_THRESHOLDS

class MacroSensitivityAnalyzer:
    def __init__(self, annual_factor: int = 252):
        self.annual_factor = annual_factor
        self.regression_calc = MacroRegressionCalculator(annual_factor=annual_factor)
    
    def analyze(
        self,
        portfolio_returns: pd.Series,
        macro_factors: pd.DataFrame
    ) -> Dict:

        result = self.regression_calc.calculate_multifactor(
            portfolio_returns,
            macro_factors
        )

        betas_sorted = sorted(
            result.betas.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        high_threshold = SENSITIVITY_THRESHOLDS['high']
        moderate_threshold = SENSITIVITY_THRESHOLDS['moderate']

        high_exposure = [
            (name, beta) for name, beta in betas_sorted
            if abs(beta) > high_threshold
        ]
        
        moderate_exposure = [
            (name, beta) for name, beta in betas_sorted
            if moderate_threshold <= abs(beta) <= high_threshold
        ]
        
        low_exposure = [
            (name, beta) for name, beta in betas_sorted
            if abs(beta) < moderate_threshold
        ]
        
        return {
            'betas': result.betas,
            't_stats': result.t_stats,
            'p_values': result.p_values,
            'betas_ranked': dict(betas_sorted),
            'high_exposure': dict(high_exposure),
            'moderate_exposure': dict(moderate_exposure),
            'low_exposure': dict(low_exposure),
            'dominant_factor': betas_sorted[0] if betas_sorted else None
        }
    
    def analyze_rolling(
        self,
        portfolio_returns: pd.Series,
        macro_factors: pd.DataFrame,
        window: int = 252
    ) -> pd.DataFrame:

        return self.regression_calc.calculate_rolling(
            portfolio_returns,
            macro_factors,
            window
        )