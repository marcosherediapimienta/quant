import pandas as pd
from typing import Dict
from ..components.macro_regression import (
    multifactor_regression,
    rolling_multifactor_regression
)

class MacroSensitivityAnalyzer:

    def __init__(self, annual_factor: int = 252):
        self.annual_factor = annual_factor
    
    def analyze(
        self,
        portfolio_returns: pd.Series,
        macro_factors: pd.DataFrame
    ) -> Dict:

        result = multifactor_regression(
            portfolio_returns,
            macro_factors,
            self.annual_factor
        )

        betas_sorted = sorted(
            result.betas.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        high_exposure = [
            (name, beta) for name, beta in betas_sorted
            if abs(beta) > 0.5
        ]
        
        moderate_exposure = [
            (name, beta) for name, beta in betas_sorted
            if 0.2 <= abs(beta) <= 0.5
        ]
        
        low_exposure = [
            (name, beta) for name, beta in betas_sorted
            if abs(beta) < 0.2
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

        return rolling_multifactor_regression(
            portfolio_returns,
            macro_factors,
            window,
            annual_factor=self.annual_factor
        )