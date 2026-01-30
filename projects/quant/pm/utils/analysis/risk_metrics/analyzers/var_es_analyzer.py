import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from ..components.var import VaRCalculator
from ..components.es import ESCalculator
from ....tools.config import (
    ANNUAL_FACTOR, 
    DEFAULT_CONFIDENCE_LEVEL,
    MONTE_CARLO_SIMULATIONS,
    MONTE_CARLO_SEED
)

class VarEsAnalyzer:
    def __init__(self, annual_factor: float = None):
        self.annual_factor = annual_factor or ANNUAL_FACTOR
        self.var_calc = VaRCalculator(self.annual_factor)
        self.es_calc = ESCalculator(self.annual_factor)
    
    def calculate_multi_level(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        confidence_levels: Tuple[float, ...] = None,
        methods: List[str] | None = None,
        n_simulations: int = None,
        seed: int = None
    ) -> Dict[float, Dict[str, Dict[str, float]]]:

        if confidence_levels is None:
            confidence_levels = (0.90, DEFAULT_CONFIDENCE_LEVEL, 0.99)
        
        if methods is None:
            methods = ['historical', 'parametric', 'monte_carlo']
        
        if n_simulations is None:
            n_simulations = MONTE_CARLO_SIMULATIONS
        
        if seed is None:
            seed = MONTE_CARLO_SEED
        
        results = {}
        
        for cl in confidence_levels:
            results[cl] = {}
            
            for method in methods:

                if method == 'monte_carlo':
                    var_result = self.var_calc.calculate_monte_carlo(
                        returns, weights, cl,
                        n_simulations=n_simulations, 
                        seed=seed
                    )
                    es_result = self.es_calc.calculate_monte_carlo(
                        returns, weights, cl,
                        n_simulations=n_simulations, 
                        seed=seed
                    )
                else:
                    var_result = self.var_calc.calculate(
                        returns, weights, cl, method
                    )
                    es_result = self.es_calc.calculate(
                        returns, weights, cl, method
                    )
                results[cl][method] = {
                    **var_result,
                    **es_result
                }
        
        return results