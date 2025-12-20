import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from ..components.var import VaRCalculator
from ..components.es import ESCalculator

class VarEsAnalyzer:
    
    def __init__(self, annual_factor: float = 252.0):
        self.annual_factor = annual_factor
        self.var_calc = VaRCalculator(annual_factor)
        self.es_calc = ESCalculator(annual_factor)
    
    def calculate_multi_level(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        confidence_levels: Tuple[float, ...] = (0.95, 0.99),
        methods: List[str] | None = None,
        n_simulations: int = 10000,
        seed: int = 42
    ) -> Dict[float, Dict[str, Dict[str, float]]]:

        if methods is None:
            methods = ['historical', 'parametric', 'monte_carlo']
        
        results = {}
        
        for cl in confidence_levels:
            results[cl] = {}
            
            for method in methods:
                var_result = self.var_calc.calculate(
                    returns, weights, cl, method,
                    n_simulations=n_simulations, seed=seed
                )
                es_result = self.es_calc.calculate(
                    returns, weights, cl, method,
                    n_simulations=n_simulations, seed=seed
                )
                
                results[cl][method] = {
                    **var_result,
                    **es_result
                }
        
        return results