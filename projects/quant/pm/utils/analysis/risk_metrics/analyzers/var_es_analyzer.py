import numpy as np
import pandas as pd
from typing import Dict, Tuple, List


class VarEsAnalyzer:
    
    def __init__(self, risk_analysis):
        self.risk_analysis = risk_analysis
        self.var_calc = risk_analysis.var
        self.es_calc = risk_analysis.es
    
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
    
    def compare_methods(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        confidence_level: float = 0.95,
        n_simulations: int = 10000,
        seed: int = 42
    ) -> pd.DataFrame:

        methods = ['historical', 'parametric', 'monte_carlo']
        results_list = []
        
        for method in methods:
            var_result = self.var_calc.calculate(
                returns, weights, confidence_level, method,
                n_simulations=n_simulations, seed=seed
            )
            es_result = self.es_calc.calculate(
                returns, weights, confidence_level, method,
                n_simulations=n_simulations, seed=seed
            )
            
            results_list.append({
                'method': method.capitalize(),
                'VaR_daily_%': var_result['var_daily_pct'],
                'ES_daily_%': es_result['es_daily_pct'],
                'VaR_annual_%': var_result['var_annual_pct'],
                'ES_annual_%': es_result['es_annual_pct']
            })
        
        return pd.DataFrame(results_list).set_index('method')