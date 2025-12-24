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
    """
    Analyzer para calcular VaR y ES en múltiples niveles de confianza.
    
    Responsabilidad: Coordinar cálculos de VaR y ES para diferentes métodos 
    y niveles de confianza.
    """
    
    def __init__(self, annual_factor: float = None):
        """
        Args:
            annual_factor: Factor de anualización. Por defecto usa config.ANNUAL_FACTOR
        """
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
        """
        Calcula VaR y ES para múltiples niveles de confianza y métodos.
        
        Args:
            returns: DataFrame de retornos diarios
            weights: Array de pesos del portafolio
            confidence_levels: Tupla de niveles de confianza
            methods: Lista de métodos a usar
            n_simulations: Número de simulaciones para Monte Carlo
            seed: Semilla para reproducibilidad
            
        Returns:
            Dict con resultados organizados por nivel de confianza y método
        """
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