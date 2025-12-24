import numpy as np
import pandas as pd
from typing import Dict
from ..analyzers.var_es_analyzer import VarEsAnalyzer
from ....tools.config import (
    DEFAULT_CONFIDENCE_LEVEL,
    MONTE_CARLO_SIMULATIONS,
    MONTE_CARLO_SEED,
    VAR_RISK_LEVELS
)

class VarEsReporter:
    """
    Reporter para generar informes de VaR y ES.
    
    Responsabilidad: Formatear y presentar resultados de VaR/ES de forma legible.
    """

    def __init__(self, var_es_analyzer: VarEsAnalyzer):
        """
        Args:
            var_es_analyzer: Instancia de VarEsAnalyzer para cálculos
        """
        self.analyzer = var_es_analyzer
    
    def generate_report(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        confidence_level: float = None,
        n_simulations: int = None,
        seed: int = None
    ) -> None:
        """
        Genera reporte de VaR y ES comparando métodos.
        
        Args:
            returns: DataFrame de retornos diarios
            weights: Array de pesos del portafolio
            confidence_level: Nivel de confianza. Por defecto usa config
            n_simulations: Simulaciones Monte Carlo. Por defecto usa config
            seed: Semilla para reproducibilidad. Por defecto usa config
        """
        if confidence_level is None:
            confidence_level = DEFAULT_CONFIDENCE_LEVEL
        
        if n_simulations is None:
            n_simulations = MONTE_CARLO_SIMULATIONS
        
        if seed is None:
            seed = MONTE_CARLO_SEED

        results = self.analyzer.calculate_multi_level(
            returns=returns,
            weights=weights,
            confidence_levels=(confidence_level,),
            methods=['historical', 'parametric', 'monte_carlo'],
            n_simulations=n_simulations,
            seed=seed
        )

        comparison = self._results_to_dataframe(results[confidence_level])
        self.print_comparison(comparison, confidence_level)
    
    def _results_to_dataframe(self, method_results: Dict) -> pd.DataFrame:
        """Convierte resultados a DataFrame para visualización."""
        data = []
        for method, values in method_results.items():
            data.append({
                'Método': method.capitalize(),
                'VaR_daily_%': values['var_daily_pct'],
                'VaR_annual_%': values['var_annual_pct'],
                'ES_daily_%': values['es_daily_pct'],
                'ES_annual_%': values['es_annual_pct']
            })
        
        return pd.DataFrame(data).set_index('Método')
    
    def print_comparison(
        self,
        comparison: pd.DataFrame,
        confidence_level: float = None
    ) -> None:
        """Imprime comparación formateada de VaR y ES."""
        if confidence_level is None:
            confidence_level = DEFAULT_CONFIDENCE_LEVEL

        print(f"ANÁLISIS VaR y ES (Nivel de confianza: {confidence_level*100:.0f}%)".center(70))

        print("COMPARACIÓN DE MÉTODOS")
        print(f"{'Método':<15} {'VaR Diario':<15} {'VaR Anual':<15} {'ES Diario':<15} {'ES Anual':<15}")
        
        for method, row in comparison.iterrows():
            print(f"{method:<15} "
                  f"{row['VaR_daily_%']:>7.2f}%      "
                  f"{row['VaR_annual_%']:>7.2f}%      "
                  f"{row['ES_daily_%']:>7.2f}%      "
                  f"{row['ES_annual_%']:>7.2f}%")

        avg_var_daily = comparison['VaR_daily_%'].mean()
        avg_es_daily = comparison['ES_daily_%'].mean()
        
        print("INTERPRETACIÓN")
        print(f"VaR promedio diario:     {avg_var_daily:.2f}%")
        print(f"ES promedio diario:      {avg_es_daily:.2f}%")
        print(f"Pérdida máxima esperada: {avg_es_daily:.2f}% en un día adverso")
 
        abs_var = abs(avg_var_daily)
        if abs_var < VAR_RISK_LEVELS['low']:
            risk_level = "Bajo"
        elif abs_var < VAR_RISK_LEVELS['moderate']:
            risk_level = "Moderado"
        elif abs_var < VAR_RISK_LEVELS['high']:
            risk_level = "Alto"
        else:
            risk_level = "Muy Alto"
        
        print(f"  Nivel de riesgo:         {risk_level}")

    def generate_multi_level_report(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        confidence_levels: tuple = None,
        method: str = 'historical',
        n_simulations: int = None,
        seed: int = None
    ) -> None:
        """
        Genera reporte de VaR y ES para múltiples niveles de confianza.
        
        Args:
            returns: DataFrame de retornos diarios
            weights: Array de pesos del portafolio
            confidence_levels: Tupla de niveles de confianza
            method: Método a utilizar
            n_simulations: Simulaciones Monte Carlo. Por defecto usa config
            seed: Semilla para reproducibilidad. Por defecto usa config
        """
        if confidence_levels is None:
            confidence_levels = (0.90, DEFAULT_CONFIDENCE_LEVEL, 0.99)
        
        if n_simulations is None:
            n_simulations = MONTE_CARLO_SIMULATIONS
        
        if seed is None:
            seed = MONTE_CARLO_SEED

        results = self.analyzer.calculate_multi_level(
            returns=returns,
            weights=weights,
            confidence_levels=confidence_levels,
            methods=[method],
            n_simulations=n_simulations,
            seed=seed
        )
        
        print(f"VaR y ES - Método: {method.capitalize()}".center(70))
        print(f"{'Confianza':<12} {'VaR Diario':<15} {'VaR Anual':<15} {'ES Diario':<15} {'ES Anual':<15}")

        for cl in confidence_levels:
            values = results[cl][method]
            print(f"{cl*100:>5.0f}%       "
                  f"{values['var_daily_pct']:>7.2f}%      "
                  f"{values['var_annual_pct']:>7.2f}%      "
                  f"{values['es_daily_pct']:>7.2f}%      "
                  f"{values['es_annual_pct']:>7.2f}%")