import numpy as np
import pandas as pd
from typing import Dict
from ..analyzers.var_es_analyzer import VarEsAnalyzer

class VarEsReporter:

    def __init__(self, var_es_analyzer: VarEsAnalyzer):
        self.analyzer = var_es_analyzer
    
    def generate_report(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        confidence_level: float = 0.95,
        n_simulations: int = 10000,
        seed: int = 42
    ) -> None:

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
        confidence_level: float = 0.95
    ) -> None:

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
 
        if abs(avg_var_daily) < 2:
            risk_level = "Bajo"
        elif abs(avg_var_daily) < 5:
            risk_level = "Moderado"
        elif abs(avg_var_daily) < 10:
            risk_level = "Alto"
        else:
            risk_level = "Muy Alto"
        
        print(f"  Nivel de riesgo:         {risk_level}")

    def generate_multi_level_report(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        confidence_levels: tuple = (0.90, 0.95, 0.99),
        method: str = 'historical',
        n_simulations: int = 10000,
        seed: int = 42
    ) -> None:

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