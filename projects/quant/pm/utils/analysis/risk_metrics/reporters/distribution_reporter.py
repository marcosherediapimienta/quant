import pandas as pd
import numpy as np
from typing import Dict
from ..analyzers.distribution_analyzer import DistributionAnalyzer


class DistributionReporter:

    def __init__(self, distribution_analyzer: DistributionAnalyzer):
        self.analyzer = distribution_analyzer
    
    def generate_report(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray
    ) -> None:
  
        results = self.analyzer.analyze(returns, weights)
        self.print_distribution(results)
    
    def print_distribution(self, results: Dict) -> None:
  
        print("ANÁLISIS DE DISTRIBUCIÓN".center(60))

        print("ASIMETRÍA (Skewness)")
        print(f"  Valor:                   {results['skewness']:>8.3f}")
        self._interpret_skewness(results['skewness'])
        
        print("CURTOSIS (Excess Kurtosis)")
        print(f"  Valor:                   {results['excess_kurtosis']:>8.3f}")  
        self._interpret_kurtosis(results['excess_kurtosis'])  
        
        print("TEST DE NORMALIDAD (Jarque-Bera)")
        print(f"  Estadístico JB:          {results['jb_statistic']:>8.2f}")  
        print(f"  p-value:                 {results['jb_p_value']:>8.4f}")  
        is_normal = results['is_normal'] 
        print(f"  Distribución normal:     {'[SI]' if is_normal else '[NO]'}")

    def _interpret_skewness(self, skew: float) -> None:

        if skew > 0.5:
            print(f"  Interpretación:          Asimetría positiva (cola derecha)")
            print(f"                           -> Más ganancias extremas que pérdidas")
        elif skew < -0.5:
            print(f"  Interpretación:          Asimetría negativa (cola izquierda)")
            print(f"                           -> Más pérdidas extremas que ganancias")
        else:
            print(f"  Interpretación:          Aproximadamente simétrica")
    
    def _interpret_kurtosis(self, kurt: float) -> None:

        if kurt > 3:
            print(f"  Interpretación:          Leptocúrtica (colas pesadas)")
            print(f"                           -> Mayor riesgo de eventos extremos")
        elif kurt < -1:
            print(f"  Interpretación:          Platicúrtica (colas ligeras)")
            print(f"                           -> Menos eventos extremos")
        else:
            print(f"  Interpretación:          Similar a distribución normal")