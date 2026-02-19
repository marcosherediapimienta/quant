import pandas as pd
import numpy as np
from typing import Dict
from ..analyzers.distribution_analyzer import DistributionAnalyzer
from ....tools.config import SKEWNESS_THRESHOLDS, KURTOSIS_THRESHOLDS

class DistributionReporter:
    def __init__(self, distribution_analyzer: DistributionAnalyzer):
        self.analyzer = distribution_analyzer

    def generate_report(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
    ) -> None:
        results = self.analyzer.analyze(returns, weights)
        self._print_distribution(results)

    @staticmethod
    def _classify_skewness(skew: float) -> str:
        if skew > SKEWNESS_THRESHOLDS['positive']:
            return "Asimetría positiva (cola derecha) -> Más ganancias extremas que pérdidas"
        if skew < SKEWNESS_THRESHOLDS['negative']:
            return "Asimetría negativa (cola izquierda) -> Más pérdidas extremas que ganancias"
        return "Aproximadamente simétrica"

    @staticmethod
    def _classify_kurtosis(kurt: float) -> str:
        if kurt > KURTOSIS_THRESHOLDS['leptokurtic']:
            return "Leptocúrtica (colas pesadas) -> Mayor riesgo de eventos extremos"
        if kurt < KURTOSIS_THRESHOLDS['platykurtic']:
            return "Platicúrtica (colas ligeras) -> Menos eventos extremos"
        return "Similar a distribución normal"

    def _print_distribution(self, r: Dict) -> None:
        print("ANÁLISIS DE DISTRIBUCIÓN".center(60))

        print("ASIMETRÍA (Skewness)")
        print(f"  Valor:                   {r['skewness']:>8.3f}")
        print(f"  Interpretación:          {self._classify_skewness(r['skewness'])}")

        print("CURTOSIS (Excess Kurtosis)")
        print(f"  Valor:                   {r['excess_kurtosis']:>8.3f}")
        print(f"  Interpretación:          {self._classify_kurtosis(r['excess_kurtosis'])}")

        print("TEST DE NORMALIDAD (Jarque-Bera)")
        print(f"  Estadístico JB:          {r['jb_statistic']:>8.2f}")
        print(f"  p-value:                 {r['jb_p_value']:>8.4f}")
        print(f"  Distribución normal:     {'[SI]' if r['is_normal'] else '[NO]'}")
