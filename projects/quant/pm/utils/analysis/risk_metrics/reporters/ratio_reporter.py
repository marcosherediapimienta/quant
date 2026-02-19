import numpy as np
import pandas as pd
from typing import Dict
from ..analyzers.ratio_analyzer import RatioAnalyzer
from ....tools.config import RATIO_INTERPRETATION, SORTINO_THRESHOLDS

class RatioReporter:
    def __init__(self, ratio_analyzer: RatioAnalyzer):
        self.analyzer = ratio_analyzer

    def generate_report(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        risk_free_rate: float,
        ddof: int = 0,
    ) -> None:
        results = self.analyzer.calculate_all_ratios(returns, weights, risk_free_rate, ddof)
        self._print_ratios(results)

    @staticmethod
    def _classify_sharpe(sharpe: float) -> str:
        t = RATIO_INTERPRETATION['sharpe']
        if sharpe > t['excellent']:
            return "Excelente"
        if sharpe > t['very_good']:
            return "Muy bueno"
        if sharpe > t['acceptable']:
            return "Aceptable"
        return "Por debajo del aceptable"

    @staticmethod
    def _classify_sortino(sortino: float) -> str:
        if sortino > SORTINO_THRESHOLDS['excellent']:
            return "Excelente control de downside"
        if sortino > SORTINO_THRESHOLDS['good']:
            return "Buen control de downside"
        if sortino > SORTINO_THRESHOLDS['acceptable']:
            return "Control aceptable"
        return "Downside risk elevado"

    def _print_ratios(self, results: Dict) -> None:
        print("ANÁLISIS DE RATIOS DE RENDIMIENTO".center(60))

        print("SHARPE RATIO")
        print(f"  Valor:                   {results['sharpe_ratio']:>8.3f}")
        print(f"  Interpretación:          {self._classify_sharpe(results['sharpe_ratio'])}")

        print("SORTINO RATIO")
        print(f"  Valor:                   {results['sortino_ratio']:>8.3f}")
        print(f"  Interpretación:          {self._classify_sortino(results['sortino_ratio'])}")

        print("MÉTRICAS ADICIONALES")
        print(f"  Retorno Anual:           {results['annual_return'] * 100:>8.2f}%")
        print(f"  Volatilidad Anual:       {results['annual_volatility'] * 100:>8.2f}%")
        print(f"  Volatilidad Downside:    {results['downside_volatility'] * 100:>8.2f}%")
        print(f"  Exceso de Retorno:       {results['excess_return'] * 100:>8.2f}%")

    @staticmethod
    def _print_rolling_stats(label: str, series: pd.Series) -> None:
        print(label)
        print(f"  Promedio:                {series.mean():>8.3f}")
        print(f"  Mínimo:                  {series.min():>8.3f}")
        print(f"  Máximo:                  {series.max():>8.3f}")
        print(f"  Último:                  {series.iloc[-1]:>8.3f}")

    def print_rolling_summary(self, rolling_data: pd.DataFrame) -> None:
        print("RESUMEN DE MÉTRICAS ROLLING".center(60))
        self._print_rolling_stats("SHARPE RATIO ROLLING", rolling_data['sharpe_rolling'])
        self._print_rolling_stats("SORTINO RATIO ROLLING", rolling_data['sortino_rolling'])
