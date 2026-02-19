import numpy as np
import pandas as pd
from typing import Dict
from ..analyzers.drawdown_analyzer import DrawdownAnalyzer
from ....tools.config import DRAWDOWN_RISK_LEVELS, RATIO_INTERPRETATION

class DrawdownReporter:
    def __init__(self, drawdown_analyzer: DrawdownAnalyzer):
        self.analyzer = drawdown_analyzer

    def generate_report(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
    ) -> None:
        results = self.analyzer.analyze(returns, weights)
        self._print_drawdown(results)

    @staticmethod
    def _classify_drawdown_risk(max_dd_pct: float) -> str:
        if max_dd_pct < DRAWDOWN_RISK_LEVELS['low']:
            return "Bajo"
        if max_dd_pct < DRAWDOWN_RISK_LEVELS['moderate']:
            return "Moderado"
        if max_dd_pct < DRAWDOWN_RISK_LEVELS['high']:
            return "Alto"
        return "Muy Alto"

    @staticmethod
    def _classify_calmar(calmar: float) -> str:
        t = RATIO_INTERPRETATION['calmar']
        if calmar > t['excellent']:
            return "Excelente retorno vs drawdown"
        if calmar > t['good']:
            return "Buena compensación"
        return "Riesgo elevado para el retorno"

    def _print_drawdown(self, r: Dict) -> None:
        print("ANÁLISIS DE DRAWDOWN".center(60))

        print("MAX DRAWDOWN")
        print(f"  Magnitud:                {r['max_drawdown'] * 100:>8.2f}%")
        print(f"  Fecha:                   {r['max_drawdown_date']}")
        print(f"  Duración:                {r['max_underwater_duration']} días")

        print("RATIOS DE DRAWDOWN")
        print(f"  Calmar Ratio:            {r['calmar_ratio']:>8.3f}")
        print(f"  Sterling Ratio:          {r['sterling_ratio']:>8.3f}")

        print("RETORNO ANUAL")
        print(f"  Retorno Anual:           {r['annual_return'] * 100:>8.2f}%")

        print("INTERPRETACIÓN")
        print(f"  Nivel de riesgo:         {self._classify_drawdown_risk(abs(r['max_drawdown']) * 100)}")
        print(f"  Calmar:                  {self._classify_calmar(r['calmar_ratio'])}")
