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
        weights: np.ndarray
    ) -> None:

        results = self.analyzer.analyze(returns, weights)
        self.print_drawdown(results)
    
    def print_drawdown(self, results: Dict) -> None:

        print("ANÁLISIS DE DRAWDOWN".center(60))

        print("MAX DRAWDOWN")
        print(f"  Magnitud:                {results['max_drawdown']*100:>8.2f}%")
        print(f"  Fecha:                   {results['max_drawdown_date']}")
        print(f"  Duración:                {results['max_underwater_duration']} días")
        
        print("RATIOS DE DRAWDOWN")
        print(f"  Calmar Ratio:            {results['calmar_ratio']:>8.3f}")
        print(f"  Sterling Ratio:          {results['sterling_ratio']:>8.3f}")
        
        print("RETORNO ANUAL")
        print(f"  Retorno Anual:           {results['annual_return']*100:>8.2f}%")
  
        self._interpret_drawdown(results)

    def _interpret_drawdown(self, results: Dict) -> None:

        print("INTERPRETACIÓN")
        
        max_dd = abs(results['max_drawdown']) * 100
        if max_dd < DRAWDOWN_RISK_LEVELS['low']:
            risk_level = "Bajo"
        elif max_dd < DRAWDOWN_RISK_LEVELS['moderate']:
            risk_level = "Moderado"
        elif max_dd < DRAWDOWN_RISK_LEVELS['high']:
            risk_level = "Alto"
        else: 
            risk_level = "Muy Alto"
        
        print(f"  Nivel de riesgo:         {risk_level}")

        calmar_thresholds = RATIO_INTERPRETATION['calmar']
        calmar = results['calmar_ratio']
        
        if calmar > calmar_thresholds['excellent']:
            print(f"  Calmar:                  Excelente retorno vs drawdown")
        elif calmar > calmar_thresholds['good']:
            print(f"  Calmar:                  Buena compensación")
        else:
            print(f"  Calmar:                  Riesgo elevado para el retorno")