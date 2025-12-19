import numpy as np
import pandas as pd
from typing import Dict
from ..analyzers.drawdown_analyzer import DrawdownAnalyzer


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
        print(f"  Fecha:                   {results['max_drawdown_date']}")  # ✅ Corregido
        print(f"  Duración:                {results['max_underwater_duration']} días")  # ✅ Corregido
        
        print("RATIOS DE DRAWDOWN")
        print(f"  Calmar Ratio:            {results['calmar_ratio']:>8.3f}")
        print(f"  Sterling Ratio:          {results['sterling_ratio']:>8.3f}")
        
        print("RETORNO ANUAL")
        print(f"  Retorno Anual:           {results['annual_return']*100:>8.2f}%")
  
        self._interpret_drawdown(results)

    def _interpret_drawdown(self, results: Dict) -> None:

        print("INTERPRETACIÓN")
        
        max_dd = abs(results['max_drawdown']) * 100
        if max_dd < 10:
            risk_level = "Bajo"
        elif max_dd < 20:
            risk_level = "Moderado"
        elif max_dd < 30:
            risk_level = "Alto"
        else:
            risk_level = "Muy Alto"
        
        print(f"  Nivel de riesgo:         {risk_level}")
        
        calmar = results['calmar_ratio']
        if calmar > 1.0:
            print(f"  Calmar:                  Excelente retorno vs drawdown")
        elif calmar > 0.5:
            print(f"  Calmar:                  Buena compensación")
        else:
            print(f"  Calmar:                  Riesgo elevado para el retorno")