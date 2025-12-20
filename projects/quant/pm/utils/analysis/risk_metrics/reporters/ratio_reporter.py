import numpy as np
import pandas as pd
from typing import Dict
from ..analyzers.ratio_analyzer import RatioAnalyzer
from ....tools.config import RATIO_INTERPRETATION

class RatioReporter:

    def __init__(self, ratio_analyzer: RatioAnalyzer):
        self.analyzer = ratio_analyzer
    
    def generate_report(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        risk_free_rate: float,
        ddof: int = 0
    ) -> None:

        results = self.analyzer.calculate_all_ratios(
            returns, weights, risk_free_rate, ddof
        )
        self.print_ratios(results)
    
    def print_ratios(self, results: Dict) -> None:

        print("ANÁLISIS DE RATIOS DE RENDIMIENTO".center(60))
        
        print("SHARPE RATIO")
        print(f"  Valor:                   {results['sharpe_ratio']:>8.3f}")
        self._interpret_sharpe(results['sharpe_ratio'])
        
        print("SORTINO RATIO")
        print(f"  Valor:                   {results['sortino_ratio']:>8.3f}")
        self._interpret_sortino(results['sortino_ratio'])
        
        print("MÉTRICAS ADICIONALES")
        print(f"  Retorno Anual:           {results['annual_return']*100:>8.2f}%")
        print(f"  Volatilidad Anual:       {results['annual_volatility']*100:>8.2f}%")
        print(f"  Volatilidad Downside:    {results['downside_volatility']*100:>8.2f}%")
        print(f"  Exceso de Retorno:       {results['excess_return']*100:>8.2f}%")

    
    def _interpret_sharpe(self, sharpe: float) -> None:
        thresholds = RATIO_INTERPRETATION['sharpe']
        if sharpe > thresholds['excellent']:
            print("  Interpretación:          Excelente")
        elif sharpe > thresholds['very_good']:
            print("  Interpretación:          Muy bueno")
        elif sharpe > thresholds['acceptable']:
            print("  Interpretación:          Aceptable")

    def _interpret_sortino(self, sortino: float) -> None:

        if sortino > 2.0:
            print("  Interpretación:          Excelente control de downside")
        elif sortino > 1.0:
            print("  Interpretación:          Buen control de downside")
        elif sortino > 0.5:
            print("  Interpretación:          Control aceptable")
        else:
            print("  Interpretación:          Downside risk elevado")
    
    def print_rolling_summary(self, rolling_data: pd.DataFrame) -> None:

        print("RESUMEN DE MÉTRICAS ROLLING".center(60))
        
        print("SHARPE RATIO ROLLING")
        print(f"  Promedio:                {rolling_data['sharpe_rolling'].mean():>8.3f}")
        print(f"  Mínimo:                  {rolling_data['sharpe_rolling'].min():>8.3f}")
        print(f"  Máximo:                  {rolling_data['sharpe_rolling'].max():>8.3f}")
        print(f"  Último:                  {rolling_data['sharpe_rolling'].iloc[-1]:>8.3f}")
        
        print("SORTINO RATIO ROLLING")
        print(f"  Promedio:                {rolling_data['sortino_rolling'].mean():>8.3f}")
        print(f"  Mínimo:                  {rolling_data['sortino_rolling'].min():>8.3f}")
        print(f"  Máximo:                  {rolling_data['sortino_rolling'].max():>8.3f}")
        print(f"  Último:                  {rolling_data['sortino_rolling'].iloc[-1]:>8.3f}")
