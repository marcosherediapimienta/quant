import numpy as np
import pandas as pd
from typing import Dict
from ..analyzers.ratio_analyzer import RatioAnalyzer
from ....tools.config import RATIO_INTERPRETATION, SORTINO_THRESHOLDS

class RatioReporter:
    """
    Reporter para generar informes de ratios de rendimiento/riesgo.
    
    Responsabilidad: Formatear y presentar resultados de ratios de forma legible.
    """

    def __init__(self, ratio_analyzer: RatioAnalyzer):
        """
        Args:
            ratio_analyzer: Instancia de RatioAnalyzer para cálculos
        """
        self.analyzer = ratio_analyzer
    
    def generate_report(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        risk_free_rate: float,
        ddof: int = 0
    ) -> None:
        """
        Genera reporte de todos los ratios de rendimiento/riesgo.
        
        Args:
            returns: DataFrame de retornos diarios
            weights: Array de pesos del portafolio
            risk_free_rate: Tasa libre de riesgo anualizada
            ddof: Grados de libertad para cálculos estadísticos
        """
        results = self.analyzer.calculate_all_ratios(
            returns, weights, risk_free_rate, ddof
        )
        self.print_ratios(results)
    
    def print_ratios(self, results: Dict) -> None:
        """Imprime ratios formateados con interpretaciones."""
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
        """Interpreta el Sharpe Ratio usando thresholds de config."""
        thresholds = RATIO_INTERPRETATION['sharpe']
        if sharpe > thresholds['excellent']:
            print("  Interpretación:          Excelente")
        elif sharpe > thresholds['very_good']:
            print("  Interpretación:          Muy bueno")
        elif sharpe > thresholds['acceptable']:
            print("  Interpretación:          Aceptable")
        else:
            print("  Interpretación:          Por debajo del aceptable")

    def _interpret_sortino(self, sortino: float) -> None:
        """Interpreta el Sortino Ratio usando thresholds de config."""
        if sortino > SORTINO_THRESHOLDS['excellent']:
            print("  Interpretación:          Excelente control de downside")
        elif sortino > SORTINO_THRESHOLDS['good']:
            print("  Interpretación:          Buen control de downside")
        elif sortino > SORTINO_THRESHOLDS['acceptable']:
            print("  Interpretación:          Control aceptable")
        else:
            print("  Interpretación:          Downside risk elevado")
    
    def print_rolling_summary(self, rolling_data: pd.DataFrame) -> None:
        """Imprime resumen de métricas rolling."""
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