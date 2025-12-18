import pandas as pd
import numpy as np
from typing import Dict, Tuple
from ..analyzers import ComprehensiveAnalyzer


class RiskReporter:

    def __init__(self, risk_analysis):
        self.comprehensive = ComprehensiveAnalyzer(risk_analysis)
    
    def print_full_report(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        risk_free_rate: float,
        confidence_levels: Tuple[float, ...] = (0.95, 0.99),
        var_method: str = 'historical',
        ddof: int = 0
    ) -> Dict[str, any]:
  
        print("REPORTE COMPLETO DE ANÁLISIS DE RIESGO")
        summary = self.comprehensive.calculate_summary(
            returns, weights, risk_free_rate, confidence_levels, var_method, ddof
        )

        print("\nPERFORMANCE")
        print(f"Retorno anualizado:     {summary['annual_return']*100:>8.2f}%")
        print(f"Volatilidad anualizada: {summary['annual_volatility']*100:>8.2f}%")
        print(f"Vol. Downside:          {summary['downside_volatility']*100:>8.2f}%")
 
        print("RATIOS")
        sharpe = summary['sharpe_ratio']
        sortino = summary['sortino_ratio']
        print(f"Sharpe Ratio:           {sharpe:>8.3f}" if sharpe is not None else "Sharpe:  N/A")
        print(f"Sortino Ratio:          {sortino:>8.3f}" if sortino is not None else "Sortino: N/A")
        
        # Distribución
        print("DISTRIBUCIÓN")
        print(f"Skewness:               {summary['skewness']:>8.3f}")
        print(f"Excess Kurtosis:        {summary['excess_kurtosis']:>8.3f}")
        print(f"¿Normal?:               {'Sí' if summary['is_normal_distribution'] else 'No':>8}")
        
        # VaR/ES
        print("VAR & ES")
        print(f"Método: {var_method.capitalize()}")
        for cl in confidence_levels:
            cl_key = int(cl * 100)
            print(f"\n{cl_key}%:")
            print(f"  VaR: {summary[f'VaR_{cl_key}_pct']:>7.2f}%")
            print(f"  ES:  {summary[f'ES_{cl_key}_pct']:>7.2f}%")
        
        return summary