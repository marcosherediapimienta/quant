from typing import Dict
from ..analyzers.macro_sensitivity_analyzer import MacroSensitivityAnalyzer


class MacroSensitivityReporter:

    def __init__(self, analyzer: MacroSensitivityAnalyzer):
        self.analyzer = analyzer
    
    def print_analysis(self, results: Dict) -> None:
        print("ANÁLISIS DE SENSIBILIDADES MACRO".center(70))
        self._print_exposures(results)
        self._print_dominant_factor(results)
    
    def _print_exposures(self, results: Dict) -> None:
        print("EXPOSICIONES POR MAGNITUD")

        high_exp = results['high_exposure']
        if high_exp:
            print("ALTA EXPOSICIÓN (|β| > 0.5)")
            for factor, beta in high_exp.items(): 
                direction = "↑" if beta > 0 else "↓"
                print(f"  {direction} {factor}: β = {beta:>7.3f}")

        mod_exp = results['moderate_exposure']
        if mod_exp:
            print("EXPOSICIÓN MODERADA (0.2 ≤ |β| ≤ 0.5)")
            for factor, beta in mod_exp.items():  
                direction = "↑" if beta > 0 else "↓"
                print(f"  {direction} {factor}: β = {beta:>7.3f}")

        low_exp = results['low_exposure']
        if low_exp:
            print("BAJA EXPOSICIÓN (|β| < 0.2)")

            for i, (factor, beta) in enumerate(low_exp.items()):  
                if i >= 5:
                    break
                print(f"    {factor}: β = {beta:>7.3f}")
    
    def _print_dominant_factor(self, results: Dict) -> None:
        dominant = results['dominant_factor']
        
        if dominant:
            factor, beta = dominant
            print("FACTOR DOMINANTE")
            print(f"  {factor}: β = {beta:.3f}")

            if abs(beta) > 1.0:
                print(f"  Interpretación: Exposición muy alta")
            elif abs(beta) > 0.5:
                print(f"  Interpretación: Exposición significativa")
            else:
                print(f"  Interpretación: Exposición moderada")