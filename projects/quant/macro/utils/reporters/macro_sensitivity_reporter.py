from typing import Dict
from ..analyzers.macro_sensitivity_analyzer import MacroSensitivityAnalyzer

class MacroSensitivityReporter:
    def __init__(self, analyzer: MacroSensitivityAnalyzer):
        self.analyzer = analyzer
    
    def print_analysis(self, results: Dict) -> None:
        print("MACRO SENSITIVITY ANALYSIS".center(70))
        self._print_exposures(results)
        self._print_dominant_factor(results)
    
    def _print_exposures(self, results: Dict) -> None:
        print("EXPOSURES BY MAGNITUDE")

        _SECTIONS = (
            ('high_exposure',     "HIGH EXPOSURE (|β| > 0.5)",          None),
            ('moderate_exposure', "MODERATE EXPOSURE (0.2 ≤ |β| ≤ 0.5)", None),
            ('low_exposure',      "LOW EXPOSURE (|β| < 0.2)",           5),
        )

        for key, header, limit in _SECTIONS:
            items = results[key]
            if not items:
                continue
            print(header)
            for i, (factor, beta) in enumerate(items.items()):
                if limit is not None and i >= limit:
                    break
                direction = "↑" if beta > 0 else "↓"
                print(f"  {direction} {factor}: β = {beta:>7.3f}")
    
    def _print_dominant_factor(self, results: Dict) -> None:
        dominant = results['dominant_factor']
        
        if not dominant:
            return

        factor, beta = dominant
        print("DOMINANT FACTOR")
        print(f"  {factor}: β = {beta:.3f}")

        abs_beta = abs(beta)
        if abs_beta > 1.0:
            interp = "Very high exposure"
        elif abs_beta > 0.5:
            interp = "Significant exposure"
        else:
            interp = "Moderate exposure"
        print(f"  Interpretation: {interp}")
