from typing import Dict
from ..analyzers.macro_correlation_analyzer import MacroCorrelationAnalyzer

class MacroCorrelationReporter:

    def __init__(self, analyzer: MacroCorrelationAnalyzer):
        self.analyzer = analyzer

    def print_analysis(self, results: Dict) -> None:
        print("ANÁLISIS DE CORRELACIONES MACRO".center(70))
        self._print_best_correlations(results)
        self._print_lagged_factors(results)
    
    def _print_best_correlations(self, results: Dict) -> None:
        best_lags = results['best_lagged_correlations']

        print("MEJORES CORRELACIONES (con lag óptimo)")
        print(f"{'Factor':<20} {'Corr':>8} {'Lag':>6} {'t-stat':>10} {'p-value':>10}")

        for _, row in best_lags.head(10).iterrows():
            print(f"{row['factor']:<20} {row['corr']:>8.3f} {int(row['lag']):>6d} "
                  f"{row['t']:>10.3f} {row['p']:>10.4f}")

    def _print_lagged_factors(self, results: Dict) -> None:
        best_lags = results['best_lagged_correlations']
        leading = best_lags[best_lags['lag'] < 0].head(5)
        if not leading.empty:
            print("\nFACTORES LEADING (predicen portafolio)")
            for _, row in leading.iterrows():
                print(f"  • {row['factor']}: {row['lag']} días adelantado "
                      f"(corr={row['corr']:.3f})")

        lagging = best_lags[best_lags['lag'] > 0].head(5)
        if not lagging.empty:
            print("FACTORES LAGGING (siguen al portafolio)")
            for _, row in lagging.iterrows():
                print(f"  • {row['factor']}: {row['lag']} días retrasado "
                      f"(corr={row['corr']:.3f})")