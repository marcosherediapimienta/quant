import pandas as pd
import numpy as np
from typing import Dict
from ..analyzers.correlation_analyzer import CorrelationAnalyzer
from ....tools.config import CORRELATION_REPORT

_TOP_N = CORRELATION_REPORT['top_n_pairs']

class CorrelationReporter:
    def __init__(self, correlation_analyzer: CorrelationAnalyzer):
        self.analyzer = correlation_analyzer

    def generate_report(self, returns: pd.DataFrame) -> None:
        results = self.analyzer.analyze(returns)
        self._print_summary(results)

    @staticmethod
    def _upper_triangle(corr_matrix: pd.DataFrame) -> pd.Series:
        mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        return corr_matrix.where(mask).stack()

    def _print_summary(self, results: Dict) -> None:
        corr_matrix = results['correlation_matrix']

        print("CORRELATION ANALYSIS".center(60))

        print("CORRELATION MATRIX")
        print(corr_matrix.round(3))

        print("\nCORRELATION STATISTICS")
        print(f"  Average correlation:     {results['mean_correlation']:>8.3f}")
        print(f"  Maximum correlation:     {results['max_correlation']:>8.3f}")
        print(f"  Minimum correlation:     {results['min_correlation']:>8.3f}")
        print(f"  Standard deviation:      {results['std_correlation']:>8.3f}")

        flat = self._upper_triangle(corr_matrix)

        print("MOST CORRELATED PAIRS")
        for (a1, a2), corr in flat.nlargest(_TOP_N).items():
            print(f"  {a1} - {a2}:  {corr:>6.3f}")

        print("LEAST CORRELATED PAIRS")
        for (a1, a2), corr in flat.nsmallest(_TOP_N).items():
            print(f"  {a1} - {a2}:  {corr:>6.3f}")
