import pandas as pd
import numpy as np
from typing import Dict
from ..analyzers.distribution_analyzer import DistributionAnalyzer
from ....tools.config import SKEWNESS_THRESHOLDS, KURTOSIS_THRESHOLDS

class DistributionReporter:
    def __init__(self, distribution_analyzer: DistributionAnalyzer):
        self.analyzer = distribution_analyzer

    def generate_report(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
    ) -> None:
        results = self.analyzer.analyze(returns, weights)
        self._print_distribution(results)

    @staticmethod
    def _classify_skewness(skew: float) -> str:
        if skew > SKEWNESS_THRESHOLDS['positive']:
            return "Positive skew (right tail) -> More extreme gains than losses"
        if skew < SKEWNESS_THRESHOLDS['negative']:
            return "Negative skew (left tail) -> More extreme losses than gains"
        return "Approximately symmetric"

    @staticmethod
    def _classify_kurtosis(kurt: float) -> str:
        if kurt > KURTOSIS_THRESHOLDS['leptokurtic']:
            return "Leptokurtic (heavy tails) -> Higher risk of extreme events"
        if kurt < KURTOSIS_THRESHOLDS['platykurtic']:
            return "Platykurtic (light tails) -> Fewer extreme events"
        return "Similar to normal distribution"

    def _print_distribution(self, r: Dict) -> None:
        print("DISTRIBUTION ANALYSIS".center(60))

        print("SKEWNESS")
        print(f"  Value:                   {r['skewness']:>8.3f}")
        print(f"  Interpretation:          {self._classify_skewness(r['skewness'])}")

        print("KURTOSIS (Excess Kurtosis)")
        print(f"  Value:                   {r['excess_kurtosis']:>8.3f}")
        print(f"  Interpretation:          {self._classify_kurtosis(r['excess_kurtosis'])}")

        print("NORMALITY TEST (Jarque-Bera)")
        print(f"  JB Statistic:            {r['jb_statistic']:>8.2f}")
        print(f"  p-value:                 {r['jb_p_value']:>8.4f}")
        print(f"  Normal distribution:     {'[YES]' if r['is_normal'] else '[NO]'}")
