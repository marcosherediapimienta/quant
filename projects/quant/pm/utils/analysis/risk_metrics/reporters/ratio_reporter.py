import numpy as np
import pandas as pd
from typing import Dict
from ..analyzers.ratio_analyzer import RatioAnalyzer
from ....tools.config import RATIO_INTERPRETATION, SORTINO_THRESHOLDS

class RatioReporter:
    def __init__(self, ratio_analyzer: RatioAnalyzer):
        self.analyzer = ratio_analyzer

    def generate_report(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        risk_free_rate: float,
        ddof: int = 0,
    ) -> None:
        results = self.analyzer.calculate_all_ratios(returns, weights, risk_free_rate, ddof)
        self._print_ratios(results)

    @staticmethod
    def _classify_sharpe(sharpe: float) -> str:
        t = RATIO_INTERPRETATION['sharpe']
        if sharpe > t['excellent']:
            return "Excellent"
        if sharpe > t['very_good']:
            return "Very good"
        if sharpe > t['acceptable']:
            return "Acceptable"
        return "Below acceptable"

    @staticmethod
    def _classify_sortino(sortino: float) -> str:
        if sortino > SORTINO_THRESHOLDS['excellent']:
            return "Excellent downside control"
        if sortino > SORTINO_THRESHOLDS['good']:
            return "Good downside control"
        if sortino > SORTINO_THRESHOLDS['acceptable']:
            return "Acceptable control"
        return "Elevated downside risk"

    def _print_ratios(self, results: Dict) -> None:
        print("PERFORMANCE RATIOS ANALYSIS".center(60))

        print("SHARPE RATIO")
        print(f"  Value:                   {results['sharpe_ratio']:>8.3f}")
        print(f"  Interpretation:          {self._classify_sharpe(results['sharpe_ratio'])}")

        print("SORTINO RATIO")
        print(f"  Value:                   {results['sortino_ratio']:>8.3f}")
        print(f"  Interpretation:          {self._classify_sortino(results['sortino_ratio'])}")

        print("ADDITIONAL METRICS")
        print(f"  Annual Return:           {results['annual_return'] * 100:>8.2f}%")
        print(f"  Annual Volatility:       {results['annual_volatility'] * 100:>8.2f}%")
        print(f"  Downside Volatility:     {results['downside_volatility'] * 100:>8.2f}%")
        print(f"  Excess Return:           {results['excess_return'] * 100:>8.2f}%")

    @staticmethod
    def _print_rolling_stats(label: str, series: pd.Series) -> None:
        print(label)
        print(f"  Average:                 {series.mean():>8.3f}")
        print(f"  Minimum:                 {series.min():>8.3f}")
        print(f"  Maximum:                 {series.max():>8.3f}")
        print(f"  Latest:                  {series.iloc[-1]:>8.3f}")

    def print_rolling_summary(self, rolling_data: pd.DataFrame) -> None:
        print("ROLLING METRICS SUMMARY".center(60))
        self._print_rolling_stats("ROLLING SHARPE RATIO", rolling_data['sharpe_rolling'])
        self._print_rolling_stats("ROLLING SORTINO RATIO", rolling_data['sortino_rolling'])
