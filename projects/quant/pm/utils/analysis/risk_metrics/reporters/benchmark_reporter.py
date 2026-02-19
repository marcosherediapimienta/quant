import pandas as pd
import numpy as np
from typing import Dict
from ..analyzers.benchmark_analyzer import BenchmarkAnalyzer
from ....tools.config import (
    TRACKING_ERROR_THRESHOLDS,
    INFORMATION_RATIO_THRESHOLDS,
    BETA_THRESHOLDS,
    ALPHA_THRESHOLDS,
)

class BenchmarkReporter:
    def __init__(self, benchmark_analyzer: BenchmarkAnalyzer):
        self.analyzer = benchmark_analyzer

    def generate_report(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        benchmark_returns: pd.Series,
        risk_free_rate: float,
    ) -> None:
        results = self.analyzer.analyze(returns, weights, benchmark_returns, risk_free_rate)
        self._print_analysis(results)

    @staticmethod
    def _classify_te(te_annual: float) -> str:
        te_pct = te_annual * 100
        if te_pct < TRACKING_ERROR_THRESHOLDS['very_close']:
            return "Very close to benchmark"
        if te_pct < TRACKING_ERROR_THRESHOLDS['moderate']:
            return "Moderate deviation"
        if te_pct < TRACKING_ERROR_THRESHOLDS['active']:
            return "Notable active management"
        return "High deviation from benchmark"

    @staticmethod
    def _classify_ir(ir: float) -> str:
        if ir > INFORMATION_RATIO_THRESHOLDS['excellent']:
            return "Excellent - outperforms benchmark"
        if ir > INFORMATION_RATIO_THRESHOLDS['positive']:
            return "Positive - adds value"
        if ir > INFORMATION_RATIO_THRESHOLDS['slightly_below']:
            return "Slightly below"
        return "Significant underperformance"

    @staticmethod
    def _classify_beta(beta: float) -> str:
        if beta > BETA_THRESHOLDS['aggressive']:
            return "High sensitivity (aggressive)"
        if beta > BETA_THRESHOLDS['market']:
            return "Similar to market"
        if beta > 0:
            return "Low sensitivity (defensive)"
        return "Inverse correlation"

    @staticmethod
    def _classify_alpha(alpha_annual: float) -> str:
        alpha_pct = alpha_annual * 100
        if alpha_pct > ALPHA_THRESHOLDS['excellent']:
            return "Excellent - exceeds expectations"
        if alpha_pct > ALPHA_THRESHOLDS['positive']:
            return "Positive - generates value"
        if alpha_pct > ALPHA_THRESHOLDS['slightly_below']:
            return "Slightly below"
        return "Notable underperformance"

    def _print_analysis(self, r: Dict) -> None:
        print("BENCHMARK ANALYSIS".center(60))

        print("TRACKING ERROR")
        print(f"  Tracking Error (daily):   {r['tracking_error_daily'] * 100:>8.2f}%")
        print(f"  Tracking Error (annual):  {r['tracking_error_annual'] * 100:>8.2f}%")
        print(f"  Interpretation:           {self._classify_te(r['tracking_error_annual'])}")

        print("INFORMATION RATIO")
        print(f"  Information Ratio:        {r['information_ratio']:>8.3f}")
        print(f"  Interpretation:           {self._classify_ir(r['information_ratio'])}")

        print("BETA")
        print(f"  Beta:                     {r['beta']:>8.3f}")
        print(f"  R-squared:                {r['r_squared']:>8.3f}")
        print(f"  Correlation:              {r['correlation']:>8.3f}")
        print(f"  Interpretation:           {self._classify_beta(r['beta'])}")

        print("ALPHA (Jensen)")
        print(f"  Alpha (annualized):       {r['alpha_annual'] * 100:>8.2f}%")
        print(f"  Portfolio return:         {r['portfolio_return_annual'] * 100:>8.2f}%")
        print(f"  Benchmark return:         {r['benchmark_return_annual'] * 100:>8.2f}%")
        print(f"  Expected return (CAPM):   {r['expected_return'] * 100:>8.2f}%")
        print(f"  Interpretation:           {self._classify_alpha(r['alpha_annual'])}")
