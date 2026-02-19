import numpy as np
import pandas as pd
from typing import Dict
from ..analyzers.drawdown_analyzer import DrawdownAnalyzer
from ....tools.config import DRAWDOWN_RISK_LEVELS, RATIO_INTERPRETATION

class DrawdownReporter:
    def __init__(self, drawdown_analyzer: DrawdownAnalyzer):
        self.analyzer = drawdown_analyzer

    def generate_report(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        risk_free_rate: float = None,
    ) -> None:
        results = self.analyzer.analyze(returns, weights, risk_free_rate)
        self._print_drawdown(results)

    @staticmethod
    def _classify_drawdown_risk(max_dd_pct: float) -> str:
        if max_dd_pct < DRAWDOWN_RISK_LEVELS['low']:
            return "Low"
        if max_dd_pct < DRAWDOWN_RISK_LEVELS['moderate']:
            return "Moderate"
        if max_dd_pct < DRAWDOWN_RISK_LEVELS['high']:
            return "High"
        return "Very High"

    @staticmethod
    def _classify_calmar(calmar: float) -> str:
        t = RATIO_INTERPRETATION['calmar']
        if calmar > t['excellent']:
            return "Excellent return vs drawdown"
        if calmar > t['good']:
            return "Good compensation"
        return "Elevated risk for the return"

    def _print_drawdown(self, r: Dict) -> None:
        print("DRAWDOWN ANALYSIS".center(60))

        print("MAX DRAWDOWN")
        print(f"  Magnitude:               {r['max_drawdown'] * 100:>8.2f}%")
        print(f"  Date:                    {r['max_drawdown_date']}")
        print(f"  Duration:                {r['max_underwater_duration']} days")

        print("DRAWDOWN RATIOS")
        print(f"  Calmar Ratio:            {r['calmar_ratio']:>8.3f}")
        print(f"  Sterling Ratio:          {r['sterling_ratio']:>8.3f}")

        print("ANNUAL RETURN")
        print(f"  Annual Return:           {r['annual_return'] * 100:>8.2f}%")

        print("INTERPRETATION")
        print(f"  Risk level:              {self._classify_drawdown_risk(abs(r['max_drawdown']) * 100)}")
        print(f"  Calmar:                  {self._classify_calmar(r['calmar_ratio'])}")
