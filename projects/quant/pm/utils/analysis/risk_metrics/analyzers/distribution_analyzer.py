import pandas as pd
import numpy as np
from typing import Dict


class DistributionAnalyzer:

    def __init__(self, risk_analysis):
        self.risk_analysis = risk_analysis
        self.moments_calc = risk_analysis.moments
    
    def analyze(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray
    ) -> Dict[str, float]:

        return self.moments_calc.calculate_all(returns, weights)