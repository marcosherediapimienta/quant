import pandas as pd
import numpy as np
from typing import Dict
from ..components.momentum import DistributionMoments

class DistributionAnalyzer:
    def __init__(self):
        self.moments_calc = DistributionMoments()
    
    def analyze(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray
    ) -> Dict[str, float]:

        return self.moments_calc.calculate_all(returns, weights)