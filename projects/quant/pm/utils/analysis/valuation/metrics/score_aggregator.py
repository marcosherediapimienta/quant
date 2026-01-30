from typing import Dict
from ....tools.config import SCORE_AGGREGATION_WEIGHTS

class ScoreAggregator:
    def __init__(self, weights: Dict[str, float] = None):
        default_weights = SCORE_AGGREGATION_WEIGHTS['total']
        self.weights = weights or default_weights.copy()

        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            self.weights = {k: v/total for k, v in self.weights.items()}
    
    def aggregate(
        self, 
        valuation_score: float,
        fundamental_score: float,
        technical_score: float
    ) -> float:

        return (
            valuation_score * self.weights['valuation'] +
            fundamental_score * self.weights['fundamental'] +
            technical_score * self.weights['technical']
        )