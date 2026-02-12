from typing import Dict
from ....tools.config import SCORE_AGGREGATION_WEIGHTS

class FundamentalAggregator:
    def __init__(self, weights: Dict[str, float] = None):
        default_weights = SCORE_AGGREGATION_WEIGHTS['fundamental']
        self.weights = weights or default_weights.copy()

        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            self.weights = {k: v/total for k, v in self.weights.items()}
    
    def aggregate(
        self,
        profitability_score: float,
        health_score: float,
        growth_score: float
    ) -> float:

        return (
            profitability_score * self.weights['profitability'] +
            health_score * self.weights['health'] +
            growth_score * self.weights['growth']
        )