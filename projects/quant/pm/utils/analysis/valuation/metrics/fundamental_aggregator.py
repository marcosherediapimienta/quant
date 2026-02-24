from typing import Dict
from ....tools.config import SCORE_AGGREGATION_WEIGHTS

class FundamentalAggregator:
    _REQUIRED_KEYS = ('profitability', 'health', 'growth')

    def __init__(self, weights: Dict[str, float] = None):
        default_weights = SCORE_AGGREGATION_WEIGHTS['fundamental'].copy()
        merged_weights = default_weights.copy()
        if weights:
            merged_weights.update(weights)

        self.weights = {k: float(merged_weights[k]) for k in self._REQUIRED_KEYS}

        total = sum(self.weights.values())
        if total <= 0:
            self.weights = {k: float(default_weights[k]) for k in self._REQUIRED_KEYS}
        elif abs(total - 1.0) > 0.01:
            self.weights = {k: v / total for k, v in self.weights.items()}
    
    def aggregate(
        self,
        profitability_score: float,
        health_score: float,
        growth_score: float
    ) -> float:

        score_map = {
            'profitability': profitability_score,
            'health': health_score,
            'growth': growth_score,
        }
        valid_items = [
            (key, float(value))
            for key, value in score_map.items()
            if value is not None and value == value
        ]

        if not valid_items:
            return 0.0

        valid_weight_sum = sum(self.weights[key] for key, _ in valid_items)
        if valid_weight_sum <= 0:
            return 0.0

        return sum(
            value * (self.weights[key] / valid_weight_sum)
            for key, value in valid_items
        )