import math
from typing import Dict
from ....tools.config import SCORE_AGGREGATION_WEIGHTS

class ScoreAggregator:
    _REQUIRED_KEYS = ('valuation', 'fundamental', 'technical')

    def __init__(self, weights: Dict[str, float] = None):
        default_weights = SCORE_AGGREGATION_WEIGHTS['total'].copy()
        if weights:
            default_weights.update(weights)

        self.weights = {
            key: float(default_weights.get(key, 0.0))
            for key in self._REQUIRED_KEYS
        }

        total = sum(self.weights.values())
        if total <= 0:
            base = SCORE_AGGREGATION_WEIGHTS['total']
            self.weights = {k: float(base.get(k, 0.0)) for k in self._REQUIRED_KEYS}
            total = sum(self.weights.values())

        if abs(total - 1.0) > 0.01:
            self.weights = {k: v / total for k, v in self.weights.items()}
    
    def aggregate(
        self, 
        valuation_score: float,
        fundamental_score: float,
        technical_score: float
    ) -> float:
        scores = {
            'valuation': valuation_score,
            'fundamental': fundamental_score,
            'technical': technical_score,
        }
        valid_scores = {
            key: value for key, value in scores.items()
            if self._is_finite_number(value)
        }
        if not valid_scores:
            return 0.0

        effective_weight_sum = sum(self.weights[key] for key in valid_scores.keys())
        if effective_weight_sum <= 0:
            return 0.0

        return sum(
            valid_scores[key] * (self.weights[key] / effective_weight_sum)
            for key in valid_scores.keys()
        )

    @staticmethod
    def _is_finite_number(value: float) -> bool:
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False