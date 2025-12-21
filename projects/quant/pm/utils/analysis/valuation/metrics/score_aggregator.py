from typing import Dict

class ScoreAggregator:

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            'valuation': 0.40,      
            'fundamental': 0.50,   
            'technical': 0.10
        }
    
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