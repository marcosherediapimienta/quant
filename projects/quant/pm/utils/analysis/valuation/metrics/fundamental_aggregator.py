class FundamentalAggregator:

    def __init__(self, weights: dict = None):
        self.weights = weights or {
            'profitability': 0.35,
            'health': 0.35,
            'growth': 0.30
        }
    
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