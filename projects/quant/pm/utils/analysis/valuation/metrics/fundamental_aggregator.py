from typing import Dict
from ....tools.config import SCORE_AGGREGATION_WEIGHTS

class FundamentalAggregator:
    """
    Agrega scores fundamentales en un score fundamental total.
    
    Responsabilidad: Combinar profitability, health y growth scores
    según pesos configurables.
    
    Metodología:
    - Score fundamental = weighted average de los tres pilares
    - Pesos por defecto: 35% rentabilidad, 35% salud, 30% crecimiento
    """

    def __init__(self, weights: Dict[str, float] = None):
        """
        Args:
            weights: Diccionario con pesos {'profitability', 'health', 'growth'}.
                    Por defecto usa SCORE_AGGREGATION_WEIGHTS['fundamental']
        """
        default_weights = SCORE_AGGREGATION_WEIGHTS['fundamental']
        self.weights = weights or default_weights.copy()
        
        # Validar que sumen ~1.0
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            # Normalizar si no suman 1
            self.weights = {k: v/total for k, v in self.weights.items()}
    
    def aggregate(
        self,
        profitability_score: float,
        health_score: float,
        growth_score: float
    ) -> float:
        """
        Agrega scores fundamentales.
        
        Args:
            profitability_score: Score de rentabilidad (0-100)
            health_score: Score de salud financiera (0-100)
            growth_score: Score de crecimiento (0-100)
            
        Returns:
            Score fundamental total (0-100)
        """
        return (
            profitability_score * self.weights['profitability'] +
            health_score * self.weights['health'] +
            growth_score * self.weights['growth']
        )