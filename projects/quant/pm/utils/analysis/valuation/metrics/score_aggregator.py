from typing import Dict
from ....tools.config import SCORE_AGGREGATION_WEIGHTS

class ScoreAggregator:
    """
    Agrega scores parciales en un score total ponderado.
    
    Responsabilidad: Combinar valuation_score, fundamental_score y technical_score
    según pesos configurables.
    
    Metodología:
    - Score total = weighted average de los tres pilares
    - Pesos por defecto: 40% valoración, 50% fundamental, 10% técnico
    """

    def __init__(self, weights: Dict[str, float] = None):
        """
        Args:
            weights: Diccionario con pesos {'valuation', 'fundamental', 'technical'}.
                    Por defecto usa SCORE_AGGREGATION_WEIGHTS['total']
        """
        default_weights = SCORE_AGGREGATION_WEIGHTS['total']
        self.weights = weights or default_weights.copy()
        
        # Validar que sumen ~1.0
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            # Normalizar si no suman 1
            self.weights = {k: v/total for k, v in self.weights.items()}
    
    def aggregate(
        self, 
        valuation_score: float,
        fundamental_score: float,
        technical_score: float
    ) -> float:
        """
        Agrega scores parciales en score total.
        
        Args:
            valuation_score: Score de valoración (0-100)
            fundamental_score: Score fundamental (0-100)
            technical_score: Score técnico (0-100)
            
        Returns:
            Score total ponderado (0-100)
        """
        return (
            valuation_score * self.weights['valuation'] +
            fundamental_score * self.weights['fundamental'] +
            technical_score * self.weights['technical']
        )