from typing import Dict
from ....tools.config import DEFAULT_NA_SCORE

class ScoreExtractor:
    """
    Extrae scores de resultados de análisis.
    
    Responsabilidad: Proporcionar acceso seguro a scores con valores por defecto.
    """
    
    def __init__(self, default_score: float = None):
        """
        Args:
            default_score: Score por defecto cuando no hay datos (usa config si None)
        """
        self.default_score = default_score if default_score is not None else DEFAULT_NA_SCORE
    
    def extract_valuation(self, analysis: Dict) -> float:
        """Extrae score de valoración."""
        return analysis.get('valuation', {}).get('score', self.default_score)
    
    def extract_profitability(self, analysis: Dict) -> float:
        """Extrae score de rentabilidad."""
        return analysis.get('profitability', {}).get('score', self.default_score)
    
    def extract_health(self, analysis: Dict) -> float:
        """Extrae score de salud financiera."""
        return analysis.get('financial_health', {}).get('score', self.default_score)
    
    def extract_growth(self, analysis: Dict) -> float:
        """Extrae score de crecimiento."""
        return analysis.get('growth', {}).get('score', self.default_score)
    
    def extract_efficiency(self, analysis: Dict) -> float:
        """Extrae score de eficiencia."""
        return analysis.get('efficiency', {}).get('score', self.default_score)
    
    def extract_all(self, analysis: Dict) -> Dict[str, float]:
        """Extrae todos los scores disponibles."""
        return {
            'valuation': self.extract_valuation(analysis),
            'profitability': self.extract_profitability(analysis),
            'financial_health': self.extract_health(analysis),
            'growth': self.extract_growth(analysis),
            'efficiency': self.extract_efficiency(analysis)
        }