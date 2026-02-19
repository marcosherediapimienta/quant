from typing import Dict
from ....tools.config import DEFAULT_NA_SCORE

_CATEGORIES = ('valuation', 'profitability', 'financial_health', 'growth', 'efficiency')

class ScoreExtractor:
    def __init__(self, default_score: float = None):
        self.default_score = default_score if default_score is not None else DEFAULT_NA_SCORE
    
    def extract(self, analysis: Dict, category: str) -> float:
        return analysis.get(category, {}).get('score', self.default_score)
    
    def extract_all(self, analysis: Dict) -> Dict[str, float]:
        return {cat: self.extract(analysis, cat) for cat in _CATEGORIES}

    def extract_valuation(self, analysis: Dict) -> float:
        return self.extract(analysis, 'valuation')
    
    def extract_profitability(self, analysis: Dict) -> float:
        return self.extract(analysis, 'profitability')
    
    def extract_health(self, analysis: Dict) -> float:
        return self.extract(analysis, 'financial_health')
    
    def extract_growth(self, analysis: Dict) -> float:
        return self.extract(analysis, 'growth')
    
    def extract_efficiency(self, analysis: Dict) -> float:
        return self.extract(analysis, 'efficiency')
