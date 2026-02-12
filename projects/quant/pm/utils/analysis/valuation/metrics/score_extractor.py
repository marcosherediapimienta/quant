from typing import Dict
from ....tools.config import DEFAULT_NA_SCORE

class ScoreExtractor:
    def __init__(self, default_score: float = None):
        self.default_score = default_score if default_score is not None else DEFAULT_NA_SCORE
    
    def extract_valuation(self, analysis: Dict) -> float:
        return analysis.get('valuation', {}).get('score', self.default_score)
    
    def extract_profitability(self, analysis: Dict) -> float:
        return analysis.get('profitability', {}).get('score', self.default_score)
    
    def extract_health(self, analysis: Dict) -> float:
        return analysis.get('financial_health', {}).get('score', self.default_score)
    
    def extract_growth(self, analysis: Dict) -> float:
        return analysis.get('growth', {}).get('score', self.default_score)
    
    def extract_efficiency(self, analysis: Dict) -> float:
        return analysis.get('efficiency', {}).get('score', self.default_score)
    
    def extract_all(self, analysis: Dict) -> Dict[str, float]:
        return {
            'valuation': self.extract_valuation(analysis),
            'profitability': self.extract_profitability(analysis),
            'financial_health': self.extract_health(analysis),
            'growth': self.extract_growth(analysis),
            'efficiency': self.extract_efficiency(analysis)
        }