import logging
from typing import Any, Dict, Mapping
from ....tools.config import DEFAULT_NA_SCORE

_CATEGORIES = ('valuation', 'profitability', 'financial_health', 'growth', 'efficiency')
logger = logging.getLogger(__name__)

class ScoreExtractor:
    def __init__(self, default_score: float = None):
        self.default_score = float(DEFAULT_NA_SCORE if default_score is None else default_score)
    
    def extract(self, analysis: Mapping[str, Any], category: str) -> float:
        if category not in _CATEGORIES:
            logger.warning("Unknown score category '%s'. Falling back to default score.", category)
            return self.default_score

        section = analysis.get(category) or {}
        if not isinstance(section, dict):
            return self.default_score
        return section.get('score', self.default_score)
    
    def extract_all(self, analysis: Mapping[str, Any]) -> Dict[str, float]:
        return {cat: self.extract(analysis, cat) for cat in _CATEGORIES}

    def extract_valuation(self, analysis: Mapping[str, Any]) -> float:
        return self.extract(analysis, 'valuation')
    
    def extract_profitability(self, analysis: Mapping[str, Any]) -> float:
        return self.extract(analysis, 'profitability')
    
    def extract_health(self, analysis: Mapping[str, Any]) -> float:
        return self.extract(analysis, 'financial_health')
    
    def extract_growth(self, analysis: Mapping[str, Any]) -> float:
        return self.extract(analysis, 'growth')
    
    def extract_efficiency(self, analysis: Mapping[str, Any]) -> float:
        return self.extract(analysis, 'efficiency')
