from typing import Dict

class ScoreExtractor:
    
    def extract_valuation(self, analysis: Dict) -> float:
        """Extrae score de valoración"""
        return analysis.get('valuation', {}).get('score', 50)
    
    def extract_profitability(self, analysis: Dict) -> float:
        """Extrae score de rentabilidad"""
        return analysis.get('profitability', {}).get('score', 50)
    
    def extract_health(self, analysis: Dict) -> float:
        """Extrae score de salud financiera"""
        return analysis.get('financial_health', {}).get('score', 50)
    
    def extract_growth(self, analysis: Dict) -> float:
        """Extrae score de crecimiento"""
        return analysis.get('growth', {}).get('score', 50)