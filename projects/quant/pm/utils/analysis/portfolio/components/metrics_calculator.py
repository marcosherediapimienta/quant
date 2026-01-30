from typing import List, Dict
from ....tools.config import PORTFOLIO_CONFIG

class PortfolioMetricsCalculator:
    def __init__(self, default_sector: str = ''):
        defaults = PORTFOLIO_CONFIG['defaults']
        self.default_sector = default_sector if default_sector else defaults['sector_name']
    
    def calculate(
        self,
        tickers: List[str],
        weights: Dict[str, float],
        analysis_results: Dict
    ) -> Dict:

        return {
            'total_score': self._calculate_weighted_score(tickers, weights, analysis_results),
            'sector_allocation': self._calculate_sector_allocation(tickers, weights, analysis_results),
            'num_companies': len(tickers)
        }
    
    def _calculate_weighted_score(
        self,
        tickers: List[str],
        weights: Dict[str, float],
        analysis_results: Dict
    ) -> float:

        return sum(
            analysis_results[t].get('scores', {}).get('total', 0) * weights[t]
            for t in tickers
        )
    
    def _calculate_sector_allocation(
        self,
        tickers: List[str],
        weights: Dict[str, float],
        analysis_results: Dict
    ) -> Dict[str, float]:

        sectors = {}
        for t in tickers:
            sector = analysis_results[t].get('sector', self.default_sector)
            sectors[sector] = sectors.get(sector, 0) + weights[t]
        return sectors