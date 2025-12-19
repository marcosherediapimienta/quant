import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Callable
from dataclasses import dataclass
from .company_analyzer import CompanyAnalyzer


@dataclass
class PercentileInterpretation:

    top_performer: float = 80
    above_average: float = 60
    average: float = 40
    below_average: float = 20
    
    labels: Dict[str, str] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {
                'top': 'Top performer del sector',
                'above': 'Por encima del promedio',
                'average': 'En el promedio del sector',
                'below': 'Por debajo del promedio',
                'bottom': 'Rezagado del sector'
            }


class SectorAnalyzer:

    def __init__(
        self,
        company_analyzer: CompanyAnalyzer = None,
        percentile_config: PercentileInterpretation = None,
        peer_fetcher: Callable[[str, str, str], List[str]] = None,
        max_peers: int = 5
    ):
 
        self.company_analyzer = company_analyzer or CompanyAnalyzer()
        self.percentile_config = percentile_config or PercentileInterpretation()
        self._peer_fetcher = peer_fetcher or self._default_peer_fetcher
        self.max_peers = max_peers
    
    def _default_peer_fetcher(
        self, 
        ticker: str, 
        industry: str, 
        sector: str
    ) -> List[str]:
     
        peers = []
        
        try:
            stock = yf.Ticker(ticker)
    
            if hasattr(stock, 'recommendations') and stock.recommendations is not None:
                pass
            
        except Exception:
            pass
        
        return peers
    
    def analyze_vs_peers(
        self, 
        ticker: str,
        peers: List[str] = None,
        fetch_peers: bool = True
    ) -> Dict:

        # Analizar empresa objetivo
        company_result = self.company_analyzer.analyze(ticker)
        
        if not company_result.get('success'):
            return company_result
        
        sector = company_result.get('sector')
        industry = company_result.get('industry')
        
        # Obtener peers si no se proveen
        if peers is None and fetch_peers:
            peers = self._peer_fetcher(ticker, industry, sector)
        
        peers = peers or []
        
        # Filtrar el ticker objetivo de peers
        peers = [p for p in peers if p.upper() != ticker.upper()][:self.max_peers]
        
        # Analizar peers
        peer_results = {}
        if peers:
            peer_results = self.company_analyzer.analyze_multiple(peers)
        
        # Calcular posición relativa
        relative_position = self._calculate_relative_position(
            company_result, peer_results
        )
        
        # Percentiles
        percentiles = self._calculate_percentiles(company_result, peer_results)
        
        return {
            'company': company_result,
            'sector': sector,
            'industry': industry,
            'peers_analyzed': peers,
            'peer_count': len([p for p in peer_results.values() if p.get('success')]),
            'peer_results': peer_results,
            'relative_position': relative_position,
            'percentiles': percentiles,
            'comparison_df': self._create_comparison_df(company_result, peer_results),
            'success': True
        }
    
    def _calculate_relative_position(
        self, 
        company: Dict, 
        peers: Dict[str, Dict]
    ) -> Dict:

        if not peers:
            return {'note': 'Sin peers para comparar'}
        
        valid_peers = {t: r for t, r in peers.items() if r.get('success')}
        
        if not valid_peers:
            return {'note': 'Ningún peer analizado exitosamente'}
        
        categories = ['profitability', 'financial_health', 'growth', 'efficiency', 'valuation', 'total']
        position = {}
        
        for cat in categories:
            company_score = company.get('scores', {}).get(cat)
            
            peer_scores = [
                r.get('scores', {}).get(cat)
                for r in valid_peers.values()
                if pd.notna(r.get('scores', {}).get(cat))
            ]
            
            if pd.notna(company_score) and peer_scores:
                better_than = sum(1 for p in peer_scores if company_score > p)
                position[cat] = {
                    'company_score': company_score,
                    'peer_avg': float(np.mean(peer_scores)),
                    'peer_median': float(np.median(peer_scores)),
                    'rank': better_than + 1,
                    'total_compared': len(peer_scores) + 1,
                    'vs_avg': company_score - float(np.mean(peer_scores)),
                    'vs_median': company_score - float(np.median(peer_scores))
                }
        
        return position
    
    def _calculate_percentiles(
        self, 
        company: Dict, 
        peers: Dict[str, Dict]
    ) -> Dict:

        valid_peers = {t: r for t, r in peers.items() if r.get('success')}
        
        if not valid_peers:
            return {'note': 'Sin peers para calcular percentil'}
        
        company_score = company.get('scores', {}).get('total')
        
        if pd.isna(company_score):
            return {'note': 'Score de empresa no disponible'}
        
        all_scores = [company_score]
        all_scores.extend([
            r.get('scores', {}).get('total')
            for r in valid_peers.values()
            if pd.notna(r.get('scores', {}).get('total'))
        ])
        
        if len(all_scores) < 2:
            return {'note': 'Datos insuficientes para percentil'}
        
        # Calcular percentil
        below_count = sum(1 for s in all_scores if s < company_score)
        percentile = (below_count / len(all_scores)) * 100
        
        return {
            'percentile': percentile,
            'sample_size': len(all_scores),
            'interpretation': self._interpret_percentile(percentile)
        }
    
    def _interpret_percentile(self, percentile: float) -> str:
        cfg = self.percentile_config
        labels = cfg.labels
        
        if percentile >= cfg.top_performer:
            return labels.get('top', 'Top performer')
        elif percentile >= cfg.above_average:
            return labels.get('above', 'Por encima del promedio')
        elif percentile >= cfg.average:
            return labels.get('average', 'En el promedio')
        elif percentile >= cfg.below_average:
            return labels.get('below', 'Por debajo del promedio')
        
        return labels.get('bottom', 'Rezagado')
    
    def _create_comparison_df(
        self, 
        company: Dict, 
        peers: Dict[str, Dict]
    ) -> pd.DataFrame:

        all_results = {company['ticker']: company}
        all_results.update(peers)
        return self.company_analyzer.get_summary_df(all_results)