import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore', category=pd.errors.Pandas4Warning, module='yfinance')

from typing import Dict, List, Callable
from dataclasses import dataclass

from .company_analyzer import CompanyAnalyzer
from ....tools.config import SECTOR_ANALYSIS_CONFIG

@dataclass
class PercentileInterpretation:
    top_performer: float = None
    above_average: float = None
    average: float = None
    below_average: float = None
    labels: Dict[str, str] = None
    
    def __post_init__(self):
        cfg = SECTOR_ANALYSIS_CONFIG
        thresholds = cfg['percentile_thresholds']
        for field in ('top_performer', 'above_average', 'average', 'below_average'):
            if getattr(self, field) is None:
                setattr(self, field, thresholds[field])
        self.labels = self.labels or cfg['percentile_labels'].copy()

class SectorAnalyzer:
    def __init__(
        self,
        company_analyzer: CompanyAnalyzer = None,
        percentile_config: PercentileInterpretation = None,
        peer_fetcher: Callable[[str, str, str], List[str]] = None,
        max_peers: int = None
    ):
        self.company_analyzer = company_analyzer or CompanyAnalyzer()
        self.percentile_config = percentile_config or PercentileInterpretation()
        self._peer_fetcher = peer_fetcher or self._default_peer_fetcher
        self.max_peers = max_peers or SECTOR_ANALYSIS_CONFIG['max_peers']
    
    def _default_peer_fetcher(
        self, 
        ticker: str, 
        industry: str, 
        sector: str
    ) -> List[str]:

        peers = []
        try:
            from ....tools.config import SECTOR_PEERS
            sector_key = (sector or '').strip()

            if sector_key.endswith(' Sector'):
                sector_key = sector_key.replace(' Sector', '').strip()

            if sector_key and sector_key in SECTOR_PEERS:
                candidates = [p for p in SECTOR_PEERS[sector_key] if p.upper() != ticker.upper()]
                peers = candidates[: self.max_peers]
        except Exception:
            pass
        return peers
    
    def analyze_vs_peers(
        self, 
        ticker: str,
        peers: List[str] = None,
        fetch_peers: bool = True
    ) -> Dict:

        company_result = self.company_analyzer.analyze(ticker)
        
        if not company_result.get('success'):
            return company_result
        
        sector = company_result.get('sector')
        industry = company_result.get('industry')
   
        if peers is None and fetch_peers:
            peers = self._peer_fetcher(ticker, industry, sector)
        
        peers = peers or []
        peers = [p for p in peers if p.upper() != ticker.upper()][:self.max_peers]
        
        peer_results = {}
        if peers:
            peer_results = self.company_analyzer.analyze_multiple(peers)

        relative_position = self._calculate_relative_position(
            company_result, peer_results
        )

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
    
    @staticmethod
    def _calculate_relative_position(
        company: Dict, 
        peers: Dict[str, Dict]
    ) -> Dict:

        if not peers:
            return {'note': 'No peers available for comparison'}
        
        valid_peers = {t: r for t, r in peers.items() if r.get('success')}
        
        if not valid_peers:
            return {'note': 'No peers were analyzed successfully'}
        
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
            return {'note': 'No peers available for percentile calculation'}
        
        company_score = company.get('scores', {}).get('total')
        
        if pd.isna(company_score):
            return {'note': 'Company score not available'}
        
        all_scores = [company_score]
        all_scores.extend([
            r.get('scores', {}).get('total')
            for r in valid_peers.values()
            if pd.notna(r.get('scores', {}).get('total'))
        ])
        
        if len(all_scores) < 2:
            return {'note': 'Insufficient data for percentile'}

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
        
        levels = [
            (cfg.top_performer, 'top', 'Top performer'),
            (cfg.above_average, 'above', 'Above average'),
            (cfg.average, 'average', 'At average'),
            (cfg.below_average, 'below', 'Below average'),
        ]
        
        for threshold, key, fallback in levels:
            if percentile >= threshold:
                return labels.get(key, fallback)
        
        return labels.get('bottom', 'Sector laggard')
    
    def _create_comparison_df(
        self, 
        company: Dict, 
        peers: Dict[str, Dict]
    ) -> pd.DataFrame:
    
        all_results = {company['ticker']: company}
        all_results.update(peers)
        return self.company_analyzer.get_summary_df(all_results)
