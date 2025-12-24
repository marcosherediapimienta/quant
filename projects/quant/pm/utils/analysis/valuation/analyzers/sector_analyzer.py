import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Callable
from dataclasses import dataclass
from .company_analyzer import CompanyAnalyzer
from ....tools.config import SECTOR_ANALYSIS_CONFIG

@dataclass
class PercentileInterpretation:
    """Configuración de interpretación de percentiles."""
    top_performer: float = None
    above_average: float = None
    average: float = None
    below_average: float = None
    labels: Dict[str, str] = None
    
    def __post_init__(self):
        # Cargar desde config si no se proporciona
        cfg = SECTOR_ANALYSIS_CONFIG
        self.top_performer = self.top_performer or cfg['percentile_thresholds']['top_performer']
        self.above_average = self.above_average or cfg['percentile_thresholds']['above_average']
        self.average = self.average or cfg['percentile_thresholds']['average']
        self.below_average = self.below_average or cfg['percentile_thresholds']['below_average']
        self.labels = self.labels or cfg['percentile_labels'].copy()

class SectorAnalyzer:
    """
    Analiza empresa vs peers del sector.
    
    Responsabilidad: Comparar métricas de empresa con competidores del mismo sector.
    
    Metodología:
    - Identifica peers (mismo sector/industria)
    - Calcula posición relativa en cada métrica
    - Determina percentil general (top 20%, promedio, etc.)
    """

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
        """
        Fetcher por defecto de peers (placeholder).
        
        En producción, esto debería conectarse a:
        - API de clasificación sectorial
        - Base de datos de empresas por sector
        - Screening de peers similares
        """
        peers = []
        
        try:
            stock = yf.Ticker(ticker)
            # yfinance no proporciona peers directamente
            # Aquí iría lógica personalizada de identificación de peers
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
        """
        Analiza empresa vs peers del sector.
        
        Args:
            ticker: Ticker de la empresa
            peers: Lista opcional de peers (si None, intenta fetch)
            fetch_peers: Si True, intenta buscar peers automáticamente
            
        Returns:
            Dict con análisis completo vs peers
        """
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
    
    def _calculate_relative_position(
        self, 
        company: Dict, 
        peers: Dict[str, Dict]
    ) -> Dict:
        """Calcula posición relativa en cada categoría de score."""
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
        """Calcula percentil de la empresa vs peers."""
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

        below_count = sum(1 for s in all_scores if s < company_score)
        percentile = (below_count / len(all_scores)) * 100
        
        return {
            'percentile': percentile,
            'sample_size': len(all_scores),
            'interpretation': self._interpret_percentile(percentile)
        }
    
    def _interpret_percentile(self, percentile: float) -> str:
        """Interpreta percentil usando config."""
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
        """Crea DataFrame comparativo."""
        all_results = {company['ticker']: company}
        all_results.update(peers)
        return self.company_analyzer.get_summary_df(all_results)