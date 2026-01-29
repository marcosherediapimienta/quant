import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass
from .helpers import nan_if_missing, safe_div, score_metric
from ....tools.config import (
    VALUATION_THRESHOLDS, 
    VALUATION_SCORING, 
    OVERALL_VALUATION_LOGIC,
    ALERT_THRESHOLDS,
    VALUATION_MULTIPLES_FALLBACKS
)

@dataclass
class ValuationThresholds:
    """Umbrales para clasificación de múltiplos de valoración."""
    pe_ratio: Dict[str, float] = None
    ev_ebitda: Dict[str, float] = None
    pb_ratio: Dict[str, float] = None
    fcf_yield: Dict[str, float] = None
    
    def __post_init__(self):
        logic = OVERALL_VALUATION_LOGIC['thresholds']
        fallbacks = VALUATION_MULTIPLES_FALLBACKS
        
        self.pe_ratio = self.pe_ratio or {
            'cheap': logic['pe']['cheap'],
            'fair': fallbacks['pe_ratio']['fair'],
            'expensive': logic['pe']['expensive'],
            'very_expensive': fallbacks['pe_ratio']['very_expensive']
        }
        self.ev_ebitda = self.ev_ebitda or {
            'cheap': logic['ev_ebitda']['cheap'],
            'fair': fallbacks['ev_ebitda']['fair'],
            'expensive': logic['ev_ebitda']['expensive'],
            'very_expensive': fallbacks['ev_ebitda']['very_expensive']
        }
        self.pb_ratio = self.pb_ratio or VALUATION_THRESHOLDS['valuation_multiples']['pb_ratio']
        self.fcf_yield = self.fcf_yield or {
            'excellent': logic['fcf_yield']['cheap'],
            'good': fallbacks['fcf_yield']['good'],
            'fair': fallbacks['fcf_yield']['fair'],
            'poor': logic['fcf_yield']['expensive']
        }

class ValuationMultiples:
    """
    Calcula múltiplos de valoración.
    
    Responsabilidad: Evaluar si el precio actual es atractivo vs fundamentales.
    
    Múltiplos clave:
    - P/E (Price to Earnings): Cuánto pagas por cada $ de beneficio
    - EV/EBITDA: Valoración enterprise vs EBITDA
    - P/B (Price to Book): Precio vs valor en libros
    - FCF Yield: Rendimiento de caja libre (inverso de P/FCF)
    - PEG: P/E ajustado por crecimiento
    
    Benchmarks históricos (S&P 500):
    - P/E medio: ~16x
    - EV/EBITDA medio: ~12x
    - P/B medio: ~3x
    """
    
    def __init__(self, thresholds: ValuationThresholds = None):
        self.thresholds = thresholds or ValuationThresholds()
        self.config = VALUATION_SCORING
        self.overall_logic = OVERALL_VALUATION_LOGIC
    
    def calculate(self, data: Dict) -> Dict:
        """Calcula múltiplos, clasificaciones y score de valoración."""
        pe_ttm = nan_if_missing(data.get('trailingPE'))
        pe_fwd = nan_if_missing(data.get('forwardPE'))
        pb = nan_if_missing(data.get('priceToBook'))
        ps = nan_if_missing(data.get('priceToSalesTrailing12Months'))
        ev_ebitda = nan_if_missing(data.get('enterpriseToEbitda'))
        ev_revenue = nan_if_missing(data.get('enterpriseToRevenue'))
        peg = nan_if_missing(data.get('pegRatio'))
        market_cap = nan_if_missing(data.get('marketCap'))
        fcf = nan_if_missing(data.get('freeCashflow'))
        
        fcf_yield = safe_div(fcf, market_cap)
        earnings_yield = safe_div(1, pe_ttm) if pd.notna(pe_ttm) and pe_ttm > 0 else np.nan
        
        metrics = {
            'pe_ttm': pe_ttm,
            'pe_forward': pe_fwd,
            'pb_ratio': pb,
            'ps_ratio': ps,
            'ev_ebitda': ev_ebitda,
            'ev_revenue': ev_revenue,
            'peg_ratio': peg,
            'fcf_yield': fcf_yield,
            'earnings_yield': earnings_yield,
            'market_cap': market_cap
        }
        
        classifications = {
            'pe_class': self._classify_pe(pe_ttm),
            'ev_ebitda_class': self._classify_ev_ebitda(ev_ebitda),
            'pb_class': self._classify_pb(pb),
            'fcf_yield_class': self._classify_fcf_yield(fcf_yield),
            'overall': self._overall_valuation(metrics)
        }
        
        scores = []
        weights_list = []
        cfg_weights = self.config['weights']
        cfg_ranges = self.config['ranges']
        
        if pd.notna(pe_ttm) and pe_ttm > 0:
            scores.append(score_metric(
                pe_ttm, 
                cfg_ranges['pe_ttm']['min'],
                cfg_ranges['pe_ttm']['max'],
                higher_is_better=False
            ))
            weights_list.append(cfg_weights['pe_ttm'])
            
        if pd.notna(ev_ebitda) and ev_ebitda > 0:
            scores.append(score_metric(
                ev_ebitda, 
                cfg_ranges['ev_ebitda']['min'],
                cfg_ranges['ev_ebitda']['max'],
                higher_is_better=False
            ))
            weights_list.append(cfg_weights['ev_ebitda'])
            
        if pd.notna(pb) and pb > 0:
            scores.append(score_metric(
                pb, 
                cfg_ranges['pb_ratio']['min'],
                cfg_ranges['pb_ratio']['max'],
                higher_is_better=False
            ))
            weights_list.append(cfg_weights['pb_ratio'])
            
        if pd.notna(fcf_yield):
            scores.append(score_metric(
                fcf_yield, 
                cfg_ranges['fcf_yield']['min'],
                cfg_ranges['fcf_yield']['max']
            ))
            weights_list.append(cfg_weights['fcf_yield'])

        if pd.notna(peg) and peg > 0 and 'peg_ratio' in cfg_ranges and 'peg_ratio' in cfg_weights:
            scores.append(score_metric(
                peg,
                cfg_ranges['peg_ratio']['min'],
                cfg_ranges['peg_ratio']['max'],
                higher_is_better=False
            ))
            weights_list.append(cfg_weights['peg_ratio'])
        
        total_weight = sum(weights_list)
        valuation_score = sum(s * w for s, w in zip(scores, weights_list)) / total_weight if total_weight > 0 else np.nan
        
        return {
            'metrics': metrics,
            'classifications': classifications,
            'score': valuation_score,
            'alerts': self._generate_alerts(metrics)
        }

    def _classify_pe(self, value: float) -> str:
        """Clasifica P/E según umbrales."""
        if pd.isna(value) or value <= 0:
            return 'N/A'
        thresholds = self.thresholds.pe_ratio
        if value < thresholds['cheap']:
            return 'cheap'
        elif value < thresholds['fair']:
            return 'fair'
        elif value < thresholds['expensive']:
            return 'expensive'
        return 'very_expensive'
    
    def _classify_ev_ebitda(self, value: float) -> str:
        """Clasifica EV/EBITDA según umbrales."""
        if pd.isna(value) or value <= 0:
            return 'N/A'
        thresholds = self.thresholds.ev_ebitda
        if value < thresholds['cheap']:
            return 'cheap'
        elif value < thresholds['fair']:
            return 'fair'
        elif value < thresholds['expensive']:
            return 'expensive'
        return 'very_expensive'
    
    def _classify_pb(self, value: float) -> str:
        """Clasifica P/B según umbrales."""
        if pd.isna(value) or value <= 0:
            return 'N/A'
        thresholds = self.thresholds.pb_ratio
        if value < thresholds['cheap']:
            return 'cheap'
        elif value < thresholds['fair']:
            return 'fair'
        elif value < thresholds['expensive']:
            return 'expensive'
        return 'very_expensive'
    
    def _classify_fcf_yield(self, value: float) -> str:
        """Clasifica FCF Yield según umbrales."""
        if pd.isna(value):
            return 'N/A'
        thresholds = self.thresholds.fcf_yield
        if value >= thresholds['excellent']:
            return 'excellent'
        elif value >= thresholds['good']:
            return 'good'
        elif value >= thresholds['fair']:
            return 'fair'
        return 'poor'
    
    def _overall_valuation(self, metrics: Dict) -> str:
        """
        Determina valoración general por consenso de métricas.
        
        Lógica de votación:
        - Cada métrica vota: cheap, neutral, expensive
        - Se necesita mínimo 2 votos para decidir UNDERVALUED/OVERVALUED
        - Si señales mixtas → FAIR_VALUE
        """
        cheap_count = 0
        expensive_count = 0
        valid_metrics = 0
        
        thresholds = self.overall_logic['thresholds']
        voting = self.overall_logic['voting']

        # P/E
        pe = metrics['pe_ttm']
        if pd.notna(pe) and thresholds['pe']['min_valid'] < pe < thresholds['pe']['max_valid']:
            valid_metrics += 1
            if pe < thresholds['pe']['cheap']:
                cheap_count += 1
            elif pe > thresholds['pe']['expensive']:
                expensive_count += 1

        # EV/EBITDA
        ev = metrics['ev_ebitda']
        if pd.notna(ev) and thresholds['ev_ebitda']['min_valid'] < ev < thresholds['ev_ebitda']['max_valid']:
            valid_metrics += 1
            if ev < thresholds['ev_ebitda']['cheap']:
                cheap_count += 1
            elif ev > thresholds['ev_ebitda']['expensive']:
                expensive_count += 1

        # FCF Yield
        fcf_y = metrics['fcf_yield']
        if pd.notna(fcf_y) and thresholds['fcf_yield']['min_valid'] < fcf_y < thresholds['fcf_yield']['max_valid']:
            valid_metrics += 1
            if fcf_y > thresholds['fcf_yield']['cheap']:
                cheap_count += 1
            elif fcf_y < thresholds['fcf_yield']['expensive']:
                expensive_count += 1

        # P/B
        pb = metrics.get('pb_ratio')
        if pd.notna(pb) and thresholds['pb']['min_valid'] < pb < thresholds['pb']['max_valid']:
            valid_metrics += 1
            if pb < thresholds['pb']['cheap']:
                cheap_count += 1
            elif pb > thresholds['pb']['expensive']:
                expensive_count += 1

        # Decisión
        if valid_metrics < voting['min_valid_metrics']:
            return 'FAIR_VALUE'

        # Señales mixtas
        if cheap_count > 0 and expensive_count > 0:
            return 'FAIR_VALUE'

        # Decisión por mayoría
        if cheap_count >= voting['min_votes_for_decision']:
            return 'UNDERVALUED'
        elif expensive_count >= voting['min_votes_for_decision']:
            return 'OVERVALUED'
        
        return 'FAIR_VALUE'
    
    def _generate_alerts(self, metrics: Dict) -> List[str]:
        """Genera alertas usando umbrales de config."""
        alerts = []
        alert_cfg = ALERT_THRESHOLDS['valuation']
        
        pe = metrics['pe_ttm']
        if pd.notna(pe):
            if pe > alert_cfg['pe_very_high']:
                alerts.append(f"P/E muy alto (>{alert_cfg['pe_very_high']}): posiblemente sobrevalorado")
            elif pe < alert_cfg['pe_negative']:
                alerts.append("P/E negativo: la empresa tiene pérdidas")
        
        ev_ebitda = metrics['ev_ebitda']
        if pd.notna(ev_ebitda) and ev_ebitda > alert_cfg['ev_ebitda_high']:
            alerts.append(f"EV/EBITDA muy alto (>{alert_cfg['ev_ebitda_high']})")
        
        fcf_yield = metrics['fcf_yield']
        if pd.notna(fcf_yield) and fcf_yield < alert_cfg['fcf_yield_negative']:
            alerts.append("FCF Yield negativo: no genera caja libre")
        
        peg = metrics['peg_ratio']
        if pd.notna(peg) and peg > alert_cfg['peg_high']:
            alerts.append(f"PEG > {alert_cfg['peg_high']}: precio alto respecto al crecimiento")
        
        return alerts