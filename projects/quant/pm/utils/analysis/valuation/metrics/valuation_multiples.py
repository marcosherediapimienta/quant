import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass
from .helpers import nan_if_missing, safe_div, score_metric

@dataclass
class ValuationThresholds:

    pe_ratio: Dict[str, float] = None
    ev_ebitda: Dict[str, float] = None
    pb_ratio: Dict[str, float] = None
    fcf_yield: Dict[str, float] = None
    
    def __post_init__(self):
        self.pe_ratio = self.pe_ratio or {'cheap': 12, 'fair': 18, 'expensive': 25, 'very_expensive': 35}
        self.ev_ebitda = self.ev_ebitda or {'cheap': 8, 'fair': 12, 'expensive': 16, 'very_expensive': 20}
        self.pb_ratio = self.pb_ratio or {'cheap': 1.5, 'fair': 3.0, 'expensive': 5.0, 'very_expensive': 8.0}
        self.fcf_yield = self.fcf_yield or {'excellent': 0.08, 'good': 0.05, 'fair': 0.03, 'poor': 0.01}


class ValuationMultiples:
    
    def __init__(self, thresholds: ValuationThresholds = None):
        self.thresholds = thresholds or ValuationThresholds()
    
    def calculate(self, data: Dict) -> Dict:

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
        if pd.notna(pe_ttm) and pe_ttm > 0:
            scores.append(score_metric(pe_ttm, 5, 40, higher_is_better=False) * 0.25)
        if pd.notna(ev_ebitda) and ev_ebitda > 0:
            scores.append(score_metric(ev_ebitda, 4, 25, higher_is_better=False) * 0.25)
        if pd.notna(pb) and pb > 0:
            scores.append(score_metric(pb, 0.5, 8, higher_is_better=False) * 0.20)
        if pd.notna(fcf_yield):
            scores.append(score_metric(fcf_yield, -0.02, 0.12) * 0.30)
        
        total_weight = sum([0.25, 0.25, 0.20, 0.30][:len(scores)])
        valuation_score = sum(scores) / total_weight if total_weight > 0 else np.nan
        
        return {
            'metrics': metrics,
            'classifications': classifications,
            'score': valuation_score,
            'alerts': self._generate_alerts(metrics)
        }
    
    def _classify_pe(self, value: float) -> str:
        if pd.isna(value) or value <= 0:
            return 'N/A'
        if value < 12:
            return 'cheap'
        elif value < 18:
            return 'fair'
        elif value < 25:
            return 'expensive'
        return 'very_expensive'
    
    def _classify_ev_ebitda(self, value: float) -> str:
        if pd.isna(value) or value <= 0:
            return 'N/A'
        if value < 8:
            return 'cheap'
        elif value < 12:
            return 'fair'
        elif value < 16:
            return 'expensive'
        return 'very_expensive'
    
    def _classify_pb(self, value: float) -> str:
        if pd.isna(value) or value <= 0:
            return 'N/A'
        if value < 1.5:
            return 'cheap'
        elif value < 3:
            return 'fair'
        elif value < 5:
            return 'expensive'
        return 'very_expensive'
    
    def _classify_fcf_yield(self, value: float) -> str:
        if pd.isna(value):
            return 'N/A'
        if value >= 0.08:
            return 'excellent'
        elif value >= 0.05:
            return 'good'
        elif value >= 0.03:
            return 'fair'
        return 'poor'
    
    def _overall_valuation(self, metrics: Dict) -> str:
        cheap_count = 0
        expensive_count = 0
        
        pe = metrics['pe_ttm']
        if pd.notna(pe) and pe > 0:
            if pe < 15:
                cheap_count += 1
            elif pe > 25:
                expensive_count += 1
        
        ev = metrics['ev_ebitda']
        if pd.notna(ev) and ev > 0:
            if ev < 10:
                cheap_count += 1
            elif ev > 15:
                expensive_count += 1
        
        fcf_y = metrics['fcf_yield']
        if pd.notna(fcf_y):
            if fcf_y > 0.06:
                cheap_count += 1
            elif fcf_y < 0.02:
                expensive_count += 1
        
        if cheap_count >= 2:
            return 'UNDERVALUED'
        elif expensive_count >= 2:
            return 'OVERVALUED'
        return 'FAIR_VALUE'
    
    def _generate_alerts(self, metrics: Dict) -> List[str]:
        alerts = []
        
        if pd.notna(metrics['pe_ttm']) and metrics['pe_ttm'] > 40:
            alerts.append("P/E muy alto (>40): posiblemente sobrevalorado")
        
        if pd.notna(metrics['pe_ttm']) and metrics['pe_ttm'] < 0:
            alerts.append("P/E negativo: la empresa tiene pérdidas")
        
        if pd.notna(metrics['ev_ebitda']) and metrics['ev_ebitda'] > 20:
            alerts.append("EV/EBITDA muy alto (>20)")
        
        if pd.notna(metrics['fcf_yield']) and metrics['fcf_yield'] < 0:
            alerts.append("FCF Yield negativo: no genera caja libre")
        
        if pd.notna(metrics['peg_ratio']) and metrics['peg_ratio'] > 2:
            alerts.append("PEG > 2: precio alto respecto al crecimiento")
        
        return alerts