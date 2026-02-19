import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass
from .helpers import nan_if_missing, safe_div, classify_metric, MetricSpec, WeightedScorer
from ....tools.config import (
    VALUATION_THRESHOLDS, 
    VALUATION_SCORING, 
    OVERALL_VALUATION_LOGIC,
    OVERALL_VALUATION_METRICS,
    ALERT_THRESHOLDS,
)

@dataclass
class ValuationThresholds:
    pe_ratio: Dict[str, float] = None
    ev_ebitda: Dict[str, float] = None
    pb_ratio: Dict[str, float] = None
    fcf_yield: Dict[str, float] = None
    peg_ratio: Dict[str, float] = None
    
    def __post_init__(self):
        multiples = VALUATION_THRESHOLDS['valuation_multiples']
        for field in ('pe_ratio', 'ev_ebitda', 'pb_ratio', 'peg_ratio', 'fcf_yield'):
            if getattr(self, field) is None:
                setattr(self, field, multiples[field])

_SCORING_SPECS = [
    MetricSpec(key='pe_ttm', range_key='pe_ttm', weight_key='pe_ttm', higher_is_better=False, require_positive=True),
    MetricSpec(key='ev_ebitda', range_key='ev_ebitda', weight_key='ev_ebitda', higher_is_better=False, require_positive=True),
    MetricSpec(key='pb_ratio', range_key='pb_ratio', weight_key='pb_ratio', higher_is_better=False, require_positive=True),
    MetricSpec(key='fcf_yield', range_key='fcf_yield', weight_key='fcf_yield', higher_is_better=True),
    MetricSpec(key='peg_ratio', range_key='peg_ratio', weight_key='peg_ratio', higher_is_better=False, require_positive=True),
]

class ValuationMultiples:
    def __init__(self, thresholds: ValuationThresholds = None):
        self.thresholds = thresholds or ValuationThresholds()
        self.config = VALUATION_SCORING
        self.overall_logic = OVERALL_VALUATION_LOGIC
    
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

        peg = self._compute_peg(peg, pe_ttm, data)
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
            'pe_class': self._classify_lower_is_better(pe_ttm, self.thresholds.pe_ratio, require_positive=True),
            'ev_ebitda_class': self._classify_lower_is_better(ev_ebitda, self.thresholds.ev_ebitda, require_positive=True),
            'pb_class': self._classify_lower_is_better(pb, self.thresholds.pb_ratio, require_positive=True),
            'fcf_yield_class': classify_metric(fcf_yield, self.thresholds.fcf_yield),
            'peg_class': self._classify_lower_is_better(peg, self.thresholds.peg_ratio, require_positive=True, default='very_expensive'),
            'overall': self._overall_valuation(metrics)
        }
        
        valuation_score = WeightedScorer.calculate(
            metrics, _SCORING_SPECS,
            self.config['weights'], self.config['ranges']
        )
        
        return {
            'metrics': metrics,
            'classifications': classifications,
            'score': valuation_score,
            'alerts': self._generate_alerts(metrics)
        }

    @staticmethod
    def _classify_lower_is_better(
        value: float, thresholds: Dict, require_positive: bool = False, default: str = 'very_expensive'
    ) -> str:
        if pd.isna(value):
            return 'N/A'
        if require_positive and value <= 0:
            return 'N/A'
        return classify_metric(value, thresholds, higher_is_better=False, strict=True, default=default)

    @staticmethod
    def _compute_peg(peg: float, pe_ttm: float, data: Dict) -> float:
        if pd.notna(peg) and peg > 0:
            return peg

        earnings_growth = nan_if_missing(data.get('earningsGrowth'))
        if pd.isna(earnings_growth) or earnings_growth <= 0:
            earnings_growth = nan_if_missing(data.get('earningsQuarterlyGrowth'))

        if pd.isna(pe_ttm) or pe_ttm <= 0 or pd.isna(earnings_growth) or earnings_growth <= 0:
            return np.nan

        growth_pct = earnings_growth * 100 if earnings_growth <= 1 else earnings_growth
        peg = pe_ttm / growth_pct
        return peg if 0.1 <= peg <= 50 else np.nan
    
    def _overall_valuation(self, metrics: Dict) -> str:
        cheap_count = 0
        expensive_count = 0
        valid_metrics = 0
        thresholds = self.overall_logic['thresholds']
        voting = self.overall_logic['voting']

        for spec in OVERALL_VALUATION_METRICS:
            value = metrics.get(spec['key'])
            cfg = thresholds[spec['config_key']]

            if pd.isna(value) or not (cfg['min_valid'] < value < cfg['max_valid']):
                continue

            valid_metrics += 1
            hmc = spec['higher_means_cheaper']
            cheap_count += (value > cfg['cheap']) if hmc else (value < cfg['cheap'])
            expensive_count += (value < cfg['expensive']) if hmc else (value > cfg['expensive'])

        if valid_metrics < voting['min_valid_metrics'] or (cheap_count > 0 and expensive_count > 0):
            return 'FAIR_VALUE'
        if cheap_count >= voting['min_votes_for_decision']:
            return 'UNDERVALUED'
        if expensive_count >= voting['min_votes_for_decision']:
            return 'OVERVALUED'
        return 'FAIR_VALUE'
    
    _ALERT_SPECS = (
        ('pe_ttm',    '>', 'pe_very_high',    "P/E very high (>{t}): possibly overvalued"),
        ('pe_ttm',    '<', 'pe_negative',     "Negative P/E: company has losses"),
        ('ev_ebitda', '>', 'ev_ebitda_high',  "EV/EBITDA very high (>{t})"),
        ('fcf_yield', '<', 'fcf_yield_negative', "Negative FCF Yield: not generating free cash flow"),
        ('peg_ratio', '>', 'peg_high',        "PEG > {t}: high price relative to growth"),
    )

    def _generate_alerts(self, metrics: Dict) -> List[str]:
        alerts = []
        cfg = ALERT_THRESHOLDS['valuation']
        for key, op, threshold_key, msg in self._ALERT_SPECS:
            value = metrics[key]
            if pd.isna(value):
                continue
            t = cfg[threshold_key]
            if (value > t) if op == '>' else (value < t):
                alerts.append(msg.format(t=t))
        return alerts
