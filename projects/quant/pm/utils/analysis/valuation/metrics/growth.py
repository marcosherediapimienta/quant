import pandas as pd
from typing import Dict, List
from dataclasses import dataclass
from .helpers import nan_if_missing, classify_metric, MetricSpec, WeightedScorer
from ....tools.config import (
    VALUATION_THRESHOLDS, 
    SCORING_RANGES,
    GROWTH_SCORING_WEIGHTS,
    ALERT_THRESHOLDS
)

@dataclass
class GrowthThresholds:
    revenue_growth: Dict[str, float] = None
    earnings_growth: Dict[str, float] = None
    
    def __post_init__(self):
        growth = VALUATION_THRESHOLDS['growth']
        for field in ('revenue_growth', 'earnings_growth'):
            if getattr(self, field) is None:
                setattr(self, field, growth[field])

_SCORING_SPECS = [
    MetricSpec(key='revenue_growth_yoy', range_key='revenue', weight_key='revenue'),
    MetricSpec(key='earnings_growth_yoy', range_key='earnings', weight_key='earnings'),
]

class GrowthMetrics:
    def __init__(self, thresholds: GrowthThresholds = None):
        self.thresholds = thresholds or GrowthThresholds()
        self.ranges = SCORING_RANGES['growth']
        self.weights = GROWTH_SCORING_WEIGHTS
    
    def calculate(self, data: Dict) -> Dict:
        revenue_growth = nan_if_missing(data.get('revenueGrowth'))
        earnings_growth = nan_if_missing(data.get('earningsGrowth'))
        earnings_quarterly_growth = nan_if_missing(data.get('earningsQuarterlyGrowth'))
        revenue_per_share_growth = nan_if_missing(data.get('revenuePerShareGrowth'))
        
        metrics = {
            'revenue_growth_yoy': revenue_growth,
            'earnings_growth_yoy': earnings_growth,
            'earnings_quarterly_growth': earnings_quarterly_growth,
            'revenue_per_share_growth': revenue_per_share_growth
        }
        
        classifications = {
            'revenue_growth_class': classify_metric(revenue_growth, self.thresholds.revenue_growth),
            'earnings_growth_class': classify_metric(earnings_growth, self.thresholds.earnings_growth)
        }

        growth_score = WeightedScorer.calculate(metrics, _SCORING_SPECS, self.weights, self.ranges)
        sustainability = self._analyze_sustainability(metrics)
        
        return {
            'metrics': metrics,
            'classifications': classifications,
            'score': growth_score,
            'sustainability': sustainability,
            'alerts': self._generate_alerts(metrics)
        }
    
    def _analyze_sustainability(self, metrics: Dict) -> Dict:
        analysis = {
            'is_sustainable': True,
            'concerns': []
        }
        
        alert_cfg = ALERT_THRESHOLDS['growth']
        rev_g = metrics['revenue_growth_yoy']
        earn_g = metrics['earnings_growth_yoy']

        if pd.notna(rev_g) and pd.notna(earn_g):
            earnings_multiple = alert_cfg['earnings_vs_revenue_multiple']
            high_threshold = alert_cfg['high_earnings_growth_threshold']
            
            if earn_g > rev_g * earnings_multiple and earn_g > high_threshold:
                analysis['concerns'].append("Earnings growing faster than revenue - verify sustainability")
                analysis['is_sustainable'] = False

        if pd.notna(rev_g) and rev_g < alert_cfg['revenue_decline_mild']:
            analysis['concerns'].append("Revenue in decline")
            analysis['is_sustainable'] = False
        
        return analysis
    
    _ALERT_SPECS = (
        ('revenue_growth_yoy',  '<', 'revenue_decline_significant', "Significant revenue decline (>{t}%)"),
        ('earnings_growth_yoy', '<', 'earnings_decline_strong',     "Strong earnings decline (>{t}%)"),
        ('revenue_growth_yoy',  '>', 'growth_too_high',             "Very high growth (>{t}%) - verify sustainability"),
    )

    def _generate_alerts(self, metrics: Dict) -> List[str]:
        cfg = ALERT_THRESHOLDS['growth']
        alerts = []
        for key, op, threshold_key, msg in self._ALERT_SPECS:
            value = metrics[key]
            if pd.isna(value):
                continue
            t = cfg[threshold_key]
            if (value > t) if op == '>' else (value < t):
                alerts.append(msg.format(t=f"{abs(t)*100:.0f}"))
        return alerts
