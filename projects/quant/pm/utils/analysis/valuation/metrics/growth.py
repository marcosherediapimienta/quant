import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Mapping
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
    MetricSpec(key='earnings_growth_yoy', range_key='earnings_yoy', weight_key='earnings_yoy'),
    MetricSpec(key='earnings_quarterly_growth', range_key='earnings_quarterly', weight_key='earnings_quarterly'),
]

logger = logging.getLogger(__name__)

class GrowthMetrics:
    def __init__(self, thresholds: GrowthThresholds = None):
        self.thresholds = thresholds or GrowthThresholds()
        self.ranges = SCORING_RANGES['growth']
        self.weights = GROWTH_SCORING_WEIGHTS
        self._config_validated = False

    def _validate_scoring_config(self) -> None:
        if self._config_validated:
            return
        missing_weights = [spec.weight_key for spec in _SCORING_SPECS if spec.weight_key not in self.weights]
        missing_ranges = [spec.range_key for spec in _SCORING_SPECS if spec.range_key not in self.ranges]
        if missing_weights or missing_ranges:
            logger.warning(
                "Growth scoring config incomplete (missing weights=%s, missing ranges=%s)",
                missing_weights,
                missing_ranges,
            )
        self._config_validated = True
    
    def calculate(self, data: Mapping[str, Any]) -> Dict[str, Any]:
        self._validate_scoring_config()
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
        growth_score = self._apply_consistency_discount(growth_score, revenue_growth, earnings_growth, earnings_quarterly_growth)
        sustainability = self._analyze_sustainability(metrics)
        
        return {
            'metrics': metrics,
            'classifications': classifications,
            'score': growth_score,
            'sustainability': sustainability,
            'alerts': self._generate_alerts(metrics)
        }
    
    @staticmethod
    def _apply_consistency_discount(
        score: float,
        revenue_growth: float,
        earnings_yoy: float,
        earnings_quarterly: float,
    ) -> float:

        if pd.isna(score):
            return score

        rev = revenue_growth if pd.notna(revenue_growth) else None
        if rev is None:
            return score

        earn_candidates = [
            e for e in (earnings_yoy, earnings_quarterly)
            if pd.notna(e)
        ]
        if not earn_candidates:
            return score

        best_earn = max(earn_candidates)

        if rev >= 0.15 or best_earn < 0.20:
            return score

        rev_floor = max(rev, 0.01)
        divergence = best_earn / rev_floor

        threshold = 3.0 if rev < 0.10 else 5.0

        if divergence <= threshold:
            return score

        discount = min((divergence - threshold) * 0.08, 0.30)
        return float(np.clip(score * (1.0 - discount), 0, 100))

    def _analyze_sustainability(self, metrics: Mapping[str, Any]) -> Dict[str, Any]:
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
        ('revenue_growth_yoy',  '<', 'revenue_decline_significant', "Significant revenue decline ({t}%)"),
        ('earnings_growth_yoy', '<', 'earnings_decline_strong',     "Strong earnings decline ({t}%)"),
        ('revenue_growth_yoy',  '>', 'growth_too_high',             "Very high growth ({t}%) - verify sustainability"),
    )

    def _generate_alerts(self, metrics: Mapping[str, Any]) -> List[str]:
        cfg = ALERT_THRESHOLDS['growth']
        alerts = []
        for key, op, threshold_key, msg in self._ALERT_SPECS:
            value = metrics[key]
            if pd.isna(value):
                continue
            t = cfg[threshold_key]
            if (value > t) if op == '>' else (value < t):
                alerts.append(msg.format(t=f"{t*100:.0f}"))
        return alerts
