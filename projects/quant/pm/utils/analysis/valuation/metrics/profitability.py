import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass
from .helpers import nan_if_missing, classify_metric, MetricSpec, WeightedScorer
from ....tools.config import VALUATION_THRESHOLDS, SCORING_WEIGHTS, PROFITABILITY_SCORING_RANGES, ALERT_THRESHOLDS

@dataclass
class ProfitabilityThresholds:
    roic: Dict[str, float] = None
    roe: Dict[str, float] = None
    roa: Dict[str, float] = None
    gross_margin: Dict[str, float] = None
    operating_margin: Dict[str, float] = None
    net_margin: Dict[str, float] = None
    
    def __post_init__(self):
        profitability = VALUATION_THRESHOLDS['profitability']
        for field in ('roic', 'roe', 'roa', 'gross_margin', 'operating_margin', 'net_margin'):
            if getattr(self, field) is None:
                setattr(self, field, profitability[field])

_DATA_KEYS = {
    'roic': ('returnOnCapital', 'roic'),
    'roe': ('returnOnEquity',),
    'roa': ('returnOnAssets',),
    'gross_margin': ('grossMargins',),
    'operating_margin': ('operatingMargins',),
    'net_margin': ('profitMargins',),
}

_SCORING_SPECS = [
    MetricSpec(key='roic', range_key='roic', weight_key='roic'),
    MetricSpec(key='roe', range_key='roe', weight_key='roe'),
    MetricSpec(key='operating_margin', range_key='operating_margin', weight_key='operating_margin'),
    MetricSpec(key='net_margin', range_key='net_margin', weight_key='net_margin'),
]

class ProfitabilityMetrics:
    def __init__(self, thresholds: ProfitabilityThresholds = None):
        self.thresholds = thresholds or ProfitabilityThresholds()
        self.weights = SCORING_WEIGHTS['profitability']
        self.ranges = PROFITABILITY_SCORING_RANGES
    
    def calculate(self, data: Dict) -> Dict:
        metrics = self._extract_metrics(data)
        
        classifications = {
            f'{name}_class': classify_metric(metrics[name], getattr(self.thresholds, name))
            for name in _DATA_KEYS
        }

        score = WeightedScorer.calculate(metrics, _SCORING_SPECS, self.weights, self.ranges)
        
        return {
            'metrics': metrics,
            'classifications': classifications,
            'score': score,
            'alerts': self._generate_alerts(metrics)
        }

    @staticmethod
    def _extract_metrics(data: Dict) -> Dict:
        metrics = {}
        for metric_name, source_keys in _DATA_KEYS.items():
            value = np.nan
            for key in source_keys:
                value = nan_if_missing(data.get(key))
                if pd.notna(value):
                    break
            metrics[metric_name] = value
        return metrics
    
    _ALERT_SPECS = (
        ('roic', 'roic_low', "Low ROIC: company does not generate sufficient returns on invested capital"),
        ('roe', 'roe_negative', "Negative ROE: company is losing money"),
        ('operating_margin', 'operating_margin_low', "Very low operating margin: operational efficiency issues"),
        ('net_margin', 'net_margin_negative', "Negative net margin: company is not profitable"),
    )

    def _generate_alerts(self, metrics: Dict) -> List[str]:
        cfg = ALERT_THRESHOLDS['profitability']
        return [
            msg for key, threshold_key, msg in self._ALERT_SPECS
            if pd.notna(metrics[key]) and metrics[key] < cfg[threshold_key]
        ]
