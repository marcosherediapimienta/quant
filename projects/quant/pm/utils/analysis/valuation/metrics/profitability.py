import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass
from .helpers import nan_if_missing, classify_metric, MetricSpec, WeightedScorer
from ....tools.config import (
    VALUATION_THRESHOLDS, SCORING_WEIGHTS, PROFITABILITY_SCORING_RANGES, ALERT_THRESHOLDS,
    SECTOR_PROFITABILITY_SCORING, PROFITABILITY_SECTOR_MAP,
)

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
        self.default_config = {
            'weights': SCORING_WEIGHTS['profitability'],
            'ranges': PROFITABILITY_SCORING_RANGES,
        }

    def _resolve_sector_config(self, sector: str) -> Dict:
        sector_key = PROFITABILITY_SECTOR_MAP.get(sector)
        if sector_key and sector_key in SECTOR_PROFITABILITY_SCORING:
            return SECTOR_PROFITABILITY_SCORING[sector_key]
        return self.default_config

    def calculate(self, data: Dict) -> Dict:
        sector = data.get('sector', '')
        config = self._resolve_sector_config(sector)
        metrics = self._extract_metrics(data)

        if sector == 'Financial Services' and metrics.get('gross_margin') == 0:
            metrics['gross_margin'] = np.nan
        
        classifications = {
            f'{name}_class': classify_metric(metrics[name], getattr(self.thresholds, name))
            for name in _DATA_KEYS
        }

        score = WeightedScorer.calculate(
            metrics, _SCORING_SPECS,
            config['weights'], config['ranges']
        )
        
        return {
            'metrics': metrics,
            'classifications': classifications,
            'score': score,
            'alerts': self._generate_alerts(metrics, config)
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

    def _generate_alerts(self, metrics: Dict, config: Dict) -> List[str]:
        cfg = config.get('alerts', ALERT_THRESHOLDS['profitability'])
        return [
            msg for key, threshold_key, msg in self._ALERT_SPECS
            if pd.notna(metrics[key]) and metrics[key] < cfg[threshold_key]
        ]
