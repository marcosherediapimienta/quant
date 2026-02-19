import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass
from .helpers import nan_if_missing, safe_div, classify_metric, MetricSpec, WeightedScorer
from ....tools.config import VALUATION_THRESHOLDS, EFFICIENCY_SCORING, ALERT_THRESHOLDS, REPORTING_CONFIG

@dataclass
class EfficiencyThresholds:
    asset_turnover: Dict[str, float] = None
    inventory_turnover: Dict[str, float] = None
    dso: Dict[str, float] = None
    dio: Dict[str, float] = None
    
    def __post_init__(self):
        efficiency = VALUATION_THRESHOLDS['efficiency']
        for field in ('asset_turnover', 'inventory_turnover', 'dso', 'dio'):
            if getattr(self, field) is None:
                setattr(self, field, efficiency[field])


_SCORING_SPECS = [
    MetricSpec(key='asset_turnover', range_key='asset_turnover', weight_key='asset_turnover', higher_is_better=True),
    MetricSpec(key='days_sales_outstanding', range_key='dso', weight_key='dso', higher_is_better=False),
    MetricSpec(key='days_inventory_outstanding', range_key='dio', weight_key='dio', higher_is_better=False),
]

_CLASSIFICATION_SPECS = [
    ('asset_turnover_class', 'asset_turnover', 'asset_turnover', True),
    ('dso_class', 'days_sales_outstanding', 'dso', False),
    ('dio_class', 'days_inventory_outstanding', 'dio', False),
    ('inventory_turnover_class', 'inventory_turnover', 'inventory_turnover', True),
]


class EfficiencyMetrics:
    def __init__(self, thresholds: EfficiencyThresholds = None):
        self.thresholds = thresholds or EfficiencyThresholds()
        self.config = EFFICIENCY_SCORING
        self.days_per_year = REPORTING_CONFIG['days_per_year']
    
    def calculate(self, data: Dict) -> Dict:
        total_revenue = nan_if_missing(data.get('totalRevenue'))
        total_assets = nan_if_missing(data.get('totalAssets'))
        inventory = nan_if_missing(data.get('inventory'))
        receivables = nan_if_missing(data.get('netReceivables'))
        cogs = nan_if_missing(data.get('costOfRevenue'))
        employees = nan_if_missing(data.get('fullTimeEmployees'))
        asset_turnover = nan_if_missing(data.get('assetTurnover'))

        if pd.isna(asset_turnover) and pd.notna(total_revenue) and pd.notna(total_assets) and total_assets != 0:
            asset_turnover = total_revenue / total_assets

        inventory_turnover = safe_div(cogs, inventory)
        receivables_turnover = safe_div(total_revenue, receivables)

        dso = safe_div(self.days_per_year, receivables_turnover)
        dio = safe_div(self.days_per_year, inventory_turnover)
        revenue_per_employee = safe_div(total_revenue, employees)
        
        metrics = {
            'asset_turnover': asset_turnover,
            'inventory_turnover': inventory_turnover,
            'receivables_turnover': receivables_turnover,
            'days_sales_outstanding': dso,
            'days_inventory_outstanding': dio,
            'revenue_per_employee': revenue_per_employee,
            'total_revenue': total_revenue,
            'total_assets': total_assets,
            'employees': employees
        }
        
        classifications = {
            cls_key: classify_metric(
                metrics[metric_key],
                getattr(self.thresholds, threshold_attr),
                higher_is_better=higher_is_better
            )
            for cls_key, metric_key, threshold_attr, higher_is_better in _CLASSIFICATION_SPECS
        }

        score = WeightedScorer.calculate(
            metrics, _SCORING_SPECS,
            self.config['weights'], self.config['ranges']
        )
        
        return {
            'metrics': metrics,
            'classifications': classifications,
            'score': score,
            'alerts': self._generate_alerts(metrics)
        }
    
    def _generate_alerts(self, metrics: Dict) -> List[str]:
        alerts = []
        alert_cfg = ALERT_THRESHOLDS['efficiency']
        
        dso = metrics['days_sales_outstanding']
        if pd.notna(dso) and dso > alert_cfg['dso_high']:
            alerts.append(f"High DSO ({dso:.0f} days): slow collection")
        
        dio = metrics['days_inventory_outstanding']
        if pd.notna(dio) and dio > alert_cfg['dio_high']:
            alerts.append(f"Inventory takes {dio:.0f} days to turn over")
        
        at = metrics['asset_turnover']
        if pd.notna(at) and at < alert_cfg['asset_turnover_low']:
            alerts.append(f"Low Asset Turnover ({at:.2f}x): inefficient asset usage")
        
        return alerts
