import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass
from .helpers import nan_if_missing, safe_div, score_metric
from ....tools.config import VALUATION_THRESHOLDS, EFFICIENCY_SCORING, ALERT_THRESHOLDS, REPORTING_CONFIG

@dataclass
class EfficiencyThresholds:
    asset_turnover: Dict[str, float] = None
    inventory_turnover: Dict[str, float] = None
    dso: Dict[str, float] = None
    dio: Dict[str, float] = None
    
    def __post_init__(self):
        efficiency_thresholds = VALUATION_THRESHOLDS['efficiency']
        self.asset_turnover = self.asset_turnover or efficiency_thresholds['asset_turnover']
        self.inventory_turnover = self.inventory_turnover or efficiency_thresholds['inventory_turnover']
        self.dso = self.dso or efficiency_thresholds['dso']
        self.dio = self.dio or efficiency_thresholds['dio']


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
            'asset_turnover_class': self._classify_value(asset_turnover, self.thresholds.asset_turnover, higher_is_better=True),
            'dso_class': self._classify_value(dso, self.thresholds.dso, higher_is_better=False),
            'dio_class': self._classify_value(dio, self.thresholds.dio, higher_is_better=False),
            'inventory_turnover_class': self._classify_value(inventory_turnover, self.thresholds.inventory_turnover, higher_is_better=True)
        }

        score = self._calculate_score(metrics)
        
        return {
            'metrics': metrics,
            'classifications': classifications,
            'score': score,
            'alerts': self._generate_alerts(metrics)
        }
    
    def _classify_value(self, value: float, thresholds: Dict, higher_is_better: bool = True) -> str:

        if pd.isna(value):
            return 'N/A'
        
        levels = ['excellent', 'good', 'fair', 'poor']
        
        if higher_is_better:
            for level in levels:
                if value >= thresholds.get(level, 0):
                    return level
            return 'poor'
        else:
            for level in levels:
                if value <= thresholds.get(level, float('inf')):
                    return level
            return 'poor'
    
    def _calculate_score(self, metrics: Dict) -> float:
        scores = []
        weights = []
        cfg_weights = self.config['weights']
        cfg_ranges = self.config['ranges']
        
        if pd.notna(metrics['asset_turnover']):
            at_score = score_metric(
                metrics['asset_turnover'], 
                cfg_ranges['asset_turnover']['min'],
                cfg_ranges['asset_turnover']['max'],
                higher_is_better=True
            )
            scores.append(at_score)
            weights.append(cfg_weights['asset_turnover'])

        if pd.notna(metrics['days_sales_outstanding']):
            dso_score = score_metric(
                metrics['days_sales_outstanding'], 
                cfg_ranges['dso']['min'],
                cfg_ranges['dso']['max'],
                higher_is_better=False
            )
            scores.append(dso_score)
            weights.append(cfg_weights['dso'])

        if pd.notna(metrics['days_inventory_outstanding']):
            dio_score = score_metric(
                metrics['days_inventory_outstanding'], 
                cfg_ranges['dio']['min'],
                cfg_ranges['dio']['max'],
                higher_is_better=False
            )
            scores.append(dio_score)
            weights.append(cfg_weights['dio'])
        
        if not scores:
            return np.nan
        
        total_weight = sum(weights)
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        
        return weighted_sum / total_weight if total_weight > 0 else np.nan
    
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