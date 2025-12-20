
import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass
from .helpers import nan_if_missing, score_metric, classify_metric

@dataclass
class GrowthThresholds:
    revenue_growth: Dict[str, float] = None
    earnings_growth: Dict[str, float] = None
    
    def __post_init__(self):
        self.revenue_growth = self.revenue_growth or {'excellent': 0.25, 'good': 0.15, 'fair': 0.08, 'poor': 0.03}
        self.earnings_growth = self.earnings_growth or {'excellent': 0.30, 'good': 0.20, 'fair': 0.10, 'poor': 0.05}

class GrowthMetrics:
    
    def __init__(self, thresholds: GrowthThresholds = None):
        self.thresholds = thresholds or GrowthThresholds()
    
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

        scores = []
        if pd.notna(revenue_growth):
            scores.append(score_metric(revenue_growth, -0.20, 0.40) * 0.50)
        if pd.notna(earnings_growth):
            scores.append(score_metric(earnings_growth, -0.30, 0.50) * 0.50)
        
        total_weight = sum([0.50, 0.50][:len(scores)])
        growth_score = sum(scores) / total_weight if total_weight > 0 else np.nan
        sustainability = self._analyze_sustainability(metrics)
        
        return {
            'metrics': metrics,
            'classifications': classifications,
            'score': growth_score,
            'sustainability': sustainability,
            'alerts': self._generate_alerts(metrics)
        }
    
    def _analyze_sustainability(self, metrics: Dict) -> Dict:
        rev_g = metrics['revenue_growth_yoy']
        earn_g = metrics['earnings_growth_yoy']
        
        analysis = {
            'is_sustainable': True,
            'concerns': []
        }

        if pd.notna(rev_g) and pd.notna(earn_g):
            if earn_g > rev_g * 2 and earn_g > 0.20:
                analysis['concerns'].append("Earnings crecen más rápido que ventas - verificar si es sostenible")
                analysis['is_sustainable'] = False

        if pd.notna(rev_g) and rev_g < -0.05:
            analysis['concerns'].append("Ventas en declive")
            analysis['is_sustainable'] = False
        
        return analysis
    
    def _generate_alerts(self, metrics: Dict) -> List[str]:
        alerts = []
        
        if pd.notna(metrics['revenue_growth_yoy']) and metrics['revenue_growth_yoy'] < -0.10:
            alerts.append("Caída significativa de ingresos (>10%)")
        
        if pd.notna(metrics['earnings_growth_yoy']) and metrics['earnings_growth_yoy'] < -0.20:
            alerts.append("Caída fuerte de beneficios (>20%)")
        
        if pd.notna(metrics['revenue_growth_yoy']) and metrics['revenue_growth_yoy'] > 0.50:
            alerts.append("Crecimiento muy alto (>50%) - verificar sostenibilidad")
        
        return alerts