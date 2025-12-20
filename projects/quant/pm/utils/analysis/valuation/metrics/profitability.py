import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass
from .helpers import nan_if_missing, score_metric, classify_metric

@dataclass
class ProfitabilityThresholds:
    roic: Dict[str, float] = None
    roe: Dict[str, float] = None
    roa: Dict[str, float] = None
    gross_margin: Dict[str, float] = None
    operating_margin: Dict[str, float] = None
    net_margin: Dict[str, float] = None
    
    def __post_init__(self):
        self.roic = self.roic or {'excellent': 0.20, 'good': 0.15, 'fair': 0.10, 'poor': 0.05}
        self.roe = self.roe or {'excellent': 0.25, 'good': 0.15, 'fair': 0.10, 'poor': 0.05}
        self.roa = self.roa or {'excellent': 0.15, 'good': 0.10, 'fair': 0.05, 'poor': 0.02}
        self.gross_margin = self.gross_margin or {'excellent': 0.50, 'good': 0.35, 'fair': 0.20, 'poor': 0.10}
        self.operating_margin = self.operating_margin or {'excellent': 0.25, 'good': 0.15, 'fair': 0.10, 'poor': 0.05}
        self.net_margin = self.net_margin or {'excellent': 0.20, 'good': 0.10, 'fair': 0.05, 'poor': 0.02}


class ProfitabilityMetrics:
    
    def __init__(self, thresholds: ProfitabilityThresholds = None):
        self.thresholds = thresholds or ProfitabilityThresholds()
    
    def calculate(self, data: Dict) -> Dict:
        roic = nan_if_missing(data.get('returnOnCapital'))

        if pd.isna(roic):
            roic = nan_if_missing(data.get('roic')) 
            
        roe = nan_if_missing(data.get('returnOnEquity'))
        roa = nan_if_missing(data.get('returnOnAssets'))
        gross_margin = nan_if_missing(data.get('grossMargins'))
        operating_margin = nan_if_missing(data.get('operatingMargins'))
        net_margin = nan_if_missing(data.get('profitMargins'))
        
        metrics = {
            'roic': roic,
            'roe': roe,
            'roa': roa,
            'gross_margin': gross_margin,
            'operating_margin': operating_margin,
            'net_margin': net_margin
        }
        
        classifications = {
            'roic_class': classify_metric(roic, self.thresholds.roic),
            'roe_class': classify_metric(roe, self.thresholds.roe),
            'roa_class': classify_metric(roa, self.thresholds.roa),
            'gross_margin_class': classify_metric(gross_margin, self.thresholds.gross_margin),
            'operating_margin_class': classify_metric(operating_margin, self.thresholds.operating_margin),
            'net_margin_class': classify_metric(net_margin, self.thresholds.net_margin)
        }

        scores = []
        if pd.notna(roic):
            scores.append(score_metric(roic, -0.10, 0.30) * 0.30) 
        if pd.notna(roe):
            scores.append(score_metric(roe, -0.10, 0.35) * 0.20)  
        if pd.notna(operating_margin):
            scores.append(score_metric(operating_margin, -0.10, 0.30) * 0.25)  
        if pd.notna(net_margin):
            scores.append(score_metric(net_margin, -0.10, 0.25) * 0.25)  
        
        total_weight = sum([0.30, 0.20, 0.25, 0.25][:len(scores)])
        profitability_score = sum(scores) / total_weight if total_weight > 0 else np.nan
        
        return {
            'metrics': metrics,
            'classifications': classifications,
            'score': profitability_score,
            'alerts': self._generate_alerts(metrics)
        }
    
    def _generate_alerts(self, metrics: Dict) -> List[str]:
        alerts = []
        
        if pd.notna(metrics['roic']) and metrics['roic'] < 0.08:
            alerts.append("ROIC bajo: la empresa no genera retornos suficientes sobre el capital invertido")
        
        if pd.notna(metrics['roe']) and metrics['roe'] < 0:
            alerts.append("ROE negativo: la empresa está perdiendo dinero")
        
        if pd.notna(metrics['operating_margin']) and metrics['operating_margin'] < 0.05:
            alerts.append("Margen operativo muy bajo: problemas de eficiencia operativa")
        
        if pd.notna(metrics['net_margin']) and metrics['net_margin'] < 0:
            alerts.append("Margen neto negativo: la empresa no es rentable")
        
        return alerts