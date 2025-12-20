import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass
from .helpers import nan_if_missing, score_metric, classify_metric
from ....tools.config import VALUATION_THRESHOLDS, SCORING_WEIGHTS, ALERT_THRESHOLDS

@dataclass
class ProfitabilityThresholds:
    roic: Dict[str, float] = None
    roe: Dict[str, float] = None
    roa: Dict[str, float] = None
    gross_margin: Dict[str, float] = None
    operating_margin: Dict[str, float] = None
    net_margin: Dict[str, float] = None
    
    def __post_init__(self):
        profitability_thresholds = VALUATION_THRESHOLDS['profitability']
        self.roic = self.roic or profitability_thresholds['roic']
        self.roe = self.roe or profitability_thresholds['roe']
        self.roa = self.roa or profitability_thresholds['roa']
        self.gross_margin = self.gross_margin or profitability_thresholds['gross_margin']
        self.operating_margin = self.operating_margin or profitability_thresholds['operating_margin']
        self.net_margin = self.net_margin or profitability_thresholds['net_margin']


class ProfitabilityMetrics:
    
    def __init__(self, thresholds: ProfitabilityThresholds = None):
        self.thresholds = thresholds or ProfitabilityThresholds()
        self.weights = SCORING_WEIGHTS['profitability']
    
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
        weights_used = []
        
        if pd.notna(roic):
            scores.append(score_metric(roic, -0.10, 0.30) * self.weights['roic'])
            weights_used.append(self.weights['roic'])
        
        if pd.notna(roe):
            scores.append(score_metric(roe, -0.10, 0.35) * self.weights['roe'])
            weights_used.append(self.weights['roe'])
        
        if pd.notna(operating_margin):
            scores.append(score_metric(operating_margin, -0.10, 0.30) * self.weights['operating_margin'])
            weights_used.append(self.weights['operating_margin'])
        
        if pd.notna(net_margin):
            scores.append(score_metric(net_margin, -0.10, 0.25) * self.weights['net_margin'])
            weights_used.append(self.weights['net_margin'])
        
        total_weight = sum(weights_used)
        profitability_score = sum(scores) / total_weight if total_weight > 0 else np.nan
        
        return {
            'metrics': metrics,
            'classifications': classifications,
            'score': profitability_score,
            'alerts': self._generate_alerts(metrics)
        }
    
    def _generate_alerts(self, metrics: Dict) -> List[str]:
        alerts = []
        alert_thresholds = ALERT_THRESHOLDS['profitability']
        
        if pd.notna(metrics['roic']) and metrics['roic'] < alert_thresholds['roic_low']:
            alerts.append("ROIC bajo: la empresa no genera retornos suficientes sobre el capital invertido")
        
        if pd.notna(metrics['roe']) and metrics['roe'] < 0:
            alerts.append("ROE negativo: la empresa está perdiendo dinero")
        
        if pd.notna(metrics['operating_margin']) and metrics['operating_margin'] < alert_thresholds['operating_margin_low']:
            alerts.append("Margen operativo muy bajo: problemas de eficiencia operativa")
        
        if pd.notna(metrics['net_margin']) and metrics['net_margin'] < 0:
            alerts.append("Margen neto negativo: la empresa no es rentable")
        
        return alerts