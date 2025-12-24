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
    """
    Calcula métricas de rentabilidad.
    
    Responsabilidad: Evaluar capacidad de generar beneficios sobre capital invertido.
    
    Métricas clave:
    - ROIC (Return on Invested Capital): Buffett's favorite
    - ROE (Return on Equity): Rentabilidad sobre patrimonio
    - ROA (Return on Assets): Eficiencia en uso de activos
    - Márgenes (Gross, Operating, Net): Rentabilidad operativa
    """
    
    def __init__(self, thresholds: ProfitabilityThresholds = None):
        self.thresholds = thresholds or ProfitabilityThresholds()
        self.weights = SCORING_WEIGHTS['profitability']
        # Cargar rangos desde config
        from ....tools.config import PROFITABILITY_SCORING_RANGES
        self.ranges = PROFITABILITY_SCORING_RANGES
    
    def calculate(self, data: Dict) -> Dict:
        """Calcula scores y clasificaciones de rentabilidad."""
        # ... (extracción de métricas igual) ...
        roic = nan_if_missing(data.get('returnOnCapital'))
        if pd.isna(roic):
            roic = nan_if_missing(data.get('roic'))  # ❌ Este campo no existe
        roe = nan_if_missing(data.get('returnOnEquity'))  # Este SÍ existe
        roa = nan_if_missing(data.get('returnOnAssets'))   # Este SÍ existe
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

        # Usar rangos de config
        scores = []
        weights_used = []
        
        if pd.notna(roic):
            scores.append(score_metric(
                roic, 
                self.ranges['roic']['min'], 
                self.ranges['roic']['max']
            ) * self.weights['roic'])
            weights_used.append(self.weights['roic'])
        
        if pd.notna(roe):
            scores.append(score_metric(
                roe, 
                self.ranges['roe']['min'], 
                self.ranges['roe']['max']
            ) * self.weights['roe'])
            weights_used.append(self.weights['roe'])
        
        if pd.notna(operating_margin):
            scores.append(score_metric(
                operating_margin, 
                self.ranges['operating_margin']['min'], 
                self.ranges['operating_margin']['max']
            ) * self.weights['operating_margin'])
            weights_used.append(self.weights['operating_margin'])
        
        if pd.notna(net_margin):
            scores.append(score_metric(
                net_margin, 
                self.ranges['net_margin']['min'], 
                self.ranges['net_margin']['max']
            ) * self.weights['net_margin'])
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
        """Genera alertas usando umbrales de config."""
        from ....tools.config import ALERT_THRESHOLDS
        alerts = []
        alert_cfg = ALERT_THRESHOLDS['profitability']
        
        if pd.notna(metrics['roic']) and metrics['roic'] < alert_cfg['roic_low']:
            alerts.append("ROIC bajo: la empresa no genera retornos suficientes sobre el capital invertido")
        
        if pd.notna(metrics['roe']) and metrics['roe'] < ALERT_THRESHOLDS['financial_health'].get('roe_negative', 0):
            alerts.append("ROE negativo: la empresa está perdiendo dinero")
        
        if pd.notna(metrics['operating_margin']) and metrics['operating_margin'] < alert_cfg['operating_margin_low']:
            alerts.append("Margen operativo muy bajo: problemas de eficiencia operativa")
        
        if pd.notna(metrics['net_margin']) and metrics['net_margin'] < ALERT_THRESHOLDS['financial_health'].get('net_margin_negative', 0):
            alerts.append("Margen neto negativo: la empresa no es rentable")
        
        return alerts