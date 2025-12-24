import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass
from .helpers import nan_if_missing, score_metric, classify_metric
from ....tools.config import (
    VALUATION_THRESHOLDS, 
    SCORING_RANGES,
    GROWTH_SCORING_WEIGHTS,
    ALERT_THRESHOLDS
)

@dataclass
class GrowthThresholds:
    """Thresholds para clasificación de crecimiento."""
    revenue_growth: Dict[str, float] = None
    earnings_growth: Dict[str, float] = None
    
    def __post_init__(self):
        # Usar thresholds de config si están disponibles
        val_thresh = VALUATION_THRESHOLDS.get('growth', {})
        self.revenue_growth = self.revenue_growth or val_thresh.get('revenue_growth',
            {'excellent': 0.25, 'good': 0.15, 'fair': 0.08, 'poor': 0.03})
        self.earnings_growth = self.earnings_growth or val_thresh.get('earnings_growth',
            {'excellent': 0.30, 'good': 0.20, 'fair': 0.10, 'poor': 0.05})

class GrowthMetrics:
    """
    Calcula métricas de crecimiento.
    
    Responsabilidad: Evaluar tasas de crecimiento de revenue y earnings.
    
    Métricas clave:
    - Revenue Growth YoY: Crecimiento de ingresos año sobre año
    - Earnings Growth YoY: Crecimiento de beneficios
    - Earnings Quarterly Growth: Crecimiento trimestral
    
    Interpretación:
    - Growth >20%: Alto crecimiento (verificar sostenibilidad)
    - Growth 10-20%: Crecimiento sólido
    - Growth 5-10%: Crecimiento moderado
    - Growth <5%: Crecimiento bajo/maduro
    - Growth negativo: Declive (requiere análisis profundo)
    """
    
    def __init__(self, thresholds: GrowthThresholds = None):
        self.thresholds = thresholds or GrowthThresholds()
        # Obtener rangos y pesos desde config
        self.ranges = SCORING_RANGES['growth']
        self.weights = GROWTH_SCORING_WEIGHTS
    
    def calculate(self, data: Dict) -> Dict:
        """Calcula scores y clasificaciones de crecimiento."""
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

        # Usar rangos y pesos de config
        scores = []
        if pd.notna(revenue_growth):
            scores.append(score_metric(
                revenue_growth, 
                self.ranges['revenue']['min'], 
                self.ranges['revenue']['max']
            ) * self.weights['revenue'])
        if pd.notna(earnings_growth):
            scores.append(score_metric(
                earnings_growth, 
                self.ranges['earnings']['min'], 
                self.ranges['earnings']['max']
            ) * self.weights['earnings'])
        
        # Total weight es la suma de pesos usados
        total_weight = sum([self.weights['revenue'], self.weights['earnings']][:len(scores)])
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
        """
        Analiza sostenibilidad del crecimiento.
        
        Red flags:
        - Earnings crecen mucho más rápido que revenue (posible manipulación contable)
        - Revenue en declive sostenido
        """
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
                analysis['concerns'].append("Earnings crecen más rápido que ventas - verificar sostenibilidad")
                analysis['is_sustainable'] = False

        if pd.notna(rev_g) and rev_g < alert_cfg['revenue_decline_mild']:
            analysis['concerns'].append("Ventas en declive")
            analysis['is_sustainable'] = False
        
        return analysis
    
    def _generate_alerts(self, metrics: Dict) -> List[str]:
        """Genera alertas basadas en métricas usando umbrales de config."""
        alerts = []
        alert_cfg = ALERT_THRESHOLDS['growth']
        
        if pd.notna(metrics['revenue_growth_yoy']) and metrics['revenue_growth_yoy'] < alert_cfg['revenue_decline_significant']:
            alerts.append(f"Caída significativa de ingresos (>{abs(alert_cfg['revenue_decline_significant'])*100:.0f}%)")
        
        if pd.notna(metrics['earnings_growth_yoy']) and metrics['earnings_growth_yoy'] < alert_cfg['earnings_decline_strong']:
            alerts.append(f"Caída fuerte de beneficios (>{abs(alert_cfg['earnings_decline_strong'])*100:.0f}%)")
        
        if pd.notna(metrics['revenue_growth_yoy']) and metrics['revenue_growth_yoy'] > alert_cfg['growth_too_high']:
            alerts.append(f"Crecimiento muy alto (>{alert_cfg['growth_too_high']*100:.0f}%) - verificar sostenibilidad")
        
        return alerts