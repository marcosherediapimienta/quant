import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass
from .helpers import nan_if_missing, safe_div, score_metric, classify_metric

from ....tools.config import VALUATION_THRESHOLDS, FINANCIAL_HEALTH_SCORING, ALERT_THRESHOLDS

@dataclass
class FinancialHealthThresholds:
    debt_ebitda: Dict[str, float] = None
    debt_equity: Dict[str, float] = None
    current_ratio: Dict[str, float] = None
    interest_coverage: Dict[str, float] = None
    
    def __post_init__(self):
        health_thresholds = VALUATION_THRESHOLDS['financial_health']
        self.debt_ebitda = self.debt_ebitda or health_thresholds['debt_ebitda']
        self.debt_equity = self.debt_equity or health_thresholds['debt_equity']
        self.current_ratio = self.current_ratio or health_thresholds['current_ratio']
        self.interest_coverage = self.interest_coverage or health_thresholds['interest_coverage']


class FinancialHealthMetrics:
    """
    Calcula métricas de salud financiera.
    
    Responsabilidad: Evaluar solidez financiera y capacidad de pago.
    
    Métricas clave:
    - Debt/EBITDA: Capacidad de pago de deuda (S&P: <3x es sano)
    - Debt/Equity: Nivel de apalancamiento
    - Current Ratio: Liquidez de corto plazo (>1.5 es sano)
    - FCF: Generación de caja libre
    """

    def __init__(self, thresholds: FinancialHealthThresholds = None):
        self.thresholds = thresholds or FinancialHealthThresholds()
        self.config = FINANCIAL_HEALTH_SCORING
    
    def calculate(self, data: Dict) -> Dict:
        """Calcula scores y clasificaciones de salud financiera."""
        total_debt = nan_if_missing(data.get('totalDebt'))
        total_cash = nan_if_missing(data.get('totalCash'))
        ebitda = nan_if_missing(data.get('ebitda'))
        current_ratio = nan_if_missing(data.get('currentRatio'))
        quick_ratio = nan_if_missing(data.get('quickRatio'))
        debt_equity = nan_if_missing(data.get('debtToEquity'))

        # Convertir debt_equity si viene en porcentaje (>100)
        if pd.notna(debt_equity):
            conversion_factor = ALERT_THRESHOLDS['financial_health'].get('debt_equity_conversion_factor', 100)
            if debt_equity > 10:  # Probablemente está en %
                debt_equity = debt_equity / conversion_factor

        debt_ebitda = safe_div(total_debt, ebitda)
        net_cash = self._calculate_net_cash(total_cash, total_debt)
        net_cash_ebitda = safe_div(net_cash, ebitda)
        fcf = self._get_free_cash_flow(data)
        interest_coverage = nan_if_missing(data.get('interestCoverage'))
        
        metrics = {
            'total_debt': total_debt,
            'total_cash': total_cash,
            'net_cash': net_cash,
            'debt_ebitda': debt_ebitda,
            'debt_equity': debt_equity,
            'net_cash_ebitda': net_cash_ebitda,
            'current_ratio': current_ratio,
            'quick_ratio': quick_ratio,
            'free_cash_flow': fcf,
            'interest_coverage': interest_coverage,
            'ebitda': ebitda
        }
        
        classifications = {
            'debt_ebitda_class': self._classify_debt_ratio(debt_ebitda, self.thresholds.debt_ebitda),
            'debt_equity_class': self._classify_debt_ratio(debt_equity, self.thresholds.debt_equity),
            'current_ratio_class': classify_metric(current_ratio, self.thresholds.current_ratio),
            'fcf_class': self._classify_fcf(fcf),
            'net_cash_class': 'positive' if pd.notna(net_cash) and net_cash > 0 else 'negative' if pd.notna(net_cash) else 'N/A'
        }
        
        score = self._calculate_score(metrics)
        
        return {
            'metrics': metrics,
            'classifications': classifications,
            'score': score,
            'alerts': self._generate_alerts(metrics)
        }
    
    # ... (métodos helper iguales) ...
    
    def _calculate_score(self, metrics: Dict) -> float:
        """Calcula score usando pesos y rangos de config."""
        scores = []
        weights = []
        cfg_weights = self.config['weights']
        cfg_ranges = self.config['ranges']

        if pd.notna(metrics['debt_ebitda']):
            score = score_metric(
                metrics['debt_ebitda'], 
                cfg_ranges['debt_ebitda']['min'],
                cfg_ranges['debt_ebitda']['max'],
                higher_is_better=False
            )
            scores.append(score)
            weights.append(cfg_weights['debt_ebitda'])

        if pd.notna(metrics['debt_equity']):
            score = score_metric(
                metrics['debt_equity'], 
                cfg_ranges['debt_equity']['min'],
                cfg_ranges['debt_equity']['max'],
                higher_is_better=False
            )
            scores.append(score)
            weights.append(cfg_weights['debt_equity'])

        if pd.notna(metrics['current_ratio']):
            score = score_metric(
                metrics['current_ratio'], 
                cfg_ranges['current_ratio']['min'],
                cfg_ranges['current_ratio']['max'],
                higher_is_better=True
            )
            scores.append(score)
            weights.append(cfg_weights['current_ratio'])

        if pd.notna(metrics['net_cash_ebitda']):
            score = score_metric(
                metrics['net_cash_ebitda'], 
                cfg_ranges['net_cash_ebitda']['min'],
                cfg_ranges['net_cash_ebitda']['max'],
                higher_is_better=True
            )
            scores.append(score)
            weights.append(cfg_weights['net_cash_ebitda'])

        if pd.notna(metrics['free_cash_flow']):
            fcf_score = 100 if metrics['free_cash_flow'] > 0 else 0
            scores.append(fcf_score)
            weights.append(cfg_weights['free_cash_flow'])
        
        if not scores:
            return np.nan
        
        total_weight = sum(weights)
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        
        return weighted_sum / total_weight if total_weight > 0 else np.nan
    
    def _generate_alerts(self, metrics: Dict) -> List[str]:
        """Genera alertas usando umbrales de config."""
        alerts = []
        alert_cfg = ALERT_THRESHOLDS['financial_health']
        
        debt_ebitda = metrics['debt_ebitda']
        if pd.notna(debt_ebitda):
            if debt_ebitda > alert_cfg['debt_ebitda_danger']:
                alerts.append(f"Deuda/EBITDA muy alta ({debt_ebitda:.1f}x): riesgo de insolvencia")
            elif debt_ebitda > alert_cfg['debt_ebitda_warning']:
                alerts.append(f"Deuda/EBITDA elevada ({debt_ebitda:.1f}x): vigilar capacidad de pago")

        current_ratio = metrics['current_ratio']
        if pd.notna(current_ratio) and current_ratio < alert_cfg['current_ratio_low']:
            alerts.append(f"Current Ratio bajo ({current_ratio:.2f}): problemas de liquidez")

        fcf = metrics['free_cash_flow']
        if pd.notna(fcf) and fcf < alert_cfg.get('fcf_negative', 0):
            alerts.append("Free Cash Flow negativo: la empresa consume caja")

        net_cash = metrics['net_cash']
        if pd.notna(net_cash) and net_cash < alert_cfg.get('net_cash_negative', 0):
            alerts.append("Posición de caja neta negativa (más deuda que caja)")
        
        return alerts