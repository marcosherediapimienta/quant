import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass
from .helpers import nan_if_missing, safe_div, score_metric, classify_metric


@dataclass
class FinancialHealthThresholds:
    debt_ebitda: Dict[str, float] = None
    debt_equity: Dict[str, float] = None
    current_ratio: Dict[str, float] = None
    interest_coverage: Dict[str, float] = None
    
    def __post_init__(self):
        self.debt_ebitda = self.debt_ebitda or {'excellent': 1.0, 'good': 2.0, 'fair': 3.0, 'poor': 5.0}
        self.debt_equity = self.debt_equity or {'excellent': 0.3, 'good': 0.5, 'fair': 1.0, 'poor': 2.0}
        self.current_ratio = self.current_ratio or {'excellent': 2.5, 'good': 2.0, 'fair': 1.5, 'poor': 1.0}
        self.interest_coverage = self.interest_coverage or {'excellent': 10.0, 'good': 5.0, 'fair': 3.0, 'poor': 1.5}


class FinancialHealthMetrics:

    def __init__(self, thresholds: FinancialHealthThresholds = None):
        self.thresholds = thresholds or FinancialHealthThresholds()
    
    def calculate(self, data: Dict) -> Dict:
        total_debt = nan_if_missing(data.get('totalDebt'))
        total_cash = nan_if_missing(data.get('totalCash'))
        ebitda = nan_if_missing(data.get('ebitda'))
        current_ratio = nan_if_missing(data.get('currentRatio'))
        quick_ratio = nan_if_missing(data.get('quickRatio'))
        debt_equity = nan_if_missing(data.get('debtToEquity'))

        if pd.notna(debt_equity):
            debt_equity = debt_equity / 100

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
        
        # Score compuesto
        score = self._calculate_score(metrics)
        
        return {
            'metrics': metrics,
            'classifications': classifications,
            'score': score,
            'alerts': self._generate_alerts(metrics)
        }
    
    def _get_free_cash_flow(self, data: Dict) -> float:
        fcf = nan_if_missing(data.get('freeCashflow'))
        
        if pd.notna(fcf):
            return fcf

        operating_cf = nan_if_missing(data.get('operatingCashflow'))
        capex = nan_if_missing(data.get('capitalExpenditures'))
        
        if pd.notna(operating_cf):
            capex_val = capex if pd.notna(capex) else 0
            return operating_cf + capex_val 
        
        return np.nan
    
    def _calculate_net_cash(self, cash: float, debt: float) -> float:
        cash_val = cash if pd.notna(cash) else 0
        debt_val = debt if pd.notna(debt) else 0
        return cash_val - debt_val
    
    def _classify_debt_ratio(self, value: float, thresholds: Dict) -> str:

        if pd.isna(value):
            return 'N/A'
        
        if value <= thresholds['excellent']:
            return 'excellent'
        elif value <= thresholds['good']:
            return 'good'
        elif value <= thresholds['fair']:
            return 'fair'
        elif value <= thresholds['poor']:
            return 'warning'
        return 'danger'
    
    def _classify_fcf(self, fcf: float) -> str:

        if pd.isna(fcf):
            return 'N/A'
        if fcf > 0:
            return 'positive'
        return 'negative'
    
    def _calculate_score(self, metrics: Dict) -> float:
        scores = []
        weights = []
        
        # Debt/EBITDA (25%) - menor es mejor
        if pd.notna(metrics['debt_ebitda']):
            score = score_metric(metrics['debt_ebitda'], 0, 6, higher_is_better=False)
            scores.append(score)
            weights.append(0.25)
        
        # Debt/Equity (20%) - menor es mejor
        if pd.notna(metrics['debt_equity']):
            score = score_metric(metrics['debt_equity'], 0, 3, higher_is_better=False)
            scores.append(score)
            weights.append(0.20)
        
        # Current Ratio (20%) - mayor es mejor (hasta cierto punto)
        if pd.notna(metrics['current_ratio']):
            score = score_metric(metrics['current_ratio'], 0.5, 3.0, higher_is_better=True)
            scores.append(score)
            weights.append(0.20)
        
        # Net Cash/EBITDA (20%)
        if pd.notna(metrics['net_cash_ebitda']):
            score = score_metric(metrics['net_cash_ebitda'], -3, 3, higher_is_better=True)
            scores.append(score)
            weights.append(0.20)
        
        # FCF (15%) - positivo = 100, negativo = 0
        if pd.notna(metrics['free_cash_flow']):
            fcf_score = 100 if metrics['free_cash_flow'] > 0 else 0
            scores.append(fcf_score)
            weights.append(0.15)
        
        if not scores:
            return np.nan
        
        total_weight = sum(weights)
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        
        return weighted_sum / total_weight if total_weight > 0 else np.nan
    
    def _generate_alerts(self, metrics: Dict) -> List[str]:
        alerts = []
        
        # Alerta de deuda alta
        debt_ebitda = metrics['debt_ebitda']
        if pd.notna(debt_ebitda):
            if debt_ebitda > 4:
                alerts.append(f"Deuda/EBITDA muy alta ({debt_ebitda:.1f}x): riesgo de insolvencia")
            elif debt_ebitda > 3:
                alerts.append(f"Deuda/EBITDA elevada ({debt_ebitda:.1f}x): vigilar capacidad de pago")
        
        # Alerta de liquidez
        current_ratio = metrics['current_ratio']
        if pd.notna(current_ratio) and current_ratio < 1.0:
            alerts.append(f"Current Ratio bajo ({current_ratio:.2f}): problemas de liquidez")
        
        # Alerta de FCF negativo
        fcf = metrics['free_cash_flow']
        if pd.notna(fcf) and fcf < 0:
            alerts.append("Free Cash Flow negativo: la empresa consume caja")
        
        # Alerta de caja neta negativa
        net_cash = metrics['net_cash']
        if pd.notna(net_cash) and net_cash < 0:
            alerts.append("Posición de caja neta negativa (más deuda que caja)")
        
        return alerts