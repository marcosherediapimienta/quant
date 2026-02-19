import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass
from .helpers import nan_if_missing, safe_div, classify_metric, MetricSpec, WeightedScorer

from ....tools.config import VALUATION_THRESHOLDS, FINANCIAL_HEALTH_SCORING, ALERT_THRESHOLDS

@dataclass
class FinancialHealthThresholds:
    debt_ebitda: Dict[str, float] = None
    debt_equity: Dict[str, float] = None
    current_ratio: Dict[str, float] = None
    interest_coverage: Dict[str, float] = None
    
    def __post_init__(self):
        health = VALUATION_THRESHOLDS['financial_health']
        for field in ('debt_ebitda', 'debt_equity', 'current_ratio', 'interest_coverage'):
            if getattr(self, field) is None:
                setattr(self, field, health[field])


_SCORING_SPECS = [
    MetricSpec(key='debt_ebitda', range_key='debt_ebitda', weight_key='debt_ebitda', higher_is_better=False),
    MetricSpec(key='debt_equity', range_key='debt_equity', weight_key='debt_equity', higher_is_better=False),
    MetricSpec(key='current_ratio', range_key='current_ratio', weight_key='current_ratio', higher_is_better=True),
    MetricSpec(key='net_cash_ebitda', range_key='net_cash_ebitda', weight_key='net_cash_ebitda', higher_is_better=True),
    MetricSpec(key='free_cash_flow', range_key='free_cash_flow', weight_key='free_cash_flow', binary_positive=True),
]


class FinancialHealthMetrics:
    def __init__(self, thresholds: FinancialHealthThresholds = None):
        self.thresholds = thresholds or FinancialHealthThresholds()
        self.config = FINANCIAL_HEALTH_SCORING
    
    def calculate(self, data: Dict) -> Dict:
        total_debt = nan_if_missing(data.get('totalDebt'))
        total_cash = nan_if_missing(data.get('totalCash'))
        ebitda = nan_if_missing(data.get('ebitda'))
        current_ratio = nan_if_missing(data.get('currentRatio'))
        quick_ratio = nan_if_missing(data.get('quickRatio'))
        debt_equity = self._normalize_debt_equity(nan_if_missing(data.get('debtToEquity')))

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
            'debt_ebitda_class': classify_metric(
                debt_ebitda, self.thresholds.debt_ebitda,
                higher_is_better=False, strict=False, default='poor'
            ),
            'debt_equity_class': classify_metric(
                debt_equity, self.thresholds.debt_equity,
                higher_is_better=False, strict=False, default='poor'
            ),
            'current_ratio_class': classify_metric(current_ratio, self.thresholds.current_ratio),
            'fcf_class': self._classify_fcf(fcf),
            'net_cash_class': 'positive' if pd.notna(net_cash) and net_cash > 0 else 'negative' if pd.notna(net_cash) else 'N/A'
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
    
    @staticmethod
    def _normalize_debt_equity(debt_equity: float) -> float:
        if pd.isna(debt_equity):
            return debt_equity
        alert_cfg = ALERT_THRESHOLDS['financial_health']
        threshold = alert_cfg['debt_equity_likely_percentage_threshold']
        factor = alert_cfg['debt_equity_conversion_factor']
        if debt_equity > threshold:
            debt_equity = debt_equity / factor
        return debt_equity
    
    def _generate_alerts(self, metrics: Dict) -> List[str]:
        alerts = []
        alert_cfg = ALERT_THRESHOLDS['financial_health']
        
        debt_ebitda = metrics['debt_ebitda']
        if pd.notna(debt_ebitda):
            if debt_ebitda > alert_cfg['debt_ebitda_danger']:
                alerts.append(f"Debt/EBITDA very high ({debt_ebitda:.1f}x): insolvency risk")
            elif debt_ebitda > alert_cfg['debt_ebitda_warning']:
                alerts.append(f"Debt/EBITDA elevated ({debt_ebitda:.1f}x): monitor payment capacity")

        current_ratio = metrics['current_ratio']
        if pd.notna(current_ratio) and current_ratio < alert_cfg['current_ratio_low']:
            alerts.append(f"Current Ratio low ({current_ratio:.2f}): liquidity issues")

        fcf = metrics['free_cash_flow']
        if pd.notna(fcf) and fcf < alert_cfg['fcf_negative']:
            alerts.append("Negative Free Cash Flow: company is consuming cash")

        net_cash = metrics['net_cash']
        if pd.notna(net_cash) and net_cash < alert_cfg['net_cash_negative']:
            alerts.append("Negative net cash position (more debt than cash)")
        
        return alerts

    @staticmethod
    def _calculate_net_cash(total_cash: float, total_debt: float) -> float:
        if pd.isna(total_cash) or pd.isna(total_debt):
            return np.nan
        return total_cash - total_debt
    
    @staticmethod
    def _get_free_cash_flow(data: Dict) -> float:
        fcf = nan_if_missing(data.get('freeCashflow'))
        
        if pd.isna(fcf):
            operating_cf = nan_if_missing(data.get('operatingCashflow'))
            capex = nan_if_missing(data.get('capitalExpenditures'))
            
            if pd.notna(operating_cf) and pd.notna(capex):
                capex_adj = -abs(capex) if capex > 0 else capex
                fcf = operating_cf + capex_adj
        
        return fcf
    
    @staticmethod
    def _classify_fcf(fcf: float) -> str:
        if pd.isna(fcf):
            return 'N/A'
        return 'positive' if fcf > 0 else 'neutral' if fcf == 0 else 'negative'
