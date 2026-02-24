import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional, Sequence
from dataclasses import dataclass

import warnings
try:
    warnings.filterwarnings('ignore', category=pd.errors.Pandas4Warning, module='yfinance')
except AttributeError:
    pass

from ..metrics import (
    ProfitabilityMetrics,
    FinancialHealthMetrics,
    GrowthMetrics,
    EfficiencyMetrics,
    ValuationMultiples,
    ProfitabilityThresholds,
    FinancialHealthThresholds,
    GrowthThresholds,
    EfficiencyThresholds,
    ValuationThresholds
)
from ....tools.config import COMPANY_ANALYSIS_WEIGHTS, CONCLUSION_THRESHOLDS, ROIC_CONFIG

_ANALYSIS_CATEGORIES = ('profitability', 'financial_health', 'growth', 'efficiency', 'valuation')

_BALANCE_SHEET_FIELD_ALIASES = {
    'totalAssets': ('Total Assets', 'TotalAssets'),
    'inventory': ('Inventory',),
    'netReceivables': ('Receivables', 'Net Receivables'),
    'totalStockholderEquity': ('Stockholders Equity', 'Total Stockholder Equity', 'StockholdersEquity'),
}

_INCOME_STMT_FIELD_ALIASES = {
    'costOfRevenue': ('Cost Of Revenue', 'Cost of Revenue'),
    'operatingIncome': ('Operating Income', 'OperatingIncome'),
    'ebit': ('EBIT',),
    'taxProvision': ('Tax Provision', 'TaxProvision'),
    'pretaxIncome': ('Pretax Income', 'Pre-Tax Income', 'PretaxIncome'),
}

_ROIC_BALANCE_FIELD_ALIASES = {
    'investedCapital': ('Invested Capital', 'InvestedCapital'),
    'totalAssets': ('Total Assets', 'TotalAssets'),
    'currentLiabilities': ('Current Liabilities', 'CurrentLiabilities', 'Total Current Liabilities'),
}

_SUMMARY_COLUMNS = {
    'profitability': 'Profitability',
    'financial_health': 'Health',
    'growth': 'Growth',
    'efficiency': 'Efficiency',
    'valuation': 'Valuation',
}

@dataclass
class AnalysisWeights:
    profitability: float = None
    financial_health: float = None
    growth: float = None
    efficiency: float = None
    valuation: float = None
    
    def __post_init__(self):
        defaults = COMPANY_ANALYSIS_WEIGHTS['default']
        for field in _ANALYSIS_CATEGORIES:
            if getattr(self, field) is None:
                setattr(self, field, defaults[field])

        total = sum(getattr(self, f) for f in _ANALYSIS_CATEGORIES)
        if not np.isclose(total, 1.0):
            for field in _ANALYSIS_CATEGORIES:
                setattr(self, field, getattr(self, field) / total)

@dataclass
class ConclusionThresholds:
    excellent: float = None
    good: float = None
    fair: float = None
    weak: float = None
    labels: Dict[str, str] = None
    
    def __post_init__(self):
        cfg = CONCLUSION_THRESHOLDS
        for field in ('excellent', 'good', 'fair', 'weak'):
            if getattr(self, field) is None:
                setattr(self, field, cfg[field])
        self.labels = self.labels or cfg['labels'].copy()

class CompanyAnalyzer:
    def __init__(
        self,
        weights: AnalysisWeights = None,
        conclusion_thresholds: ConclusionThresholds = None,
        profitability_thresholds: ProfitabilityThresholds = None,
        health_thresholds: FinancialHealthThresholds = None,
        growth_thresholds: GrowthThresholds = None,
        efficiency_thresholds: EfficiencyThresholds = None,
        valuation_thresholds: ValuationThresholds = None
    ):
        self.weights = weights or AnalysisWeights()
        self.conclusion_thresholds = conclusion_thresholds or ConclusionThresholds()
        self._calculators = {
            'profitability': ProfitabilityMetrics(profitability_thresholds),
            'financial_health': FinancialHealthMetrics(health_thresholds),
            'growth': GrowthMetrics(growth_thresholds),
            'efficiency': EfficiencyMetrics(efficiency_thresholds),
            'valuation': ValuationMultiples(valuation_thresholds),
        }

    def fetch_data(self, ticker: str) -> Dict:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info or len(info) == 0:
                return {
                    'success': False,
                    'error': f'Could not retrieve data for {ticker}'
                }

            try:
                balance_sheet = stock.quarterly_balance_sheet
                income_stmt = stock.quarterly_income_stmt
                self._enrich_from_balance_sheet(info, balance_sheet)
                self._enrich_from_income_stmt(info, income_stmt)
                self._enrich_roic(info, balance_sheet, income_stmt)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"Could not retrieve financial statements for {ticker}: {e}"
                )
            
            return {
                'success': True,
                'data': info,
                'ticker': ticker
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Error fetching data for {ticker}: {e}'
            }
            
    @staticmethod
    def _safe_float(df: 'pd.DataFrame', field: str, column) -> Optional[float]:
        if field in df.index:
            val = df.loc[field, column]
            if pd.isna(val):
                return None
            try:
                return float(val)
            except (TypeError, ValueError):
                return None
        return None

    def _safe_float_alias(self, df: 'pd.DataFrame', fields: Sequence[str], column) -> Optional[float]:
        for field in fields:
            val = self._safe_float(df, field, column)
            if val is not None:
                return val
        return None

    def _enrich_from_balance_sheet(self, info: Dict, balance_sheet: 'pd.DataFrame') -> None:
        if balance_sheet.empty:
            return
        col = balance_sheet.columns[0]
        for dst, aliases in _BALANCE_SHEET_FIELD_ALIASES.items():
            val = self._safe_float_alias(balance_sheet, aliases, col)
            if val is not None:
                info[dst] = val

    def _enrich_from_income_stmt(self, info: Dict, income_stmt: 'pd.DataFrame') -> None:
        if income_stmt.empty:
            return
        col = income_stmt.columns[0]
        val = self._safe_float_alias(income_stmt, _INCOME_STMT_FIELD_ALIASES['costOfRevenue'], col)
        if val is not None:
            info['costOfRevenue'] = val

    def _enrich_roic(self, info: Dict, balance_sheet: 'pd.DataFrame', income_stmt: 'pd.DataFrame') -> None:
        if income_stmt.empty or balance_sheet.empty:
            return
        is_col = income_stmt.columns[0]
        bs_col = balance_sheet.columns[0]

        operating_income = (
            self._safe_float_alias(income_stmt, _INCOME_STMT_FIELD_ALIASES['operatingIncome'], is_col)
            or self._safe_float_alias(income_stmt, _INCOME_STMT_FIELD_ALIASES['ebit'], is_col)
        )
        if operating_income is None:
            return

        tax_rate = None
        tax_provision = self._safe_float_alias(income_stmt, _INCOME_STMT_FIELD_ALIASES['taxProvision'], is_col)
        pretax_income = self._safe_float_alias(income_stmt, _INCOME_STMT_FIELD_ALIASES['pretaxIncome'], is_col)
        if tax_provision is not None and pretax_income is not None and pretax_income != 0:
            tax_rate = tax_provision / pretax_income
        if tax_rate is None:
            tax_rate = info.get('effectiveTaxRate', ROIC_CONFIG['default_tax_rate'])
        try:
            tax_rate = float(tax_rate)
        except (TypeError, ValueError):
            tax_rate = float(ROIC_CONFIG['default_tax_rate'])
        tax_rate = max(0.0, min(1.0, tax_rate))

        nopat = operating_income * (1 - tax_rate)

        invested_capital = self._safe_float_alias(balance_sheet, _ROIC_BALANCE_FIELD_ALIASES['investedCapital'], bs_col)
        if invested_capital is None:
            total_assets = self._safe_float_alias(balance_sheet, _ROIC_BALANCE_FIELD_ALIASES['totalAssets'], bs_col)
            current_liab = self._safe_float_alias(balance_sheet, _ROIC_BALANCE_FIELD_ALIASES['currentLiabilities'], bs_col)
            if total_assets is not None and current_liab is not None:
                invested_capital = total_assets - current_liab

        if invested_capital and invested_capital > 0:
            info['returnOnCapital'] = nopat / invested_capital

    def _compute_total_score(self, scores: Dict) -> float:
        total_score = 0.0
        total_weight = 0.0
        for category in _ANALYSIS_CATEGORIES:
            score = scores[category]
            if pd.notna(score):
                weight = getattr(self.weights, category)
                total_score += score * weight
                total_weight += weight
        return total_score / total_weight if total_weight > 0 else np.nan

    def _run_all_calculators(self, data: Dict) -> Dict:
        return {cat: calc.calculate(data) for cat, calc in self._calculators.items()}

    def _build_scores(self, category_results: Dict) -> Dict:
        scores = {cat: category_results.get(cat, {}).get('score', np.nan) for cat in _ANALYSIS_CATEGORIES}
        scores['total'] = self._compute_total_score(scores)
        return scores

    def analyze(self, ticker: str, data: Dict = None) -> Dict:
        if data is None:
            fetch_result = self.fetch_data(ticker)
            if not fetch_result.get('success'):
                return {
                    'success': False,
                    'error': fetch_result.get('error'),
                    'ticker': ticker
                }
            data = fetch_result['data']
        
        try:
            category_results = self._run_all_calculators(data)
            scores = self._build_scores(category_results)
            
            result = {
                'success': True,
                'ticker': ticker,
                'scores': scores,
                'conclusion': self._determine_conclusion(scores['total']),
                'sector': data.get('sector', 'N/A'),
                'industry': data.get('industry', 'N/A'),
                'company_name': data.get('longName', data.get('shortName', ticker)),
                'country': data.get('country', 'N/A'),
            }
            result.update(category_results)
            return result
        except Exception as e:
            return {
                'success': False,
                'error': f'Error analyzing {ticker}: {e}',
                'ticker': ticker
            }

    def analyze_quick(self, ticker: str) -> Dict:
        fetch_result = self.fetch_data(ticker)
        if not fetch_result.get('success'):
            return {
                'success': False,
                'error': fetch_result.get('error'),
                'ticker': ticker
            }
        
        try:
            scores = self._build_scores(self._run_all_calculators(fetch_result['data']))
            return {
                'success': True,
                'ticker': ticker,
                'scores': scores,
                'sector': fetch_result['data'].get('sector', 'N/A'),
                'industry': fetch_result['data'].get('industry', 'N/A'),
                'company_name': fetch_result['data'].get('longName', fetch_result['data'].get('shortName', ticker)),
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Error in quick analysis of {ticker}: {e}',
                'ticker': ticker
            }
    
    def analyze_multiple(self, tickers: List[str]) -> Dict[str, Dict]:
        results = {}
        for ticker in tickers:
            results[ticker] = self.analyze(ticker)
        return results
    
    def _determine_conclusion(self, score: float) -> Dict[str, str]:
        thresholds = self.conclusion_thresholds
        labels = thresholds.labels

        if score is None or not np.isfinite(score):
            return {
                'overall': labels.get('na', 'N/A'),
                'score': score
            }
        
        levels = [
            (thresholds.excellent, 'excellent'),
            (thresholds.good, 'good'),
            (thresholds.fair, 'fair'),
        ]
        
        label = labels.get('weak', 'WEAK')
        for threshold_val, key in levels:
            if score >= threshold_val:
                label = labels.get(key, key.upper())
                break
        
        return {
            'overall': label,
            'score': score
        }
    
    def get_summary_df(self, results: Dict[str, Dict]) -> 'pd.DataFrame':
        rows = []
        for ticker, result in results.items():
            if result.get('success'):
                scores = result.get('scores', {})
                row = {
                    'Ticker': ticker,
                    'Name': result.get('company_name', ticker),
                    'Sector': result.get('sector', 'N/A'),
                }
                for key, col_name in _SUMMARY_COLUMNS.items():
                    row[col_name] = scores.get(key, 0)
                row['Total'] = scores.get('total', 0)
                row['Conclusion'] = result.get('conclusion', {}).get('overall', 'N/A')
                rows.append(row)
        
        return pd.DataFrame(rows) if rows else pd.DataFrame()
