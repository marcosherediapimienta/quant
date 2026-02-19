import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore', category=pd.errors.Pandas4Warning, module='yfinance')

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
from ....tools.config import COMPANY_ANALYSIS_WEIGHTS, CONCLUSION_THRESHOLDS

@dataclass
class AnalysisWeights:
    profitability: float = None
    financial_health: float = None
    growth: float = None
    efficiency: float = None
    valuation: float = None
    
    def __post_init__(self):
        defaults = COMPANY_ANALYSIS_WEIGHTS['default']
        self.profitability = self.profitability if self.profitability is not None else defaults['profitability']
        self.financial_health = self.financial_health if self.financial_health is not None else defaults['financial_health']
        self.growth = self.growth if self.growth is not None else defaults['growth']
        self.efficiency = self.efficiency if self.efficiency is not None else defaults['efficiency']
        self.valuation = self.valuation if self.valuation is not None else defaults['valuation']

        total = (self.profitability + self.financial_health + 
                 self.growth + self.efficiency + self.valuation)
        if not np.isclose(total, 1.0):
            self.profitability /= total
            self.financial_health /= total
            self.growth /= total
            self.efficiency /= total
            self.valuation /= total

@dataclass
class ConclusionThresholds:
    excellent: float = None
    good: float = None
    fair: float = None
    weak: float = None
    labels: Dict[str, str] = None
    
    def __post_init__(self):
        cfg = CONCLUSION_THRESHOLDS
        self.excellent = self.excellent if self.excellent is not None else cfg['excellent']
        self.good = self.good if self.good is not None else cfg['good']
        self.fair = self.fair if self.fair is not None else cfg['fair']
        self.weak = self.weak if self.weak is not None else cfg['weak']
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
        self.profitability = ProfitabilityMetrics(profitability_thresholds)
        self.health = FinancialHealthMetrics(health_thresholds)
        self.growth = GrowthMetrics(growth_thresholds)
        self.efficiency = EfficiencyMetrics(efficiency_thresholds)
        self.valuation = ValuationMultiples(valuation_thresholds)

    def fetch_data(self, ticker: str) -> Dict:

        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info or len(info) == 0:
                return {
                    'success': False,
                    'error': f'No se pudieron obtener datos para {ticker}'
                }

            try:
                balance_sheet = stock.quarterly_balance_sheet
                income_stmt = stock.quarterly_income_stmt
                self._enrich_from_balance_sheet(info, balance_sheet)
                self._enrich_from_income_stmt(info, income_stmt)
                self._enrich_roic(info, balance_sheet, income_stmt)
            except Exception as e:
                print(f"⚠️  No se pudieron obtener estados financieros para {ticker}: {str(e)}")
            
            return {
                'success': True,
                'data': info,
                'ticker': ticker
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Error obteniendo datos de {ticker}: {str(e)}'
            }
            
    @staticmethod
    def _safe_float(df: 'pd.DataFrame', field: str, column) -> float:
        if field in df.index:
            return float(df.loc[field, column])
        return None

    def _enrich_from_balance_sheet(self, info: Dict, balance_sheet: 'pd.DataFrame') -> None:
        if balance_sheet.empty:
            return
        col = balance_sheet.columns[0]
        field_map = {
            'Total Assets': 'totalAssets',
            'Inventory': 'inventory',
            'Receivables': 'netReceivables',
            'Stockholders Equity': 'totalStockholderEquity',
        }
        for src, dst in field_map.items():
            val = self._safe_float(balance_sheet, src, col)
            if val is not None:
                info[dst] = val

    def _enrich_from_income_stmt(self, info: Dict, income_stmt: 'pd.DataFrame') -> None:
        if income_stmt.empty:
            return
        col = income_stmt.columns[0]
        val = self._safe_float(income_stmt, 'Cost Of Revenue', col)
        if val is not None:
            info['costOfRevenue'] = val

    def _enrich_roic(self, info: Dict, balance_sheet: 'pd.DataFrame', income_stmt: 'pd.DataFrame') -> None:
        if income_stmt.empty or balance_sheet.empty:
            return
        is_col = income_stmt.columns[0]
        bs_col = balance_sheet.columns[0]

        operating_income = (
            self._safe_float(income_stmt, 'Operating Income', is_col)
            or self._safe_float(income_stmt, 'EBIT', is_col)
        )
        if operating_income is None:
            return

        tax_rate = None
        tax_provision = self._safe_float(income_stmt, 'Tax Provision', is_col)
        pretax_income = self._safe_float(income_stmt, 'Pretax Income', is_col)
        if tax_provision is not None and pretax_income is not None and pretax_income != 0:
            tax_rate = tax_provision / pretax_income
        if tax_rate is None:
            tax_rate = info.get('effectiveTaxRate', 0.21)

        nopat = operating_income * (1 - tax_rate)

        invested_capital = self._safe_float(balance_sheet, 'Invested Capital', bs_col)
        if invested_capital is None:
            total_assets = self._safe_float(balance_sheet, 'Total Assets', bs_col)
            current_liab = self._safe_float(balance_sheet, 'Current Liabilities', bs_col)
            if total_assets is not None and current_liab is not None:
                invested_capital = total_assets - current_liab

        if invested_capital and invested_capital > 0:
            info['returnOnCapital'] = nopat / invested_capital

    def _compute_total_score(self, scores: Dict) -> float:
        total_score = 0.0
        total_weight = 0.0
        for category in ('profitability', 'financial_health', 'growth', 'efficiency', 'valuation'):
            score = scores[category]
            if pd.notna(score):
                weight = getattr(self.weights, category)
                total_score += score * weight
                total_weight += weight
        return total_score / total_weight if total_weight > 0 else np.nan

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
            profitability_result = self.profitability.calculate(data)
            health_result = self.health.calculate(data)
            growth_result = self.growth.calculate(data)
            efficiency_result = self.efficiency.calculate(data)
            valuation_result = self.valuation.calculate(data)
            scores = {
                'profitability': profitability_result.get('score', 0),
                'financial_health': health_result.get('score', 0),
                'growth': growth_result.get('score', 0),
                'efficiency': efficiency_result.get('score', 0),
                'valuation': valuation_result.get('score', 0)
            }
            scores['total'] = self._compute_total_score(scores)
            conclusion = self._determine_conclusion(scores['total'])
            
            return {
                'success': True,
                'ticker': ticker,
                'scores': scores,
                'conclusion': conclusion,
                'profitability': profitability_result,
                'financial_health': health_result,
                'growth': growth_result,
                'efficiency': efficiency_result,
                'valuation': valuation_result,
                'sector': data.get('sector', 'N/A'),
                'industry': data.get('industry', 'N/A'),
                'company_name': data.get('longName', data.get('shortName', ticker)),
                'country': data.get('country', 'N/A'),
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Error analizando {ticker}: {str(e)}',
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
        
        data = fetch_result['data']
        
        try:
            scores = {
                'profitability': self.profitability.calculate(data).get('score', 0),
                'financial_health': self.health.calculate(data).get('score', 0),
                'growth': self.growth.calculate(data).get('score', 0),
                'efficiency': self.efficiency.calculate(data).get('score', 0),
                'valuation': self.valuation.calculate(data).get('score', 0),
            }
            scores['total'] = self._compute_total_score(scores)
            
            return {
                'success': True,
                'ticker': ticker,
                'scores': scores,
                'sector': data.get('sector', 'N/A'),
                'industry': data.get('industry', 'N/A'),
                'company_name': data.get('longName', data.get('shortName', ticker)),
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Error en análisis rápido de {ticker}: {str(e)}',
                'ticker': ticker
            }
    
    def analyze_multiple(self, tickers: List[str]) -> Dict[str, Dict]:
        results = {}
        for ticker in tickers:
            results[ticker] = self.analyze(ticker)
        return results
    
    def _determine_conclusion(self, score: float) -> Dict[str, str]:
        thresholds = self.conclusion_thresholds
        
        if score >= thresholds.excellent:
            label = thresholds.labels.get('excellent', 'Excelente')
        elif score >= thresholds.good:
            label = thresholds.labels.get('good', 'Bueno')
        elif score >= thresholds.fair:
            label = thresholds.labels.get('fair', 'Regular')
        else:
            label = thresholds.labels.get('weak', 'Débil')
        
        return {
            'overall': label,
            'score': score
        }
    
    def get_summary_df(self, results: Dict[str, Dict]) -> 'pd.DataFrame':
        rows = []
        for ticker, result in results.items():
            if result.get('success'):
                scores = result.get('scores', {})
                rows.append({
                    'Ticker': ticker,
                    'Nombre': result.get('company_name', ticker),
                    'Sector': result.get('sector', 'N/A'),
                    'Rentabilidad': scores.get('profitability', 0),
                    'Salud': scores.get('financial_health', 0),
                    'Crecimiento': scores.get('growth', 0),
                    'Eficiencia': scores.get('efficiency', 0),
                    'Valoración': scores.get('valuation', 0),
                    'Total': scores.get('total', 0),
                    'Conclusión': result.get('conclusion', {}).get('overall', 'N/A')
                })
        
        return pd.DataFrame(rows) if rows else pd.DataFrame()