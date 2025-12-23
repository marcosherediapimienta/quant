import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Optional, List
from dataclasses import dataclass, field

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

@dataclass
class AnalysisWeights:
    profitability: float = 0.25
    financial_health: float = 0.25
    growth: float = 0.20
    efficiency: float = 0.15
    valuation: float = 0.15
    
    def __post_init__(self):
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
    excellent: float = 80
    good: float = 65
    fair: float = 50
    weak: float = 35
    
    labels: Dict[str, str] = field(default_factory=lambda: {
        'excellent': 'EXCELENTE',
        'good': 'BUENA',
        'fair': 'REGULAR',
        'weak': 'DÉBIL',
        'critical': 'CRÍTICA',
        'insufficient': 'DATOS INSUFICIENTES'
    })

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
            info = stock.info or {}
   
            data = {**info}

            self._add_balance_sheet_data(stock, data)
            self._add_cashflow_data(stock, data)
            self._add_financials_data(stock, data)
            self._calculate_derived_metrics(data)
            
            return {
                'ticker': ticker.upper(),
                'name': info.get('shortName', ticker),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'country': info.get('country'),
                'currency': info.get('currency', 'USD'),
                'data': data,
                'success': True
            }
            
        except Exception as e:
            return {
                'ticker': ticker.upper(),
                'error': str(e),
                'success': False
            }
    
    def _add_balance_sheet_data(self, stock, data: Dict):
        try:
            bs = stock.balance_sheet
            if bs is None or bs.empty:
                return

            latest = bs.iloc[:, 0]
            mappings = {
                'totalAssets': ['Total Assets'],
                'totalCurrentAssets': ['Total Current Assets', 'Current Assets'],
                'totalCurrentLiabilities': ['Total Current Liabilities', 'Current Liabilities'],
                'inventory': ['Inventory'],
                'netReceivables': ['Net Receivables', 'Receivables', 'Accounts Receivable'],
                'totalStockholderEquity': ['Total Stockholder Equity', 'Stockholders Equity', 'Total Equity Gross Minority Interest', 'Common Stock Equity']
            }
            
            for key, possible_names in mappings.items():
                if key not in data or pd.isna(data.get(key)):
                    data[key] = self._get_from_series(latest, possible_names)
                    
        except Exception:
            pass
    
    def _add_cashflow_data(self, stock, data: Dict):
        try:
            cf = stock.cashflow
            if cf is None or cf.empty:
                return
            
            latest = cf.iloc[:, 0]
            
            mappings = {
                'operatingCashflow': ['Total Cash From Operating Activities', 'Operating Cash Flow', 'Cash Flow From Continuing Operating Activities'],
                'capitalExpenditures': ['Capital Expenditures', 'Capital Expenditure', 'Purchase Of Property Plant And Equipment']
            }
            
            for key, possible_names in mappings.items():
                if key not in data or pd.isna(data.get(key)):
                    data[key] = self._get_from_series(latest, possible_names)
                    
        except Exception:
            pass
    
    def _add_financials_data(self, stock, data: Dict):
        try:
            fin = stock.financials
            if fin is None or fin.empty:
                return
            
            latest = fin.iloc[:, 0]
            
            mappings = {
                'costOfRevenue': ['Cost Of Revenue', 'Cost of Revenue'],
                'ebit': ['Ebit', 'EBIT', 'Operating Income']
            }
            
            for key, possible_names in mappings.items():
                if key not in data or pd.isna(data.get(key)):
                    data[key] = self._get_from_series(latest, possible_names)

            if 'totalRevenue' not in data or pd.isna(data.get('totalRevenue')):
                data['totalRevenue'] = self._get_from_series(latest, ['Total Revenue', 'Revenue'])
                    
        except Exception:
            pass
    
    def _get_from_series(self, series: pd.Series, keys: List[str]) -> Optional[float]:
        for key in keys:
            try:
                if key in series.index:
                    val = series[key]
                    if pd.notna(val):
                        return float(val)
            except Exception:
                continue
        return np.nan
    
    def _calculate_derived_metrics(self, data: Dict):        
        if pd.isna(data.get('assetTurnover')):
            revenue = data.get('totalRevenue')
            assets = data.get('totalAssets')
            if pd.notna(revenue) and pd.notna(assets) and assets != 0:
                data['assetTurnover'] = revenue / assets

        if pd.isna(data.get('returnOnCapital')):
            ebit = data.get('ebit')
            if pd.notna(ebit):
                nopat = ebit * (1 - 0.21)  
                debt = data.get('totalDebt', 0) or 0
                equity = data.get('totalStockholderEquity')
                cash = data.get('totalCash', 0) or 0
                
                if pd.notna(equity) and equity > 0:
                    invested_capital = debt + equity - cash
                    if invested_capital > 0:
                        data['returnOnCapital'] = nopat / invested_capital

            ocf = data.get('operatingCashflow') 
            capex = data.get('capitalExpenditures', 0) or 0
            market_cap = data.get('marketCap')

            if pd.notna(ocf):
                calculated_fcf = ocf + capex
                
                if pd.notna(market_cap) and market_cap > 0:
                    fcf_yield_check = calculated_fcf / market_cap

                    if fcf_yield_check > 0.10 or fcf_yield_check < -0.10:
                        data['freeCashflow'] = np.nan
                    else:
                        data['freeCashflow'] = calculated_fcf
                else:
                        data['freeCashflow'] = calculated_fcf
    
    def analyze(self, ticker: str, data: Dict = None) -> Dict:

        if data is not None:
            company = {
                'ticker': ticker.upper(),
                'name': data.get('shortName', ticker),
                'sector': data.get('sector'),
                'industry': data.get('industry'),
                'country': data.get('country'),
                'currency': data.get('currency', 'USD'),
                'data': data,
                'success': True
            }
        else:
            company = self.fetch_data(ticker)
        
        if not company.get('success', False):
            return {
                'ticker': ticker.upper(),
                'error': company.get('error', 'Error desconocido'),
                'success': False
            }
        
        raw_data = company['data']

        profitability_result = self.profitability.calculate(raw_data)
        health_result = self.health.calculate(raw_data)
        growth_result = self.growth.calculate(raw_data)
        efficiency_result = self.efficiency.calculate(raw_data)
        valuation_result = self.valuation.calculate(raw_data)
        total_score = self._calculate_total_score(
            profitability_result['score'],
            health_result['score'],
            growth_result['score'],
            efficiency_result['score'],
            valuation_result['score']
        )

        conclusion = self._determine_conclusion(total_score)
        all_alerts = self._consolidate_alerts(
            profitability=profitability_result,
            financial_health=health_result,
            growth=growth_result,
            efficiency=efficiency_result,
            valuation=valuation_result
        )
        
        return {
            'ticker': company['ticker'],
            'name': company['name'],
            'sector': company['sector'],
            'industry': company['industry'],
            'country': company['country'],
            'currency': company['currency'],
            
            'profitability': profitability_result,
            'financial_health': health_result,
            'growth': growth_result,
            'efficiency': efficiency_result,
            'valuation': valuation_result,
            
            'scores': {
                'profitability': profitability_result['score'],
                'financial_health': health_result['score'],
                'growth': growth_result['score'],
                'efficiency': efficiency_result['score'],
                'valuation': valuation_result['score'],
                'total': total_score
            },
            
            'weights_used': {
                'profitability': self.weights.profitability,
                'financial_health': self.weights.financial_health,
                'growth': self.weights.growth,
                'efficiency': self.weights.efficiency,
                'valuation': self.weights.valuation
            },
            
            'conclusion': conclusion,
            'alerts': all_alerts,
            'success': True
        }
    
    def analyze_multiple(self, tickers: List[str]) -> Dict[str, Dict]:
        return {ticker: self.analyze(ticker) for ticker in tickers}
    
    def _calculate_total_score(self, *scores) -> float:
        weights = [
            self.weights.profitability,
            self.weights.financial_health,
            self.weights.growth,
            self.weights.efficiency,
            self.weights.valuation
        ]
        
        valid_pairs = [
            (s, w) for s, w in zip(scores, weights) 
            if pd.notna(s)
        ]
        
        if not valid_pairs:
            return np.nan
        
        total_weight = sum(w for _, w in valid_pairs)
        weighted_sum = sum(s * w for s, w in valid_pairs)
        
        return weighted_sum / total_weight if total_weight > 0 else np.nan
    
    def _determine_conclusion(self, score: float) -> str:
        labels = self.conclusion_thresholds.labels
        
        if pd.isna(score):
            return labels.get('insufficient', 'DATOS INSUFICIENTES')
        
        if score >= self.conclusion_thresholds.excellent:
            return labels.get('excellent', 'EXCELENTE')
        elif score >= self.conclusion_thresholds.good:
            return labels.get('good', 'BUENA')
        elif score >= self.conclusion_thresholds.fair:
            return labels.get('fair', 'REGULAR')
        elif score >= self.conclusion_thresholds.weak:
            return labels.get('weak', 'DÉBIL')
        
        return labels.get('critical', 'CRÍTICA')
    
    def _consolidate_alerts(self, **category_results) -> Dict[str, List[str]]:
        alerts = {}
        for category, result in category_results.items():
            if result.get('alerts'):
                alerts[category] = result['alerts']
        return alerts
    
    def get_summary_df(self, results: Dict[str, Dict]) -> pd.DataFrame:
        rows = []
        for ticker, data in results.items():
            if not data.get('success', False):
                continue
            
            rows.append({
                'Ticker': ticker,
                'Nombre': (data.get('name') or ticker)[:25],
                'Sector': (data.get('sector') or 'N/A')[:15],
                'Rentabilidad': data['scores']['profitability'],
                'Salud Fin.': data['scores']['financial_health'],
                'Crecimiento': data['scores']['growth'],
                'Eficiencia': data['scores']['efficiency'],
                'Valoración': data['scores']['valuation'],
                'TOTAL': data['scores']['total'],
                'Conclusión': data['conclusion']
            })
        
        return pd.DataFrame(rows)