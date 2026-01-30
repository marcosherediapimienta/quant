import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass
import warnings
# Suprimir warnings de Pandas4Warning de yfinance
warnings.filterwarnings('ignore', category=pd.errors.Pandas4Warning, module='yfinance')
import yfinance as yf

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
    """Pesos para combinar scores de diferentes categorías."""
    profitability: float = None
    financial_health: float = None
    growth: float = None
    efficiency: float = None
    valuation: float = None
    
    def __post_init__(self):
        # Cargar defaults desde config si no se proporcionan
        defaults = COMPANY_ANALYSIS_WEIGHTS['default']
        self.profitability = self.profitability if self.profitability is not None else defaults['profitability']
        self.financial_health = self.financial_health if self.financial_health is not None else defaults['financial_health']
        self.growth = self.growth if self.growth is not None else defaults['growth']
        self.efficiency = self.efficiency if self.efficiency is not None else defaults['efficiency']
        self.valuation = self.valuation if self.valuation is not None else defaults['valuation']
        
        # Normalizar si no suman 1.0
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
    """Umbrales para clasificación de scores."""
    excellent: float = None
    good: float = None
    fair: float = None
    weak: float = None
    labels: Dict[str, str] = None
    
    def __post_init__(self):
        # Cargar desde config si no se proporciona
        cfg = CONCLUSION_THRESHOLDS
        self.excellent = self.excellent if self.excellent is not None else cfg['excellent']
        self.good = self.good if self.good is not None else cfg['good']
        self.fair = self.fair if self.fair is not None else cfg['fair']
        self.weak = self.weak if self.weak is not None else cfg['weak']
        self.labels = self.labels or cfg['labels'].copy()

class CompanyAnalyzer:
    """
    Analizador fundamental de empresas.
    
    Responsabilidad: Coordinar análisis completo de fundamentales de una empresa.
    """

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
        """
        Obtiene datos fundamentales de la empresa usando yfinance.
        Incluye info + estados financieros + ROIC calculado.
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info or len(info) == 0:
                return {
                    'success': False,
                    'error': f'No se pudieron obtener datos para {ticker}'
                }
            
            # Obtener estados financieros adicionales
            try:
                balance_sheet = stock.quarterly_balance_sheet
                income_stmt = stock.quarterly_income_stmt
                
                # ========== BALANCE SHEET ==========
                if not balance_sheet.empty:
                    latest_bs = balance_sheet.columns[0]
                    
                    # Total Assets (para Asset Turnover)
                    if 'Total Assets' in balance_sheet.index:
                        info['totalAssets'] = float(balance_sheet.loc['Total Assets', latest_bs])
                    
                    # Inventory (para DIO)
                    if 'Inventory' in balance_sheet.index:
                        info['inventory'] = float(balance_sheet.loc['Inventory', latest_bs])
                    
                    # Receivables (para DSO)
                    if 'Receivables' in balance_sheet.index:
                        info['netReceivables'] = float(balance_sheet.loc['Receivables', latest_bs])
                    
                    # Stockholders Equity (para ROIC alternativo)
                    if 'Stockholders Equity' in balance_sheet.index:
                        info['totalStockholderEquity'] = float(balance_sheet.loc['Stockholders Equity', latest_bs])
                
                # ========== INCOME STATEMENT ==========
                if not income_stmt.empty:
                    latest_is = income_stmt.columns[0]
                    
                    # Cost of Revenue (para COGS y márgenes)
                    if 'Cost Of Revenue' in income_stmt.index:
                        info['costOfRevenue'] = float(income_stmt.loc['Cost Of Revenue', latest_is])
                
                # ========== CALCULAR ROIC ==========
                if not income_stmt.empty and not balance_sheet.empty:
                    latest_is = income_stmt.columns[0]
                    latest_bs = balance_sheet.columns[0]
                    
                    # 1. Obtener Operating Income (EBIT)
                    operating_income = None
                    if 'Operating Income' in income_stmt.index:
                        operating_income = float(income_stmt.loc['Operating Income', latest_is])
                    elif 'EBIT' in income_stmt.index:
                        operating_income = float(income_stmt.loc['EBIT', latest_is])
                    
                    # 2. Calcular Tax Rate efectivo
                    tax_rate = None
                    if 'Tax Provision' in income_stmt.index and 'Pretax Income' in income_stmt.index:
                        tax_provision = float(income_stmt.loc['Tax Provision', latest_is])
                        pretax_income = float(income_stmt.loc['Pretax Income', latest_is])
                        if pretax_income != 0:
                            tax_rate = tax_provision / pretax_income
                    
                    if tax_rate is None:
                        tax_rate = info.get('effectiveTaxRate', 0.21)  # Default 21%
                    
                    # 3. Calcular NOPAT
                    if operating_income is not None and tax_rate is not None:
                        nopat = operating_income * (1 - tax_rate)
                        
                        # 4. Obtener Invested Capital (preferido si existe)
                        invested_capital = None
                        if 'Invested Capital' in balance_sheet.index:
                            invested_capital = float(balance_sheet.loc['Invested Capital', latest_bs])
                        else:
                            # Alternativa: Total Assets - Current Liabilities
                            if 'Total Assets' in balance_sheet.index and 'Current Liabilities' in balance_sheet.index:
                                total_assets = float(balance_sheet.loc['Total Assets', latest_bs])
                                current_liabilities = float(balance_sheet.loc['Current Liabilities', latest_bs])
                                invested_capital = total_assets - current_liabilities
                        
                        # 5. Calcular ROIC = NOPAT / Invested Capital
                        if invested_capital and invested_capital > 0:
                            roic = nopat / invested_capital
                            info['returnOnCapital'] = roic
                        
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
            
    def analyze(self, ticker: str, data: Dict = None) -> Dict:
        """
        Analiza fundamentalmente una empresa.
        
        Args:
            ticker: Símbolo del ticker
            data: Datos de la empresa (si None, los obtiene automáticamente)
            
        Returns:
            Dict con análisis completo
        """
        # Obtener datos si no se proporcionan
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
            # Calcular métricas por categoría
            profitability_result = self.profitability.calculate(data)
            health_result = self.health.calculate(data)
            growth_result = self.growth.calculate(data)
            efficiency_result = self.efficiency.calculate(data)
            valuation_result = self.valuation.calculate(data)
            
            # Extraer scores
            scores = {
                'profitability': profitability_result.get('score', 0),
                'financial_health': health_result.get('score', 0),
                'growth': growth_result.get('score', 0),
                'efficiency': efficiency_result.get('score', 0),
                'valuation': valuation_result.get('score', 0)
            }
            
            # Calcular score total ponderado ignorando NaN
            total_score = 0
            total_weight = 0
            
            for category in ['profitability', 'financial_health', 'growth', 'efficiency', 'valuation']:
                score = scores[category]
                if pd.notna(score):  # Solo incluir si no es NaN
                    weight = getattr(self.weights, category)
                    total_score += score * weight
                    total_weight += weight
            
            # Normalizar por el peso total usado
            scores['total'] = total_score / total_weight if total_weight > 0 else np.nan
            
            # Determinar conclusión
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
        """
        Análisis rápido que solo calcula el score total sin detalles completos.
        Útil para filtrar empresas antes del análisis completo.
        
        Args:
            ticker: Símbolo del ticker
            
        Returns:
            Dict con success, ticker, scores.total, y datos básicos
        """
        fetch_result = self.fetch_data(ticker)
        if not fetch_result.get('success'):
            return {
                'success': False,
                'error': fetch_result.get('error'),
                'ticker': ticker
            }
        
        data = fetch_result['data']
        
        try:
            # Calcular solo los scores sin todos los detalles
            profitability_score = self.profitability.calculate(data).get('score', 0)
            health_score = self.health.calculate(data).get('score', 0)
            growth_score = self.growth.calculate(data).get('score', 0)
            efficiency_score = self.efficiency.calculate(data).get('score', 0)
            valuation_score = self.valuation.calculate(data).get('score', 0)
            
            scores = {
                'profitability': profitability_score,
                'financial_health': health_score,
                'growth': growth_score,
                'efficiency': efficiency_score,
                'valuation': valuation_score
            }
            
            # Calcular score total
            total_score = 0
            total_weight = 0
            
            for category in ['profitability', 'financial_health', 'growth', 'efficiency', 'valuation']:
                score = scores[category]
                if pd.notna(score):
                    weight = getattr(self.weights, category)
                    total_score += score * weight
                    total_weight += weight
            
            scores['total'] = total_score / total_weight if total_weight > 0 else np.nan
            
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
        """
        Analiza múltiples empresas.
        
        Args:
            tickers: Lista de tickers
            
        Returns:
            Dict con ticker como key y resultado del análisis como value
        """
        results = {}
        for ticker in tickers:
            results[ticker] = self.analyze(ticker)
        return results
    
    def _determine_conclusion(self, score: float) -> Dict[str, str]:
        """Determina la conclusión basada en el score."""
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
        """
        Crea un DataFrame resumen de múltiples análisis.
        
        Args:
            results: Dict con resultados de análisis (ticker -> resultado)
            
        Returns:
            DataFrame con resumen
        """
        import pandas as pd
        
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