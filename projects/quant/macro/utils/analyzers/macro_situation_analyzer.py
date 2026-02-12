import pandas as pd
from typing import Dict
from ..components.macro_situation import MacroSituationAnalyzer as MacroSituationCalculator

class MacroSituationAnalyzer:
    def __init__(self):
        self.calculator = MacroSituationCalculator()
    
    def analyze(self, factors_data: Dict[str, pd.Series]) -> Dict:
        analysis_result = {
            'yield_curve': self.calculator.analyze_yield_curve_usa(factors_data),
            'inflation': self.calculator.analyze_inflation_signals(factors_data),
            'credit': self.calculator.analyze_credit_conditions(factors_data),
            'global_bonds': self.calculator.analyze_global_bonds(factors_data),
            'risk_sentiment': self.calculator.analyze_risk_sentiment(factors_data),
            'snapshot': self.calculator.get_current_snapshot(factors_data)
        }

        analysis_result['summary'] = self.get_summary(analysis_result)
        
        return analysis_result
    
    def get_summary(self, analysis: Dict) -> Dict:
        risk_factors = []
        risk_score = 0
        risk_score += self._analyze_yield_curve(analysis.get('yield_curve'), risk_factors)
        risk_score += self._analyze_inflation(analysis.get('inflation'), risk_factors)
        risk_score += self._analyze_credit(analysis.get('credit'), risk_factors)
        risk_score += self._analyze_bonds(analysis.get('global_bonds', {}), risk_factors)
        risk_score += self._analyze_sentiment(analysis.get('risk_sentiment'), risk_factors)
        
        return {
            'risk_factors': risk_factors,
            'risk_score': risk_score,
            'overall_risk': self._calculate_overall_risk(risk_score)
        }

    def _analyze_yield_curve(self, yield_curve, risk_factors: list) -> int:

        if not yield_curve:
            return 0
        
        score = 0
        curve_risk = getattr(yield_curve, 'risk_level', 'N/A')
        
        if curve_risk == 'Alto':
            risk_factors.append('Curva invertida')
            score += 3
        elif curve_risk == 'Moderado':
            risk_factors.append('Curva plana')
            score += 1

        divergence = getattr(yield_curve, 'divergence_analysis', {})
        if '1y' in divergence:
            div_1y = divergence['1y']['divergence']
            if div_1y > 1.0:
                risk_factors.append('Expectativas inflacionarias elevadas')
                score += 2
            elif div_1y > 0.5:
                score += 1
        
        return score

    def _analyze_inflation(self, inflation, risk_factors: list) -> int:

        if not inflation:
            return 0
        
        score = 0
        avg_commodity_change = getattr(inflation, 'avg_commodity_change', 0)
        
        if avg_commodity_change > 15:
            risk_factors.append('Alta inflación')
            score += 3
        elif avg_commodity_change > 10:
            risk_factors.append('Inflación moderada-alta')
            score += 1
        
        # Analizar metales preciosos
        commodity_changes = getattr(inflation, 'commodity_changes', {})
        gold_change = commodity_changes.get('GOLD', 0)
        silver_change = commodity_changes.get('SILVER', 0)
        
        if gold_change > 20 or silver_change > 20:
            risk_factors.append('Fuerte presión inflacionaria (metales)')
            score += 2
        
        return score

    def _analyze_credit(self, credit, risk_factors: list) -> int:

        if not credit:
            return 0
        
        score = 0
        vix = getattr(credit, 'vix_level', 15)
        
        if vix and vix > 25:
            risk_factors.append('Alta volatilidad')
            score += 3
        elif vix and vix > 20:
            risk_factors.append('Volatilidad elevada')
            score += 1
        
        return score

    def _analyze_bonds(self, bonds: dict, risk_factors: list) -> int:

        if 'USA' not in bonds:
            return 0
        
        score = 0
        usa_bond_change = bonds['USA'].get('change_1y', 0)
        
        if usa_bond_change < -10:
            risk_factors.append('Caída severa en bonos USA')
            score += 2
        elif usa_bond_change < -5:
            risk_factors.append('Presión en bonos USA')
            score += 1
        
        return score

    def _analyze_sentiment(self, sentiment, risk_factors: list) -> int:

        if not sentiment:
            return 0
        
        score = 0
        dollar_strength = getattr(sentiment, 'dollar_strength', '')
        
        if dollar_strength and 'debilitándose' in dollar_strength.lower():
            risk_factors.append('Dólar debilitándose')
            score += 1
        
        return score

    def _calculate_overall_risk(self, risk_score: int) -> str:

        if risk_score >= 5:
            return "ALTO"
        elif risk_score >= 2:
            return "MODERADO"
        else:
            return "BAJO"