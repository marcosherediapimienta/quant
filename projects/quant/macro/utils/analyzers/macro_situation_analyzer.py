import pandas as pd
from typing import Dict
from ..components.macro_situation import (
    analyze_yield_curve_usa,
    analyze_inflation_signals,
    analyze_credit_conditions,
    analyze_global_bonds,
    analyze_risk_sentiment,
    get_current_snapshot
)

class MacroSituationAnalyzer:

    def __init__(self):
        pass
    
    def analyze(self, factors_data: Dict[str, pd.Series]) -> Dict:
        return {
            'yield_curve': analyze_yield_curve_usa(factors_data),
            'inflation': analyze_inflation_signals(factors_data),
            'credit': analyze_credit_conditions(factors_data),
            'global_bonds': analyze_global_bonds(factors_data),
            'risk_sentiment': analyze_risk_sentiment(factors_data),
            'snapshot': get_current_snapshot(factors_data)
        }
    
    def get_summary(self, analysis: Dict) -> Dict:
        summary = {}
        risk_factors = []
        risk_score = 0 

        curve_risk = analysis['yield_curve'].get('risk_level', 'N/A')
        if curve_risk == 'Alto':
            risk_factors.append('Curva invertida')
            risk_score += 3
        elif curve_risk == 'Moderado':
            risk_factors.append('Curva plana')
            risk_score += 1

        divergence = analysis['yield_curve'].get('divergence_analysis', {})
        if '1y' in divergence:
            div_1y = divergence['1y']['divergence']
            if div_1y > 1.0:  # Largo sube mucho más que corto
                risk_factors.append('Expectativas inflacionarias elevadas')
                risk_score += 2
            elif div_1y > 0.5:
                risk_score += 1

        inflation = analysis['inflation'].get('avg_commodity_change', 0)
        if inflation > 15:
            risk_factors.append('Alta inflación')
            risk_score += 3
        elif inflation > 10:
            risk_factors.append('Inflación moderada-alta')
            risk_score += 1

        gold_change = analysis['inflation'].get('gold_change_1y', 0)
        silver_change = analysis['inflation'].get('silver_change_1y', 0)
        if gold_change > 20 or silver_change > 20:
            risk_factors.append('Fuerte presión inflacionaria (metales)')
            risk_score += 2

        vix = analysis['credit'].get('vix_level', 15)
        if vix > 25:
            risk_factors.append('Alta volatilidad')
            risk_score += 3
        elif vix > 20:
            risk_factors.append('Volatilidad elevada')
            risk_score += 1

        bonds = analysis.get('global_bonds', {})
        if 'USA' in bonds:
            usa_bond_change = bonds['USA'].get('change_1y', 0)
            if usa_bond_change < -10:
                risk_factors.append('Caída severa en bonos USA')
                risk_score += 2
            elif usa_bond_change < -5:
                risk_factors.append('Presión en bonos USA')
                risk_score += 1

        sentiment = analysis.get('risk_sentiment', {})
        dollar_strength = sentiment.get('dollar_strength', '')
        if 'debilitándose' in dollar_strength.lower():
            risk_factors.append('Dólar debilitándose')
            risk_score += 1

        summary['risk_factors'] = risk_factors
        summary['risk_score'] = risk_score

        if risk_score >= 5:
            summary['overall_risk'] = "ALTO"
        elif risk_score >= 2:
            summary['overall_risk'] = "MODERADO"
        else:
            summary['overall_risk'] = "BAJO"
        
        return summary