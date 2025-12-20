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

        curve_risk = analysis['yield_curve'].get('risk_level', 'N/A')
        if curve_risk == 'Alto':
            risk_factors.append('Curva invertida')
        
        inflation = analysis['inflation'].get('avg_commodity_change', 0)
        if inflation > 15:
            risk_factors.append('Alta inflación')
 
        vix = analysis['credit'].get('vix_level', 15)
        if vix > 25:
            risk_factors.append('Alta volatilidad')
        
        summary['risk_factors'] = risk_factors
        summary['overall_risk'] = (
            "ALTO" if len(risk_factors) >= 2 else
            "MODERADO" if len(risk_factors) == 1 else
            "BAJO"
        )
        
        return summary