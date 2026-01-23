import pandas as pd
from typing import Dict
from ..components.macro_situation import MacroSituationAnalyzer as MacroSituationCalculator

class MacroSituationAnalyzer:
    """
    Analizador de alto nivel para situación macroeconómica.
    
    Responsabilidad: Orquestar análisis de situación macro y generar resúmenes ejecutivos.
    """

    def __init__(self):
        self.calculator = MacroSituationCalculator()
    
    def analyze(self, factors_data: Dict[str, pd.Series]) -> Dict:
        """
        Análisis completo de situación macroeconómica.
        
        Args:
            factors_data: Dict con series de factores macro
            
        Returns:
            Dict con todos los análisis de situación macro
        """
        # Realizar todos los análisis
        analysis_result = {
            'yield_curve': self.calculator.analyze_yield_curve_usa(factors_data),
            'inflation': self.calculator.analyze_inflation_signals(factors_data),
            'credit': self.calculator.analyze_credit_conditions(factors_data),
            'global_bonds': self.calculator.analyze_global_bonds(factors_data),
            'risk_sentiment': self.calculator.analyze_risk_sentiment(factors_data),
            'snapshot': self.calculator.get_current_snapshot(factors_data)
        }
        
        # Agregar resumen ejecutivo con nivel de riesgo global
        analysis_result['summary'] = self.get_summary(analysis_result)
        
        return analysis_result
    
    def get_summary(self, analysis: Dict) -> Dict:
        """
        Genera resumen ejecutivo del análisis macro.
        
        Args:
            analysis: Dict con resultados de analyze()
            
        Returns:
            Dict con factores de riesgo y nivel de riesgo global
        """
        summary = {}
        risk_factors = []
        risk_score = 0 

        # Análisis de curva de rendimientos
        yield_curve = analysis.get('yield_curve')
        if yield_curve:
            curve_risk = getattr(yield_curve, 'risk_level', 'N/A')
            if curve_risk == 'Alto':
                risk_factors.append('Curva invertida')
                risk_score += 3
            elif curve_risk == 'Moderado':
                risk_factors.append('Curva plana')
                risk_score += 1

            divergence = getattr(yield_curve, 'divergence_analysis', {})
            if '1y' in divergence:
                div_1y = divergence['1y']['divergence']
                if div_1y > 1.0:  # Largo sube mucho más que corto
                    risk_factors.append('Expectativas inflacionarias elevadas')
                    risk_score += 2
                elif div_1y > 0.5:
                    risk_score += 1

        # Análisis de inflación
        inflation = analysis.get('inflation')
        if inflation:
            avg_commodity_change = getattr(inflation, 'avg_commodity_change', 0)
            if avg_commodity_change > 15:
                risk_factors.append('Alta inflación')
                risk_score += 3
            elif avg_commodity_change > 10:
                risk_factors.append('Inflación moderada-alta')
                risk_score += 1

            commodity_changes = getattr(inflation, 'commodity_changes', {})
            gold_change = commodity_changes.get('GOLD', 0)
            silver_change = commodity_changes.get('SILVER', 0)
            if gold_change > 20 or silver_change > 20:
                risk_factors.append('Fuerte presión inflacionaria (metales)')
                risk_score += 2

        # Análisis de crédito y volatilidad
        credit = analysis.get('credit')
        if credit:
            vix = getattr(credit, 'vix_level', 15)
            if vix and vix > 25:
                risk_factors.append('Alta volatilidad')
                risk_score += 3
            elif vix and vix > 20:
                risk_factors.append('Volatilidad elevada')
                risk_score += 1

        # Análisis de bonos globales
        bonds = analysis.get('global_bonds', {})
        if 'USA' in bonds:
            usa_bond_change = bonds['USA'].get('change_1y', 0)
            if usa_bond_change < -10:
                risk_factors.append('Caída severa en bonos USA')
                risk_score += 2
            elif usa_bond_change < -5:
                risk_factors.append('Presión en bonos USA')
                risk_score += 1

        # Sentimiento de riesgo
        sentiment = analysis.get('risk_sentiment')
        if sentiment:
            dollar_strength = getattr(sentiment, 'dollar_strength', '')
            if dollar_strength and 'debilitándose' in dollar_strength.lower():
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