import pandas as pd
from typing import Dict
from ..components.macro_situation import MacroSituationAnalyzer as MacroSituationCalculator
from ..components.implied_yield_curve import ImpliedYieldCurveCalculator
from ..tools.config import RISK_SCORING

class MacroSituationAnalyzer:
    def __init__(self):
        self.calculator = MacroSituationCalculator()
        self.implied_curve_calc = ImpliedYieldCurveCalculator()
    
    def analyze(self, factors_data: Dict[str, pd.Series]) -> Dict:
        analysis_result = {
            'yield_curve': self.calculator.analyze_yield_curve_usa(factors_data),
            'implied_yield_curve': self.implied_curve_calc.analyze(factors_data),
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
        risk_score += self._analyze_implied_curve(analysis.get('implied_yield_curve'), risk_factors)
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
        
        cfg = RISK_SCORING['yield_curve']
        score = 0
        curve_risk = getattr(yield_curve, 'risk_level', 'N/A')
        
        if curve_risk == 'High':
            risk_factors.append('Inverted yield curve')
            score += cfg['high_risk_score']
        elif curve_risk == 'Moderate':
            risk_factors.append('Flat yield curve')
            score += cfg['moderate_risk_score']

        divergence = getattr(yield_curve, 'divergence_analysis', {})
        if '1y' in divergence:
            div_1y = divergence['1y']['divergence']
            if div_1y > cfg['divergence_elevated']:
                risk_factors.append('Elevated inflation expectations')
                score += cfg['divergence_elevated_score']
            elif div_1y > cfg['divergence_moderate']:
                score += cfg['divergence_moderate_score']
        
        return score

    def _analyze_implied_curve(self, implied_curve, risk_factors: list) -> int:
        if not implied_curve:
            return 0
        
        cfg = RISK_SCORING['implied_curve']
        score = 0

        signal = getattr(implied_curve, 'rate_path_signal', '')
        if signal == "HAWKISH":
            risk_factors.append('Forward curve signals rate hikes')
            score += cfg['hawkish_score']
        elif signal == "SLIGHTLY HAWKISH":
            risk_factors.append('Forward curve signals moderate hikes')
            score += cfg['slightly_hawkish_score']

        term_premium = getattr(implied_curve, 'term_premium', {})
        if '10Y' in term_premium:
            tp_10y = term_premium['10Y']
            if tp_10y < cfg['term_premium_deep_negative']:
                risk_factors.append(f'10Y term premium deeply negative ({tp_10y:+.2f}pp)')
                score += cfg['term_premium_deep_score']
            elif tp_10y < cfg['term_premium_negative']:
                risk_factors.append(f'10Y term premium negative ({tp_10y:+.2f}pp)')
                score += cfg['term_premium_negative_score']

        fwd_vs_spot = getattr(implied_curve, 'forward_vs_spot', {})
        if fwd_vs_spot:
            diffs = [v for v in fwd_vs_spot.values() if v is not None]
            if diffs:
                avg_diff = sum(diffs) / len(diffs)
                if avg_diff > cfg['fwd_vs_spot_high']:
                    risk_factors.append(f'Forwards well above spot ({avg_diff:+.2f}pp)')
                    score += cfg['fwd_vs_spot_high_score']
                elif avg_diff > cfg['fwd_vs_spot_moderate']:
                    risk_factors.append(f'Forwards elevated vs spot ({avg_diff:+.2f}pp)')
                    score += cfg['fwd_vs_spot_moderate_score']

        forwards = getattr(implied_curve, 'forward_rates', {})
        spot = getattr(implied_curve, 'spot_rates', {})
        if '2Y→5Y' in forwards and '2Y' in spot:
            if forwards['2Y→5Y'] < spot['2Y']:
                risk_factors.append('Inverted forward curve (2Y→5Y < spot 2Y)')
                score += cfg['inverted_fwd_score']
        
        return score

    def _analyze_inflation(self, inflation, risk_factors: list) -> int:
        if not inflation:
            return 0
        
        cfg = RISK_SCORING['inflation']
        score = 0
        avg_commodity_change = getattr(inflation, 'avg_commodity_change', 0)
        
        if avg_commodity_change > cfg['high_threshold']:
            risk_factors.append('High inflation')
            score += cfg['high_score']
        elif avg_commodity_change > cfg['moderate_threshold']:
            risk_factors.append('Moderate-high inflation')
            score += cfg['moderate_score']
        
        commodity_changes = getattr(inflation, 'commodity_changes', {})
        gold_change = commodity_changes.get('GOLD', 0)
        silver_change = commodity_changes.get('SILVER', 0)
        
        if gold_change > cfg['metals_threshold'] or silver_change > cfg['metals_threshold']:
            risk_factors.append('Strong inflationary pressure (metals)')
            score += cfg['metals_score']
        
        return score

    def _analyze_credit(self, credit, risk_factors: list) -> int:
        if not credit:
            return 0
        
        cfg = RISK_SCORING['credit']
        score = 0
        vix = getattr(credit, 'vix_level', cfg['vix_default'])
        
        if vix and vix > cfg['vix_high']:
            risk_factors.append('High volatility')
            score += cfg['vix_high_score']
        elif vix and vix > cfg['vix_elevated']:
            risk_factors.append('Elevated volatility')
            score += cfg['vix_elevated_score']
        
        return score

    def _analyze_bonds(self, bonds: dict, risk_factors: list) -> int:
        if 'USA' not in bonds:
            return 0
        
        cfg = RISK_SCORING['bonds']
        score = 0
        usa_bond_change = bonds['USA'].get('change_1y', 0)
        
        if usa_bond_change < cfg['severe_threshold']:
            risk_factors.append('Severe US bond sell-off')
            score += cfg['severe_score']
        elif usa_bond_change < cfg['pressure_threshold']:
            risk_factors.append('US bond pressure')
            score += cfg['pressure_score']
        
        return score

    def _analyze_sentiment(self, sentiment, risk_factors: list) -> int:
        if not sentiment:
            return 0
        
        score = 0
        dollar_strength = getattr(sentiment, 'dollar_strength', '')
        
        if dollar_strength and 'weakening' in dollar_strength.lower():
            risk_factors.append('Dollar weakening')
            score += RISK_SCORING['sentiment']['weakening_score']
        
        return score

    def _calculate_overall_risk(self, risk_score: int) -> str:
        cfg = RISK_SCORING['overall']
        if risk_score >= cfg['high']:
            return "HIGH"
        return "MODERATE" if risk_score >= cfg['moderate'] else "LOW"
