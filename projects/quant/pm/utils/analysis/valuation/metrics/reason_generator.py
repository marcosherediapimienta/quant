from typing import List, Optional
from ....tools.config import SCORE_INTERPRETATION

class ReasonGenerator:
    def __init__(self, thresholds: dict = None):
        self.thresholds = thresholds or SCORE_INTERPRETATION['reasons']

    def generate(
        self, 
        analysis: dict, 
        fundamental_score: float,
        technical_score: Optional[float] = None
    ) -> List[str]:

        reasons = []
        val_data = analysis.get('valuation', {})
        val_score = val_data.get('score', 50)
        reasons.extend(self._valuation_reasons(val_score))
        reasons.extend(self._fundamental_reasons(fundamental_score))
        prof_data = analysis.get('profitability', {})
        prof_score = prof_data.get('score', 50)
        reasons.extend(self._profitability_reasons(prof_score))
        health_data = analysis.get('financial_health', {})
        health_score = health_data.get('score', 50)
        reasons.extend(self._health_reasons(health_score))
        growth_data = analysis.get('growth', {})
        growth_score = growth_data.get('score', 50)
        reasons.extend(self._growth_reasons(growth_score))
        reasons.extend(self._extract_alerts(analysis))
        
        return reasons if reasons else ["No specific reasons"]
    
    def _valuation_reasons(self, score: float) -> List[str]:
        reasons = []
        thresh = self.thresholds['valuation']
        
        if score >= thresh['excellent']:
            reasons.append("💰 Attractive valuation - Price below fair value")
        elif score >= thresh['good']:
            reasons.append("💵 Reasonable valuation - Price near fair value")
        elif score <= thresh['very_poor']:
            reasons.append("⚠️ Overvalued - Price significantly above fair value")
        elif score <= thresh['poor']:
            reasons.append("⚡ High valuation - Price above fair value")
        
        return reasons
    
    def _fundamental_reasons(self, score: float) -> List[str]:
        reasons = []
        thresh = self.thresholds['fundamental']
        
        if score >= thresh['exceptional']:
            reasons.append("⭐ Exceptional quality company")
        elif score >= thresh['good']:
            reasons.append("✅ Solid fundamentals")
        elif score <= thresh['weak']:
            reasons.append("⚠️ Weak fundamentals - Higher risk")
        
        return reasons
    
    def _profitability_reasons(self, score: float) -> List[str]:
        reasons = []
        thresh = self.thresholds['profitability']
        
        if score >= thresh['excellent']:
            reasons.append("📈 Excellent profitability")
        elif score <= thresh['poor']:
            reasons.append("📉 Low or negative profitability")
        
        return reasons
    
    def _health_reasons(self, score: float) -> List[str]:
        reasons = []
        thresh = self.thresholds['health']
        
        if score >= thresh['excellent']:
            reasons.append("🛡️ Solid balance sheet - Low debt")
        elif score <= thresh['poor']:
            reasons.append("⚠️ Financial risk - High debt or liquidity issues")
        
        return reasons
    
    def _growth_reasons(self, score: float) -> List[str]:
        reasons = []
        thresh = self.thresholds['growth']
        
        if score >= thresh['high']:
            reasons.append("🚀 High sustained growth")
        elif score <= thresh['low']:
            reasons.append("📊 Limited growth or decline")
        
        return reasons
    
    def _extract_alerts(self, analysis: dict) -> List[str]:
        alerts = []
        
        for section in ['valuation', 'profitability', 'financial_health', 'growth']:
            section_alerts = analysis.get(section, {}).get('alerts', [])
            alerts.extend(section_alerts)

        return [f"⚠️ {alert}" for alert in alerts[:2]]