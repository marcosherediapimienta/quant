from typing import List, Optional

class ReasonGenerator:

    def generate(
        self, 
        analysis: dict, 
        fundamental_score: float,
        technical_score: Optional[float] = None
    ) -> List[str]:

        reasons = []
        val_data = analysis.get('valuation', {})
        val_score = val_data.get('score', 50)
        
        if val_score >= 70:
            reasons.append("💰 Valoración atractiva - Precio por debajo del valor justo")
        elif val_score >= 55:
            reasons.append("💵 Valoración razonable - Precio cerca del valor justo")
        elif val_score <= 30:
            reasons.append("⚠️ Sobrevalorada - Precio significativamente por encima del valor justo")
        elif val_score <= 45:
            reasons.append("⚡ Valoración elevada - Precio por encima del valor justo")

        if fundamental_score >= 80:
            reasons.append("⭐ Empresa de calidad excepcional")
        elif fundamental_score >= 70:
            reasons.append("✅ Sólidos fundamentales")
        elif fundamental_score <= 50:
            reasons.append("⚠️ Fundamentales débiles - Mayor riesgo")

        prof_data = analysis.get('profitability', {})
        prof_score = prof_data.get('score', 50)
        
        if prof_score >= 80:
            reasons.append("📈 Excelente rentabilidad")
        elif prof_score <= 40:
            reasons.append("📉 Rentabilidad baja o negativa")
        
        health_data = analysis.get('financial_health', {})
        health_score = health_data.get('score', 50)
        
        if health_score >= 80:
            reasons.append("🛡️ Balance sólido - Baja deuda")
        elif health_score <= 40:
            reasons.append("⚠️ Riesgo financiero - Alta deuda o problemas de liquidez")
        
        growth_data = analysis.get('growth', {})
        growth_score = growth_data.get('score', 50)
        
        if growth_score >= 75:
            reasons.append("🚀 Alto crecimiento sostenido")
        elif growth_score <= 35:
            reasons.append("📊 Crecimiento limitado o decrecimiento")

        alerts = []
        for section in ['valuation', 'profitability', 'financial_health', 'growth']:
            section_alerts = analysis.get(section, {}).get('alerts', [])
            alerts.extend(section_alerts)
        
        if alerts:
            for alert in alerts[:2]:
                reasons.append(f"⚠️ {alert}")
        
        return reasons if reasons else ["Sin razones específicas"]