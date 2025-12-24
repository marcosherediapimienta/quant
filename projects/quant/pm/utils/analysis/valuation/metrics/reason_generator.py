from typing import List, Optional
from ....tools.config import SCORE_INTERPRETATION

class ReasonGenerator:
    """
    Genera razones textuales para señales de inversión.
    
    Responsabilidad: Interpretar scores numéricos y convertirlos en mensajes accionables.
    """

    def __init__(self, thresholds: dict = None):
        """
        Args:
            thresholds: Umbrales personalizados (opcional, usa config por defecto)
        """
        self.thresholds = thresholds or SCORE_INTERPRETATION['reasons']

    def generate(
        self, 
        analysis: dict, 
        fundamental_score: float,
        technical_score: Optional[float] = None
    ) -> List[str]:
        """
        Genera lista de razones basadas en análisis.
        
        Args:
            analysis: Dict con resultados de análisis por categoría
            fundamental_score: Score fundamental agregado
            technical_score: Score técnico (opcional)
            
        Returns:
            Lista de razones (strings con emojis)
        """
        reasons = []
        
        # Razones de valoración
        val_data = analysis.get('valuation', {})
        val_score = val_data.get('score', 50)
        reasons.extend(self._valuation_reasons(val_score))
        
        # Razones fundamentales generales
        reasons.extend(self._fundamental_reasons(fundamental_score))
        
        # Razones de rentabilidad
        prof_data = analysis.get('profitability', {})
        prof_score = prof_data.get('score', 50)
        reasons.extend(self._profitability_reasons(prof_score))
        
        # Razones de salud financiera
        health_data = analysis.get('financial_health', {})
        health_score = health_data.get('score', 50)
        reasons.extend(self._health_reasons(health_score))
        
        # Razones de crecimiento
        growth_data = analysis.get('growth', {})
        growth_score = growth_data.get('score', 50)
        reasons.extend(self._growth_reasons(growth_score))
        
        # Alertas específicas (mostrar primeras 2)
        reasons.extend(self._extract_alerts(analysis))
        
        return reasons if reasons else ["Sin razones específicas"]
    
    def _valuation_reasons(self, score: float) -> List[str]:
        """Genera razones de valoración."""
        reasons = []
        thresh = self.thresholds['valuation']
        
        if score >= thresh['excellent']:
            reasons.append("💰 Valoración atractiva - Precio por debajo del valor justo")
        elif score >= thresh['good']:
            reasons.append("💵 Valoración razonable - Precio cerca del valor justo")
        elif score <= thresh['very_poor']:
            reasons.append("⚠️ Sobrevalorada - Precio significativamente por encima del valor justo")
        elif score <= thresh['poor']:
            reasons.append("⚡ Valoración elevada - Precio por encima del valor justo")
        
        return reasons
    
    def _fundamental_reasons(self, score: float) -> List[str]:
        """Genera razones fundamentales generales."""
        reasons = []
        thresh = self.thresholds['fundamental']
        
        if score >= thresh['exceptional']:
            reasons.append("⭐ Empresa de calidad excepcional")
        elif score >= thresh['good']:
            reasons.append("✅ Sólidos fundamentales")
        elif score <= thresh['weak']:
            reasons.append("⚠️ Fundamentales débiles - Mayor riesgo")
        
        return reasons
    
    def _profitability_reasons(self, score: float) -> List[str]:
        """Genera razones de rentabilidad."""
        reasons = []
        thresh = self.thresholds['profitability']
        
        if score >= thresh['excellent']:
            reasons.append("📈 Excelente rentabilidad")
        elif score <= thresh['poor']:
            reasons.append("📉 Rentabilidad baja o negativa")
        
        return reasons
    
    def _health_reasons(self, score: float) -> List[str]:
        """Genera razones de salud financiera."""
        reasons = []
        thresh = self.thresholds['health']
        
        if score >= thresh['excellent']:
            reasons.append("🛡️ Balance sólido - Baja deuda")
        elif score <= thresh['poor']:
            reasons.append("⚠️ Riesgo financiero - Alta deuda o problemas de liquidez")
        
        return reasons
    
    def _growth_reasons(self, score: float) -> List[str]:
        """Genera razones de crecimiento."""
        reasons = []
        thresh = self.thresholds['growth']
        
        if score >= thresh['high']:
            reasons.append("🚀 Alto crecimiento sostenido")
        elif score <= thresh['low']:
            reasons.append("📊 Crecimiento limitado o decrecimiento")
        
        return reasons
    
    def _extract_alerts(self, analysis: dict) -> List[str]:
        """Extrae alertas de todas las categorías (máximo 2)."""
        alerts = []
        
        for section in ['valuation', 'profitability', 'financial_health', 'growth']:
            section_alerts = analysis.get(section, {}).get('alerts', [])
            alerts.extend(section_alerts)
        
        # Retornar máximo 2 alertas
        return [f"⚠️ {alert}" for alert in alerts[:2]]