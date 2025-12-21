from typing import Tuple

class SignalDeterminer:

    def determine(
        self, 
        valuation_score: float,
        fundamental_score: float
    ) -> Tuple[str, float]:
  
        if valuation_score >= 60 and fundamental_score >= 80:
            confidence = min(95, 75 + (valuation_score - 60) * 0.4 + (fundamental_score - 80) * 0.3)
            return "COMPRA", confidence

        if valuation_score >= 60 and fundamental_score >= 70:
            confidence = min(85, 65 + (valuation_score - 60) * 0.4 + (fundamental_score - 70) * 0.2)
            return "COMPRA", confidence

        if valuation_score >= 60 and fundamental_score >= 60:
            confidence = min(75, 55 + (valuation_score - 60) * 0.4)
            return "COMPRA", confidence

        if valuation_score >= 45 and fundamental_score >= 85:
            confidence = min(75, 60 + (fundamental_score - 85) * 0.5)
            return "COMPRA", confidence

        if valuation_score <= 40 and fundamental_score <= 65:
            confidence = min(85, 55 + (40 - valuation_score) * 0.8 + (65 - fundamental_score) * 0.4)
            return "VENTA", confidence

        if valuation_score <= 20 and fundamental_score <= 60:
            confidence = min(95, 70 + (20 - valuation_score) * 0.9 + (60 - fundamental_score) * 0.5)
            return "VENTA", confidence

        if valuation_score <= 15:
            confidence = min(75, 60 + (15 - valuation_score) * 1.0)
            return "VENTA", confidence

        if valuation_score <= 40 and fundamental_score >= 85:
            confidence = 50
            return "MANTENER", confidence

        if valuation_score <= 45 and fundamental_score >= 75:
            confidence = 50
            return "MANTENER", confidence
        
        confidence = 50
        return "MANTENER", confidence