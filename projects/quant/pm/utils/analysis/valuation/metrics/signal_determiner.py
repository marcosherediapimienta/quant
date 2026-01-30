from typing import Tuple
from ....tools.config import TRADING_SIGNAL_RULES

class SignalDeterminer:
    """
    Determines trading signals (BUY/SELL/HOLD) based on scores.
    
    Responsabilidad: Aplicar reglas cuantitativas para generar señales de inversión.
    
    Metodología:
    - Evalúa combinaciones de valuation_score y fundamental_score
    - Aplica reglas jerárquicas (prioridad: strong > moderate > specific)
    - Calcula confianza ponderada según calidad de métricas
    """
    
    def __init__(self, rules: dict = None):
        """
        Args:
            rules: Diccionario de reglas de señales. Por defecto usa TRADING_SIGNAL_RULES
        """
        self.rules = rules or TRADING_SIGNAL_RULES

    def determine(
        self, 
        valuation_score: float,
        fundamental_score: float
    ) -> Tuple[str, float]:
        """
        Determina señal de trading y nivel de confianza.
        
        Args:
            valuation_score: Score de valoración (0-100, mayor = más barato)
            fundamental_score: Score fundamental (0-100, mayor = mejor calidad)
            
        Returns:
            Tupla (señal, confianza) donde:
            - señal: "BUY", "SELL", "HOLD"
            - confianza: nivel de confianza 0-100
            
        Lógica:
        1. Evalúa reglas de BUY (de más estrictas a menos)
        2. Evalúa reglas de SELL
        3. Evalúa reglas de HOLD
        4. Default: HOLD con confianza calculada dinámicamente
        """
        # ========== REGLAS DE COMPRA ==========
        
        # COMPRA FUERTE: Alta calidad + Buen precio
        buy_strong = self.rules['buy']['strong']
        if (valuation_score >= buy_strong['valuation_min'] and 
            fundamental_score >= buy_strong['fundamental_min']):
            confidence = self._calculate_confidence(
                base=buy_strong['confidence_base'],
                max_conf=buy_strong['confidence_max'],
                valuation_score=valuation_score,
                fundamental_score=fundamental_score,
                val_min=buy_strong['valuation_min'],
                fund_min=buy_strong['fundamental_min'],
                val_weight=buy_strong['valuation_weight'],
                fund_weight=buy_strong['fundamental_weight']
            )
            return "BUY", confidence

        # COMPRA MODERADA: Buena calidad + Buen precio
        buy_mod = self.rules['buy']['moderate']
        if (valuation_score >= buy_mod['valuation_min'] and 
            fundamental_score >= buy_mod['fundamental_min']):
            confidence = self._calculate_confidence(
                base=buy_mod['confidence_base'],
                max_conf=buy_mod['confidence_max'],
                valuation_score=valuation_score,
                fundamental_score=fundamental_score,
                val_min=buy_mod['valuation_min'],
                fund_min=buy_mod['fundamental_min'],
                val_weight=buy_mod['valuation_weight'],
                fund_weight=buy_mod['fundamental_weight']
            )
            return "BUY", confidence

        # COMPRA POR VALOR: Precio muy atractivo
        buy_value = self.rules['buy']['value']
        if (valuation_score >= buy_value['valuation_min'] and 
            fundamental_score >= buy_value['fundamental_min']):
            confidence = self._calculate_confidence(
                base=buy_value['confidence_base'],
                max_conf=buy_value['confidence_max'],
                valuation_score=valuation_score,
                val_min=buy_value['valuation_min'],
                val_weight=buy_value['valuation_weight']
            )
            return "BUY", confidence

        # COMPRA POR CALIDAD: Empresa excelente (aunque algo cara)
        buy_quality = self.rules['buy']['quality']
        if (valuation_score >= buy_quality['valuation_min'] and 
            fundamental_score >= buy_quality['fundamental_min']):
            confidence = self._calculate_confidence(
                base=buy_quality['confidence_base'],
                max_conf=buy_quality['confidence_max'],
                fundamental_score=fundamental_score,
                fund_min=buy_quality['fundamental_min'],
                fund_weight=buy_quality['fundamental_weight']
            )
            return "BUY", confidence

        # ========== REGLAS DE VENTA ==========
        
        # VENTA FUERTE: Mala calidad + Sobrevalorado
        sell_strong = self.rules['sell']['strong']
        if (valuation_score <= sell_strong['valuation_max'] and 
            fundamental_score <= sell_strong['fundamental_max']):
            confidence = self._calculate_confidence(
                base=sell_strong['confidence_base'],
                max_conf=sell_strong['confidence_max'],
                valuation_score=valuation_score,
                fundamental_score=fundamental_score,
                val_max=sell_strong['valuation_max'],
                fund_max=sell_strong['fundamental_max'],
                val_weight=sell_strong['valuation_weight'],
                fund_weight=sell_strong['fundamental_weight'],
                is_sell=True
            )
            return "SELL", confidence

        # VENTA MODERADA: Calidad regular + Caro
        sell_mod = self.rules['sell']['moderate']
        if (valuation_score <= sell_mod['valuation_max'] and 
            fundamental_score <= sell_mod['fundamental_max']):
            confidence = self._calculate_confidence(
                base=sell_mod['confidence_base'],
                max_conf=sell_mod['confidence_max'],
                valuation_score=valuation_score,
                fundamental_score=fundamental_score,
                val_max=sell_mod['valuation_max'],
                fund_max=sell_mod['fundamental_max'],
                val_weight=sell_mod['valuation_weight'],
                fund_weight=sell_mod['fundamental_weight'],
                is_sell=True
            )
            return "SELL", confidence

        # VENTA POR SOBREVALORACIÓN: Extremadamente caro
        sell_overval = self.rules['sell']['overvalued']
        if valuation_score <= sell_overval['valuation_max']:
            confidence = self._calculate_confidence(
                base=sell_overval['confidence_base'],
                max_conf=sell_overval['confidence_max'],
                valuation_score=valuation_score,
                val_max=sell_overval['valuation_max'],
                val_weight=sell_overval['valuation_weight'],
                is_sell=True
            )
            return "SELL", confidence

        # ========== REGLAS DE HOLD ==========
        
        # HOLD: Excelente empresa pero cara
        hold_qual = self.rules['hold']['mixed_quality_price']
        if (valuation_score <= hold_qual['valuation_max'] and 
            fundamental_score >= hold_qual['fundamental_min']):
            # Calcular confianza dinámica basada en scores
            confidence = self._calculate_hold_confidence(
                valuation_score, fundamental_score,
                hold_qual['valuation_max'], hold_qual['fundamental_min']
            )
            return "HOLD", confidence

        # HOLD: Empresa buena pero algo cara
        hold_mod = self.rules['hold']['mixed_moderate']
        if (valuation_score <= hold_mod['valuation_max'] and 
            fundamental_score >= hold_mod['fundamental_min']):
            # Calcular confianza dinámica basada en scores
            confidence = self._calculate_hold_confidence(
                valuation_score, fundamental_score,
                hold_mod['valuation_max'], hold_mod['fundamental_min']
            )
            return "HOLD", confidence
        
        # DEFAULT: HOLD con confianza basada en scores
        default_conf = self._calculate_hold_confidence(
            valuation_score, fundamental_score, 50, 50
        )
        return "HOLD", default_conf
    
    def _calculate_confidence(
        self,
        base: float,
        max_conf: float,
        valuation_score: float = None,
        fundamental_score: float = None,
        val_min: float = None,
        val_max: float = None,
        fund_min: float = None,
        fund_max: float = None,
        val_weight: float = 0,
        fund_weight: float = 0,
        is_sell: bool = False
    ) -> float:
        """
        Calcula confianza ponderada basada en cuánto superan los umbrales.
        
        Args:
            base: Confianza base
            max_conf: Confianza máxima
            valuation_score: Score de valoración
            fundamental_score: Score fundamental
            val_min/val_max: Umbrales de valoración
            fund_min/fund_max: Umbrales fundamental
            val_weight: Peso para ajuste de valoración
            fund_weight: Peso para ajuste fundamental
            is_sell: Si es señal de venta (invierte lógica)
            
        Returns:
            Confianza ajustada (base a max_conf)
        """
        confidence = base
        
        # Ajuste por valoración
        if valuation_score is not None and val_weight > 0:
            if is_sell and val_max is not None:
                # Para venta: más confianza cuanto más bajo el score (más sobrevalorado)
                excess = val_max - valuation_score
                confidence += excess * val_weight
            elif not is_sell and val_min is not None:
                # Para compra: más confianza cuanto más alto el score (más infravalorado)
                excess = valuation_score - val_min
                confidence += excess * val_weight
        
        # Ajuste por fundamental
        if fundamental_score is not None and fund_weight > 0:
            if is_sell and fund_max is not None:
                # Para venta: más confianza cuanto peor la calidad
                excess = fund_max - fundamental_score
                confidence += excess * fund_weight
            elif not is_sell and fund_min is not None:
                # Para compra: más confianza cuanto mejor la calidad
                excess = fundamental_score - fund_min
                confidence += excess * fund_weight
        
        # Limitar entre base y max_conf
        return min(max_conf, max(base, confidence))
    
    def _calculate_hold_confidence(
        self,
        valuation_score: float,
        fundamental_score: float,
        val_max: float,
        fund_min: float
    ) -> float:
        """
        Calcula confianza para señales HOLD basada en scores.
        
        La confianza varía según qué tan cerca estén los scores de los umbrales:
        - Si está muy cerca de ser BUY o SELL → menor confianza (40-50%)
        - Si está claramente en zona neutral → mayor confianza (50-65%)
        - Si tiene buen fundamental pero mala valoración → confianza media (50-60%)
        """
        # Confianza base
        base_confidence = 50.0
        
        # Calcular distancia a umbrales
        val_distance = abs(valuation_score - val_max) if val_max else 0
        fund_distance = abs(fundamental_score - fund_min) if fund_min else 0
        
        # Si está muy cerca de umbrales (zona ambigua), menor confianza
        if val_distance < 5 or fund_distance < 5:
            return max(40.0, base_confidence - 10)
        
        # Si tiene buen fundamental pero mala valoración (típico HOLD)
        if fundamental_score >= 70 and valuation_score <= 45:
            # Más confianza si el fundamental es muy bueno
            if fundamental_score >= 80:
                return 60.0
            return 55.0
        
        # Si está en zona neutral (ni muy bueno ni muy malo)
        if 45 <= valuation_score <= 55 and 60 <= fundamental_score <= 75:
            return 55.0
        
        # Default: confianza base ajustada por calidad general
        avg_score = (valuation_score + fundamental_score) / 2
        if avg_score >= 65:
            return 55.0
        elif avg_score >= 55:
            return 52.0
        else:
            return 48.0