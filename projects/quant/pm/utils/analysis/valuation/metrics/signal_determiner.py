from typing import Tuple
from ....tools.config import TRADING_SIGNAL_RULES

class SignalDeterminer:
    def __init__(self, rules: dict = None):
        self.rules = rules or TRADING_SIGNAL_RULES

    def determine(
        self, 
        valuation_score: float,
        fundamental_score: float
    ) -> Tuple[str, float]:

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

        hold_qual = self.rules['hold']['mixed_quality_price']

        if (valuation_score <= hold_qual['valuation_max'] and 
            fundamental_score >= hold_qual['fundamental_min']):
            confidence = self._calculate_hold_confidence(
                valuation_score, fundamental_score,
                hold_qual['valuation_max'], hold_qual['fundamental_min']
            )
            return "HOLD", confidence

        hold_mod = self.rules['hold']['mixed_moderate']

        if (valuation_score <= hold_mod['valuation_max'] and 
            fundamental_score >= hold_mod['fundamental_min']):
            confidence = self._calculate_hold_confidence(
                valuation_score, fundamental_score,
                hold_mod['valuation_max'], hold_mod['fundamental_min']
            )
            return "HOLD", confidence

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

        confidence = base

        if valuation_score is not None and val_weight > 0:
            if is_sell and val_max is not None:
                excess = val_max - valuation_score
                confidence += excess * val_weight
            elif not is_sell and val_min is not None:
                excess = valuation_score - val_min
                confidence += excess * val_weight

        if fundamental_score is not None and fund_weight > 0:
            if is_sell and fund_max is not None:
                excess = fund_max - fundamental_score
                confidence += excess * fund_weight
            elif not is_sell and fund_min is not None:
                excess = fundamental_score - fund_min
                confidence += excess * fund_weight

        return min(max_conf, max(base, confidence))
    
    def validate_with_upside(
        self,
        signal: str,
        confidence: float,
        upside: float
    ) -> Tuple[str, float]:
        """
        Sanity check: evita contradicciones entre señal y upside potential.
        - SELL con upside alto → HOLD (señales mixtas)
        - BUY con upside negativo → HOLD (señales mixtas)
        """
        sanity = self.rules.get('sanity_check')
        if not sanity:
            return signal, confidence

        if signal == "SELL":
            threshold = sanity['sell_override_to_hold']['upside_min']
            if upside > threshold:
                return "HOLD", sanity['sell_override_to_hold']['confidence']

        if signal == "BUY":
            threshold = sanity['buy_override_to_hold']['upside_max']
            if upside < threshold:
                return "HOLD", sanity['buy_override_to_hold']['confidence']

        return signal, confidence

    def _calculate_hold_confidence(
        self,
        valuation_score: float,
        fundamental_score: float,
        val_max: float,
        fund_min: float
    ) -> float:

        base_confidence = 50.0
        val_distance = abs(valuation_score - val_max) if val_max else 0
        fund_distance = abs(fundamental_score - fund_min) if fund_min else 0
        
        if val_distance < 5 or fund_distance < 5:
            return max(40.0, base_confidence - 10)

        if fundamental_score >= 70 and valuation_score <= 45:
            if fundamental_score >= 80:
                return 60.0
            return 55.0
        
        if 45 <= valuation_score <= 55 and 60 <= fundamental_score <= 75:
            return 55.0

        avg_score = (valuation_score + fundamental_score) / 2
        if avg_score >= 65:
            return 55.0
        elif avg_score >= 55:
            return 52.0
        else:
            return 48.0