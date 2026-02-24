import math
from typing import Any, Mapping, Tuple
from ....tools.config import TRADING_SIGNAL_RULES, SIGNAL_EVALUATION_ORDER, DEFAULT_NA_SCORE

_SIGNAL_MAP = {'buy': 'BUY', 'sell': 'SELL', 'hold': 'HOLD'}

class SignalDeterminer:
    def __init__(self, rules: dict = None):
        self.rules = rules or TRADING_SIGNAL_RULES

    def determine(
        self, 
        valuation_score: float,
        fundamental_score: float
    ) -> Tuple[str, float]:
        valuation_score = self._normalize_score(valuation_score)
        fundamental_score = self._normalize_score(fundamental_score)

        scores = {'valuation': valuation_score, 'fundamental': fundamental_score}

        for signal_type, rule_name in SIGNAL_EVALUATION_ORDER:
            signal_key = str(signal_type).lower()
            bucket = (
                self.rules.get(signal_type)
                or self.rules.get(str(signal_type).lower())
                or self.rules.get(str(signal_type).upper())
                or {}
            )
            rule = bucket.get(rule_name)
            if rule is None or not self._matches_rule(rule, scores):
                continue

            if signal_key == 'hold':
                confidence = self._calculate_hold_confidence(
                    valuation_score, fundamental_score,
                    rule.get('valuation_max', DEFAULT_NA_SCORE),
                    rule.get('fundamental_min', DEFAULT_NA_SCORE)
                )
            else:
                confidence = self._calculate_confidence(rule, scores, is_sell=(signal_key == 'sell'))

            return _SIGNAL_MAP.get(signal_key, "HOLD"), confidence

        return "HOLD", self._calculate_hold_confidence(valuation_score, fundamental_score, DEFAULT_NA_SCORE, DEFAULT_NA_SCORE)

    @staticmethod
    def _matches_rule(rule: Mapping[str, Any], scores: Mapping[str, float]) -> bool:
        for name, score in scores.items():
            lo = rule.get(f'{name}_min')
            hi = rule.get(f'{name}_max')

            if lo is not None:
                try:
                    lo = float(lo)
                except (TypeError, ValueError):
                    lo = float('-inf')
            else:
                lo = float('-inf')

            if hi is not None:
                try:
                    hi = float(hi)
                except (TypeError, ValueError):
                    hi = float('inf')
            else:
                hi = float('inf')

            if not (lo <= score <= hi):
                return False
        return True

    @staticmethod
    def _calculate_confidence(rule: Mapping[str, Any], scores: Mapping[str, float], is_sell: bool) -> float:
        base = rule['confidence_base']
        confidence = base

        dimensions = (
            (scores['valuation'], rule.get('valuation_min'), rule.get('valuation_max'), rule.get('valuation_weight', 0)),
            (scores['fundamental'], rule.get('fundamental_min'), rule.get('fundamental_max'), rule.get('fundamental_weight', 0)),
        )

        for score, lo, hi, weight in dimensions:
            if weight <= 0:
                continue
            ref = hi if is_sell else lo
            if ref is not None:
                try:
                    ref = float(ref)
                except (TypeError, ValueError):
                    continue
                excess = (ref - score) if is_sell else (score - ref)
                confidence += excess * weight

        return min(rule['confidence_max'], max(base, confidence))
    
    def validate_with_upside(
        self,
        signal: str,
        confidence: float,
        upside: float
    ) -> Tuple[str, float]:
        upside = self._normalize_upside(upside)
        signal = signal.strip().upper()

        sanity = self.rules.get('sanity_check')
        if not sanity:
            return signal, confidence

        key = f'{signal.lower()}_override_to_hold'
        override = sanity.get(key)
        if override is None:
            return signal, confidence

        if signal == "SELL" and upside > override['upside_min']:
            return "HOLD", override['confidence']
        if signal == "BUY" and upside < override['upside_max']:
            return "HOLD", override['confidence']

        return signal, confidence

    def _calculate_hold_confidence(
        self,
        valuation_score: float,
        fundamental_score: float,
        val_max: float,
        fund_min: float
    ) -> float:

        cfg = self.rules.get('hold_confidence') or {}
        proximity_threshold = float(cfg.get('proximity_threshold', 10.0))
        proximity_min = float(cfg.get('proximity_min', 35.0))
        base = float(cfg.get('base', 40.0))
        proximity_penalty = float(cfg.get('proximity_penalty', 5.0))
        tiers = cfg.get('tiers', [])
        avg_tiers = cfg.get('avg_tiers', [])
        default_confidence = float(cfg.get('default_confidence', 40.0))

        val_distance = abs(valuation_score - val_max)
        fund_distance = abs(fundamental_score - fund_min)
        
        if val_distance < proximity_threshold or fund_distance < proximity_threshold:
            return max(proximity_min, base - proximity_penalty)

        for tier in tiers:
            if self._matches_tier(tier, valuation_score, fundamental_score):
                return tier['confidence']

        avg_score = (valuation_score + fundamental_score) / 2
        for tier in avg_tiers:
            if avg_score >= tier['min_avg']:
                return tier['confidence']

        return default_confidence

    @staticmethod
    def _matches_tier(tier: Mapping[str, Any], valuation_score: float, fundamental_score: float) -> bool:
        scores = {'val': valuation_score, 'fund': fundamental_score}
        for prefix, score in scores.items():
            if f'{prefix}_min' in tier and score < tier[f'{prefix}_min']:
                return False
            if f'{prefix}_max' in tier and score > tier[f'{prefix}_max']:
                return False
            if f'{prefix}_range' in tier:
                lo, hi = tier[f'{prefix}_range']
                if not (lo <= score <= hi):
                    return False
        return True

    @staticmethod
    def _normalize_score(score: float) -> float:
        try:
            score = float(score)
        except (TypeError, ValueError):
            return float(DEFAULT_NA_SCORE)
        return score if math.isfinite(score) else float(DEFAULT_NA_SCORE)

    @staticmethod
    def _normalize_upside(upside: float) -> float:
        try:
            upside = float(upside)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(upside):
            return 0.0
        if abs(upside) > 2:
            return upside / 100.0
        return upside
