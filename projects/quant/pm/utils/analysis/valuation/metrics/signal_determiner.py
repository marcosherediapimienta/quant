from typing import Tuple
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

        scores = {'valuation': valuation_score, 'fundamental': fundamental_score}

        for signal_type, rule_name in SIGNAL_EVALUATION_ORDER:
            rule = self.rules[signal_type].get(rule_name)
            if rule is None or not self._matches_rule(rule, scores):
                continue

            if signal_type == 'hold':
                confidence = self._calculate_hold_confidence(
                    valuation_score, fundamental_score,
                    rule.get('valuation_max', DEFAULT_NA_SCORE),
                    rule.get('fundamental_min', DEFAULT_NA_SCORE)
                )
            else:
                confidence = self._calculate_confidence(rule, scores, is_sell=(signal_type == 'sell'))

            return _SIGNAL_MAP[signal_type], confidence

        return "HOLD", self._calculate_hold_confidence(valuation_score, fundamental_score, DEFAULT_NA_SCORE, DEFAULT_NA_SCORE)

    @staticmethod
    def _matches_rule(rule: dict, scores: dict) -> bool:
        return all(
            score >= rule.get(f'{name}_min', float('-inf')) and
            score <= rule.get(f'{name}_max', float('inf'))
            for name, score in scores.items()
        )

    @staticmethod
    def _calculate_confidence(rule: dict, scores: dict, is_sell: bool) -> float:
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
                excess = (ref - score) if is_sell else (score - ref)
                confidence += excess * weight

        return min(rule['confidence_max'], max(base, confidence))
    
    def validate_with_upside(
        self,
        signal: str,
        confidence: float,
        upside: float
    ) -> Tuple[str, float]:

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

        cfg = self.rules['hold_confidence']

        val_distance = abs(valuation_score - val_max)
        fund_distance = abs(fundamental_score - fund_min)
        
        if val_distance < cfg['proximity_threshold'] or fund_distance < cfg['proximity_threshold']:
            return max(cfg['proximity_min'], cfg['base'] - cfg['proximity_penalty'])

        for tier in cfg['tiers']:
            if self._matches_tier(tier, valuation_score, fundamental_score):
                return tier['confidence']

        avg_score = (valuation_score + fundamental_score) / 2
        for tier in cfg['avg_tiers']:
            if avg_score >= tier['min_avg']:
                return tier['confidence']

        return cfg['default_confidence']

    @staticmethod
    def _matches_tier(tier: dict, valuation_score: float, fundamental_score: float) -> bool:
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
