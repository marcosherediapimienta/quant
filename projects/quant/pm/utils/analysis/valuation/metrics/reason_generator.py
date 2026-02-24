import logging
from typing import Any, Dict, List, Mapping, Optional
from ....tools.config import SCORE_INTERPRETATION, DEFAULT_NA_SCORE

logger = logging.getLogger(__name__)

_OPS = {'>=': lambda s, t: s >= t, '<=': lambda s, t: s <= t}

_REASON_SPECS = [
    {
        'category': 'valuation',
        'source': 'valuation',
        'rules': [
            ('>=', 'excellent', "Attractive valuation - Price below fair value"),
            ('>=', 'good', "Reasonable valuation - Price near fair value"),
            ('<=', 'very_poor', "Overvalued - Price significantly above fair value"),
            ('<=', 'poor', "⚡ High valuation - Price above fair value"),
        ]
    },
    {
        'category': 'fundamental',
        'source': None,
        'rules': [
            ('>=', 'exceptional', "Exceptional quality company"),
            ('>=', 'good', "Solid fundamentals"),
            ('<=', 'weak', "Weak fundamentals - Higher risk"),
        ]
    },
    {
        'category': 'profitability',
        'source': 'profitability',
        'rules': [
            ('>=', 'excellent', "Excellent profitability"),
            ('<=', 'poor', "Low or negative profitability"),
        ]
    },
    {
        'category': 'health',
        'source': 'financial_health',
        'rules': [
            ('>=', 'excellent', "Solid balance sheet - Low debt"),
            ('<=', 'poor', "Financial risk - High debt or liquidity issues"),
        ]
    },
    {
        'category': 'growth',
        'source': 'growth',
        'rules': [
            ('>=', 'high', "High sustained growth"),
            ('<=', 'low', "Limited growth or decline"),
        ]
    },
]

_ALERT_SECTIONS = ('valuation', 'profitability', 'financial_health', 'growth')
_MAX_ALERTS = 2

class ReasonGenerator:
    def __init__(self, thresholds: dict = None):
        self.thresholds = thresholds or SCORE_INTERPRETATION['reasons']
        self._validate_thresholds()

    def _validate_thresholds(self) -> None:
        for spec in _REASON_SPECS:
            category = spec['category']
            category_thresholds = self.thresholds.get(category, {})
            missing = [key for _, key, _ in spec['rules'] if key not in category_thresholds]
            if missing:
                # Keep it non-fatal: reasons gracefully degrade to defaults.
                logger.warning("Missing thresholds for '%s': %s", category, missing)

    def generate(
        self, 
        analysis: Mapping[str, Any], 
        fundamental_score: float,
        signal: Optional[str] = None,
        upside: Optional[float] = None
    ) -> List[str]:
        signal_norm = signal.strip().upper() if isinstance(signal, str) else None
        upside_norm = upside
        if upside_norm is not None and abs(upside_norm) > 2:
            upside_norm = upside_norm / 100.0

        score_map = {
            'fundamental': fundamental_score,
        }

        reasons = []
        for spec in _REASON_SPECS:
            category = spec['category']

            if spec['source'] is not None:
                score = analysis.get(spec['source'], {}).get('score', DEFAULT_NA_SCORE)
            else:
                score = score_map.get(category, DEFAULT_NA_SCORE)

            thresh = self.thresholds.get(category, {})
            reasons.extend(self._evaluate_rules(score, thresh, spec['rules']))

        reasons.extend(self._extract_alerts(analysis))

        if signal_norm and upside_norm is not None:
            reasons.extend(self._signal_reasons(signal_norm, upside_norm))

        reasons = list(dict.fromkeys(r.strip() for r in reasons if r and r.strip()))

        return reasons if reasons else ["No specific reasons"]

    @staticmethod
    def _signal_reasons(signal: str, upside: float) -> List[str]:
        pct = f"{upside:+.0%}"
        _SIGNAL_REASONS = [
            ("BUY",  lambda u: u > 0.30,  f"Strong upside potential ({pct})"),
            ("BUY",  lambda u: u > 0.10,  f"Moderate upside potential ({pct})"),
            ("SELL", lambda u: u < -0.10, f"Downside risk ({pct})"),
            ("SELL", lambda u: u > 0.10,  f"Overvalued despite positive target ({pct})"),
            ("HOLD", lambda u: u > 0.20,  f"Upside potential ({pct}) but mixed signals"),
            ("HOLD", lambda u: u < -0.10, f"Downside risk ({pct}) - proceed with caution"),
        ]

        for sig, condition, message in _SIGNAL_REASONS:
            if sig == signal and condition(upside):
                return [message]
        return []
    
    @staticmethod
    def _evaluate_rules(score: float, thresholds: Mapping[str, Any], rules: list) -> List[str]:
        for operator, key, message in rules:
            op_fn = _OPS.get(operator)
            if op_fn is None:
                continue
            threshold = thresholds.get(key)
            if threshold is not None and op_fn(score, threshold):
                return [message]
        return []
    
    @staticmethod
    def _extract_alerts(analysis: Mapping[str, Any]) -> List[str]:
        alerts = []
        for section in _ALERT_SECTIONS:
            section_alerts = analysis.get(section, {}).get('alerts', [])
            alerts.extend(section_alerts)
        out = []
        for alert in alerts[:_MAX_ALERTS]:
            s = str(alert).strip()
            if s:
                out.append(s)
        return out
