from typing import List, Optional
from ....tools.config import SCORE_INTERPRETATION, DEFAULT_NA_SCORE

_OPS = {'>=': lambda s, t: s >= t, '<=': lambda s, t: s <= t}

_REASON_SPECS = [
    {
        'category': 'valuation',
        'source': 'valuation',
        'rules': [
            ('>=', 'excellent', " Attractive valuation - Price below fair value"),
            ('>=', 'good', " Reasonable valuation - Price near fair value"),
            ('<=', 'very_poor', " Overvalued - Price significantly above fair value"),
            ('<=', 'poor', "⚡ High valuation - Price above fair value"),
        ]
    },
    {
        'category': 'fundamental',
        'source': None,
        'rules': [
            ('>=', 'exceptional', " Exceptional quality company"),
            ('>=', 'good', " Solid fundamentals"),
            ('<=', 'weak', " Weak fundamentals - Higher risk"),
        ]
    },
    {
        'category': 'profitability',
        'source': 'profitability',
        'rules': [
            ('>=', 'excellent', " Excellent profitability"),
            ('<=', 'poor', " Low or negative profitability"),
        ]
    },
    {
        'category': 'health',
        'source': 'financial_health',
        'rules': [
            ('>=', 'excellent', " Solid balance sheet - Low debt"),
            ('<=', 'poor', " Financial risk - High debt or liquidity issues"),
        ]
    },
    {
        'category': 'growth',
        'source': 'growth',
        'rules': [
            ('>=', 'high', " High sustained growth"),
            ('<=', 'low', " Limited growth or decline"),
        ]
    },
]

_ALERT_SECTIONS = ('valuation', 'profitability', 'financial_health', 'growth')
_MAX_ALERTS = 2

class ReasonGenerator:
    def __init__(self, thresholds: dict = None):
        self.thresholds = thresholds or SCORE_INTERPRETATION['reasons']

    def generate(
        self, 
        analysis: dict, 
        fundamental_score: float,
        signal: Optional[str] = None,
        upside: Optional[float] = None
    ) -> List[str]:

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

        if signal and upside is not None:
            reasons.extend(self._signal_reasons(signal, upside))

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
    def _evaluate_rules(score: float, thresholds: dict, rules: list) -> List[str]:
        for operator, key, message in rules:
            threshold = thresholds.get(key)
            if threshold is not None and _OPS[operator](score, threshold):
                return [message]
        return []
    
    @staticmethod
    def _extract_alerts(analysis: dict) -> List[str]:
        alerts = []
        for section in _ALERT_SECTIONS:
            section_alerts = analysis.get(section, {}).get('alerts', [])
            alerts.extend(section_alerts)
        return [f" {alert}" for alert in alerts[:_MAX_ALERTS]]
