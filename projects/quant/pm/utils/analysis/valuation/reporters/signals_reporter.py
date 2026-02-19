import pandas as pd
from typing import List
from dataclasses import dataclass
from .formatters import (
    fmt_pct, fmt_money,
    score_bar, score_emoji,
    FormatConfig
)

from ....tools.config import REPORTING_CONFIG

_SIGNAL_EMOJIS = {
    'BUY': '🟢',
    'SELL': '🔴',
    'HOLD': '🟡',
}

@dataclass
class SignalsReportSections:
    individual: bool = True
    summary: bool = True
    top_opportunities: bool = True
    full_table: bool = True


class SignalsReporter:
    def __init__(
        self,
        format_config: FormatConfig = None,
        sections: SignalsReportSections = None
    ):
        self.config = format_config or FormatConfig()
        self.sections = sections or SignalsReportSections()
        self.reporting_cfg = REPORTING_CONFIG
    
    def print_signal(self, signal) -> None:

        if not self.sections.individual:
            return
        
        w = self.config.line_width
        print("=" * w)
        print(f"INVESTMENT SIGNAL: {signal.ticker}")
        print("=" * w)
        
        emoji = _SIGNAL_EMOJIS.get(signal.signal.upper(), '⚪')
        print(f"\n{emoji} {signal.signal} (Confidence: {signal.confidence:.1f}%)")

        print(f"📊 SCORES:")
        print(f"   Valuation:    {score_emoji(signal.valuation_score)} {score_bar(signal.valuation_score)} {signal.valuation_score:.1f}")
        print(f"   Fundamental:   {score_emoji(signal.fundamental_score)} {score_bar(signal.fundamental_score)} {signal.fundamental_score:.1f}")

        print(f"💰 PRICES:")
        print(f"   Current:       {fmt_money(signal.current_price)}")
        print(f"   Target:        {fmt_money(signal.price_target)}")
        print(f"   Potential:     {fmt_pct(signal.upside_potential)}")

        if signal.reasons:
            print(f"\n💡 REASONS:")
            max_display = self.reporting_cfg['max_reasons_display']
            for reason in signal.reasons[:max_display + 1]: 
                print(f"   {reason}")
        
        print("=" * w)
    
    def print_summary(self, signals: List) -> None:

        if not self.sections.summary:
            return
        
        by_type = self._group_by_signal(signals)
        
        w = self.config.line_width
        print("\n" + "=" * w)
        print("SIGNALS SUMMARY")
        print("=" * w)

        top_n = self.reporting_cfg['top_opportunities']
        
        print(f"\n🟢 BUYS: {len(by_type['BUY'])}")
        if by_type['BUY']:
            for s in sorted(by_type['BUY'], key=lambda x: x.confidence, reverse=True)[:top_n]:
                print(f"   {s.ticker}: {s.confidence:.1f}% confidence, {fmt_pct(s.upside_potential)} potential")
        
        print(f"\n🔴 SELLS: {len(by_type['SELL'])}")
        if by_type['SELL']:
            for s in sorted(by_type['SELL'], key=lambda x: x.confidence, reverse=True)[:top_n]:
                print(f"   {s.ticker}: {s.confidence:.1f}% confidence")
        
        print(f"\n🟡 HOLDS: {len(by_type['HOLD'])}")
        print("=" * w)

    def print_top_opportunities(self, signals: List, top_n: int = None) -> None:

        if not self.sections.top_opportunities:
            return
        
        if top_n is None:
            top_n = self.reporting_cfg['top_opportunities']
        
        by_type = self._group_by_signal(signals)
        buys = by_type['BUY']

        if not buys:
            print("\n⚠️ No buy opportunities identified")
            return
        
        w = self.config.line_width
        print("\n" + "=" * w)
        print(f"TOP {top_n} BUY OPPORTUNITIES")
        print("=" * w)

        top_buys = sorted(
            buys, 
            key=lambda x: (x.confidence, x.upside_potential), 
            reverse=True
        )[:top_n]
        
        max_reasons = self.reporting_cfg['max_reasons_display']
        
        for i, s in enumerate(top_buys, 1):
            print(f"\n{i}. {s.ticker}")
            print(f"   Confidence: {s.confidence:.1f}%")
            print(f"   Potential: {fmt_pct(s.upside_potential)}")
            print(f"   Price: {fmt_money(s.current_price)} → {fmt_money(s.price_target)}")
            print(f"   Valuation: {s.valuation_score:.1f} | Fundamental: {s.fundamental_score:.1f}")
            if s.reasons:
                print(f"   Reasons: {', '.join(s.reasons[:max_reasons])}")

    def to_dataframe(self, signals: List) -> pd.DataFrame:
        return pd.DataFrame([{
            'Ticker': s.ticker,
            'Signal': s.signal,
            'Confidence': f"{s.confidence:.1f}%",
            'Valuation': f"{s.valuation_score:.1f}",
            'Fundamental': f"{s.fundamental_score:.1f}",
            'Current Price': fmt_money(s.current_price),
            'Target Price': fmt_money(s.price_target),
            'Potential': fmt_pct(s.upside_potential)
        } for s in signals])
    
    @staticmethod
    def _group_by_signal(signals: List) -> dict:
        groups = {'BUY': [], 'SELL': [], 'HOLD': []}
        for s in signals:
            key = s.signal.upper()
            if key in groups:
                groups[key].append(s)
            else:
                groups['HOLD'].append(s)
        return groups
