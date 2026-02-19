import numpy as np
from typing import Dict
from ..analyzers.macro_situation_analyzer import MacroSituationAnalyzer

def _attr(obj, name, default=None):

    if hasattr(obj, 'get'):
        return getattr(obj, name, obj.get(name, default))
    return getattr(obj, name, default)


def _fmt_change(value, suffix="%") -> str:
    return f"{value:>+6.2f}{suffix}" if not np.isnan(value) else "N/A"


class MacroSituationReporter:
    def __init__(self, analyzer: MacroSituationAnalyzer = None):
        self.analyzer = analyzer if analyzer is not None else MacroSituationAnalyzer()
    
    def print_situation(self, analysis: Dict) -> None:
        print("GLOBAL MACROECONOMIC SITUATION".center(80))
        summary = self.analyzer.get_summary(analysis)
        self._print_executive_summary(summary)
        self._print_yield_curve(analysis['yield_curve'])
        if analysis.get('implied_yield_curve'):
            self._print_implied_yield_curve(analysis['implied_yield_curve'])
        self._print_inflation(analysis['inflation'])
        self._print_credit(analysis['credit'])
        self._print_global_bonds(analysis['global_bonds'])
        self._print_risk_sentiment(analysis['risk_sentiment'])
    
    def _print_executive_summary(self, summary: Dict) -> None:
        print("EXECUTIVE SUMMARY")
        risk_level = summary.get('overall_risk', 'N/A')
        _RISK_LABELS = {'HIGH': '[!!] HIGH', 'MODERATE': '[!] MODERATE'}
        risk_display = _RISK_LABELS.get(risk_level, '[OK] LOW')
        
        print(f"\n  Global risk level: {risk_display}")
        risk_factors = summary.get('risk_factors', [])

        if risk_factors:
            print(f"\n  Risk factors detected:")
            for factor in risk_factors:
                print(f"    - {factor}")
        else:
            print(f"\n  [OK] No significant risk factors detected")
    
    def _print_yield_curve(self, curve) -> None:
        print("US YIELD CURVE")
        levels = _attr(curve, 'levels', {})
        if levels:
            print("\n  Current levels:")
            for tenor, rate in sorted(levels.items()):
                print(f"    {tenor:>4}: {rate:>6.2f}%")

        spreads = _attr(curve, 'spreads', {})
        if spreads:
            print("\n  Spreads:")
            for spread_name, value in spreads.items():
                tag = "[-]" if value < 0 else "[+]"
                print(f"    {tag} {spread_name:>10}: {value:>+6.2f} pp")

        rate_changes = _attr(curve, 'rate_changes', {})
        if rate_changes:
            print("\n  Rate changes:")
            print(f"    {'Tenor':<6} {'1 Month':<12} {'3 Months':<12} {'1 Year':<12}")
            for tenor in sorted(rate_changes.keys()):
                changes = rate_changes[tenor]
                vals = [_fmt_change(changes.get(k, np.nan), " pp") for k in ('1m', '3m', '1y')]
                print(f"    {tenor:<6} {vals[0]:<12} {vals[1]:<12} {vals[2]:<12}")

        divergence = _attr(curve, 'divergence_analysis', {})
        if divergence:
            print("\n  Short vs Long Rate Divergence:")
            for period, label in (('3m', '3 months'), ('1y', '1 year')):
                if period not in divergence:
                    continue
                d = divergence[period]
                print(f"    {label}:")
                print(f"      Short (2Y): {d['short']:>+6.2f} pp")
                print(f"      Long (10Y): {d['long']:>+6.2f} pp")
                div_val = d['divergence']
                if div_val > 0.5:
                    note = " (Long rising faster -- inflation expectations)"
                elif div_val < -0.5:
                    note = " (Short rising faster -- restrictive policy)"
                else:
                    note = " (Aligned movements)"
                print(f"      Divergence: {div_val:>+6.2f} pp{note}")

        interpretation = _attr(curve, 'interpretation', 'N/A')
        print(f"\n  {interpretation}")
    
    def _print_implied_yield_curve(self, implied_analysis) -> None:
        print("IMPLIED YIELD CURVE (FORWARD RATES)")
        
        forwards = _attr(implied_analysis, 'forward_rates', {})
        fwd_vs_spot = _attr(implied_analysis, 'forward_vs_spot', {})
        term_premium = _attr(implied_analysis, 'term_premium', {})

        print("\n  Spot vs Implied Forward Curve:")
        print(f"    {'Segment':<14} {'Forward (%)':<14} {'vs Spot':<14} {'Signal'}")
        print(f"    {'─'*56}")
        
        for segment, fwd_rate in forwards.items():
            diff = fwd_vs_spot.get(segment, np.nan)
            if not np.isnan(fwd_rate):
                if not np.isnan(diff):
                    tag = "[UP]" if diff > 0.3 else "[DN]" if diff < -0.3 else "[--]"
                    print(f"    {segment:<14} {fwd_rate:>6.2f}%       {diff:>+6.2f} pp     {tag}")
                else:
                    print(f"    {segment:<14} {fwd_rate:>6.2f}%")

        if term_premium:
            print(f"\n  Estimated Term Premium:")
            for tenor, tp in term_premium.items():
                tag = "[-]" if tp < -0.2 else "[+]" if tp > 0.5 else "[=]"
                print(f"    {tag} {tenor}: {tp:>+6.2f} pp")

        expectations = _attr(implied_analysis, 'curve_expectations', 'N/A')
        signal = _attr(implied_analysis, 'rate_path_signal', 'N/A')
        print(f"\n  Expectations: {expectations}")
        print(f"  Monetary policy signal: {signal}")

    def _print_inflation(self, inflation) -> None:
        print("INFLATION SIGNALS (Commodities)")
        commodity_changes = _attr(inflation, 'commodity_changes', {})
        commodity_names = _attr(inflation, 'commodity_names', {})
        
        print("\n  12-month change:")
        for key, name in commodity_names.items():
            if key in commodity_changes:
                change = commodity_changes[key]
                tag = "[!!]" if change > 15 else "[!]" if change > 5 else "[OK]"
                print(f"    {tag} {name:<12}: {change:>+7.2f}%")

        pressure = _attr(inflation, 'inflation_pressure', 'N/A')
        avg_change = _attr(inflation, 'avg_commodity_change', np.nan)
        
        print(f"\n  {pressure}")
        if not np.isnan(avg_change):
            print(f"  Average change: {avg_change:>+.2f}%")
    
    def _print_credit(self, credit) -> None:
        print("CREDIT CONDITIONS & VOLATILITY")
        vix = _attr(credit, 'vix_level')
        if vix is not None:
            tag = "[!!]" if vix > 30 else "[!]" if vix > 20 else "[OK]"
            print(f"\n  {tag} VIX (volatility):      {vix:>7.2f}")

        condition = _attr(credit, 'market_condition', 'N/A')
        print(f"\n  {condition}")

        hyg = _attr(credit, 'hyg_level')
        lqd = _attr(credit, 'lqd_level')
        if hyg is not None and lqd is not None:
            print(f"\n  Credit ETF levels:")
            print(f"    HYG (High Yield):        ${hyg:>7.2f}")
            print(f"    LQD (Investment Grade):  ${lqd:>7.2f}")
    
    def _print_global_bonds(self, bonds: Dict) -> None:
        print("GLOBAL SOVEREIGN BONDS")

        if not bonds:
            print("\n  [!] No global bond data available")
            return
        
        print(f"  {'Region':<15} {'Level':<12} {'1 Month':<12} {'1 Year':<12}")

        for region, data in sorted(bonds.items()):
            level = data.get('level', np.nan)
            change_1m = data.get('change_1m', np.nan)
            change_1y = data.get('change_1y', np.nan)

            def _bond_tag(val, threshold):
                if np.isnan(val):
                    return ""
                return "[-]" if val < -threshold else "[+]" if val > threshold else "[=]"

            tag_1m = _bond_tag(change_1m, 5)
            tag_1y = _bond_tag(change_1y, 10)
            
            level_str = f"${level:.2f}" if not np.isnan(level) else "N/A"
            change_1m_str = f"{tag_1m} {change_1m:>+6.2f}%" if not np.isnan(change_1m) else "N/A"
            change_1y_str = f"{tag_1y} {change_1y:>+6.2f}%" if not np.isnan(change_1y) else "N/A"
            
            print(f"  {region:<15} {level_str:<12} {change_1m_str:<12} {change_1y_str:<12}")
    
    def _print_risk_sentiment(self, sentiment) -> None:
        print("RISK SENTIMENT")
        fear = _attr(sentiment, 'fear_level')

        if fear:
            print(f"\n  Fear level:               {fear}")

        _TREND_SPECS = (
            ('dollar_strength', 'Dollar strength', 'dxy_trend'),
            ('safe_haven',      'Safe-haven demand', 'gold_trend'),
        )

        for attr_name, label, trend_prefix in _TREND_SPECS:
            value = _attr(sentiment, attr_name)
            if not value:
                continue
            parts = []
            for suffix, period in (('_1w', '1W'), ('_1m', '1M'), ('_3m', '3M')):
                t = _attr(sentiment, f'{trend_prefix}{suffix}')
                if t is not None:
                    parts.append(f"{period}: {t:+.2f}%")
            trend_str = f"({', '.join(parts)})" if parts else ""
            print(f"  {label + ':':<28}{value} {trend_str}")

    def print_compact(self, analysis: Dict) -> None:
        print("MACRO SNAPSHOT".center(60))
        curve = analysis['yield_curve']
        spreads = _attr(curve, 'spreads', {})
        if '10Y-2Y' in spreads:
            spread = spreads['10Y-2Y']
            tag = "[-]" if spread < 0 else "[+]"
            print(f"\n  {tag} Curve 10Y-2Y: {spread:+.2f} pp")

        inflation = analysis['inflation']
        pressure = _attr(inflation, 'inflation_pressure', 'N/A')
        print(f"  Inflation: {pressure}")
        
        credit = analysis['credit']
        vix = _attr(credit, 'vix_level')
        if vix is not None:
            tag = "[!!]" if vix > 25 else "[!]" if vix > 20 else "[OK]"
            print(f"  {tag} VIX: {vix:.1f}")

        summary = self.analyzer.get_summary(analysis)
        risk = summary.get('overall_risk', 'N/A')
        print(f"\n  [!] Global risk: {risk}")
        
        print("="*60)
