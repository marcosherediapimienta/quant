import pandas as pd
from typing import Dict
from dataclasses import dataclass

from .formatters import (
    fmt_pct, fmt_num, fmt_money, fmt_multiple,
    score_bar, score_emoji,
    separator, FormatConfig
)

_SCORE_CATEGORIES = [
    ('Profitability', 'profitability'),
    ('Financial Health', 'financial_health'),
    ('Growth', 'growth'),
    ('Efficiency', 'efficiency'),
    ('Valuation', 'valuation'),
]

@dataclass 
class ReportSections:
    summary: bool = True
    profitability: bool = True
    financial_health: bool = True
    growth: bool = True
    efficiency: bool = True
    valuation: bool = True
    alerts: bool = True

class CompanyReporter:
    def __init__(
        self, 
        format_config: FormatConfig = None,
        sections: ReportSections = None
    ):
        self.config = format_config or FormatConfig()
        self.sections = sections or ReportSections()
    
    def render(self, result: Dict) -> str:

        if not result.get('success'):
            return self._render_error(result)
        
        section_renderers = [
            ('summary', self._render_summary),
            ('profitability', self._render_profitability),
            ('financial_health', self._render_financial_health),
            ('growth', self._render_growth),
            ('efficiency', self._render_efficiency),
            ('valuation', self._render_valuation),
            ('alerts', self._render_alerts),
        ]

        lines = [self._render_header(result)]

        for section_name, renderer in section_renderers:
            if getattr(self.sections, section_name, False):
                rendered = renderer(result)
                if rendered:
                    lines.append(rendered)

        lines.append(self._render_footer(result))
        return "\n".join(filter(None, lines))
    
    @staticmethod
    def _render_error(result: Dict) -> str:
        ticker = result.get('ticker', 'N/A')
        error = result.get('error', 'Unknown error')
        return f"❌ Error analyzing {ticker}: {error}"
    
    def _render_header(self, r: Dict) -> str:
        w = self.config.line_width
        lines = [
            "=" * w,
            f"ANALYSIS: {r.get('company_name', r['ticker'])} ({r['ticker']})",
            "=" * w,
            f"Sector: {r.get('sector') or 'N/A'} | Industry: {r.get('industry') or 'N/A'}",
            f"Country: {r.get('country') or 'N/A'} | Currency: {r.get('currency', 'USD')}",
            ""
        ]
        return "\n".join(lines)
    
    def _render_summary(self, r: Dict) -> str:
        w = self.config.line_width
        bw = self.config.bar_width
        s = r['scores']
        
        lines = [
            separator("─", w),
            "SCORE SUMMARY",
            separator("─", w),
        ]
        
        for name, key in _SCORE_CATEGORIES:
            score = s.get(key)
            emoji = score_emoji(score)
            bar = score_bar(score, bw)
            score_str = fmt_num(score, 1) if pd.notna(score) else "N/A"
            lines.append(f"  {emoji} {name:<18} {bar} {score_str:>6}")
        
        lines.append(separator("─", w))
        total_emoji = score_emoji(s['total'])
        total_bar = score_bar(s['total'], bw)
        lines.append(f"  {total_emoji} {'TOTAL':<18} {total_bar} {fmt_num(s['total'], 1):>6}")
        lines.append(f"  📋 Conclusion: {r['conclusion']['overall']}")
        lines.append("")
        
        return "\n".join(lines)
    
    def _render_profitability(self, r: Dict) -> str:
        w = self.config.line_width
        prof = r['profitability']['metrics']
        cls = r['profitability']['classifications']
        score = r['scores']['profitability']
        
        lines = [
            separator("─", w),
            f"PROFITABILITY (Score: {fmt_num(score, 1)})",
            separator("─", w),
            f"  ROIC:             {fmt_pct(prof['roic']):<12} {cls['roic_class']}",
            f"  ROE:              {fmt_pct(prof['roe']):<12} {cls['roe_class']}",
            f"  ROA:              {fmt_pct(prof['roa']):<12} {cls['roa_class']}",
            f"  Gross Margin:     {fmt_pct(prof['gross_margin']):<12} {cls['gross_margin_class']}",
            f"  Operating Margin: {fmt_pct(prof['operating_margin']):<12} {cls['operating_margin_class']}",
            f"  Net Margin:       {fmt_pct(prof['net_margin']):<12} {cls['net_margin_class']}",
            ""
        ]
        return "\n".join(lines)
    
    def _render_financial_health(self, r: Dict) -> str:
        w = self.config.line_width
        h = r['financial_health']['metrics']
        cls = r['financial_health']['classifications']
        score = r['scores']['financial_health']
        
        lines = [
            separator("─", w),
            f"FINANCIAL HEALTH (Score: {fmt_num(score, 1)})",
            separator("─", w),
            f"  Total Debt:       {fmt_money(h['total_debt'])}",
            f"  Total Cash:       {fmt_money(h['total_cash'])}",
            f"  Net Cash:         {fmt_money(h['net_cash'])}",
            f"  Debt/EBITDA:      {fmt_multiple(h['debt_ebitda']):<12} {cls['debt_ebitda_class']}",
            f"  Debt/Equity:      {fmt_multiple(h['debt_equity']):<12} {cls['debt_equity_class']}",
            f"  Current Ratio:    {fmt_num(h['current_ratio']):<12} {cls['current_ratio_class']}",
            f"  Quick Ratio:      {fmt_num(h['quick_ratio'])}",
            f"  Free Cash Flow:   {fmt_money(h['free_cash_flow'])}",
            ""
        ]
        return "\n".join(lines)
    
    def _render_growth(self, r: Dict) -> str:
        w = self.config.line_width
        g = r['growth']['metrics']
        cls = r['growth']['classifications']
        score = r['scores']['growth']
        
        lines = [
            separator("─", w),
            f"GROWTH (Score: {fmt_num(score, 1)})",
            separator("─", w),
            f"  Revenue YoY:      {fmt_pct(g['revenue_growth_yoy']):<12} {cls['revenue_growth_class']}",
            f"  Earnings YoY:     {fmt_pct(g['earnings_growth_yoy']):<12} {cls['earnings_growth_class']}",
            f"  Earnings Qtr:     {fmt_pct(g['earnings_quarterly_growth'])}",
        ]
        
        sust = r['growth'].get('sustainability', {})
        if sust.get('concerns'):
            lines.append(f"  ⚠️ {', '.join(sust['concerns'])}")
        
        lines.append("")
        return "\n".join(lines)
    
    def _render_efficiency(self, r: Dict) -> str:
        w = self.config.line_width
        e = r['efficiency']['metrics']
        score = r['scores']['efficiency']
        
        lines = [
            separator("─", w),
            f"EFFICIENCY (Score: {fmt_num(score, 1)})",
            separator("─", w),
            f"  Asset Turnover:   {fmt_multiple(e['asset_turnover'])}",
            f"  DSO:              {fmt_num(e['days_sales_outstanding'], 0)} days",
            f"  DIO:              {fmt_num(e['days_inventory_outstanding'], 0)} days",
            f"  Revenue/Employee: {fmt_money(e['revenue_per_employee'])}",
            ""
        ]
        return "\n".join(lines)
    
    def _render_valuation(self, r: Dict) -> str:
        w = self.config.line_width
        v = r['valuation']['metrics']
        cls = r['valuation']['classifications']
        score = r['scores']['valuation']
        
        lines = [
            separator("─", w),
            f"VALUATION (Score: {fmt_num(score, 1)})",
            separator("─", w),
            f"  Market Cap:       {fmt_money(v['market_cap'])}",
            f"  P/E (TTM):        {fmt_multiple(v['pe_ttm']):<12} {cls['pe_class']}",
            f"  P/E (Forward):    {fmt_multiple(v['pe_forward'])}",
            f"  EV/EBITDA:        {fmt_multiple(v['ev_ebitda']):<12} {cls['ev_ebitda_class']}",
            f"  P/B:              {fmt_multiple(v['pb_ratio']):<12} {cls['pb_class']}",
            f"  P/S:              {fmt_multiple(v['ps_ratio'])}",
            f"  FCF Yield:        {fmt_pct(v['fcf_yield']):<12} {cls['fcf_yield_class']}",
            f"  PEG Ratio:        {fmt_multiple(v['peg_ratio']):<12} {cls.get('peg_class', 'N/A')}",
            f"  Valuation:        {cls['overall']}",
            ""
        ]
        return "\n".join(lines)
    
    def _render_alerts(self, r: Dict) -> str:
        categories = ('valuation', 'profitability', 'financial_health', 'growth', 'efficiency')
        alerts = {cat: r.get(cat, {}).get('alerts', []) for cat in categories}

        if not any(alerts[cat] for cat in categories):
            return ""
        
        w = self.config.line_width
        lines = [
            separator("─", w),
            "🚨 ALERTS",
            separator("─", w),
        ]
        
        for category in categories:
            for alert in alerts[category]:
                lines.append(f"  ⚠️ [{category}] {alert}")
        
        lines.append("")
        return "\n".join(lines)
    
    def _render_footer(self, r: Dict) -> str:
        w = self.config.line_width
        s = r['scores']
        
        lines = [
            "=" * w,
            f"🎯 FINAL SCORE: {fmt_num(s['total'], 1)}/100 → {r['conclusion']['overall']}",
            "=" * w,
        ]
        return "\n".join(lines)
