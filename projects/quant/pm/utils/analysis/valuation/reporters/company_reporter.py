import pandas as pd
from typing import Dict
from dataclasses import dataclass

from .formatters import (
    fmt_pct, fmt_num, fmt_money, fmt_multiple,
    score_bar, score_emoji,
    separator, FormatConfig
)


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
        
        lines = []
        
        # Header
        lines.append(self._render_header(result))
        
        # Secciones
        if self.sections.summary:
            lines.append(self._render_summary(result))
        
        if self.sections.profitability:
            lines.append(self._render_profitability(result))
        
        if self.sections.financial_health:
            lines.append(self._render_financial_health(result))
        
        if self.sections.growth:
            lines.append(self._render_growth(result))
        
        if self.sections.efficiency:
            lines.append(self._render_efficiency(result))
        
        if self.sections.valuation:
            lines.append(self._render_valuation(result))
        
        if self.sections.alerts:
            lines.append(self._render_alerts(result))
        
        # Footer
        lines.append(self._render_footer(result))
        
        return "\n".join(filter(None, lines))
    
    def _render_error(self, result: Dict) -> str:
        ticker = result.get('ticker', 'N/A')
        error = result.get('error', 'Error desconocido')
        return f"❌ Error analizando {ticker}: {error}"
    
    def _render_header(self, r: Dict) -> str:
        w = self.config.line_width
        lines = [
            "=" * w,
            f"ANÁLISIS: {r.get('name', r['ticker'])} ({r['ticker']})",
            "=" * w,
            f"Sector: {r.get('sector') or 'N/A'} | Industria: {r.get('industry') or 'N/A'}",
            f"País: {r.get('country') or 'N/A'} | Moneda: {r.get('currency', 'USD')}",
            ""
        ]
        return "\n".join(lines)
    
    def _render_summary(self, r: Dict) -> str:
        w = self.config.line_width
        bw = self.config.bar_width
        s = r['scores']
        
        categories = [
            ('Rentabilidad', 'profitability'),
            ('Salud Financiera', 'financial_health'),
            ('Crecimiento', 'growth'),
            ('Eficiencia', 'efficiency'),
            ('Valoración', 'valuation'),
        ]
        
        lines = [
            separator("─", w),
            "RESUMEN DE SCORES",
            separator("─", w),
        ]
        
        for name, key in categories:
            score = s.get(key)
            emoji = score_emoji(score)
            bar = score_bar(score, bw)
            score_str = fmt_num(score, 1) if pd.notna(score) else "N/A"
            lines.append(f"  {emoji} {name:<18} {bar} {score_str:>6}")
        
        lines.append(separator("─", w))
        total_emoji = score_emoji(s['total'])
        total_bar = score_bar(s['total'], bw)
        lines.append(f"  {total_emoji} {'TOTAL':<18} {total_bar} {fmt_num(s['total'], 1):>6}")
        lines.append(f"  📋 Conclusión: {r['conclusion']}")
        lines.append("")
        
        return "\n".join(lines)
    
    def _render_profitability(self, r: Dict) -> str:
        w = self.config.line_width
        prof = r['profitability']['metrics']
        cls = r['profitability']['classifications']
        score = r['scores']['profitability']
        
        lines = [
            separator("─", w),
            f"RENTABILIDAD (Score: {fmt_num(score, 1)})",
            separator("─", w),
            f"  ROIC:             {fmt_pct(prof['roic']):<12} {cls['roic_class']}",
            f"  ROE:              {fmt_pct(prof['roe']):<12} {cls['roe_class']}",
            f"  ROA:              {fmt_pct(prof['roa']):<12} {cls['roa_class']}",
            f"  Margen Bruto:     {fmt_pct(prof['gross_margin']):<12} {cls['gross_margin_class']}",
            f"  Margen Operativo: {fmt_pct(prof['operating_margin']):<12} {cls['operating_margin_class']}",
            f"  Margen Neto:      {fmt_pct(prof['net_margin']):<12} {cls['net_margin_class']}",
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
            f"SALUD FINANCIERA (Score: {fmt_num(score, 1)})",
            separator("─", w),
            f"  Deuda Total:      {fmt_money(h['total_debt'])}",
            f"  Caja Total:       {fmt_money(h['total_cash'])}",
            f"  Caja Neta:        {fmt_money(h['net_cash'])}",
            f"  Deuda/EBITDA:     {fmt_multiple(h['debt_ebitda']):<12} {cls['debt_ebitda_class']}",
            f"  Deuda/Equity:     {fmt_multiple(h['debt_equity']):<12} {cls['debt_equity_class']}",
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
            f"📈 CRECIMIENTO (Score: {fmt_num(score, 1)})",
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
            f"EFICIENCIA (Score: {fmt_num(score, 1)})",
            separator("─", w),
            f"  Asset Turnover:   {fmt_multiple(e['asset_turnover'])}",
            f"  DSO:              {fmt_num(e['days_sales_outstanding'], 0)} días",
            f"  DIO:              {fmt_num(e['days_inventory_outstanding'], 0)} días",
            f"  Revenue/Empleado: {fmt_money(e['revenue_per_employee'])}",
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
            f"VALORACIÓN (Score: {fmt_num(score, 1)})",
            separator("─", w),
            f"  Market Cap:       {fmt_money(v['market_cap'])}",
            f"  P/E (TTM):        {fmt_multiple(v['pe_ttm']):<12} {cls['pe_class']}",
            f"  P/E (Forward):    {fmt_multiple(v['pe_forward'])}",
            f"  EV/EBITDA:        {fmt_multiple(v['ev_ebitda']):<12} {cls['ev_ebitda_class']}",
            f"  P/B:              {fmt_multiple(v['pb_ratio']):<12} {cls['pb_class']}",
            f"  P/S:              {fmt_multiple(v['ps_ratio'])}",
            f"  FCF Yield:        {fmt_pct(v['fcf_yield']):<12} {cls['fcf_yield_class']}",
            f"  📊 Valoración:    {cls['overall']}",
            ""
        ]
        return "\n".join(lines)
    
    def _render_alerts(self, r: Dict) -> str:
        alerts = r.get('alerts', {})
        
        if not any(alerts.values()):
            return ""
        
        w = self.config.line_width
        lines = [
            separator("─", w),
            "🚨 ALERTAS",
            separator("─", w),
        ]
        
        for category, alert_list in alerts.items():
            for alert in alert_list:
                lines.append(f"  ⚠️ [{category}] {alert}")
        
        lines.append("")
        return "\n".join(lines)
    
    def _render_footer(self, r: Dict) -> str:
        w = self.config.line_width
        s = r['scores']
        
        lines = [
            "=" * w,
            f"🎯 SCORE FINAL: {fmt_num(s['total'], 1)}/100 → {r['conclusion']}",
            "=" * w,
        ]
        return "\n".join(lines)