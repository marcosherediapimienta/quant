import pandas as pd
from typing import List
from dataclasses import dataclass
from .formatters import (
    fmt_pct, fmt_money,
    score_bar, score_emoji,
    FormatConfig
)
from ....tools.config import REPORTING_CONFIG

@dataclass
class SignalsReportSections:
    """Configuración de secciones a incluir en reporte."""
    individual: bool = True
    summary: bool = True
    top_opportunities: bool = True
    full_table: bool = True


class SignalsReporter:
    """
    Reporter para señales de trading.
    
    Responsabilidad: Formatear y presentar señales de inversión de forma clara.
    """

    def __init__(
        self,
        format_config: FormatConfig = None,
        sections: SignalsReportSections = None
    ):
        self.config = format_config or FormatConfig()
        self.sections = sections or SignalsReportSections()
        # Cargar configuración de reporting
        self.reporting_cfg = REPORTING_CONFIG
    
    def print_signal(self, signal) -> None:
        """Imprime señal individual."""
        if not self.sections.individual:
            return
        
        w = self.config.line_width
        print("=" * w)
        print(f"SEÑAL DE INVERSIÓN: {signal.ticker}")
        print("=" * w)
        
        emoji = self._get_signal_emoji(signal.signal)
        print(f"\n{emoji} {signal.signal} (Confianza: {signal.confidence:.1f}%)")

        print(f"📊 SCORES:")
        print(f"   Valoración:    {score_emoji(signal.valuation_score)} {score_bar(signal.valuation_score)} {signal.valuation_score:.1f}")
        print(f"   Fundamental:   {score_emoji(signal.fundamental_score)} {score_bar(signal.fundamental_score)} {signal.fundamental_score:.1f}")
        
        # Precios
        print(f"💰 PRECIOS:")
        print(f"   Actual:        {fmt_money(signal.current_price)}")
        print(f"   Objetivo:      {fmt_money(signal.price_target)}")
        print(f"   Potencial:     {fmt_pct(signal.upside_potential / 100)}")

        if signal.reasons:
            print(f"\n💡 RAZONES:")
            max_display = self.reporting_cfg['max_reasons_display']
            for reason in signal.reasons[:max_display + 1]:  # Mostrar un poco más que el límite
                print(f"   {reason}")
        
        print("=" * w)
    
    def print_summary(self, signals: List) -> None:
        """Imprime resumen de señales."""
        if not self.sections.summary:
            return
        
        compras = [s for s in signals if s.signal == "COMPRA"]
        ventas = [s for s in signals if s.signal == "VENTA"]
        mantener = [s for s in signals if s.signal == "MANTENER"]
        
        w = self.config.line_width
        print("\n" + "=" * w)
        print("RESUMEN DE SEÑALES")
        print("=" * w)

        top_n = self.reporting_cfg['top_opportunities']
        
        print(f"\n🟢 COMPRAS: {len(compras)}")
        if compras:
            for s in sorted(compras, key=lambda x: x.confidence, reverse=True)[:top_n]:
                print(f"   {s.ticker}: {s.confidence:.1f}% confianza, {fmt_pct(s.upside_potential / 100)} potencial")
        
        print(f"\n🔴 VENTAS: {len(ventas)}")
        if ventas:
            for s in sorted(ventas, key=lambda x: x.confidence, reverse=True)[:top_n]:
                print(f"   {s.ticker}: {s.confidence:.1f}% confianza")
        
        print(f"\n🟡 MANTENER: {len(mantener)}")
        print("=" * w)

    def print_top_opportunities(self, signals: List, top_n: int = None) -> None:
        """Imprime top oportunidades de compra."""
        if not self.sections.top_opportunities:
            return
        
        if top_n is None:
            top_n = self.reporting_cfg['top_opportunities']
        
        compras = [s for s in signals if s.signal == "COMPRA"]
        if not compras:
            print("\n⚠️ No hay oportunidades de compra identificadas")
            return
        
        w = self.config.line_width
        print("\n" + "=" * w)
        print(f"TOP {top_n} OPORTUNIDADES DE COMPRA")
        print("=" * w)

        top_compras = sorted(
            compras, 
            key=lambda x: (x.confidence, x.upside_potential), 
            reverse=True
        )[:top_n]
        
        max_reasons = self.reporting_cfg['max_reasons_display']
        
        for i, s in enumerate(top_compras, 1):
            print(f"\n{i}. {s.ticker}")
            print(f"   Confianza: {s.confidence:.1f}%")
            print(f"   Potencial: {fmt_pct(s.upside_potential / 100)}")
            print(f"   Precio: {fmt_money(s.current_price)} → {fmt_money(s.price_target)}")
            print(f"   Valoración: {s.valuation_score:.1f} | Fundamental: {s.fundamental_score:.1f}")
            if s.reasons:
                print(f"   Razones: {', '.join(s.reasons[:max_reasons])}")

    def to_dataframe(self, signals: List) -> pd.DataFrame:
        """Convierte señales a DataFrame."""
        return pd.DataFrame([{
            'Ticker': s.ticker,
            'Señal': s.signal,
            'Confianza': f"{s.confidence:.1f}%",
            'Valoración': f"{s.valuation_score:.1f}",
            'Fundamental': f"{s.fundamental_score:.1f}",
            'Precio Actual': fmt_money(s.current_price),
            'Precio Objetivo': fmt_money(s.price_target),
            'Potencial': fmt_pct(s.upside_potential / 100)
        } for s in signals])
    
    def _get_signal_emoji(self, signal: str) -> str:
        """Obtiene emoji para la señal."""
        mapping = {
            'COMPRA': '🟢',
            'VENTA': '🔴',
            'MANTENER': '🟡'
        }
        return mapping.get(signal, '⚪')