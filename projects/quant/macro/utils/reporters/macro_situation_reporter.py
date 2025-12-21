import numpy as np
from typing import Dict
from ..analyzers.macro_situation_analyzer import MacroSituationAnalyzer


class MacroSituationReporter:
    
    def __init__(self, analyzer: MacroSituationAnalyzer = None):
        self.analyzer = analyzer if analyzer is not None else MacroSituationAnalyzer()
    
    def print_situation(self, analysis: Dict) -> None:
        print("SITUACIÓN MACROECONÓMICA GLOBAL".center(80))
        summary = self.analyzer.get_summary(analysis)
        self._print_executive_summary(summary)
        self._print_yield_curve(analysis['yield_curve'])
        self._print_inflation(analysis['inflation'])
        self._print_credit(analysis['credit'])
        self._print_global_bonds(analysis['global_bonds'])
        self._print_risk_sentiment(analysis['risk_sentiment'])
    
    def _print_executive_summary(self, summary: Dict) -> None:
        print("RESUMEN EJECUTIVO")

        risk_level = summary.get('overall_risk', 'N/A')

        if risk_level == "ALTO":
            risk_display = "🔴 ALTO"
        elif risk_level == "MODERADO":
            risk_display = "🟡 MODERADO"
        else:
            risk_display = "🟢 BAJO"
        
        print(f"\n  Nivel de riesgo global: {risk_display}")
        
        risk_factors = summary.get('risk_factors', [])
        if risk_factors:
            print(f"\n  Factores de riesgo detectados:")
            for factor in risk_factors:
                print(f"    • {factor}")
        else:
            print(f"\n  ✅ No se detectan factores de riesgo significativos")
    
    def _print_yield_curve(self, curve: Dict) -> None:
        print("CURVA DE TIPOS DE INTERÉS (USA)")

        levels = curve.get('levels', {})
        if levels:
            print("\n  Niveles actuales:")
            for tenor, rate in sorted(levels.items()):
                print(f"    {tenor:>4}: {rate:>6.2f}%")

        spreads = curve.get('spreads', {})
        if spreads:
            print("\n  Spreads:")
            for spread_name, value in spreads.items():
                symbol = "🔴" if value < 0 else "🟢"
                print(f"    {symbol} {spread_name:>10}: {value:>+6.2f} pp")

        interpretation = curve.get('interpretation', 'N/A')
        print(f"\n  💡 {interpretation}")
    
    def _print_inflation(self, inflation: Dict) -> None:
        print("SEÑALES DE INFLACIÓN (Commodities)")

        commodities = [
            ('gold', 'Oro'),
            ('silver', 'Plata'),
            ('oil', 'Petróleo'),
            ('copper', 'Cobre'),
            ('wheat', 'Trigo'),
            ('corn', 'Maíz')
        ]
        
        print("\n  Cambio últimos 12 meses:")
        for key, name in commodities:
            change_key = f'{key}_change_1y'
            if change_key in inflation:
                change = inflation[change_key]
                symbol = "🔴" if change > 15 else "🟡" if change > 5 else "🟢"
                print(f"    {symbol} {name:<12}: {change:>+7.2f}%")

        pressure = inflation.get('inflation_pressure', 'N/A')
        avg_change = inflation.get('avg_commodity_change', np.nan)
        
        print(f"\n  💡 {pressure}")
        if not np.isnan(avg_change):
            print(f"     Cambio promedio: {avg_change:>+.2f}%")
    
    def _print_credit(self, credit: Dict) -> None:
        print("CONDICIONES DE CRÉDITO Y VOLATILIDAD")

        if 'vix_level' in credit:
            vix = credit['vix_level']
            vix_symbol = "🔴" if vix > 30 else "🟡" if vix > 20 else "🟢"
            print(f"\n  {vix_symbol} VIX (volatilidad):      {vix:>7.2f}")

        condition = credit.get('market_condition', 'N/A')
        print(f"\n  💡 {condition}")

        if 'hyg_level' in credit and 'lqd_level' in credit:
            print(f"\n  Niveles ETFs crédito:")
            print(f"    HYG (High Yield):        ${credit['hyg_level']:>7.2f}")
            print(f"    LQD (Investment Grade):  ${credit['lqd_level']:>7.2f}")
    
    def _print_global_bonds(self, bonds: Dict) -> None:
        print("BONOS SOBERANOS GLOBALES")

        if not bonds:
            print("\n  ⚠️  Sin datos de bonos globales")
            return
        
        print("{'Región':<15} {'Nivel':<12} {'1 Mes':<12} {'1 Año':<12}")

        for region, data in sorted(bonds.items()):
            level = data.get('level', np.nan)
            change_1m = data.get('change_1m', np.nan)
            change_1y = data.get('change_1y', np.nan)

            symbol_1m = (
                "🔴" if not np.isnan(change_1m) and change_1m < -5 else
                "🟢" if not np.isnan(change_1m) and change_1m > 5 else
                "⚪"
            )
            symbol_1y = (
                "🔴" if not np.isnan(change_1y) and change_1y < -10 else
                "🟢" if not np.isnan(change_1y) and change_1y > 10 else
                "⚪"
            )
            
            level_str = f"${level:.2f}" if not np.isnan(level) else "N/A"
            change_1m_str = f"{symbol_1m} {change_1m:>+6.2f}%" if not np.isnan(change_1m) else "N/A"
            change_1y_str = f"{symbol_1y} {change_1y:>+6.2f}%" if not np.isnan(change_1y) else "N/A"
            
            print(f"  {region:<15} {level_str:<12} {change_1m_str:<12} {change_1y_str:<12}")
    
    def _print_risk_sentiment(self, sentiment: Dict) -> None:
        print("SENTIMIENTO DE RIESGO")

        if 'fear_level' in sentiment:
            fear = sentiment['fear_level']
            print(f"\n  Nivel de miedo:           {fear}")

        if 'dollar_strength' in sentiment:
            dollar = sentiment['dollar_strength']

            trends = []
            if 'dxy_trend_1w' in sentiment:
                trends.append(f"1 sem: {sentiment['dxy_trend_1w']:+.2f}%")
            if 'dxy_trend_1m' in sentiment:
                trends.append(f"1 mes: {sentiment['dxy_trend_1m']:+.2f}%")
            if 'dxy_trend_3m' in sentiment:
                trends.append(f"3 mes: {sentiment['dxy_trend_3m']:+.2f}%")
            
            trend_str = f"({', '.join(trends)})" if trends else ""
            print(f"  Fortaleza del dólar:      {dollar} {trend_str}")

        if 'safe_haven' in sentiment:
            safe_haven = sentiment['safe_haven']
            
            trends = []
            if 'gold_trend_1w' in sentiment:
                trends.append(f"1 sem: {sentiment['gold_trend_1w']:+.2f}%")
            if 'gold_trend_1m' in sentiment:
                trends.append(f"1 mes: {sentiment['gold_trend_1m']:+.2f}%")
            if 'gold_trend_3m' in sentiment:
                trends.append(f"3 mes: {sentiment['gold_trend_3m']:+.2f}%")
            
            trend_str = f"({', '.join(trends)})" if trends else ""
            print(f"  Demanda de refugio:       {safe_haven} {trend_str}")

    def print_compact(self, analysis: Dict) -> None:
        print("SNAPSHOT MACRO".center(60))

        curve = analysis['yield_curve']
        spreads = curve.get('spreads', {})
        if '10Y-2Y' in spreads:
            spread = spreads['10Y-2Y']
            symbol = "🔴" if spread < 0 else "🟢"
            print(f"\n  {symbol} Curva 10Y-2Y: {spread:+.2f} pp")

        inflation = analysis['inflation']
        pressure = inflation.get('inflation_pressure', 'N/A')
        print(f"Inflación: {pressure}")
        
        credit = analysis['credit']
        if 'vix_level' in credit:
            vix = credit['vix_level']
            symbol = "🔴" if vix > 25 else "🟡" if vix > 20 else "🟢"
            print(f"  {symbol} VIX: {vix:.1f}")

        summary = self.analyzer.get_summary(analysis)
        risk = summary.get('overall_risk', 'N/A')
        print(f"\n  ⚠️  Riesgo global: {risk}")
        
        print("="*60)