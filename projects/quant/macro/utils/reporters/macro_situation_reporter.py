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
        if 'implied_yield_curve' in analysis and analysis['implied_yield_curve']:
            self._print_implied_yield_curve(analysis['implied_yield_curve'])
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
    
    def _print_yield_curve(self, curve) -> None:
        print("CURVA DE TIPOS DE INTERÉS (USA)")
        levels = getattr(curve, 'levels', curve.get('levels', {})) if hasattr(curve, 'get') else curve.levels
        if levels:
            print("\n  Niveles actuales:")
            for tenor, rate in sorted(levels.items()):
                print(f"    {tenor:>4}: {rate:>6.2f}%")

        spreads = getattr(curve, 'spreads', curve.get('spreads', {})) if hasattr(curve, 'get') else curve.spreads
        if spreads:
            print("\n  Spreads:")
            for spread_name, value in spreads.items():
                symbol = "🔴" if value < 0 else "🟢"
                print(f"    {symbol} {spread_name:>10}: {value:>+6.2f} pp")

        rate_changes = getattr(curve, 'rate_changes', curve.get('rate_changes', {})) if hasattr(curve, 'get') else curve.rate_changes
        if rate_changes:
            print("\n  Cambios en tasas:")
            print(f"    {'Tenor':<6} {'1 Mes':<12} {'3 Meses':<12} {'1 Año':<12}")
            for tenor in sorted(rate_changes.keys()):
                changes = rate_changes[tenor]
                change_1m = changes.get('1m', np.nan)
                change_3m = changes.get('3m', np.nan)
                change_1y = changes.get('1y', np.nan)
                
                change_1m_str = f"{change_1m:>+6.2f} pp" if not np.isnan(change_1m) else "N/A"
                change_3m_str = f"{change_3m:>+6.2f} pp" if not np.isnan(change_3m) else "N/A"
                change_1y_str = f"{change_1y:>+6.2f} pp" if not np.isnan(change_1y) else "N/A"
                
                print(f"    {tenor:<6} {change_1m_str:<12} {change_3m_str:<12} {change_1y_str:<12}")

        divergence = getattr(curve, 'divergence_analysis', curve.get('divergence_analysis', {})) if hasattr(curve, 'get') else curve.divergence_analysis
        if divergence:
            print("\n  📊 Divergencia Corto vs Largo Plazo:")
            if '3m' in divergence:
                d3m = divergence['3m']
                print(f"    3 meses:")
                print(f"      Corto (2Y): {d3m['short']:>+6.2f} pp")
                print(f"      Largo (10Y): {d3m['long']:>+6.2f} pp")
                print(f"      Divergencia: {d3m['divergence']:>+6.2f} pp", end="")
                if d3m['divergence'] > 0.5:
                    print(" 🔴 (Largo sube más que corto - expectativas inflacionarias)")
                elif d3m['divergence'] < -0.5:
                    print(" 🟢 (Corto sube más que largo - política restrictiva)")
                else:
                    print(" ⚪ (Movimientos alineados)")
            
            if '1y' in divergence:
                d1y = divergence['1y']
                print(f"    1 año:")
                print(f"      Corto (2Y): {d1y['short']:>+6.2f} pp")
                print(f"      Largo (10Y): {d1y['long']:>+6.2f} pp")
                print(f"      Divergencia: {d1y['divergence']:>+6.2f} pp", end="")
                if d1y['divergence'] > 0.5:
                    print(" 🔴 (Largo sube más que corto - expectativas inflacionarias)")
                elif d1y['divergence'] < -0.5:
                    print(" 🟢 (Corto sube más que largo - política restrictiva)")
                else:
                    print(" ⚪ (Movimientos alineados)")

        interpretation = getattr(curve, 'interpretation', curve.get('interpretation', 'N/A')) if hasattr(curve, 'get') else curve.interpretation
        print(f"\n  💡 {interpretation}")
    
    def _print_implied_yield_curve(self, implied_analysis) -> None:
        print("CURVA DE YIELD IMPLÍCITA (FORWARD RATES)")
        
        spot = getattr(implied_analysis, 'spot_rates', {})
        forwards = getattr(implied_analysis, 'forward_rates', {})
        fwd_vs_spot = getattr(implied_analysis, 'forward_vs_spot', {})
        term_premium = getattr(implied_analysis, 'term_premium', {})

        print("\n  📈 Curva Spot vs Forward Implícito:")
        print(f"    {'Tramo':<14} {'Forward (%)':<14} {'vs Spot':<14} {'Señal'}")
        print(f"    {'─'*56}")
        
        for tramo, fwd_rate in forwards.items():
            diff = fwd_vs_spot.get(tramo, np.nan)
            if not np.isnan(fwd_rate):
                if not np.isnan(diff):
                    symbol = "🔴↑" if diff > 0.3 else "🟢↓" if diff < -0.3 else "⚪→"
                    print(f"    {tramo:<14} {fwd_rate:>6.2f}%       {diff:>+6.2f} pp     {symbol}")
                else:
                    print(f"    {tramo:<14} {fwd_rate:>6.2f}%")

        if term_premium:
            print(f"\n  💰 Term Premium Estimado:")
            for tenor, tp in term_premium.items():
                symbol = "🔴" if tp < -0.2 else "🟢" if tp > 0.5 else "⚪"
                print(f"    {symbol} {tenor}: {tp:>+6.2f} pp")

        expectations = getattr(implied_analysis, 'curve_expectations', 'N/A')
        signal = getattr(implied_analysis, 'rate_path_signal', 'N/A')
        print(f"\n  💡 Expectativas: {expectations}")
        print(f"  🏦 Señal política monetaria: {signal}")

    def _print_inflation(self, inflation) -> None:
        print("SEÑALES DE INFLACIÓN (Commodities)")
        commodity_changes = getattr(inflation, 'commodity_changes', inflation.get('commodity_changes', {})) if hasattr(inflation, 'get') else inflation.commodity_changes
        commodity_names = getattr(inflation, 'commodity_names', inflation.get('commodity_names', {})) if hasattr(inflation, 'get') else inflation.commodity_names
        
        print("\n  Cambio últimos 12 meses:")
        for key, name in commodity_names.items():
            if key in commodity_changes:
                change = commodity_changes[key]
                symbol = "🔴" if change > 15 else "🟡" if change > 5 else "🟢"
                print(f"    {symbol} {name:<12}: {change:>+7.2f}%")

        pressure = getattr(inflation, 'inflation_pressure', inflation.get('inflation_pressure', 'N/A')) if hasattr(inflation, 'get') else inflation.inflation_pressure
        avg_change = getattr(inflation, 'avg_commodity_change', inflation.get('avg_commodity_change', np.nan)) if hasattr(inflation, 'get') else inflation.avg_commodity_change
        
        print(f"\n  💡 {pressure}")
        if not np.isnan(avg_change):
            print(f"     Cambio promedio: {avg_change:>+.2f}%")
    
    def _print_credit(self, credit) -> None:
        print("CONDICIONES DE CRÉDITO Y VOLATILIDAD")
        vix = getattr(credit, 'vix_level', credit.get('vix_level')) if hasattr(credit, 'get') else credit.vix_level
        if vix is not None:
            vix_symbol = "🔴" if vix > 30 else "🟡" if vix > 20 else "🟢"
            print(f"\n  {vix_symbol} VIX (volatilidad):      {vix:>7.2f}")

        condition = getattr(credit, 'market_condition', credit.get('market_condition', 'N/A')) if hasattr(credit, 'get') else credit.market_condition
        print(f"\n  💡 {condition}")

        hyg = getattr(credit, 'hyg_level', credit.get('hyg_level')) if hasattr(credit, 'get') else credit.hyg_level
        lqd = getattr(credit, 'lqd_level', credit.get('lqd_level')) if hasattr(credit, 'get') else credit.lqd_level
        if hyg is not None and lqd is not None:
            print(f"\n  Niveles ETFs crédito:")
            print(f"    HYG (High Yield):        ${hyg:>7.2f}")
            print(f"    LQD (Investment Grade):  ${lqd:>7.2f}")
    
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
    
    def _print_risk_sentiment(self, sentiment) -> None:
        print("SENTIMIENTO DE RIESGO")
        fear = getattr(sentiment, 'fear_level', sentiment.get('fear_level')) if hasattr(sentiment, 'get') else sentiment.fear_level

        if fear:
            print(f"\n  Nivel de miedo:           {fear}")

        dollar = getattr(sentiment, 'dollar_strength', sentiment.get('dollar_strength')) if hasattr(sentiment, 'get') else sentiment.dollar_strength

        if dollar:
            trends = []
            dxy_1w = getattr(sentiment, 'dxy_trend_1w', sentiment.get('dxy_trend_1w')) if hasattr(sentiment, 'get') else sentiment.dxy_trend_1w
            dxy_1m = getattr(sentiment, 'dxy_trend_1m', sentiment.get('dxy_trend_1m')) if hasattr(sentiment, 'get') else sentiment.dxy_trend_1m
            dxy_3m = getattr(sentiment, 'dxy_trend_3m', sentiment.get('dxy_trend_3m')) if hasattr(sentiment, 'get') else sentiment.dxy_trend_3m
            
            if dxy_1w is not None:
                trends.append(f"1 sem: {dxy_1w:+.2f}%")
            if dxy_1m is not None:
                trends.append(f"1 mes: {dxy_1m:+.2f}%")
            if dxy_3m is not None:
                trends.append(f"3 mes: {dxy_3m:+.2f}%")
            
            trend_str = f"({', '.join(trends)})" if trends else ""
            print(f"  Fortaleza del dólar:      {dollar} {trend_str}")

        safe_haven = getattr(sentiment, 'safe_haven', sentiment.get('safe_haven')) if hasattr(sentiment, 'get') else sentiment.safe_haven
        if safe_haven:
            trends = []
            gold_1w = getattr(sentiment, 'gold_trend_1w', sentiment.get('gold_trend_1w')) if hasattr(sentiment, 'get') else sentiment.gold_trend_1w
            gold_1m = getattr(sentiment, 'gold_trend_1m', sentiment.get('gold_trend_1m')) if hasattr(sentiment, 'get') else sentiment.gold_trend_1m
            gold_3m = getattr(sentiment, 'gold_trend_3m', sentiment.get('gold_trend_3m')) if hasattr(sentiment, 'get') else sentiment.gold_trend_3m
            
            if gold_1w is not None:
                trends.append(f"1 sem: {gold_1w:+.2f}%")
            if gold_1m is not None:
                trends.append(f"1 mes: {gold_1m:+.2f}%")
            if gold_3m is not None:
                trends.append(f"3 mes: {gold_3m:+.2f}%")
            
            trend_str = f"({', '.join(trends)})" if trends else ""
            print(f"  Demanda de refugio:       {safe_haven} {trend_str}")

    def print_compact(self, analysis: Dict) -> None:
        print("SNAPSHOT MACRO".center(60))
        curve = analysis['yield_curve']
        spreads = getattr(curve, 'spreads', curve.get('spreads', {})) if hasattr(curve, 'get') else curve.spreads
        if '10Y-2Y' in spreads:
            spread = spreads['10Y-2Y']
            symbol = "🔴" if spread < 0 else "🟢"
            print(f"\n  {symbol} Curva 10Y-2Y: {spread:+.2f} pp")

        inflation = analysis['inflation']
        pressure = getattr(inflation, 'inflation_pressure', inflation.get('inflation_pressure', 'N/A')) if hasattr(inflation, 'get') else inflation.inflation_pressure
        print(f"Inflación: {pressure}")
        
        credit = analysis['credit']
        vix = getattr(credit, 'vix_level', credit.get('vix_level')) if hasattr(credit, 'get') else credit.vix_level
        if vix is not None:
            symbol = "🔴" if vix > 25 else "🟡" if vix > 20 else "🟢"
            print(f"  {symbol} VIX: {vix:.1f}")

        summary = self.analyzer.get_summary(analysis)
        risk = summary.get('overall_risk', 'N/A')
        print(f"\n  ⚠️  Riesgo global: {risk}")
        
        print("="*60)