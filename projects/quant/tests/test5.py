"""
Módulo de Análisis de Valoración de Stocks (v2)
===============================================

Mejoras clave respecto a la versión original:
- Fallbacks robustos para precio actual (fast_info, history).
- Cálculo de 52 semanas con media real (Close.mean()) y protecciones /0.
- RSI con suavizado de Wilder (opcional) y manejo de NaN.
- Sistema de scoring por-factor (se guardan contribuciones por métrica).
- Umbrales añadidos para P/S, EV/EBITDA, D/E y margen neto.
- Evita tratar faltantes como 0 (usa NaN) y excluye del score si no hay dato.
- Limpieza de redundancias y correcciones varias.

Autor: Sistema de Análisis Cuantitativo
Fecha: 2025
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# Estilo para gráficos (opcional)
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


# ------------------------------- Utilidades ---------------------------------

def _nan_if_missing(x):
    return np.nan if (x is None or (isinstance(x, float) and math.isnan(x))) else x


def _safe_div(numer, denom):
    try:
        if denom is None or numer is None:
            return np.nan
        denom = float(denom)
        numer = float(numer)
        return np.nan if denom == 0 else numer / denom
    except Exception:
        return np.nan


def _compute_rsi(series: pd.Series, window: int = 14, method: str = "wilder") -> pd.Series:
    """RSI clásico con opción de suavizado de Wilder.
    Devuelve una serie con el RSI (0-100). Si faltan datos suficientes, devuelve NaN.
    """
    if series is None or series.empty:
        return pd.Series(dtype=float)
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    if method.lower() == "wilder":
        # Suavizado exponencial con alpha = 1/window (Wilder)
        roll_up = up.ewm(alpha=1/window, adjust=False).mean()
        roll_down = down.ewm(alpha=1/window, adjust=False).mean()
    else:
        roll_up = up.rolling(window=window, min_periods=window).mean()
        roll_down = down.rolling(window=window, min_periods=window).mean()

    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi


@dataclass
class ResultadoAnalisis:
    ticker: str
    score: float
    conclusion: str
    color: str
    factores: List[str]
    datos_fundamentales: Dict[str, float]
    metricas_valoracion: Dict[str, float]
    contribuciones: Dict[str, float]


# -------------------------- Clase principal v2 -------------------------------

class AnalizadorValoracion:
    """Analiza la valoración de un stock con múltiples métricas fundamentales."""

    def __init__(self, ticker: str, periodo: str = '1y', rsi_method: str = 'wilder'):
        self.ticker = ticker.upper()
        self.periodo = periodo
        self.rsi_method = rsi_method
        self.stock: Optional[yf.Ticker] = None
        self.datos_fundamentales: Dict[str, float] = {}
        self.metricas_valoracion: Dict[str, float] = {}

    # ---------------------------- Datos base ---------------------------------

    def _precio_actual_fallback(self) -> Optional[float]:
        """Intenta obtener el precio actual con varias fuentes."""
        try:
            # 1) fast_info (rápido y suele ser fiable)
            fi = getattr(self.stock, 'fast_info', None)
            if fi and 'last_price' in fi and fi['last_price']:
                return float(fi['last_price'])
        except Exception:
            pass
        try:
            # 2) regularMarketPrice en info
            info = getattr(self.stock, 'info', {}) or {}
            p = info.get('regularMarketPrice') or info.get('currentPrice')
            if p:
                return float(p)
        except Exception:
            pass
        try:
            # 3) Último cierre del histórico reciente
            hist = self.stock.history(period='5d')
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
        except Exception:
            pass
        return None

    def obtener_datos(self) -> bool:
        """Obtiene info de la empresa y guarda datos fundamentales clave."""
        try:
            self.stock = yf.Ticker(self.ticker)
            info = {}
            try:
                info = self.stock.info or {}
            except Exception:
                info = {}

            precio = self._precio_actual_fallback()

            # Usar NaN para faltantes (no 0) y evitar sesgos en el score
            self.datos_fundamentales = {
                'precio_actual': _nan_if_missing(precio),
                'market_cap': _nan_if_missing(info.get('marketCap')),
                'shares_outstanding': _nan_if_missing(info.get('sharesOutstanding')),
                'pe_ratio': _nan_if_missing(info.get('trailingPE')),
                'forward_pe': _nan_if_missing(info.get('forwardPE')),
                'peg_ratio': _nan_if_missing(info.get('pegRatio')),
                'pb_ratio': _nan_if_missing(info.get('priceToBook')),
                'ps_ratio': _nan_if_missing(info.get('priceToSalesTrailing12Months')),
                'ev_to_ebitda': _nan_if_missing(info.get('enterpriseToEbitda')),
                'ev_to_revenue': _nan_if_missing(info.get('enterpriseToRevenue')),
                'debt_to_equity': _nan_if_missing(info.get('debtToEquity')),
                'return_on_equity': _nan_if_missing(info.get('returnOnEquity')),
                'return_on_assets': _nan_if_missing(info.get('returnOnAssets')),
                'gross_margin': _nan_if_missing(info.get('grossMargins')),
                'operating_margin': _nan_if_missing(info.get('operatingMargins')),
                'profit_margin': _nan_if_missing(info.get('profitMargins')),
                'revenue_growth': _nan_if_missing(info.get('revenueGrowth')),
                'earnings_growth': _nan_if_missing(info.get('earningsGrowth')),
                'dividend_yield': _nan_if_missing(info.get('dividendYield')),
                'beta': _nan_if_missing(info.get('beta')),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A')
            }
            return True
        except Exception as e:
            print(f"❌ Error al obtener datos para {self.ticker}: {e}")
            return False

    # -------------------------- Métricas extra --------------------------------

    def calcular_metricas_valoracion(self) -> bool:
        """Calcula métricas como 52w, volatilidad y RSI."""
        if not self.datos_fundamentales:
            print("❌ Primero debe obtener los datos del stock")
            return False
        try:
            # Histórico para 52 semanas siempre de 1 año
            hist_1y = self.stock.history(period='1y', interval='1d')
            if hist_1y.empty:
                print("❌ No hay histórico de 1y disponible")
                return False

            # Histórico para volatilidad según periodo solicitado
            hist = self.stock.history(period=self.periodo, interval='1d')
            if hist.empty:
                hist = hist_1y.copy()

            # 52w metrics
            precio_52w_high = float(hist_1y['High'].max())
            precio_52w_low = float(hist_1y['Low'].min())
            precio_52w_mean = float(hist_1y['Close'].mean())  # media real

            # Volatilidad anualizada (std de retornos diarios)
            retornos = hist['Close'].pct_change().dropna()
            volatilidad = float(retornos.std() * np.sqrt(252)) if not retornos.empty else np.nan

            # RSI
            rsi_series = _compute_rsi(hist['Close'], window=14, method=self.rsi_method)
            rsi_actual = float(rsi_series.iloc[-1]) if not rsi_series.empty else np.nan
            if np.isnan(rsi_actual):
                rsi_actual = 50.0

            precio_actual = self.datos_fundamentales.get('precio_actual')
            vs_high = _safe_div(precio_actual, precio_52w_high)
            vs_low = _safe_div(precio_actual, precio_52w_low)
            vs_mean = _safe_div(precio_actual, precio_52w_mean)

            self.metricas_valoracion = {
                'precio_52w_high': precio_52w_high,
                'precio_52w_low': precio_52w_low,
                'precio_52w_mean': precio_52w_mean,
                'volatilidad': volatilidad,
                'rsi': rsi_actual,
                'precio_vs_52w_high_pct': float(vs_high * 100) if not np.isnan(vs_high) else np.nan,
                'precio_vs_52w_low_pct': float(vs_low * 100) if not np.isnan(vs_low) else np.nan,
                'precio_vs_52w_mean_pct': float(vs_mean * 100) if not np.isnan(vs_mean) else np.nan,
            }
            return True
        except Exception as e:
            print(f"❌ Error al calcular métricas: {e}")
            return False

    # ------------------------------ Scoring -----------------------------------

    def _add_contrib(self, contrib: Dict[str, float], key: str, value: float, factores: List[str], msg_pos: str, msg_neg: str):
        contrib[key] = contrib.get(key, 0) + value
        if value > 0:
            factores.append(f"✅ {msg_pos}")
        elif value < 0:
            factores.append(f"❌ {msg_neg}")

    def analizar_valoracion(self) -> Optional[ResultadoAnalisis]:
        if not self.obtener_datos():
            return None
        if not self.calcular_metricas_valoracion():
            return None

        factores: List[str] = []
        contrib: Dict[str, float] = {}

        df = self.datos_fundamentales
        mv = self.metricas_valoracion

        # 1) P/E
        pe = df.get('pe_ratio')
        if pe and pe > 0 and np.isfinite(pe):
            if pe < 15:
                self._add_contrib(contrib, 'P/E', 2, factores, 'P/E bajo (subvaluado)', 'P/E muy alto (sobrevaluado)')
            elif pe < 20:
                self._add_contrib(contrib, 'P/E', 1, factores, 'P/E moderado', 'P/E alto')
            elif pe < 30:
                self._add_contrib(contrib, 'P/E', -1, factores, 'P/E moderado', 'P/E alto')
            else:
                self._add_contrib(contrib, 'P/E', -2, factores, 'P/E moderado', 'P/E muy alto (sobrevaluado)')

        # 2) P/B
        pb = df.get('pb_ratio')
        if pb and pb > 0 and np.isfinite(pb):
            if pb < 1:
                self._add_contrib(contrib, 'P/B', 2, factores, 'P/B < 1 (subvaluado)', 'P/B muy alto')
            elif pb < 2:
                self._add_contrib(contrib, 'P/B', 1, factores, 'P/B moderado', 'P/B alto')
            elif pb < 3:
                self._add_contrib(contrib, 'P/B', -1, factores, 'P/B moderado', 'P/B alto')
            else:
                self._add_contrib(contrib, 'P/B', -2, factores, 'P/B moderado', 'P/B muy alto')

        # 3) PEG
        peg = df.get('peg_ratio')
        if peg and peg > 0 and np.isfinite(peg):
            if peg < 1:
                self._add_contrib(contrib, 'PEG', 2, factores, 'PEG < 1 (subvaluado)', 'PEG alto')
            elif peg < 1.5:
                self._add_contrib(contrib, 'PEG', 1, factores, 'PEG moderado', 'PEG alto')
            else:
                self._add_contrib(contrib, 'PEG', -1, factores, 'PEG moderado', 'PEG alto')

        # 4) P/S
        ps = df.get('ps_ratio')
        if ps and ps > 0 and np.isfinite(ps):
            if ps < 2:
                self._add_contrib(contrib, 'P/S', 1, factores, 'P/S atractivo (<2)', 'P/S muy alto (>8)')
            elif ps > 8:
                self._add_contrib(contrib, 'P/S', -1, factores, 'P/S atractivo (<2)', 'P/S muy alto (>8)')

        # 5) EV/EBITDA
        ev_e = df.get('ev_to_ebitda')
        if ev_e and ev_e > 0 and np.isfinite(ev_e):
            if ev_e < 10:
                self._add_contrib(contrib, 'EV/EBITDA', 1, factores, 'EV/EBITDA < 10', 'EV/EBITDA > 20')
            elif ev_e > 20:
                self._add_contrib(contrib, 'EV/EBITDA', -1, factores, 'EV/EBITDA < 10', 'EV/EBITDA > 20')

        # 6) Posición vs 52w mean
        pos_52w = mv.get('precio_vs_52w_mean_pct')
        if pos_52w and np.isfinite(pos_52w):
            if pos_52w < 80:
                self._add_contrib(contrib, '52w', 1, factores, 'Precio < 80% de la media 52w', 'Precio > 120% de la media 52w')
            elif pos_52w > 120:
                self._add_contrib(contrib, '52w', -1, factores, 'Precio < 80% de la media 52w', 'Precio > 120% de la media 52w')

        # 7) RSI
        rsi = mv.get('rsi')
        if rsi and np.isfinite(rsi):
            if rsi < 30:
                self._add_contrib(contrib, 'RSI', 1, factores, 'RSI en sobreventa (<30)', 'RSI en sobrecompra (>70)')
            elif rsi > 70:
                self._add_contrib(contrib, 'RSI', -1, factores, 'RSI en sobreventa (<30)', 'RSI en sobrecompra (>70)')

        # 8) Crecimiento de ganancias
        eg = df.get('earnings_growth')
        if eg is not None and np.isfinite(eg):
            if eg > 0.10:
                self._add_contrib(contrib, 'Crecimiento', 1, factores, 'Crecimiento de ganancias > 10%', 'Crecimiento de ganancias negativo')
            elif eg < 0:
                self._add_contrib(contrib, 'Crecimiento', -1, factores, 'Crecimiento de ganancias > 10%', 'Crecimiento de ganancias negativo')

        # 9) Rentabilidad (ROE)
        roe = df.get('return_on_equity')
        if roe is not None and np.isfinite(roe):
            if roe > 0.15:
                self._add_contrib(contrib, 'ROE', 1, factores, 'ROE alto (>15%)', 'ROE bajo (<5%)')
            elif roe < 0.05:
                self._add_contrib(contrib, 'ROE', -1, factores, 'ROE alto (>15%)', 'ROE bajo (<5%)')

        # 10) Apalancamiento (D/E)
        de = df.get('debt_to_equity')
        if de is not None and np.isfinite(de) and de >= 0:
            if de < 100:  # yfinance suele dar D/E en % (p.ej. 50 == 0.5)
                self._add_contrib(contrib, 'D/E', 1, factores, 'Deuda/Equity baja', 'Deuda/Equity alta')
            elif de > 200:
                self._add_contrib(contrib, 'D/E', -1, factores, 'Deuda/Equity baja', 'Deuda/Equity alta')

        # 11) Margen Neto
        pm = df.get('profit_margin')
        if pm is not None and np.isfinite(pm):
            if pm > 0.10:
                self._add_contrib(contrib, 'Margen', 1, factores, 'Margen neto saludable (>10%)', 'Margen neto negativo')
            elif pm < 0:
                self._add_contrib(contrib, 'Margen', -1, factores, 'Margen neto saludable (>10%)', 'Margen neto negativo')

        total_score = float(sum(contrib.values()))

        if total_score >= 4:
            conclusion, color = "INFRAVALUADO", "🟢"
        elif total_score >= 1:
            conclusion, color = "VALORACIÓN JUSTA", "🟡"
        else:
            conclusion, color = "SOBREVALUADO", "🔴"

        return ResultadoAnalisis(
            ticker=self.ticker,
            score=total_score,
            conclusion=conclusion,
            color=color,
            factores=factores,
            datos_fundamentales=self.datos_fundamentales,
            metricas_valoracion=self.metricas_valoracion,
            contribuciones=contrib,
        )

    # ------------------------------ Reportes ----------------------------------

    def generar_reporte(self, analisis: ResultadoAnalisis) -> None:
        if not analisis:
            return
        print(f"\n{'='*60}")
        print(f"📊 REPORTE DE VALORACIÓN - {analisis.ticker}")
        print(f"{'='*60}")

        df = analisis.datos_fundamentales
        mv = analisis.metricas_valoracion

        # Información básica
        precio_actual = df['precio_actual']
        precio_str = f"${precio_actual:.2f}" if pd.notna(precio_actual) else "N/D"
        mcap = df['market_cap']
        mcap_str = f"${mcap:,.0f}" if pd.notna(mcap) else "N/D"

        print(f"\n📈 INFORMACIÓN BÁSICA:")
        print(f"   Precio Actual: {precio_str}")
        print(f"   Market Cap: {mcap_str}")
        print(f"   Sector: {df.get('sector')}")
        print(f"   Industria: {df.get('industry')}")

        # Métricas de valoración
        def _fmt(x):
            return f"{x:.2f}" if pd.notna(x) else "N/D"
        print(f"\n💰 MÉTRICAS DE VALORACIÓN:")
        print(f"   P/E Ratio: {_fmt(df.get('pe_ratio'))}")
        print(f"   P/B Ratio: {_fmt(df.get('pb_ratio'))}")
        print(f"   PEG Ratio: {_fmt(df.get('peg_ratio'))}")
        print(f"   P/S Ratio: {_fmt(df.get('ps_ratio'))}")
        print(f"   EV/EBITDA: {_fmt(df.get('ev_to_ebitda'))}")

        # Rentabilidad
        print(f"\n📊 RENTABILIDAD:")
        for label, key in [
            ("ROE", 'return_on_equity'),
            ("ROA", 'return_on_assets'),
            ("Margen Bruto", 'gross_margin'),
            ("Margen Operativo", 'operating_margin'),
            ("Margen Neto", 'profit_margin'),
        ]:
            val = df.get(key)
            s = f"{val:.2%}" if pd.notna(val) else "N/D"
            print(f"   {label}: {s}")

        # Técnico
        print(f"\n📈 ANÁLISIS TÉCNICO:")
        rsi = mv.get('rsi')
        vol = mv.get('volatilidad')
        vs = mv.get('precio_vs_52w_mean_pct')
        print(f"   RSI: {rsi:.1f}" if pd.notna(rsi) else "   RSI: N/D")
        print(f"   Volatilidad: {vol:.1%}" if pd.notna(vol) else "   Volatilidad: N/D")
        print(f"   Precio vs Media 52w: {vs:.1f}%" if pd.notna(vs) else "   Precio vs Media 52w: N/D")

        # Factores
        print(f"\n🔍 FACTORES DE VALORACIÓN:")
        if analisis.factores:
            for factor in analisis.factores:
                print(f"   {factor}")
        else:
            print("   (Sin factores: faltan datos confiables)")

        print(f"\n🎯 CONCLUSIÓN:\n   {analisis.color} {analisis.conclusion} (Score: {analisis.score:.1f})")
        print(f"\n{'='*60}")

    def crear_graficos_valoracion(self, analisis: ResultadoAnalisis) -> None:
        if not analisis:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Análisis de Valoración - {analisis.ticker}', fontsize=16, fontweight='bold')

        # 1) Ratios clave
        metricas = ['P/E', 'P/B', 'P/S', 'PEG']
        valores = [
            analisis.datos_fundamentales.get('pe_ratio'),
            analisis.datos_fundamentales.get('pb_ratio'),
            analisis.datos_fundamentales.get('ps_ratio'),
            analisis.datos_fundamentales.get('peg_ratio'),
        ]
        m_validas, v_validos = [], []
        for m, v in zip(metricas, valores):
            if v and v > 0 and not np.isnan(v):
                m_validas.append(m)
                v_validos.append(v)
        if m_validas:
            axes[0, 0].bar(m_validas, v_validos)
            axes[0, 0].set_title('Ratios de Valoración', fontweight='bold')
            axes[0, 0].set_ylabel('Valor')
            axes[0, 0].tick_params(axis='x', rotation=45)

        # 2) Rentabilidad
        rent_labels = ['ROE', 'ROA', 'Margen Bruto', 'Margen Operativo', 'Margen Neto']
        rent_keys = ['return_on_equity', 'return_on_assets', 'gross_margin', 'operating_margin', 'profit_margin']
        r_labels, r_vals = [], []
        for label, key in zip(rent_labels, rent_keys):
            v = analisis.datos_fundamentales.get(key)
            if v is not None and pd.notna(v):
                r_labels.append(label)
                r_vals.append(v * 100)
        if r_labels:
            axes[0, 1].bar(r_labels, r_vals)
            axes[0, 1].set_title('Métricas de Rentabilidad (%)', fontweight='bold')
            axes[0, 1].set_ylabel('Porcentaje (%)')
            axes[0, 1].tick_params(axis='x', rotation=45)

        # 3) Posición 52w
        posiciones = ['52w Low', 'Actual', '52w Mean', '52w High']
        precios = [
            analisis.metricas_valoracion.get('precio_52w_low'),
            analisis.datos_fundamentales.get('precio_actual'),
            analisis.metricas_valoracion.get('precio_52w_mean'),
            analisis.metricas_valoracion.get('precio_52w_high'),
        ]
        p_labels, p_vals = [], []
        for lbl, val in zip(posiciones, precios):
            if val is not None and pd.notna(val):
                p_labels.append(lbl)
                p_vals.append(val)
        if p_labels:
            axes[1, 0].bar(p_labels, p_vals)
            axes[1, 0].set_title('Posición del Precio (52 semanas)', fontweight='bold')
            axes[1, 0].set_ylabel('Precio ($)')
            axes[1, 0].tick_params(axis='x', rotation=45)

        # 4) Contribuciones del score
        if analisis.contribuciones:
            items = list(analisis.contribuciones.items())
            items.sort(key=lambda x: x[1])  # ordenar por contribución
            cats = [k for k, _ in items]
            vals = [v for _, v in items]
            axes[1, 1].barh(cats, vals)
            axes[1, 1].set_title('Contribución al Score por Métrica', fontweight='bold')
            axes[1, 1].set_xlabel('Puntos')
            axes[1, 1].axvline(x=0, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()

    # --------------------------- Comparaciones --------------------------------

    def comparar_con_mercado(self, tickers_comparacion: List[str]) -> List[Dict[str, object]]:
        print(f"\n🔄 Comparando {self.ticker} con otros stocks del mercado...")
        comparaciones: List[Dict[str, object]] = []

        # Análisis del actual (una sola vez)
        analisis_actual = self.analizar_valoracion()
        if analisis_actual:
            comparaciones.append({
                'ticker': analisis_actual.ticker,
                'score': analisis_actual.score,
                'pe_ratio': analisis_actual.datos_fundamentales.get('pe_ratio'),
                'pb_ratio': analisis_actual.datos_fundamentales.get('pb_ratio'),
                'conclusion': analisis_actual.conclusion
            })

        for tk in tickers_comparacion:
            tk = tk.upper()
            if tk == self.ticker:
                continue
            try:
                analizador = AnalizadorValoracion(tk, self.periodo, self.rsi_method)
                analisis = analizador.analizar_valoracion()
                if analisis:
                    comparaciones.append({
                        'ticker': analisis.ticker,
                        'score': analisis.score,
                        'pe_ratio': analisis.datos_fundamentales.get('pe_ratio'),
                        'pb_ratio': analisis.datos_fundamentales.get('pb_ratio'),
                        'conclusion': analisis.conclusion
                    })
            except Exception as e:
                print(f"❌ Error analizando {tk}: {e}")

        if comparaciones:
            dfc = pd.DataFrame(comparaciones).sort_values('score', ascending=False)

            print(f"\n📊 COMPARACIÓN CON EL MERCADO:")
            print(f"{'Ticker':<8} {'Score':<6} {'P/E':<8} {'P/B':<8} {'Conclusión':<15}")
            print("-" * 60)
            for _, row in dfc.iterrows():
                pe = row['pe_ratio'] if pd.notna(row['pe_ratio']) else float('nan')
                pb = row['pb_ratio'] if pd.notna(row['pb_ratio']) else float('nan')
                pe_str = f"{pe:.2f}" if pd.notna(pe) else "N/D"
                pb_str = f"{pb:.2f}" if pd.notna(pb) else "N/D"
                print(f"{row['ticker']:<8} {row['score']:<6.1f} {pe_str:<8} {pb_str:<8} {row['conclusion']:<15}")

        return comparaciones


# ------------------------- Funciones de conveniencia -------------------------

def analizar_stock(ticker: str, periodo: str = '1y', mostrar_graficos: bool = True, comparar_con: Optional[List[str]] = None) -> Optional[ResultadoAnalisis]:
    print(f"🚀 Iniciando análisis de valoración para {ticker.upper()}")
    analizador = AnalizadorValoracion(ticker, periodo)
    analisis = analizador.analizar_valoracion()

    if analisis:
        analizador.generar_reporte(analisis)
        if mostrar_graficos:
            analizador.crear_graficos_valoracion(analisis)
        if comparar_con:
            analizador.comparar_con_mercado(comparar_con)
        return analisis
    else:
        print(f"❌ No se pudo completar el análisis para {ticker}")
        return None


def analizar_portafolio(tickers: List[str], periodo: str = '1y') -> Optional[pd.DataFrame]:
    print(f"📊 Analizando portafolio de {len(tickers)} stocks...")
    resultados: List[ResultadoAnalisis] = []

    for tk in tickers:
        print(f"\n{'='*40}\nAnalizando {tk.upper()}\n{'='*40}")
        r = analizar_stock(tk, periodo, mostrar_graficos=False)
        if r:
            resultados.append(r)

    if resultados:
        print(f"\n{'='*60}\n📈 RESUMEN DEL PORTAFOLIO\n{'='*60}")
        df_resumen = pd.DataFrame([
            {
                'Ticker': r.ticker,
                'Score': r.score,
                'P/E': r.datos_fundamentales.get('pe_ratio'),
                'P/B': r.datos_fundamentales.get('pb_ratio'),
                'ROE': r.datos_fundamentales.get('return_on_equity'),
                'Conclusión': r.conclusion
            }
            for r in resultados
        ])
        df_resumen = df_resumen.sort_values('Score', ascending=False)
        pd.options.display.float_format = '{:,.2f}'.format
        print(df_resumen.to_string(index=False))

        print(f"\n📊 ESTADÍSTICAS DEL PORTAFOLIO:")
        print(f"   Promedio Score: {df_resumen['Score'].mean():.2f}")
        print(f"   Stocks Infravaluados: {len(df_resumen[df_resumen['Score'] >= 4])}")
        print(f"   Stocks Sobrevaluados: {len(df_resumen[df_resumen['Score'] < 1])}")
        print(f"   Score Máximo: {df_resumen['Score'].max():.1f}")
        print(f"   Score Mínimo: {df_resumen['Score'].min():.1f}")
        return df_resumen

    return None


# ------------------------------- Ejemplos ------------------------------------
if __name__ == "__main__":
    print("🎯 MÓDULO DE ANÁLISIS DE VALORACIÓN DE STOCKS - v2")
    print("=" * 50)

    # Ejemplo 1: Análisis individual
    print("\n1️⃣ Análisis individual de AAPL:")
    analizar_stock("AAPL", periodo='1y', mostrar_graficos=True)

    # Ejemplo 2: Portafolio tecnológico
    print("\n2️⃣ Análisis de portafolio tecnológico:")
    tickers_tech = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    analizar_portafolio(tickers_tech, periodo='1y')

    # Ejemplo 3: Comparación con competidores
    print("\n3️⃣ Comparación de AAPL con competidores:")
    analizador_aapl = AnalizadorValoracion("AAPL")
    analisis_aapl = analizador_aapl.analizar_valoracion()
    if analisis_aapl:
        analizador_aapl.comparar_con_mercado(["MSFT", "GOOGL", "AMZN"])  
