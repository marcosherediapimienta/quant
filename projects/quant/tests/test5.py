"""
Módulo de Análisis de Valoración de Stocks (v3, parcheado)
==========================================================

Refactor completo con:
- Métricas relativas por sector/industria (percentiles/z-scores, opcional)
- Incorporación de calidad (ROIC, márgenes), FCF yield, crecimiento y momentum
- Manejo de P/B, ROE y EV/EBITDA extremos con winsorización contra peers
- Reportes reproducibles y trazables con fuentes y fechas
- Sanitización de múltiplos (evitar tratar múltiplos ≤0 como “baratos”)
- Formateo seguro para NaN/valores faltantes
- Conteo correcto de peers disponibles (excluyendo el propio ticker)
- Parche de índice temporal (naive) y búsquedas robustas de precios para evitar “panel vacío”

Autor: Sistema de Análisis Cuantitativo
Fecha: 2025
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# Estilo para gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ================================ CONFIGURACIÓN ================================

# Pesos de los bloques de análisis
WEIGHTS = {
    "valuation": 0.40,  # Valoración (P/E, EV/EBITDA, P/S, FCF yield, P/B)
    "quality": 0.30,    # Calidad (ROIC, márgenes, estabilidad, deuda)
    "growth": 0.20,     # Crecimiento (ventas, EPS YoY y forward)
    "momentum": 0.10    # Momentum (rendimientos 6-12m, RSI)
}

# Configuración de winsorización para métricas extremas
WINSOR = {
    "pb": [0.01, 0.99],   # P/B winsorizado al 1% y 99% (contra peers)
    "roe": [0.01, 0.99],  # ROE winsorizado al 1% y 99% (contra peers)
    "ev_ebitda": [0.01, 0.99]  # EV/EBITDA winsorizado contra peers
}

# Ventanas de momentum
MOMENTUM_WINDOW = {
    "m6": 126,   # 6 meses (aprox. 126 días hábiles)
    "m12": 252,  # 12 meses (aprox. 252 días hábiles)
    "skip": 21   # Saltar último mes (21 días hábiles)
}

# Alcance para comparaciones relativas
RELATIVE_SCOPE = "industry"  # "industry" o "sector"

# Bandas para conclusiones
NEUTRAL_BAND = 0.30  # Banda neutral: [-0.30, 0.30]

# Percentiles para considerar caro/barato (no usado directamente en score, útil para flags)
PCTILES_NEG_HIGH = 0.8  # Percentil 80% = caro
PCTILES_POS_LOW = 0.2   # Percentil 20% = barato

# Configuración de RSI
RSI_WINDOW = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
RSI_MAX_ADJUSTMENT = 0.1  # Ajuste máximo del RSI al score

# Mínimo de peers para análisis relativo
MIN_PEERS = 5

# Fallback si no hay suficientes peers
FALLBACK_SCOPE = {
    "industry": "sector",
    "sector": "universe"
}

# ================================ UTILIDADES ================================

def _nan_if_missing(x):
    """Convierte valores faltantes o inválidos a NaN."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return np.nan
    return x

def _safe_div(numer, denom):
    """División segura que devuelve NaN si el denominador es 0 o inválido."""
    try:
        if denom is None or numer is None:
            return np.nan
        denom = float(denom)
        numer = float(numer)
        return np.nan if denom == 0 else numer / denom
    except Exception:
        return np.nan

def winsorize_against_peers(value: float, peers: List[float], p_low=0.01, p_high=0.99) -> float:
    """Aplica winsorización a un valor basado en percentiles de peers."""
    vals = np.array([v for v in peers if pd.notna(v)], dtype=float)
    if pd.isna(value) or vals.size == 0:
        return value
    lo, hi = np.quantile(vals, [p_low, p_high])
    return float(np.clip(value, lo, hi))

def winsorize(x: float, p_low: float, p_high: float, data: List[float] = None) -> float:
    """
    Winsorización corregida.
    - Si se pasa 'data', calcula los cortes por percentil y recorta 'x' a [p_low, p_high] de 'data'.
    - Si no hay 'data', interpreta p_low/p_high como límites absolutos (solo por compatibilidad).
    """
    if pd.isna(x):
        return x
    try:
        x = float(x)
    except Exception:
        return np.nan
    if data is not None and len(data) > 0:
        vals = np.array([v for v in data if pd.notna(v)], dtype=float)
        if vals.size == 0:
            return x
        lo, hi = np.quantile(vals, [p_low, p_high])
        return float(np.clip(x, lo, hi))
    # Modo absoluto (evitar en producción)
    return float(np.clip(x, p_low, p_high))

def percentile_rank(value: float, peer_values: List[float]) -> float:
    """Calcula percentil rank de un valor respecto a sus peers (0..1)."""
    if pd.isna(value) or not peer_values:
        return np.nan
    valid_values = [v for v in peer_values if pd.notna(v)]
    if not valid_values:
        return np.nan
    n = len(valid_values)
    rank = sum(1 for v in valid_values if v < value) + 0.5 * sum(1 for v in valid_values if v == value)
    return rank / n

def normalize_signed(value: float, higher_is_better: bool = True) -> float:
    """Normaliza un valor a rango [-1, 1] con saturación suave (tanh)."""
    if pd.isna(value):
        return np.nan
    signed_value = value if higher_is_better else -value
    return np.tanh(signed_value)

def _calculate_rsi(prices: np.ndarray, window: int = 14) -> float:
    """Calcula RSI de una serie de precios."""
    if len(prices) < window + 1:
        return np.nan
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gains = np.mean(gains[-window:])
    avg_losses = np.mean(losses[-window:])
    if avg_losses == 0:
        return 100.0
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ---------- Helpers de formateo y sanitizado ----------

def fmt_num(x, fmt="{:,.2f}", na="N/D"):
    try:
        return na if pd.isna(x) or not np.isfinite(float(x)) else fmt.format(float(x))
    except Exception:
        return na

def fmt_int(x, na="N/D"):
    try:
        return na if pd.isna(x) or not np.isfinite(float(x)) else f"{int(round(float(x))):,}"
    except Exception:
        return na

def fmt_pct(x, fmt="{:.2%}", na="N/D"):
    try:
        return na if pd.isna(x) or not np.isfinite(float(x)) else fmt.format(float(x))
    except Exception:
        return na

def _clean_multiple(x: float, allow_zero: bool = False) -> float:
    """
    Devuelve NaN si el múltiplo es inválido/no finito o ≤0 (o <0 si allow_zero=True).
    Evita tratar múltiplos “negativos” como baratos (suelen reflejar pérdidas).
    """
    if pd.isna(x):
        return np.nan
    try:
        x = float(x)
    except Exception:
        return np.nan
    if not np.isfinite(x):
        return np.nan
    if allow_zero:
        return np.nan if x < 0 else x
    else:
        return np.nan if x <= 0 else x

# ================================ INTERFAZ DE DATOS ================================

def fetch_snapshot(ticker: str) -> Dict:
    """Obtiene snapshot completo de datos para un ticker."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        # Precio actual con fallbacks
        precio_actual = _get_current_price(stock)

        # Datos básicos
        market_cap = _nan_if_missing(info.get('marketCap'))
        shares_outstanding = _nan_if_missing(info.get('sharesOutstanding'))

        # Métricas de valoración
        pe_ttm = _nan_if_missing(info.get('trailingPE'))
        pe_fwd = _nan_if_missing(info.get('forwardPE'))
        pb = _nan_if_missing(info.get('priceToBook'))
        ps = _nan_if_missing(info.get('priceToSalesTrailing12Months'))
        ev_ebitda = _nan_if_missing(info.get('enterpriseToEbitda'))
        ev_revenue = _nan_if_missing(info.get('enterpriseToRevenue'))

        # Calidad
        roe = _nan_if_missing(info.get('returnOnEquity'))
        roa = _nan_if_missing(info.get('returnOnAssets'))
        roic = _nan_if_missing(info.get('returnOnInvestedCapital'))
        gross_margin = _nan_if_missing(info.get('grossMargins'))
        operating_margin = _nan_if_missing(info.get('operatingMargins'))
        profit_margin = _nan_if_missing(info.get('profitMargins'))

        # Crecimiento
        revenue_growth = _nan_if_missing(info.get('revenueGrowth'))
        earnings_growth = _nan_if_missing(info.get('earningsGrowth'))

        # Deuda
        debt_equity = _nan_if_missing(info.get('debtToEquity'))
        debt_ebitda = _nan_if_missing(info.get('debtToEbitda'))

        # FCF yield (calculado)
        fcf = _nan_if_missing(info.get('freeCashflow'))
        fcf_yield = _safe_div(fcf, market_cap) if pd.notna(fcf) and pd.notna(market_cap) and market_cap > 0 else np.nan

        # Net cash/EBITDA
        total_cash = _nan_if_missing(info.get('totalCash'))
        total_debt = _nan_if_missing(info.get('totalDebt'))
        net_cash = (total_cash if pd.notna(total_cash) else 0.0) - (total_debt if pd.notna(total_debt) else 0.0)
        ebitda = _nan_if_missing(info.get('ebitda'))
        net_cash_ebitda = _safe_div(net_cash, ebitda) if pd.notna(ebitda) and ebitda != 0 else np.nan

        return {
            'ticker': ticker.upper(),
            'price': precio_actual,
            'market_cap': market_cap,
            'shares_outstanding': shares_outstanding,
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),

            # Valoración
            'pe_ttm': pe_ttm,
            'pe_fwd': pe_fwd,
            'pb': pb,
            'ps': ps,
            'ev_ebitda': ev_ebitda,
            'ev_revenue': ev_revenue,
            'fcf_yield': fcf_yield,

            # Calidad
            'roe': roe,
            'roa': roa,
            'roic': roic,
            'gross_margin': gross_margin,
            'operating_margin': operating_margin,
            'profit_margin': profit_margin,

            # Crecimiento
            'revenue_growth': revenue_growth,
            'earnings_growth': earnings_growth,

            # Deuda
            'debt_equity': debt_equity,
            'debt_ebitda': debt_ebitda,
            'net_cash_ebitda': net_cash_ebitda,

            # Metadatos
            'as_of': {
                'price_date': _get_latest_trading_date(),
                'fund_date': _get_fundamental_date(info)
            },
            'source': 'yfinance'
        }

    except Exception as e:
        print(f"❌ Error obteniendo snapshot para {ticker}: {e}")
        return {
            'ticker': ticker.upper(),
            'price': np.nan,
            'market_cap': np.nan,
            'sector': 'N/A',
            'industry': 'N/A',
            'as_of': {'price_date': 'N/D', 'fund_date': 'N/D'},
            'source': 'error'
        }

def fetch_history(ticker: str, days: int = 252) -> pd.Series:
    """Obtiene histórico de precios para un ticker."""
    try:
        stock = yf.Ticker(ticker)
        period = f"{max(1, math.ceil(days/252))}y"
        hist = stock.history(period=period, interval='1d')
        if hist.empty:
            return pd.Series(dtype=float)
        s = hist['Close'].dropna()
        return s.tail(days)
    except Exception as e:
        print(f"❌ Error obteniendo histórico para {ticker}: {e}")
        return pd.Series(dtype=float)

def fetch_peers(scope: str, name: str, ticker: str) -> List[str]:
    """Obtiene lista de peers del mismo sector/industria."""
    if scope == "industry":
        industry_peers = {
            "Consumer Electronics": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NFLX"],
            "Software": ["MSFT", "GOOGL", "ORCL", "CRM", "ADBE", "NOW"],
            "E-commerce": ["AMZN", "EBAY", "SHOP", "ETSY", "MELI"],
            "Automotive": ["TSLA", "F", "GM", "TM", "HMC"],
            "Financial Services": ["JPM", "BAC", "WFC", "GS", "MS"]
        }
        return industry_peers.get(name, [ticker])
    elif scope == "sector":
        sector_peers = {
            "Technology": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NFLX", "ORCL", "CRM"],
            "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE"],
            "Financial Services": ["JPM", "BAC", "WFC", "GS", "MS", "V", "MA"],
            "Healthcare": ["JNJ", "PFE", "UNH", "ABBV", "MRK"],
            "Energy": ["XOM", "CVX", "COP", "EOG", "SLB"]
        }
        return sector_peers.get(name, [ticker])
    elif scope == "universe":
        return ["AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","ORCL","ADBE","CRM"]
    return [ticker]

def _get_current_price(stock) -> float:
    """Obtiene precio actual con múltiples fallbacks."""
    try:
        fi = getattr(stock, "fast_info", None)
        price = getattr(fi, "last_price", None) if fi is not None else None
        if price is not None:
            return float(price)
    except Exception:
        pass
    try:
        info = getattr(stock, "info", {}) or {}
        p = info.get("regularMarketPrice") or info.get("currentPrice")
        if p is not None:
            return float(p)
    except Exception:
        pass
    try:
        hist = stock.history(period="5d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return np.nan

def _get_latest_trading_date() -> str:
    """Obtiene la fecha de trading más reciente."""
    try:
        stock = yf.Ticker("AAPL")
        hist = stock.history(period='5d')
        if not hist.empty:
            return hist.index[-1].strftime('%Y-%m-%d')
    except Exception:
        pass
    return datetime.now().strftime('%Y-%m-%d')

def _get_fundamental_date(info: Dict) -> str:
    """Obtiene fecha de datos fundamentales."""
    v = info.get('lastFiscalYearEnd')
    if v is None:
        return datetime.now().strftime('%Y-%m-%d')
    try:
        if isinstance(v, (int, float)):
            if v > 10_000_000_000:  # epoch ms
                v = v / 1000.0
            return datetime.utcfromtimestamp(v).strftime('%Y-%m-%d')
        return str(v)
    except Exception:
        return datetime.now().strftime('%Y-%m-%d')

# ================================ FACTORES ================================

def valuation_subscore(ticker_data: Dict, peers_data: List[Dict]) -> Tuple[float, Dict]:
    """Calcula subscore de valoración basado en percentiles de peers con sanitizado y winsorización clave."""
    # Extraer métricas (sanitizadas)
    pe_ttm = _clean_multiple(ticker_data.get('pe_ttm'))          # P/E ≤0 => NaN
    pe_fwd = _clean_multiple(ticker_data.get('pe_fwd'))          # P/E FWD ≤0 => NaN
    ev_ebitda_raw = ticker_data.get('ev_ebitda')
    ev_ebitda = _clean_multiple(ev_ebitda_raw)                   # EV/EBITDA ≤0 => NaN
    ps = _clean_multiple(ticker_data.get('ps'))                  # P/S ≤0 => NaN
    pb = _clean_multiple(ticker_data.get('pb'))                  # P/B ≤0 => NaN
    fcf_yield = ticker_data.get('fcf_yield')                     # FCF yield puede ser negativo (aceptable)

    # Obtener arrays de peers (mismos filtros)
    pe_peers = [_clean_multiple(p.get('pe_ttm')) for p in peers_data]
    pe_peers = [v for v in pe_peers if pd.notna(v)]

    ev_peers = [_clean_multiple(p.get('ev_ebitda')) for p in peers_data]
    ev_peers = [v for v in ev_peers if pd.notna(v)]

    ps_peers = [_clean_multiple(p.get('ps')) for p in peers_data]
    ps_peers = [v for v in ps_peers if pd.notna(v)]

    pb_peers = [_clean_multiple(p.get('pb')) for p in peers_data]
    pb_peers = [v for v in pb_peers if pd.notna(v)]

    fcf_peers = [p.get('fcf_yield') for p in peers_data if pd.notna(p.get('fcf_yield', np.nan))]

    scores = []
    weights = []
    details = {}

    # P/E TTM (más alto = más caro)
    if pd.notna(pe_ttm) and pe_peers:
        pe_pct = percentile_rank(pe_ttm, pe_peers)
        if pd.notna(pe_pct):
            pe_score = -(2 * pe_pct - 1)
            scores.append(pe_score)
            weights.append(0.25)
            details['pe_pct'] = pe_pct

    # P/E Forward (si disponible)
    if pd.notna(pe_fwd) and pe_peers:
        pe_fwd_pct = percentile_rank(pe_fwd, pe_peers)
        if pd.notna(pe_fwd_pct):
            pe_fwd_score = -(2 * pe_fwd_pct - 1)
            scores.append(pe_fwd_score)
            weights.append(0.20)
            details['pe_fwd_pct'] = pe_fwd_pct

    # EV/EBITDA (más alto = más caro) con winsorización contra peers
    if pd.notna(ev_ebitda) and ev_peers:
        ev_w = winsorize_against_peers(ev_ebitda, ev_peers, *WINSOR['ev_ebitda'])
        ev_peers_w = [winsorize_against_peers(v, ev_peers, *WINSOR['ev_ebitda']) for v in ev_peers]
        ev_pct = percentile_rank(ev_w, ev_peers_w)
        if pd.notna(ev_pct):
            ev_score = -(2 * ev_pct - 1)
            scores.append(ev_score)
            weights.append(0.25)
            details['ev_ebitda_pct'] = ev_pct

    # P/S (más alto = más caro)
    if pd.notna(ps) and ps_peers:
        ps_pct = percentile_rank(ps, ps_peers)
        if pd.notna(ps_pct):
            ps_score = -(2 * ps_pct - 1)
            scores.append(ps_score)
            weights.append(0.20)
            details['ps_pct'] = ps_pct

    # FCF Yield (más alto = mejor, peso mayor)
    if pd.notna(fcf_yield) and fcf_peers:
        fcf_pct = percentile_rank(fcf_yield, fcf_peers)
        if pd.notna(fcf_pct):
            fcf_score = (2 * fcf_pct - 1)
            scores.append(fcf_score)
            weights.append(0.35)
            details['fcf_pct'] = fcf_pct

    # P/B (peso reducido y winsorizado contra peers)
    if pd.notna(pb) and pb_peers:
        pb_winsor = winsorize_against_peers(pb, pb_peers, *WINSOR['pb'])
        pb_peers_winsor = [winsorize_against_peers(p, pb_peers, *WINSOR['pb']) for p in pb_peers]
        pb_pct = percentile_rank(pb_winsor, pb_peers_winsor)
        if pd.notna(pb_pct):
            pb_score = -(2 * pb_pct - 1)
            scores.append(pb_score)
            weights.append(0.10)  # Peso reducido
            details['pb_pct'] = pb_pct

    # Calcular subscore ponderado
    if scores and weights:
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
            subscore = sum(s * w for s, w in zip(scores, weights))
        else:
            subscore = 0.0
    else:
        subscore = 0.0

    return subscore, details

def quality_subscore(ticker_data: Dict, peers_data: List[Dict]) -> Tuple[float, Dict]:
    """Calcula subscore de calidad basado en ROIC, márgenes y estabilidad."""
    # Extraer métricas de calidad
    roic = ticker_data.get('roic')
    operating_margin = ticker_data.get('operating_margin')
    profit_margin = ticker_data.get('profit_margin')
    net_cash_ebitda = ticker_data.get('net_cash_ebitda')
    roe = ticker_data.get('roe')

    # Obtener arrays de peers
    roic_peers = [p.get('roic') for p in peers_data if pd.notna(p.get('roic', np.nan))]
    op_margin_peers = [p.get('operating_margin') for p in peers_data if pd.notna(p.get('operating_margin', np.nan))]
    profit_margin_peers = [p.get('profit_margin') for p in peers_data if pd.notna(p.get('profit_margin', np.nan))]
    net_cash_peers = [p.get('net_cash_ebitda') for p in peers_data if pd.notna(p.get('net_cash_ebitda', np.nan))]
    roe_peers = [p.get('roe') for p in peers_data if pd.notna(p.get('roe', np.nan))]

    scores = []
    weights = []
    details = {}

    # ROIC (peso alto, más alto = mejor)
    if pd.notna(roic) and roic_peers:
        roic_pct = percentile_rank(roic, roic_peers)
        if pd.notna(roic_pct):
            roic_score = (2 * roic_pct - 1)
            scores.append(roic_score)
            weights.append(0.40)
            details['roic_pct'] = roic_pct

    # Margen operativo (más alto = mejor)
    if pd.notna(operating_margin) and op_margin_peers:
        op_margin_pct = percentile_rank(operating_margin, op_margin_peers)
        if pd.notna(op_margin_pct):
            op_margin_score = (2 * op_margin_pct - 1)
            scores.append(op_margin_score)
            weights.append(0.25)
            details['op_margin_pct'] = op_margin_pct

    # Margen neto (más alto = mejor)
    if pd.notna(profit_margin) and profit_margin_peers:
        profit_margin_pct = percentile_rank(profit_margin, profit_margin_peers)
        if pd.notna(profit_margin_pct):
            profit_margin_score = (2 * profit_margin_pct - 1)
            scores.append(profit_margin_score)
            weights.append(0.20)
            details['profit_margin_pct'] = profit_margin_pct

    # Net cash/EBITDA (más alto = mejor)
    if pd.notna(net_cash_ebitda) and net_cash_peers:
        net_cash_pct = percentile_rank(net_cash_ebitda, net_cash_peers)
        if pd.notna(net_cash_pct):
            net_cash_score = (2 * net_cash_pct - 1)
            scores.append(net_cash_score)
            weights.append(0.15)
            details['net_cash_pct'] = net_cash_pct

    # ROE (peso bajo y winsorizado contra peers)
    if pd.notna(roe) and roe_peers:
        roe_winsor = winsorize_against_peers(roe, roe_peers, *WINSOR['roe'])
        roe_peers_winsor = [winsorize_against_peers(p, roe_peers, *WINSOR['roe']) for p in roe_peers]
        roe_pct = percentile_rank(roe_winsor, roe_peers_winsor)
        if pd.notna(roe_pct):
            roe_score = (2 * roe_pct - 1)
            scores.append(roe_score)
            weights.append(0.10)  # Peso bajo
            details['roe_pct'] = roe_pct

    # Calcular subscore ponderado
    if scores and weights:
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
            subscore = sum(s * w for s, w in zip(scores, weights))
        else:
            subscore = 0.0
    else:
        subscore = 0.0

    return subscore, details

def growth_subscore(ticker_data: Dict, peers_data: List[Dict]) -> Tuple[float, Dict]:
    """Calcula subscore de crecimiento basado en ventas y EPS."""
    # Extraer métricas de crecimiento
    revenue_growth = ticker_data.get('revenue_growth')
    earnings_growth = ticker_data.get('earnings_growth')

    # Obtener arrays de peers
    rev_growth_peers = [p.get('revenue_growth') for p in peers_data if pd.notna(p.get('revenue_growth', np.nan))]
    earn_growth_peers = [p.get('earnings_growth') for p in peers_data if pd.notna(p.get('earnings_growth', np.nan))]

    scores = []
    weights = []
    details = {}

    # Crecimiento de ventas (más alto = mejor)
    if pd.notna(revenue_growth) and rev_growth_peers:
        rev_pct = percentile_rank(revenue_growth, rev_growth_peers)
        if pd.notna(rev_pct):
            rev_score = (2 * rev_pct - 1)
            scores.append(rev_score)
            weights.append(0.60)
            details['revenue_growth_pct'] = rev_pct

    # Crecimiento de ganancias (más alto = mejor)
    if pd.notna(earnings_growth) and earn_growth_peers:
        earn_pct = percentile_rank(earnings_growth, earn_growth_peers)
        if pd.notna(earn_pct):
            earn_score = (2 * earn_pct - 1)
            scores.append(earn_score)
            weights.append(0.40)
            details['earnings_growth_pct'] = earn_pct

    # Calcular subscore ponderado
    if scores and weights:
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
            subscore = sum(s * w for s, w in zip(scores, weights))
        else:
            subscore = 0.0
    else:
        subscore = 0.0

    return subscore, details

def momentum_subscore(ticker_data: Dict, price_history: np.ndarray) -> Tuple[float, Dict]:
    """Calcula subscore de momentum basado en rendimientos y RSI."""
    w12 = MOMENTUM_WINDOW['m12']
    w6 = MOMENTUM_WINDOW['m6']
    wskip = MOMENTUM_WINDOW['skip']

    if len(price_history) < 200:  # Necesitamos al menos ~8 meses de datos
        return 0.0, {}

    # Calcular rendimientos (aritméticos; opcional: usar log-retornos)
    returns = np.diff(price_history) / price_history[:-1]

    # Momentum 12m menos 1m
    if len(returns) >= w12:
        mom12 = np.prod(1 + returns[-w12:]) - 1  # 12 meses
        mom1 = np.prod(1 + returns[-wskip:]) - 1  # 1 mes
        mom12_1 = mom12 - mom1
    else:
        mom12_1 = 0.0

    # Momentum 6m
    if len(returns) >= w6:
        mom6 = np.prod(1 + returns[-w6:]) - 1   # 6 meses
    else:
        mom6 = 0.0

    # RSI
    rsi = _calculate_rsi(price_history, window=RSI_WINDOW)

    # Volatilidad anualizada
    vol_252 = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0

    # Calcular scores
    scores = []
    weights = []
    details = {}

    # Momentum 12m-1m (peso principal)
    if mom12_1 != 0.0:
        mom12_1_score = normalize_signed(mom12_1, higher_is_better=True)
        scores.append(mom12_1_score)
        weights.append(0.70)
        details['mom12_1'] = mom12_1

    # Momentum 6m
    if mom6 != 0.0:
        mom6_score = normalize_signed(mom6, higher_is_better=True)
        scores.append(mom6_score)
        weights.append(0.30)
        details['mom6'] = mom6

    # RSI (ajuste leve)
    rsi_adjustment = 0.0
    if pd.notna(rsi):
        if rsi > RSI_OVERBOUGHT:
            rsi_adjustment = -RSI_MAX_ADJUSTMENT * (rsi - RSI_OVERBOUGHT) / (100 - RSI_OVERBOUGHT)
        elif rsi < RSI_OVERSOLD:
            rsi_adjustment = RSI_MAX_ADJUSTMENT * (RSI_OVERSOLD - rsi) / RSI_OVERSOLD

        details['rsi'] = rsi
        details['rsi_adjustment'] = rsi_adjustment

    # Calcular subscore base
    if scores and weights:
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
            base_score = sum(s * w for s, w in zip(scores, weights))
        else:
            base_score = 0.0
    else:
        base_score = 0.0

    # Aplicar ajuste de RSI
    final_score = base_score + rsi_adjustment
    final_score = np.clip(final_score, -1.0, 1.0)

    details['vol_252'] = vol_252

    return final_score, details

# ================================ SCORING PRINCIPAL ================================

def compute_score(ticker: str) -> Dict:
    """Calcula score completo para un ticker."""
    # Obtener datos del ticker
    ticker_data = fetch_snapshot(ticker)
    if pd.isna(ticker_data['price']):
        return _create_error_result(ticker, "No se pudieron obtener datos")

    # Obtener histórico de precios
    price_history = fetch_history(ticker)
    if len(price_history) < 200:  # Reducido de 252 a 200 días para ser más flexible
        return _create_error_result(ticker, "Histórico insuficiente")

    # Obtener peers
    peers = _get_peers_with_fallback(ticker_data)
    if not peers:
        return _create_error_result(ticker, "No se encontraron peers")

    # Obtener datos de peers
    peers_data = []
    for peer in peers:
        if peer != ticker:
            peer_data = fetch_snapshot(peer)
            if pd.notna(peer_data['price']):
                peers_data.append(peer_data)

    # Incluir el ticker actual en los peers para comparación
    peers_data.append(ticker_data)

    # Verificar disponibilidad relativa (excluyendo el propio)
    relative_available = sum(1 for p in peers if p != ticker) >= MIN_PEERS

    # Calcular subscores
    valuation_score, valuation_details = valuation_subscore(ticker_data, peers_data)
    quality_score, quality_details = quality_subscore(ticker_data, peers_data)
    growth_score, growth_details = growth_subscore(ticker_data, peers_data)
    momentum_score, momentum_details = momentum_subscore(ticker_data, price_history.values)

    # Calcular score total ponderado
    total_score = (
        valuation_score * WEIGHTS['valuation'] +
        quality_score * WEIGHTS['quality'] +
        growth_score * WEIGHTS['growth'] +
        momentum_score * WEIGHTS['momentum']
    )

    # Determinar conclusión
    if total_score > NEUTRAL_BAND:
        conclusion = "INFRAVALORADO"
    elif total_score < -NEUTRAL_BAND:
        conclusion = "SOBREVALUADO"
    else:
        conclusion = "NEUTRAL"

    # Crear resultado
    result = {
        'ticker': ticker.upper(),
        'basics': {
            'price': ticker_data['price'],
            'sector': ticker_data['sector'],
            'industry': ticker_data['industry'],
            'market_cap': ticker_data['market_cap']
        },
        'valuation': {
            'pe_ttm': ticker_data['pe_ttm'],
            'pe_fwd': ticker_data['pe_fwd'],
            'ev_ebitda': ticker_data['ev_ebitda'],
            'ps': ticker_data['ps'],
            'pb': ticker_data['pb'],
            'fcf_yield': ticker_data['fcf_yield'],
            'subscore': valuation_score,
            'detail': valuation_details
        },
        'quality': {
            'roic': ticker_data['roic'],
            'op_margin': ticker_data['operating_margin'],
            'profit_margin': ticker_data['profit_margin'],
            'net_cash_ebitda': ticker_data['net_cash_ebitda'],
            'roe': ticker_data['roe'],
            'subscore': quality_score,
            'detail': quality_details
        },
        'growth': {
            'rev_yoy': ticker_data['revenue_growth'],
            'eps_yoy': ticker_data['earnings_growth'],
            'subscore': growth_score,
            'detail': growth_details
        },
        'momentum': {
            'r12_1': momentum_details.get('mom12_1', np.nan),
            'r6': momentum_details.get('mom6', np.nan),
            'rsi14': momentum_details.get('rsi', np.nan),
            'vol_252': momentum_details.get('vol_252', np.nan),
            'subscore': momentum_score,
            'detail': momentum_details
        },
        'score': total_score,
        'conclusion': conclusion,
        'as_of': ticker_data['as_of'],
        'source': ticker_data['source'],
        'peers_used': len(peers_data),
        'peers_scope': RELATIVE_SCOPE,
        'relative_available': relative_available
    }

    return result

def _get_peers_with_fallback(ticker_data: Dict) -> List[str]:
    """Obtiene peers con fallback si no hay suficientes."""
    scope = RELATIVE_SCOPE
    name = ticker_data.get('industry' if scope == 'industry' else 'sector', '')

    peers = fetch_peers(scope, name, ticker_data['ticker'])

    # Si no hay suficientes peers, intentar fallback
    if len(peers) < MIN_PEERS:
        fallback_scope = FALLBACK_SCOPE.get(scope)
        if fallback_scope:
            fallback_name = ticker_data.get('sector' if fallback_scope == 'sector' else 'industry', '')
            peers = fetch_peers(fallback_scope, fallback_name, ticker_data['ticker'])

    # Si aún no hay suficientes, usar universe
    if len(peers) < MIN_PEERS:
        peers = fetch_peers("universe", "", ticker_data['ticker'])

    return peers

def _create_error_result(ticker: str, error_msg: str) -> Dict:
    """Crea resultado de error."""
    return {
        'ticker': ticker.upper(),
        'error': error_msg,
        'score': np.nan,
        'conclusion': 'ERROR',
        'as_of': {'price_date': 'N/D', 'fund_date': 'N/D'},
        'source': 'error'
    }

# ================================ REPORTES ================================

def render_stock_report(result: Dict) -> str:
    """Genera reporte individual de un stock (formateo seguro)."""
    if 'error' in result:
        return f"❌ Error analizando {result['ticker']}: {result['error']}"

    ticker = result['ticker']
    basics = result['basics']
    valuation = result['valuation']
    quality = result['quality']
    growth = result['growth']
    momentum = result['momentum']

    # Header
    report = f"\n{'='*60}\n"
    report += f"📊 REPORTE DE VALORACIÓN v3 - {ticker}\n"
    report += f"{'='*60}\n"

    # Información básica
    report += f"\n INFORMACIÓN BÁSICA:\n"
    report += f"   Precio: ${fmt_num(basics['price'])}\n"
    report += f"   Market Cap: ${fmt_int(basics['market_cap'])}\n"
    report += f"   Sector: {basics['sector']}\n"
    report += f"   Industria: {basics['industry']}\n"

    # Métricas de valoración
    report += f"\n VALORACIÓN:\n"
    pe_line = f"   P/E (TTM): {fmt_num(valuation['pe_ttm'])}"
    if pd.notna(valuation['pe_fwd']):
        pe_line += f" | P/E (Forward): {fmt_num(valuation['pe_fwd'])}"
    report += pe_line + "\n"
    report += f"   EV/EBITDA: {fmt_num(valuation['ev_ebitda'])}\n"
    report += f"   P/S: {fmt_num(valuation['ps'])}\n"
    report += f"   P/B: {fmt_num(valuation['pb'])}\n"
    report += f"   FCF Yield: {fmt_pct(valuation['fcf_yield'])}\n"
    report += f"   Valuation Subscore: {fmt_num(valuation['subscore'], fmt='{:.2f}')}\n"

    # Calidad
    report += f"\n🏆 CALIDAD:\n"
    report += f"   ROIC: {fmt_pct(quality['roic'])}\n"
    report += f"   Margen Operativo: {fmt_pct(quality['op_margin'])}\n"
    report += f"   Margen Neto: {fmt_pct(quality['profit_margin'])}\n"
    report += f"   ROE: {fmt_pct(quality['roe'])}\n"
    report += f"   Quality Subscore: {fmt_num(quality['subscore'], fmt='{:.2f}')}\n"

    # Crecimiento
    report += f"\n CRECIMIENTO:\n"
    report += f"   Ventas YoY: {fmt_pct(growth['rev_yoy'])}\n"
    report += f"   EPS YoY: {fmt_pct(growth['eps_yoy'])}\n"
    report += f"   Growth Subscore: {fmt_num(growth['subscore'], fmt='{:.2f}')}\n"

    # Momentum
    report += f"\n⚡ MOMENTUM:\n"
    report += f"   R12-1m: {fmt_pct(momentum['r12_1'])}\n"
    report += f"   R6m: {fmt_pct(momentum['r6'])}\n"
    report += f"   RSI(14): {fmt_num(momentum['rsi14'], fmt='{:.1f}')}\n"
    report += f"   Volatilidad: {fmt_pct(momentum['vol_252'])}\n"
    report += f"   Momentum Subscore: {fmt_num(momentum['subscore'], fmt='{:.2f}')}\n"

    # Score total y conclusión
    report += f"\n CONCLUSIÓN:\n"
    report += f"   Score Total: {fmt_num(result['score'], fmt='{:.2f}')}\n"
    report += f"   Conclusión: {result['conclusion']}\n"

    # Metadatos
    report += f"\n📅 METADATOS:\n"
    report += f"   Fecha de precios: {result['as_of']['price_date']}\n"
    report += f"   Fecha de fundamentales: {result['as_of']['fund_date']}\n"
    report += f"   Fuente: {result['source']}\n"
    report += f"   Peers usados: {result['peers_used']} ({result['peers_scope']})\n"

    report += f"\n{'='*60}\n"

    return report

def render_portfolio_summary(results: List[Dict]) -> str:
    """Genera resumen de portafolio."""
    if not results:
        return "❌ No hay resultados para mostrar"

    # Filtrar errores
    valid_results = [r for r in results if 'error' not in r]
    if not valid_results:
        return "❌ No hay resultados válidos para mostrar"

    # Ordenar por score
    valid_results.sort(key=lambda x: x.get('score', -999), reverse=True)

    # Header
    summary = f"\n{'='*80}\n"
    summary += f"📊 RESUMEN DE PORTAFOLIO - VALORACIÓN v3\n"
    summary += f"{'='*80}\n"

    # Tabla de resultados
    summary += f"\n{'Ticker':<8} {'Score':<8} {'Valoración':<12} {'Calidad':<12} {'Crecimiento':<12} {'Momentum':<12} {'Conclusión':<15}\n"
    summary += f"{'-'*80}\n"

    for result in valid_results:
        ticker = result['ticker']
        score = result['score']
        val_score = result['valuation']['subscore']
        qual_score = result['quality']['subscore']
        growth_score = result['growth']['subscore']
        mom_score = result['momentum']['subscore']
        conclusion = result['conclusion']

        summary += f"{ticker:<8} {score:<8.2f} {val_score:<12.2f} {qual_score:<12.2f} {growth_score:<12.2f} {mom_score:<12.2f} {conclusion:<15}\n"

    # Estadísticas
    scores = [r['score'] for r in valid_results]
    summary += f"\n📊 ESTADÍSTICAS:\n"
    summary += f"   Promedio Score: {np.mean(scores):.2f}\n"
    summary += f"   Score Máximo: {np.max(scores):.2f}\n"
    summary += f"   Score Mínimo: {np.min(scores):.2f}\n"
    summary += f"   Infravaluados: {len([s for s in scores if s > NEUTRAL_BAND])}\n"
    summary += f"   Sobrevaluados: {len([s for s in scores if s < -NEUTRAL_BAND])}\n"
    summary += f"   Neutrales: {len([s for s in scores if -NEUTRAL_BAND <= s <= NEUTRAL_BAND])}\n"

    summary += f"\n{'='*80}\n"

    return summary

# ================================ FUNCIONES PRINCIPALES ================================

def analizar_stock(ticker: str) -> Dict:
    """Analiza un stock individual."""
    print(f"🔎 Analizando {ticker.upper()}...")
    result = compute_score(ticker)

    if 'error' not in result:
        print(render_stock_report(result))
    else:
        print(f"❌ Error: {result['error']}")

    return result

def analizar_portafolio(tickers: List[str]) -> List[Dict]:
    """Analiza un portafolio completo."""
    print(f"📊 Analizando portafolio de {len(tickers)} stocks...")
    results = []

    for ticker in tickers:
        print(f"\n{'='*40}")
        print(f"Analizando {ticker.upper()}")
        print(f"{'='*40}")

        result = compute_score(ticker)
        results.append(result)

        if 'error' not in result:
            print(render_stock_report(result))
        else:
            print(f"❌ Error: {result['error']}")

    # Mostrar resumen del portafolio
    print(render_portfolio_summary(results))

    return results

def crear_graficos_valoracion(result: Dict) -> None:
    """Crea gráficos de valoración para un stock."""
    if 'error' in result:
        print(f"❌ No se pueden crear gráficos para {result['ticker']}: {result['error']}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Análisis de Valoración v3 - {result["ticker"]}', fontsize=16, fontweight='bold')

    # 1) Subscores por bloque
    blocks = ['Valoración', 'Calidad', 'Crecimiento', 'Momentum']
    scores = [
        result['valuation']['subscore'],
        result['quality']['subscore'],
        result['growth']['subscore'],
        result['momentum']['subscore']
    ]

    colors = ['red' if s < -0.3 else 'green' if s > 0.3 else 'orange' for s in scores]
    bars = axes[0, 0].bar(blocks, scores, color=colors, alpha=0.7)
    axes[0, 0].set_title('Subscores por Bloque', fontweight='bold')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[0, 0].axhline(y=NEUTRAL_BAND, color='green', linestyle='--', alpha=0.5, label='Banda Neutral')
    axes[0, 0].axhline(y=-NEUTRAL_BAND, color='red', linestyle='--', alpha=0.5)
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Añadir valores en las barras
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                       f'{score:.2f}', ha='center', va='bottom' if height >= 0 else 'top')

    # 2) Métricas de valoración
    val_metrics = ['P/E TTM', 'P/E FWD', 'EV/EBITDA', 'P/S', 'P/B', 'FCF Yield']
    val_values = [
        result['valuation']['pe_ttm'],
        result['valuation']['pe_fwd'],
        result['valuation']['ev_ebitda'],
        result['valuation']['ps'],
        result['valuation']['pb'],
        result['valuation']['fcf_yield'] * 100 if not np.isnan(result['valuation']['fcf_yield']) else np.nan
    ]

    # Filtrar valores válidos
    valid_metrics = []
    valid_values = []
    for m, v in zip(val_metrics, val_values):
        if not np.isnan(v) and v is not None:
            valid_metrics.append(m)
            valid_values.append(v)

    if valid_metrics:
        axes[0, 1].bar(valid_metrics, valid_values, alpha=0.7, color='skyblue')
        axes[0, 1].set_title('Métricas de Valoración', fontweight='bold')
        axes[0, 1].set_ylabel('Valor')
        axes[0, 1].tick_params(axis='x', rotation=45)

    # 3) Métricas de calidad
    qual_metrics = ['ROIC', 'Margen Op.', 'Margen Neto', 'ROE']
    qual_values = [
        result['quality']['roic'] * 100 if not np.isnan(result['quality']['roic']) else np.nan,
        result['quality']['op_margin'] * 100 if not np.isnan(result['quality']['op_margin']) else np.nan,
        result['quality']['profit_margin'] * 100 if not np.isnan(result['quality']['profit_margin']) else np.nan,
        result['quality']['roe'] * 100 if not np.isnan(result['quality']['roe']) else np.nan
    ]

    # Filtrar valores válidos
    valid_qual_metrics = []
    valid_qual_values = []
    for m, v in zip(qual_metrics, qual_values):
        if not np.isnan(v) and v is not None:
            valid_qual_metrics.append(m)
            valid_qual_values.append(v)

    if valid_qual_metrics:
        axes[1, 0].bar(valid_qual_metrics, valid_qual_values, alpha=0.7, color='lightgreen')
        axes[1, 0].set_title('Métricas de Calidad (%)', fontweight='bold')
        axes[1, 0].set_ylabel('Porcentaje (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)

    # 4) Momentum y técnico
    mom_metrics = ['R12-1m', 'R6m', 'RSI(14)', 'Volatilidad']
    mom_values = [
        result['momentum']['r12_1'] * 100 if not np.isnan(result['momentum']['r12_1']) else np.nan,
        result['momentum']['r6'] * 100 if not np.isnan(result['momentum']['r6']) else np.nan,
        result['momentum']['rsi14'] if not np.isnan(result['momentum']['rsi14']) else np.nan,
        result['momentum']['vol_252'] * 100 if not np.isnan(result['momentum']['vol_252']) else np.nan
    ]

    # Filtrar valores válidos
    valid_mom_metrics = []
    valid_mom_values = []
    for m, v in zip(mom_metrics, mom_values):
        if not np.isnan(v) and v is not None:
            valid_mom_metrics.append(m)
            valid_mom_values.append(v)

    if valid_mom_metrics:
        colors = ['red' if m == 'RSI(14)' and v > 70 else 'green' if m == 'RSI(14)' and v < 30 else 'blue' for m, v in zip(valid_mom_metrics, valid_mom_values)]
        axes[1, 1].bar(valid_mom_metrics, valid_mom_values, alpha=0.7, color=colors)
        axes[1, 1].set_title('Momentum y Técnico', fontweight='bold')
        axes[1, 1].set_ylabel('Valor')
        axes[1, 1].tick_params(axis='x', rotation=45)

        # Líneas de referencia para RSI
        if 'RSI(14)' in valid_mom_metrics:
            axes[1, 1].axhline(y=70, color='red', linestyle='--', alpha=0.5, label='RSI > 70')
            axes[1, 1].axhline(y=30, color='green', linestyle='--', alpha=0.5, label='RSI < 30')
            axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

def comparar_stocks(tickers: List[str]) -> None:
    """Compara múltiples stocks en una tabla."""
    print(f"🔄 Comparando {len(tickers)} stocks...")

    results = []
    for ticker in tickers:
        result = compute_score(ticker)
        results.append(result)

    # Crear DataFrame para comparación
    comparison_data = []
    for result in results:
        if 'error' not in result:
            comparison_data.append({
                'Ticker': result['ticker'],
                'Score Total': result['score'],
                'Valoración': result['valuation']['subscore'],
                'Calidad': result['quality']['subscore'],
                'Crecimiento': result['growth']['subscore'],
                'Momentum': result['momentum']['subscore'],
                'P/E TTM': result['valuation']['pe_ttm'],
                'P/E FWD': result['valuation']['pe_fwd'],
                'EV/EBITDA': result['valuation']['ev_ebitda'],
                'FCF Yield': result['valuation']['fcf_yield'],
                'ROIC': result['quality']['roic'],
                'ROE': result['quality']['roe'],
                'Conclusión': result['conclusion']
            })

    if comparison_data:
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Score Total', ascending=False)

        print(f"\n{'='*100}")
        print(f"📊 COMPARACIÓN DE STOCKS - VALORACIÓN v3")
        print(f"{'='*100}")

        # Mostrar tabla principal
        print("\n📈 RESUMEN EJECUTIVO:")
        summary_cols = ['Ticker', 'Score Total', 'Valoración', 'Calidad', 'Crecimiento', 'Momentum', 'Conclusión']
        print(df[summary_cols].to_string(index=False, float_format='%.2f'))

        # Mostrar métricas detalladas
        print(f"\n💰 MÉTRICAS DE VALORACIÓN:")
        val_cols = ['Ticker', 'P/E TTM', 'P/E FWD', 'EV/EBITDA', 'FCF Yield']
        print(df[val_cols].to_string(index=False, float_format='%.2f'))

        print(f"\n🏆 MÉTRICAS DE CALIDAD:")
        qual_cols = ['Ticker', 'ROIC', 'ROE']
        # Formatear porcentajes manualmente
        qual_df = df[qual_cols].copy()
        for col in ['ROIC', 'ROE']:
            if col in qual_df.columns:
                qual_df[col] = qual_df[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/D")
        print(qual_df.to_string(index=False))

        # Estadísticas del portafolio
        scores = df['Score Total'].values
        print(f"\n📊 ESTADÍSTICAS DEL PORTAFOLIO:")
        print(f"   Promedio Score: {np.mean(scores):.2f}")
        print(f"   Score Máximo: {np.max(scores):.2f}")
        print(f"   Score Mínimo: {np.min(scores):.2f}")
        print(f"   Desviación Estándar: {np.std(scores):.2f}")
        print(f"   Infravaluados: {len(df[df['Score Total'] > NEUTRAL_BAND])}")
        print(f"   Sobrevaluados: {len(df[df['Score Total'] < -NEUTRAL_BAND])}")
        print(f"   Neutrales: {len(df[(df['Score Total'] >= -NEUTRAL_BAND) & (df['Score Total'] <= NEUTRAL_BAND)])}")

        print(f"\n{'='*100}")

        return df
    else:
        print("❌ No hay datos válidos para comparar")
        return None

def generar_reporte_detallado(ticker: str) -> None:
    """Genera un reporte detallado con gráficos para un stock."""
    print(f"📋 Generando reporte detallado para {ticker.upper()}...")

    result = compute_score(ticker)

    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        return

    # Mostrar reporte
    print(render_stock_report(result))

    # Crear gráficos
    crear_graficos_valoracion(result)

    # Análisis adicional
    print(f"\n📈 ANÁLISIS ADICIONAL:")

    # Top 2 factores positivos y negativos
    factors = {
        'Valoración': result['valuation']['subscore'],
        'Calidad': result['quality']['subscore'],
        'Crecimiento': result['growth']['subscore'],
        'Momentum': result['momentum']['subscore']
    }

    sorted_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)

    print(f"   ✅ Factores más positivos:")
    for i, (factor, score) in enumerate(sorted_factors[:2]):
        if score > 0:
            print(f"      {i+1}. {factor}: {score:.2f}")

    print(f"   ❌ Factores más negativos:")
    for i, (factor, score) in enumerate(sorted_factors[-2:]):
        if score < 0:
            print(f"      {i+1}. {factor}: {score:.2f}")

    # Recomendación
    score = result['score']
    if score > 0.5:
        recommendation = "FUERTE COMPRA"
        color = "🟢"
    elif score > 0.3:
        recommendation = "COMPRA"
        color = "🟢"
    elif score > -0.3:
        recommendation = "MANTENER"
        color = "🟡"
    elif score > -0.5:
        recommendation = "VENTA"
        color = "🟠"
    else:
        recommendation = "FUERTE VENTA"
        color = "🔴"

    print(f"\n🎯 RECOMENDACIÓN: {color} {recommendation}")
    print(f"   Score: {score:.2f} | Conclusión: {result['conclusion']}")

# ================================ TESTS ================================

def test_scoring():
    """Pruebas unitarias del sistema de scoring."""
    print("🧪 Ejecutando tests del sistema de scoring...")

    # Test 1: AAPL con datos reales
    print("\n1️⃣ Test con AAPL:")
    result_aapl = compute_score("AAPL")
    if 'error' not in result_aapl:
        print(f"   ✅ AAPL analizado correctamente")
        print(f"   Score: {result_aapl['score']:.2f}")
        print(f"   Conclusión: {result_aapl['conclusion']}")
    else:
        print(f"   ❌ Error con AAPL: {result_aapl['error']}")

    # Test 2: Portafolio tecnológico
    print("\n2️⃣ Test con portafolio tecnológico:")
    tech_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    results_tech = analizar_portafolio(tech_tickers)

    valid_results = [r for r in results_tech if 'error' not in r]
    if len(valid_results) >= 3:
        print(f"   ✅ Portafolio analizado correctamente ({len(valid_results)}/{len(tech_tickers)} stocks)")
    else:
        print(f"   ❌ Solo {len(valid_results)}/{len(tech_tickers)} stocks analizados correctamente")

    # Test 3: Verificar winsorización (contra peers)
    print("\n3️⃣ Test de winsorización:")
    test_pb = 50.0  # Valor extremo
    test_peers = [1.0, 2.0, 3.0, 4.0, 5.0]
    winsorized = winsorize_against_peers(test_pb, test_peers, p_low=0.01, p_high=0.99)
    print(f"   P/B original: {test_pb}")
    print(f"   P/B winsorizado: {winsorized}")
    print(f"   ✅ Winsorización funcionando" if winsorized < test_pb else "   ❌ Winsorización no funcionando")

    # Test 4: Verificar percentiles
    print("\n4️⃣ Test de percentiles:")
    test_value = 3.0
    test_peers = [1.0, 2.0, 4.0, 5.0, 6.0]
    pct = percentile_rank(test_value, test_peers)
    print(f"   Valor: {test_value}")
    print(f"   Peers: {test_peers}")
    print(f"   Percentil: {pct:.2f}")
    print(f"   ✅ Percentiles funcionando" if 0 <= pct <= 1 else "   ❌ Percentiles no funcionando")

    print("\n🎉 Tests completados!")

# ================================ EJEMPLOS DE USO ================================

def ejemplo_analisis_individual():
    """Ejemplo de análisis individual."""
    print("📊 EJEMPLO: ANÁLISIS INDIVIDUAL")
    print("="*50)

    # Analizar un ticker (puedes cambiar META por AAPL si prefieres)
    generar_reporte_detallado("META")

def ejemplo_comparacion():
    """Ejemplo de comparación de stocks."""
    print("📊 EJEMPLO: COMPARACIÓN DE STOCKS")
    print("="*50)

    # Comparar grandes tecnológicas
    faang_tickers = ["AAPL","MSFT","NVDA","META"]
    comparar_stocks(faang_tickers)

# ================================ EMPÍRICO: RE-CALIBRACIÓN DE PESOS ================================
# Este bloque construye un panel "punto en tiempo" (PIT), evalúa estrategias long-short
# y busca los pesos (valuation, quality, growth, momentum) que maximizan Sharpe OOS.

from functools import lru_cache
from dateutil.relativedelta import relativedelta

# ---------- Utilidades de fechas y cache ----------

def _month_end(d: pd.Timestamp) -> pd.Timestamp:
    return (pd.Timestamp(d).to_period('M') + 0).to_timestamp('M')

@lru_cache(maxsize=256)
def _prices_series_cached(ticker: str, years: int = 6) -> pd.Series:
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=f"{years}y", interval="1d")
        if hist.empty:
            return pd.Series(dtype=float)
        s = hist["Close"].astype(float).dropna().copy()
        # Índice -> naive (sin tz) + ordenado + sin duplicados
        idx = pd.to_datetime(s.index)
        try:
            idx = idx.tz_localize(None)
        except Exception:
            try:
                idx = idx.tz_convert(None)
            except Exception:
                pass
        s.index = idx
        s = s[~s.index.duplicated(keep="last")].sort_index()
        return s
    except Exception:
        return pd.Series(dtype=float)

@lru_cache(maxsize=128)
def _funds_cached(ticker: str):
    """Devuelve (quarterly_financials, quarterly_balance_sheet, quarterly_cashflow, info) cacheado."""
    try:
        t = yf.Ticker(ticker)
        qfin = t.quarterly_financials
        qbs = t.quarterly_balance_sheet
        qcf = t.quarterly_cashflow
        info = t.info or {}
        
        # Asegurar que no devolvemos None
        qfin = qfin if qfin is not None else pd.DataFrame()
        qbs = qbs if qbs is not None else pd.DataFrame()
        qcf = qcf if qcf is not None else pd.DataFrame()
        info = info if info is not None else {}
        
        return (qfin, qbs, qcf, info)
    except Exception as e:
        print(f"⚠️ Error obteniendo datos financieros para {ticker}: {e}")
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {})

def _last_quarter_leq(df: pd.DataFrame, cutoff: pd.Timestamp) -> Optional[pd.Timestamp]:
    """Obtiene la última columna (quarter end) <= cutoff."""
    if df is None or df.empty:
        return None
    cols = df.columns
    try:
        cols = pd.to_datetime(cols)
    except Exception:
        return None
    cols = [c for c in cols if c <= cutoff]
    if not cols:
        return None
    return max(cols)

def _getq(df: pd.DataFrame, row_name: str, col: pd.Timestamp) -> float:
    """Saca un valor (float) del DF trimestral en fila row_name y columna col."""
    if df is None or df.empty or col is None:
        return np.nan
    if row_name not in df.index:
        return np.nan
    try:
        return float(df.loc[row_name, col])
    except Exception:
        return np.nan

def _sum_ttm(df: pd.DataFrame, row_name: str, col: pd.Timestamp, n_quarters: int = 4) -> float:
    """Suma TTM de la fila row_name terminando en 'col' (incluido)."""
    if df is None or df.empty or col is None:
        return np.nan
    if row_name not in df.index:
        return np.nan
    try:
        cols_sorted = sorted(pd.to_datetime(df.columns))
        if col not in cols_sorted:
            return np.nan
        idx = cols_sorted.index(col)
        start = max(0, idx - (n_quarters - 1))
        window = cols_sorted[start:idx + 1]
        vals = [df.loc[row_name, c] for c in window if row_name in df.index]
        vals = [float(v) for v in vals if pd.notna(v)]
        return float(np.nansum(vals)) if vals else np.nan
    except Exception:
        return np.nan

def _shares_outstanding_from_info(info: Dict) -> float:
    x = info.get('sharesOutstanding')
    try:
        return float(x) if x is not None and np.isfinite(float(x)) else np.nan
    except Exception:
        return np.nan

def _price_on_or_before(ticker: str, date: pd.Timestamp) -> float:
    s = _prices_series_cached(ticker)
    if s.empty:
        return np.nan
    ts = pd.Timestamp(date)
    # Asegura naive
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    # Búsqueda binaria: última barra <= ts
    pos = s.index.searchsorted(ts, side="right") - 1
    if pos >= 0:
        return float(s.iloc[pos])
    # Fallback: primera barra >= ts (por si no hay anterior)
    pos = s.index.searchsorted(ts, side="left")
    return float(s.iloc[pos]) if pos < len(s) else np.nan

def _price_on_after(ticker: str, date: pd.Timestamp) -> float:
    s = _prices_series_cached(ticker)
    if s.empty:
        return np.nan
    ts = pd.Timestamp(date)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    pos = s.index.searchsorted(ts, side="left")
    if pos < len(s):
        return float(s.iloc[pos])
    # Fallback: última barra disponible
    return float(s.iloc[-1])

def _create_empty_snapshot(ticker: str, asof: pd.Timestamp) -> Dict:
    """Crea un snapshot vacío cuando no hay datos financieros disponibles."""
    return {
        'ticker': ticker.upper(),
        'price': np.nan,
        'market_cap': np.nan,
        'sector': 'N/A',
        'industry': 'N/A',
        
        # Valoración
        'pe_ttm': np.nan,
        'pe_fwd': np.nan,
        'pb': np.nan,
        'ps': np.nan,
        'ev_ebitda': np.nan,
        'ev_revenue': np.nan,
        'fcf_yield': np.nan,
        
        # Calidad
        'roe': np.nan,
        'roa': np.nan,
        'roic': np.nan,
        'gross_margin': np.nan,
        'operating_margin': np.nan,
        'profit_margin': np.nan,
        
        # Crecimiento
        'revenue_growth': np.nan,
        'earnings_growth': np.nan,
        
        # Deuda
        'debt_equity': np.nan,
        'debt_ebitda': np.nan,
        'net_cash_ebitda': np.nan,
        
        'as_of': {'price_date': str(pd.Timestamp(asof).date()), 'fund_date': 'N/D'},
        'source': 'empty'
    }

# ---------- Snapshot PIT (fundamentales y múltiplos con lag) ----------

def build_pit_snapshot(ticker: str, asof: pd.Timestamp, lag_days: int = 30) -> Dict:
    """
    Construye un snapshot 'punto en tiempo' para 'asof'.
    - Usa el último trimestre con cierre <= asof - lag_days (para evitar look-ahead).
    - Calcula TTM de Revenue, Net Income, EBIT/EBITDA aprox, CFO/CapEx y deriva múltiplos.
    """
    try:
        qfin, qbs, qcf, info = _funds_cached(ticker)
        cutoff = pd.Timestamp(asof) - pd.Timedelta(days=lag_days)
        
        # Verificar que tenemos datos financieros básicos
        if qfin is None or qfin.empty:
            print(f"⚠️ Sin datos financieros para {ticker} en {asof.date()}")
            return _create_empty_snapshot(ticker, asof)
    except Exception as e:
        print(f"⚠️ Error en build_pit_snapshot para {ticker}: {e}")
        return _create_empty_snapshot(ticker, asof)

    # Quarter más reciente disponible antes del cutoff
    qcol_fin = _last_quarter_leq(qfin, cutoff)
    qcol_bs  = _last_quarter_leq(qbs, cutoff)
    qcol_cf  = _last_quarter_leq(qcf, cutoff)
    
    # Debug: verificar si tenemos quarters válidos
    if qcol_fin is None:
        print(f"⚠️ {ticker} en {asof.date()}: sin quarters financieros antes de {cutoff.date()}")
        return _create_empty_snapshot(ticker, asof)

    # TTM básicos
    rev_ttm = _sum_ttm(qfin, 'Total Revenue', qcol_fin)
    ni_ttm  = _sum_ttm(qfin, 'Net Income', qcol_fin)

    # EBITDA TTM (intentamos EBITDA; si no, EBIT + Depreciation)
    ebitda_ttm = _sum_ttm(qfin, 'Ebitda', qcol_fin)
    if pd.isna(ebitda_ttm):
        ebit_ttm = _sum_ttm(qfin, 'Ebit', qcol_fin)
        d_and_a_ttm = _sum_ttm(qfin, 'Depreciation', qcol_fin)
        if pd.isna(d_and_a_ttm):
            d_and_a_ttm = _sum_ttm(qfin, 'Depreciation & Amortization', qcol_fin)
        if pd.isna(ebit_ttm) and not pd.isna(_sum_ttm(qfin, 'Operating Income', qcol_fin)):
            ebit_ttm = _sum_ttm(qfin, 'Operating Income', qcol_fin)
        ebitda_ttm = (ebit_ttm if pd.notna(ebit_ttm) else 0.0) + (d_and_a_ttm if pd.notna(d_and_a_ttm) else 0.0)
        if ebitda_ttm == 0.0:
            ebitda_ttm = np.nan

    # Márgenes (TTM)
    gross_ttm   = _sum_ttm(qfin, 'Gross Profit', qcol_fin)
    op_inc_ttm  = _sum_ttm(qfin, 'Operating Income', qcol_fin)
    profit_m    = _safe_div(ni_ttm, rev_ttm)
    op_m        = _safe_div(op_inc_ttm, rev_ttm)
    gross_m     = _safe_div(gross_ttm, rev_ttm)

    # Balance y CF (último quarter)
    total_debt = _getq(qbs, 'Total Debt', qcol_bs)
    if pd.isna(total_debt):
        # aproximar con deuda corto+largo si existe
        sd = _getq(qbs, 'Short Long Term Debt', qcol_bs)
        ld = _getq(qbs, 'Long Term Debt', qcol_bs)
        total_debt = (sd if pd.notna(sd) else 0.0) + (ld if pd.notna(ld) else 0.0)
        if total_debt == 0.0:
            total_debt = np.nan
    cash = _getq(qbs, 'Cash And Cash Equivalents', qcol_bs)
    if pd.isna(cash):
        cash = _getq(qbs, 'CashAndCashEquivalents', qcol_bs)
    total_equity = _getq(qbs, 'Total Stockholder Equity', qcol_bs)

    cfo_ttm   = _sum_ttm(qcf, 'Total Cash From Operating Activities', qcol_cf)
    if pd.isna(cfo_ttm):
        cfo_ttm = _sum_ttm(qcf, 'Operating Cash Flow', qcol_cf)
    capex_ttm = _sum_ttm(qcf, 'Capital Expenditures', qcol_cf)

    fcf_ttm = np.nan
    if pd.notna(cfo_ttm) and pd.notna(capex_ttm):
        fcf_ttm = cfo_ttm - abs(capex_ttm)

    # Precio/MC/EV a fecha asof
    price = _price_on_or_before(ticker, asof)
    shares = _shares_outstanding_from_info(info)  # no PIT, pero usable como proxy
    market_cap = price * shares if (pd.notna(price) and pd.notna(shares)) else np.nan
    ev = np.nan
    if pd.notna(market_cap) and pd.notna(total_debt) and pd.notna(cash):
        ev = market_cap + total_debt - cash

    # Múltiplos TTM
    eps_ttm = _safe_div(ni_ttm, shares)
    pe_ttm  = _safe_div(price, eps_ttm) if pd.notna(eps_ttm) else np.nan
    ps      = _safe_div(market_cap, rev_ttm)
    pb      = _safe_div(market_cap, total_equity)
    ev_ebitda = _safe_div(ev, ebitda_ttm) if pd.notna(ev) and pd.notna(ebitda_ttm) else np.nan
    fcf_yield = _safe_div(fcf_ttm, market_cap) if pd.notna(fcf_ttm) and pd.notna(market_cap) else np.nan

    # Crecimiento YoY (TTM contra TTM-4Q)
    rev_ttm_prev = np.nan
    ni_ttm_prev  = np.nan
    if qcol_fin is not None:
        four_q_back = qcol_fin - pd.DateOffset(months=12)
        rev_ttm_prev = _sum_ttm(qfin, 'Total Revenue', four_q_back)
        ni_ttm_prev  = _sum_ttm(qfin, 'Net Income', four_q_back)

    rev_yoy  = _safe_div(rev_ttm - rev_ttm_prev, rev_ttm_prev) if pd.notna(rev_ttm_prev) else np.nan
    eps_ttm_prev = _safe_div(ni_ttm_prev, shares) if pd.notna(ni_ttm_prev) and pd.notna(shares) else np.nan
    eps_yoy  = _safe_div(eps_ttm - eps_ttm_prev, eps_ttm_prev) if pd.notna(eps_ttm_prev) else np.nan

    # Net cash / EBITDA
    net_cash = (cash if pd.notna(cash) else 0.0) - (total_debt if pd.notna(total_debt) else 0.0)
    net_cash_ebitda = _safe_div(net_cash, ebitda_ttm) if pd.notna(ebitda_ttm) and ebitda_ttm != 0 else np.nan

    # Sector/Industria (de info; no PIT, pero estable)
    sector = info.get('sector', 'N/A')
    industry = info.get('industry', 'N/A')

    # Armar dict "ticker_data" con las mismas claves que usan tus subscores
    return {
        'ticker': ticker.upper(),
        'price': price,
        'market_cap': market_cap,
        'sector': sector,
        'industry': industry,

        # Valoración
        'pe_ttm': pe_ttm,
        'pe_fwd': np.nan,  # forward PE no PIT con yfinance; se deja NaN
        'pb': pb,
        'ps': ps,
        'ev_ebitda': ev_ebitda,
        'ev_revenue': np.nan,
        'fcf_yield': fcf_yield,

        # Calidad
        'roe': _safe_div(ni_ttm, total_equity),
        'roa': np.nan,
        'roic': np.nan,  # opcional: calcular proxy si añades NOPAT/Capital
        'gross_margin': gross_m,
        'operating_margin': op_m,
        'profit_margin': profit_m,

        # Crecimiento
        'revenue_growth': rev_yoy,
        'earnings_growth': eps_yoy,

        # Deuda
        'debt_equity': _safe_div(total_debt, total_equity),
        'debt_ebitda': _safe_div(total_debt, ebitda_ttm) if pd.notna(ebitda_ttm) else np.nan,
        'net_cash_ebitda': net_cash_ebitda,

        'as_of': {'price_date': str(pd.Timestamp(asof).date()), 'fund_date': str(qcol_fin.date()) if qcol_fin else 'N/D'},
        'source': 'yfinance-pit'
    }

# ---------- Construcción del panel PIT ----------

def _get_peers_same_scope(snapshots_by_ticker: Dict[str, Dict], ticker: str) -> List[str]:
    """Peers según RELATIVE_SCOPE con fallbacks como en tu módulo."""
    tdata = snapshots_by_ticker[ticker]
    scope = RELATIVE_SCOPE
    name = tdata.get('industry' if scope == 'industry' else 'sector', '')
    # Candidatos = todos los de la fecha con mismo scope
    if scope == 'industry':
        peers = [k for k, v in snapshots_by_ticker.items() if v.get('industry') == name]
        if len(peers) < MIN_PEERS:
            # fallback sector
            sname = tdata.get('sector', '')
            peers = [k for k, v in snapshots_by_ticker.items() if v.get('sector') == sname]
    else:
        peers = [k for k, v in snapshots_by_ticker.items() if v.get('sector') == name]
        if len(peers) < MIN_PEERS:
            # fallback industria
            iname = tdata.get('industry', '')
            peers = [k for k, v in snapshots_by_ticker.items() if v.get('industry') == iname]
    if len(peers) < MIN_PEERS:
        peers = list(snapshots_by_ticker.keys())
    return peers

def _momentum_prices_to_date(ticker: str, asof: pd.Timestamp, min_days: int = 220) -> np.ndarray:
    s = _prices_series_cached(ticker)
    if s.empty:
        return np.array([])
    s = s.loc[:pd.Timestamp(asof)]
    if len(s) < min_days:
        return np.array([])
    # usa hasta 3 años de histórico para momentum
    s = s.tail(3*252)
    return s.values.astype(float)

def build_panel_pit(tickers: List[str], start: str, end: str, freq: str = 'M', lag_days: int = 30) -> pd.DataFrame:
    """
    Construye un panel con índices (date, ticker) y columnas:
    ['valuation','quality','growth','momentum','ret_fwd','sector'].
    Calcula subscores usando snapshots PIT por fecha y un retorno 1M forward.
    - Robusto a fines de semana/festivos (precio 'smart' en/antes o en/después).
    - Normaliza fechas a fin de mes si freq='M'.
    """
    # Normaliza la rejilla temporal (si es mensual, usa fin de mes)
    dates = pd.date_range(start=start, end=end, freq=freq)
    dates = pd.to_datetime(dates).to_period('M').to_timestamp('M') if freq.upper().startswith('M') else dates

    rows: List[Dict] = []

    # Precio "inteligente": intenta <= dt; si no hay, usa >= dt
    def _price_smart(tkr: str, dt: pd.Timestamp) -> float:
        p = _price_on_or_before(tkr, dt)
        if pd.isna(p):
            p = _price_on_after(tkr, dt)
        return p

    for dt in dates:
        dt = pd.Timestamp(dt).normalize()

        # Snapshots PIT para todos los tickers en la fecha dt
        snaps: Dict[str, Dict] = {}
        for t in tickers:
            try:
                snap = build_pit_snapshot(t, dt, lag_days=lag_days)
                # Verificar que el snapshot no esté vacío y tenga precio válido
                if snap and snap.get('source') != 'empty':
                    price_ok = snap.get('price', np.nan)
                    if pd.notna(price_ok) and np.isfinite(float(price_ok)):
                        snaps[t] = snap
            except Exception as e:
                print(f"⚠️ Error procesando {t} en {dt.date()}: {e}")
                continue

        # Debug: mostrar información sobre snapshots por fecha
        if len(snaps) < 3:
            print(f"⚠️ Fecha {dt.date()}: solo {len(snaps)} snapshots válidos (necesarios: 3+)")
            continue

        # Precalcular precios de t y t+1 para ret_fwd (1M forward)
        dt_next = (dt + pd.offsets.MonthEnd(1)).normalize()
        price_t  = {t: _price_smart(t, dt)      for t in snaps.keys()}
        price_t1 = {t: _price_smart(t, dt_next) for t in snaps.keys()}

        # Para cada ticker, calcular subscores con peers de esa fecha
        for t in list(snaps.keys()):
            tdata = snaps[t]
            peers_list = _get_peers_same_scope(snaps, t)
            peers_data = [snaps[p] for p in peers_list]  # incluye al propio

            # Momentum history hasta dt
            phist = _momentum_prices_to_date(t, dt)

            val_s, _ = valuation_subscore(tdata, peers_data)
            qual_s, _ = quality_subscore(tdata, peers_data)
            grow_s, _ = growth_subscore(tdata, peers_data)
            mom_s,  _ = momentum_subscore(tdata, phist)

            # Retorno futuro 1m (t -> t+1)
            p0 = price_t.get(t, np.nan)
            p1 = price_t1.get(t, np.nan)
            ret_fwd = _safe_div(p1 - p0, p0) if (pd.notna(p0) and pd.notna(p1)) else np.nan

            rows.append({
                'date': dt,
                'ticker': t,
                'valuation': val_s,
                'quality': qual_s,
                'growth': grow_s,
                'momentum': mom_s,
                'ret_fwd': ret_fwd,
                'sector': tdata.get('sector', 'N/A'),
            })

    if not rows:
        return pd.DataFrame()

    panel = pd.DataFrame(rows).set_index(['date', 'ticker']).sort_index()

    # Limpieza: quitar filas sin ret_fwd o no finitas
    mask_ret = pd.notna(panel['ret_fwd'])
    try:
        mask_ret &= np.isfinite(panel['ret_fwd'].astype(float))
    except Exception:
        pass
    panel = panel[mask_ret]

    # (Opcional) quitar filas con todos los factores NaN
    fac_cols = ['valuation', 'quality', 'growth', 'momentum']
    if set(fac_cols).issubset(panel.columns):
        panel = panel[~panel[fac_cols].isna().all(axis=1)]

    return panel

# ---------- Motor de evaluación y búsqueda de pesos ----------

def _portfolio_long_short(scores: pd.Series, rets: pd.Series, sectors: Optional[pd.Series],
                          costs_bps: float = 5, neutralize_sector: bool = False) -> float:
    df = pd.DataFrame({'score': scores, 'ret': rets})
    
    # Verificar que tenemos datos válidos
    valid_data = df.dropna()
    if len(valid_data) < 3:
        return np.nan
    
    if neutralize_sector and sectors is not None:
        df['sector'] = sectors
        ls_by_sector = []
        for sctr, g in df.groupby('sector'):
            if len(g) < 10:
                continue
            q = g['score'].quantile([0.2, 0.8])
            long_ret = g[g['score'] >= q.loc[0.8]]['ret'].mean()
            short_ret = g[g['score'] <= q.loc[0.2]]['ret'].mean()
            if pd.notna(long_ret) and pd.notna(short_ret):
                ls_by_sector.append(long_ret - short_ret)
        if not ls_by_sector:
            return np.nan
        ls = float(np.mean(ls_by_sector))
    else:
        q = df['score'].quantile([0.2, 0.8])
        long_ret = df[df['score'] >= q.loc[0.8]]['ret'].mean()
        short_ret = df[df['score'] <= q.loc[0.2]]['ret'].mean()
        if pd.isna(long_ret) or pd.isna(short_ret):
            return np.nan
        ls = long_ret - short_ret

    # Costes por lado (simplificación): 2 * bps
    ls_net = ls - 2 * (costs_bps / 1e4)
    return ls_net

def _evaluate_weights(panel: pd.DataFrame, w: np.ndarray, neutralize_sector: bool = False, costs_bps: float = 5) -> Tuple[float, Dict]:
    assert np.isclose(w.sum(), 1.0), "Pesos deben sumar 1"
    assert (w >= 0).all(), "Pesos no negativos"

    # Score compuesto por fecha
    def _row_s(x):
        return w[0]*x['valuation'] + w[1]*x['quality'] + w[2]*x['growth'] + w[3]*x['momentum']

    panel_ = panel.copy()
    panel_['score'] = panel_[['valuation','quality','growth','momentum']].apply(_row_s, axis=1)

    # Long-short mensual
    ls_ret = []
    for dt, df_t in panel_.groupby(level=0):
        scores = df_t['score']
        rets   = df_t['ret_fwd']
        sectors = df_t['sector'] if 'sector' in df_t.columns else None
        r = _portfolio_long_short(scores, rets, sectors, costs_bps=costs_bps, neutralize_sector=neutralize_sector)
        if pd.notna(r) and np.isfinite(r):
            ls_ret.append(r)

    if not ls_ret:
        return -np.inf, {}

    rets = np.array(ls_ret, dtype=float)
    mu, sd = np.nanmean(rets), np.nanstd(rets, ddof=1)
    sharpe = (mu / (sd + 1e-9)) * np.sqrt(12)  # mensual -> anual
    stats = {
        'ann_ret': (1.0 + mu)**12 - 1.0,
        'ann_vol': sd * np.sqrt(12),
        'sharpe': sharpe,
        'n_periods': int(len(rets)),
    }
    return sharpe, stats

def _grid_simplex(grid: int = 5) -> List[np.ndarray]:
    vals = np.linspace(0, 1, grid)
    cand = []
    for wv in vals:
        for wq in vals:
            for wg in vals:
                wm = 1 - (wv + wq + wg)
                if wm < 0:
                    continue
                w = np.array([wv, wq, wg, wm], dtype=float)
                cand.append(w)
    return cand

def grid_search_weights(panel: pd.DataFrame, grid: int = 6, neutralize_sector: bool = True, costs_bps: float = 5) -> Tuple[float, np.ndarray, Dict]:
    best = (-np.inf, None, None)
    valid_combinations = 0
    
    for w in _grid_simplex(grid):
        sharpe, stats = _evaluate_weights(panel, w, neutralize_sector=neutralize_sector, costs_bps=costs_bps)
        if np.isfinite(sharpe) and sharpe > -np.inf:
            valid_combinations += 1
            if sharpe > best[0]:
                best = (sharpe, w, stats)
    
    print(f"🔍 Grid search: {valid_combinations} combinaciones válidas de {len(_grid_simplex(grid))}")
    if valid_combinations == 0:
        print("⚠️ Ninguna combinación de pesos produjo retornos válidos")
    
    return best  # (best_sharpe, best_weights, stats)

# ---------- Orquestador: construir panel + buscar pesos ----------

# ---------- Orquestador: construir panel + buscar pesos ----------

def recalibrar_pesos_empirico(tickers: List[str],
                              start: str = "2019-01-01",
                              end: str   = None,
                              freq: str = "M",
                              lag_days: int = 30,
                              grid: int = 6,
                              neutralize_sector: bool = True,
                              costs_bps: float = 5.0) -> Dict:
    """
    Construye panel PIT (mensual), ejecuta grid search de pesos y devuelve el mejor set.
    Retorna: {'weights': array([wV,wQ,wG,wM]), 'sharpe': float, 'stats': dict, 'panel_shape': tuple}
    """
    try:
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")
        
        print(f"🔨 Construyendo panel PIT para {len(tickers)} tickers...")
        panel = build_panel_pit(tickers, start=start, end=end, freq=freq, lag_days=lag_days)
        
        if panel.empty:
            print("❌ Panel vacío. Revisa tickers/fechas o disponibilidad de datos.")
            return {}

        print(f"📊 Panel construido: {panel.shape[0]} observaciones, {panel.shape[1]} columnas")
        
        best_sharpe, best_w, stats = grid_search_weights(panel, grid=grid, neutralize_sector=neutralize_sector, costs_bps=costs_bps)
        
        if best_w is None:
            print("❌ No se pudieron encontrar pesos óptimos.")
            return {}
            
    except Exception as e:
        print(f"❌ Error en recalibrar_pesos_empirico: {e}")
        return {}

    # Mostrar resultado
    print("\n🧪 Recalibración empírica de pesos (long-short Top20% vs Bottom20%)")
    print(f"Periodo: {panel.index.get_level_values(0).min().date()} → {panel.index.get_level_values(0).max().date()}  |  Muestras mensuales: {stats.get('n_periods', 'N/A')}")
    print(f"Neutralización por sector: {'Sí' if neutralize_sector else 'No'}  |  Costes: {costs_bps} bps por lado")
    print(f"Mejor Sharpe anualizado: {stats.get('sharpe', np.nan):.2f}")
    print(f"Retorno anualizado (aprox): {stats.get('ann_ret', np.nan):.2%}  |  Vol: {stats.get('ann_vol', np.nan):.2%}")
    print(f"Pesos óptimos: valuation={best_w[0]:.2f}  quality={best_w[1]:.2f}  growth={best_w[2]:.2f}  momentum={best_w[3]:.2f}")
    return {'weights': best_w, 'sharpe': best_sharpe, 'stats': stats, 'panel_shape': panel.shape}


# ================================ MAIN ================================

if __name__ == "__main__":
    print("📦 MÓDULO DE ANÁLISIS DE VALORACIÓN DE STOCKS v3 (parcheado)")
    print("="*60)
    print("Sistema de valoración relativa por sector/industria")
    print("Con métricas de calidad, crecimiento y momentum")
    print("="*60)

    # Ejecutar tests (opcional)
    # test_scoring()

    print("\n" + "="*60)
    print("🧭 EJEMPLOS DE USO")
    print("="*60)

    # Ejemplo 1: Análisis individual
    ejemplo_analisis_individual()

    # Ejemplo 2: Comparación
    ejemplo_comparacion()

    # Ejemplo 3: Recalibración empírica de pesos (opcional)
    print("\n" + "="*60)
    print("🧪 Recalibración empírica de WEIGHTS (long-short Top20% vs Bottom20%)")
    print("="*60)
    try:
        # Smoke test (opcional, para verificar histórico disponible)
        try:
            s = _prices_series_cached("AAPL", years=6)
            print("🔎 AAPL barras:", len(s), "| última fecha:", (s.index[-1] if not s.empty else None))
        except Exception as _e:
            print("⚠️ No se pudo leer histórico de AAPL:", _e)

        tickers_empirico = ["AAPL","MSFT","GOOGL","META","NVDA","AMZN","ORCL","ADBE","CRM"]
        res = recalibrar_pesos_empirico(
            tickers=tickers_empirico,
            start="2020-01-01",
            end=datetime.now().strftime("%Y-%m-%d"),
            freq="M",            # rebalanceo mensual
            lag_days=5,          # lag reducido para tener más datos
            grid=6,              # densidad del grid en el simplex
            neutralize_sector=False,  # desactivar neutralización por sector
            costs_bps=5.0        # costes por lado (bps)
        )
        if res and 'weights' in res:
            w = res['weights']
            print(f"\n🔧 Pesos sugeridos (empírico): valuation={w[0]:.2f} quality={w[1]:.2f} growth={w[2]:.2f} momentum={w[3]:.2f}")
            print(f"   Sharpe anualizado (OOS aprox): {res.get('sharpe', float('nan')):.2f}")
            print(f"   Panel (filas, cols): {res.get('panel_shape', None)}")

            # Aplicación opcional de los pesos hallados:
            APPLY_EMPIRICAL_WEIGHTS = False  # <- pon True si deseas actualizar WEIGHTS automáticamente
            if APPLY_EMPIRICAL_WEIGHTS:
                WEIGHTS.update({
                    "valuation": float(w[0]),
                    "quality":   float(w[1]),
                    "growth":    float(w[2]),
                    "momentum":  float(w[3]),
                })
                print("✅ WEIGHTS actualizados:", WEIGHTS)
        else:
            print("⚠️ No se pudo recalibrar pesos (panel vacío o error).")
    except NameError:
        print("⚠️ Falta 'recalibrar_pesos_empirico'. Asegúrate de pegar el bloque EMPÍRICO completo.")
    except Exception as e:
        print(f"⚠️ Recalibración empírica omitida por error: {e}")

    print("\n🎉 Análisis completado!")
    print("="*60)
