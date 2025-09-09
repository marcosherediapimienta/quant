"""
Módulo de Análisis de Valoración de Stocks (v4)
==========================================================
Cambios clave v4:
- FCF TTM: no invertir signo de CapEx; se resta tal como viene.
- P/E Forward: percentil contra peers de P/E Forward (no TTM).
- Deuda: evitar doble conteo; preferir Total Debt si existe.
- relative_available: contar peers realmente válidos (con datos).
- Fecha de precios: basada en el ticker analizado (fallback: SPY).
- Robustez: ordenar columnas por fecha en DF de yfinance.
"""

from __future__ import annotations

import math
import warnings
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

# ====== utilidades para extraer TTM y balance de yfinance ======
CF_CFO_KEYS = [
    "Total Cash From Operating Activities", "Operating Cash Flow",
    "Net Cash Provided by Operating Activities", "Net CashFrom Operating Activities"
]
CF_CAPEX_KEYS = [
    "Capital Expenditures", "Capital Expenditure", "Purchase Of Property Plant And Equipment"
]
BS_EQUITY_KEYS = [
    "Total Stockholder Equity", "Total Stockholders Equity", "Stockholders Equity",
    "Total Equity Gross Minority Interest"
]
BS_CASH_KEYS = [
    "Cash And Cash Equivalents", "Cash", "Cash And Cash Equivalents And Short Term Investments"
]
BS_DEBT_KEYS = [
    "Total Debt", "Short Long Term Debt", "Long Term Debt",
    "Current Debt And Capital Lease Obligation", "Long Term Debt And Capital Lease Obligation"
]

def _order_cols_by_date(df: pd.DataFrame) -> pd.DataFrame:
    """Intenta ordenar columnas por fecha descendente (más reciente primero)."""
    try:
        cols = pd.to_datetime(df.columns, errors="coerce")
        if getattr(cols, "notna", lambda: False)().any():
            order = np.argsort(cols)[::-1]
            return df.iloc[:, order]
    except Exception:
        pass
    return df

def _get_row_sum_last_n(df: pd.DataFrame, keys: List[str], n: int = 4) -> Optional[float]:
    if df is None or df.empty:
        return None
    df = _order_cols_by_date(df)
    for k in keys:
        if k in df.index:
            s = df.loc[k].dropna()
            if not s.empty:
                return float(s.iloc[:n].sum())
    return None

def _get_row_last(df: pd.DataFrame, keys: List[str]) -> Optional[float]:
    if df is None or df.empty:
        return None
    df = _order_cols_by_date(df)
    for k in keys:
        if k in df.index:
            s = df.loc[k].dropna()
            if not s.empty:
                return float(s.iloc[0])
    return None

def _compute_fcfe_ttm(stock: yf.Ticker) -> Optional[float]:
    """
    FCFE TTM (yfinance) = CFO_TTM − CapEx_TTM
    Nota: en yfinance el CapEx suele venir NEGATIVO (outflow). Por eso:
          CFO − CapEx_neg = CFO + |CapEx|.
    """
    try:
        qcf = stock.quarterly_cashflow
    except Exception:
        qcf = None

    cfo_4q   = _get_row_sum_last_n(qcf, CF_CFO_KEYS, 4)
    capex_4q = _get_row_sum_last_n(qcf, CF_CAPEX_KEYS, 4)

    # Si no hay TTM, usar anual
    if cfo_4q is None and capex_4q is None:
        try:
            acf = stock.cashflow
        except Exception:
            acf = None
        if acf is None or acf.empty:
            return None
        cfo   = _get_row_last(acf, CF_CFO_KEYS)
        capex = _get_row_last(acf, CF_CAPEX_KEYS)
        if cfo is None or capex is None:
            return None
        # FCFE = CFO − CapEx (CapEx suele ser negativo)
        return float(cfo - capex)

    if cfo_4q is None or capex_4q is None:
        return None

    # FCFE = CFO − CapEx (CapEx suele ser negativo)
    return float(cfo_4q - capex_4q)

def _compute_fcff_ttm(stock: yf.Ticker, tax_rate: float = 0.21) -> Optional[float]:
    """
    FCFF TTM (yfinance) = CFO_TTM − CapEx_TTM + Interest_Expense_TTM * (1 − tax_rate)
    Razón: CFO ya está después de intereses en US GAAP; para FCFF hay que
            sumar de vuelta el gasto financiero neto después de impuestos.
    Convenciones de signos en yfinance:
      - CapEx suele venir NEGATIVO (outflow)  ⇒ usar (− CapEx)  ≡  +|CapEx|
      - Interest Expense suele venir NEGATIVO ⇒ usar abs() antes de (1 − t)
    """
    try:
        qcf = stock.quarterly_cashflow
    except Exception:
        qcf = None

    cfo_4q   = _get_row_sum_last_n(qcf, CF_CFO_KEYS, 4)
    capex_4q = _get_row_sum_last_n(qcf, CF_CAPEX_KEYS, 4)

    # Interest Expense (4Q)
    interest_expense_keys = [
        "Interest Expense", "Interest Paid", "Net Interest Income",
        "Interest Income", "Interest And Debt Expense"
    ]
    interest_4q = _get_row_sum_last_n(qcf, interest_expense_keys, 4)

    # Fallback anual si falta TTM
    if cfo_4q is None and capex_4q is None:
        try:
            acf = stock.cashflow
        except Exception:
            acf = None
        if acf is None or acf.empty:
            return None
        cfo      = _get_row_last(acf, CF_CFO_KEYS)
        capex    = _get_row_last(acf, CF_CAPEX_KEYS)
        interest = _get_row_last(acf, interest_expense_keys)
        if cfo is None or capex is None:
            return None

        fcff = (cfo - capex)  # − CapEx (CapEx suele ser negativo)
        if interest is not None:
            fcff += abs(interest) * (1 - tax_rate)
        return float(fcff)

    if cfo_4q is None or capex_4q is None:
        return None

    fcff = (cfo_4q - capex_4q)  # − CapEx (CapEx suele ser negativo)
    if interest_4q is not None:
        fcff += abs(interest_4q) * (1 - tax_rate)

    return float(fcff)

def _get_equity_last(stock: yf.Ticker) -> Optional[float]:
    for getter in ("quarterly_balance_sheet", "balance_sheet"):
        try:
            bs = getattr(stock, getter)
            eq = _get_row_last(bs, BS_EQUITY_KEYS)
            if eq is not None:
                return float(eq)
        except Exception:
            continue
    return None

def _get_cash_debt_last(stock: yf.Ticker) -> Tuple[Optional[float], Optional[float]]:
    """
    Obtiene caja y deuda evitando doble conteo.
    - Si existe 'Total Debt', úsalo directamente.
    - Si no, suma componentes relevantes.
    """
    cash = debt = None
    for getter in ("quarterly_balance_sheet", "balance_sheet"):
        try:
            bs = getattr(stock, getter)
            if bs is None or bs.empty:
                continue

            if cash is None:
                cash = _get_row_last(bs, BS_CASH_KEYS)

            if debt is None:
                if "Total Debt" in bs.index:
                    debt = _get_row_last(bs, ["Total Debt"])
                else:
                    comp_keys = [
                        "Short Long Term Debt",
                        "Current Debt And Capital Lease Obligation",
                        "Long Term Debt",
                        "Long Term Debt And Capital Lease Obligation",
                    ]
                    vals = []
                    for k in comp_keys:
                        v = _get_row_last(bs, [k])
                        if v is not None:
                            vals.append(float(v))
                    debt = float(sum(vals)) if vals else None

            if cash is not None and debt is not None:
                break
        except Exception:
            continue
    return cash, debt

def _compute_ev_corporativo(info: Dict, stock: yf.Ticker) -> Optional[float]:
    try:
        mc = float(info.get("marketCap") or 0)
        cash_i = info.get("totalCash")
        debt_i = info.get("totalDebt")
        cash_bs, debt_bs = _get_cash_debt_last(stock)
        cash = float(cash_bs if cash_bs is not None else (cash_i or 0))
        debt = float(debt_bs if debt_bs is not None else (debt_i or 0))
        if mc <= 0:
            return None
        return mc + debt - cash
    except Exception:
        return None

def _equity_nonpositive(stock: yf.Ticker) -> bool:
    eq = _get_equity_last(stock)
    try:
        return (eq is not None) and (float(eq) <= 0)
    except Exception:
        return False

def _latest_fund_date(stock: yf.Ticker) -> Optional[str]:
    # Toma la columna más reciente de los DF trimestrales/anuales disponibles
    dfs = []
    for attr in ("quarterly_financials", "quarterly_cashflow", "quarterly_balance_sheet",
                 "financials", "cashflow", "balance_sheet"):
        try:
            df = getattr(stock, attr)
            if df is not None and not df.empty:
                dfs.append(df)
        except Exception:
            pass
    if not dfs:
        return None
    try:
        latest = max([c for df in dfs for c in df.columns if pd.notna(c)])
        if isinstance(latest, (pd.Timestamp, np.datetime64)):
            return pd.to_datetime(latest).strftime("%Y-%m-%d")
        return str(latest)
    except Exception:
        return None

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
    """Calcula RSI de una serie de precios (promedio simple)."""
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
    """Formatea números (mantener para compatibilidad)."""
    return formatear_numero(x, fmt, na)

def fmt_int(x, na="N/D"):
    """Formatea enteros."""
    return formatear_numero(x, "{:,.0f}", na)

def fmt_pct(x, fmt="{:.2%}", na="N/D"):
    """Formatea porcentajes (mantener para compatibilidad)."""
    return formatear_porcentaje(x, fmt, na)

def _clean_multiple(x: float, allow_zero: bool = False) -> float:
    """
    Devuelve NaN si el múltiplo es inválido/no finito o ≤0 (o <0 si allow_zero=True).
    Evita tratar múltiplos "negativos" como baratos (suelen reflejar pérdidas).
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

        # FCFE TTM (Free Cash Flow to Equity) - Flujo para accionistas
        fcfe_ttm = _compute_fcfe_ttm(stock)
        fcfe_yield = _safe_div(fcfe_ttm, market_cap) if pd.notna(fcfe_ttm) and pd.notna(market_cap) and market_cap > 0 else np.nan

        # FCFF TTM (Free Cash Flow to Firm) - Flujo para la empresa
        fcff_ttm = _compute_fcff_ttm(stock, tax_rate=0.21)
        
        # EV corporativo
        ev_corp = _compute_ev_corporativo(info, stock)
        fcff_ev_yield = _safe_div(fcff_ttm, ev_corp) if pd.notna(fcff_ttm) and pd.notna(ev_corp) and ev_corp > 0 else np.nan

        # Mantener compatibilidad con fcf_ttm (alias FCFE)
        fcf_ttm = fcfe_ttm
        fcf_yield = fcfe_yield
        fcf_ev_yield = fcff_ev_yield

        # Equity nonpositive
        equity_nonpositive = _equity_nonpositive(stock)
        if equity_nonpositive:
            pb = np.nan
            roe = np.nan

        # Net cash/EBITDA
        total_cash = _nan_if_missing(info.get('totalCash'))
        total_debt = _nan_if_missing(info.get('totalDebt'))
        net_cash = (total_cash if pd.notna(total_cash) else 0.0) - (total_debt if pd.notna(total_debt) else 0.0)
        ebitda = _nan_if_missing(info.get('ebitda'))
        net_cash_ebitda = _safe_div(net_cash, ebitda) if pd.notna(ebitda) and ebitda != 0 else np.nan

        # Fecha de fundamentales real (o fallback)
        fund_date = _latest_fund_date(stock) or _get_fundamental_date(info)

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
            'fcf_ttm': fcf_ttm,
            'fcf_ev_yield': fcf_ev_yield,
            'fcfe_ttm': fcfe_ttm,
            'fcfe_yield': fcfe_yield,
            'fcff_ttm': fcff_ttm,
            'fcff_ev_yield': fcff_ev_yield,

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
                'price_date': _get_latest_trading_date(ticker),
                'fund_date': fund_date
            },
            'source': 'yfinance',
            'ev_corp': ev_corp,
            'equity_nonpositive': equity_nonpositive
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
            'source': 'error',
            'ev_corp': np.nan,
            'equity_nonpositive': False
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
            "Financial Services": ["JPM", "BAC", "WFC", "GS", "MS"],
            "Credit Services": ["V", "MA", "AXP", "COF", "SYF", "FI", "FIS", "GPN", "ADYEN.AS", "PYPL", "AFRM", "SOFI", "LC", "UPST"],
            "Payments": ["V", "MA", "AXP", "COF", "SYF", "FI", "FIS", "GPN", "ADYEN.AS", "PYPL", "AFRM", "SOFI", "LC", "UPST"]
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

def _get_latest_trading_date(ticker: str = "SPY") -> str:
    """Obtiene la fecha de trading más reciente usando el ticker dado (fallback: SPY)."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='5d')
        if not hist.empty:
            return hist.index[-1].strftime('%Y-%m-%d')
    except Exception:
        pass
    return datetime.now().strftime('%Y-%m-%d')

def _get_fundamental_date(info: Dict) -> str:
    """Obtiene fecha de datos fundamentales (fallback al último día del año anterior)."""
    año_actual = datetime.now().year
    año_anterior = año_actual - 1
    fecha_fundamentales = datetime(año_anterior, 12, 31)
    return fecha_fundamentales.strftime('%Y-%m-%d')

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

    # 🔧 peers correctos para P/E forward
    pe_fwd_peers = [_clean_multiple(p.get('pe_fwd')) for p in peers_data]
    pe_fwd_peers = [v for v in pe_fwd_peers if pd.notna(v)]

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

    # P/E Forward (usar peers forward)
    if pd.notna(pe_fwd) and pe_fwd_peers:
        pe_fwd_pct = percentile_rank(pe_fwd, pe_fwd_peers)
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

    # Calcular rendimientos (aritméticos)
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

# ================================ DCF (VALORACIÓN ABSOLUTA) ================================

def _get_net_debt_for_ticker(ticker: str) -> Optional[float]:
    """
    Deuda neta = Deuda total - Caja, usando balance (quarterly si existe).
    Prioriza 'Total Debt'; si no está, suma componentes de deuda.
    """
    try:
        stock = yf.Ticker(ticker)
        cash, debt = _get_cash_debt_last(stock)  # ya existe en tu módulo
        if cash is None and debt is None:
            info = stock.info or {}
            cash = info.get("totalCash", None)
            debt = info.get("totalDebt", None)
        if cash is None or debt is None:
            return None
        return float(debt - cash)
    except Exception:
        return None

def dcf_2stage_fcff(
    fcf0: float,
    wacc: float,
    g_years: float = 0.08,
    years: int = 5,
    g_terminal: float = 0.025,
    mid_year: bool = True
) -> float:
    """
    DCF 2 etapas con FCFF:
      - Proyecta FCFF durante 'years' años con crecimiento g_years.
      - Valor terminal = FCFF_yearN * (1 + g_terminal) / (WACC - g_terminal).
      - Devuelve EV (Enterprise Value) en valor presente.
      - mid_year=True: descuenta cada flujo con t-0.5 periodos.
    Nota: requiere WACC > g_terminal.
    """
    if fcf0 is None or not np.isfinite(fcf0):
        return np.nan
    if wacc <= g_terminal:
        return np.nan

    fcfs = []
    fcf_t = float(fcf0)
    for t in range(1, years + 1):
        fcf_t *= (1.0 + g_years)
        fcfs.append(fcf_t)

    # Descuento con convención mid-year opcional
    if mid_year:
        pv_fcfs = sum(f / ((1.0 + wacc) ** (t - 0.5)) for t, f in enumerate(fcfs, start=1))
        fcf_term = fcfs[-1] * (1.0 + g_terminal)
        tv = fcf_term / (wacc - g_terminal)
        pv_tv = tv / ((1.0 + wacc) ** (years - 0.5))
    else:
        pv_fcfs = sum(f / ((1.0 + wacc) ** t) for t, f in enumerate(fcfs, start=1))
        fcf_term = fcfs[-1] * (1.0 + g_terminal)
        tv = fcf_term / (wacc - g_terminal)
        pv_tv = tv / ((1.0 + wacc) ** years)

    ev = pv_fcfs + pv_tv
    return float(ev)

def dcf_2stage_fcfe(
    fcfe0: float,
    cost_of_equity: float,
    g_years: float = 0.08,
    years: int = 5,
    g_terminal: float = 0.025,
    mid_year: bool = True
) -> float:
    """
    DCF 2 etapas con FCFE:
      - Proyecta FCFE durante 'years' años con crecimiento g_years.
      - Valor terminal = FCFE_yearN * (1 + g_terminal) / (Re - g_terminal).
      - Devuelve Equity Value directamente (no resta deuda neta).
      - mid_year=True: descuenta cada flujo con t-0.5 periodos.
    Nota: requiere cost_of_equity > g_terminal.
    """
    if fcfe0 is None or not np.isfinite(fcfe0):
        return np.nan
    if cost_of_equity <= g_terminal:
        return np.nan

    fcfes = []
    fcfe_t = float(fcfe0)
    for t in range(1, years + 1):
        fcfe_t *= (1.0 + g_years)
        fcfes.append(fcfe_t)

    # Descuento con convención mid-year opcional
    if mid_year:
        pv_fcfes = sum(f / ((1.0 + cost_of_equity) ** (t - 0.5)) for t, f in enumerate(fcfes, start=1))
        fcfe_term = fcfes[-1] * (1.0 + g_terminal)
        tv = fcfe_term / (cost_of_equity - g_terminal)
        pv_tv = tv / ((1.0 + cost_of_equity) ** (years - 0.5))
    else:
        pv_fcfes = sum(f / ((1.0 + cost_of_equity) ** t) for t, f in enumerate(fcfes, start=1))
        fcfe_term = fcfes[-1] * (1.0 + g_terminal)
        tv = fcfe_term / (cost_of_equity - g_terminal)
        pv_tv = tv / ((1.0 + cost_of_equity) ** years)

    equity_value = pv_fcfes + pv_tv
    return float(equity_value)

def compute_dcf_for_ticker_fcff(
    ticker: str,
    wacc: float = 0.09,
    g_years: float = 0.08,
    years: int = 5,
    g_terminal: float = 0.025,
    mid_year: bool = True,
    use_net_debt: bool = True
) -> Dict:
    """
    Calcula DCF 2 etapas usando FCFF (Free Cash Flow to Firm):
      - FCFF TTM (de _compute_fcff_ttm via fetch_snapshot)
      - Acciones en circulación
      - Deuda neta (si use_net_debt=True)
    Retorna valor razonable/acción y upside.
    """
    snap = fetch_snapshot(ticker)
    fcff_ttm = snap.get("fcff_ttm", np.nan)
    shares = snap.get("shares_outstanding", np.nan)
    price = snap.get("price", np.nan)

    if pd.isna(fcff_ttm) or not np.isfinite(fcff_ttm) or fcff_ttm <= 0:
        return {"ticker": ticker.upper(), "error": "FCFF TTM no disponible o no positivo (DCF no fiable)"}
    if pd.isna(shares) or not np.isfinite(shares) or shares <= 0:
        return {"ticker": ticker.upper(), "error": "Shares outstanding no disponible (DCF no calculable)"}

    ev = dcf_2stage_fcff(fcf0=fcff_ttm, wacc=wacc, g_years=g_years, years=years, g_terminal=g_terminal, mid_year=mid_year)
    if pd.isna(ev):
        return {"ticker": ticker.upper(), "error": "Parámetros inválidos: WACC debe ser > g_terminal."}

    net_debt = _get_net_debt_for_ticker(ticker) if use_net_debt else 0.0
    if pd.isna(net_debt):
        net_debt = 0.0  # fallback prudente

    equity_value = ev - net_debt
    fair_value_per_share = equity_value / shares

    upside = np.nan
    if pd.notna(price) and np.isfinite(price) and price > 0:
        upside = (fair_value_per_share / price) - 1.0

    return {
        "ticker": ticker.upper(),
        "method": "FCFF",
        "assumptions": {
            "wacc": wacc, "g_years": g_years, "years": years,
            "g_terminal": g_terminal, "mid_year": mid_year, "use_net_debt": use_net_debt
        },
        "inputs": {
            "fcff_ttm": fcff_ttm, "shares_outstanding": shares,
            "net_debt": net_debt, "price": price
        },
        "outputs": {
            "enterprise_value": ev,
            "equity_value": equity_value,
            "fair_value_per_share": fair_value_per_share,
            "upside": upside
        },
        "as_of": snap.get("as_of", {})
    }

def compute_dcf_for_ticker_fcfe(
    ticker: str,
    cost_of_equity: float = 0.12,
    g_years: float = 0.08,
    years: int = 5,
    g_terminal: float = 0.025,
    mid_year: bool = True
) -> Dict:
    """
    Calcula DCF 2 etapas usando FCFE (Free Cash Flow to Equity):
      - FCFE TTM (de _compute_fcfe_ttm via fetch_snapshot)
      - Acciones en circulación
      - No resta deuda neta (ya está implícito en FCFE)
    Retorna valor razonable/acción y upside.
    """
    snap = fetch_snapshot(ticker)
    fcfe_ttm = snap.get("fcfe_ttm", np.nan)
    shares = snap.get("shares_outstanding", np.nan)
    price = snap.get("price", np.nan)

    if pd.isna(fcfe_ttm) or not np.isfinite(fcfe_ttm) or fcfe_ttm <= 0:
        return {"ticker": ticker.upper(), "error": "FCFE TTM no disponible o no positivo (DCF no fiable)"}
    if pd.isna(shares) or not np.isfinite(shares) or shares <= 0:
        return {"ticker": ticker.upper(), "error": "Shares outstanding no disponible (DCF no calculable)"}

    equity_value = dcf_2stage_fcfe(fcfe0=fcfe_ttm, cost_of_equity=cost_of_equity, g_years=g_years, years=years, g_terminal=g_terminal, mid_year=mid_year)
    if pd.isna(equity_value):
        return {"ticker": ticker.upper(), "error": "Parámetros inválidos: cost_of_equity debe ser > g_terminal."}

    fair_value_per_share = equity_value / shares

    upside = np.nan
    if pd.notna(price) and np.isfinite(price) and price > 0:
        upside = (fair_value_per_share / price) - 1.0

    return {
        "ticker": ticker.upper(),
        "method": "FCFE",
        "assumptions": {
            "cost_of_equity": cost_of_equity, "g_years": g_years, "years": years,
            "g_terminal": g_terminal, "mid_year": mid_year
        },
        "inputs": {
            "fcfe_ttm": fcfe_ttm, "shares_outstanding": shares, "price": price
        },
        "outputs": {
            "equity_value": equity_value,
            "fair_value_per_share": fair_value_per_share,
            "upside": upside
        },
        "as_of": snap.get("as_of", {})
    }

def compute_dcf_for_ticker(
    ticker: str,
    wacc: float = 0.09,
    g_years: float = 0.08,
    years: int = 5,
    g_terminal: float = 0.025,
    use_net_debt: bool = True,
    method: str = "fcff",
    mid_year: bool = True
) -> Dict:
    """
    Función de conveniencia que llama a FCFF o FCFE según el método.
    Por defecto usa FCFF+WACC.
    """
    if method.lower() == "fcfe":
        # Para FCFE necesitamos estimar cost_of_equity
        wacc_data = estimate_wacc(ticker)
        if "error" in wacc_data:
            cost_of_equity = 0.12  # fallback
        else:
            cost_of_equity = wacc_data["cost_of_equity"]
        return compute_dcf_for_ticker_fcfe(ticker, cost_of_equity, g_years, years, g_terminal, mid_year)
    else:
        return compute_dcf_for_ticker_fcff(ticker, wacc, g_years, years, g_terminal, mid_year, use_net_debt)

def dcf_sensitivity(
    ticker: str,
    wacc_grid: List[float] = None,
    gterm_grid: List[float] = None,
    g_years: float = 0.08,
    years: int = 5,
    use_net_debt: bool = True
) -> pd.DataFrame:
    """
    Matriz de sensibilidad del DCF: filas = WACC, columnas = g_terminal.
    Devuelve DataFrame con valor razonable por acción.
    """
    if wacc_grid is None:
        wacc_grid = [0.07, 0.08, 0.09, 0.10, 0.11]
    if gterm_grid is None:
        gterm_grid = [0.01, 0.02, 0.025, 0.03, 0.035]

    snap = fetch_snapshot(ticker)
    fcf_ttm = snap.get("fcf_ttm", np.nan)
    shares = snap.get("shares_outstanding", np.nan)
    if pd.isna(fcf_ttm) or fcf_ttm <= 0 or pd.isna(shares) or shares <= 0:
        return pd.DataFrame()

    net_debt = _get_net_debt_for_ticker(ticker) if use_net_debt else 0.0
    if pd.isna(net_debt):
        net_debt = 0.0

    table = []
    for w in wacc_grid:
        row = []
        for gt in gterm_grid:
            if w <= gt:
                row.append(np.nan)
                continue
            ev = dcf_2stage_fcff(fcf0=fcf_ttm, wacc=w, g_years=g_years, years=years, g_terminal=gt)
            eq = (ev - net_debt) if pd.notna(ev) else np.nan
            pps = (eq / shares) if (pd.notna(eq) and shares > 0) else np.nan
            row.append(pps)
        table.append(row)

    df = pd.DataFrame(table, index=[f"WACC {w*100:.1f}%" for w in wacc_grid],
                      columns=[f"gT {gt*100:.1f}%" for gt in gterm_grid])
    return df

def demo_dcf(ticker: str = "AAPL"):
    """Demo rápida: imprime DCF y una pequeña sensibilidad."""
    print(f"\n🧮 DCF (2 etapas) para {ticker.upper()}")
    res = compute_dcf_for_ticker(ticker, wacc=0.09, g_years=0.08, years=5, g_terminal=0.025)
    if "error" in res:
        print("❌", res["error"])
        return

    fv = res["outputs"]["fair_value_per_share"]
    px = res["inputs"]["price"]
    up = res["outputs"]["upside"]

    print(f"  Valor razonable/acción: {fmt_num(fv)}")
    if pd.notna(px):
        print(f"  Precio actual:         {fmt_num(px)}")
    if pd.notna(up):
        print(f"  Upside estimado:       {fmt_pct(up)}")

    print("\n🔁 Sensibilidad (precio/acción):")
    sens = dcf_sensitivity(ticker, wacc_grid=[0.08, 0.09, 0.10], gterm_grid=[0.02, 0.025, 0.03])
    if sens.empty:
        print("   (No disponible: requiere FCF>0 y acciones)")
    else:
        sens_fmt = sens.applymap(lambda x: fmt_num(x) if pd.notna(x) else "N/D")
        print(sens_fmt.to_string())

def estimate_wacc(ticker: str, rf: float = 0.042, erp: float = 0.05, tax_rate: float = 0.21) -> Dict:
    """
    Estima el WACC para un ticker usando CAPM + estructura de capital con deuda bruta.
    
    Args:
        ticker: símbolo (ej. "META", "AAPL").
        rf: tasa libre de riesgo (ej. bono 10Y USD = 4.2%).
        erp: equity risk premium (USA ~5%).
        tax_rate: tasa impositiva efectiva (USA ~21%).
        
    Returns:
        dict con desglose de R_e, R_d, pesos E/D (deuda bruta) y WACC.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        
        # Market Cap y deuda bruta (no deuda neta para pesos)
        E = float(info.get("marketCap", 0) or 0)
        D = float(info.get("totalDebt", 0) or 0)
        
        # Pesos con deuda bruta
        V = E + D
        we = E / V if V > 0 else 1.0  # si no hay deuda, 100% equity
        wd = D / V if V > 0 else 0.0
        
        # Beta
        beta = info.get("beta", 1.0) or 1.0
        
        # Coste de equity (CAPM)
        Re = rf + beta * erp
        
        # Coste de deuda empírico
        Rd = _estimate_cost_of_debt(stock, info, D)
        Rd_after_tax = Rd * (1 - tax_rate)
        
        # WACC
        WACC = we * Re + wd * Rd_after_tax
        
        return {
            "ticker": ticker.upper(),
            "beta": beta,
            "cost_of_equity": Re,
            "cost_of_debt": Rd_after_tax,
            "cost_of_debt_before_tax": Rd,
            "weights": {"equity": we, "debt": wd},
            "wacc": WACC,
            "market_cap": E,
            "total_debt": D
        }
    except Exception as e:
        return {"ticker": ticker.upper(), "error": str(e)}

def _estimate_cost_of_debt(stock: yf.Ticker, info: Dict, total_debt: float) -> float:
    """
    Estima el coste de deuda empíricamente usando Interest Expense / Total Debt.
    Fallback por tramos de D/EBITDA si no hay datos de interés.
    """
    if total_debt <= 0:
        return 0.0
    
    # Intentar usar Interest Expense 4Q
    try:
        qcf = stock.quarterly_cashflow
        if qcf is not None and not qcf.empty:
            interest_expense_keys = [
                "Interest Expense", "Interest Paid", "Net Interest Income", 
                "Interest Income", "Interest And Debt Expense"
            ]
            interest_4q = _get_row_sum_last_n(qcf, interest_expense_keys, 4)
            if interest_4q is not None and interest_4q != 0:
                # Interest Expense suele venir negativo, usar abs()
                rd_empirical = abs(interest_4q) / total_debt
                return float(np.clip(rd_empirical, 0.01, 0.12))  # clamp 1%-12%
    except Exception:
        pass
    
    # Fallback anual
    try:
        acf = stock.cashflow
        if acf is not None and not acf.empty:
            interest_expense_keys = [
                "Interest Expense", "Interest Paid", "Net Interest Income", 
                "Interest Income", "Interest And Debt Expense"
            ]
            interest_annual = _get_row_last(acf, interest_expense_keys)
            if interest_annual is not None and interest_annual != 0:
                rd_empirical = abs(interest_annual) / total_debt
                return float(np.clip(rd_empirical, 0.01, 0.12))  # clamp 1%-12%
    except Exception:
        pass
    
    # Fallback por tramos de D/EBITDA
    ebitda = info.get("ebitda", None)
    if ebitda and ebitda > 0:
        ratio = total_debt / ebitda
        if ratio < 1:
            return 0.04  # 4%
        elif ratio < 2:
            return 0.05  # 5%
        elif ratio < 3:
            return 0.06  # 6%
        elif ratio < 4:
            return 0.07  # 7%
        else:
            return 0.08  # 8%
    
    # Fallback genérico
    return 0.045  # 4.5%
    
def _series_from_financials(df: pd.DataFrame, key: str) -> pd.Series:
    """Extrae una serie anual ordenada (más reciente primero) para 'key' desde un DF de yfinance."""
    if df is None or df.empty or key not in df.index:
        return pd.Series(dtype=float)
    df = _order_cols_by_date(df)
    s = df.loc[key].dropna()
    # columnas suelen ser timestamps (años). Devolver en orden DESC ya viene tras _order_cols_by_date
    return s

def _series_fcf_annual(stock: yf.Ticker) -> pd.Series:
    """
    Construye FCF anual = CFO - CapEx a partir de 'stock.cashflow' (anual).
    Nota: CapEx suele venir negativo en Yahoo; NO invertir signo.
    """
    try:
        acf = stock.cashflow
    except Exception:
        acf = None
    if acf is None or acf.empty:
        return pd.Series(dtype=float)
    acf = _order_cols_by_date(acf)
    # claves compatibles (ya usadas en tu módulo)
    cfo = None
    for k in CF_CFO_KEYS:
        if k in acf.index:
            cfo = acf.loc[k].dropna()
            break
    capex = None
    for k in CF_CAPEX_KEYS:
        if k in acf.index:
            capex = acf.loc[k].dropna()
            break
    if cfo is None or capex is None or cfo.empty or capex.empty:
        return pd.Series(dtype=float)
    # Alinear por columnas (fechas)
    aligned = pd.concat([cfo, capex], axis=1, join="inner")
    aligned.columns = ["CFO", "CapEx"]
    fcf = aligned["CFO"] + aligned["CapEx"]  # sin invertir signo
    # Ya está en orden DESC por _order_cols_by_date
    return fcf.dropna()

def _cagr_from_series_robusta(series: pd.Series, min_points: int = 3, max_points: int = 5) -> Optional[float]:
    """
    Calcula CAGR robusto usando entre 3 y 5 observaciones más recientes (anuales).
    
    Estrategias:
    1. Si todos los valores >0 → CAGR clásico
    2. Si hay ≤0 → mediana de crecimientos YoY sobre pares positivos
    3. Si no hay suficientes puntos válidos → None
    
    Args:
        series: Serie temporal ordenada (más reciente primero)
        min_points: Mínimo de puntos requeridos
        max_points: Máximo de puntos a usar
        
    Returns:
        CAGR o mediana de crecimientos YoY, o None si no hay suficientes datos
    """
    if series is None or series.empty:
        return None
    s = series.dropna().astype(float)
    if s.size < min_points:
        return None
    
    # Tomar las N más recientes (ya en orden DESC)
    N = min(max_points, s.size)
    sN = s.iloc[:N].iloc[::-1]  # ordenar ASC temporalmente (antiguo -> reciente)
    
    # Estrategia 1: Si todos los valores son positivos, usar CAGR clásico
    if (sN > 0).all():
        try:
            first, last = float(sN.iloc[0]), float(sN.iloc[-1])
            periods = N - 1
            if periods > 0 and first > 0 and last > 0:
                return (last / first) ** (1.0 / periods) - 1.0
        except Exception:
            pass
    
    # Estrategia 2: Calcular mediana de crecimientos YoY sobre pares positivos
    growth_rates = []
    for i in range(1, len(sN)):
        prev_val = float(sN.iloc[i-1])
        curr_val = float(sN.iloc[i])
        if prev_val > 0 and curr_val > 0:
            growth_rate = (curr_val / prev_val) - 1.0
            growth_rates.append(growth_rate)
    
    if len(growth_rates) >= min_points - 1:  # Necesitamos al menos min_points-1 crecimientos
        return float(np.median(growth_rates))
    
    return None

def _cagr_from_series(series: pd.Series, min_points: int = 3, max_points: int = 5) -> Optional[float]:
    """
    Alias para _cagr_from_series_robusta por compatibilidad.
    """
    return _cagr_from_series_robusta(series, min_points, max_points)

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def estimate_growth(ticker: str) -> Dict:
    """
    Estima g_years (fase explícita) y g_terminal a partir de:
      - CAGR de ingresos (3-5y)
      - CAGR de FCF (3-5y)
      - Heurística por país/sector para g_terminal
    Devuelve dict con sugerencias y diagnósticos.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        country = (info.get("country") or "USA").upper()
        sector = info.get("sector", "") or ""
        
        # 1) CAGR de ingresos (anual)
        try:
            fin_a = stock.financials  # anual
        except Exception:
            fin_a = None
        rev_series = _series_from_financials(fin_a, "Total Revenue")
        rev_cagr = _cagr_from_series(rev_series, min_points=3, max_points=5)

        # 2) CAGR de FCF (anual)
        fcf_series = _series_fcf_annual(stock)
        fcf_cagr = _cagr_from_series(fcf_series, min_points=3, max_points=5)

        # 3) Fusión de señales para g_years:
        #    - promediado con pesos (ingresos 60%, FCF 40%) si ambos existen.
        #    - si falta uno, usa el otro.
        if rev_cagr is not None and fcf_cagr is not None:
            g_years_raw = 0.60 * rev_cagr + 0.40 * fcf_cagr
        elif rev_cagr is not None:
            g_years_raw = rev_cagr
        elif fcf_cagr is not None:
            g_years_raw = fcf_cagr
        else:
            # fallback por sector si no hay datos
            if "Technology" in sector:
                g_years_raw = 0.07
            elif "Healthcare" in sector:
                g_years_raw = 0.05
            elif "Financial" in sector:
                g_years_raw = 0.04
            else:
                g_years_raw = 0.05

        # Saneado para no irnos a extremos en fase explícita
        # (en empresas de alto crecimiento podrías ampliar el techo)
        g_years_suggested = _clamp(g_years_raw, lo=0.00, hi=0.18)

        # 4) g_terminal por país/sector con límites prudentes (1%–3%)
        #    Regla base: PIB real + inflación objetivo (~2%–2.5% USA).
        if "USA" in country:
            base_gt = 0.022  # 2.2% como centro
        elif country in ("IRELAND", "NETHERLANDS", "SINGAPORE", "SWITZERLAND"):
            base_gt = 0.020
        elif country in ("BRAZIL", "INDIA", "MEXICO", "INDONESIA"):
            base_gt = 0.025  # economías con mayor crecimiento tendencial, pero prudentes
        else:
            base_gt = 0.020

        # Ajuste leve por sector (tecnología global tiende algo > media)
        if "Technology" in sector:
            base_gt += 0.002  # +0.2 pp
        elif "Utilities" in sector or "Energy" in sector:
            base_gt -= 0.003  # -0.3 pp

        g_terminal_suggested = _clamp(base_gt, lo=0.010, hi=0.030)

        return {
            "ticker": ticker.upper(),
            "suggestions": {
                "g_years": g_years_suggested,
                "g_terminal": g_terminal_suggested
            },
            "diagnostics": {
                "country": country,
                "sector": sector,
                "revenue_cagr_3to5y": rev_cagr,
                "fcf_cagr_3to5y": fcf_cagr,
                "revenue_series_recent": rev_series.iloc[:5].to_dict() if rev_series is not None and not rev_series.empty else {},
                "fcf_series_recent": fcf_series.iloc[:5].to_dict() if fcf_series is not None and not fcf_series.empty else {}
            }
        }
    except Exception as e:
        return {"ticker": ticker.upper(), "error": f"estimate_growth failed: {e}"}

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

    # Verificar disponibilidad relativa basada en peers válidos (excluyendo el propio)
    valid_peer_count = sum(1 for p in peers_data if p['ticker'] != ticker and pd.notna(p.get('price')))
    relative_available = valid_peer_count >= MIN_PEERS

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
            'market_cap': ticker_data['market_cap'],
            'ev_corp': ticker_data.get('ev_corp', np.nan)
        },
        'valuation': {
            'pe_ttm': ticker_data['pe_ttm'],
            'pe_fwd': ticker_data['pe_fwd'],
            'ev_ebitda': ticker_data['ev_ebitda'],
            'ps': ticker_data['ps'],
            'pb': ticker_data['pb'],
            'fcf_yield': ticker_data['fcf_yield'],
            'fcf_ttm': ticker_data.get('fcf_ttm', np.nan),
            'fcf_ev_yield': ticker_data.get('fcf_ev_yield', np.nan),
            'fcfe_ttm': ticker_data.get('fcfe_ttm', np.nan),
            'fcfe_yield': ticker_data.get('fcfe_yield', np.nan),
            'fcff_ttm': ticker_data.get('fcff_ttm', np.nan),
            'fcff_ev_yield': ticker_data.get('fcff_ev_yield', np.nan),
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
        'peers_used': valid_peer_count,       # peers válidos realmente usados
        'peers_scope': RELATIVE_SCOPE,
        'relative_available': relative_available,
        'meta': {
            'equity_nonpositive': ticker_data.get('equity_nonpositive', False)
        }
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
    report += f"📊 REPORTE DE VALORACIÓN v4 - {ticker}\n"
    report += f"{'='*60}\n"

    # Información básica
    report += f"\n INFORMACIÓN BÁSICA:\n"
    report += f"   Precio: ${fmt_num(basics['price'])}\n"
    report += f"   Market Cap: ${fmt_int(basics['market_cap'])}\n"
    if pd.notna(basics.get('ev_corp', np.nan)):
        report += f"   EV (corp): ${fmt_int(basics['ev_corp'])}\n"
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
    
    # P/B y ROE con renderizado NM
    equity_nonpos = result.get('meta', {}).get('equity_nonpositive', False)
    pb_val = valuation['pb']
    roe_val = quality['roe']
    
    pb_str = "NM" if (equity_nonpos and pd.isna(pb_val)) else fmt_num(pb_val)
    roe_str = "NM" if (equity_nonpos and pd.isna(roe_val)) else fmt_pct(roe_val)
    
    report += f"   P/B: {pb_str}\n"
    report += f"   FCFE Yield: {fmt_pct(valuation.get('fcfe_yield', np.nan))}\n"
    
    # Métricas adicionales FCFE/FCFF
    if pd.notna(valuation.get('fcfe_ttm', np.nan)):
        report += f"   FCFE TTM: ${fmt_int(valuation['fcfe_ttm'])}\n"
    if pd.notna(basics.get('ev_corp', np.nan)) and pd.notna(valuation.get('fcff_ev_yield', np.nan)):
        report += f"   FCFF/EV Yield: {fmt_pct(valuation['fcff_ev_yield'], fmt='{:.1%}')}\n"
    if pd.notna(valuation.get('fcff_ttm', np.nan)):
        report += f"   FCFF TTM: ${fmt_int(valuation['fcff_ttm'])}\n"
    
    report += f"   Valuation Subscore: {fmt_num(valuation['subscore'], fmt='{:.2f}')}\n"

    # Calidad
    report += f"\n🏆 CALIDAD:\n"
    report += f"   ROIC: {fmt_pct(quality['roic'])}\n"
    report += f"   Margen Operativo: {fmt_pct(quality['op_margin'])}\n"
    report += f"   Margen Neto: {fmt_pct(quality['profit_margin'])}\n"
    report += f"   ROE: {roe_str}\n"
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
    report += f"   Peers usados (válidos): {result['peers_used']} ({result['peers_scope']})\n"

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
    summary += f"📊 RESUMEN DE PORTAFOLIO - VALORACIÓN v4\n"
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

def crear_graficos_valoracion(result: Dict, output_path: str = "./plots") -> None:
    """Crea gráficos de valoración para un stock."""
    if 'error' in result:
        print(f"❌ No se pueden crear gráficos para {result['ticker']}: {result['error']}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Análisis de Valoración v4 - {result["ticker"]}', fontsize=16, fontweight='bold')

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

    for bar, score in zip(bars, scores):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                       f'{score:.2f}', ha='center', va='bottom' if height >= 0 else 'top')

    # 2) Métricas de valoración
    val_metrics = ['P/E TTM', 'P/E FWD', 'EV/EBITDA', 'P/S', 'P/B', 'FCF Yield', 'FCF/EV Yield']
    val_values = [
        result['valuation']['pe_ttm'],
        result['valuation']['pe_fwd'],
        result['valuation']['ev_ebitda'],
        result['valuation']['ps'],
        result['valuation']['pb'],
        result['valuation']['fcf_yield'] * 100 if not np.isnan(result['valuation']['fcf_yield']) else np.nan,
        result['valuation']['fcf_ev_yield'] * 100 if not np.isnan(result['valuation']['fcf_ev_yield']) else np.nan
    ]

    valid_metrics = []
    valid_values = []
    for m, v in zip(val_metrics, val_values):
        if v is not None and not np.isnan(v):
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

    valid_qual_metrics = []
    valid_qual_values = []
    for m, v in zip(qual_metrics, qual_values):
        if v is not None and not np.isnan(v):
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

    valid_mom_metrics = []
    valid_mom_values = []
    for m, v in zip(mom_metrics, mom_values):
        if v is not None and not np.isnan(v):
            valid_mom_metrics.append(m)
            valid_mom_values.append(v)

    if valid_mom_metrics:
        colors = ['red' if m == 'RSI(14)' and v > 70 else 'green' if m == 'RSI(14)' and v < 30 else 'blue' for m, v in zip(valid_mom_metrics, valid_mom_values)]
        axes[1, 1].bar(valid_mom_metrics, valid_mom_values, alpha=0.7, color=colors)
        axes[1, 1].set_title('Momentum y Técnico', fontweight='bold')
        axes[1, 1].set_ylabel('Valor')
        axes[1, 1].tick_params(axis='x', rotation=45)

        if 'RSI(14)' in valid_mom_metrics:
            axes[1, 1].axhline(y=70, color='red', linestyle='--', alpha=0.5, label='RSI > 70')
            axes[1, 1].axhline(y=30, color='green', linestyle='--', alpha=0.5, label='RSI < 30')
            axes[1, 1].legend()

    plt.tight_layout()
    
    # Crear directorio de salida si no existe
    import os
    os.makedirs(output_path, exist_ok=True)
    
    # Guardar gráfico
    ticker = result['ticker']
    filename = f"{output_path}/analisis_valoracion_{ticker}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico guardado en: {filename}")
    
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
        print(f"📊 COMPARACIÓN DE STOCKS - VALORACIÓN v4")
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

    # Tests extra v4: FCFE/FCFF y P/E forward
    print("\n5️⃣ Test FCFE con CapEx negativo (simulado)")
    _cfo = 300.0; _capex = -150.0
    fcfe_expected = _cfo + _capex  # 150 (FCFE = CFO + CapEx)
    fcfe_calc = _cfo + _capex
    print(f"   Esperado: {fcfe_expected}, Calculado: {fcfe_calc}")
    print("   ✅" if abs(fcfe_expected - fcfe_calc) < 1e-6 else "   ❌")

    print("\n6️⃣ Test FCFF con Interest Expense (simulado)")
    _cfo = 300.0; _capex = -150.0; _interest = -20.0; _tax_rate = 0.21
    fcff_expected = _cfo + _capex + abs(_interest) * (1 - _tax_rate)  # 300 + (-150) + 20*0.79 = 165.8
    fcff_calc = _cfo + _capex + abs(_interest) * (1 - _tax_rate)
    print(f"   Esperado: {fcff_expected}, Calculado: {fcff_calc}")
    print("   ✅" if abs(fcff_expected - fcff_calc) < 1e-6 else "   ❌")

    print("\n7️⃣ Test peers P/E forward vs TTM (deben diferir)")
    peers = [{'pe_ttm': 10, 'pe_fwd': 8}, {'pe_ttm': 20, 'pe_fwd': 18}]
    v = {'pe_ttm': 15, 'pe_fwd': 12}
    pct_ttm = percentile_rank(v['pe_ttm'], [p['pe_ttm'] for p in peers])
    pct_fwd = percentile_rank(v['pe_fwd'], [p['pe_fwd'] for p in peers])
    print(f"   pct_ttm={pct_ttm:.2f}, pct_fwd={pct_fwd:.2f}")
    print("   ✅" if abs(pct_ttm - pct_fwd) > 1e-6 else "   ❌")

    # Test 8: DCF smoke tests
    print("\n8️⃣ Test DCF smoke (parámetros válidos)")
    try:
        # Test FCFF DCF
        fcff_res = compute_dcf_for_ticker_fcff("AAPL", wacc=0.09, g_years=0.08, years=5, g_terminal=0.025)
        if "error" not in fcff_res and fcff_res["outputs"]["fair_value_per_share"] > 0:
            print("   ✅ DCF FCFF genera valor > 0")
        else:
            print("   ❌ DCF FCFF falló")
        
        # Test FCFE DCF
        fcfe_res = compute_dcf_for_ticker_fcfe("AAPL", cost_of_equity=0.12, g_years=0.08, years=5, g_terminal=0.025)
        if "error" not in fcfe_res and fcfe_res["outputs"]["fair_value_per_share"] > 0:
            print("   ✅ DCF FCFE genera valor > 0")
        else:
            print("   ❌ DCF FCFE falló")
    except Exception as e:
        print(f"   ❌ DCF tests fallaron: {e}")

    # Test 9: WACC validation
    print("\n9️⃣ Test WACC validation")
    try:
        wacc_res = estimate_wacc("AAPL")
        if "error" not in wacc_res:
            we = wacc_res["weights"]["equity"]
            wd = wacc_res["weights"]["debt"]
            wacc = wacc_res["wacc"]
            re = wacc_res["cost_of_equity"]
            rd = wacc_res["cost_of_debt"]
            
            # Verificar pesos suman ~1
            weights_ok = abs(we + wd - 1.0) < 0.01
            # Verificar WACC está entre min y max de costes
            wacc_range_ok = min(re, rd) <= wacc <= max(re, rd)
            # Verificar Rd razonable
            rd_ok = 0.01 <= rd <= 0.15
            
            print(f"   Pesos suman ~1: {weights_ok} (we={we:.3f}, wd={wd:.3f})")
            print(f"   WACC en rango: {wacc_range_ok} (WACC={wacc:.3f}, Re={re:.3f}, Rd={rd:.3f})")
            print(f"   Rd razonable: {rd_ok}")
            print("   ✅" if weights_ok and wacc_range_ok and rd_ok else "   ❌")
        else:
            print("   ❌ WACC estimation falló")
    except Exception as e:
        print(f"   ❌ WACC test falló: {e}")

    # Test 10: Mid-year DCF
    print("\n🔟 Test mid-year vs end-year DCF")
    try:
        # Mismo ticker, mismos parámetros, solo cambia mid_year
        res_mid = compute_dcf_for_ticker_fcff("AAPL", wacc=0.09, g_years=0.08, years=5, g_terminal=0.025, mid_year=True)
        res_end = compute_dcf_for_ticker_fcff("AAPL", wacc=0.09, g_years=0.08, years=5, g_terminal=0.025, mid_year=False)
        
        if "error" not in res_mid and "error" not in res_end:
            mid_val = res_mid["outputs"]["fair_value_per_share"]
            end_val = res_end["outputs"]["fair_value_per_share"]
            mid_higher = mid_val > end_val
            print(f"   Mid-year: {mid_val:.2f}, End-year: {end_val:.2f}")
            print(f"   Mid-year > End-year: {mid_higher}")
            print("   ✅" if mid_higher else "   ❌")
        else:
            print("   ❌ Mid-year DCF test falló")
    except Exception as e:
        print(f"   ❌ Mid-year test falló: {e}")

    print("\n🎉 Tests completados!")

def ejemplo_analisis_individual():
    """Ejecuta un análisis individual sencillo."""
    ticker = "AAPL"
    print("\n" + "="*60)
    print(f"🔹 Ejemplo 1: Análisis individual ({ticker})")
    print("="*60)
    analizar_stock(ticker)

def ejemplo_comparacion():
    """Compara un conjunto de grandes tecnológicas."""
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
    print("\n" + "="*60)
    print(f"🔹 Ejemplo 2: Comparación entre tickers {tickers}")
    print("="*60)
    comparar_stocks(tickers)


# ================================ FUNCIONES DE UTILIDAD ===============================

def formatear_numero(x, formato="{:,.2f}", na="N/D"):
    """Función unificada de formateo de números."""
    try:
        return na if pd.isna(x) or not np.isfinite(float(x)) else formato.format(float(x))
    except Exception:
        return na

def formatear_porcentaje(x, formato="{:.2%}", na="N/D"):
    """Formatea porcentajes."""
    try:
        return na if pd.isna(x) or not np.isfinite(float(x)) else formato.format(float(x))
    except Exception:
        return na

# ================================ FUNCIONES PRINCIPALES ===============================

def menu_principal():
    """Menú principal del sistema de análisis."""
    while True:
        print("\n" + "="*60)
        print("📊 SISTEMA DE ANÁLISIS DE VALORACIÓN DE STOCKS v4")
        print("="*60)
        print("1. Analizar stock individual")
        print("2. Comparar múltiples stocks")
        print("3. Análisis de portafolio")
        print("4. Generar reporte detallado con gráficos")
        print("5. Validar métricas de valoración")
        print("6. Ejecutar tests del sistema")
        print("7. Salir")
        print("8. Valorar por DCF (beta)")   # <--- NUEVO
        print("="*60)
        
        opcion = input("Seleccione una opción (1-8): ").strip()
        
        if opcion == "1":
            ticker = input("Ingrese el ticker del stock (ej: AAPL): ").strip().upper()
            if ticker:
                analizar_stock(ticker)
            else:
                print("❌ Ticker inválido")
                
        elif opcion == "2":
            tickers_input = input("Ingrese los tickers separados por comas (ej: AAPL,MSFT,GOOGL): ").strip()
            if tickers_input:
                tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
                if len(tickers) >= 2:
                    comparar_stocks(tickers)
                else:
                    print("❌ Se necesitan al menos 2 tickers para comparar")
            else:
                print("❌ Lista de tickers inválida")
                
        elif opcion == "3":
            tickers_input = input("Ingrese los tickers del portafolio separados por comas: ").strip()
            if tickers_input:
                tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
                if tickers:
                    analizar_portafolio(tickers)
                else:
                    print("❌ Lista de tickers inválida")
            else:
                print("❌ Lista de tickers inválida")
                
        elif opcion == "4":
            ticker = input("Ingrese el ticker para reporte detallado (ej: AAPL): ").strip().upper()
            if ticker:
                generar_reporte_detallado(ticker)
            else:
                print("❌ Ticker inválido")
                
        elif opcion == "5":
            ejemplo_validacion_metricas()
            
        elif opcion == "6":
            test_scoring()
            
        elif opcion == "7":
            print("👋 ¡Hasta luego!")
            break

        elif opcion == "8":  # DCF con WACC y crecimientos auto
            ticker = input("Ticker para DCF (ej: AAPL): ").strip().upper()
            if not ticker:
                print("❌ Ticker inválido")
                continue

            # 1) WACC auto
            res_wacc = estimate_wacc(ticker)
            if "error" in res_wacc:
                wacc = 0.09
                print("⚠️ WACC auto no disponible, usando 9%")
            else:
                wacc = res_wacc["wacc"]
                print(f"📊 WACC estimado: {wacc:.2%} (β={res_wacc['beta']:.2f})")

            # 2) Crecimientos auto
            eg = estimate_growth(ticker)
            if "error" in eg:
                g_years_auto = 0.08
                g_term_auto = 0.022
                print("⚠️ Crecimientos auto no disponibles; usando 8% y 2.2%")
            else:
                g_years_auto = eg["suggestions"]["g_years"]
                g_term_auto  = eg["suggestions"]["g_terminal"]
                print(f"📈 Sugerencias de crecimiento → g_years={g_years_auto:.2%}, g_terminal={g_term_auto:.2%}")

            # 3) Permitir override rápido
            try:
                g_years = float(input(f"Crecimiento años explícitos [{g_years_auto:.3f}]: ") or f"{g_years_auto:.6f}")
                years   = int(input("Años explícitos (por defecto 5): ") or "5")
                g_term  = float(input(f"Crecimiento terminal [{g_term_auto:.3f}]: ") or f"{g_term_auto:.6f}")
            except Exception:
                print("❌ Parámetros inválidos")
                continue

            # 4) Ejecutar DCF
            res = compute_dcf_for_ticker(ticker, wacc=wacc, g_years=g_years, years=years, g_terminal=g_term)
            if "error" in res:
                print("❌", res["error"])
            else:
                out = res["outputs"]
                print("\n🧮 RESULTADO DCF")
                print(f"   Valor razonable/acción: {fmt_num(out['fair_value_per_share'])}")
                if pd.notna(out.get("upside", np.nan)):
                    print(f"   Upside: {fmt_pct(out['upside'])}")
                if "enterprise_value" in out:
                    print(f"   EV (PV): {fmt_int(out['enterprise_value'])} | Equity: {fmt_int(out['equity_value'])}")
                else:
                    print(f"   Equity Value: {fmt_int(out['equity_value'])}")
                print(f"   Asumptions: WACC={wacc:.2%}, g_exp={g_years:.2%}, years={years}, gT={g_term:.2%}")

def demo_automatico():
    """Ejecuta una demostración automática del sistema."""
    print("🚀 DEMOSTRACIÓN AUTOMÁTICA DEL SISTEMA")
    print("="*60)
    
    # Demo 1: Análisis individual
    print("\n1️⃣ ANÁLISIS INDIVIDUAL - AAPL")
    print("-" * 40)
    analizar_stock("AAPL")
    
    # Demo 2: Comparación
    print("\n2️⃣ COMPARACIÓN DE TECNOLÓGICAS")
    print("-" * 40)
    tech_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    comparar_stocks(tech_tickers)
    
    # Demo 3: Test del sistema
    print("\n3️⃣ VERIFICACIÓN DEL SISTEMA")
    print("-" * 40)
    test_scoring()

# ================================ VALIDACIÓN DE MÉTRICAS ===============================

def validar_metricas_valoracion(datos_metricas: List[Dict]) -> Dict:
    """
    Valida y formatea métricas de valoración para megacaps tech.
    
    Args:
        datos_metricas: Lista de diccionarios con métricas de cada ticker
        
    Returns:
        Dict con tabla formateada, CSV, alertas y metadatos
    """
    tabla_formateada = []
    alertas = []
    
    for ticker_data in datos_metricas:
        ticker = ticker_data.get('ticker', '')
        pe_ttm = ticker_data.get('pe_ttm', np.nan)
        pe_fwd = ticker_data.get('pe_fwd', np.nan)
        ev_ebitda = ticker_data.get('ev_ebitda', np.nan)
        fcf_yield = ticker_data.get('fcf_yield', np.nan)
        
        # Convertir FCF Yield a porcentaje (0.02 -> 2.0)
        if pd.notna(fcf_yield):
            fcf_yield_pct = fcf_yield * 100
        else:
            fcf_yield_pct = np.nan
        
        fila = {
            'ticker': ticker,
            'pe_ttm': round(pe_ttm, 2) if pd.notna(pe_ttm) else np.nan,
            'pe_fwd': round(pe_fwd, 2) if pd.notna(pe_fwd) else np.nan,
            'ev_ebitda': round(ev_ebitda, 2) if pd.notna(ev_ebitda) else np.nan,
            'fcf_yield_pct': round(fcf_yield_pct, 1) if pd.notna(fcf_yield_pct) else np.nan
        }
        
        tabla_formateada.append(fila)
        
        # Sanity checks
        if pd.notna(pe_fwd) and pd.notna(pe_ttm) and pe_fwd > pe_ttm:
            alertas.append(f"• {ticker}: P/E FWD ({pe_fwd}) > P/E TTM ({pe_ttm}) - Posible caída de BPA esperado o datos inconsistentes")
        
        if pd.notna(fcf_yield_pct) and fcf_yield_pct == 0.0:
            alertas.append(f"• {ticker}: FCF Yield ≈ 0.0% - FCF reciente escaso/volátil")
        
        if pd.notna(pe_ttm) and pe_ttm > 80:
            alertas.append(f"• {ticker}: P/E TTM = {pe_ttm} (múltiplo elevado; confirmar supuestos)")
        
        if pd.notna(ev_ebitda) and ev_ebitda > 60:
            alertas.append(f"• {ticker}: EV/EBITDA = {ev_ebitda} (múltiplo elevado; confirmar supuestos)")
    
    # Verificar FCF Yield idéntico
    fcf_values = [fila['fcf_yield_pct'] for fila in tabla_formateada if pd.notna(fila['fcf_yield_pct'])]
    if len(set(fcf_values)) < len(fcf_values) and len(fcf_values) > 1:
        alertas.append("• FCF Yield idéntico en varias compañías - Posible redondeo o placeholder; revisar")
    
    # Generar CSV
    csv_lines = ["Ticker,P/E TTM,P/E FWD,EV/EBITDA,FCF Yield (%)"]
    for fila in tabla_formateada:
        csv_line = f"{fila['ticker']},{fila['pe_ttm']},{fila['pe_fwd']},{fila['ev_ebitda']},{fila['fcf_yield_pct']}"
        csv_lines.append(csv_line)
    csv_content = "\n".join(csv_lines)
    
    return {
        'tabla_formateada': tabla_formateada,
        'csv_content': csv_content,
        'alertas': alertas,
        'fecha_corte': 'desconocida'
    }

def generar_reporte_metricas(datos_metricas: List[Dict]) -> str:
    """
    Genera reporte completo de validación de métricas de valoración.
    
    Args:
        datos_metricas: Lista de diccionarios con métricas de cada ticker
        
    Returns:
        String con reporte formateado en Markdown
    """
    resultado = validar_metricas_valoracion(datos_metricas)
    
    # Header
    reporte = "## Tabla de Métricas de Valoración - Megacaps Tech\n\n"
    
    # Tabla Markdown
    reporte += "| Ticker | P/E TTM | P/E FWD | EV/EBITDA | FCF Yield (%) |\n"
    reporte += "|--------|---------|---------|-----------|---------------|\n"
    
    for fila in resultado['tabla_formateada']:
        pe_ttm = f"{fila['pe_ttm']:.2f}" if pd.notna(fila['pe_ttm']) else "N/D"
        pe_fwd = f"{fila['pe_fwd']:.2f}" if pd.notna(fila['pe_fwd']) else "N/D"
        ev_ebitda = f"{fila['ev_ebitda']:.2f}" if pd.notna(fila['ev_ebitda']) else "N/D"
        fcf_yield = f"{fila['fcf_yield_pct']:.1f}" if pd.notna(fila['fcf_yield_pct']) else "N/D"
        
        reporte += f"| {fila['ticker']:<6} | {pe_ttm:>7} | {pe_fwd:>7} | {ev_ebitda:>9} | {fcf_yield:>13} |\n"
    
    reporte += f"\n**Fecha de corte:** ({resultado['fecha_corte']})\n\n"
    
    # CSV
    reporte += "---\n\n## Versión CSV\n\n```csv\n"
    reporte += resultado['csv_content']
    reporte += "\n```\n\n"
    
    # Alertas
    if resultado['alertas']:
        reporte += "---\n\n## Alertas\n\n"
        for alerta in resultado['alertas']:
            reporte += f"{alerta}\n"
        reporte += "\n"
    
    return reporte

def ejemplo_validacion_metricas():
    """Ejemplo de uso de la validación de métricas."""
    print("🔍 Ejemplo de Validación de Métricas de Valoración")
    print("="*60)
    
    # Datos de ejemplo (megacaps tech)
    datos_ejemplo = [
        {'ticker': 'META', 'pe_ttm': 27.33, 'pe_fwd': 29.74, 'ev_ebitda': 20.07, 'fcf_yield': 0.02},
        {'ticker': 'GOOGL', 'pe_ttm': 24.92, 'pe_fwd': 26.12, 'ev_ebitda': 19.72, 'fcf_yield': 0.02},
        {'ticker': 'MSFT', 'pe_ttm': 36.50, 'pe_fwd': 33.32, 'ev_ebitda': 23.77, 'fcf_yield': 0.02},
        {'ticker': 'AAPL', 'pe_ttm': 36.10, 'pe_fwd': 28.63, 'ev_ebitda': 25.24, 'fcf_yield': 0.03},
        {'ticker': 'TSLA', 'pe_ttm': 208.67, 'pe_fwd': 106.91, 'ev_ebitda': 96.46, 'fcf_yield': 0.00}
    ]
    
    # Generar reporte
    reporte = generar_reporte_metricas(datos_ejemplo)
    print(reporte)

# ================================ MAIN ÚNICO ===============================

if __name__ == "__main__":
    print("📦 MÓDULO DE ANÁLISIS DE VALORACIÓN DE STOCKS v4")
    print("="*60)
    print("Sistema de valoración relativa por sector/industria")
    print("Con métricas de calidad, crecimiento y momentum")
    print("="*60)
    
    # Verificar si se pasó argumento para demo automática
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_automatico()
    else:
        # Modo interactivo
        try:
            menu_principal()
        except KeyboardInterrupt:
            print("\n\n👋 Programa interrumpido por el usuario. ¡Hasta luego!")
        except Exception as e:
            print(f"\n❌ Error inesperado: {e}")
            print("Por favor, reporte este error al desarrollador.")
