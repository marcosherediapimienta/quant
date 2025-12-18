"""
Módulo de Análisis de Valoración de Stocks 
==========================================================
Novedades v4.1 (DCF más realista):
- "Cash-like": caja = Cash & Equivalents + Short-Term/Marketable Investments (si existe).
- FCFF normalizado: mediana de 3 años (CFO + CapEx + |Interest|*(1-t)), opcional restar SBC.
- Opción DCF 3-etapas (g1 -> g2 -> gT) además del 2-etapas clásico.
- Desglose de TV share (porcentaje de valor presente que proviene del valor terminal).
- Bloque DEBUG en el flujo DCF del menú para inspeccionar insumos (precio, FCFF, acciones, caja, deuda).

También incluye:
- Fallback de ROIC (NOPAT/Capital invertido medio).
- Corregido el tratamiento de CapEx (no invertir signo; en Yahoo suele ser negativo).
- EV corporativo robusto y net debt limpio.
- Scoring con valoración, calidad, crecimiento, momentum.
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

# ===== Estilo para gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ================================ CONFIGURACIÓN ================================
WEIGHTS = {
    "valuation": 0.40,
    "quality":   0.30,
    "growth":    0.20,
    "momentum":  0.10
}

WINSOR = {
    "pb":        [0.01, 0.99],
    "roe":       [0.01, 0.99],
    "ev_ebitda": [0.01, 0.99]
}

MOMENTUM_WINDOW = {"m6": 126, "m12": 252, "skip": 21}
RELATIVE_SCOPE = "industry"
NEUTRAL_BAND = 0.30
RSI_WINDOW = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
RSI_MAX_ADJUSTMENT = 0.10
MIN_PEERS = 5
FALLBACK_SCOPE = {"industry":"sector","sector":"universe"}

INCOME_EBIT_KEYS = ["Ebit", "EBIT", "Operating Income"]
INCOME_TAX_EXPENSE_KEYS = ["Tax Provision", "Income Tax Expense"]
INCOME_PRETAX_KEYS = ["Income Before Tax", "Pretax Income", "Ebt", "Earnings Before Tax"]

# Cash Flow keys
CF_CFO_KEYS = [
    "Total Cash From Operating Activities","Operating Cash Flow",
    "Net Cash Provided by Operating Activities","Net CashFrom Operating Activities"
]
CF_CAPEX_KEYS = [
    "Capital Expenditures","Capital Expenditure","Purchase Of Property Plant And Equipment"
]

# Balance Sheet keys
BS_EQUITY_KEYS = [
    "Total Stockholder Equity","Total Stockholders Equity",
    "Stockholders Equity","Total Equity Gross Minority Interest"
]
BS_CASH_KEYS = [
    "Cash And Cash Equivalents","Cash","Cash And Cash Equivalents And Short Term Investments"
]
BS_ST_INV_KEYS = [
    "Short Term Investments","Marketable Securities","Other Short Term Investments"
]
BS_DEBT_KEYS = [
    "Total Debt","Short Long Term Debt","Long Term Debt",
    "Current Debt And Capital Lease Obligation","Long Term Debt And Capital Lease Obligation"
]

# ================================ UTILIDADES ================================

def _order_cols_by_date(df: pd.DataFrame) -> pd.DataFrame:
    try:
        cols = pd.to_datetime(df.columns, errors="coerce")
        if getattr(cols, "notna", lambda: False)().any():
            order = np.argsort(cols)[::-1]  # más reciente primero
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

def _row_series(df: pd.DataFrame, keys: List[str]) -> pd.Series:
    """Serie completa (DESC por fecha) de la primera key disponible."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    df = _order_cols_by_date(df)
    for k in keys:
        if k in df.index:
            return df.loc[k].dropna()
    return pd.Series(dtype=float)

def _nan_if_missing(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return np.nan
    return x

def _safe_div(numer, denom):
    try:
        if denom is None or numer is None:
            return np.nan
        denom = float(denom); numer = float(numer)
        return np.nan if denom == 0 else numer / denom
    except Exception:
        return np.nan

def winsorize_against_peers(value: float, peers: List[float], p_low=0.01, p_high=0.99) -> float:
    vals = np.array([v for v in peers if pd.notna(v)], dtype=float)
    if pd.isna(value) or vals.size == 0:
        return value
        
    lo, hi = np.quantile(vals, [p_low, p_high])
    return float(np.clip(value, lo, hi))

def winsorize(x: float, p_low: float, p_high: float, data: List[float] = None) -> float:
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
    if pd.isna(value) or not peer_values:
        return np.nan
    valid_values = [v for v in peer_values if pd.notna(v)]
    if not valid_values:
        return np.nan
    n = len(valid_values)
    rank = sum(1 for v in valid_values if v < value) + 0.5 * sum(1 for v in valid_values if v == value)
    return rank / n

def normalize_signed(value: float, higher_is_better: bool = True) -> float:
    if pd.isna(value):
        return np.nan
    signed_value = value if higher_is_better else -value
    return np.tanh(signed_value)

def _calculate_rsi(prices: np.ndarray, window: int = 14) -> float:
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
    return 100 - (100 / (1 + rs))

def _clean_multiple(x: float, allow_zero: bool = False) -> float:
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

# ---------- Formateo ----------
def formatear_numero(x, formato="{:,.2f}", na="N/D", fmt=None):
    try:
        # Si se pasa fmt, usar ese formato en lugar de formato
        if fmt is not None:
            formato = fmt
        return na if pd.isna(x) or not np.isfinite(float(x)) else formato.format(float(x))
    except Exception:
        return na
def formatear_porcentaje(x, formato="{:.2%}", na="N/D", fmt=None):
    try:
        # Si se pasa fmt, usar ese formato en lugar de formato
        if fmt is not None:
            formato = fmt
        return na if pd.isna(x) or not np.isfinite(float(x)) else formato.format(float(x))
    except Exception:
        return na
fmt_num = formatear_numero
def fmt_int(x, na="N/D"): return formatear_numero(x, "{:,.0f}", na)
def fmt_pct(x, fmt="{:.2%}", na="N/D"): return formatear_porcentaje(x, fmt, na)

# ================================ CAPTURA DE DATOS ================================

def _get_current_price(stock) -> float:
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
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='5d')
        if not hist.empty:
            return hist.index[-1].strftime('%Y-%m-%d')
    except Exception:
        pass
    return datetime.now().strftime('%Y-%m-%d')

def _get_fundamental_date(info: Dict) -> str:
    año_anterior = datetime.now().year - 1
    return datetime(año_anterior, 12, 31).strftime('%Y-%m-%d')

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

def _get_cash_like(bs: pd.DataFrame) -> Optional[float]:
    """Cash-like = Cash & Equivalents (+) Short-Term/Marketable Investments si aplica."""
    if bs is None or bs.empty:
        return None
    bs = _order_cols_by_date(bs)

    combined = _get_row_last(bs, ["Cash And Cash Equivalents And Short Term Investments"])
    if combined is not None:
        return float(combined)

    cash = _get_row_last(bs, BS_CASH_KEYS)
    stinv = None
    for k in BS_ST_INV_KEYS:
        if k in bs.index:
            v = _get_row_last(bs, [k])
            if v is not None:
                stinv = v
                break
    total = 0.0
    if cash is not None: total += float(cash)
    if stinv is not None: total += float(stinv)
    return total if total > 0 else None

def _debt_at(df: pd.DataFrame, col_idx: int) -> Optional[float]:
    if df is None or df.empty:
        return None
    if "Total Debt" in df.index:
        v = df.loc["Total Debt"].iloc[col_idx]
        return float(v) if pd.notna(v) else None
    comp_keys = [
        "Short Long Term Debt",
        "Current Debt And Capital Lease Obligation",
        "Long Term Debt",
        "Long Term Debt And Capital Lease Obligation",
    ]
    vals = []
    for k in comp_keys:
        if k in df.index and pd.notna(df.loc[k].iloc[col_idx]):
            vals.append(float(df.loc[k].iloc[col_idx]))
    return float(sum(vals)) if vals else None

def _get_cash_debt_last(stock: yf.Ticker) -> Tuple[Optional[float], Optional[float]]:
    cash = debt = None
    for getter in ("quarterly_balance_sheet", "balance_sheet"):
        try:
            bs = getattr(stock, getter)
            if bs is None or bs.empty:
                continue

            if cash is None:
                cash = _get_cash_like(bs)

            if debt is None:
                if "Total Debt" in bs.index:
                    debt = _get_row_last(bs, ["Total Debt"])
                else:
                    debt = _debt_at(bs, 0)

            if cash is not None and debt is not None:
                break
        except Exception:
            continue
    return cash, debt

def _equity_nonpositive(stock: yf.Ticker) -> bool:
    eq = _get_equity_last(stock)
    try:
        return (eq is not None) and (float(eq) <= 0)
    except Exception:
        return False

def _latest_fund_date(stock: yf.Ticker) -> Optional[str]:
    dfs = []
    for attr in ("quarterly_financials","quarterly_cashflow","quarterly_balance_sheet",
                 "financials","cashflow","balance_sheet"):
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

def _compute_fcfe_ttm(stock: yf.Ticker) -> Optional[float]:
    try:
        qcf = stock.quarterly_cashflow
    except Exception:
        qcf = None
    cfo_4q   = _get_row_sum_last_n(qcf, CF_CFO_KEYS, 4)
    capex_4q = _get_row_sum_last_n(qcf, CF_CAPEX_KEYS, 4)
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
        return float(cfo + capex)
    if cfo_4q is None or capex_4q is None:
        return None
    return float(cfo_4q + capex_4q)

def _compute_fcff_ttm(stock: yf.Ticker, tax_rate: float = 0.21) -> Optional[float]:
    try:
        qcf = stock.quarterly_cashflow
    except Exception:
        qcf = None
    cfo_4q   = _get_row_sum_last_n(qcf, CF_CFO_KEYS, 4)
    capex_4q = _get_row_sum_last_n(qcf, CF_CAPEX_KEYS, 4)
    interest_expense_keys = ["Interest Expense","Interest And Debt Expense","Interest Paid"]
    interest_4q = _get_row_sum_last_n(qcf, interest_expense_keys, 4)
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
        fcff = (cfo + capex)
        if interest is not None:
            fcff += abs(interest) * (1 - tax_rate)
        return float(fcff)
    if cfo_4q is None or capex_4q is None:
        return None
    fcff = (cfo_4q + capex_4q)
    if interest_4q is not None:
        fcff += abs(interest_4q) * (1 - tax_rate)
    return float(fcff)

def _fcff_normalized(stock: yf.Ticker, tax_rate: float = 0.21, years: int = 3, adjust_sbc: bool = False) -> Optional[float]:
    try:
        acf = stock.cashflow
    except Exception:
        acf = None
    if acf is None or acf.empty:
        return None
    acf = _order_cols_by_date(acf)
    cfo   = _row_series(acf, CF_CFO_KEYS).iloc[:years]
    capex = _row_series(acf, CF_CAPEX_KEYS).iloc[:years]
    intr  = _row_series(acf, ["Interest Expense","Interest And Debt Expense"]).abs().iloc[:years]
    if cfo.empty or capex.empty:
        return None
    fcff_y = cfo + capex
    if not intr.empty:
        fcff_y = fcff_y + intr * (1 - tax_rate)
    if adjust_sbc:
        sbc = _row_series(acf, ["Stock Based Compensation"]).iloc[:years].fillna(0.0)
        if not sbc.empty:
            fcff_y = fcff_y - sbc
    fcff_norm = fcff_y.dropna()
    if fcff_norm.empty:
        return None
    val = float(np.median(fcff_norm))
    return val if np.isfinite(val) else None

def fetch_snapshot(ticker: str) -> Dict:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        precio_actual = _get_current_price(stock)

        market_cap = _nan_if_missing(info.get('marketCap'))
        shares_outstanding = _nan_if_missing(info.get('sharesOutstanding'))

        pe_ttm = _nan_if_missing(info.get('trailingPE'))
        pe_fwd = _nan_if_missing(info.get('forwardPE'))
        pb = _nan_if_missing(info.get('priceToBook'))
        ps = _nan_if_missing(info.get('priceToSalesTrailing12Months'))
        ev_ebitda = _nan_if_missing(info.get('enterpriseToEbitda'))
        ev_revenue = _nan_if_missing(info.get('enterpriseToRevenue'))

        roe = _nan_if_missing(info.get('returnOnEquity'))
        roa = _nan_if_missing(info.get('returnOnAssets'))
        roic = _nan_if_missing(info.get('returnOnInvestedCapital'))
        gross_margin = _nan_if_missing(info.get('grossMargins'))
        operating_margin = _nan_if_missing(info.get('operatingMargins'))
        profit_margin = _nan_if_missing(info.get('profitMargins'))

        if pd.isna(roic):
            try:
                roic_fb = _estimate_roic_fallback(stock, info)
                if roic_fb is not None and np.isfinite(roic_fb):
                    roic = float(roic_fb)
            except Exception:
                pass

        revenue_growth = _nan_if_missing(info.get('revenueGrowth'))
        earnings_growth = _nan_if_missing(info.get('earningsGrowth'))

        debt_equity = _nan_if_missing(info.get('debtToEquity'))
        debt_ebitda = _nan_if_missing(info.get('debtToEbitda'))

        fcfe_ttm = _compute_fcfe_ttm(stock)
        fcff_ttm = _compute_fcff_ttm(stock, tax_rate=0.21)

        # EV corporativo limpio
        ev_corp = _compute_ev_corporativo(info, stock)

        fcf_yield = _safe_div(fcfe_ttm, market_cap) if pd.notna(fcfe_ttm) and pd.notna(market_cap) and market_cap > 0 else np.nan
        fcff_ev_yield = _safe_div(fcff_ttm, ev_corp) if pd.notna(fcff_ttm) and pd.notna(ev_corp) and ev_corp > 0 else np.nan

        # Equity nonpositive
        equity_nonpositive = _equity_nonpositive(stock)
        if equity_nonpositive:
            pb = np.nan
            roe = np.nan

        total_cash = _nan_if_missing(info.get('totalCash'))
        total_debt = _nan_if_missing(info.get('totalDebt'))
        net_cash = (total_cash if pd.notna(total_cash) else 0.0) - (total_debt if pd.notna(total_debt) else 0.0)
        ebitda = _nan_if_missing(info.get('ebitda'))
        net_cash_ebitda = _safe_div(net_cash, ebitda) if pd.notna(ebitda) and ebitda != 0 else np.nan

        fund_date = _latest_fund_date(stock) or _get_fundamental_date(info)

        return {
            'ticker': ticker.upper(),
            'price': precio_actual,
            'market_cap': market_cap,
            'shares_outstanding': shares_outstanding,
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),

            'pe_ttm': pe_ttm, 'pe_fwd': pe_fwd, 'pb': pb, 'ps': ps,
            'ev_ebitda': ev_ebitda, 'ev_revenue': ev_revenue,
            'fcf_yield': fcf_yield, 'fcf_ttm': fcfe_ttm, 'fcfe_ttm': fcfe_ttm,
            'fcfe_yield': fcf_yield, 'fcff_ttm': fcff_ttm, 'fcff_ev_yield': fcff_ev_yield,

            'roe': roe, 'roa': roa, 'roic': roic,
            'gross_margin': gross_margin, 'operating_margin': operating_margin, 'profit_margin': profit_margin,

            'revenue_growth': revenue_growth, 'earnings_growth': earnings_growth,

            'debt_equity': debt_equity, 'debt_ebitda': debt_ebitda, 'net_cash_ebitda': net_cash_ebitda,

            'as_of': {'price_date': _get_latest_trading_date(ticker), 'fund_date': fund_date},
            'source': 'yfinance',
            'ev_corp': ev_corp,
            'equity_nonpositive': equity_nonpositive
        }

    except Exception as e:
        print(f"❌ Error obteniendo snapshot para {ticker}: {e}")
        return {'ticker': ticker.upper(),'price': np.nan,'market_cap': np.nan,'sector': 'N/A','industry': 'N/A',
                'as_of': {'price_date': 'N/D', 'fund_date': 'N/D'},'source': 'error','ev_corp': np.nan,
                'equity_nonpositive': False}

def _compute_ev_corporativo(info: Dict, stock: yf.Ticker) -> Optional[float]:
    try:
        mc = float(info.get("marketCap") or 0)
        cash_i = info.get("totalCash"); debt_i = info.get("totalDebt")
        cash_bs, debt_bs = _get_cash_debt_last(stock)
        cash = float(cash_bs if cash_bs is not None else (cash_i or 0))
        debt = float(debt_bs if debt_bs is not None else (debt_i or 0))
        if mc <= 0:
            return None
        return mc + debt - cash
    except Exception:
        return None

def fetch_history(ticker: str, days: int = 252) -> pd.Series:
    try:
        stock = yf.Ticker(ticker)
        period = f"{max(1, math.ceil(days/252))}y"
        hist = stock.history(period=period, interval='1d')
        if hist.empty:
            return pd.Series(dtype=float)
        return hist['Close'].dropna().tail(days)
    except Exception as e:
        print(f"❌ Error obteniendo histórico para {ticker}: {e}")
        return pd.Series(dtype=float)

def fetch_peers(scope: str, name: str, ticker: str) -> List[str]:
    if scope == "industry":
        industry_peers = {
            "Consumer Electronics": ["AAPL","MSFT","GOOGL","AMZN","META","NFLX"],
            "Software": ["MSFT","GOOGL","ORCL","CRM","ADBE","NOW"],
            "E-commerce": ["AMZN","EBAY","SHOP","ETSY","MELI"],
            "Automotive": ["TSLA","F","GM","TM","HMC"],
            "Financial Services": ["JPM","BAC","WFC","GS","MS"],
            "Credit Services": ["V","MA","AXP","COF","SYF","FI","FIS","GPN","ADYEN.AS","PYPL","AFRM","SOFI","LC","UPST"],
            "Payments": ["V","MA","AXP","COF","SYF","FI","FIS","GPN","ADYEN.AS","PYPL","AFRM","SOFI","LC","UPST"]
        }
        return industry_peers.get(name, [ticker])
    elif scope == "sector":
        sector_peers = {
            "Technology": ["AAPL","MSFT","GOOGL","AMZN","META","NFLX","ORCL","CRM"],
            "Consumer Discretionary": ["AMZN","TSLA","HD","MCD","NKE"],
            "Financial Services": ["JPM","BAC","WFC","GS","MS","V","MA"],
            "Healthcare": ["JNJ","PFE","UNH","ABBV","MRK"],
            "Energy": ["XOM","CVX","COP","EOG","SLB"]
        }
        return sector_peers.get(name, [ticker])
    elif scope == "universe":
        return ["AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","ORCL","ADBE","CRM"]
    return [ticker]

# ================================ SUBSCORES ================================

def valuation_subscore(ticker_data: Dict, peers_data: List[Dict]) -> Tuple[float, Dict]:
    pe_ttm = _clean_multiple(ticker_data.get('pe_ttm'))
    pe_fwd = _clean_multiple(ticker_data.get('pe_fwd'))
    ev_ebitda_raw = ticker_data.get('ev_ebitda')
    ev_ebitda = _clean_multiple(ev_ebitda_raw)
    ps = _clean_multiple(ticker_data.get('ps'))
    pb = _clean_multiple(ticker_data.get('pb'))
    fcf_yield = ticker_data.get('fcf_yield')

    pe_peers = [v for v in (_clean_multiple(p.get('pe_ttm')) for p in peers_data) if pd.notna(v)]
    pe_fwd_peers = [v for v in (_clean_multiple(p.get('pe_fwd')) for p in peers_data) if pd.notna(v)]
    ev_peers = [v for v in (_clean_multiple(p.get('ev_ebitda')) for p in peers_data) if pd.notna(v)]
    ps_peers = [v for v in (_clean_multiple(p.get('ps')) for p in peers_data) if pd.notna(v)]
    pb_peers = [v for v in (_clean_multiple(p.get('pb')) for p in peers_data) if pd.notna(v)]
    fcf_peers = [p.get('fcf_yield') for p in peers_data if pd.notna(p.get('fcf_yield', np.nan))]

    scores, weights, details = [], [], {}

    if pd.notna(pe_ttm) and pe_peers:
        pe_pct = percentile_rank(pe_ttm, pe_peers)
        if pd.notna(pe_pct):
            scores.append(-(2*pe_pct-1)); weights.append(0.25); details['pe_pct']=pe_pct

    if pd.notna(pe_fwd) and pe_fwd_peers:
        pe_fwd_pct = percentile_rank(pe_fwd, pe_fwd_peers)
        if pd.notna(pe_fwd_pct):
            scores.append(-(2*pe_fwd_pct-1)); weights.append(0.20); details['pe_fwd_pct']=pe_fwd_pct

    if pd.notna(ev_ebitda) and ev_peers:
        ev_w = winsorize_against_peers(ev_ebitda, ev_peers, *WINSOR['ev_ebitda'])
        ev_peers_w = [winsorize_against_peers(v, ev_peers, *WINSOR['ev_ebitda']) for v in ev_peers]
        ev_pct = percentile_rank(ev_w, ev_peers_w)
        if pd.notna(ev_pct):
            scores.append(-(2*ev_pct-1)); weights.append(0.25); details['ev_ebitda_pct']=ev_pct

    if pd.notna(ps) and ps_peers:
        ps_pct = percentile_rank(ps, ps_peers)
        if pd.notna(ps_pct):
            scores.append(-(2*ps_pct-1)); weights.append(0.20); details['ps_pct']=ps_pct

    if pd.notna(fcf_yield) and fcf_peers:
        fcf_pct = percentile_rank(fcf_yield, fcf_peers)
        if pd.notna(fcf_pct):
            scores.append( (2*fcf_pct-1) ); weights.append(0.35); details['fcf_pct']=fcf_pct

    if pd.notna(pb) and pb_peers:
        pb_w = winsorize_against_peers(pb, pb_peers, *WINSOR['pb'])
        pb_peers_w = [winsorize_against_peers(p, pb_peers, *WINSOR['pb']) for p in pb_peers]
        pb_pct = percentile_rank(pb_w, pb_peers_w)
        if pd.notna(pb_pct):
            scores.append(-(2*pb_pct-1)); weights.append(0.10); details['pb_pct']=pb_pct

    if scores and weights:
        tw = sum(weights); weights = [w/tw for w in weights] if tw>0 else weights
        subscore = sum(s*w for s,w in zip(scores,weights))
    else:
        subscore = 0.0
    return subscore, details

def quality_subscore(ticker_data: Dict, peers_data: List[Dict]) -> Tuple[float, Dict]:
    roic = ticker_data.get('roic')
    operating_margin = ticker_data.get('operating_margin')
    profit_margin = ticker_data.get('profit_margin')
    net_cash_ebitda = ticker_data.get('net_cash_ebitda')
    roe = ticker_data.get('roe')

    roic_peers = [p.get('roic') for p in peers_data if pd.notna(p.get('roic', np.nan))]
    op_margin_peers = [p.get('operating_margin') for p in peers_data if pd.notna(p.get('operating_margin', np.nan))]
    profit_margin_peers = [p.get('profit_margin') for p in peers_data if pd.notna(p.get('profit_margin', np.nan))]
    net_cash_peers = [p.get('net_cash_ebitda') for p in peers_data if pd.notna(p.get('net_cash_ebitda', np.nan))]
    roe_peers = [p.get('roe') for p in peers_data if pd.notna(p.get('roe', np.nan))]

    scores, weights, details = [], [], {}

    if pd.notna(roic) and roic_peers:
        pct = percentile_rank(roic, roic_peers)
        if pd.notna(pct):
            scores.append(2*pct-1); weights.append(0.40); details['roic_pct']=pct

    if pd.notna(operating_margin) and op_margin_peers:
        pct = percentile_rank(operating_margin, op_margin_peers)
        if pd.notna(pct):
            scores.append(2*pct-1); weights.append(0.25); details['op_margin_pct']=pct

    if pd.notna(profit_margin) and profit_margin_peers:
        pct = percentile_rank(profit_margin, profit_margin_peers)
        if pd.notna(pct):
            scores.append(2*pct-1); weights.append(0.20); details['profit_margin_pct']=pct

    if pd.notna(net_cash_ebitda) and net_cash_peers:
        pct = percentile_rank(net_cash_ebitda, net_cash_peers)
        if pd.notna(pct):
            scores.append(2*pct-1); weights.append(0.15); details['net_cash_pct']=pct

    if pd.notna(roe) and roe_peers:
        roe_w = winsorize_against_peers(roe, roe_peers, *WINSOR['roe'])
        roe_peers_w = [winsorize_against_peers(p, roe_peers, *WINSOR['roe']) for p in roe_peers]
        pct = percentile_rank(roe_w, roe_peers_w)
        if pd.notna(pct):
            scores.append(2*pct-1); weights.append(0.10); details['roe_pct']=pct

    if scores and weights:
        tw = sum(weights); weights = [w/tw for w in weights] if tw>0 else weights
        subscore = sum(s*w for s,w in zip(scores,weights))
    else:
        subscore = 0.0
    return subscore, details

def growth_subscore(ticker_data: Dict, peers_data: List[Dict]) -> Tuple[float, Dict]:
    revenue_growth = ticker_data.get('revenue_growth')
    earnings_growth = ticker_data.get('earnings_growth')
    rev_peers = [p.get('revenue_growth') for p in peers_data if pd.notna(p.get('revenue_growth', np.nan))]
    eps_peers = [p.get('earnings_growth') for p in peers_data if pd.notna(p.get('earnings_growth', np.nan))]
    scores, weights, details = [], [], {}
    if pd.notna(revenue_growth) and rev_peers:
        pct = percentile_rank(revenue_growth, rev_peers)
        if pd.notna(pct):
            scores.append(2*pct-1); weights.append(0.60); details['revenue_growth_pct']=pct
    if pd.notna(earnings_growth) and eps_peers:
        pct = percentile_rank(earnings_growth, eps_peers)
        if pd.notna(pct):
            scores.append(2*pct-1); weights.append(0.40); details['earnings_growth_pct']=pct
    if scores and weights:
        tw = sum(weights); weights = [w/tw for w in weights] if tw>0 else weights
        subscore = sum(s*w for s,w in zip(scores,weights))
    else:
        subscore = 0.0
    return subscore, details

def momentum_subscore(ticker_data: Dict, price_history: np.ndarray) -> Tuple[float, Dict]:
    w12 = MOMENTUM_WINDOW['m12']; w6 = MOMENTUM_WINDOW['m6']; wskip = MOMENTUM_WINDOW['skip']
    if len(price_history) < 200:
        return 0.0, {}
    returns = np.diff(price_history) / price_history[:-1]

    mom12_1 = 0.0
    if len(returns) >= w12:
        mom12 = np.prod(1 + returns[-w12:]) - 1
        mom1 = np.prod(1 + returns[-wskip:]) - 1
        mom12_1 = mom12 - mom1
    mom6 = np.prod(1 + returns[-w6:]) - 1 if len(returns) >= w6 else 0.0

    rsi = _calculate_rsi(price_history, window=RSI_WINDOW)
    vol_252 = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0

    scores, weights, details = [], [], {}
    if mom12_1 != 0.0:
        scores.append(normalize_signed(mom12_1, True)); weights.append(0.70); details['mom12_1']=mom12_1
    if mom6 != 0.0:
        scores.append(normalize_signed(mom6, True)); weights.append(0.30); details['mom6']=mom6

    rsi_adjustment = 0.0
    if pd.notna(rsi):
        if rsi > RSI_OVERBOUGHT:
            rsi_adjustment = -RSI_MAX_ADJUSTMENT * (rsi - RSI_OVERBOUGHT) / (100 - RSI_OVERBOUGHT)
        elif rsi < RSI_OVERSOLD:
            rsi_adjustment =  RSI_MAX_ADJUSTMENT * (RSI_OVERSOLD - rsi) / RSI_OVERSOLD
        details['rsi']=rsi; details['rsi_adjustment']=rsi_adjustment

    if scores and weights:
        tw = sum(weights); weights = [w/tw for w in weights] if tw>0 else weights
        base = sum(s*w for s,w in zip(scores,weights))
    else:
        base = 0.0
    final_score = np.clip(base + rsi_adjustment, -1.0, 1.0)
    details['vol_252'] = vol_252
    return final_score, details

# ================================ DCF (VALORACIÓN ABSOLUTA) ================================

def _get_net_debt_for_ticker(ticker: str) -> Optional[float]:
    try:
        stock = yf.Ticker(ticker)
        cash, debt = _get_cash_debt_last(stock)
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
    fcf0: float, wacc: float, g_years: float = 0.08, years: int = 5,
    g_terminal: float = 0.025, mid_year: bool = True
) -> float:
    if fcf0 is None or not np.isfinite(fcf0): return np.nan
    if wacc <= g_terminal: return np.nan
    fcfs=[]; f=float(fcf0)
    for t in range(1, years+1):
        f *= (1+g_years); fcfs.append(f)
    if mid_year:
        pv_fcfs = sum(cf/((1+wacc)**(t-0.5)) for t,cf in enumerate(fcfs,1))
        fterm = fcfs[-1]*(1+g_terminal)
        tv = fterm/(wacc-g_terminal)
        pv_tv = tv/((1+wacc)**(years-0.5))
    else:
        pv_fcfs = sum(cf/((1+wacc)**t) for t,cf in enumerate(fcfs,1))
        fterm = fcfs[-1]*(1+g_terminal)
        tv = fterm/(wacc-g_terminal)
        pv_tv = tv/((1+wacc)**years)
    return float(pv_fcfs + pv_tv)

def dcf_2stage_fcff_with_tvshare(
    fcf0: float, wacc: float, g_years: float = 0.08, years: int = 5,
    g_terminal: float = 0.025, mid_year: bool = True
) -> Tuple[float,float]:
    if fcf0 is None or not np.isfinite(fcf0) or wacc <= g_terminal:
        return (np.nan, np.nan)
    fcfs=[]; f=float(fcf0)
    for t in range(1, years+1):
        f *= (1+g_years); fcfs.append(f)
    if mid_year:
        pv_fcfs = sum(cf/((1+wacc)**(t-0.5)) for t,cf in enumerate(fcfs,1))
        fterm = fcfs[-1]*(1+g_terminal)
        tv = fterm/(wacc-g_terminal)
        pv_tv = tv/((1+wacc)**(years-0.5))
    else:
        pv_fcfs = sum(cf/((1+wacc)**t) for t,cf in enumerate(fcfs,1))
        fterm = fcfs[-1]*(1+g_terminal)
        tv = fterm/(wacc-g_terminal)
        pv_tv = tv/((1+wacc)**years)
    ev = pv_fcfs + pv_tv
    tv_share = pv_tv/ev if ev and np.isfinite(ev) and ev>0 else np.nan
    return float(ev), float(tv_share)

def dcf_3stage_fcff(
    fcf0: float, wacc: float,
    g1: float = 0.14, y1: int = 3,
    g2: float = 0.08, y2: int = 3,
    gT: float = 0.022, mid_year: bool = True
) -> float:
    if any(x is None or not np.isfinite(x) for x in [fcf0,wacc,g1,g2,gT]) or wacc <= gT:
        return np.nan
    def disc(cf,t): return cf/((1+wacc)**(t-0.5 if mid_year else t))
    fcfs=[]; f=float(fcf0)
    for t in range(1, y1+1): f*=(1+g1); fcfs.append((t,f))
    for t in range(y1+1, y1+y2+1): f*=(1+g2); fcfs.append((t,f))
    fterm = f*(1+gT); tv = fterm/(wacc-gT)
    pv_fcfs = sum(disc(v,t) for t,v in fcfs)
    pv_tv   = disc(tv, y1+y2)
    return float(pv_fcfs + pv_tv)

def dcf_2stage_fcfe(
    fcfe0: float, cost_of_equity: float, g_years: float = 0.08, years: int = 5,
    g_terminal: float = 0.025, mid_year: bool = True
) -> float:
    if fcfe0 is None or not np.isfinite(fcfe0): return np.nan
    if cost_of_equity <= g_terminal: return np.nan
    fcfes=[]; f=float(fcfe0)
    for t in range(1, years+1):
        f *= (1+g_years); fcfes.append(f)
    if mid_year:
        pv = sum(cf/((1+cost_of_equity)**(t-0.5)) for t,cf in enumerate(fcfes,1))
        fterm = fcfes[-1]*(1+g_terminal)
        tv = fterm/(cost_of_equity-g_terminal)
        pv_tv = tv/((1+cost_of_equity)**(years-0.5))
    else:
        pv = sum(cf/((1+cost_of_equity)**t) for t,cf in enumerate(fcfes,1))
        fterm = fcfes[-1]*(1+g_terminal)
        tv = fterm/(cost_of_equity-g_terminal)
        pv_tv = tv/((1+cost_of_equity)**years)
    return float(pv + pv_tv)

def compute_dcf_for_ticker_fcff(
    ticker: str,
    wacc: float = 0.09,
    g_years: float = 0.08,
    years: int = 5,
    g_terminal: float = 0.025,
    mid_year: bool = True,
    use_net_debt: bool = True,
    use_normalized_fcff: bool = True,
    normalized_years: int = 3,
    adjust_sbc: bool = False,
    three_stage: bool = False,
    g1: float = 0.12, y1: int = 3,
    g2: float = 0.08, y2: int = 3
) -> Dict:
    snap = fetch_snapshot(ticker)
    fcff_ttm = snap.get("fcff_ttm", np.nan)
    shares   = snap.get("shares_outstanding", np.nan)
    price    = snap.get("price", np.nan)

    fcff_used = fcff_ttm
    tv_share = np.nan
    if use_normalized_fcff:
        stock = yf.Ticker(ticker)
        fcff_norm = _fcff_normalized(stock, years=normalized_years, adjust_sbc=adjust_sbc)
        if fcff_norm is not None and np.isfinite(fcff_norm) and fcff_norm > 0:
            fcff_used = fcff_norm

    if pd.isna(fcff_used) or not np.isfinite(fcff_used) or fcff_used <= 0:
        return {"ticker": ticker.upper(), "error": "FCFF (TTM/normalizado) no disponible o no positivo (DCF no fiable)"}
    if pd.isna(shares) or not np.isfinite(shares) or shares <= 0:
        return {"ticker": ticker.upper(), "error": "Shares outstanding no disponible (DCF no calculable)"}

    if three_stage:
        ev = dcf_3stage_fcff(fcf0=fcff_used, wacc=wacc, g1=g1, y1=y1, g2=g2, y2=y2, gT=g_terminal, mid_year=mid_year)
    else:
        ev, tv_share = dcf_2stage_fcff_with_tvshare(
            fcf0=fcff_used, wacc=wacc, g_years=g_years, years=years, g_terminal=g_terminal, mid_year=mid_year
        )

    if pd.isna(ev):
        return {"ticker": ticker.upper(), "error": "Parámetros inválidos: WACC debe ser > g_terminal."}

    net_debt = _get_net_debt_for_ticker(ticker) if use_net_debt else 0.0
    if pd.isna(net_debt) or not np.isfinite(net_debt): net_debt = 0.0

    equity_value = ev - net_debt
    fair_value_per_share = equity_value / shares

    upside = np.nan
    if pd.notna(price) and np.isfinite(price) and price > 0:
        upside = (fair_value_per_share / price) - 1.0

    return {
        "ticker": ticker.upper(),
        "method": "FCFF",
        "assumptions": {
            "wacc": wacc, "g_years": g_years, "years": years, "g_terminal": g_terminal,
            "mid_year": mid_year, "use_net_debt": use_net_debt,
            "use_normalized_fcff": use_normalized_fcff, "normalized_years": normalized_years,
            "adjust_sbc": adjust_sbc, "three_stage": three_stage,
            "g1": g1, "y1": y1, "g2": g2, "y2": y2
        },
        "inputs": {
            "fcff_used": fcff_used, "fcff_ttm": fcff_ttm, "shares_outstanding": shares,
            "net_debt": net_debt, "price": price, "tv_share": tv_share
        },
        "outputs": {
            "enterprise_value": ev, "equity_value": equity_value,
            "fair_value_per_share": fair_value_per_share, "upside": upside
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
        "assumptions": {"cost_of_equity": cost_of_equity, "g_years": g_years, "years": years, "g_terminal": g_terminal, "mid_year": mid_year},
        "inputs": {"fcfe_ttm": fcfe_ttm, "shares_outstanding": shares, "price": price},
        "outputs": {"equity_value": equity_value, "fair_value_per_share": fair_value_per_share, "upside": upside},
        "as_of": snap.get("as_of", {})
    }

def estimate_wacc(ticker: str, rf: float = 0.042, erp: float = 0.05, tax_rate: float = 0.21) -> Dict:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        E = float(info.get("marketCap", 0) or 0)
        D = float(info.get("totalDebt", 0) or 0)
        V = E + D
        we = E / V if V > 0 else 1.0
        wd = D / V if V > 0 else 0.0
        beta = info.get("beta", 1.0) or 1.0
        Re = rf + beta * erp
        Rd = _estimate_cost_of_debt(stock, info, D)
        Rd_after_tax = Rd * (1 - tax_rate)
        WACC = we * Re + wd * Rd_after_tax
        return {
            "ticker": ticker.upper(), "beta": beta, "cost_of_equity": Re,
            "cost_of_debt": Rd_after_tax, "cost_of_debt_before_tax": Rd,
            "weights": {"equity": we, "debt": wd}, "wacc": WACC,
            "market_cap": E, "total_debt": D
        }
    except Exception as e:
        return {"ticker": ticker.upper(), "error": str(e)}

def _estimate_cost_of_debt(stock: yf.Ticker, info: Dict, total_debt: float) -> float:
    if total_debt <= 0: return 0.0
    try:
        qcf = stock.quarterly_cashflow
        if qcf is not None and not qcf.empty:
            interest_expense_keys = ["Interest Expense","Interest Paid","Interest And Debt Expense","Net Interest Income","Interest Income"]
            interest_4q = _get_row_sum_last_n(qcf, interest_expense_keys, 4)
            if interest_4q is not None and interest_4q != 0:
                rd = abs(interest_4q) / total_debt
                return float(np.clip(rd, 0.01, 0.12))
    except Exception:
        pass
    try:
        acf = stock.cashflow
        if acf is not None and not acf.empty:
            interest_expense_keys = ["Interest Expense","Interest Paid","Interest And Debt Expense","Net Interest Income","Interest Income"]
            interest_an = _get_row_last(acf, interest_expense_keys)
            if interest_an is not None and interest_an != 0:
                rd = abs(interest_an) / total_debt
                return float(np.clip(rd, 0.01, 0.12))
    except Exception:
        pass
    ebitda = info.get("ebitda", None)
    if ebitda and ebitda > 0:
        ratio = total_debt / ebitda
        if ratio < 1: return 0.04
        elif ratio < 2: return 0.05
        elif ratio < 3: return 0.06
        elif ratio < 4: return 0.07
        else: return 0.08
    return 0.045

def compute_dcf_for_ticker(
    ticker: str,
    wacc: float = 0.09,
    g_years: float = 0.08,
    years: int = 5,
    g_terminal: float = 0.025,
    use_net_debt: bool = True,
    method: str = "fcff",
    mid_year: bool = True,
    use_normalized_fcff: bool = True,
    normalized_years: int = 3,
    adjust_sbc: bool = False,
    three_stage: bool = False,
    g1: float = 0.12, y1: int = 3,
    g2: float = 0.08, y2: int = 3
) -> Dict:
    if method.lower() == "fcfe":
        wacc_data = estimate_wacc(ticker)
        cost_of_equity = 0.12 if "error" in wacc_data else wacc_data["cost_of_equity"]
        return compute_dcf_for_ticker_fcfe(ticker, cost_of_equity, g_years, years, g_terminal, mid_year)
    else:
        return compute_dcf_for_ticker_fcff(
            ticker, wacc, g_years, years, g_terminal, mid_year, use_net_debt,
            use_normalized_fcff, normalized_years, adjust_sbc, three_stage, g1, y1, g2, y2
        )

def dcf_sensitivity(
    ticker: str,
    wacc_grid: List[float] = None,
    gterm_grid: List[float] = None,
    g_years: float = 0.08,
    years: int = 5,
    use_net_debt: bool = True,
    mid_year: bool = True,
    use_normalized_fcff: bool = True,
    normalized_years: int = 3,
    adjust_sbc: bool = False
) -> pd.DataFrame:
    if wacc_grid is None: wacc_grid = [0.07,0.08,0.09,0.10,0.11]
    if gterm_grid is None: gterm_grid = [0.01,0.02,0.025,0.03,0.035]

    snap = fetch_snapshot(ticker)
    fcff_ttm = snap.get("fcff_ttm", np.nan)
    shares   = snap.get("shares_outstanding", np.nan)

    if use_normalized_fcff:
        stock = yf.Ticker(ticker)
        fcff_norm = _fcff_normalized(stock, years=normalized_years, adjust_sbc=adjust_sbc)
        if fcff_norm is not None and np.isfinite(fcff_norm) and fcff_norm > 0:
            fcff_ttm = fcff_norm  # usar normalizado

    if pd.isna(fcff_ttm) or not np.isfinite(fcff_ttm) or fcff_ttm <= 0:
        return pd.DataFrame()
    if pd.isna(shares) or not np.isfinite(shares) or shares <= 0:
        return pd.DataFrame()

    net_debt = _get_net_debt_for_ticker(ticker) if use_net_debt else 0.0
    if pd.isna(net_debt) or not np.isfinite(net_debt): net_debt = 0.0

    table=[]
    for w in wacc_grid:
        row=[]
        for gt in gterm_grid:
            if (not np.isfinite(w)) or (not np.isfinite(gt)) or (w <= gt):
                row.append(np.nan); continue
            ev = dcf_2stage_fcff(fcf0=fcff_ttm, wacc=w, g_years=g_years, years=years, g_terminal=gt, mid_year=mid_year)
            if pd.isna(ev):
                row.append(np.nan); continue
            equity_value = ev - net_debt
            pps = equity_value / shares if shares > 0 else np.nan
            row.append(pps if np.isfinite(pps) else np.nan)
        table.append(row)

    df = pd.DataFrame(table, index=[f"WACC {w*100:.1f}%" for w in wacc_grid], columns=[f"gT {gt*100:.1f}%" for gt in gterm_grid])
    return df

# ================================ CRECIMIENTOS (SUGERENCIAS) ================================

def _series_from_financials(df: pd.DataFrame, key: str) -> pd.Series:
    if df is None or df.empty or key not in df.index:
        return pd.Series(dtype=float)
    df = _order_cols_by_date(df)
    return df.loc[key].dropna()

def _series_fcf_annual(stock: yf.Ticker) -> pd.Series:
    try:
        acf = stock.cashflow
    except Exception:
        acf = None
    if acf is None or acf.empty:
        return pd.Series(dtype=float)
    acf = _order_cols_by_date(acf)
    cfo = None
    for k in CF_CFO_KEYS:
        if k in acf.index:
            cfo = acf.loc[k].dropna(); break
    capex = None
    for k in CF_CAPEX_KEYS:
        if k in acf.index:
            capex = acf.loc[k].dropna(); break
    if cfo is None or capex is None or cfo.empty or capex.empty:
        return pd.Series(dtype=float)
    aligned = pd.concat([cfo, capex], axis=1, join="inner")
    aligned.columns = ["CFO","CapEx"]
    fcf = aligned["CFO"] + aligned["CapEx"]
    return fcf.dropna()

def _cagr_from_series_robusta(series: pd.Series, min_points: int = 3, max_points: int = 5) -> Optional[float]:
    if series is None or series.empty: return None
    s = series.dropna().astype(float)
    if s.size < min_points: return None
    N = min(max_points, s.size)
    sN = s.iloc[:N].iloc[::-1]
    if (sN > 0).all():
        try:
            first, last = float(sN.iloc[0]), float(sN.iloc[-1])
            periods = N - 1
            if periods > 0 and first > 0 and last > 0:
                return (last/first)**(1.0/periods) - 1.0
        except Exception:
            pass
    growth_rates=[]
    for i in range(1,len(sN)):
        prev_val = float(sN.iloc[i-1]); curr_val = float(sN.iloc[i])
        if prev_val > 0 and curr_val > 0:
            growth_rates.append((curr_val/prev_val) - 1.0)
    if len(growth_rates) >= min_points-1:
        return float(np.median(growth_rates))
    return None

_cagr_from_series = _cagr_from_series_robusta

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def estimate_growth(ticker: str) -> Dict:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        country = (info.get("country") or "USA").upper()
        sector = info.get("sector", "") or ""
        try:
            fin_a = stock.financials
        except Exception:
            fin_a = None
        rev_series = _series_from_financials(fin_a, "Total Revenue")
        rev_cagr = _cagr_from_series(rev_series, min_points=3, max_points=5)
        fcf_series = _series_fcf_annual(stock)
        fcf_cagr = _cagr_from_series(fcf_series, min_points=3, max_points=5)
        if rev_cagr is not None and fcf_cagr is not None:
            g_years_raw = 0.60*rev_cagr + 0.40*fcf_cagr
        elif rev_cagr is not None:
            g_years_raw = rev_cagr
        elif fcf_cagr is not None:
            g_years_raw = fcf_cagr
        else:
            if "Technology" in sector: g_years_raw = 0.07
            elif "Healthcare" in sector: g_years_raw = 0.05
            elif "Financial" in sector: g_years_raw = 0.04
            else: g_years_raw = 0.05
        g_years_suggested = _clamp(g_years_raw, lo=0.00, hi=0.18)
        if "USA" in country: base_gt = 0.022
        elif country in ("IRELAND","NETHERLANDS","SINGAPORE","SWITZERLAND"): base_gt = 0.020
        elif country in ("BRAZIL","INDIA","MEXICO","INDONESIA"): base_gt = 0.025
        else: base_gt = 0.020
        if "Technology" in sector: base_gt += 0.002
        elif "Utilities" in sector or "Energy" in sector: base_gt -= 0.003
        g_terminal_suggested = _clamp(base_gt, lo=0.010, hi=0.030)
        return {
            "ticker": ticker.upper(),
            "suggestions": {"g_years": g_years_suggested, "g_terminal": g_terminal_suggested},
            "diagnostics": {
                "country": country, "sector": sector,
                "revenue_cagr_3to5y": rev_cagr, "fcf_cagr_3to5y": fcf_cagr,
                "revenue_series_recent": rev_series.iloc[:5].to_dict() if rev_series is not None and not rev_series.empty else {},
                "fcf_series_recent": fcf_series.iloc[:5].to_dict() if fcf_series is not None and not fcf_series.empty else {}
            }
        }
    except Exception as e:
        return {"ticker": ticker.upper(), "error": f"estimate_growth failed: {e}"}

# ================================ SCORING PRINCIPAL ================================

def _get_peers_with_fallback(ticker_data: Dict) -> List[str]:
    scope = RELATIVE_SCOPE
    name = ticker_data.get('industry' if scope == 'industry' else 'sector', '')
    peers = fetch_peers(scope, name, ticker_data['ticker'])
    if len(peers) < MIN_PEERS:
        fallback_scope = FALLBACK_SCOPE.get(scope)
        if fallback_scope:
            fallback_name = ticker_data.get('sector' if fallback_scope == 'sector' else 'industry', '')
            peers = fetch_peers(fallback_scope, fallback_name, ticker_data['ticker'])
    if len(peers) < MIN_PEERS:
        peers = fetch_peers("universe", "", ticker_data['ticker'])
    return peers

def _create_error_result(ticker: str, error_msg: str) -> Dict:
    return {
        'ticker': ticker.upper(),'error': error_msg,'score': np.nan,'conclusion': 'ERROR',
        'as_of': {'price_date': 'N/D','fund_date': 'N/D'},'source': 'error'
    }

def compute_score(ticker: str) -> Dict:
    ticker_data = fetch_snapshot(ticker)
    if pd.isna(ticker_data['price']): return _create_error_result(ticker, "No se pudieron obtener datos")
    price_history = fetch_history(ticker)
    if len(price_history) < 200: return _create_error_result(ticker, "Histórico insuficiente")
    peers = _get_peers_with_fallback(ticker_data)
    if not peers: return _create_error_result(ticker, "No se encontraron peers")

    peers_data=[]
    for peer in peers:
        if peer != ticker:
            peer_data = fetch_snapshot(peer)
            if pd.notna(peer_data['price']):
                peers_data.append(peer_data)
    peers_data.append(ticker_data)

    valid_peer_count = sum(1 for p in peers_data if p['ticker'] != ticker and pd.notna(p.get('price')))

    valuation_score, valuation_details = valuation_subscore(ticker_data, peers_data)
    quality_score, quality_details     = quality_subscore(ticker_data, peers_data)
    growth_score, growth_details       = growth_subscore(ticker_data, peers_data)
    momentum_score, momentum_details   = momentum_subscore(ticker_data, price_history.values)

    total_score = (
        valuation_score*WEIGHTS['valuation'] +
        quality_score*WEIGHTS['quality'] +
        growth_score*WEIGHTS['growth'] +
        momentum_score*WEIGHTS['momentum']
    )

    if total_score > NEUTRAL_BAND: conclusion = "INFRAVALORADO"
    elif total_score < -NEUTRAL_BAND: conclusion = "SOBREVALUADO"
    else: conclusion = "NEUTRAL"

    return {
        'ticker': ticker.upper(),
        'basics': {
            'price': ticker_data['price'], 'sector': ticker_data['sector'], 'industry': ticker_data['industry'],
            'market_cap': ticker_data['market_cap'], 'ev_corp': ticker_data.get('ev_corp', np.nan)
        },
        'valuation': {
            'pe_ttm': ticker_data['pe_ttm'],'pe_fwd': ticker_data['pe_fwd'],'ev_ebitda': ticker_data['ev_ebitda'],
            'ps': ticker_data['ps'],'pb': ticker_data['pb'],'fcf_yield': ticker_data['fcf_yield'],
            'fcf_ttm': ticker_data.get('fcf_ttm', np.nan),'fcf_ev_yield': ticker_data.get('fcf_ev_yield', np.nan),
            'fcfe_ttm': ticker_data.get('fcfe_ttm', np.nan),'fcfe_yield': ticker_data.get('fcfe_yield', np.nan),
            'fcff_ttm': ticker_data.get('fcff_ttm', np.nan),'fcff_ev_yield': ticker_data.get('fcff_ev_yield', np.nan),
            'subscore': valuation_score,'detail': valuation_details
        },
        'quality': {
            'roic': ticker_data['roic'],'op_margin': ticker_data['operating_margin'],
            'profit_margin': ticker_data['profit_margin'],'net_cash_ebitda': ticker_data['net_cash_ebitda'],
            'roe': ticker_data['roe'],'subscore': quality_score,'detail': quality_details
        },
        'growth': {
            'rev_yoy': ticker_data['revenue_growth'],'eps_yoy': ticker_data['earnings_growth'],
            'subscore': growth_score,'detail': growth_details
        },
        'momentum': {
            'r12_1': momentum_details.get('mom12_1', np.nan),'r6': momentum_details.get('mom6', np.nan),
            'rsi14': momentum_details.get('rsi', np.nan),'vol_252': momentum_details.get('vol_252', np.nan),
            'subscore': momentum_score,'detail': momentum_details
        },
        'score': total_score,'conclusion': conclusion,'as_of': ticker_data['as_of'],'source': ticker_data['source'],
        'peers_used': valid_peer_count,'peers_scope': RELATIVE_SCOPE,'relative_available': valid_peer_count >= MIN_PEERS,
        'meta': {'equity_nonpositive': ticker_data.get('equity_nonpositive', False)}
    }

# ================================ REPORTES BREVES ================================

def render_stock_report(result: Dict) -> str:
    if 'error' in result:
        return f"❌ Error analizando {result['ticker']}: {result['error']}"
    t = result['ticker']; b = result['basics']; v = result['valuation']; q = result['quality']; g = result['growth']; m = result['momentum']
    equity_nonpos = result.get('meta', {}).get('equity_nonpositive', False)
    pb_str  = "NM" if (equity_nonpos and pd.isna(v['pb'])) else fmt_num(v['pb'])
    roe_str = "NM" if (equity_nonpos and pd.isna(q['roe'])) else fmt_pct(q['roe'])
    rep  = f"\n{'='*60}\n📊 REPORTE DE VALORACIÓN v4.1 - {t}\n{'='*60}\n"
    rep += f"\n INFORMACIÓN BÁSICA:\n   Precio: ${fmt_num(b['price'])}\n   Market Cap: ${fmt_int(b['market_cap'])}\n"
    if pd.notna(b.get('ev_corp', np.nan)): rep += f"   EV (corp): ${fmt_int(b['ev_corp'])}\n"
    rep += f"   Sector: {b['sector']}\n   Industria: {b['industry']}\n"
    rep += f"\n VALORACIÓN:\n   P/E (TTM): {fmt_num(v['pe_ttm'])} | P/E (Forward): {fmt_num(v['pe_fwd'])}\n"
    rep += f"   EV/EBITDA: {fmt_num(v['ev_ebitda'])}\n   P/S: {fmt_num(v['ps'])}\n   P/B: {pb_str}\n"
    rep += f"   FCFE Yield: {fmt_pct(v.get('fcfe_yield', np.nan))}\n"
    if pd.notna(v.get('fcfe_ttm', np.nan)): rep += f"   FCFE TTM: ${fmt_int(v['fcfe_ttm'])}\n"
    if pd.notna(b.get('ev_corp', np.nan)) and pd.notna(v.get('fcff_ev_yield', np.nan)):
        rep += f"   FCFF/EV Yield: {fmt_pct(v['fcff_ev_yield'], fmt='{:.1%}')}\n"
    if pd.notna(v.get('fcff_ttm', np.nan)): rep += f"   FCFF TTM: ${fmt_int(v['fcff_ttm'])}\n"
    rep += f"   Valuation Subscore: {fmt_num(v['subscore'], fmt='{:.2f}')}\n"
    rep += f"\n🏆 CALIDAD:\n   ROIC: {fmt_pct(q['roic'])}\n   Margen Operativo: {fmt_pct(q['op_margin'])}\n   Margen Neto: {fmt_pct(q['profit_margin'])}\n   ROE: {roe_str}\n   Quality Subscore: {fmt_num(q['subscore'], fmt='{:.2f}')}\n"
    rep += f"\n CRECIMIENTO:\n   Ventas YoY: {fmt_pct(g['rev_yoy'])}\n   EPS YoY: {fmt_pct(g['eps_yoy'])}\n   Growth Subscore: {fmt_num(g['subscore'], fmt='{:.2f}')}\n"
    rep += f"\n⚡ MOMENTUM:\n   R12-1m: {fmt_pct(m['r12_1'])}\n   R6m: {fmt_pct(m['r6'])}\n   RSI(14): {fmt_num(m['rsi14'], fmt='{:.1f}')}\n   Volatilidad: {fmt_pct(m['vol_252'])}\n   Momentum Subscore: {fmt_num(m['subscore'], fmt='{:.2f}')}\n"
    rep += f"\n CONCLUSIÓN:\n   Score Total: {fmt_num(result['score'], fmt='{:.2f}')}\n   Conclusión: {result['conclusion']}\n"
    rep += f"\n📅 METADATOS:\n   Fecha de precios: {result['as_of']['price_date']}\n   Fecha de fundamentales: {result['as_of']['fund_date']}\n   Fuente: {result['source']}\n   Peers usados (válidos): {result['peers_used']} ({result['peers_scope']})\n"
    rep += f"\n{'='*60}\n"
    return rep

# ================================ ROIC FALLBACK ================================

def _get_ebit_ttm(stock: yf.Ticker) -> Optional[float]:
    qfin = getattr(stock, "quarterly_financials", None)
    ebit_4q = _get_row_sum_last_n(qfin, INCOME_EBIT_KEYS, 4) if qfin is not None else None
    if ebit_4q is not None: return float(ebit_4q)
    fin = getattr(stock, "financials", None)
    ebit_annual = _get_row_last(fin, INCOME_EBIT_KEYS) if fin is not None else None
    return float(ebit_annual) if ebit_annual is not None else None

def _estimate_effective_tax_rate(stock: yf.Ticker, info: Dict, default: float = 0.21) -> float:
    try:
        t = info.get("effectiveTaxRate", None)
        if t is not None and np.isfinite(float(t)): return float(np.clip(float(t), 0.0, 0.40))
    except Exception:
        pass
    qfin = getattr(stock, "quarterly_financials", None)
    tax_4q = _get_row_sum_last_n(qfin, INCOME_TAX_EXPENSE_KEYS, 4) if qfin is not None else None
    pretax_4q = _get_row_sum_last_n(qfin, INCOME_PRETAX_KEYS, 4) if qfin is not None else None
    if tax_4q is not None and pretax_4q and pretax_4q > 0:
        rate = abs(float(tax_4q)) / float(pretax_4q)
        return float(np.clip(rate, 0.0, 0.40))
    fin = getattr(stock, "financials", None)
    tax_an = _get_row_last(fin, INCOME_TAX_EXPENSE_KEYS) if fin is not None else None
    pretax_an = _get_row_last(fin, INCOME_PRETAX_KEYS) if fin is not None else None
    if tax_an is not None and pretax_an and pretax_an > 0:
        rate = abs(float(tax_an)) / float(pretax_an)
        return float(np.clip(rate, 0.0, 0.40))
    return default

def _value_at(df: pd.DataFrame, keys: List[str], col_idx: int) -> Optional[float]:
    if df is None or df.empty: return None
    for k in keys:
        if k in df.index:
            v = df.loc[k].iloc[col_idx]
            if pd.notna(v): return float(v)
    return None

def _invested_capital_avg(stock: yf.Ticker) -> Optional[float]:
    for getter in ("quarterly_balance_sheet","balance_sheet"):
        try:
            bs = getattr(stock, getter)
            if bs is None or bs.empty: continue
            bs = _order_cols_by_date(bs)
            cols = bs.columns
            if len(cols) == 0: continue
            def ic_at(idx: int) -> Optional[float]:
                debt = _debt_at(bs, idx)
                equity = _value_at(bs, BS_EQUITY_KEYS, idx)
                cash = _get_cash_like(bs) if idx == 0 else _value_at(bs, BS_CASH_KEYS, idx)
                if debt is None or equity is None or cash is None: return None
                if equity <= 0: return None
                return float(debt + equity - cash)
            ic0 = ic_at(0); ic1 = ic_at(1) if len(cols) > 1 else None
            if ic0 is not None and ic1 is not None and ic0 > 0 and ic1 > 0:
                return float((ic0 + ic1) / 2.0)
            if ic0 is not None and ic0 > 0:
                return float(ic0)
        except Exception:
            continue
    return None

def _estimate_roic_fallback(stock: yf.Ticker, info: Dict) -> Optional[float]:
    ebit_ttm = _get_ebit_ttm(stock)
    if ebit_ttm is None or not np.isfinite(ebit_ttm): return None
    tax_rate = _estimate_effective_tax_rate(stock, info, default=0.21)
    nopat = float(ebit_ttm) * (1.0 - float(tax_rate))
    ic_avg = _invested_capital_avg(stock)
    if ic_avg is None or not np.isfinite(ic_avg) or ic_avg <= 0: return None
    return float(nopat / ic_avg)

# ================================ MENÚ INTERACTIVO ================================

def analizar_stock(ticker: str) -> Dict:
    print(f"🔎 Analizando {ticker.upper()}...")
    result = compute_score(ticker)
    if 'error' not in result:
        print(render_stock_report(result))
    else:
        print(f"❌ Error: {result['error']}")
    return result

def menu_principal():
    """Menú principal del sistema de análisis."""
    while True:
        print("\n" + "="*60)
        print("📊 SISTEMA DE ANÁLISIS DE VALORACIÓN DE STOCKS v4.1")
        print("="*60)
        print("1. Analizar stock individual")
        print("2. Valorar por DCF (FCFF, 2 o 3 etapas)")
        print("3. Salir")
        print("="*60)

        opcion = input("Seleccione una opción (1-3): ").strip()

        if opcion == "1":
            ticker = input("Ingrese el ticker del stock (ej: AAPL): ").strip().upper()
            if ticker: analizar_stock(ticker)
            else: print("❌ Ticker inválido")

        elif opcion == "2":
            ticker = input("Ticker para DCF (ej: AAPL): ").strip().upper()
            if not ticker:
                print("❌ Ticker inválido"); continue

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
                g_years_auto = 0.08; g_term_auto = 0.022
                print("⚠️ Crecimientos auto no disponibles; usando 8% y 2.2%")
            else:
                g_years_auto = eg["suggestions"]["g_years"]
                g_term_auto  = eg["suggestions"]["g_terminal"]
                print(f"📈 Sugerencias de crecimiento → g_years={g_years_auto:.2%}, g_terminal={g_term_auto:.2%}")

            # 3) Overrides rápidos
            try:
                g_years = float(input(f"Crecimiento años explícitos [{g_years_auto:.3f}]: ") or f"{g_years_auto:.6f}")
                years   = int(input("Años explícitos (por defecto 5): ") or "5")
                g_term  = float(input(f"Crecimiento terminal [{g_term_auto:.3f}]: ") or f"{g_term_auto:.6f}")
            except Exception:
                print("❌ Parámetros inválidos"); continue

            # 3b) Opciones DCF realista
            use_norm = (input("¿Usar FCFF normalizado 3Y? [S/n]: ").strip().lower() or "s") != "n"
            adj_sbc  = (input("¿Restar SBC en FCFF? [s/N]: ").strip().lower() or "n") == "s"
            three_st = (input("¿Usar DCF 3 etapas (g1→g2→gT)? [s/N]: ").strip().lower() or "n") == "s"
            if three_st:
                try:
                    g1 = float(input("  g1 (alta) [0.12]: ") or "0.12")
                    y1 = int(input("  años g1 [3]: ") or "3")
                    g2 = float(input("  g2 (transición) [0.08]: ") or "0.08")
                    y2 = int(input("  años g2 [3]: ") or "3")
                except Exception:
                    print("❌ Parámetros 3 etapas inválidos"); continue
            else:
                g1=y1=g2=y2=None,None,None,None  # no usados

            # 4) Ejecutar DCF
            res = compute_dcf_for_ticker(
                ticker, wacc=wacc, g_years=g_years, years=years, g_terminal=g_term,
                use_net_debt=True, method="fcff", mid_year=True,
                use_normalized_fcff=use_norm, normalized_years=3, adjust_sbc=adj_sbc,
                three_stage=three_st, g1=(g1 or 0.12), y1=(y1 or 3), g2=(g2 or 0.08), y2=(y2 or 3)
            )

            if "error" in res:
                print("❌", res["error"])
            else:
                out = res["outputs"]; inp = res["inputs"]
                print("\n🧮 RESULTADO DCF")
                print(f"   Valor razonable/acción: {fmt_num(out['fair_value_per_share'])}")
                if pd.notna(out.get("upside", np.nan)):
                    print(f"   Upside: {fmt_pct(out['upside'])}")
                print(f"   EV (PV): {fmt_int(out['enterprise_value'])} | Equity: {fmt_int(out['equity_value'])}")
                if pd.notna(inp.get("tv_share", np.nan)):
                    print(f"   TV share (PV TV / EV): {fmt_pct(inp['tv_share'])}")
                print(f"   Asumptions: WACC={res['assumptions']['wacc']:.2%}, "
                      f"g_exp={res['assumptions']['g_years']:.2%}, years={res['assumptions']['years']}, "
                      f"gT={res['assumptions']['g_terminal']:.2%}, "
                      f"{'3 etapas' if res['assumptions']['three_stage'] else '2 etapas'}; "
                      f"FCFF usado={'normalizado 3Y' if res['assumptions']['use_normalized_fcff'] else 'TTM'}"
                      f"{' (SBC restado)' if res['assumptions']['adjust_sbc'] else ''}")

                # 5) DEBUG de insumos
                snap = fetch_snapshot(ticker)
                stock = yf.Ticker(ticker)
                cash_bs, debt_bs = _get_cash_debt_last(stock)
                nd = (debt_bs or 0) - (cash_bs or 0)
                print("\n DEBUG DCF INPUTS —", ticker.upper())
                print(f"  Price: {fmt_num(snap['price'])}")
                print(f"  FCFF_TTM: {fmt_int(snap['fcff_ttm'])} | FCFF_usado: {fmt_int(inp['fcff_used'])}")
                print(f"  Shares: {fmt_int(snap['shares_outstanding'])}")
                print(f"  Cash-like (BS): {fmt_int(cash_bs)}  Debt (BS): {fmt_int(debt_bs)}  NetDebt: {fmt_int(nd)}")
                print(f"  As of: {snap['as_of']}")

        elif opcion == "3":
            print("👋 ¡Hasta luego!")
            break
        else:
            print("❌ Opción inválida")

# ================================ MAIN ================================

if __name__ == "__main__":
    print("📦 MÓDULO DE ANÁLISIS DE VALORACIÓN DE STOCKS v4.1")
    print("="*60)
    try:
        menu_principal()
    except KeyboardInterrupt:
        print("\n\n👋 Programa interrumpido por el usuario. ¡Hasta luego!")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        print("Por favor, reporte este error al desarrollador.")
