# -*- coding: utf-8 -*-
"""
PIPELINE MACRO PARA GESTIÓN DE CARTERA — SOLO YAHOO FINANCE
-----------------------------------------------------------
- SIN FRED (todo desde Yahoo Finance)
- Regímenes walk-forward (KMeans, sin look-ahead)
- Correlaciones con lags ±126 y p-values HAC (Newey–West)
- Señales: VIX, curva 10–3M (Y10 Yahoo /10 − ^IRX), DXY (fallback UUP)
- Residuación multifactor (SPX, Δ10Y Yahoo, USD)
- Superficie de IV real: múltiples expiraciones, limpieza de quotes laxa, q por paridad
"""

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from scipy.stats import norm
from scipy.optimize import brentq
import statsmodels.api as sm

# Cache HTTP opcional (Yahoo)
try:
    import requests_cache
    requests_cache.install_cache('macro_cache', expire_after=24*3600)
except Exception:
    pass

# ================================ CONFIG =====================================

START_DATE = "2015-01-01"
END_DATE = None  # hoy si None
USE_RESIDUAL_CORRS = True
MAX_LAG = 126  # ~6 meses hábiles

# Activos/indicadores SOLO Yahoo
YF_TICKERS = {
    'VIX': '^VIX',
    'DXY': 'DX-Y.NYB',     # fallback: UUP normalizado
    'EUR_USD': 'EURUSD=X',
    'GBP_USD': 'GBPUSD=X',
    'GOLD': 'GC=F',
    'OIL': 'CL=F',
    'COPPER': 'HG=F',
    'SP500': '^GSPC',
    'NASDAQ': '^IXIC',
    'DOW': '^DJI',
    'RUSSELL2000': '^RUT',
    'TLT': 'TLT',
    'IEF': 'IEF',
    'SHY': 'SHY',
    'Y10': '^TNX',     # 10y yield (x10 en Yahoo)
    'Y30': '^TYX',     # 30y yield (x10)
    'Y03M': '^IRX',    # 13-week T-bill (%)
}

# Cartera (se renormalizan pesos a los presentes)
PORTFOLIO_TICKERS = ["0P0001CLDK.F","NVDA","MSFT","AAPL","GOOGL","IBM","AMZN","META","TSLA","JPM","BRK-A","BTC-EUR","GLD"]
PORTFOLIO_WEIGHTS = np.array([0.514, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.486])

# ================================ HELPERS =====================================

def _today_str():
    return datetime.today().strftime('%Y-%m-%d')

def _ensure_end_date(end_date):
    return _today_str() if end_date is None else end_date

def _safe_series(s):
    return pd.Series(s).dropna() if s is not None else pd.Series(dtype=float)

def _set_style():
    try:
        plt.style.use('seaborn-v0_8')
    except Exception:
        plt.style.use('default')

def pct_log(s: pd.Series) -> pd.Series:
    return np.log(s).diff()

def to_business_daily(s: pd.Series) -> pd.Series:
    return s.asfreq('B').ffill()

def _now_naive():
    # UTC sin tz (naive) para casar con expiraciones de Yahoo
    return pd.Timestamp.utcnow().tz_localize(None)

# ============================ DESCARGA DE DATOS ===============================

def download_yahoo_series(ticker_list, start=START_DATE, end=None):
    """Descarga cierres (auto_adjust=True) de Yahoo. Devuelve dict[ticker] -> pd.Series."""
    end = _ensure_end_date(end)
    tickers = list(ticker_list) if isinstance(ticker_list, (list, tuple, set)) else [ticker_list]
    out = {}
    try:
        df = yf.download(
            tickers, start=start, end=end,
            progress=False, auto_adjust=True, threads=True
        )

        # Caso 1: un solo ticker
        if len(tickers) == 1:
            if isinstance(df, pd.DataFrame) and ('Close' in df.columns):
                s = _safe_series(df['Close'])
            else:
                s = _safe_series(df.squeeze())
            if len(s):
                s.name = tickers[0]
                out[tickers[0]] = s
            else:
                print(f"[Yahoo] sin datos: {tickers[0]}")
            return out

        # Caso 2: varios tickers → normalmente MultiIndex
        if isinstance(df.columns, pd.MultiIndex) and ('Close' in df.columns.get_level_values(0)):
            close = df['Close']
        elif 'Close' in df.columns:
            close = df[['Close']]
        else:
            close = df

        for t in tickers:
            s = _safe_series(close.get(t))
            if len(s):
                s.name = t
                out[t] = s
            else:
                print(f"[Yahoo] sin datos: {t}")

    except Exception as e:
        print(f"[Yahoo] Error batch: {e}")

    return out

def download_yahoo_assets(start=START_DATE, end=None):
    print("Descargando activos Yahoo...")
    batch = download_yahoo_series(list(YF_TICKERS.values()), start, end)
    out = {}
    for name, t in YF_TICKERS.items():
        if t in batch and len(batch[t]):
            out[name] = batch[t]

    # Fallback DXY → UUP normalizado
    if ('DXY' not in out) or (not len(out['DXY'])):
        print("DXY no disponible. Usando UUP como proxy normalizado.")
        uup = download_yahoo_series(['UUP'], start, end).get('UUP', pd.Series(dtype=float))
        if len(uup):
            out['DXY'] = uup / uup.iloc[0] * 100.0

    return out

def download_portfolio_prices(tickers, start=START_DATE, end=None):
    print("Descargando precios de cartera...")
    out = download_yahoo_series(tickers, start, end)
    for t in tickers:
        if t not in out:
            print(f"  - {t}: sin datos; se excluye")
    return out

def build_portfolio_returns(portfolio_prices: dict, weights: np.ndarray):
    if not portfolio_prices:
        raise ValueError("No hay precios de cartera.")
    prices = pd.concat(portfolio_prices, axis=1, join='inner').dropna(how='all')
    prices.columns = list(portfolio_prices.keys())
    rets = prices.apply(pct_log).dropna(how='all')  # log-returns

    cols = rets.columns
    w = pd.Series(0.0, index=cols)
    for i, c in enumerate(PORTFOLIO_TICKERS):
        if c in cols:
            w[c] = float(weights[i])
    if w.sum() == 0:
        raise ValueError("Pesos efectivos suman 0.")
    w = w / w.sum()
    port = (rets * w).sum(axis=1, min_count=1).dropna()
    return rets, port, w

# ========================== TRANSFORMACIONES MACRO ============================

def transform_macro_for_correlation(macro: dict, idx_target: pd.DatetimeIndex):
    """
    Transformaciones SOLO para Yahoo:
    - Yields (^TNX,^TYX → /10) y ^IRX → Δ (pp)
    - Precios (FX/commodities/índices) → log-returns
    """
    X = {}
    for k, s in macro.items():
        s = s.dropna()
        if not len(s):
            continue
        if k in ('Y03M',):
            X[k] = s.diff()
        elif k in ('Y10', 'Y30'):
            X[k] = (s/10.0).diff()  # Yahoo escala x10
        else:
            X[k] = pct_log(s)

    df = pd.DataFrame(index=idx_target)
    for k, s in X.items():
        df[k] = to_business_daily(s).reindex(df.index).ffill()
    return X, df

# ============================ CORRELACIONES (LAGS) ============================

def lagged_corr_df(y: pd.Series, x: pd.Series, max_lag=126) -> pd.DataFrame:
    res = []
    for k in range(-max_lag, max_lag+1):
        ys = y.copy()
        xs = x.shift(k)
        df = pd.concat([ys, xs], axis=1).dropna()
        if len(df) < 60:
            continue
        rho = df.iloc[:, 0].corr(df.iloc[:, 1])
        X = sm.add_constant(df.iloc[:, 1].values)
        model = sm.OLS(df.iloc[:, 0].values, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
        res.append((k, rho, float(model.tvalues[1]), float(model.pvalues[1]), len(df)))
    return pd.DataFrame(res, columns=['lag', 'corr', 't', 'p', 'n'])

def best_lagged_correlation(port_series: pd.Series, macro_df: pd.DataFrame, max_lag=126):
    out = []
    for col in macro_df.columns:
        try:
            stats = lagged_corr_df(port_series, macro_df[col], max_lag)
            if not stats.empty:
                best = stats.iloc[stats['corr'].abs().values.argmax()]
                out.append({'indicador': col,
                            'lag': int(best['lag']),
                            'corr': float(best['corr']),
                            't': float(best['t']),
                            'p': float(best['p']),
                            'n': int(best['n'])})
        except Exception:
            continue
    res = pd.DataFrame(out).sort_values('corr', key=np.abs, ascending=False)
    return res

# ======================= RESIDUACIÓN MULTIFACTOR (HAC) =======================

def orthogonalize_portfolio(y: pd.Series, macro: dict):
    """
    Factores SOLO Yahoo: SPX ret, Δ10Y (Yahoo), USD ret (DXY).
    """
    x_cols = {}
    if 'SP500' in macro: x_cols['SPX']  = pct_log(macro['SP500'])
    if 'Y10'   in macro: x_cols['d10y'] = (macro['Y10']/10.0).diff()
    if 'DXY'   in macro: x_cols['USD']  = pct_log(macro['DXY'])

    X = pd.DataFrame(x_cols).dropna()
    y_ = y.reindex(X.index).dropna()
    X = X.reindex(y_.index).dropna()
    y_ = y_.reindex(X.index)
    if len(X) < 100:
        return y.rename('portfolio_resid'), None

    Xc = sm.add_constant(X)
    model = sm.OLS(y_.values, Xc.values).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
    resid = pd.Series(y_.values - model.fittedvalues, index=y_.index, name='portfolio_resid')
    return resid, model

# ============================== SEÑALES (Yahoo) ===============================

def rolling_percentile_signal(series: pd.Series, win=252, upper=0.8, lower=0.2):
    s = series.dropna()
    if len(s) < win: return 'NEUTRAL', np.nan
    window = s.iloc[-win:]
    pctl = (window <= window.iloc[-1]).mean()
    if pctl >= upper: return 'RISK_OFF', pctl
    if pctl <= lower: return 'RISK_ON', pctl
    return 'NEUTRAL', pctl

def generate_signals_percentiles(macro: dict):
    print("Generando señales (Yahoo)...")
    sig = {}

    # VIX
    if 'VIX' in macro and len(macro['VIX']) > 0:
        lvl = macro['VIX']
        state, p = rolling_percentile_signal(lvl, win=252, upper=0.8, lower=0.2)
        sig['VIX'] = {'value': float(lvl.dropna().iloc[-1]), 'signal': state, 'pct': p}

    # Curva: 10-3M (solo Yahoo)
    ten = macro.get('Y10'); thr = macro.get('Y03M')
    if ten is not None and thr is not None and len(ten) and len(thr):
        s = (ten/10.0 - thr).dropna()
        if len(s):
            state, p = rolling_percentile_signal(s, win=252, upper=0.3, lower=0.1)
            signal = 'INVERTED' if s.iloc[-1] < 0 else state
            sig['YIELD_CURVE_10_3M'] = {'value': float(s.iloc[-1]), 'signal': signal, 'pct': p}

    # USD (DXY)
    if 'DXY' in macro and len(macro['DXY']) >= 60:
        state, p = rolling_percentile_signal(macro['DXY'], win=252, upper=0.8, lower=0.2)
        sig['DXY'] = {'value': float(macro['DXY'].dropna().iloc[-1]),
                      'signal': 'USD_STRENGTH' if state=='RISK_OFF' else ('USD_WEAKNESS' if state=='RISK_ON' else 'USD_NEUTRAL'),
                      'pct': p}
    return sig

# ============================== REGÍMENES W-F =================================

def walkforward_regimes(returns: pd.Series, rf_series: pd.Series=None, window=252, step=21, n_clusters=3):
    """
    rf_series (opcional): usa ^IRX (Y03M) si está; si no, Sharpe sin rf.
    """
    idx = returns.index
    rf_daily = None
    if rf_series is not None and len(rf_series):
        rf_daily = to_business_daily(rf_series).reindex(idx).ffill()/100.0/252.0
    excess = returns - (rf_daily if rf_daily is not None else 0.0)

    reg_feats = pd.DataFrame(index=idx)
    reg_feats['vol'] = excess.rolling(window).std()*np.sqrt(252)
    reg_feats['ret'] = excess.rolling(window).mean()*252
    cum = (1+returns.fillna(0)).cumprod()
    reg_feats['mdd'] = (cum / cum.rolling(window).max() - 1).rolling(window).min()
    reg_feats['sharpe'] = reg_feats['ret'] / reg_feats['vol'].replace(0, np.nan)

    labels = pd.Series(index=idx, dtype=int)
    scaler = StandardScaler()

    for t in range(window*2, len(idx), step):
        sub = reg_feats.iloc[:t].dropna()
        if len(sub) < 100: 
            continue
        X = scaler.fit_transform(sub[['vol','ret','sharpe','mdd']])
        km = KMeans(n_clusters=n_clusters, n_init=25, random_state=42).fit(X)
        pred = km.predict(X)[-step:]
        labels.iloc[t-step:t] = pred

    stats_ = pd.DataFrame({'lab': labels, 'vol': reg_feats['vol'], 'ret': reg_feats['ret']}).dropna()
    name_map = {}
    if not stats_.empty:
        m = stats_.groupby('lab')[['vol','ret']].mean()
        for r in m.index:
            vol, ret = m.loc[r, 'vol'], m.loc[r, 'ret']
            if vol > m['vol'].quantile(0.7):
                name_map[r] = 'High Vol Bull' if ret > 0 else 'High Vol Bear'
            else:
                name_map[r] = 'Low Vol Bull' if ret > 0 else 'Low Vol Bear'
    labeled = pd.DataFrame({'regime': labels})
    labeled['regime_label'] = labeled['regime'].map(name_map)
    return labeled

# ========================== IV REAL: CALL/PUT + q =============================

def black_scholes_call(S, K, T, r, q, sigma):
    d1 = (np.log(S/K)+(r - q + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def black_scholes_put(S, K, T, r, q, sigma):
    d1 = (np.log(S/K)+(r - q + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)

def implied_vol(option_type, S, K, T, r, q, price):
    f = (lambda s: black_scholes_call(S,K,T,r,q,s)-price) if option_type=='Call' \
        else (lambda s: black_scholes_put(S,K,T,r,q,s)-price)
    try:
        return brentq(f, 1e-4, 3.0, xtol=1e-6, maxiter=200)
    except Exception:
        return np.nan

def _mid_from_row(row):
    b = row.get('bid', np.nan); a = row.get('ask', np.nan); lp = row.get('lastPrice', np.nan)
    if np.isfinite(b) and np.isfinite(a) and (a > 0) and (b > 0) and (a >= b):
        m = (b + a) / 2
        if np.isfinite(lp) and (b <= lp <= a):
            return float(lp)
        return float(m)
    if np.isfinite(lp) and lp > 0: return float(lp)
    if np.isfinite(a) and a > 0:   return float(a)
    if np.isfinite(b) and b > 0:   return float(b)
    return np.nan

def infer_q_from_parity(tk, S, r, exp_str):
    """q implícito por put–call parity en strike ATM (naive time)."""
    try:
        ch = tk.option_chain(exp_str); calls, puts = ch.calls, ch.puts
        if calls is None or puts is None or calls.empty or puts.empty: return None
        strikes = np.intersect1d(np.array(calls['strike']), np.array(puts['strike']))
        if len(strikes) == 0: return None
        K = float(strikes[np.argmin(np.abs(strikes - S))])
        c_row = calls[calls['strike'] == K].iloc[0]
        p_row = puts[puts['strike'] == K].iloc[0]
        C = _mid_from_row(c_row); P = _mid_from_row(p_row)
        if not np.isfinite(C) or not np.isfinite(P): return None

        now = _now_naive()
        T = max((pd.Timestamp(exp_str) - now).days, 1) / 365.0

        disc_r = np.exp(-r * T)
        emqT = (C - P + K * disc_r) / S  # e^{-qT}
        if emqT <= 0: return None
        q = -np.log(emqT) / T
        return max(0.0, float(q))
    except Exception:
        return None

def _clean_quotes(df, typ, S, T, r, q,
                  min_vol=0,
                  require_bidask=False,
                  min_time_value=0.0,
                  mny_range=(0.6, 1.4),
                  ub_rel_tol=0.10,
                  lb_abs_tol=0.01):
    """Limpieza flexible para maximizar nº de opciones válidas."""
    if df is None or df.empty: 
        print(f"  {typ}: DataFrame vacío o None")
        return pd.DataFrame()
    out = df.copy()
    if 'strike' not in out.columns: 
        print(f"  {typ}: Sin columna 'strike'")
        return pd.DataFrame()
    
    print(f"  {typ}: {len(out)} opciones iniciales")

    # requisitos mínimos de liquidez (muy flexibles)
    if 'openInterest' in out.columns:
        before = len(out)
        out = out[out['openInterest'].fillna(0) >= 1]  # muy flexible: solo > 0
        print(f"    Después de openInterest: {len(out)} (eliminadas {before-len(out)})")
    if 'volume' in out.columns:
        before = len(out)
        out = out[out['volume'].fillna(0) >= 0]  # sin filtro de volumen
        print(f"    Después de volume: {len(out)} (eliminadas {before-len(out)})")
    if 'lastTradeDate' in out.columns:
        before = len(out)
        recent = pd.to_datetime(out['lastTradeDate'], errors='coerce')
        recent = recent.dt.tz_localize(None)
        # Solo aplicar filtro si hay fechas válidas, sino mantener todas
        valid_dates = recent.notna()
        if valid_dates.any():
            recent_valid = recent[valid_dates]
            recent_threshold = _now_naive() - pd.Timedelta(days=30)  # 30 días más flexible
            recent_mask = recent_valid >= recent_threshold
            # Mantener opciones con fechas recientes O sin fecha (asumiendo que están activas)
            keep_mask = valid_dates & recent_mask | ~valid_dates
            out = out[keep_mask]
            print(f"    Después de lastTradeDate: {len(out)} (eliminadas {before-len(out)})")
        else:
            print(f"    Después de lastTradeDate: {len(out)} (sin fechas válidas, manteniendo todas)")
    else:
        print(f"    Después de lastTradeDate: {len(out)} (sin columna lastTradeDate)")

    # usar SIEMPRE mid y controlar spread
    req = {'bid','ask'}.issubset(out.columns)
    if not req: 
        print(f"    {typ}: Sin columnas bid/ask")
        return pd.DataFrame()
    before = len(out)
    out = out[(out['bid']>0) & (out['ask']>0) & (out['ask']>=out['bid'])]
    print(f"    Después de bid/ask válidos: {len(out)} (eliminadas {before-len(out)})")
    
    if out.empty:
        print(f"    {typ}: Sin opciones con bid/ask válidos")
        return pd.DataFrame()
        
    mid = (out['bid'] + out['ask'])/2.0
    spread_rel = (out['ask'] - out['bid'])/mid
    before = len(out)
    out = out[spread_rel <= 1.0]  # muy flexible: hasta 100% de spread
    print(f"    Después de spread: {len(out)} (eliminadas {before-len(out)})")
    out['px'] = mid

    # no-arbitraje y valor temporal mínimo
    disc_r = np.exp(-r*T); disc_q = np.exp(-q*T)
    call_lb = np.maximum(0.0, S*disc_q - out['strike']*disc_r)
    put_lb  = np.maximum(0.0, out['strike']*disc_r - S*disc_q)
    lb = call_lb if typ=='Call' else put_lb
    ub = (S*disc_q) if typ=='Call' else (out['strike']*disc_r)
    before = len(out)
    out = out[(out['px'] >= (lb + 0.01)) & (out['px'] <= (ub*1.10))]
    print(f"    Después de no-arbitraje: {len(out)} (eliminadas {before-len(out)})")

    tv = out['px'] - lb
    min_tv = max(0.001, 0.0001 * S * np.sqrt(T))   # muy flexible: casi sin filtro
    before = len(out)
    out = out[tv >= min_tv]
    print(f"    Después de valor temporal: {len(out)} (eliminadas {before-len(out)})")
    
    if out.empty: 
        print(f"    {typ}: Sin opciones tras filtros finales")
        return pd.DataFrame()

    # IV (usamos impliedVolatility de Yahoo como seed si viene)
    y_iv = pd.to_numeric(out.get('impliedVolatility', np.nan), errors='coerce')
    iv = []
    for i, row in out.iterrows():
        K = float(row['strike']); px = float(row['px'])
        guess = float(y_iv.loc[i]) if np.isfinite(y_iv.loc[i]) and 0.01 <= y_iv.loc[i] <= 3.0 else None
        a, b = (0.005, 3.0) if guess is None else (max(0.003, guess/5), min(3.0, guess*5))
        f = (lambda s: black_scholes_call(S,K,T,r,q,s)-px) if typ=='Call' else (lambda s: black_scholes_put(S,K,T,r,q,s)-px)
        try:
            sigma = brentq(f, a, b, xtol=1e-6, maxiter=200)
        except Exception:
            sigma = guess if guess is not None else np.nan
        iv.append(sigma)
    out['iv'] = iv
    before = len(out)
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=['iv'])
    print(f"    Después de limpiar IV: {len(out)} (eliminadas {before-len(out)})")
    
    before = len(out)
    out = out[(out['iv'] > 0.001) & (out['iv'] < 5.0)]  # muy flexible: 0.1% a 500%
    print(f"    Después de rango IV: {len(out)} (eliminadas {before-len(out)})")

    # moneyness (más flexible)
    lo, hi = mny_range
    out['moneyness'] = out['strike'] / S
    before = len(out)
    # Usar rango más amplio si el rango original es muy restrictivo
    lo_flex = min(lo, 0.5)  # al menos 0.5
    hi_flex = max(hi, 2.0)  # al menos 2.0
    out = out[(out['moneyness'] >= lo_flex) & (out['moneyness'] <= hi_flex)]
    print(f"    Después de moneyness: {len(out)} (eliminadas {before-len(out)}, rango {lo_flex:.1f}-{hi_flex:.1f})")

    out['type'] = typ
    print(f"  {typ}: {len(out)} opciones finales válidas")
    return out.sort_values('moneyness')


def _pick_expirations_many(tk, n=12, min_T_days=5, max_T_days=None):
    """
    Devuelve hasta n expiraciones futuras (las más próximas), con T>=min_T_days.
    Evita .abs() en TimedeltaIndex; trabaja en días enteros.
    """
    if not tk.options:
        return []
    exps = pd.to_datetime(tk.options)           # naive
    today = _now_naive().normalize()            # naive
    td = (exps - today)
    try:
        days = td.to_numpy(dtype="timedelta64[D]").astype(int)
    except TypeError:
        days = td.values.astype("timedelta64[D]").astype(int)

    keep = days >= min_T_days
    if max_T_days is not None:
        keep &= (days <= max_T_days)
    exps, days = exps[keep], days[keep]
    if len(exps) == 0:
        return []

    order = np.argsort(days)
    chosen = exps[order][:n]
    return [e.strftime("%Y-%m-%d") for e in chosen]

def real_vol_surface(symbol="SPY", r_annual=None, save=True,
                     n_expirations=12,     # más expiraciones
                     min_T_days=7,
                     max_T_days=None,
                     mny_range=(0.8, 1.2),
                     min_vol=5,
                     require_bidask=False,
                     min_time_value=0.10):
    _set_style()
    tk = yf.Ticker(symbol)
    if not tk.options:
        print("Sin expiraciones disponibles.")
        return None
    spot_hist = tk.history(period="1d")
    if spot_hist is None or spot_hist.empty:
        print("Sin precio spot.")
        return None
    S = float(spot_hist['Close'].iloc[-1])

    # r: ^IRX si disponible; si no, 1% de fallback
    if r_annual is None:
        try:
            irx = yf.download(['^IRX'], period="1mo", progress=False, auto_adjust=False)['Close']['^IRX']
            r_annual = float(irx.dropna().iloc[-1]) / 100.0
        except Exception:
            r_annual = 0.01
    r = float(r_annual)

    expirations = _pick_expirations_many(tk, n=n_expirations, min_T_days=min_T_days, max_T_days=max_T_days)
    if not expirations:
        print("No expiraciones suficientes.")
        return None

    now = _now_naive()
    frames = []
    for exp in expirations:
        exp_dt = pd.Timestamp(exp)              # naive
        T = max((exp_dt - now).days, 1) / 365.0
        q_est = infer_q_from_parity(tk, S, r, exp)
        q = 0.0 if q_est is None else float(q_est)

        ch = tk.option_chain(exp)
        calls, puts = ch.calls, ch.puts

        # ventana de moneyness un poco más ancha si T es corta
        dyn_range = mny_range
        if T*365 <= 45:
            dyn_range = (max(0.55, mny_range[0]), min(1.5, mny_range[1]))

        cc = _clean_quotes(calls, 'Call', S, T, r, q,
                           min_vol=min_vol, require_bidask=require_bidask,
                           min_time_value=min_time_value, mny_range=dyn_range)
        pp = _clean_quotes(puts,  'Put',  S, T, r, q,
                           min_vol=min_vol, require_bidask=require_bidask,
                           min_time_value=min_time_value, mny_range=dyn_range)

        if not cc.empty: cc['exp'] = exp; cc['T'] = T
        if not pp.empty: pp['exp'] = exp; pp['T'] = T
        frames += [cc, pp]
        print(f"[{exp}] T={int(T*365)}d  q={q:.3%}  calls={len(cc)}  puts={len(pp)}")

    # Filtrar frames no vacíos antes de concatenar
    valid_frames = [f for f in frames if f is not None and not f.empty]
    if not valid_frames:
        print("No hay datos de opciones válidos tras limpieza.")
        return None
    
    data = pd.concat(valid_frames, ignore_index=True)
    if data.empty:
        print("Datos de opciones vacíos tras limpieza.")
        return None

    fig, ax = plt.subplots(figsize=(12, 8))
    for exp, grp in data.groupby('exp'):
        ax.scatter(grp['moneyness'], grp['iv']*100, s=14, alpha=0.6, label=f"{exp} ({int(grp['T'].iloc[0]*365)}d)")
        if len(grp) >= 7:
            z = np.polyfit(grp['moneyness'], grp['iv']*100, 2)
            xs = np.linspace(grp['moneyness'].min(), grp['moneyness'].max(), 140)
            ax.plot(xs, np.poly1d(z)(xs), ls='--', lw=1.5)
    ax.axvline(1.0, color='r', ls='--', alpha=0.7, label='ATM')
    ax.set_title(f"Superficie de IV - {symbol} (expiraciones={len(expirations)})")
    ax.set_xlabel("K/S"); ax.set_ylabel("IV (%)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8, ncol=2)
    if save:
        out = Path.cwd() / "png"; out.mkdir(parents=True, exist_ok=True)
        plt.savefig(out / "real_vol_surface.png", dpi=300, bbox_inches='tight')
        print(f"✅ Superficie guardada en: {out/'real_vol_surface.png'}")
    plt.close(fig)
    return data

# ================================ DASHBOARD ===================================

def create_dashboard(macro, port_ret, corr_best_df, signals, regimes_df, save=True):
    _set_style()
    fig = plt.figure(figsize=(22,16))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25,
                          left=0.05, right=0.97, top=0.93, bottom=0.06)
    fig.suptitle('DASHBOARD MACRO (Solo Yahoo)', fontsize=20, fontweight='bold')

    # 1) Indicadores normalizados
    ax1 = fig.add_subplot(gs[0,0])
    for k in ['VIX','DXY','GOLD','OIL','SP500']:
        if k in macro and len(macro[k]):
            s = macro[k].dropna()
            base = s/s.iloc[0]*100
            ax1.plot(base.index, base.values, lw=2, label=k)
    ax1.set_title("Indicadores (Base=100)"); ax1.grid(True, alpha=0.3); ax1.legend(fontsize=9)

    # 2) Correlaciones (mejor lag)
    ax2 = fig.add_subplot(gs[0,1])
    if corr_best_df is not None and not corr_best_df.empty:
        top = corr_best_df.head(12).iloc[::-1]
        ax2.barh(top['indicador'], top['corr'], alpha=0.85)
        for i,(c,lg,pv) in enumerate(zip(top['corr'], top['lag'], top['p'])):
            ax2.text(c + (0.01 if c>0 else -0.01), i, f"{c:.2f} | lag {lg} | p {pv:.3f}",
                     va='center', ha='left' if c>0 else 'right', fontsize=9, fontweight='bold')
        ax2.axvline(0, color='k', lw=1)
    ax2.set_title("Correlaciones macro ↔ cartera (mejor lag)"); ax2.grid(True, axis='x', alpha=0.3)

    # 3) Señales
    ax3 = fig.add_subplot(gs[1,0])
    names = list(signals.keys()); vals = [signals[n]['value'] for n in names]
    cols = []
    for n in names:
        st = signals[n]['signal']
        cols.append('#e74c3c' if ('RISK_OFF' in st or 'INVERTED' in st) else ('#f39c12' if 'NEUTRAL' in st else '#27ae60'))
    bars = ax3.bar(names, vals, color=cols, edgecolor='black', lw=1, alpha=0.85)
    for b,n in zip(bars,names):
        ax3.text(b.get_x()+b.get_width()/2, b.get_height(), signals[n]['signal'], ha='center', va='bottom', fontsize=9, fontweight='bold', rotation=90)
    ax3.set_title("Señales de riesgo"); ax3.grid(True, axis='y', alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)

    # 4) Regímenes
    ax4 = fig.add_subplot(gs[1,1])
    if regimes_df is not None and regimes_df['regime'].notna().sum()>0:
        vol = port_ret.rolling(252).std()*np.sqrt(252)
        ret = port_ret.rolling(252).mean()*252
        tmp = pd.DataFrame({'vol':vol, 'ret':ret, 'lab':regimes_df['regime_label']}).dropna()
        for lab, d in tmp.groupby('lab'):
            ax4.scatter(d['vol'], d['ret'], alpha=0.7, s=25, label=lab)
        ax4.axhline(0, ls='--', color='k', alpha=0.5); ax4.axvline(0, ls='--', color='k', alpha=0.5)
        ax4.grid(True, alpha=0.3); ax4.legend(fontsize=9)
    ax4.set_title("Regímenes (walk-forward)")

    # 5) Portafolio (crecimiento)
    ax5 = fig.add_subplot(gs[2,:])
    eq = (1+port_ret.fillna(0)).cumprod()
    ax5.plot(eq.index, eq.values, lw=2)
    ax5.set_title("Crecimiento del portafolio (log-returns)"); ax5.grid(True, alpha=0.3)

    if save:
        out = Path.cwd()/ "png"; out.mkdir(exist_ok=True, parents=True)
        plt.savefig(out/"macro_dashboard_yahoo.png", dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Dashboard guardado en: {out/'macro_dashboard_yahoo.png'}")
    plt.close(fig)

# ================================ MAIN ========================================

def main():
    print("\n=== PIPELINE MACRO — SOLO YAHOO ===")
    end = _ensure_end_date(END_DATE)

    # Datos (solo Yahoo)
    macro = download_yahoo_assets(START_DATE, end)
    if not macro: raise RuntimeError("No se pudieron descargar datos macro (Yahoo).")

    # Cartera
    portfolio_prices = download_portfolio_prices(PORTFOLIO_TICKERS, START_DATE, end)
    if not portfolio_prices: raise RuntimeError("No se pudieron descargar precios de cartera.")
    rets_by_ticker, port_ret, eff_weights = build_portfolio_returns(portfolio_prices, PORTFOLIO_WEIGHTS)
    print(f"Obs cartera: {len(port_ret)} | Tickers efectivos: {list(eff_weights[eff_weights>0].index)}")

    # RF diario: ^IRX si existe
    rf_daily = macro.get('Y03M')

    # Regímenes
    regimes_df = walkforward_regimes(port_ret, rf_series=rf_daily, window=252, step=21, n_clusters=3)

    # Correlaciones
    _, macro_aligned = transform_macro_for_correlation(macro, port_ret.index)
    corr_best = best_lagged_correlation(port_ret, macro_aligned, max_lag=MAX_LAG)

    # Residuación + correlaciones sobre residuo (opcional)
    corr_best_resid = None
    if USE_RESIDUAL_CORRS:
        port_resid, _ = orthogonalize_portfolio(port_ret, macro)
        corr_best_resid = best_lagged_correlation(port_resid, macro_aligned, max_lag=MAX_LAG)

    # Señales
    signals = generate_signals_percentiles(macro)

    # Dashboard
    create_dashboard(
        macro, port_ret,
        corr_best_resid if (USE_RESIDUAL_CORRS and corr_best_resid is not None) else corr_best,
        signals, regimes_df, save=True
    )

    # Superficie de IV real (SPY) — más expiraciones y filtros laxos
    try:
        _ = real_vol_surface(
            "SPY",
            r_annual=None,          # usa ^IRX o 1% fallback
            n_expirations=10,
            min_T_days=10,
            mny_range=(0.85, 1.15),
            min_vol=10,
            require_bidask=True,
            min_time_value=0.10,
            save=True
        )
    except Exception as e:
        print(f"(IV) Aviso: {e}")

    # Export
    outdir = Path.cwd()/ "outputs"; outdir.mkdir(exist_ok=True, parents=True)
    port_ret.to_csv(outdir/"portfolio_returns.csv", header=['ret'])
    if corr_best is not None: corr_best.to_csv(outdir/"corr_best_lag.csv", index=False)
    if corr_best_resid is not None: corr_best_resid.to_csv(outdir/"corr_best_lag_resid.csv", index=False)

    # Resumen
    print("\n=== RESUMEN ===")
    print(f"Indicadores macro (Yahoo): {len(macro)}")
    print(f"Señales: {len(signals)} | Regímenes etiquetados: {regimes_df['regime'].notna().sum()} días")
    if corr_best is not None and not corr_best.empty:
        top = corr_best.head(5)
        print("Top correlaciones (cartera):")
        for _, r in top.iterrows():
            print(f"  {r['indicador']:<12s} corr={r['corr']:+.3f} lag={int(r['lag'])} p={r['p']:.3f}")
    if corr_best_resid is not None and not corr_best_resid.empty:
        top = corr_best_resid.head(5)
        print("Top correlaciones (residuo):")
        for _, r in top.iterrows():
            print(f"  {r['indicador']:<12s} corr={r['corr']:+.3f} lag={int(r['lag'])} p={r['p']:.3f}")
    print("Listo. Revisa ./png/macro_dashboard_yahoo.png y ./png/real_vol_surface.png")

    return {
        'macro_data': macro,
        'portfolio_prices': portfolio_prices,
        'portfolio_returns_by_ticker': rets_by_ticker,
        'portfolio': port_ret,
        'effective_weights': eff_weights,
        'regimes': regimes_df,
        'corr_best': corr_best,
        'corr_best_resid': corr_best_resid,
        'signals': signals
    }

# =============================== EJECUCIÓN ====================================

if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")
    except Exception as e:
        print(f"\nError inesperado: {e}")
        import traceback; traceback.print_exc()
