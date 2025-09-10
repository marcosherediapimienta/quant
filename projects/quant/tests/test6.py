"""
MACRO ECONOMIC ANALYSIS FOR PORTFOLIO MANAGEMENT
================================================
Yahoo Finance (activos) + FRED (indicadores macro) - Gratis

- Regime detection (KMeans) con Sharpe por exceso (rf = TB3MS)
- Señales: VIX, curva 10-2 (preferente) o 10-3M (fallback), USD (DXY vs MA50)
- Correlaciones macro ↔ cartera con transformaciones correctas
- Opción de correlaciones sobre residuo vs S&P 500
- Dashboard matplotlib: ./png/macro_dashboard_avanzado.png

Autor: Quant PM (refactor)
Versión: 2.3
"""

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------- Cache HTTP opcional (evita re-descargas/timeout FRED) ----------
try:
    import requests_cache
    requests_cache.install_cache('fred_cache', expire_after=24*3600)
except Exception:
    pass

# ================================ CONFIG =====================================

START_DATE = "2015-01-01"
END_DATE = None  # usa hoy si None
USE_RESIDUAL_CORRS = True  # correlaciones sobre residuo vs S&P 500

# Activos Yahoo (precios/índices)
YF_TICKERS = {
    'VIX': '^VIX',
    'DXY': 'DX-Y.NYB',   # si falla, se usa UUP como proxy normalizado
    'EUR_USD': 'EURUSD=X',
    'GBP_USD': 'GBPUSD=X',  # <-- añadido: se usa en el dashboard de divisas
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
    'Y10': '^TNX',        # 10y yield (x10)
    'Y30': '^TYX',        # 30y yield (x10)
    'Y03M': '^IRX'        # 13-week T-bill (aprox 3M), en %
}

# Macro FRED (niveles, en % para yields/ffr/tbills)
FRED_SERIES = {
    'Y02': 'DGS2',          # 2y
    'Y10_FRED': 'DGS10',    # 10y
    'FEDFUNDS': 'FEDFUNDS', # Fed Funds
    'CPI': 'CPIAUCSL',      # CPI nivel
    'PPI': 'PPIACO',        # PPI nivel
    'TED_SPREAD': 'TEDRATE',
    'TBILL_3M': 'TB3MS'     # RF mensual aprox
}

# Cartera (los tickers que fallen se excluyen y se renormalizan pesos)
PORTFOLIO_TICKERS = ["0P0001CLDK.F", "AMZN", "GLD", "BTC-EUR"]
PORTFOLIO_WEIGHTS = np.array([0.8, 0.05, 0.10, 0.05])  # se renormalizan

# ================================ HELPERS ====================================

def _today_str():
    return datetime.today().strftime('%Y-%m-%d')

def _ensure_end_date(end_date):
    return _today_str() if end_date is None else end_date

def _safe_series(s):
    return pd.Series(s).dropna() if s is not None else pd.Series(dtype=float)

def _set_style():
    """Usa seaborn-v0_8 si existe; cae a default si no."""
    try:
        plt.style.use('seaborn-v0_8')
    except Exception:
        plt.style.use('default')

# ======================== DESCARGA YAHOO + FALLBACK ===========================

def download_yahoo_series(ticker_list, start=START_DATE, end=None):
    """Descarga en batch desde Yahoo y devuelve dict[ticker]=Series Close."""
    end = _ensure_end_date(end)
    out = {}
    try:
        df = yf.download(
            ticker_list, start=start, end=end,
            progress=False, auto_adjust=True
        )
        close = df['Close'] if isinstance(df.columns, pd.MultiIndex) and 'Close' in df else df
        for t in ticker_list:
            s = _safe_series(close.get(t))
            if len(s):
                out[t] = s
            else:
                print(f"[Yahoo] sin datos: {t}")
    except Exception as e:
        print(f"[Yahoo] Error batch: {e}")
    return out

def download_yahoo_assets(start=START_DATE, end=None):
    """Descarga activos/índices definidos en YF_TICKERS y aplica fallbacks."""
    end = _ensure_end_date(end)
    print("Descargando activos Yahoo...")
    out = {}
    batch = download_yahoo_series(list(YF_TICKERS.values()), start, end)
    for name, t in YF_TICKERS.items():
        if t in batch and len(batch[t]):
            out[name] = batch[t]

    # Fallback DXY → UUP (proxy normalizado)
    if 'DXY' not in out or not len(out['DXY']):
        print("DXY no disponible. Usando UUP (ETF) como proxy normalizado.")
        uup = download_yahoo_series(['UUP'], start, end).get('UUP', pd.Series(dtype=float))
        if len(uup):
            out['DXY'] = uup / uup.iloc[0] * 100.0

    # Fallback 2Y yield por Yahoo si no hay FRED
    if 'Y02' not in out:
        for y2 in ["^UST2Y", "^US2Y"]:
            cand = download_yahoo_series([y2], start, end).get(y2)
            if cand is not None and len(cand):
                out['Y02'] = cand  # misma clave que FRED para reutilizar lógica
                print(f"Usando {y2} como 2Y (fallback).")
                break
    return out

# ================================ FRED ========================================

def download_fred_series(series_map, start=START_DATE, end=None):
    """Descarga desde FRED usando pandas-datareader; devuelve dict[name]=Series."""
    end = _ensure_end_date(end)
    out = {}
    try:
        from pandas_datareader import data as pdr
        fred = pdr.DataReader(list(series_map.values()), 'fred', start=start, end=end).ffill()
        for name, code in series_map.items():
            s = _safe_series(fred.get(code))
            if len(s): out[name] = s
    except Exception as e:
        print(f"[FRED] No disponible: {e}")
    return out

# ======================== PORTFOLIO: PRECIOS → RETORNOS =======================

def download_portfolio_prices(tickers, start=START_DATE, end=None):
    end = _ensure_end_date(end)
    print("Descargando precios de cartera...")
    out = download_yahoo_series(tickers, start, end)
    for t in tickers:
        if t not in out:
            print(f"  - {t}: sin datos; se excluye")
    return out

def build_portfolio_returns(portfolio_prices: dict, weights: np.ndarray):
    """Alinea por inner join y calcula retornos + retorno de cartera."""
    if not portfolio_prices:
        raise ValueError("No hay precios de cartera.")
    prices = pd.concat(portfolio_prices, axis=1, join='inner').dropna(how='all')
    prices.columns = list(portfolio_prices.keys())
    rets = prices.pct_change().dropna(how='all')
    # Renormaliza pesos a los tickers presentes
    cols = rets.columns
    w = pd.Series(0.0, index=cols)
    for i, c in enumerate(PORTFOLIO_TICKERS):
        if c in cols:
            w[c] = float(weights[i])
    if w.sum() == 0:
        raise ValueError("Los pesos efectivos suman 0; revisa tickers.")
    w = w / w.sum()
    port = (rets * w).sum(axis=1, min_count=1).dropna()
    return rets, port, w

# ======================== TRANSFORMACIONES MACRO ==============================

def transform_macro_for_correlation(macro: dict, portfolio_idx: pd.DatetimeIndex):
    """
    Devuelve dict transformado y un DataFrame alineado a índice de la cartera (diario):
    - Yields / FedFunds / TBills → diferencias (pp). Para ^TNX/^TYX se ajusta /10 antes de diferenciar.
    - CPI / PPI → YoY (variación 12m). Resample mensual (last).
    - Resto (precios/índices/FX/commodities) → pct_change()
    """
    X = {}
    for k, s in macro.items():
        s = s.dropna()
        if not len(s):
            continue
        if k in ('Y02', 'Y10_FRED', 'FEDFUNDS', 'TBILL_3M', 'Y03M'):
            X[k] = s.diff()
        elif k in ('Y10', 'Y30'):           # Yahoo yields x10
            X[k] = (s / 10.0).diff()
        elif k in ('CPI', 'PPI'):
            m = s.resample('M').last()
            X[k] = m.pct_change(12)         # YoY
        else:
            X[k] = s.pct_change()

    # Alinea a frecuencia diaria de la cartera con ffill
    df = pd.DataFrame(index=portfolio_idx)
    for k, s in X.items():
        df[k] = s.asfreq('B').ffill().reindex(df.index).ffill()
    return X, df

# =============================== CORRELACIONES ================================

def analyze_macro_correlations(portfolio_ret: pd.Series, macro: dict):
    """Correlaciones contemporáneas (puedes extender con lags si quieres)."""
    print("Analizando correlaciones macro-cartera...")
    _, df = transform_macro_for_correlation(macro, portfolio_ret.index)
    data = pd.concat([portfolio_ret.rename('portfolio'), df], axis=1).dropna()
    corrs = {k: data['portfolio'].corr(data[k]) for k in df.columns}
    # Top absoluto
    top = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    for name, c in top:
        print(f"  {name}: {c:.3f}")
    return corrs, data

def residualize_vs_equity(port_ret: pd.Series, spx_ret: pd.Series) -> pd.Series:
    """Quita la beta al S&P 500 para ver 'macro puro' en correlaciones."""
    y = port_ret.dropna()
    x = spx_ret.reindex(y.index).dropna()
    idx = y.index.intersection(x.index)
    if len(idx) < 50:
        return port_ret.rename('portfolio_resid')
    X = x.loc[idx].values
    Y = y.loc[idx].values
    beta = (X.T @ Y) / (X.T @ X)
    resid = pd.Series(Y - X * beta, index=idx, name='portfolio_resid')
    return resid

# =============================== SEÑALES RIESGO ===============================

def generate_risk_signals(macro: dict):
    """VIX, Curva 10-2 (preferente) o 10-3M (fallback), DXY vs MA50."""
    print("Generando señales de riesgo...")
    signals = {}

    # VIX
    vix = macro.get('VIX')
    if vix is not None and len(vix):
        v = float(vix.iloc[-1])
        if v > 30: lvl, sig = 'HIGH', 'RISK_OFF'
        elif v > 20: lvl, sig = 'MEDIUM', 'NEUTRAL'
        else: lvl, sig = 'LOW', 'RISK_ON'
        signals['VIX'] = {'level': lvl, 'value': v, 'signal': sig}

    # Curva: 10-2 si hay; si no, 10-3M (Y03M = ^IRX)
    ten = None; two = None; three_m = None

    if 'Y10_FRED' in macro and len(macro['Y10_FRED']):
        ten = float(macro['Y10_FRED'].iloc[-1])
    elif 'Y10' in macro and len(macro['Y10']):
        ten = float(macro['Y10'].iloc[-1]) / 10.0  # ^TNX escala x10

    if 'Y02' in macro and len(macro['Y02']):
        two = float(macro['Y02'].iloc[-1])

    if 'Y03M' in macro and len(macro['Y03M']):
        three_m = float(macro['Y03M'].iloc[-1])

    def slope_signal(spread, label):
        if spread < 0:
            signals[label] = {'level': 'INVERTED', 'value': spread, 'signal': 'RECESSION_RISK'}
        elif spread < 0.5:
            signals[label] = {'level': 'FLAT', 'value': spread, 'signal': 'CAUTION'}
        else:
            signals[label] = {'level': 'NORMAL', 'value': spread, 'signal': 'HEALTHY'}

    if (ten is not None) and (two is not None):
        slope_signal(ten - two, 'YIELD_CURVE_10_2')
    elif (ten is not None) and (three_m is not None):
        slope_signal(ten - three_m, 'YIELD_CURVE_10_3M')

    # USD (DXY vs MA50)
    dxy = macro.get('DXY')
    if dxy is not None and len(dxy) >= 50:
        d = float(dxy.iloc[-1]); dma = float(dxy.rolling(50).mean().iloc[-1])
        if d > dma * 1.05:
            signals['DXY'] = {'level': 'STRONG', 'value': d, 'signal': 'USD_STRENGTH'}
        elif d < dma * 0.95:
            signals['DXY'] = {'level': 'WEAK', 'value': d, 'signal': 'USD_WEAKNESS'}
        else:
            signals['DXY'] = {'level': 'NEUTRAL', 'value': d, 'signal': 'USD_NEUTRAL'}

    return signals

# =============================== REGÍMENES ====================================

def detect_market_regimes(returns_data: pd.Series, window=252, rf_series: pd.Series=None):
    """Regímenes con KMeans (n_init alto) y Sharpe por exceso."""
    print("Detectando regímenes de mercado...")
    idx = returns_data.index
    rf_daily = None
    if rf_series is not None and len(rf_series):
        # rf en % anual → diario aprox
        rf_daily = rf_series.asfreq('B').ffill().reindex(idx).ffill() / 100.0 / 252.0
    excess = returns_data - (rf_daily if rf_daily is not None else 0.0)

    reg = pd.DataFrame(index=idx)
    reg['volatility'] = excess.rolling(window).std() * np.sqrt(252)
    reg['return']     = excess.rolling(window).mean() * 252
    reg['sharpe']     = reg['return'] / reg['volatility'].replace(0, np.nan)

    cumulative = (1 + returns_data.fillna(0)).cumprod()
    rolling_max = cumulative.rolling(window).max()
    drawdown = (cumulative / rolling_max) - 1
    reg['max_drawdown'] = drawdown.rolling(window).min()
    reg = reg.dropna()

    if len(reg) < 50:
        print("  Aviso: datos insuficientes para regímenes")
        return None

    scaler = StandardScaler()
    scaled = scaler.fit_transform(reg)

    kmeans = KMeans(n_clusters=3, n_init=25, random_state=42)
    regimes = kmeans.fit_predict(scaled)

    regime_df = reg.copy()
    regime_df['regime'] = regimes

    # Etiquetas legibles + únicas
    stats_ = regime_df.groupby('regime')[['volatility','return']].mean()
    base = {}
    for r in stats_.index:
        vol, ret = stats_.loc[r, 'volatility'], stats_.loc[r, 'return']
        if vol > stats_['volatility'].quantile(0.7):
            base[r] = 'High Vol Bull' if ret > 0 else 'High Vol Bear'
        else:
            base[r] = 'Low Vol Bull' if ret > 0 else 'Low Vol Bear'
    counts = {}
    labels = {}
    for r in sorted(base.keys()):
        name = base[r]
        counts[name] = counts.get(name, 0) + 1
        suffix = f" #{counts[name]}" if counts[name] > 1 else ""
        labels[r] = f"{name}{suffix}"
    regime_df['regime_label'] = regime_df['regime'].map(labels)

    print(f"  Regímenes detectados: {len(set(labels.values()))}")
    return regime_df

# =============================== DASHBOARD ====================================

def create_macro_dashboard(macro, correlations, signals, regime_df, portfolio_ret=None, save=True):
    print("Creando dashboard macro avanzado...")
    _set_style()

    fig = plt.figure(figsize=(24, 20))
    fig.suptitle('DASHBOARD MACROECONÓMICO PROFESIONAL - ANÁLISIS COMPLETO', 
                 fontsize=24, fontweight='bold', color='#1f4e79')
    
    # Grid más grande para más gráficos
    gs = fig.add_gridspec(5, 4, hspace=0.35, wspace=0.25,
                          left=0.03, right=0.97, top=0.93, bottom=0.03)

    # === GRÁFICO 1: INDICADORES MACRO NORMALIZADOS ===
    ax1 = fig.add_subplot(gs[0, :2])
    indicators_to_plot = ['VIX', 'DXY', 'GOLD', 'OIL', 'SP500']
    colors = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71', '#9b59b6']
    
    for i, ind in enumerate(indicators_to_plot):
        if ind in macro and len(macro[ind]):
            s = macro[ind].dropna()
            if len(s) > 0:
                base = s / s.iloc[0] * 100.0
                ax1.plot(base.index, base.values, label=ind, linewidth=2.5, 
                        color=colors[i], alpha=0.8)
    
    ax1.set_title('📊 INDICADORES MACRO CLAVE (Normalizados Base 100)', 
                  fontsize=14, fontweight='bold', color='#1f4e79')
    ax1.set_ylabel('Valor Normalizado', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=100, color='black', linestyle='--', alpha=0.5)

    # === GRÁFICO 2: HEATMAP DE CORRELACIONES ===
    ax2 = fig.add_subplot(gs[0, 2:])
    if correlations:
        corr_data = pd.DataFrame(list(correlations.items()), 
                                 columns=['Indicador', 'Correlación'])
        corr_data = corr_data.sort_values('Correlación', key=abs, ascending=True)
        y_pos = np.arange(len(corr_data))
        colors_heatmap = ['#e74c3c' if x < 0 else '#27ae60' for x in corr_data['Correlación']]
        bars = ax2.barh(y_pos, corr_data['Correlación'], color=colors_heatmap, alpha=0.7)
        for bar, value in zip(bars, corr_data['Correlación']):
            ax2.text(value + (0.01 if value > 0 else -0.01), bar.get_y() + bar.get_height()/2,
                     f'{value:.3f}', ha='left' if value > 0 else 'right', 
                     va='center', fontweight='bold', fontsize=9)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(corr_data['Indicador'], fontsize=9)
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_title('🔥 CORRELACIONES MACRO-CARTERA', 
                  fontsize=14, fontweight='bold', color='#1f4e79')
    ax2.set_xlabel('Coeficiente de Correlación', fontsize=12)

    # === GRÁFICO 3: REGÍMENES DE MERCADO ===
    ax3 = fig.add_subplot(gs[1, :2])
    if regime_df is not None and not regime_df.empty:
        regime_colors = ['#e74c3c', '#f39c12', '#27ae60', '#3498db', '#9b59b6']
        for i, r in enumerate(sorted(regime_df['regime'].unique())):
            d = regime_df[regime_df['regime'] == r]
            ax3.scatter(d['volatility'], d['return'], s=60, alpha=0.7, 
                        color=regime_colors[i % len(regime_colors)],
                        label=d['regime_label'].iloc[0], edgecolors='black', linewidth=0.5)
        ax3.set_xlabel('Volatilidad Anualizada (%)', fontsize=12)
        ax3.set_ylabel('Retorno Anualizado (%)', fontsize=12)
        ax3.set_title('🎯 REGÍMENES DE MERCADO DETECTADOS', 
                      fontsize=14, fontweight='bold', color='#1f4e79')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)

    # === GRÁFICO 4: SEÑALES DE RIESGO ===
    ax4 = fig.add_subplot(gs[1, 2:])
    if signals:
        names = list(signals.keys())
        vals = [signals[n]['value'] for n in names]
        cols, labels = [], []
        for n in names:
            st = signals[n]['signal']
            if ('RISK' in st) or ('RECESSION' in st):
                cols.append('#e74c3c'); labels.append('🔴 ALERTA')
            elif ('CAUTION' in st):
                cols.append('#f39c12'); labels.append('🟡 PRECAUCIÓN')
            else:
                cols.append('#27ae60'); labels.append('🟢 OK')
        bars = ax4.bar(names, vals, color=cols, alpha=0.8, edgecolor='black', linewidth=1)
        for bar, value, label in zip(bars, vals, labels):
            h = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2, h + (abs(h)*0.01 if h!=0 else 0.01),
                     f'{value:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
            ax4.text(bar.get_x() + bar.get_width()/2, h/2, label,
                     ha='center', va='center', fontweight='bold', fontsize=8, color='white')
    ax4.set_title('⚠️ SEÑALES DE RIESGO EN TIEMPO REAL', 
                  fontsize=14, fontweight='bold', color='#1f4e79')
    ax4.set_ylabel('Valor', fontsize=12)
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')

    # === GRÁFICO 5: EVOLUCIÓN TEMPORAL DETALLADA ===
    ax5 = fig.add_subplot(gs[2, :])
    key_indicators = ['VIX', 'DXY', 'GOLD', 'OIL', 'SP500']
    colors_temp = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71', '#9b59b6']
    for i, ind in enumerate(key_indicators):
        if ind in macro and len(macro[ind]):
            s = macro[ind].dropna()
            if len(s) > 0:
                ax5.plot(s.index, s.values, label=ind, linewidth=2.5, 
                         color=colors_temp[i], alpha=0.8)
    ax5.set_title('📈 EVOLUCIÓN TEMPORAL DE INDICADORES CLAVE', 
                  fontsize=14, fontweight='bold', color='#1f4e79')
    ax5.set_ylabel('Valor', fontsize=12)
    ax5.legend(loc='upper left', fontsize=10)
    ax5.grid(True, alpha=0.3)

    # === GRÁFICO 6: ANÁLISIS DE VOLATILIDAD ===
    ax6 = fig.add_subplot(gs[3, :2])
    if 'VIX' in macro and len(macro['VIX']):
        vix_data = macro['VIX'].dropna()
        ax6.plot(vix_data.index, vix_data.values, color='#e74c3c', linewidth=2, alpha=0.8)
        ax6.axhline(y=20, color='orange', linestyle='--', alpha=0.7, label='Umbral Medio (20)')
        ax6.axhline(y=30, color='red', linestyle='--', alpha=0.7, label='Umbral Alto (30)')
        ax6.fill_between(vix_data.index, 0, vix_data.values, where=(vix_data.values > 30),
                         color='red', alpha=0.2, label='Zona de Pánico')
        ax6.fill_between(vix_data.index, 0, vix_data.values,
                         where=((vix_data.values > 20) & (vix_data.values <= 30)),
                         color='orange', alpha=0.2, label='Zona de Precaución')
        ax6.fill_between(vix_data.index, 0, vix_data.values,
                         where=(vix_data.values <= 20), color='green', alpha=0.2, label='Zona Tranquila')
    ax6.set_title('📊 ANÁLISIS DE VOLATILIDAD (VIX)', fontsize=14, fontweight='bold', color='#1f4e79')
    ax6.set_ylabel('VIX', fontsize=12)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)

    # === GRÁFICO 7: ANÁLISIS DE DIVISAS ===
    ax7 = fig.add_subplot(gs[3, 2:])
    currency_indicators = ['DXY', 'EUR_USD', 'GBP_USD']
    colors_curr = ['#3498db', '#e74c3c', '#f39c12']
    for i, curr in enumerate(currency_indicators):
        if curr in macro and len(macro[curr]):
            s = macro[curr].dropna()
            if len(s) > 0:
                normalized = s / s.iloc[0] * 100
                ax7.plot(normalized.index, normalized.values, label=curr, linewidth=2,
                         color=colors_curr[i], alpha=0.8)
    ax7.set_title('💱 ANÁLISIS DE DIVISAS (Normalizado)', fontsize=14, fontweight='bold', color='#1f4e79')
    ax7.set_ylabel('Valor Normalizado', fontsize=12)
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.3)
    ax7.axhline(y=100, color='black', linestyle='--', alpha=0.5)

    # === GRÁFICO 8: RESUMEN EJECUTIVO ===
    ax8 = fig.add_subplot(gs[4, :])
    ax8.axis('off')

    # Rango de fechas global robusto
    try:
        idx_list = [s.dropna().index for s in macro.values() if hasattr(s, "index") and len(s.dropna())]
        period_start = min([idx.min() for idx in idx_list]).strftime('%Y-%m-%d') if idx_list else 'N/A'
        period_end   = max([idx.max() for idx in idx_list]).strftime('%Y-%m-%d') if idx_list else 'N/A'
    except Exception:
        period_start = period_end = 'N/A'
    
    summary_text = "🎯 RESUMEN EJECUTIVO DEL ANÁLISIS MACROECONÓMICO\n" + "="*70 + "\n\n"
    if signals:
        summary_text += "📊 SEÑALES DE RIESGO:\n"
        for n, sig in signals.items():
            lvl = sig['level']; val = sig['value']; st = sig['signal']
            if 'RISK' in st or 'RECESSION' in st:
                summary_text += f"🔴 ALERTA: {n} - {lvl} ({val:.2f}) → {st}\n"
            elif 'CAUTION' in st:
                summary_text += f"🟡 PRECAUCIÓN: {n} - {lvl} ({val:.2f}) → {st}\n"
            else:
                summary_text += f"🟢 OK: {n} - {lvl} ({val:.2f}) → {st}\n"
        summary_text += "\n"
    if correlations:
        summary_text += "📈 CORRELACIONES DESTACADAS:\n"
        top_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
        for ind, c in top_corr:
            emoji = "📈" if c > 0 else "📉"
            strength = "Fuerte" if abs(c) > 0.5 else "Moderada" if abs(c) > 0.3 else "Débil"
            summary_text += f"   {emoji} {ind}: {c:.3f} ({strength})\n"
        summary_text += "\n"
    if regime_df is not None and not regime_df.empty:
        summary_text += "🎯 REGÍMENES DETECTADOS:\n"
        regime_counts = regime_df['regime_label'].value_counts()
        for regime, count in regime_counts.items():
            percentage = (count / len(regime_df)) * 100
            summary_text += f"   • {regime}: {count} períodos ({percentage:.1f}%)\n"
    summary_text += f"\n📅 Período de análisis: {period_start} a {period_end}"
    summary_text += f"\n📊 Total de indicadores analizados: {len(macro)}"

    ax8.text(0.02, 0.95, summary_text, transform=ax8.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.8", facecolor='#f8f9fa', 
                       edgecolor='#1f4e79', linewidth=2, alpha=0.9))

    if save:
        try:
            base_dir = Path(__file__).resolve().parent
        except NameError:
            base_dir = Path.cwd()
        png_dir = base_dir / "png"
        png_dir.mkdir(parents=True, exist_ok=True)
        outpath = png_dir / "macro_dashboard_avanzado.png"
        plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Dashboard avanzado guardado en: {outpath}")
    
    return fig

def create_individual_charts(macro, correlations, signals, regime_df, save=True):
    """Crea gráficos individuales más detallados para análisis específicos."""
    print("Creando gráficos individuales detallados...")
    _set_style()
    
    # === GRÁFICO 1: ANÁLISIS DETALLADO DE CORRELACIONES ===
    if correlations:
        fig1, ax1 = plt.subplots(figsize=(14, 10))
        corr_data = pd.DataFrame(list(correlations.items()), 
                                 columns=['Indicador', 'Correlación'])
        corr_data = corr_data.sort_values('Correlación', key=abs, ascending=True)
        y_pos = np.arange(len(corr_data))
        colors = ['#e74c3c' if x < 0 else '#27ae60' for x in corr_data['Correlación']]
        bars = ax1.barh(y_pos, corr_data['Correlación'], color=colors, alpha=0.8)
        for bar, value in zip(bars, corr_data['Correlación']):
            ax1.text(value + (0.01 if value > 0 else -0.01), bar.get_y() + bar.get_height()/2,
                     f'{value:.3f}', ha='left' if value > 0 else 'right',
                     va='center', fontweight='bold', fontsize=11)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(corr_data['Indicador'], fontsize=12)
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.set_xlabel('Coeficiente de Correlación', fontsize=14)
        ax1.set_title('🔥 CORRELACIONES MACRO-CARTERA DETALLADAS', 
                      fontsize=16, fontweight='bold', color='#1f4e79')
        if save:
            try:
                base_dir = Path(__file__).resolve().parent
            except NameError:
                base_dir = Path.cwd()
            png_dir = base_dir / "png"
            png_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(png_dir / "correlaciones_detalladas.png", dpi=300, bbox_inches='tight')
            print(f"📊 Gráfico de correlaciones guardado en: {png_dir / 'correlaciones_detalladas.png'}")
        plt.close()

    # === GRÁFICO 2: ANÁLISIS DE REGÍMENES DETALLADO ===
    if regime_df is not None and not regime_df.empty:
        fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(16, 8))
        regime_colors = ['#e74c3c', '#f39c12', '#27ae60', '#3498db', '#9b59b6']
        for i, r in enumerate(sorted(regime_df['regime'].unique())):
            d = regime_df[regime_df['regime'] == r]
            ax2a.scatter(d['volatility'], d['return'], s=80, alpha=0.7, 
                         color=regime_colors[i % len(regime_colors)],
                         label=d['regime_label'].iloc[0], edgecolors='black', linewidth=1)
        ax2a.set_xlabel('Volatilidad Anualizada (%)', fontsize=12)
        ax2a.set_ylabel('Retorno Anualizado (%)', fontsize=12)
        ax2a.set_title('🎯 REGÍMENES DE MERCADO DETECTADOS', 
                       fontsize=14, fontweight='bold', color='#1f4e79')
        ax2a.legend(fontsize=11)
        ax2a.grid(True, alpha=0.3)
        ax2a.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2a.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        regime_counts = regime_df['regime_label'].value_counts()
        colors_pie = regime_colors[:len(regime_counts)]
        ax2b.pie(regime_counts.values, labels=regime_counts.index, 
                 autopct='%1.1f%%', colors=colors_pie, startangle=90)
        ax2b.set_title('📊 DISTRIBUCIÓN DE REGÍMENES', fontsize=14, fontweight='bold', color='#1f4e79')
        if save:
            try:
                base_dir = Path(__file__).resolve().parent
            except NameError:
                base_dir = Path.cwd()
            png_dir = base_dir / "png"
            png_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(png_dir / "regimenes_detallados.png", dpi=300, bbox_inches='tight')
            print(f"📊 Gráfico de regímenes guardado en: {png_dir / 'regimenes_detallados.png'}")
        plt.close()

    # === GRÁFICO 3: ANÁLISIS DE VOLATILIDAD DETALLADO ===
    if 'VIX' in macro and len(macro['VIX']):
        fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(16, 12))
        vix_data = macro['VIX'].dropna()
        ax3a.plot(vix_data.index, vix_data.values, color='#e74c3c', linewidth=2, alpha=0.8)
        ax3a.axhline(y=20, color='orange', linestyle='--', alpha=0.7, label='Umbral Medio (20)')
        ax3a.axhline(y=30, color='red', linestyle='--', alpha=0.7, label='Umbral Alto (30)')
        ax3a.fill_between(vix_data.index, 0, vix_data.values, where=(vix_data.values > 30),
                          color='red', alpha=0.2, label='Zona de Pánico')
        ax3a.fill_between(vix_data.index, 0, vix_data.values,
                          where=((vix_data.values > 20) & (vix_data.values <= 30)),
                          color='orange', alpha=0.2, label='Zona de Precaución')
        ax3a.fill_between(vix_data.index, 0, vix_data.values,
                          where=(vix_data.values <= 20), color='green', alpha=0.2, label='Zona Tranquila')
        ax3a.set_title('📊 ANÁLISIS DE VOLATILIDAD (VIX) - ZONAS DE RIESGO', fontsize=14, fontweight='bold', color='#1f4e79')
        ax3a.set_ylabel('VIX', fontsize=12)
        ax3a.legend(fontsize=10)
        ax3a.grid(True, alpha=0.3)
        ax3b.hist(vix_data.values, bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
        ax3b.axvline(x=20, color='orange', linestyle='--', alpha=0.7, label='Umbral Medio (20)')
        ax3b.axvline(x=30, color='red', linestyle='--', alpha=0.7, label='Umbral Alto (30)')
        ax3b.set_title('📈 DISTRIBUCIÓN DE VALORES VIX', fontsize=14, fontweight='bold', color='#1f4e79')
        ax3b.set_xlabel('VIX', fontsize=12); ax3b.set_ylabel('Frecuencia', fontsize=12)
        ax3b.legend(fontsize=10); ax3b.grid(True, alpha=0.3)
        if save:
            try:
                base_dir = Path(__file__).resolve().parent
            except NameError:
                base_dir = Path.cwd()
            png_dir = base_dir / "png"
            png_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(png_dir / "vix_detallado.png", dpi=300, bbox_inches='tight')
            print(f"📊 Gráfico de VIX guardado en: {png_dir / 'vix_detallado.png'}")
        plt.close()

    # === GRÁFICO 4: COMPARACIÓN DE INDICADORES MACRO ===
    fig4, ax4 = plt.subplots(figsize=(16, 10))
    indicators = ['VIX', 'DXY', 'GOLD', 'OIL', 'SP500', 'NASDAQ']
    colors = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71', '#9b59b6', '#8e44ad']
    for i, ind in enumerate(indicators):
        if ind in macro and len(macro[ind]):
            s = macro[ind].dropna()
            if len(s) > 0:
                normalized = s / s.iloc[0] * 100
                ax4.plot(normalized.index, normalized.values, label=ind, linewidth=2.5,
                         color=colors[i], alpha=0.8)
    ax4.set_title('📈 COMPARACIÓN DE INDICADORES MACRO (Normalizados Base 100)', 
                  fontsize=16, fontweight='bold', color='#1f4e79')
    ax4.set_ylabel('Valor Normalizado', fontsize=14)
    ax4.legend(loc='upper left', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=100, color='black', linestyle='--', alpha=0.5)
    if save:
        try:
            base_dir = Path(__file__).resolve().parent
        except NameError:
            base_dir = Path.cwd()
        png_dir = base_dir / "png"
        png_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(png_dir / "indicadores_comparacion.png", dpi=300, bbox_inches='tight')
        print(f"📊 Gráfico de comparación guardado en: {png_dir / 'indicadores_comparacion.png'}")
    plt.close()

def create_yield_curve_analysis(macro, save=True):
    """Crea análisis detallado de la curva de rendimientos (yield curve)."""
    print("Creando análisis de curva de rendimientos...")
    _set_style()
    
    # Obtener datos de yields
    yields_data = {}
    yield_mapping = {
        'Y03M': ('3M', '^IRX'),
        'Y02': ('2Y', 'DGS2'), 
        'Y10': ('10Y', '^TNX'),
        'Y10_FRED': ('10Y', 'DGS10'),
        'Y30': ('30Y', '^TYX')
    }
    for key, (name, _) in yield_mapping.items():
        if key in macro and len(macro[key]):
            yields_data[name] = (macro[key] / 10.0) if key in ['Y10', 'Y30'] else macro[key]
    if len(yields_data) < 2:
        print("⚠️ Datos insuficientes para análisis de yield curve")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('📈 ANÁLISIS DETALLADO DE CURVA DE RENDIMIENTOS (YIELD CURVE)', 
                 fontsize=20, fontweight='bold', color='#1f4e79')
    
    # Curva actual
    maturity_order = ['3M', '2Y', '10Y', '30Y']
    available_yields = {k: v for k, v in yields_data.items() if k in maturity_order}
    if len(available_yields) >= 2:
        current_yields = {name: float(data.dropna().iloc[-1]) for name, data in available_yields.items() if len(data.dropna())}
        maturities = list(current_yields.keys()); rates = list(current_yields.values())
        if len(rates) >= 2 and '10Y' in current_yields and '2Y' in current_yields:
            spread_10_2 = current_yields['10Y'] - current_yields['2Y']
            if spread_10_2 < 0: color, curve_type = '#e74c3c', 'INVERTIDA'
            elif spread_10_2 < 0.5: color, curve_type = '#f39c12', 'PLANA'
            else: color, curve_type = '#27ae60', 'NORMAL'
        else:
            color, curve_type = '#3498db', 'NORMAL'
        ax1.plot(maturities, rates, 'o-', linewidth=3, markersize=8, color=color, alpha=0.8, label=f'Curva {curve_type}')
        for i, (maturity, rate) in enumerate(zip(maturities, rates)):
            ax1.annotate(f'{rate:.2f}%', (i, rate), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold', fontsize=10)
        ax1.set_title(f'🎯 CURVA DE RENDIMIENTOS ACTUAL - {curve_type}', fontsize=14, fontweight='bold', color='#1f4e79')
        ax1.set_xlabel('Vencimiento', fontsize=12); ax1.set_ylabel('Rendimiento (%)', fontsize=12)
        ax1.grid(True, alpha=0.3); ax1.legend(fontsize=12)
        if '10Y' in current_yields and '2Y' in current_yields:
            spread = current_yields['10Y'] - current_yields['2Y']
            ax1.text(0.02, 0.98, f'Spread 10Y-2Y: {spread:.2f}%', transform=ax1.transAxes, fontsize=12, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7), verticalalignment='top')
    
    # Evolución temporal
    colors_yields = ['#e74c3c', '#f39c12', '#27ae60', '#3498db', '#9b59b6']
    for i, (name, data) in enumerate(yields_data.items()):
        if len(data.dropna()) > 0:
            ax2.plot(data.index, data.values, label=name, linewidth=2, color=colors_yields[i % len(colors_yields)], alpha=0.8)
    ax2.set_title('📊 EVOLUCIÓN TEMPORAL DE RENDIMIENTOS', fontsize=14, fontweight='bold', color='#1f4e79')
    ax2.set_ylabel('Rendimiento (%)', fontsize=12); ax2.legend(fontsize=10); ax2.grid(True, alpha=0.3)
    
    # Spread 10Y-2Y histórico
    if 'Y10' in macro and 'Y02' in macro and len(macro['Y10']) and len(macro['Y02']):
        y10_data = macro['Y10'] / 10.0
        y02_data = macro['Y02']
        common_index = y10_data.index.intersection(y02_data.index)
        if len(common_index):
            spread_data = (y10_data.loc[common_index] - y02_data.loc[common_index]).dropna()
            ax3.plot(spread_data.index, spread_data.values, linewidth=2, color='#8e44ad', alpha=0.8)
            ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Curva Invertida')
            ax3.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Curva Plana')
            ax3.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Curva Normal')
            ax3.fill_between(spread_data.index, spread_data.values, 0, where=(spread_data.values < 0), color='red', alpha=0.2)
            ax3.set_title('📈 SPREAD 10Y-2Y HISTÓRICO', fontsize=14, fontweight='bold', color='#1f4e79')
            ax3.set_ylabel('Spread (%)', fontsize=12); ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

    # Matriz de correlaciones entre yields
    if len(yields_data) >= 3:
        yields_df = pd.DataFrame(yields_data).dropna()
        if len(yields_df):
            corr_matrix = yields_df.corr()
            im = ax4.imshow(corr_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
            for i in range(len(corr_matrix)):
                for j in range(len(corr_matrix.columns)):
                    ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha="center", va="center", color="black", fontweight='bold')
            ax4.set_xticks(range(len(corr_matrix.columns))); ax4.set_yticks(range(len(corr_matrix.index)))
            ax4.set_xticklabels(corr_matrix.columns); ax4.set_yticklabels(corr_matrix.index)
            ax4.set_title('🔗 MATRIZ DE CORRELACIONES ENTRE YIELDS', fontsize=14, fontweight='bold', color='#1f4e79')
            cbar = plt.colorbar(im, ax=ax4); cbar.set_label('Correlación', fontsize=12)
    
    plt.tight_layout()
    if save:
        try:
            base_dir = Path(__file__).resolve().parent
        except NameError:
            base_dir = Path.cwd()
        png_dir = base_dir / "png"
        png_dir.mkdir(parents=True, exist_ok=True)
        outpath = png_dir / "yield_curve_analysis.png"
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        print(f"📊 Análisis de yield curve guardado en: {outpath}")
    return fig

def create_volatility_smile_analysis(macro, save=True):
    """Crea análisis de la sonrisa de volatilidad implícita (simulada con VIX)."""
    print("Creando análisis de sonrisa de volatilidad...")
    _set_style()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('😊 ANÁLISIS DE SONRISA DE VOLATILIDAD IMPLÍCITA', fontsize=20, fontweight='bold', color='#1f4e79')
    
    # VIX histórico con zonas
    if 'VIX' in macro and len(macro['VIX']):
        vix_data = macro['VIX'].dropna()
        ax1.plot(vix_data.index, vix_data.values, linewidth=2, color='#e74c3c', alpha=0.8)
        ax1.axhline(y=12, color='green', linestyle='--', alpha=0.7, label='Muy Baja (<12)')
        ax1.axhline(y=20, color='orange', linestyle='--', alpha=0.7, label='Normal (12-20)')
        ax1.axhline(y=30, color='red', linestyle='--', alpha=0.7, label='Alta (20-30)')
        ax1.axhline(y=40, color='purple', linestyle='--', alpha=0.7, label='Muy Alta (>30)')
        ax1.fill_between(vix_data.index, 0, vix_data.values, where=(vix_data.values < 12), color='green', alpha=0.2)
        ax1.fill_between(vix_data.index, 0, vix_data.values, where=((vix_data.values >= 12) & (vix_data.values < 20)), color='lightgreen', alpha=0.2)
        ax1.fill_between(vix_data.index, 0, vix_data.values, where=((vix_data.values >= 20) & (vix_data.values < 30)), color='orange', alpha=0.2)
        ax1.fill_between(vix_data.index, 0, vix_data.values, where=(vix_data.values >= 30), color='red', alpha=0.2)
        ax1.set_title('📊 VIX HISTÓRICO CON ZONAS DE VOLATILIDAD', fontsize=14, fontweight='bold', color='#1f4e79')
        ax1.set_ylabel('VIX', fontsize=12); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
    
    # Sonrisa simulada
    strikes = np.array([0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2])
    current_vix = float(macro['VIX'].iloc[-1]) if 'VIX' in macro and len(macro['VIX']) else 20.0
    atm_vol = current_vix / 100.0
    smile_vols = atm_vol * (1 + 0.3 * (strikes - 1.0)**2)
    ax2.plot(strikes, smile_vols * 100, 'o-', linewidth=3, markersize=8, color='#8e44ad', alpha=0.8, label='Sonrisa de Volatilidad')
    ax2.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='ATM (At-the-Money)')
    ax2.set_title('😊 SONRISA DE VOLATILIDAD IMPLÍCITA SIMULADA', fontsize=14, fontweight='bold', color='#1f4e79')
    ax2.set_xlabel('Strike / Spot Ratio', fontsize=12); ax2.set_ylabel('Volatilidad Implícita (%)', fontsize=12)
    ax2.legend(fontsize=10); ax2.grid(True, alpha=0.3)
    for strike, vol in zip(strikes, smile_vols * 100):
        ax2.annotate(f'{vol:.1f}%', (strike, vol), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold', fontsize=9)
    
    # Distribución de VIX
    if 'VIX' in macro and len(macro['VIX']):
        vix_data = macro['VIX'].dropna()
        ax3.hist(vix_data.values, bins=50, color='#e74c3c', alpha=0.7, edgecolor='black', density=True, label='Distribución VIX')
        for p, color in zip([10, 25, 50, 75, 90], ['green', 'lightgreen', 'orange', 'red', 'purple']):
            val = np.percentile(vix_data.values, p); ax3.axvline(x=val, color=color, linestyle='--', alpha=0.7, label=f'P{p}: {val:.1f}')
        ax3.set_title('📈 DISTRIBUCIÓN DE VOLATILIDAD (VIX)', fontsize=14, fontweight='bold', color='#1f4e79')
        ax3.set_xlabel('VIX', fontsize=12); ax3.set_ylabel('Densidad', fontsize=12)
        ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)
    
    # VIX vs retornos S&P500
    if 'VIX' in macro and 'SP500' in macro:
        vix_data = macro['VIX'].dropna()
        sp500_data = macro['SP500'].pct_change().dropna() * 100
        common_index = vix_data.index.intersection(sp500_data.index)
        if len(common_index):
            vix_aligned = vix_data.loc[common_index]; spx_aligned = sp500_data.loc[common_index]
            scatter = ax4.scatter(vix_aligned.values, spx_aligned.values, c=range(len(vix_aligned)), cmap='viridis', alpha=0.6, s=20)
            z = np.polyfit(vix_aligned.values, spx_aligned.values, 1); p = np.poly1d(z)
            ax4.plot(vix_aligned.values, p(vix_aligned.values), "r--", alpha=0.8, linewidth=2, label=f'Tendencia (slope: {z[0]:.3f})')
            ax4.set_title('📊 VOLATILIDAD vs RENDIMIENTOS S&P 500', fontsize=14, fontweight='bold', color='#1f4e79')
            ax4.set_xlabel('VIX', fontsize=12); ax4.set_ylabel('Retorno S&P 500 (%)', fontsize=12)
            ax4.legend(fontsize=10); ax4.grid(True, alpha=0.3)
            cbar = plt.colorbar(scatter, ax=ax4); cbar.set_label('Tiempo', fontsize=10)
    
    plt.tight_layout()
    if save:
        try:
            base_dir = Path(__file__).resolve().parent
        except NameError:
            base_dir = Path.cwd()
        png_dir = base_dir / "png"
        png_dir.mkdir(parents=True, exist_ok=True)
        outpath = png_dir / "volatility_smile_analysis.png"
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        print(f"📊 Análisis de sonrisa de volatilidad guardado en: {outpath}")
    return fig

# ====================== Black-Scholes & opciones (reales) =====================

def black_scholes_call(S, K, T, r, sigma):
    """Calcula el precio de una opción call usando Black-Scholes."""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def implied_volatility_call(S, K, T, r, market_price, max_iter=100, tolerance=1e-6):
    """Calcula la volatilidad implícita de una opción call usando bisección."""
    def objective(sigma):
        return black_scholes_call(S, K, T, r, sigma) - market_price
    try:
        return brentq(objective, 0.001, 2.0, xtol=tolerance, maxiter=max_iter)
    except Exception:
        return np.nan

def download_options_data(symbol="SPY", expiration_dates=None):
    """Descarga datos de opciones reales desde Yahoo Finance. SIEMPRE devuelve 4 valores."""
    print(f"Descargando datos de opciones para {symbol}...")
    try:
        ticker = yf.Ticker(symbol)
        # Fechas de expiración
        if expiration_dates is None:
            exp_dates = ticker.options
            if not exp_dates:
                print(f"❌ No hay fechas de expiración disponibles para {symbol}")
                return None, None, None, None
            today = datetime.now().date()
            future_dates = [d for d in exp_dates if datetime.strptime(d, '%Y-%m-%d').date() > today]
            if not future_dates:
                print(f"❌ No hay fechas de expiración futuras para {symbol}")
                return None, None, None, None
            exp_date = future_dates[0]
        else:
            exp_date = expiration_dates

        print(f"  📅 Usando fecha de expiración: {exp_date}")
        chain = ticker.option_chain(exp_date)
        calls = getattr(chain, 'calls', pd.DataFrame())
        puts  = getattr(chain, 'puts',  pd.DataFrame())

        if (calls is None or calls.empty) and (puts is None or puts.empty):
            print(f"❌ No hay datos de opciones para {symbol} en {exp_date}")
            return None, None, None, exp_date

        hist = ticker.history(period="1d")
        if hist is None or hist.empty:
            print(f"❌ No se pudo obtener precio spot de {symbol}")
            return calls, puts, None, exp_date
        current_price = float(hist['Close'].iloc[-1])

        print(f"  💰 Precio actual de {symbol}: ${current_price:.2f}")
        print(f"  📊 Calls disponibles: {len(calls)}")
        print(f"  📊 Puts disponibles: {len(puts)}")
        return calls, puts, current_price, exp_date
    except Exception as e:
        print(f"❌ Error descargando opciones de {symbol}: {e}")
        return None, None, None, None

def create_real_volatility_smile(symbol="SPY", r=None, save=True):
    """Crea análisis de sonrisa de volatilidad REAL usando datos de opciones."""
    print("Creando análisis de sonrisa de volatilidad REAL...")
    _set_style()
    
    calls, puts, current_price, exp_date = download_options_data(symbol)
    if calls is None or exp_date is None or current_price is None:
        print("❌ No se pudieron obtener datos de opciones")
        return None
    if calls.empty and puts.empty:
        print("❌ No hay opciones válidas para análisis")
        return None
    
    # Tiempo a vencimiento
    exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
    days_to_exp = max((exp_datetime - datetime.now()).days, 1)
    T = days_to_exp / 365.0

    # Tasa libre de riesgo
    if r is None:
        r = 0.05  # 5% anual aprox si no hay TB3MS
    print(f"  📅 Exp: {exp_date} | ⏰ Días: {days_to_exp} | r: {r:.3%}")

    # Validación de datos
    def _valid(df):
        if df is None or df.empty: return pd.DataFrame()
        df = df.dropna(subset=['lastPrice', 'strike'])
        if 'impliedVolatility' in df.columns:
            df = df[df['impliedVolatility'] > 0]
        return df

    valid_calls = _valid(calls).copy()
    valid_puts  = _valid(puts).copy()
    if valid_calls.empty and valid_puts.empty:
        print("❌ No hay opciones válidas tras filtrado")
        return None

    # Calcular IV real si no viene
    def _ensure_iv(df, typ):
        if df.empty: return df
        if 'impliedVolatility' in df.columns:
            df['real_iv'] = df['impliedVolatility']
        else:
            ivs = []
            for _, row in df.iterrows():
                ivs.append(implied_volatility_call(current_price, row['strike'], T, r, row['lastPrice']))
            df['real_iv'] = ivs
        df = df.dropna(subset=['real_iv'])
        df = df[df['real_iv'] > 0]
        df['moneyness'] = df['strike'] / current_price
        df['option_type'] = typ
        return df.sort_values('moneyness')

    valid_calls = _ensure_iv(valid_calls, 'Call')
    valid_puts  = _ensure_iv(valid_puts,  'Put')
    if valid_calls.empty and valid_puts.empty:
        print("❌ No se pudieron calcular IVs válidas")
        return None

    combined = pd.concat([df for df in [valid_calls, valid_puts] if not df.empty], ignore_index=True)

    # Gráficos
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(f'😊 SONRISA DE VOLATILIDAD REAL - {symbol} (Exp: {exp_date})', fontsize=20, fontweight='bold', color='#1f4e79')

    # Sonrisa
    if not valid_calls.empty:
        ax1.scatter(valid_calls['moneyness'], valid_calls['real_iv'] * 100, color='#8e44ad', alpha=0.7, s=50, label=f'Calls ({len(valid_calls)})')
    if not valid_puts.empty:
        ax1.scatter(valid_puts['moneyness'], valid_puts['real_iv'] * 100, color='#e74c3c', alpha=0.7, s=50, label=f'Puts ({len(valid_puts)})')
    if len(combined) > 2:
        z = np.polyfit(combined['moneyness'], combined['real_iv'] * 100, 2); p = np.poly1d(z)
        x_trend = np.linspace(combined['moneyness'].min(), combined['moneyness'].max(), 100)
        ax1.plot(x_trend, p(x_trend), 'k--', linewidth=2, alpha=0.8, label='Tendencia')
    ax1.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='ATM')
    ax1.set_title('🎯 SONRISA DE VOLATILIDAD REAL (Calls + Puts)', fontsize=14, fontweight='bold', color='#1f4e79')
    ax1.set_xlabel('Strike / Spot Ratio', fontsize=12); ax1.set_ylabel('Volatilidad Implícita (%)', fontsize=12)
    ax1.legend(fontsize=10); ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.98, f'Spot: ${current_price:.2f}\nDías: {days_to_exp}\nCalls: {len(valid_calls)} | Puts: {len(valid_puts)}',
             transform=ax1.transAxes, fontsize=10, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7),
             verticalalignment='top')

    # Distribución de strikes
    ax2.hist(combined['strike'].values, bins=20, color='#3498db', alpha=0.7, edgecolor='black', label='Strikes')
    ax2.axvline(x=current_price, color='red', linestyle='--', alpha=0.7, label=f'Spot (${current_price:.2f})')
    ax2.set_title('📊 DISTRIBUCIÓN DE STRIKES (Calls + Puts)', fontsize=14, fontweight='bold', color='#1f4e79')
    ax2.set_xlabel('Strike Price ($)', fontsize=12); ax2.set_ylabel('Frecuencia', fontsize=12)
    ax2.legend(fontsize=10); ax2.grid(True, alpha=0.3)

    # IV vs Precio opcional
    if not valid_calls.empty:
        ax3.scatter(valid_calls['lastPrice'], valid_calls['real_iv'] * 100, color='#8e44ad', alpha=0.7, s=50, label='Calls')
    if not valid_puts.empty:
        ax3.scatter(valid_puts['lastPrice'], valid_puts['real_iv'] * 100, color='#e74c3c', alpha=0.7, s=50, label='Puts')
    ax3.set_title('💰 VOLATILIDAD vs PRECIO DE LA OPCIÓN', fontsize=14, fontweight='bold', color='#1f4e79')
    ax3.set_xlabel('Precio de la Opción ($)', fontsize=12); ax3.set_ylabel('Volatilidad Implícita (%)', fontsize=12)
    ax3.legend(fontsize=10); ax3.grid(True, alpha=0.3)

    # IV vs Volumen (si hay)
    if 'volume' in combined.columns:
        calls_data = combined[combined['option_type'] == 'Call']
        puts_data  = combined[combined['option_type'] == 'Put']
        if not calls_data.empty:
            sc1 = ax4.scatter(calls_data['moneyness'], calls_data['real_iv'] * 100, c=calls_data['volume'], cmap='Purples', alpha=0.7, s=50, label='Calls', edgecolors='black', linewidth=0.5)
        if not puts_data.empty:
            sc2 = ax4.scatter(puts_data['moneyness'], puts_data['real_iv'] * 100, c=puts_data['volume'], cmap='Reds', alpha=0.7, s=50, label='Puts', edgecolors='black', linewidth=0.5)
        ax4.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='ATM')
        ax4.set_title('📈 VOLATILIDAD vs VOLUMEN (Calls + Puts)', fontsize=14, fontweight='bold', color='#1f4e79')
        ax4.set_xlabel('Strike / Spot Ratio', fontsize=12); ax4.set_ylabel('Volatilidad Implícita (%)', fontsize=12)
        ax4.legend(fontsize=10); ax4.grid(True, alpha=0.3)
        if not calls_data.empty: plt.colorbar(sc1, ax=ax4, label='Volumen Calls')
        if not puts_data.empty:  plt.colorbar(sc2, ax=ax4, label='Volumen Puts')
    else:
        if not valid_calls.empty:
            ax4.scatter(valid_calls['moneyness'], valid_calls['real_iv'] * 100, color='#8e44ad', alpha=0.7, s=50, label='Calls')
        if not valid_puts.empty:
            ax4.scatter(valid_puts['moneyness'], valid_puts['real_iv'] * 100, color='#e74c3c', alpha=0.7, s=50, label='Puts')
        ax4.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='ATM')
        ax4.set_title('📈 VOLATILIDAD IMPLÍCITA REAL (Calls + Puts)', fontsize=14, fontweight='bold', color='#1f4e79')
        ax4.set_xlabel('Strike / Spot Ratio', fontsize=12); ax4.set_ylabel('Volatilidad Implícita (%)', fontsize=12)
        ax4.legend(fontsize=10); ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        try:
            base_dir = Path(__file__).resolve().parent
        except NameError:
            base_dir = Path.cwd()
        png_dir = base_dir / "png"
        png_dir.mkdir(parents=True, exist_ok=True)
        outpath = png_dir / "real_volatility_smile.png"
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        print(f"📊 Sonrisa de volatilidad REAL guardada en: {outpath}")
    return fig

# ================================ MAIN ========================================

def main():
    print("\n=== ANÁLISIS MACROECONÓMICO PROFESIONAL ===")

    # 1) Datos macro (Yahoo) + FRED
    macro_yf = download_yahoo_assets(START_DATE, END_DATE)
    macro_fred = download_fred_series(FRED_SERIES, START_DATE, END_DATE)
    macro = {**macro_yf, **macro_fred}
    if not macro:
        raise RuntimeError("No se pudieron descargar datos macro.")

    # 2) Cartera
    portfolio_prices = download_portfolio_prices(PORTFOLIO_TICKERS, START_DATE, END_DATE)
    if not portfolio_prices:
        raise RuntimeError("No se pudieron descargar precios de cartera.")
    rets_by_ticker, port_ret, eff_weights = build_portfolio_returns(portfolio_prices, PORTFOLIO_WEIGHTS)
    print(f"Observaciones de cartera: {len(port_ret)} | Tickers efectivos: {list(eff_weights[eff_weights>0].index)}")

    # 3) Regímenes (rf = TBILL_3M si está)
    rf = macro.get('TBILL_3M')
    regime_df = detect_market_regimes(port_ret, window=252, rf_series=rf)

    # 4) Correlaciones macro (cartera)
    correlations, all_data = analyze_macro_correlations(port_ret, macro)

    # 4b) Correlaciones del residuo vs S&P 500 (opcional)
    correlations_resid = None
    if USE_RESIDUAL_CORRS and 'SP500' in macro:
        spx_ret = macro['SP500'].pct_change().dropna()
        port_resid = residualize_vs_equity(port_ret, spx_ret)
        print("\nCorrelaciones sobre residuo (cartera sin beta SPX):")
        correlations_resid, _ = analyze_macro_correlations(port_resid, macro)

    # 5) Señales de riesgo
    signals = generate_risk_signals(macro)

    # 6) Dashboard principal
    create_macro_dashboard(macro, correlations, signals, regime_df, port_ret, save=True)
    
    # 6b) Gráficos individuales detallados
    create_individual_charts(macro, correlations, signals, regime_df, save=True)
    
    # 6c) Análisis de Yield Curve
    create_yield_curve_analysis(macro, save=True)
    
    # 6d) Análisis de Sonrisa de Volatilidad (simulada)
    create_volatility_smile_analysis(macro, save=True)
    
    # 6e) Sonrisa de Volatilidad REAL (datos de opciones)
    r_annual = None
    if 'TBILL_3M' in macro and len(macro['TBILL_3M']):
        try:
            r_annual = float(macro['TBILL_3M'].dropna().iloc[-1]) / 100.0
        except Exception:
            r_annual = None
    create_real_volatility_smile("SPY", r=r_annual, save=True)

    # 7) Resumen
    print("\n=== RESUMEN EJECUTIVO ===")
    print(f"Indicadores macro descargados: {len(macro)}")
    print(f"Correlaciones calculadas: {len(correlations)}")
    if correlations_resid is not None:
        print(f"Correlaciones (residuo) calculadas: {len(correlations_resid)}")
    print(f"Señales de riesgo: {len(signals)}")
    if regime_df is not None:
        print(f"Regímenes detectados: {regime_df['regime'].nunique()}")
    print("Análisis completado.")

    return {
        'macro_data': macro,
        'portfolio_prices': portfolio_prices,
        'portfolio_returns': rets_by_ticker,
        'portfolio': port_ret,
        'effective_weights': eff_weights,
        'regime_df': regime_df,
        'correlations': correlations,
        'correlations_resid': correlations_resid,
        'all_data': all_data,
        'signals': signals
    }

# =============================== EJECUCIÓN ====================================

if __name__ == "__main__":
    try:
        results = main()
        print("\nListo. Revisa ./png/macro_dashboard_avanzado.png")
    except KeyboardInterrupt:
        print("\nProceso interrumpido por el usuario.")
    except Exception as e:      
        print(f"\nError inesperado: {e}")
        import traceback; traceback.print_exc()
