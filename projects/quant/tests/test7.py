# -*- coding: utf-8 -*-
"""
ANÁLISIS CAPM CON FRONTERA EFICIENTE - BASE EUR
===============================================
- Convierte TODO a EUR (incluido benchmark en USD) usando EURUSD=X
- CAPM con simple returns + rf diaria compuesta (en EUR)
- Frontera eficiente (sin short) + CML
- SML coherente (en EUR)
- CSV con métricas y PNG con gráficos
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pathlib import Path
from datetime import datetime

# =========================
# Configuración
# =========================
# Ejemplos de activos en EUR: bolsa española (.MC), francesa (.PA), alemana (.DE), BTC-EUR, etc.
PORTFOLIO_TICKERS = ["0P0001CLDK.F","NVDA","MSFT","AAPL","GOOGL","IBM","AMZN","META","TSLA","JPM","BRK-A","BTC-EUR","GLD"]
PORTFOLIO_WEIGHTS = np.array([0.514, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.486])

# Benchmark en USD (se convertirá a EUR automáticamente); si prefieres EU, prueba "^STOXX50E"
BENCHMARK_TICKER = "EUNL.DE"

# Tasa libre de riesgo ANUAL en EUR (ajústala a tu referencia: Bund/€STR, etc.)
RISK_FREE_RATE = 0.0425

START_DATE = "2018-01-01"
END_DATE = None               # hasta hoy
BETA_START_DATE = "2019-01-01"  # ~5 años para beta
BASE_CURRENCY = "EUR"           # ¡clave! todo el análisis en EUR

# =========================
# Utilidades de descarga
# =========================
def download_close(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    if "Close" in getattr(df, "columns", []):
        df = df["Close"]
    if isinstance(df, pd.Series):
        name = tickers if isinstance(tickers, str) else tickers[0]
        df = df.to_frame(name=name)
    return df.dropna(how="all")

_CCY_CACHE = {}
def infer_currency(ticker: str) -> str:
    """Intenta obtener la divisa real del ticker; fallback por sufijo."""
    t = ticker.upper()
    if t in _CCY_CACHE:
        return _CCY_CACHE[t]
    cur = None
    try:
        cur = yf.Ticker(t).fast_info.get("currency", None)
    except Exception:
        cur = None
    if cur:
        cur = cur.upper()
    else:
        # Heurísticas de sufijos habituales
        eur_suffixes = (".MC",".PA",".DE",".MI",".AS",".BE",".BR",".LS",".F")
        if t.endswith("-EUR") or t.endswith(eur_suffixes):
            cur = "EUR"
        elif t.endswith("-USD") or t.startswith("^") or "." not in t:
            cur = "USD"
        else:
            cur = "USD"  # por defecto
    _CCY_CACHE[t] = cur
    return cur

def convert_df_to_base_currency(prices_df: pd.DataFrame, base_ccy: str, tickers=None,
                                start=None, end=None):
    """Convierte cada columna a la moneda base usando EURUSD=X cuando haga falta (EUR<->USD)."""
    if tickers is None:
        tickers = list(prices_df.columns)
    if start is None or end is None:
        start = prices_df.index.min().strftime("%Y-%m-%d")
        end = prices_df.index.max().strftime("%Y-%m-%d")

    out = {}
    # Serie FX única para toda la ventana
    fx = download_close("EURUSD=X", start, end).squeeze().ffill().bfill()  # USD por EUR
    for t in tickers:
        s = prices_df[t].dropna().copy()
        if s.empty:
            continue
        src = infer_currency(t)
        if src == base_ccy:
            out[t] = s
        else:
            if {src, base_ccy} != {"EUR","USD"}:
                raise ValueError(f"Conversión {src}->{base_ccy} no soportada ({t}).")
            fx_aligned = fx.reindex(s.index).ffill().bfill()
            if src == "USD" and base_ccy == "EUR":
                out[t] = s / fx_aligned         # USD -> EUR
            elif src == "EUR" and base_ccy == "USD":
                out[t] = s * fx_aligned         # EUR -> USD
            else:
                out[t] = s                      # no-op
    out_df = pd.DataFrame(out).sort_index().dropna(how="all")
    return out_df

def ensure_weights(assets_cols, weights, tickers_original):
    """Reordena y normaliza pesos respecto a columnas efectivas."""
    wmap = {t: w for t, w in zip(tickers_original, weights)}
    kept = [t for t in assets_cols if t in wmap]
    w = np.array([wmap[t] for t in kept], dtype=float)
    if not np.isclose(w.sum(), 1.0):
        print(f"⚠️  Los pesos no suman 1 (suman {w.sum():.4f}). Se normalizan.")
        w = w / w.sum()
    return kept, w

# =========================
# Optimización / frontera
# =========================
def calculate_efficient_frontier(returns: pd.DataFrame, n_points=50):
    """Frontera eficiente sin ventas en corto (bounds [0,1]) con anualización simple."""
    mean_returns = returns.mean() * 252
    cov_matrix  = returns.cov() * 252

    def port_var(w): return float(w.T @ cov_matrix @ w)
    def port_ret(w): return float(np.sum(w * mean_returns))

    n = len(returns.columns)
    bounds = tuple((0, 1) for _ in range(n))
    targets = np.linspace(mean_returns.min(), mean_returns.max(), n_points)

    er, ev, ew = [], [], []
    for tr in targets:
        cons = [{"type":"eq","fun":lambda x: np.sum(x)-1},
                {"type":"eq","fun":lambda x, target=tr: port_ret(x)-target}]
        res = minimize(port_var, x0=np.full(n, 1/n), method="SLSQP", bounds=bounds, constraints=cons)
        if res.success:
            w = res.x
            ew.append(w)
            er.append(port_ret(w))
            ev.append(np.sqrt(port_var(w)))
    return np.array(er), np.array(ev), np.array(ew)

def calculate_cml(eff_ret, eff_vol, risk_free_rate):
    """Capital Market Line según punto tangente (máx Sharpe)."""
    sharpe = (eff_ret - risk_free_rate) / np.maximum(eff_vol, 1e-12)
    i = int(np.argmax(sharpe))
    tr, tv = eff_ret[i], eff_vol[i]
    slope = (tr - risk_free_rate) / tv
    vol_grid = np.linspace(0, eff_vol.max()*1.2, 100)
    ret_grid = risk_free_rate + slope * vol_grid
    return ret_grid, vol_grid, (tr, tv)

# =========================
# CAPM / métricas
# =========================
def capm_regression(y_asset, x_bench, rf_daily):
    """OLS: (y - rf) = alpha + beta*(x - rf) + eps."""
    y = y_asset - rf_daily
    x = x_bench - rf_daily
    X = np.c_[np.ones_like(x), x]
    alpha, beta = np.linalg.lstsq(X, y, rcond=None)[0]
    corr = np.corrcoef(x, y)[0, 1]
    return alpha, beta, corr

# =========================
# Main
# =========================
def main():
    print("🚀 ANÁLISIS CAPM (BASE EUR)")
    print("="*60)

    # -------- Descarga
    print("📈 Descargando precios ajustados…")
    start_all = min(pd.to_datetime(START_DATE), pd.to_datetime(BETA_START_DATE)).strftime("%Y-%m-%d")
    end_str = END_DATE or datetime.today().strftime("%Y-%m-%d")

    portfolio_prices      = download_close(PORTFOLIO_TICKERS, start=START_DATE,     end=END_DATE)
    benchmark_prices      = download_close(BENCHMARK_TICKER,   start=START_DATE,     end=END_DATE).squeeze().to_frame(BENCHMARK_TICKER)
    portfolio_prices_beta = download_close(PORTFOLIO_TICKERS, start=BETA_START_DATE, end=END_DATE)
    benchmark_prices_beta = download_close(BENCHMARK_TICKER,   start=BETA_START_DATE, end=END_DATE).squeeze().to_frame(BENCHMARK_TICKER)

    # -------- Conversión a EUR
    print("💱 Convirtiendo todo a EUR…")
    portfolio_prices      = convert_df_to_base_currency(portfolio_prices,      BASE_CURRENCY, PORTFOLIO_TICKERS, START_DATE,     end_str)
    benchmark_prices      = convert_df_to_base_currency(benchmark_prices,      BASE_CURRENCY, [BENCHMARK_TICKER], START_DATE,     end_str)
    portfolio_prices_beta = convert_df_to_base_currency(portfolio_prices_beta, BASE_CURRENCY, PORTFOLIO_TICKERS, BETA_START_DATE, end_str)
    benchmark_prices_beta = convert_df_to_base_currency(benchmark_prices_beta, BASE_CURRENCY, [BENCHMARK_TICKER], BETA_START_DATE, end_str)

    # -------- Simple returns
    R_assets      = portfolio_prices.pct_change().dropna(how="all")
    R_bench       = benchmark_prices.pct_change().dropna(how="all").squeeze()
    R_assets_beta = portfolio_prices_beta.pct_change().dropna(how="all")
    R_bench_beta  = benchmark_prices_beta.pct_change().dropna(how="all").squeeze()

    # Alinear
    common_full = R_assets.index.intersection(R_bench.index)
    R_assets = R_assets.loc[common_full]
    R_bench  = R_bench.loc[common_full]

    common_beta = R_assets_beta.index.intersection(R_bench_beta.index)
    R_assets_beta = R_assets_beta.loc[common_beta]
    R_bench_beta  = R_bench_beta.loc[common_beta]

    print(f"📊 Observaciones (completo): {len(R_assets)}")
    print(f"📊 Observaciones (beta 5y): {len(R_assets_beta)}")

    # -------- Filtro columnas con pocos datos
    min_obs = 252*2
    valid_cols = [c for c in R_assets.columns if R_assets[c].count() >= min_obs and R_assets_beta[c].count() >= min_obs/2]
    dropped = [c for c in R_assets.columns if c not in valid_cols]
    if dropped:
        print(f"⚠️  Se excluyen por datos insuficientes: {', '.join(dropped)}")
        R_assets      = R_assets[valid_cols]
        R_assets_beta = R_assets_beta[valid_cols]

    # -------- Pesos
    kept_tickers, W = ensure_weights(list(R_assets.columns), PORTFOLIO_WEIGHTS, PORTFOLIO_TICKERS)

    # -------- rf diaria (simple, en EUR)
    rf_daily = (1 + RISK_FREE_RATE)**(1/252) - 1

    # -------- Métricas por activo (beta en ventana 5y)
    metrics = {}
    for a in kept_tickers:
        a_full = R_assets[a].dropna()
        a_beta = R_assets_beta[a].dropna()

        idx_full = a_full.index.intersection(R_bench.index)
        idx_beta = a_beta.index.intersection(R_bench_beta.index)

        a_full = a_full.loc[idx_full]
        m_full = R_bench.loc[idx_full]

        a_beta = a_beta.loc[idx_beta]
        m_beta = R_bench_beta.loc[idx_beta]

        if len(a_full) < 30 or len(a_beta) < 30:
            print(f"⚠️  {a}: insuficientes datos tras alinear.")
            continue

        alpha_d, beta, corr = capm_regression(a_beta.values, m_beta.values, rf_daily)
        mean_ret = a_full.mean() * 252
        vol = a_full.std() * np.sqrt(252)
        sharpe = (mean_ret - RISK_FREE_RATE) / (vol if vol > 0 else np.nan)
        jensen_alpha = (1 + alpha_d)**252 - 1

        metrics[a] = {
            "beta": float(beta),
            "alpha_daily": float(alpha_d),
            "jensen_alpha": float(jensen_alpha),
            "correlation": float(corr),
            "mean_return": float(mean_ret),
            "volatility": float(vol),
            "sharpe_ratio": float(sharpe),
        }
        print(f"✅ {a}: Beta={beta:.3f}, Alpha={jensen_alpha:.2%}, Sharpe={sharpe:.3f}")

    if not metrics:
        raise RuntimeError("No hay activos válidos para el análisis.")

    # -------- Cartera
    kept_tickers, W = ensure_weights(list(metrics.keys()), W, kept_tickers)
    R_full = R_assets[kept_tickers].dropna()
    R_beta = R_assets_beta[kept_tickers].dropna()

    port_full = (R_full * W).sum(axis=1)
    port_beta = (R_beta * W).sum(axis=1)

    idx_full = port_full.index.intersection(R_bench.index)
    idx_beta = port_beta.index.intersection(R_bench_beta.index)

    port_full = port_full.loc[idx_full]
    m_full    = R_bench.loc[idx_full]

    port_beta = port_beta.loc[idx_beta]
    m_beta    = R_bench_beta.loc[idx_beta]

    alpha_d_p, beta_p, corr_p = capm_regression(port_beta.values, m_beta.values, rf_daily)
    mean_ret_p = port_full.mean() * 252
    vol_p = port_full.std() * np.sqrt(252)
    sharpe_p = (mean_ret_p - RISK_FREE_RATE) / (vol_p if vol_p > 0 else np.nan)
    jensen_alpha_p = (1 + alpha_d_p)**252 - 1

    portfolio_metrics = {
        "beta": float(beta_p),
        "alpha_daily": float(alpha_d_p),
        "jensen_alpha": float(jensen_alpha_p),
        "correlation": float(corr_p),
        "mean_return": float(mean_ret_p),
        "volatility": float(vol_p),
        "sharpe_ratio": float(sharpe_p),
    }

    print("\n🎯 CARTERA (EUR):")
    print(f"Beta: {beta_p:.3f}")
    print(f"Alpha: {jensen_alpha_p:.2%}")
    print(f"Retorno: {mean_ret_p:.2%}")
    print(f"Volatilidad: {vol_p:.2%}")
    print(f"Sharpe: {sharpe_p:.3f}")

    # -------- Frontera eficiente
    print("\n📐 Calculando frontera eficiente…")
    eff_ret, eff_vol, eff_w = calculate_efficient_frontier(R_full)
    if len(eff_ret) < 10:
        print("⚠️  Pocos puntos válidos en la frontera.")

    # CML
    cml_ret, cml_vol, tangent = calculate_cml(eff_ret, eff_vol, RISK_FREE_RATE)

    # -------- SML (todo en EUR)
    market_return = m_full.mean() * 252
    beta_axis = np.linspace(0, max(1.2, max([metrics[a]["beta"] for a in metrics])*1.2), 100)
    sml_returns = RISK_FREE_RATE + (market_return - RISK_FREE_RATE) * beta_axis

    # -------- Gráficos
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Análisis CAPM con Frontera Eficiente (EUR)', fontsize=16, fontweight='bold')

    # (1) SML
    betas = [metrics[a]["beta"] for a in metrics]
    rets  = [metrics[a]["mean_return"] for a in metrics]
    ax1.scatter(betas, rets, s=100, alpha=0.7, label='Activos')
    ax1.scatter(portfolio_metrics['beta'], portfolio_metrics['mean_return'],
                s=150, marker='*', label='Cartera', zorder=5)
    ax1.plot(beta_axis, sml_returns, linestyle='--', label='SML')
    ax1.set_xlabel('Beta')
    ax1.set_ylabel('Retorno Esperado (anual, EUR)')
    ax1.set_title('Línea del Mercado de Valores (SML)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # (2) Frontera + CML
    ax2.plot(eff_vol, eff_ret, linewidth=2, label='Frontera Eficiente')
    ax2.plot(cml_vol, cml_ret, linestyle='--', linewidth=2, label='CML')
    ax2.scatter(portfolio_metrics['volatility'], portfolio_metrics['mean_return'],
                s=150, marker='*', label='Cartera', zorder=5)
    for a in metrics:
        ax2.scatter(metrics[a]['volatility'], metrics[a]['mean_return'], s=80, alpha=0.7, label=a)
    h, l = ax2.get_legend_handles_labels()
    uniq = dict(zip(l, h))
    ax2.legend(uniq.values(), uniq.keys(), fontsize=8, ncol=2)
    ax2.set_xlabel('Volatilidad (anual, EUR)')
    ax2.set_ylabel('Retorno Esperado (anual, EUR)')
    ax2.set_title('Frontera Eficiente y CML')
    ax2.grid(True, alpha=0.3)

    # (3) Alpha vs Beta
    alphas = [metrics[a]["jensen_alpha"] for a in metrics]
    ax3.scatter(betas, alphas, s=100, alpha=0.7, label='Activos')
    ax3.scatter(portfolio_metrics['beta'], portfolio_metrics['jensen_alpha'],
                s=150, marker='*', label='Cartera', zorder=5)
    ax3.axhline(0, linestyle='--', alpha=0.5)
    ax3.set_xlabel('Beta')
    ax3.set_ylabel("Jensen's Alpha (anual, EUR)")
    ax3.set_title('Alpha vs Beta')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # (4) Sharpe vs Volatilidad
    sharpes = [metrics[a]['sharpe_ratio'] for a in metrics]
    vols    = [metrics[a]['volatility'] for a in metrics]
    ax4.scatter(vols, sharpes, s=100, alpha=0.7, label='Activos')
    ax4.scatter(portfolio_metrics['volatility'], portfolio_metrics['sharpe_ratio'],
                s=150, marker='*', label='Cartera', zorder=5)
    ax4.set_xlabel('Volatilidad (anual, EUR)')
    ax4.set_ylabel('Sharpe Ratio (anual)')
    ax4.set_title('Sharpe vs Volatilidad')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    Path("png").mkdir(exist_ok=True)
    out_png = Path("png") / "capm_enhanced_eur.png"
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\n📊 Gráfico guardado en: {out_png}")

    # CSV métricas
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.loc["__PORTFOLIO__"] = portfolio_metrics
    # Crear directorio outputs si no existe
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    
    metrics_df.to_csv(outputs_dir / "capm_metrics_eur.csv", float_format="%.6f")
    print(f"📄 Métricas guardadas en: {outputs_dir / 'capm_metrics_eur.csv'}")

    # Resumen
    print("\n📋 RESUMEN:")
    print(f"Período completo: {R_assets.index[0].date()} → {R_assets.index[-1].date()}")
    print(f"Período beta:     {R_assets_beta.index[0].date()} → {R_assets_beta.index[-1].date()}")
    print(f"Benchmark: {BENCHMARK_TICKER} (convertido a EUR)")
    print(f"Tasa libre de riesgo (EUR): {RISK_FREE_RATE:.1%}")

    print("\n🔍 INTERPRETACIÓN:")
    b = portfolio_metrics['beta']
    if b > 1:
        print(f"• Cartera más sensible que el mercado (β = {b:.3f})")
    elif b < 1:
        print(f"• Cartera menos sensible que el mercado (β = {b:.3f})")
    else:
        print(f"• Sensibilidad similar al mercado (β = {b:.3f})")

    if portfolio_metrics['jensen_alpha'] > 0:
        print(f"• Alpha positivo: {portfolio_metrics['jensen_alpha']:.2%}")
    else:
        print(f"• Alpha negativo: {portfolio_metrics['jensen_alpha']:.2%}")

    r = portfolio_metrics['correlation']
    if r > 0.7:
        print(f"• Alta correlación con el mercado (R = {r:.3f})")
    elif r > 0.3:
        print(f"• Correlación moderada con el mercado (R = {r:.3f})")
    else:
        print(f"• Baja correlación con el mercado (R = {r:.3f})")

    return metrics, portfolio_metrics, eff_ret, eff_vol, eff_w

if __name__ == "__main__":
    try:
        _ = main()
        print("\n✅ Análisis CAPM (EUR) completado con éxito.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
