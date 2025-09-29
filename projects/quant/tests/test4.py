"""
Módulo Avanzado de Gestión de Riesgo para Hedge Funds (refactor + Tracking Error MSCI)
=====================================================================================

Novedades clave en esta versión:
- Conversión coherente de **todos** los activos USD → EUR (heurística por sufijo).
- Benchmark MSCI (proxy por defecto **ACWI**) con conversión opcional a EUR.
- **Tracking Error** (diario/anual) e **Information Ratio** (externo) con `ddof` parametrizable.
- **Rolling Tracking Error** (por defecto 63 días ≈ 1 trimestre) y print del último valor.
- **ES paramétrico** con signo correcto para cola izquierda.
- Correlación *rolling* robusta para 2 activos (y promedio para N>2).
- **Stress Testing** con escenarios por `DEFAULT` (aplica a todos los tickers si no se especifica).
- Reporte visual con drawdown en **%** y magnitudes de riesgo positivas en barras.
- Opción en `risk_metrics_summary` para calcular IR contra benchmark **externo**.

Notas:
- VaR/ES histórico se anualiza por √252 como aproximación (iid); véase cautela al interpretar.
- Si tu benchmark ya cotiza en EUR (p.ej. EUNL.DE), pon `BENCH_IN_USD=False` y no habrá conversión.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import yfinance as yf
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# ----------------------
# Parámetros del usuario
# ----------------------
TICKERS = ["0P0001CLDK.F","NVDA","MSFT","AAPL","GOOGL","IBM","AMZN","META","TSLA","JPM","BRK-A","BTC-EUR","GLD"]
FX_TICKER = "EURUSD=X"                   # USD por 1 EUR
Rf = 0.045                               # Tasa libre de riesgo anual (p.ej. 4.5%)
START = "2018-03-20"
END = "2025-09-02"                  # Ajusta según necesites
PORTFOLIO_WEIGHTS = np.array([0.674, 0.044, 0.007, 0.275])  # Ajusta a tus pesos actuales

# Benchmark (MSCI)
BENCH_TICKER = "MSCI_WORLD"      # Proxy MSCI ACWI (USD). Cambia a tu MSCI preferido (p.ej., "EUNL.DE" en EUR)
BENCH_IN_USD = False        # Pon a False si el benchmark ya cotiza en EUR

# Semilla para Monte Carlo (reproducible)
SEED = 42

# Estilo global
plt.style.use('seaborn-v0_8')


# ----------------------
# Utilidades de descarga
# ----------------------

def download_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    print(f"📈 Descargando precios de activos ({', '.join(tickers)})…")
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=True,
        group_by="ticker",
        threads=True,
    )
    return data


def download_fx(fx_ticker: str, start: str, end: str) -> pd.DataFrame:
    print("💱 Descargando tipo de cambio USD/EUR…")
    fx = yf.download(
        tickers=fx_ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=True,
        group_by="ticker",
        threads=True,
    )
    return fx


def extract_adj_close(df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """Devuelve un DataFrame (fecha x ticker) con Adj Close. Soporta MultiIndex de yfinance."""
    if df.empty:
        return pd.DataFrame()

    out = {}
    if isinstance(df.columns, pd.MultiIndex):
        for tk in tickers:
            col = (tk, "Adj Close")
            if col in df.columns:
                out[tk] = df[col].rename(tk)
    else:
        if "Adj Close" in df.columns and len(tickers) == 1:
            out[tickers[0]] = df["Adj Close"].rename(tickers[0])
    return pd.concat(out, axis=1) if out else pd.DataFrame()


def usd_to_eur(series_usd: pd.Series, fx_df: pd.DataFrame, fx_ticker: str = FX_TICKER) -> pd.Series:
    """Convierte precios en USD a EUR usando EURUSD=X (USD por 1 EUR): EUR = USD / EURUSD."""
    if fx_df.empty:
        print("⚠️  No hay datos de FX; manteniendo USD.")
        return series_usd

    if isinstance(fx_df.columns, pd.MultiIndex):
        fx_col = (fx_ticker, "Adj Close")
    else:
        fx_col = "Adj Close"

    if fx_col not in fx_df.columns:
        print("⚠️  Columna de FX no encontrada; manteniendo USD.")
        return series_usd

    fx_rate = fx_df[fx_col].reindex(series_usd.index).ffill()
    converted = series_usd / fx_rate
    return converted


def guess_usd_tickers(tickers: List[str]) -> List[str]:
    """
    Heurística simple:
      - Considera EUR si termina en .DE, .F, .AS, .PA, .MI, .MC o en -EUR (ej. BTC-EUR).
      - El resto se asume USD.
    """
    eur_suffixes = (".DE", ".F", ".AS", ".PA", ".MI", ".MC", "-EUR")
    return [tk for tk in tickers if not tk.endswith(eur_suffixes)]


# ----------------------
# Preparación de datos
# ----------------------

def prepare_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("📥 Descargando datos históricos…")
    px = download_prices(TICKERS, START, END)
    fx = download_fx(FX_TICKER, START, END)

    adj = extract_adj_close(px, TICKERS)
    if adj.empty:
        raise SystemExit("No se pudo obtener 'Adj Close' para los tickers.")

    # Convertir TODOS los tickers en USD a EUR
    usd_tickers = [tk for tk in guess_usd_tickers(TICKERS) if tk in adj.columns]
    for ticker in usd_tickers:
        adj[ticker] = usd_to_eur(adj[ticker], fx, FX_TICKER)
        print(f"💱 Convertido {ticker} de USD a EUR usando {FX_TICKER}")

    adj = adj.dropna(how="all").sort_index()
    print(f"📅 Rango de fechas disponible: {adj.index[0].date()} a {adj.index[-1].date()}")

    # Fechas comunes (todos los activos)
    common = adj.dropna(subset=TICKERS, how="any")
    if common.empty:
        raise SystemExit("No hay fechas comunes con datos para todos los activos.")

    start_date = common.index[0]
    print(f"✅ Fecha de inicio común: {start_date.date()}")
    print(f"📊 Observaciones desde inicio común: {len(common)}")

    # (Opcional) forzar frecuencia laboral para series limpias:
    # common = common.asfreq('B').ffill()

    # Retornos **simples** diarios
    rets = common.pct_change().dropna(how="any")
    return common, rets, fx


# ----------------------
# Benchmark y Tracking Error
# ----------------------

def prepare_benchmark_returns(fx: Optional[pd.DataFrame] = None) -> pd.Series:
    """Descarga el benchmark, lo convierte a EUR si es USD y devuelve retornos simples diarios."""
    bench_px = download_prices([BENCH_TICKER], START, END)
    bench_adj = extract_adj_close(bench_px, [BENCH_TICKER])
    if bench_adj.empty:
        raise SystemExit(f"No se pudo obtener 'Adj Close' para el benchmark {BENCH_TICKER}.")

    s = bench_adj[BENCH_TICKER]
    if BENCH_IN_USD:
        if fx is None:
            fx = download_fx(FX_TICKER, START, END)
        s = usd_to_eur(s, fx, FX_TICKER)
        print(f"💱 Convertido benchmark {BENCH_TICKER} de USD a EUR")

    bench_rets = s.pct_change().dropna()
    return bench_rets


def portfolio_series(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    weights = np.asarray(weights, dtype=float)
    assert returns.shape[1] == len(weights), "Dimensión de pesos ≠ número de activos"
    if not np.isclose(weights.sum(), 1.0):
        weights = weights / weights.sum()
    return (returns * weights).sum(axis=1)


def tracking_error(portfolio_returns: pd.Series, benchmark_returns: pd.Series, ddof: int = 1) -> dict:
    """Tracking Error clásico: std de retornos activos; anualizado con √252; IR con exceso medio anual."""
    df = (
        portfolio_returns.rename('p').to_frame()
        .join(benchmark_returns.rename('b').to_frame(), how='inner')
        .dropna()
    )
    if df.empty:
        return {'te_daily': np.nan, 'te_annual': np.nan, 'information_ratio': np.nan}
    active = df['p'] - df['b']
    te_daily = float(active.std(ddof=ddof))
    te_annual = float(te_daily * np.sqrt(252))
    excess_ann = float((df['p'].mean() - df['b'].mean()) * 252)
    info_ratio = float(excess_ann / te_annual) if te_annual > 0 else np.nan
    return {'te_daily': te_daily, 'te_annual': te_annual, 'information_ratio': info_ratio}


def rolling_tracking_error(portfolio_returns: pd.Series, benchmark_returns: pd.Series, window: int = 63, ddof: int = 1) -> pd.Series:
    """Tracking Error *rolling* anualizado (por defecto 63 días ≈ 1 trimestre)."""
    df = (
        portfolio_returns.rename('p').to_frame()
        .join(benchmark_returns.rename('b').to_frame(), how='inner')
        .dropna()
    )
    if df.empty:
        return pd.Series(dtype=float)
    active = df['p'] - df['b']
    return active.rolling(window).std(ddof=ddof) * np.sqrt(252)


# ----------------------
# Funciones de análisis de riesgo
# ----------------------

def calculate_var_es(returns: pd.DataFrame, weights: np.ndarray,
                     confidence_levels=(0.95, 0.99), method='historical', n_sim=10000, seed: int = SEED):
    print("\n🎯 CÁLCULO DE VaR Y ES")
    print("=" * 50)

    pr = portfolio_series(returns, weights)
    out = {}

    for cl in confidence_levels:
        alpha = 1 - cl
        if method == 'historical':
            var = np.quantile(pr, alpha)
            es = pr[pr <= var].mean()
        elif method == 'parametric':
            mu, sd = pr.mean(), pr.std(ddof=0)
            z = stats.norm.ppf(alpha)
            var = mu + sd * z
            # ES Normal (cola izquierda) -> signo correcto
            es = mu - sd * stats.norm.pdf(z) / alpha
        elif method == 'monte_carlo':
            mu, sd = pr.mean(), pr.std(ddof=0)
            rng = np.random.default_rng(seed)
            sims = rng.normal(mu, sd, n_sim)
            var = np.quantile(sims, alpha)
            es = sims[sims <= var].mean()
        else:
            raise ValueError("method debe ser 'historical', 'parametric' o 'monte_carlo'")

        out[cl] = {
            'VaR': float(var),
            'ES': float(es),
            'VaR_daily_pct': float(var * 100),
            'ES_daily_pct': float(es * 100),
            'VaR_annual_pct': float(var * np.sqrt(252) * 100),
            'ES_annual_pct': float(es * np.sqrt(252) * 100),
        }

        print(f"Confianza {int(cl*100)}% → VaR diario: {var*100:.2f}% | ES diario: {es*100:.2f}%")
    return out


def calculate_drawdown_analysis(returns: pd.DataFrame, weights: np.ndarray):
    print("\n📉 ANÁLISIS DE DRAWDOWN")
    print("=" * 40)

    pr = portfolio_series(returns, weights)
    cum = (1 + pr).cumprod()
    running_max = cum.cummax()
    dd = (cum / running_max) - 1.0

    max_dd = dd.min()
    max_dd_date = dd.idxmin()

    underwater = dd < 0
    current, longest = 0, 0
    for val in underwater.astype(int):
        if val:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    ann_ret = pr.mean() * 252
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else np.nan

    dd_monthly = dd.resample('M').min()
    worst3 = dd_monthly.nsmallest(3)
    sterling = (ann_ret - Rf) / abs(worst3.mean()) if len(worst3) >= 1 and worst3.mean() < 0 else np.nan

    print(f"Drawdown máximo: {max_dd*100:.2f}% en {max_dd_date.date()}")
    print(f"Duración máx. bajo agua: {longest} días")
    print(f"Calmar Ratio: {calmar:.3f}")
    print(f"Sterling Ratio (peores 3 meses): {sterling:.3f}")

    return {
        'max_drawdown': float(max_dd),
        'max_drawdown_date': max_dd_date,
        'max_underwater_duration': int(longest),
        'calmar_ratio': float(calmar) if np.isfinite(calmar) else None,
        'sterling_ratio': float(sterling) if np.isfinite(sterling) else None,
        'drawdown_series': dd,
        'cumulative_returns': cum,
    }


def stress_testing(returns: pd.DataFrame, weights: np.ndarray, tickers: List[str],
                   stress_scenarios: Optional[Dict[str, Dict[str, float]]] = None):
    print("\n⚡ STRESS TESTING")
    print("=" * 30)

    if stress_scenarios is None:
        # Usa 'DEFAULT' si no hay valor para un ticker; aplica a todos.
        stress_scenarios = {
            'Crisis Financiera 2008': {'DEFAULT': -0.40},
            'COVID-19 2020':         {'DEFAULT': -0.30},
            'Crisis Deuda 2011':     {'DEFAULT': -0.25},
            'Escenario Moderado':    {'DEFAULT': -0.15},
        }

    weights = np.asarray(weights, dtype=float)
    if not np.isclose(weights.sum(), 1.0):
        weights = weights / weights.sum()

    current_value = 1_000_000  # EUR

    results = {}
    for name, shocks in stress_scenarios.items():
        shock_vec = np.array([shocks.get(tk, shocks.get('DEFAULT', 0.0)) for tk in tickers])
        impact = float(np.dot(weights, shock_vec))
        new_value = current_value * (1 + impact)
        loss = current_value - new_value
        results[name] = {
            'portfolio_impact': impact,
            'new_value': new_value,
            'loss': loss,
            'loss_pct': impact * 100,
        }
        print(f"{name}: Impacto {impact*100:.2f}% | Pérdida €{loss:,.0f} | Nuevo valor €{new_value:,.0f}")

    return results


def correlation_analysis(returns: pd.DataFrame, window: int = 252):
    print("\n🔗 ANÁLISIS DE CORRELACIÓN DINÁMICA")
    print("=" * 45)

    cols = list(returns.columns)
    if len(cols) < 2:
        raise ValueError("Se requieren ≥2 activos para correlación.")

    # Matriz y promedio
    corr_matrix = returns.corr()
    avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()

    # "Crisis": tramos de alta volatilidad media
    vol = returns.rolling(window).std()
    vol_mean = vol.mean(axis=1)
    threshold = vol_mean.quantile(0.85)
    crisis_mask = vol_mean >= threshold
    if crisis_mask.any():
        crisis_corr_matrix = returns.loc[crisis_mask].corr()
        crisis_corr = crisis_corr_matrix.values[np.triu_indices_from(crisis_corr_matrix.values, k=1)].mean()
    else:
        crisis_corr = avg_corr

    # Volatilidad de la correlación para 2 activos
    if len(cols) == 2:
        corr_series = returns[cols[0]].rolling(window).corr(returns[cols[1]]).dropna()
        corr_volatility = float(corr_series.std(ddof=0)) if not corr_series.empty else 0.0
    else:
        corr_volatility = 0.0  # simplificación para N>2

    print(f"Correlación media: {avg_corr:.3f}")
    print(f"Correlación en crisis: {crisis_corr:.3f}")
    print(f"Volatilidad de la correlación (2 activos): {corr_volatility:.3f}")

    return {
        'correlation_matrix': corr_matrix,
        'average_correlation': float(avg_corr),
        'crisis_correlation': float(crisis_corr),
        'correlation_volatility': corr_volatility,
    }


def risk_metrics_summary(returns: pd.DataFrame, weights: np.ndarray, risk_free_rate: float,
                         external_benchmark: Optional[pd.Series] = None):
    print("\n📊 RESUMEN DE MÉTRICAS DE RIESGO")
    print("=" * 45)

    pr = portfolio_series(returns, weights)

    ann_ret = pr.mean() * 252
    ann_vol = pr.std(ddof=0) * np.sqrt(252)
    sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else np.nan

    skew = pr.skew()
    kurt = pr.kurtosis()

    var_95 = np.quantile(pr, 0.05)
    var_99 = np.quantile(pr, 0.01)
    es_95 = pr[pr <= var_95].mean()
    es_99 = pr[pr <= var_99].mean()

    cum = (1 + pr).cumprod()
    dd = (cum / cum.cummax()) - 1
    max_dd = dd.min()

    downside = pr[pr < 0]
    d_vol = downside.std(ddof=0) * np.sqrt(252)
    sortino = (ann_ret - risk_free_rate) / d_vol if d_vol > 0 else np.nan

    # IR: si hay benchmark externo úsalo; si no, usa un proxy interno EW
    if external_benchmark is not None:
        df = pr.to_frame('p').join(external_benchmark.rename('b'), how='inner').dropna()
        if not df.empty:
            excess = df['p'] - df['b']
        else:
            excess = pr - returns.mean(axis=1)
    else:
        excess = pr - returns.mean(axis=1)

    tr_err = excess.std(ddof=0) * np.sqrt(252)
    info_ratio = (excess.mean() * 252) / tr_err if tr_err > 0 else np.nan

    print(f"Rend. anual: {ann_ret*100:.2f}% | Vol. anual: {ann_vol*100:.2f}%")
    print(f"Sharpe: {sharpe:.3f} | Sortino: {sortino:.3f} | IR: {info_ratio:.3f}")
    print(f"Skew: {skew:.3f} | Kurtosis: {kurt:.3f}")
    print(f"VaR95: {var_95*100:.2f}% | VaR99: {var_99*100:.2f}% | ES95: {es_95*100:.2f}% | ES99: {es_99*100:.2f}%")
    print(f"Max Drawdown: {max_dd*100:.2f}%")

    return {
        'annual_return': float(ann_ret),
        'annual_volatility': float(ann_vol),
        'sharpe_ratio': float(sharpe) if np.isfinite(sharpe) else None,
        'sortino_ratio': float(sortino) if np.isfinite(sortino) else None,
        'information_ratio': float(info_ratio) if np.isfinite(info_ratio) else None,
        'skewness': float(skew),
        'kurtosis': float(kurt),
        'var_95': float(var_95),
        'var_99': float(var_99),
        'es_95': float(es_95),
        'es_99': float(es_99),
        'max_drawdown': float(max_dd),
    }


def generate_risk_report(returns: pd.DataFrame, weights: np.ndarray, tickers: List[str]):
    print("\n📋 GENERANDO REPORTE DE RIESGO PROFESIONAL…")

    pr = portfolio_series(returns, weights)
    cum = (1 + pr).cumprod()
    dd = (cum / cum.cummax()) - 1
    dd_pct = dd * 100  # graficamos en %

    # Estilo profesional
    sns.set_palette("husl")
    colors = {
        'primary': '#1f4e79',
        'secondary': '#2e75b6',
        'accent': '#70ad47',
        'warning': '#ffc000',
        'danger': '#c5504b',
        'neutral': '#7f7f7f',
        'light': '#f2f2f2',
        'dark': '#404040'
    }

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('DASHBOARD DE GESTIÓN DE RIESGO - HEDGE FUND',
                 fontsize=24, fontweight='bold', color=colors['primary'], y=0.98)

    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3,
                          left=0.05, right=0.95, top=0.92, bottom=0.05)

    # === KPI ===
    ax_kpi = fig.add_subplot(gs[0, :])
    ax_kpi.axis('off')

    ann_ret = pr.mean() * 252
    ann_vol = pr.std(ddof=0) * np.sqrt(252)
    sharpe = (ann_ret - Rf) / ann_vol if ann_vol > 0 else np.nan
    max_dd = dd.min()
    var_95 = np.quantile(pr, 0.05)
    var_99 = np.quantile(pr, 0.01)

    kpi_data = [
        ('Rendimiento Anual', f'{ann_ret*100:.2f}%', colors['accent']),
        ('Volatilidad Anual', f'{ann_vol*100:.2f}%', colors['warning']),
        ('Sharpe Ratio', f'{sharpe:.3f}', colors['primary']),
        ('Max Drawdown', f'{max_dd*100:.2f}%', colors['danger']),
        ('VaR 95%', f'{var_95*100:.2f}%', colors['danger']),
        ('VaR 99%', f'{var_99*100:.2f}%', colors['danger'])
    ]

    for i, (label, value, color) in enumerate(kpi_data):
        x_pos = i * 0.16 + 0.02
        ax_kpi.add_patch(plt.Rectangle((x_pos, 0.1), 0.14, 0.8,
                                       facecolor=color, alpha=0.1, edgecolor=color, linewidth=2))
        ax_kpi.text(x_pos + 0.07, 0.7, label, ha='center', va='center',
                    fontsize=12, fontweight='bold', color=colors['dark'])
        ax_kpi.text(x_pos + 0.07, 0.3, value, ha='center', va='center',
                    fontsize=16, fontweight='bold', color=color)

    # === GRÁFICO 1: EVOLUCIÓN DE CARTERA CON BANDAS ===
    ax1 = fig.add_subplot(gs[1, :2])
    ax1.plot(cum.index, cum.values, linewidth=3, color=colors['primary'], label='Cartera')
    rolling_25 = cum.rolling(252).quantile(0.25)
    rolling_75 = cum.rolling(252).quantile(0.75)
    ax1.fill_between(cum.index, rolling_25, rolling_75, alpha=0.2, color=colors['primary'],
                     label='Banda de Confianza (25-75%)')
    ax1.axhline(y=1.0, color=colors['neutral'], linestyle='--', alpha=0.7, label='Valor Inicial')
    ax1.set_title('EVOLUCIÓN DE CARTERA CON BANDAS DE CONFIANZA',
                  fontsize=14, fontweight='bold', color=colors['primary'])
    ax1.set_ylabel('Valor Acumulado', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.tick_params(axis='x', rotation=45)

    # === GRÁFICO 2: DRAWDOWN AVANZADO (en %) ===
    ax2 = fig.add_subplot(gs[1, 2:])
    ax2.fill_between(dd_pct.index, dd_pct.values, 0, alpha=0.8,
                     color=colors['danger'], label='Drawdown')
    ax2.axhline(y=0, color=colors['neutral'], linestyle='-', alpha=0.5)
    max_dd_idx = dd.idxmin()
    ax2.axvline(x=max_dd_idx, color=colors['danger'], linestyle='--', alpha=0.8,
                label=f'Max DD: {dd.min()*100:.1f}%')
    ax2.set_title('ANÁLISIS DE DRAWDOWN', fontsize=14, fontweight='bold', color=colors['primary'])
    ax2.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.legend(loc='lower left', framealpha=0.9)
    ax2.tick_params(axis='x', rotation=45)

    # === GRÁFICO 3: DISTRIBUCIÓN DE RENDIMIENTOS ===
    ax3 = fig.add_subplot(gs[2, :2])
    n, bins, patches = ax3.hist(pr, bins=60, alpha=0.7, density=True,
                                color=colors['secondary'], edgecolor='white', linewidth=0.5)
    mu, sigma = pr.mean(), pr.std()
    x = np.linspace(pr.min(), pr.max(), 200)
    normal_curve = stats.norm.pdf(x, mu, sigma)
    ax3.plot(x, normal_curve, 'r-', linewidth=2, label='Distribución Normal')
    ax3.axvline(pr.mean(), color=colors['accent'], linestyle='--', linewidth=2, label='Media')
    ax3.axvline(pr.median(), color=colors['warning'], linestyle='--', linewidth=2, label='Mediana')
    p5, p95 = np.percentile(pr, [5, 95])
    ax3.axvline(p5, color=colors['danger'], linestyle=':', alpha=0.7, label='Percentil 5%')
    ax3.axvline(p95, color=colors['danger'], linestyle=':', alpha=0.7, label='Percentil 95%')
    ax3.set_title('DISTRIBUCIÓN DE RENDIMIENTOS DIARIOS',
                  fontsize=14, fontweight='bold', color=colors['primary'])
    ax3.set_xlabel('Rendimiento Diario', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Densidad', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax3.legend(loc='upper right', framealpha=0.9)

    # === GRÁFICO 4: HEATMAP DE CORRELACIÓN ===
    ax4 = fig.add_subplot(gs[2, 2:])
    if len(tickers) >= 2:
        corr_matrix = returns.corr()
        im = ax4.imshow(corr_matrix.values, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
        ax4.set_xticks(range(len(tickers)))
        ax4.set_yticks(range(len(tickers)))
        ax4.set_xticklabels(tickers, fontsize=10, fontweight='bold', rotation=45)
        ax4.set_yticklabels(tickers, fontsize=10, fontweight='bold')
        for i in range(len(tickers)):
            for j in range(len(tickers)):
                ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                         ha="center", va="center", color="black", fontweight='bold')
        cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
        cbar.set_label('Correlación', fontsize=10, fontweight='bold')
        ax4.set_title('MATRIZ DE CORRELACIÓN', fontsize=14, fontweight='bold', color=colors['primary'])
    else:
        ax4.text(0.5, 0.5, 'Correlación no disponible\n(Se requieren ≥2 activos)',
                 ha='center', va='center', transform=ax4.transAxes, fontsize=12, color=colors['neutral'])
        ax4.set_title('MATRIZ DE CORRELACIÓN', fontsize=14, fontweight='bold', color=colors['primary'])

    # === GRÁFICO 5: MÉTRICAS DE RIESGO AVANZADAS ===
    ax5 = fig.add_subplot(gs[3, :2])
    risk_metrics = ['VaR 95%', 'VaR 99%', 'ES 95%', 'ES 99%']
    # Magnitudes positivas para ser legibles
    risk_values = [-var_95*100, -var_99*100,
                   -pr[pr <= var_95].mean()*100, -pr[pr <= var_99].mean()*100]
    risk_colors = [colors['danger'], colors['danger'], colors['warning'], colors['warning']]
    bars = ax5.bar(risk_metrics, risk_values, color=risk_colors, alpha=0.8,
                   edgecolor='white', linewidth=2)
    for bar, value in zip(bars, risk_values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                 f'{value:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax5.set_title('MÉTRICAS DE RIESGO AVANZADAS', fontsize=14, fontweight='bold', color=colors['primary'])
    ax5.set_ylabel('Pérdida Potencial (%)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
    ax5.tick_params(axis='x', rotation=45)

    # === GRÁFICO 6: EXPOSICIÓN Y DIVERSIFICACIÓN ===
    ax6 = fig.add_subplot(gs[3, 2:])
    weights_arr = np.asarray(weights, dtype=float)
    if not np.isclose(weights_arr.sum(), 1.0):
        weights_arr = weights_arr / weights_arr.sum()
    # paleta para pie (repite si hay más tickers)
    base_colors = [colors['primary'], colors['secondary'], colors['accent'], colors['warning'], colors['neutral']]
    pie_colors = (base_colors * ((len(tickers) + len(base_colors) - 1) // len(base_colors)))[:len(tickers)]
    explode = [0.05] * len(tickers)
    wedges, texts, autotexts = ax6.pie(weights_arr, labels=tickers, autopct='%1.1f%%',
                                       startangle=90, colors=pie_colors,
                                       explode=explode, shadow=True,
                                       textprops={'fontweight': 'bold', 'fontsize': 12})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    ax6.set_title('EXPOSICIÓN POR ACTIVO', fontsize=14, fontweight='bold', color=colors['primary'])

    # === INFO ===
    info_text = f"""
    📊 ANÁLISIS REALIZADO: {cum.index[0].strftime('%d/%m/%Y')} - {cum.index[-1].strftime('%d/%m/%Y')}
    📈 OBSERVACIONES: {len(pr):,} días | 📅 PERÍODO: {(cum.index[-1] - cum.index[0]).days} días
    💼 CARTERA: {', '.join(tickers)} | ⚖️ PESOS: {dict(zip(tickers, np.round(weights_arr, 3)))}
    """
    fig.text(0.05, 0.02, info_text, fontsize=10, color=colors['neutral'],
             bbox=dict(boxstyle="round,pad=0.5", facecolor=colors['light'], alpha=0.8))

    # Guardar con alta calidad
    out_dir = (Path(__file__).resolve().parent / 'png') if '__file__' in globals() else (Path.cwd() / 'png')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / 'reporte_riesgo_hedge_fund_profesional.png'
    out_pdf = out_dir / 'reporte_riesgo_hedge_fund_profesional.pdf'
    plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"📊 Reporte profesional guardado como:\n  - {out_png}\n  - {out_pdf}")

    return fig, out_png


# ----------------------
# Ejecución principal
# ----------------------

def main():
    print("🏦 MÓDULO DE GESTIÓN DE RIESGO PARA HEDGE FUNDS")
    print("=" * 60)

    print("\n" + "="*60)
    print("INICIANDO ANÁLISIS COMPLETO DE RIESGO")
    print("="*60)

    prices, rets, fx = prepare_data()

    weights_norm = PORTFOLIO_WEIGHTS / PORTFOLIO_WEIGHTS.sum()
    print(f"📊 Pesos de cartera: {dict(zip(TICKERS, np.round(weights_norm, 2)))}")

    # --- Tracking Error vs MSCI ---
    bench_rets = prepare_benchmark_returns(fx)
    pr = portfolio_series(rets, PORTFOLIO_WEIGHTS)
    te_res = tracking_error(pr, bench_rets, ddof=1)
    print(f"📐 Tracking Error anual: {te_res['te_annual']*100:.2f}% | diario: {te_res['te_daily']*100:.2f}% | Information Ratio: {te_res['information_ratio']:.3f}")
    roll_te_63 = rolling_tracking_error(pr, bench_rets, window=63, ddof=1)
    if not roll_te_63.empty:
        print(f"📈 Rolling TE 63d (último): {roll_te_63.iloc[-1]*100:.2f}%")

    # 1. VaR y ES
    _ = calculate_var_es(rets, PORTFOLIO_WEIGHTS)

    # 2. Drawdown
    _ = calculate_drawdown_analysis(rets, PORTFOLIO_WEIGHTS)

    # 3. Stress Testing
    _ = stress_testing(rets, PORTFOLIO_WEIGHTS, TICKERS)

    # 4. Correlación
    _ = correlation_analysis(rets)

    # 5. Métricas (IR contra benchmark externo)
    _ = risk_metrics_summary(rets, PORTFOLIO_WEIGHTS, Rf, external_benchmark=bench_rets)

    # 6. Reporte visual
    _, out_path = generate_risk_report(rets, PORTFOLIO_WEIGHTS, TICKERS)

    print("\n" + "="*60)
    print("✅ ANÁLISIS DE RIESGO COMPLETADO")
    print("="*60)

    # Reglas simples de alerta (ejemplo)
    print("\n🎯 RECOMENDACIONES DE RIESGO:")
    print("-" * 40)
    alerts = []
    if te_res['te_annual'] > 0.08:  # TE anual > 8%
        alerts.append("• TE > 8%: revisar desalineación con benchmark y límites de tracking.")
    if roll_te_63.notna().tail(1).gt(0.10).any():
        alerts.append("• Rolling TE 63d > 10%: volatilidad relativa elevada en el trimestre.")
    dd_analysis = calculate_drawdown_analysis(rets, PORTFOLIO_WEIGHTS)
    if dd_analysis['max_drawdown'] < -0.25:
        alerts.append("• Max Drawdown < -25%: considerar coberturas o reducción de riesgo.")
    if not alerts:
        print("• Sin alertas críticas según umbrales de ejemplo.")
    else:
        for a in alerts:
            print(a)


if __name__ == "__main__":
    main()

