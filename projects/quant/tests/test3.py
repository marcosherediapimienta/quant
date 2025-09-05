"""
Optimizador de Cartera de Hedge Funds 
- Descarga datos con yfinance
- Convierte divisa (USD→EUR) cuando corresponda
- Calcula métricas anuales (rendimiento, volatilidad, Sharpe)
- Optimiza: Máx. Sharpe, Mín. Varianza, Risk Parity (contribuciones iguales),
  Cartera con restricciones, y Black–Litterman (simplificado pero consistente)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")  

from scipy.optimize import minimize

# =============================
# Parámetros del usuario
# =============================
TICKERS = ["0P0001CLDK.F", "AMZN", "GLD", "BTC-EUR"]
FECHA_INICIO = "2018-03-20"
FECHA_FIN = "2025-09-02"
FX_TICKER = "EURUSD=X"  # USD por 1 EUR

# Defínelo a tu gusto (anual):
RISK_FREE = 0.045
MONTHLY_CONTRIB = 300.0

# ¿Qué tickers vienen en USD? (para convertirlos a EUR)
USD_TICKERS = {"AMZN", "GLD"}

print("📦 OPTIMIZADOR DE CARTERA DE HEDGE FUNDS")
print("=" * 60)

# =============================
# Descarga de datos
# =============================
print("📈 Descargando precios...")
market = yf.download(TICKERS, start=FECHA_INICIO, end=FECHA_FIN,
                     auto_adjust=False, progress=False, group_by="ticker")
fx = yf.download(FX_TICKER, start=FECHA_INICIO, end=FECHA_FIN,
                 auto_adjust=False, progress=False)

# Helpers -------------------------------------------------------------

def get_adj_close_series(df_grouped_by_ticker: pd.DataFrame, ticker: str) -> pd.Series | None:
    """Extrae la serie 'Adj Close' para un ticker cuando group_by='ticker'.
    Devuelve SIEMPRE una Serie 1D o None.
    """
    # Caso MultiIndex (cuando hay varios tickers)
    if isinstance(df_grouped_by_ticker.columns, pd.MultiIndex):
        col = (ticker, "Adj Close")
        if col in df_grouped_by_ticker.columns:
            s = df_grouped_by_ticker[col].dropna().sort_index()
            if isinstance(s, pd.DataFrame):
                s = s.squeeze()
            return s if isinstance(s, pd.Series) else None
        return None
    # Caso columnas simples (raro con group_by='ticker', pero por robustez)
    if "Adj Close" in df_grouped_by_ticker.columns:
        s = df_grouped_by_ticker["Adj Close"].dropna().sort_index()
        if isinstance(s, pd.DataFrame):
            s = s.squeeze()
        return s if isinstance(s, pd.Series) else None
    return None


def convert_usd_to_eur(price_series: pd.Series, fx_df: pd.DataFrame) -> pd.Series:
    """Convierte una serie de precios en USD a EUR usando EURUSD=X (USD por EUR).
    EUR = USD / (USD/EUR). Se alinea por fecha y ffill.
    """
    if price_series is None or fx_df is None or fx_df.empty:
        return price_series
    # Soporta tanto columnas simples como MultiIndex
    if isinstance(fx_df.columns, pd.MultiIndex):
        fx_col = ("Adj Close", FX_TICKER) if ("Adj Close", FX_TICKER) in fx_df.columns else None
    else:
        fx_col = "Adj Close" if "Adj Close" in fx_df.columns else None
    if fx_col is None:
        return price_series
    fx_rate = fx_df[fx_col].dropna()
    if isinstance(fx_rate, pd.DataFrame):
        fx_rate = fx_rate.squeeze()
    aligned_fx = fx_rate.reindex(price_series.index, method="ffill")
    # Fuerza a Serie 1D
    if isinstance(price_series, pd.DataFrame):
        price_series = price_series.squeeze()
    return price_series / aligned_fx
    fx_col = "Adj Close" if "Adj Close" in fx_df.columns else None
    if fx_col is None:
        return price_series
    fx_rate = fx_df[fx_col].dropna()  # USD por EUR
    aligned_fx = fx_rate.reindex(price_series.index, method="ffill")
    return price_series / aligned_fx

# Procesa series y conversiones --------------------------------------
processed = {}
for t in TICKERS:
    s = get_adj_close_series(market, t)
    if s is None or s.empty:
        print(f"⚠️  No hay 'Adj Close' para {t}. Se omite.")
        continue
    if t in USD_TICKERS:
        s_eur = convert_usd_to_eur(s, fx)
        processed[t] = s_eur
    else:
        processed[t] = s

if not processed:
    raise SystemExit("❌ No se pudo obtener ninguna serie de precios.")

# Rendimientos logarítmicos diarios ----------------------------------
# Construimos el panel de precios de forma segura para evitar objetos 2D inesperados
prices_df = pd.concat({t: (s.squeeze() if isinstance(s, (pd.Series, pd.DataFrame)) else pd.Series(s))
                       for t, s in processed.items()}, axis=1)
# Asegura nombres de columnas planos
prices_df.columns = list(prices_df.columns.get_level_values(0)) if isinstance(prices_df.columns, pd.MultiIndex) else prices_df.columns
# Convierte a numérico por si hay basura
prices_df = prices_df.apply(pd.to_numeric, errors='coerce')
# Calcula log-retornos
returns = np.log(prices_df).diff().dropna(how="any")
assets = list(returns.columns)

print(f"📊 Datos procesados: {len(returns):,} observaciones")
print(f"📅 Período: {returns.index[0].date()} a {returns.index[-1].date()}")

# Métricas básicas (anualizadas) -------------------------------------
mean_ret = returns.mean() * 252.0  # serie (por activo)
cov = returns.cov() * 252.0        # DataFrame
vol = pd.Series(np.sqrt(np.diag(cov)), index=cov.index)

print("\n📈 MÉTRICAS BÁSICAS:")
for a in assets:
    sr = (mean_ret[a] - RISK_FREE) / vol[a]
    print(f"  {a}:")
    print(f"    Rendimiento anual: {mean_ret[a]*100:6.2f}%")
    print(f"    Volatilidad anual: {vol[a]*100:6.2f}%")
    print(f"    Sharpe (rf={RISK_FREE*100:.1f}%): {sr:6.3f}")

# ==========================================================
# Funciones de cartera y objetivos
# ==========================================================

def portfolio_performance(w: np.ndarray, mu: pd.Series, sigma: pd.DataFrame):
    """Retorno y volatilidad anualizados de la cartera."""
    r = float(np.dot(w, mu.values))
    v = float(np.sqrt(w @ sigma.values @ w))
    return r, v


def negative_sharpe(w: np.ndarray, mu: pd.Series, sigma: pd.DataFrame, rf: float) -> float:
    r, v = portfolio_performance(w, mu, sigma)
    return -((r - rf) / v)


def portfolio_variance(w: np.ndarray, sigma: pd.DataFrame) -> float:
    return float(w @ sigma.values @ w)


def risk_parity_objective(w: np.ndarray, sigma: pd.DataFrame) -> float:
    """Minimiza la desviación de las contribuciones porcentuales al riesgo respecto a 1/N."""
    w = np.asarray(w)
    port_vol = np.sqrt(w @ sigma.values @ w)
    # contribución marginal: (Σ w)_i
    marginal = sigma.values @ w
    # contribución absoluta al riesgo: w_i * marginal_i / port_vol
    abs_contrib = w * marginal / port_vol
    # contribución porcentual (suma = 1)
    pct_contrib = abs_contrib / port_vol
    target = np.ones_like(w) / w.size
    return float(np.sum((pct_contrib - target) ** 2))

# ==========================================================
# Restricciones comunes
# ==========================================================
N = len(assets)
if N < 1:
    raise SystemExit("❌ Sin activos tras limpieza de datos.")

x0 = np.full(N, 1.0 / N)
cons = ({"type": "eq", "fun": lambda x: np.sum(x) - 1.0},)
bnds = tuple((0.0, 1.0) for _ in range(N))

# ==========================================================
# 1) Máximo Sharpe
# ==========================================================
print("\n🚀 1) OPTIMIZACIÓN DE MARKOWITZ — Máx. Sharpe")
res_max_sharpe = minimize(negative_sharpe, x0=x0, args=(mean_ret[assets], cov.loc[assets, assets], RISK_FREE),
                          method="SLSQP", bounds=bnds, constraints=cons)
if not res_max_sharpe.success:
    print(f"⚠️  Máx. Sharpe no convergió: {res_max_sharpe.message}")
w_ms = res_max_sharpe.x
ret_ms, vol_ms = portfolio_performance(w_ms, mean_ret[assets], cov.loc[assets, assets])
shp_ms = (ret_ms - RISK_FREE) / vol_ms
print("  🏆 Pesos Máx. Sharpe:")
for a, w in zip(assets, w_ms):
    print(f"    {a}: {w*100:5.1f}%")
print(f"    Rendimiento: {ret_ms*100:6.2f}% | Volatilidad: {vol_ms*100:6.2f}% | Sharpe: {shp_ms:6.3f}")

# ==========================================================
# 1b) Mínima Varianza
# ==========================================================
print("\n🛡️ 1b) OPTIMIZACIÓN — Mínima Varianza")
res_min_var = minimize(portfolio_variance, x0=x0, args=(cov.loc[assets, assets],),
                       method="SLSQP", bounds=bnds, constraints=cons)
if not res_min_var.success:
    print(f"⚠️  Mínima Varianza no convergió: {res_min_var.message}")
w_mv = res_min_var.x
ret_mv, vol_mv = portfolio_performance(w_mv, mean_ret[assets], cov.loc[assets, assets])
print("  🔧 Pesos Mín. Varianza:")
for a, w in zip(assets, w_mv):
    print(f"    {a}: {w*100:5.1f}%")
print(f"    Rendimiento: {ret_mv*100:6.2f}% | Volatilidad: {vol_mv*100:6.2f}%")

# ==========================================================
# 2) Risk Parity (con contribuciones iguales)
# ==========================================================
print("\n⚖️ 2) OPTIMIZACIÓN — Risk Parity (ERC)")
res_rp = minimize(risk_parity_objective, x0=x0, args=(cov.loc[assets, assets],),
                  method="SLSQP", bounds=bnds, constraints=cons)
if not res_rp.success:
    print(f"⚠️  Risk Parity no convergió: {res_rp.message}")
w_rp = res_rp.x
ret_rp, vol_rp = portfolio_performance(w_rp, mean_ret[assets], cov.loc[assets, assets])
print("  ⚖️ Pesos Risk Parity:")
for a, w in zip(assets, w_rp):
    print(f"    {a}: {w*100:5.1f}%")
print(f"    Rendimiento: {ret_rp*100:6.2f}% | Volatilidad: {vol_rp*100:6.2f}%")

# ==========================================================
# 3) Restricciones avanzadas (ej. 10%–80% por activo)
# ==========================================================
print("\n🎯 3) OPTIMIZACIÓN — Restricciones (10%–80% por activo)")
bnds_strict = tuple((0.10, 0.80) for _ in range(N))
res_constr = minimize(negative_sharpe, x0=x0, args=(mean_ret[assets], cov.loc[assets, assets], RISK_FREE),
                      method="SLSQP", bounds=bnds_strict, constraints=cons)
if not res_constr.success:
    print(f"⚠️  Cartera con restricciones no convergió: {res_constr.message}")
w_cs = res_constr.x
ret_cs, vol_cs = portfolio_performance(w_cs, mean_ret[assets], cov.loc[assets, assets])
shp_cs = (ret_cs - RISK_FREE) / vol_cs
print("  🔒 Pesos restringidos:")
for a, w in zip(assets, w_cs):
    print(f"    {a}: {w*100:5.1f}%")
print(f"    Rendimiento: {ret_cs*100:6.2f}% | Volatilidad: {vol_cs*100:6.2f}% | Sharpe: {shp_cs:6.3f}")

# ==========================================================
# 4) Black–Litterman (simplificado)
# ==========================================================
print("\n🧠 4) OPTIMIZACIÓN — Black–Litterman (simplificado)")

# (opcional) Pesos de mercado (si los conoces, ponlos aquí; deben sumar 1)
# Por ejemplo: MARKET_WEIGHTS = {"0P0001CLDK.F": 0.5, "EEM": 0.5}
MARKET_WEIGHTS: dict[str, float] | None = None

# Parámetros BL
TAU = 0.05
DELTA = 2.5  # aversión al riesgo típica (aprox)

# Views opcionales: diccionario activo->retorno esperado anual (en el mismo scale)
# Si None, usamos solo el equilibrio.
VIEWS: dict[str, float] | None = None

sigma = cov.loc[assets, assets].values
mu_series = mean_ret[assets]

# Pi de equilibrio
if MARKET_WEIGHTS is not None:
    w_mkt = np.array([MARKET_WEIGHTS.get(a, 0.0) for a in assets])
    w_mkt = w_mkt / w_mkt.sum() if w_mkt.sum() > 0 else np.full(N, 1.0 / N)
    Pi = DELTA * sigma @ w_mkt  # retorno implícito
else:
    Pi = mu_series.values  # fallback explícito
    print("  ℹ️  Sin pesos de mercado: Pi=media histórica (fallback)")

# Matriz de views P y vector Q
if VIEWS is None:
    P = np.eye(N)
    Q = Pi.copy()
else:
    P = []
    Q = []
    for i, a in enumerate(assets):
        if a in VIEWS:
            row = np.zeros(N)
            row[i] = 1.0
            P.append(row)
            Q.append(VIEWS[a])
    P = np.array(P) if len(P) > 0 else np.eye(N)
    Q = np.array(Q) if len(Q) > 0 else Pi.copy()

# Incertidumbre de views Ω: diagonal proporcional a la varianza
Omega = np.diag(np.diag(sigma)) * 0.25  # confianza moderada

# Posterior BL
# mu_bl = [ (τΣ)^−1 + P'Ω^−1P ]^−1 [ (τΣ)^−1 Π + P'Ω^−1 Q ]
# Sigma_bl = [ (τΣ)^−1 + P'Ω^−1P ]^−1
from numpy.linalg import inv
A = inv(TAU * sigma)
middle = A + P.T @ inv(Omega) @ P
Sigma_bl = inv(middle)
mu_bl = Sigma_bl @ (A @ Pi + P.T @ inv(Omega) @ Q)

mu_bl_series = pd.Series(mu_bl, index=assets)
Sigma_bl_df = pd.DataFrame(Sigma_bl, index=assets, columns=assets)

res_bl = minimize(negative_sharpe, x0=x0, args=(mu_bl_series, Sigma_bl_df, RISK_FREE),
                  method="SLSQP", bounds=bnds, constraints=cons)
if not res_bl.success:
    print(f"⚠️  Black–Litterman no convergió: {res_bl.message}")

w_bl = res_bl.x
ret_bl, vol_bl = portfolio_performance(w_bl, mu_bl_series, Sigma_bl_df)
shp_bl = (ret_bl - RISK_FREE) / vol_bl

print("  🧠 Pesos BL:")
for a, w in zip(assets, w_bl):
    print(f"    {a}: {w*100:5.1f}%")
print(f"    Rendimiento: {ret_bl*100:6.2f}% | Volatilidad: {vol_bl*100:6.2f}% | Sharpe: {shp_bl:6.3f}")

# ==========================================================
# 5) Comparación de estrategias
# ==========================================================
print("\n📊 5) COMPARACIÓN DE ESTRATEGIAS")
strategies = {
    "Maximum Sharpe": w_ms,
    "Minimum Variance": w_mv,
    "Risk Parity": w_rp,
    "Constrained": w_cs,
    "Black–Litterman": w_bl,
}

def describe_strategy(name: str, w: np.ndarray, mu: pd.Series, sigma: pd.DataFrame):
    r, v = portfolio_performance(w, mu, sigma)
    s = (r - RISK_FREE) / v
    print(f"\n{name}:")
    for a, wi in zip(assets, w):
        print(f"  {a}: {wi*100:5.1f}%")
    print(f"  Rendimiento: {r*100:6.2f}% | Volatilidad: {v*100:6.2f}% | Sharpe: {s:6.3f}")

for name, w in strategies.items():
    # Para coherencia, medimos contra la covarianza histórica (misma métrica)
    describe_strategy(name, w, mean_ret[assets], cov.loc[assets, assets])

# ==========================================================
# 6) Recomendación final (mejor Sharpe sobre cov histórica)
# ==========================================================
print("\n🏆 6) RECOMENDACIÓN FINAL")

best_name, best_w, best_r, best_v, best_s = None, None, -9e9, 0.0, -9e9
for name, w in strategies.items():
    r, v = portfolio_performance(w, mean_ret[assets], cov.loc[assets, assets])
    s = (r - RISK_FREE) / v
    if s > best_s:
        best_name, best_w, best_r, best_v, best_s = name, w, r, v, s

print(f"Estrategia recomendada: {best_name}")
print("\n💶 APORTACIONES RECOMENDADAS (300 EUR/mes total):")
for a, w in zip(assets, best_w):
    aport = MONTHLY_CONTRIB * w
    print(f"  {a}: {aport:7.0f} EUR/mes ({w*100:5.1f}%)")

print("\n📈 Proyección (con métricas históricas):")
print(f"  Rendimiento esperado: {best_r*100:6.2f}% anual")
print(f"  Volatilidad esperada: {best_v*100:6.2f}% anual")
print(f"  Sharpe Ratio (rf={RISK_FREE*100:.1f}%): {best_s:6.3f}")

print("\n🎉 Optimización completada.")

