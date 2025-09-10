"""
Optimizador de Cartera (Markowitz, ERC, BL, Restricciones, Frontera Eficiente, TE)
- Descarga precios con yfinance
- Convierte USD→EUR (EURUSD=X = USD por 1 EUR) cuando corresponde
- Cálculo de métricas anualizadas
- Optimiza: Máx. Sharpe, Mín. Var, ERC, Restricciones y Black–Litterman (μ posterior con Σ histórica)
- Frontera eficiente (min-var para retornos objetivo)
- Tracking Error frente a benchmark opcional (cálculo y restricción)
- Covarianza Ledoit–Wolf si está disponible (fallback a shrinkage simple)
- Exporta gráficos a ./png
Requisitos:
    pip install yfinance pandas numpy scipy
    (opcional) pip install scikit-learn  # para Ledoit–Wolf
"""

import warnings
warnings.filterwarnings("ignore")

from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import minimize

# =============================
# Parámetros del usuario
# =============================
TICKERS = ["0P0001CLDK.F", "AMZN", "GLD", "BTC-EUR"]
FECHA_INICIO = "2018-03-20"
FECHA_FIN = "2025-09-02"

FX_TICKER = "EURUSD=X"      # USD por 1 EUR
USD_TICKERS = {"AMZN", "GLD"}

RISK_FREE = 0.045           # anual
MONTHLY_CONTRIB = 300.0     # EUR/mes totales

# Covarianza: usar Ledoit–Wolf si hay sklearn
USE_LEDOIT_WOLF = True
# Si no hay sklearn o se desactiva, aplicar shrinkage simple hacia la diagonal:
SIMPLE_SHRINKAGE_LAMBDA = 0.05  # 0.0 para desactivar

# ====== Restricciones por activo (ejemplos realistas) ======
# Si un ticker no aparece en este dict, se usa (0,1) por defecto.
PER_ASSET_BOUNDS: Dict[str, Tuple[float, float]] = {
    "BTC-EUR": (0.00, 0.15),     # techo al cripto
    "GLD": (0.00, 0.40),         # oro hasta 40%
    "AMZN": (0.00, 0.35),        # single-stock hasta 35%
    # "0P0001CLDK.F": (0.10, 0.80),  # ejemplo para un fondo
}

# ====== Restricciones por grupos (opcional) ======
# Define grupos y cotas sobre la suma de pesos del grupo.
GROUPS: Dict[str, List[str]] = {
    "Renta_Variable": ["AMZN", "0P0001CLDK.F"],
    "Alternativos": ["GLD", "BTC-EUR"],
}
GROUP_MAX: Dict[str, float] = {
    "Alternativos": 0.60,     # máx 60% en alternativos
}
GROUP_MIN: Dict[str, float] = {
    # "Renta_Variable": 0.30, # ej: al menos 30% en RV
}

# ====== Benchmark y Tracking Error (opcional) ======
# Define un benchmark como pesos sobre los MISMO activos (suma=1).
# Si None, no se calcula TE.
BENCH_WEIGHTS: Optional[Dict[str, float]] = {
    # ejemplo: 60/40 "proxy" dentro del universo actual
    # "0P0001CLDK.F": 0.60, "GLD": 0.40
    # O usa un 25/25/25/25:
    # a:0.25 para todos los activos presentes
}
# Si quieres restringir TE respecto al benchmark: establece un máximo (anual, en sigma)
TE_MAX: Optional[float] = None  # ej: 0.06 significa TE anual ≤ 6%

# =============================
# Helpers de datos
# =============================
def get_price_series(df: pd.DataFrame, ticker: Optional[str] = None) -> Optional[pd.Series]:
    """
    Devuelve la serie de precios priorizando 'Adj Close' y, si no existe, 'Close'.
    Soporta DataFrames de yfinance con MultiIndex en cualquiera de las orientaciones.
    """
    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        candidates = []
        if ticker is not None:
            candidates += [(ticker, "Adj Close"), (ticker, "Close"),
                           ("Adj Close", ticker), ("Close", ticker)]
        else:
            for col0 in ("Adj Close", "Close"):
                for tk in df.columns.get_level_values(0).unique():
                    candidates.append((tk, col0))
                for tk in df.columns.get_level_values(1).unique():
                    candidates.append((col0, tk))
        for col in candidates:
            if col in df.columns:
                s = df[col]
                if isinstance(s, pd.DataFrame):
                    s = s.squeeze()
                return s.dropna().sort_index()
        return None

    for c in ("Adj Close", "Close"):
        if c in df.columns:
            s = df[c]
            if isinstance(s, pd.DataFrame):
                s = s.squeeze()
            return s.dropna().sort_index()
    return None


def convert_usd_to_eur(price_series: Optional[pd.Series],
                       fx_series: Optional[pd.Series]) -> Optional[pd.Series]:
    """
    Convierte USD→EUR con EURUSD=X (USD por 1 EUR): EUR = USD / (USD/EUR).
    """
    if price_series is None or fx_series is None or fx_series.empty:
        return price_series
    aligned_fx = fx_series.reindex(price_series.index).ffill().bfill()
    return price_series / aligned_fx


def shrink_covariance_simple(cov: pd.DataFrame, lam: float) -> pd.DataFrame:
    """Shrinkage simple hacia la diagonal: (1-λ)Σ + λ*diag(Σ)."""
    lam = float(lam)
    if lam <= 0:
        return cov
    d = np.diag(np.diag(cov.values))
    s = (1 - lam) * cov.values + lam * d
    s = (s + s.T) / 2.0
    return pd.DataFrame(s, index=cov.index, columns=cov.columns)


def weights_from_dict(assets: List[str], wdict: Dict[str, float]) -> np.ndarray:
    w = np.array([wdict.get(a, 0.0) for a in assets], dtype=float)
    s = w.sum()
    if s > 0:
        w = w / s
    return w


# =============================
# Funciones de cartera
# =============================
def portfolio_performance(w: np.ndarray,
                          mu: pd.Series,
                          sigma: pd.DataFrame) -> Tuple[float, float]:
    r = float(np.dot(w, mu.values))
    v = float(np.sqrt(max(w @ sigma.values @ w, 0.0)))
    return r, v


def negative_sharpe(w: np.ndarray,
                    mu: pd.Series,
                    sigma: pd.DataFrame,
                    rf: float) -> float:
    r, v = portfolio_performance(w, mu, sigma)
    return -((r - rf) / (v + 1e-12))


def portfolio_variance(w: np.ndarray, sigma: pd.DataFrame) -> float:
    return float(max(w @ sigma.values @ w, 0.0))


def risk_parity_objective(w: np.ndarray, sigma: pd.DataFrame) -> float:
    """
    ERC: minimiza la desviación de las contribuciones porcentuales al riesgo respecto a 1/N.
    """
    w = np.asarray(w)
    port_vol = np.sqrt(max(w @ sigma.values @ w, 1e-18))
    marginal = sigma.values @ w
    abs_contrib = w * marginal / (port_vol + 1e-12)
    pct_contrib = abs_contrib / (port_vol + 1e-12)  # suma ~ 1
    target = np.ones_like(w) / w.size
    return float(np.sum((pct_contrib - target) ** 2))


def risk_contributions(w: np.ndarray, sigma: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Devuelve (contribuciones absolutas, contribuciones porcentuales)."""
    port_vol = np.sqrt(max(w @ sigma.values @ w, 1e-18))
    marginal = sigma.values @ w
    abs_contrib = w * marginal / (port_vol + 1e-12)
    pct_contrib = abs_contrib / (port_vol + 1e-12)
    return abs_contrib, pct_contrib


def tracking_error_annual(w: np.ndarray,
                          w_bench: np.ndarray,
                          sigma: pd.DataFrame) -> float:
    diff = w - w_bench
    var_te = float(max(diff @ sigma.values @ diff, 0.0))
    return float(np.sqrt(var_te))


def te_constraint_ineq(w: np.ndarray,
                       w_bench: np.ndarray,
                       sigma: pd.DataFrame,
                       te_max: float) -> float:
    """Devuelve >=0 si se cumple: TE_max^2 - (w-wb)'Σ(w-wb) ≥ 0"""
    diff = w - w_bench
    return float(te_max ** 2 - max(diff @ sigma.values @ diff, 0.0))


def per_asset_bounds(assets: List[str],
                     bounds_dict: Dict[str, Tuple[float, float]]) -> Tuple[Tuple[float, float], ...]:
    return tuple(bounds_dict.get(a, (0.0, 1.0)) for a in assets)


def build_group_constraints(assets: List[str],
                            groups: Dict[str, List[str]],
                            gmax: Optional[Dict[str, float]] = None,
                            gmin: Optional[Dict[str, float]] = None) -> List[dict]:
    cons = []
    name_to_idx = {name: [assets.index(a) for a in members if a in assets] for name, members in groups.items()}
    if gmax:
        for name, cap in gmax.items():
            idx = name_to_idx.get(name, [])
            if idx:
                cons.append({"type": "ineq", "fun": lambda w, idx=idx, cap=cap: cap - np.sum(w[idx])})
    if gmin:
        for name, floor in gmin.items():
            idx = name_to_idx.get(name, [])
            if idx:
                cons.append({"type": "ineq", "fun": lambda w, idx=idx, floor=floor: np.sum(w[idx]) - floor})
    return cons


# =============================
# Descarga de datos
# =============================
print("📦 OPTIMIZADOR DE CARTERA — versión completa")
print("=" * 70)
print("📈 Descargando precios...")
market = yf.download(TICKERS, start=FECHA_INICIO, end=FECHA_FIN,
                     auto_adjust=False, progress=False, group_by="ticker")
fx_df = yf.download(FX_TICKER, start=FECHA_INICIO, end=FECHA_FIN,
                    auto_adjust=False, progress=False)

# Serie FX (EURUSD=X) -> USD por 1 EUR
fx_series = get_price_series(fx_df, FX_TICKER)
if fx_series is None:
    print("⚠️  No fue posible obtener EURUSD=X; los activos en USD quedarán en USD.")

# Procesa series de precios y aplica conversión cuando corresponda
processed: Dict[str, pd.Series] = {}
for t in TICKERS:
    s = get_price_series(market, t)
    if s is None or s.empty:
        print(f"⚠️  No hay precio para {t}. Se omite.")
        continue
    if t in USD_TICKERS and fx_series is not None:
        s = convert_usd_to_eur(s, fx_series)
        print(f"💱 Convertido {t} de USD a EUR con {FX_TICKER}.")
    processed[t] = s

if not processed:
    raise SystemExit("❌ No se pudo obtener ninguna serie de precios.")

# Panel de precios y retornos (intersección de fechas)
prices_df = pd.concat(processed, axis=1).sort_index()
returns = np.log(prices_df).diff().dropna(how="any")
assets = list(returns.columns)

if not assets:
    raise SystemExit("❌ Sin activos tras limpiar fechas y NaN.")

print(f"📊 Observaciones: {len(returns):,}")
print(f"📅 Período: {returns.index[0].date()} → {returns.index[-1].date()}")
print(f"✅ Activos usados: {', '.join(assets)}")

# =============================
# Métricas anualizadas y Σ
# =============================
mean_ret = returns.mean() * 252.0

cov_daily = None
if USE_LEDOIT_WOLF:
    try:
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf().fit(returns.values)
        cov_daily = pd.DataFrame(lw.covariance_, index=assets, columns=assets)
        print("🧪 Σ diaria por Ledoit–Wolf (sklearn).")
    except Exception as e:
        print(f"⚠️  Ledoit–Wolf no disponible ({e}). Usando muestra con shrinkage simple λ={SIMPLE_SHRINKAGE_LAMBDA}.")
        cov_daily = returns.cov()
else:
    cov_daily = returns.cov()
    if SIMPLE_SHRINKAGE_LAMBDA > 0:
        print(f"🧪 Σ diaria muestral con shrinkage simple λ={SIMPLE_SHRINKAGE_LAMBDA}.")
    else:
        print("🧪 Σ diaria muestral (sin shrinkage).")

cov = cov_daily * 252.0
if not USE_LEDOIT_WOLF and SIMPLE_SHRINKAGE_LAMBDA > 0:
    cov = shrink_covariance_simple(cov, SIMPLE_SHRINKAGE_LAMBDA)

vol = pd.Series(np.sqrt(np.diag(cov.values)), index=cov.index)

print("\n📈 MÉTRICAS BÁSICAS (anualizadas):")
for a in assets:
    s = (mean_ret[a] - RISK_FREE) / (vol[a] + 1e-12)
    print(f"  {a}: Ret {mean_ret[a]*100:6.2f}% | Vol {vol[a]*100:6.2f}% | Sharpe {s:6.3f}")

# =============================
# Optimización base y restricciones
# =============================
N = len(assets)
x0 = np.full(N, 1.0 / N)
sum_to_one = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
bnds = per_asset_bounds(assets, PER_ASSET_BOUNDS)
group_cons = build_group_constraints(assets, GROUPS, GROUP_MAX, GROUP_MIN)

opt_base = dict(method="SLSQP", bounds=bnds, constraints=(sum_to_one, *group_cons),
                options={"maxiter": 500, "ftol": 1e-9})

# Benchmark y TE
w_bench = None
if BENCH_WEIGHTS:
    w_bench = weights_from_dict(assets, BENCH_WEIGHTS)
    print("\n📏 Benchmark definido (sobre los mismos activos).")
    print("   w_bench:", {a: round(w, 4) for a, w in zip(assets, w_bench)})
    if TE_MAX is not None:
        print(f"   Restricción de TE: TE ≤ {TE_MAX:.2%} (anual).")

# =============================
# 1) Máximo Sharpe
# =============================
print("\n🚀 1) Máximo Sharpe")
cons_ms = [sum_to_one, *group_cons]
if w_bench is not None and TE_MAX is not None:
    cons_ms.append({"type": "ineq",
                    "fun": lambda w, wb=w_bench, S=cov, te=TE_MAX: te_constraint_ineq(w, wb, S, te)})

res_ms = minimize(negative_sharpe, x0=x0,
                  args=(mean_ret[assets], cov.loc[assets, assets], RISK_FREE),
                  method="SLSQP", bounds=bnds, constraints=cons_ms,
                  options={"maxiter": 500, "ftol": 1e-9})
if not res_ms.success:
    print(f"⚠️  Máx. Sharpe no convergió: {res_ms.message}")
w_ms = res_ms.x
ret_ms, vol_ms = portfolio_performance(w_ms, mean_ret[assets], cov.loc[assets, assets])
shp_ms = (ret_ms - RISK_FREE) / (vol_ms + 1e-12)
print("  Pesos MS:", {a: f"{w*100:5.1f}%" for a, w in zip(assets, w_ms)})
print(f"  Ret {ret_ms*100:6.2f}% | Vol {vol_ms*100:6.2f}% | Sharpe {shp_ms:6.3f}")
if w_bench is not None:
    te_ms = tracking_error_annual(w_ms, w_bench, cov.loc[assets, assets])
    print(f"  TE vs bench: {te_ms*100:5.2f}%")

# =============================
# 1b) Mínima Varianza
# =============================
print("\n🛡️ 1b) Mínima Varianza")
cons_mv = [sum_to_one, *group_cons]
if w_bench is not None and TE_MAX is not None:
    cons_mv.append({"type": "ineq",
                    "fun": lambda w, wb=w_bench, S=cov, te=TE_MAX: te_constraint_ineq(w, wb, S, te)})

res_mv = minimize(portfolio_variance, x0=x0,
                  args=(cov.loc[assets, assets],),
                  method="SLSQP", bounds=bnds, constraints=cons_mv,
                  options={"maxiter": 500, "ftol": 1e-9})
if not res_mv.success:
    print(f"⚠️  Mín. Var no convergió: {res_mv.message}")
w_mv = res_mv.x
ret_mv, vol_mv = portfolio_performance(w_mv, mean_ret[assets], cov.loc[assets, assets])
print("  Pesos MV:", {a: f"{w*100:5.1f}%" for a, w in zip(assets, w_mv)})
print(f"  Ret {ret_mv*100:6.2f}% | Vol {vol_mv*100:6.2f}%")
if w_bench is not None:
    te_mv = tracking_error_annual(w_mv, w_bench, cov.loc[assets, assets])
    print(f"  TE vs bench: {te_mv*100:5.2f}%")

# =============================
# 2) Risk Parity (ERC)
# =============================
print("\n⚖️ 2) Risk Parity (ERC)")
cons_rp = [sum_to_one, *group_cons]
if w_bench is not None and TE_MAX is not None:
    cons_rp.append({"type": "ineq",
                    "fun": lambda w, wb=w_bench, S=cov, te=TE_MAX: te_constraint_ineq(w, wb, S, te)})

res_rp = minimize(risk_parity_objective, x0=x0,
                  args=(cov.loc[assets, assets],),
                  method="SLSQP", bounds=bnds, constraints=cons_rp,
                  options={"maxiter": 500, "ftol": 1e-9})
if not res_rp.success:
    print(f"⚠️  ERC no convergió: {res_rp.message}")
w_rp = res_rp.x
ret_rp, vol_rp = portfolio_performance(w_rp, mean_ret[assets], cov.loc[assets, assets])
print("  Pesos RP:", {a: f"{w*100:5.1f}%" for a, w in zip(assets, w_rp)})
print(f"  Ret {ret_rp*100:6.2f}% | Vol {vol_rp*100:6.2f}%")
if w_bench is not None:
    te_rp = tracking_error_annual(w_rp, w_bench, cov.loc[assets, assets])
    print(f"  TE vs bench: {te_rp*100:5.2f}%")

# =============================
# 3) Restricciones (mismas cotas y grupos; Sharpe)
# =============================
print("\n🎯 3) Restricciones (Sharpe con cotas y grupos)")
cons_cs = [sum_to_one, *group_cons]
if w_bench is not None and TE_MAX is not None:
    cons_cs.append({"type": "ineq",
                    "fun": lambda w, wb=w_bench, S=cov, te=TE_MAX: te_constraint_ineq(w, wb, S, te)})

res_cs = minimize(negative_sharpe, x0=x0,
                  args=(mean_ret[assets], cov.loc[assets, assets], RISK_FREE),
                  method="SLSQP", bounds=bnds, constraints=cons_cs,
                  options={"maxiter": 500, "ftol": 1e-9})
if not res_cs.success:
    print(f"⚠️  Cartera restringida no convergió: {res_cs.message}")
w_cs = res_cs.x
ret_cs, vol_cs = portfolio_performance(w_cs, mean_ret[assets], cov.loc[assets, assets])
shp_cs = (ret_cs - RISK_FREE) / (vol_cs + 1e-12)
print("  Pesos CS:", {a: f"{w*100:5.1f}%" for a, w in zip(assets, w_cs)})
print(f"  Ret {ret_cs*100:6.2f}% | Vol {vol_cs*100:6.2f}% | Sharpe {shp_cs:6.3f}")
if w_bench is not None:
    te_cs = tracking_error_annual(w_cs, w_bench, cov.loc[assets, assets])
    print(f"  TE vs bench: {te_cs*100:5.2f}%")

# =============================
# 4) Black–Litterman (μ posterior, Σ histórica)
# =============================
print("\n🧠 4) Black–Litterman (simplificado)")

TAU = 0.05     # incertidumbre del prior
DELTA = 2.5    # aversión al riesgo típica
MARKET_WEIGHTS: Optional[Dict[str, float]] = None   # si se tienen
VIEWS: Optional[Dict[str, float]] = None            # ej: {"AMZN": 0.12, "GLD": 0.04}

Sigma = cov.loc[assets, assets].values
mu_hist = mean_ret[assets].values

# prior de equilibrio
if MARKET_WEIGHTS:
    w_mkt = weights_from_dict(assets, MARKET_WEIGHTS)
else:
    w_mkt = np.full(N, 1.0 / N)
Pi = DELTA * (Sigma @ w_mkt)

# views
if not VIEWS:
    P = np.eye(N); Q = Pi.copy()
else:
    P_rows, Q_vals = [], []
    for i, a in enumerate(assets):
        if a in VIEWS:
            row = np.zeros(N); row[i] = 1.0
            P_rows.append(row); Q_vals.append(float(VIEWS[a]))
    if len(P_rows) == 0:
        P = np.eye(N); Q = Pi.copy()
    else:
        P = np.array(P_rows); Q = np.array(Q_vals)

from numpy.linalg import inv
Omega = np.diag(np.diag(Sigma)) * 0.25
A = inv(TAU * Sigma)
middle = A + P.T @ inv(Omega) @ P
Sigma_mu = inv(middle)                      # cov de la incertidumbre de μ (NO de riesgo)
mu_bl = Sigma_mu @ (A @ Pi + P.T @ inv(Omega) @ Q)
mu_bl_series = pd.Series(mu_bl, index=assets)

cons_bl = [sum_to_one, *group_cons]
if w_bench is not None and TE_MAX is not None:
    cons_bl.append({"type": "ineq",
                    "fun": lambda w, wb=w_bench, S=cov, te=TE_MAX: te_constraint_ineq(w, wb, S, te)})

res_bl = minimize(negative_sharpe, x0=x0,
                  args=(mu_bl_series, cov.loc[assets, assets], RISK_FREE),
                  method="SLSQP", bounds=bnds, constraints=cons_bl,
                  options={"maxiter": 500, "ftol": 1e-9})
if not res_bl.success:
    print(f"⚠️  BL no convergió: {res_bl.message}")
w_bl = res_bl.x
ret_bl, vol_bl = portfolio_performance(w_bl, mu_bl_series, cov.loc[assets, assets])
shp_bl = (ret_bl - RISK_FREE) / (vol_bl + 1e-12)
print("  Pesos BL:", {a: f"{w*100:5.1f}%" for a, w in zip(assets, w_bl)})
print(f"  Ret {ret_bl*100:6.2f}% | Vol {vol_bl*100:6.2f}% | Sharpe {shp_bl:6.3f}")
if w_bench is not None:
    te_bl = tracking_error_annual(w_bl, w_bench, cov.loc[assets, assets])
    print(f"  TE vs bench: {te_bl*100:5.2f}%")

# =============================
# 5) Frontera Eficiente (min-var para retornos objetivo)
# =============================
print("\n📐 5) Frontera eficiente")

def min_var_for_target(mu: pd.Series, Sigma_df: pd.DataFrame,
                       r_target: float,
                       bounds: Tuple[Tuple[float, float], ...],
                       base_cons: List[dict]) -> Optional[np.ndarray]:
    cons = list(base_cons) + [{"type": "eq", "fun": lambda w, mu=mu, rt=r_target: float(np.dot(w, mu.values) - rt)}]
    res = minimize(portfolio_variance, x0=x0,
                   args=(Sigma_df,),
                   method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-9})
    if res.success:
        return res.x
    return None

# rango de retornos objetivo (clamp dentro de percentiles razonables)
mu_vals = mean_ret[assets].values
r_min, r_max = np.percentile(mu_vals, 5), np.percentile(mu_vals, 95)
targets = np.linspace(max(0.5*r_min, r_min - 0.02), r_max + 0.02, 40)

base_cons_fe = [sum_to_one, *group_cons]
if w_bench is not None and TE_MAX is not None:
    base_cons_fe.append({"type": "ineq",
                         "fun": lambda w, wb=w_bench, S=cov, te=TE_MAX: te_constraint_ineq(w, wb, S, te)})

ef_weights, ef_rets, ef_vols = [], [], []
for rt in targets:
    w_fe = min_var_for_target(mean_ret[assets], cov.loc[assets, assets], rt, bnds, base_cons_fe)
    if w_fe is not None:
        r, v = portfolio_performance(w_fe, mean_ret[assets], cov.loc[assets, assets])
        ef_weights.append(w_fe); ef_rets.append(r); ef_vols.append(v)

# =============================
# 6) Comparación, Recomendación y TE
# =============================
print("\n📊 6) COMPARACIÓN DE ESTRATEGIAS (medidas con Σ histórica)")
strategies = {
    "Maximum Sharpe": (w_ms, ret_ms, vol_ms, shp_ms),
    "Minimum Variance": (w_mv, ret_mv, vol_mv, (ret_mv - RISK_FREE) / (vol_mv + 1e-12)),
    "Risk Parity": (w_rp, ret_rp, vol_rp, (ret_rp - RISK_FREE) / (vol_rp + 1e-12)),
    "Constrained": (w_cs, ret_cs, vol_cs, shp_cs),
    "Black–Litterman": (w_bl, ret_bl, vol_bl, shp_bl),
}

for name, (w, r, v, s) in strategies.items():
    print(f"\n{name}:")
    print("  Pesos:", {a: f"{wi*100:5.1f}%" for a, wi in zip(assets, w)})
    print(f"  Ret {r*100:6.2f}% | Vol {v*100:6.2f}% | Sharpe {s:6.3f}")
    if w_bench is not None:
        te = tracking_error_annual(w, w_bench, cov.loc[assets, assets])
        print(f"  TE vs bench: {te*100:5.2f}%")

# Recomendación por Sharpe (histórico)
print("\n🏆 RECOMENDACIÓN FINAL (máx. Sharpe histórico)")
best_name, best_w, best_r, best_v, best_s = None, None, -1e9, 1e9, -1e9
for name, (w, r, v, s) in strategies.items():
    if s > best_s:
        best_name, best_w, best_r, best_v, best_s = name, w, r, v, s

print(f"Estrategia recomendada: {best_name}")
remanente = MONTHLY_CONTRIB
print("\n💶 APORTACIONES RECOMENDADAS (~300 EUR/mes total):")
for a, w in zip(assets, best_w):
    aport = round(MONTHLY_CONTRIB * w)
    remanente -= aport
    print(f"  {a}: {aport:7.0f} EUR/mes (~{w*100:5.1f}%)")
if abs(remanente) > 0:
    print(f"  Ajuste remanente: {remanente:+.0f} EUR/mes")

if w_bench is not None:
    te_best = tracking_error_annual(best_w, w_bench, cov.loc[assets, assets])
    print(f"  TE vs bench: {te_best*100:5.2f}%")

# =============================
# 7) Gráficos y exportación
# =============================
import os
from pathlib import Path
try:
    base_dir = Path(__file__).resolve().parent
except NameError:
    base_dir = Path.cwd()
png_dir = base_dir / "png"
png_dir.mkdir(parents=True, exist_ok=True)

# Frontera eficiente + puntos de estrategias
fig, ax = plt.subplots(figsize=(10, 7))
if ef_vols:
    ax.plot(ef_vols, ef_rets, lw=2, label="Frontera Eficiente")
# puntos de estrategias
colors = {"Maximum Sharpe": "tab:green", "Minimum Variance": "tab:blue",
          "Risk Parity": "tab:orange", "Constrained": "tab:red", "Black–Litterman": "tab:purple"}
for name, (w, r, v, s) in strategies.items():
    ax.scatter([v], [r], s=60, label=name, color=colors.get(name, None), zorder=3)
    ax.annotate(name, (v, r), textcoords="offset points", xytext=(6,6), fontsize=8)
if w_bench is not None:
    v_b, r_b = portfolio_performance(w_bench, mean_ret[assets], cov.loc[assets, assets])
    ax.scatter([v_b], [r_b], s=60, label="Benchmark", marker="D", zorder=3)
    ax.annotate("Benchmark", (v_b, r_b), textcoords="offset points", xytext=(6,-12), fontsize=8)

ax.set_xlabel("Volatilidad anual")
ax.set_ylabel("Retorno anual")
ax.set_title("Frontera Eficiente y Carteras")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(png_dir / "frontera_eficiente.png", dpi=300)

# Barras de pesos por estrategia
fig2, axes2 = plt.subplots(3, 2, figsize=(12, 12))
axes2 = axes2.flatten()
for i, (name, (w, r, v, s)) in enumerate(strategies.items()):
    ax2 = axes2[i]
    ax2.bar(assets, w)
    ax2.set_ylim(0, 1)
    ax2.set_title(f"Pesos — {name}")
    ax2.tick_params(axis='x', rotation=45)
for j in range(i+1, len(axes2)):
    axes2[j].axis("off")
fig2.tight_layout()
fig2.savefig(png_dir / "pesos_estrategias.png", dpi=300)

# Contribuciones al riesgo por estrategia (porcentuales)
fig3, axes3 = plt.subplots(3, 2, figsize=(12, 12))
axes3 = axes3.flatten()
for i, (name, (w, r, v, s)) in enumerate(strategies.items()):
    ax3 = axes3[i]
    _, pct = risk_contributions(w, cov.loc[assets, assets])
    ax3.bar(assets, pct)
    ax3.set_ylim(0, 1)
    ax3.set_title(f"Contribución al riesgo — {name}")
    ax3.tick_params(axis='x', rotation=45)
for j in range(i+1, len(axes3)):
    axes3[j].axis("off")
fig3.tight_layout()
fig3.savefig(png_dir / "contribuciones_riesgo.png", dpi=300)

print(f"\n🖼️  Gráficos guardados en: {png_dir.resolve()}")
print("   - frontera_eficiente.png")
print("   - pesos_estrategias.png")
print("   - contribuciones_riesgo.png")
print("\n🎉 Optimización completada.")
