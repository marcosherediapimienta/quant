"""
Optimizador de Cartera (Markowitz, ERC, BL, Restricciones, Frontera Eficiente, TE)
- Descarga precios con yfinance
- Convierte USD→EUR (EURUSD=X = USD por 1 EUR) cuando corresponde
- Cálculo de métricas anualizadas
- Optimiza: Máx. Sharpe, Mín. Var, ERC, MDR (antes "Constrained") y Black–Litterman (μ posterior con Σ histórica)
- Frontera eficiente (min-var para retornos objetivo) + Frontera ACTIVA (Sharpe vs TE)
- Tracking Error frente a benchmark externo (SP500/MSCI WORLD): restricción ex–ante y cálculo ex–post en backtest
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
TICKERS = ["0P0001CLDK.F","NVDA","MSFT","AAPL","GOOGL","IBM","AMZN","META","TSLA","JPM","BRK-A","BTC-EUR","GLD"]

FECHA_INICIO = "2018-03-20"
FECHA_FIN = "2025-09-02"

FX_TICKER = "EURUSD=X"      # USD por 1 EUR

USD_TICKERS = {"NVDA","MSFT","AAPL","GOOGL","IBM","AMZN","META","TSLA","JPM","BRK-A","GLD"}

RISK_FREE = 0.045           # anual
MONTHLY_CONTRIB = 300.0     # EUR/mes totales

# Covarianza: usar Ledoit–Wolf si hay sklearn
USE_LEDOIT_WOLF = True
# Si no hay sklearn o se desactiva, aplicar shrinkage simple hacia la diagonal:
SIMPLE_SHRINKAGE_LAMBDA = 0.05  # 0.0 para desactivar

# ====== Restricciones por activo (ejemplos realistas) ======
# Si un ticker no aparece en este dict, se usa (0,1) por defecto.
PER_ASSET_BOUNDS: Dict[str, Tuple[float, float]] = {
    "BTC-EUR": (0.00, 0.00),     # techo al cripto
    #"GLD": (0.00, 0.40),         # oro hasta 40%
    #"AMZN": (0.00, 0.35),        # single-stock hasta 35%
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

# ====== Benchmark externo y Tracking Error (opcional) ======
# Elige: "SP500" o "MSCI_WORLD" o None
BENCHMARK = "MSCI_WORLD"  # cambia a "SP500" si lo prefieres

# Tickers candidatos (por orden de preferencia).
# EUNL.DE cotiza en EUR (evita FX); los demás se convierten a EUR con EURUSD=X cuando son USD.
BENCHMARK_TICKERS = {
    "SP500": ["SPY", "^GSPC"],                             # SPY (ETF USD) o índice ^GSPC
    "MSCI_WORLD": ["EUNL.DE", "URTH", "IWDA.AS", "SWDA.L"] # prioriza EUNL.DE (EUR)
}

# Máximo Tracking Error anual permitido en la OPTIMIZACIÓN (None para desactivar)
TE_MAX: Optional[float] = None  # p.ej. 6% anual

# --- Control de factibilidad y robustez del TE ---
AUTO_RELAX_TE      = True   # si TE_MAX < TE_min, relajar automáticamente
TE_RELAX_EPS_ABS   = 0.01   # +1% absoluto sobre TE_min
TE_PENALTY_LAMBDA  = 10.0   # penalización suave (hinge) por exceder TE_MAX
N_RANDOM_STARTS    = 6      # multistart

# --- Black–Litterman: prior y vistas ---
BL_USE_TRACKING_PRIOR = True  # usar w_TEmin como prior de mercado si está disponible
MARKET_WEIGHTS: Optional[Dict[str, float]] = None   # se rellenará con w_TEmin si BL_USE_TRACKING_PRIOR=True
VIEWS: Optional[Dict[str, float]] = None            # ej: {"AMZN": 0.12, "GLD": 0.04}

# =============================
# Helpers de datos
# =============================
def get_price_series(df: pd.DataFrame, ticker: Optional[str] = None) -> Optional[pd.Series]:
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
    if price_series is None or fx_series is None or fx_series.empty:
        return price_series
    aligned_fx = fx_series.reindex(price_series.index).ffill().bfill()
    return price_series / aligned_fx


def shrink_covariance_simple(cov: pd.DataFrame, lam: float) -> pd.DataFrame:
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

def per_asset_bounds(assets: List[str],
                     bounds_dict: Dict[str, Tuple[float, float]]
                    ) -> Tuple[Tuple[float, float], ...]:
    return tuple(bounds_dict.get(a, (0.0, 1.0)) for a in assets)


def build_group_constraints(assets: List[str],
                            groups: Dict[str, List[str]],
                            gmax: Optional[Dict[str, float]] = None,
                            gmin: Optional[Dict[str, float]] = None
                           ) -> List[dict]:
    cons: List[dict] = []
    name_to_idx = {name: [assets.index(a) for a in members if a in assets]
                   for name, members in groups.items()}

    if gmax:
        for name, cap in gmax.items():
            idx = name_to_idx.get(name, [])
            if idx:
                cons.append({"type": "ineq",
                             "fun": (lambda w, idx=idx, cap=cap: cap - np.sum(w[idx]))})

    if gmin:
        for name, floor in gmin.items():
            idx = name_to_idx.get(name, [])
            if idx:
                cons.append({"type": "ineq",
                             "fun": (lambda w, idx=idx, floor=floor: np.sum(w[idx]) - floor)})

    return cons

# ====== Helpers de benchmark y TE-índice ======
def download_benchmark_series(benchmark: str,
                              start: str, end: str,
                              fx_series: Optional[pd.Series]) -> tuple[str, Optional[pd.Series]]:
    tickers = BENCHMARK_TICKERS.get(benchmark, [])
    for tk in tickers:
        if tk.endswith(".L"):
            continue
        try:
            df = yf.download(tk, start=start, end=end, auto_adjust=False,
                             progress=False, group_by="ticker")
            s = get_price_series(df, tk if isinstance(df.columns, pd.MultiIndex) else None)
            if s is None or s.empty:
                continue
            needs_fx = False
            if tk in {"SPY", "URTH"} or tk.startswith("^"):
                needs_fx = True
            if tk.endswith((".DE", ".PA", ".MI", ".AS", ".EU")):
                needs_fx = False
            if needs_fx and fx_series is not None:
                s = convert_usd_to_eur(s, fx_series)
            return tk, s.dropna().sort_index()
        except Exception:
            continue
    return (tickers[0] if tickers else "N/A", None)


def prepare_te_objects(assets_rets: pd.DataFrame,
                       bench_rets: pd.Series) -> tuple[pd.DataFrame, pd.Series, float]:
    df = assets_rets.join(bench_rets.rename("B"), how="inner")
    R = df[assets_rets.columns]
    B = df["B"]
    Sigma_ann = R.cov() * 252.0
    c_ann = R.apply(lambda x: x.cov(B)) * 252.0
    var_b_ann = float(B.var() * 252.0)
    return Sigma_ann, c_ann, var_b_ann


def te_index_constraint_ineq(w: np.ndarray,
                             Sigma_ann: pd.DataFrame,
                             c_ann: pd.Series,
                             var_b_ann: float,
                             te_max: float) -> float:
    w = np.asarray(w)
    te2 = float(w @ Sigma_ann.values @ w - 2.0 * (w @ c_ann.values) + var_b_ann)
    return float(te_max**2 - max(te2, 0.0))


def te_index_ex_ante(w: np.ndarray,
                     Sigma_ann: Optional[pd.DataFrame],
                     c_ann: Optional[pd.Series],
                     var_b_ann: Optional[float]) -> Optional[float]:
    if Sigma_ann is None or c_ann is None or var_b_ann is None:
        return None
    te2 = float(w @ Sigma_ann.values @ w - 2.0 * (w @ c_ann.values) + var_b_ann)
    return float(np.sqrt(max(te2, 0.0)))

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
    w = np.asarray(w)
    port_vol = np.sqrt(max(w @ sigma.values @ w, 1e-18))
    marginal = sigma.values @ w
    abs_contrib = w * marginal / (port_vol + 1e-12)
    pct_contrib = abs_contrib / (port_vol + 1e-12)
    target = np.ones_like(w) / w.size
    return float(np.sum((pct_contrib - target) ** 2))


def risk_contributions(w: np.ndarray, sigma: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    port_vol = np.sqrt(max(w @ sigma.values @ w, 1e-18))
    marginal = sigma.values @ w
    abs_contrib = w * marginal / (port_vol + 1e-12)
    pct_contrib = abs_contrib / (port_vol + 1e-12)
    return abs_contrib, pct_contrib

# --- Objetivo de Max Diversification Ratio (MDR)
def neg_diversification_ratio(w: np.ndarray, Sigma_df: pd.DataFrame) -> float:
    sigma_i = np.sqrt(np.diag(Sigma_df.values))
    numer = float(np.dot(np.abs(w), sigma_i))
    denom = float(np.sqrt(max(w @ Sigma_df.values @ w, 1e-18)))
    return -(numer / (denom + 1e-12))

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
# Benchmark externo (índice) en EUR
# =============================
bench_price = None
bench_ret = None
bench_tk_used = None
Sigma_te = None
c_te = None
varb_te = None

if BENCHMARK is not None:
    bench_tk_used, bench_price = download_benchmark_series(
        BENCHMARK, FECHA_INICIO, FECHA_FIN, fx_series
    )
    if bench_price is not None:
        bench_ret = np.log(bench_price).diff().dropna()
        bench_ret = bench_ret.reindex(returns.index).dropna()
        Sigma_te, c_te, varb_te = prepare_te_objects(returns[assets], bench_ret)
        print(f"\n📏 Benchmark: {BENCHMARK} usando '{bench_tk_used}' (EUR).")
        if TE_MAX is not None:
            print(f"   Restricción de TE: TE ≤ {TE_MAX:.2%} (anual).")
    else:
        print(f"⚠️  No se pudo obtener el benchmark '{BENCHMARK}'. Se desactiva TE_MAX.")
        TE_MAX = None

# =============================
# Factibilidad de TE, relajación y helpers
# =============================
def te_quadratic(w: np.ndarray, S_df: pd.DataFrame, c_ser: pd.Series) -> float:
    S = S_df.values; c = c_ser.values
    return float(w @ S @ w - 2.0 * (w @ c))

def te_quadratic_grad(w: np.ndarray, S_df: pd.DataFrame, c_ser: pd.Series) -> np.ndarray:
    S = S_df.values; c = c_ser.values
    return 2.0 * (S @ w - c)

def te_constraint_jac(w: np.ndarray, S_df: pd.DataFrame, c_ser: pd.Series) -> np.ndarray:
    S = S_df.values; c = c_ser.values
    return 2.0 * (c - S @ w)

def make_te_ineq_with_jac(S_df, c_ser, var_b, te_max):
    return {
        "type": "ineq",
        "fun":  lambda w, S=S_df, c=c_ser, vb=var_b, te=te_max: te_index_constraint_ineq(w, S, c, vb, te),
        "jac":  lambda w, S=S_df, c=c_ser: te_constraint_jac(w, S, c),
    }

TE_min, w_TEmin = None, None
if bench_ret is not None:
    cons_te = [ {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                *build_group_constraints(assets, GROUPS, GROUP_MAX, GROUP_MIN) ]
    res_te = minimize(
        fun=lambda w: te_quadratic(w, Sigma_te, c_te),
        x0=np.full(len(assets), 1.0/len(assets)),
        jac=lambda w: te_quadratic_grad(w, Sigma_te, c_te),
        method="SLSQP",
        bounds=per_asset_bounds(assets, PER_ASSET_BOUNDS),
        constraints=cons_te,
        options={"maxiter": 1000, "ftol": 1e-12}
    )
    if res_te.success:
        w_TEmin = res_te.x
        TE_min = te_index_ex_ante(w_TEmin, Sigma_te, c_te, varb_te)
        print(f"\n🧭 TE mínimo factible con este universo: {TE_min*100:5.2f}%")
        if TE_MAX is not None and TE_min is not None and TE_MAX < TE_min - 1e-6:
            if AUTO_RELAX_TE:
                old = TE_MAX
                TE_MAX = float(TE_min + TE_RELAX_EPS_ABS)
                print(f"⚠️  TE_MAX={old:.2%} inviable; ajusto a {TE_MAX:.2%} (= TE_min + {TE_RELAX_EPS_ABS:.2%}).")
            else:
                print(f"⚠️  TE_MAX={TE_MAX:.2%} inviable; desactívalo o amplía universo/bounds.")
    else:
        print(f"⚠️  No se pudo calcular TE mínimo: {res_te.message}")

# Mostrar cartera de mínimo TE y preparar para strategies
w_te_track = None; r_te = None; v_te = None; s_te = None
if w_TEmin is not None:
    print("\n🎯 Cartera de MÍNIMO TE (vs índice)")
    print("  Pesos TE_min:", {a: f"{w*100:5.1f}%" for a, w in zip(assets, w_TEmin)})
    te_te = te_index_ex_ante(w_TEmin, Sigma_te, c_te, varb_te)
    print(f"  TE vs índice (ex–ante): {te_te*100:5.2f}%")
    r_te, v_te = portfolio_performance(w_TEmin, mean_ret[assets], cov.loc[assets, assets])
    s_te = (r_te - RISK_FREE) / (v_te + 1e-12)
    w_te_track = w_TEmin

# Si queremos que BL use w_TEmin como prior de "mercado"
if w_TEmin is not None and BL_USE_TRACKING_PRIOR:
    MARKET_WEIGHTS = {a: float(w) for a, w in zip(assets, w_TEmin)}
    print("🧩 BL prior: usando la cartera de mínimo TE como 'mercado' (MARKET_WEIGHTS).")

# Penalización “suave” de TE para el objetivo
def te_hinge_penalty(w: np.ndarray,
                     S_te: Optional[pd.DataFrame],
                     c_te: Optional[pd.Series],
                     vb_te: Optional[float],
                     te_cap: Optional[float],
                     lam: float) -> float:
    if S_te is None or c_te is None or vb_te is None or te_cap is None or lam <= 0:
        return 0.0
    te2 = float(w @ S_te.values @ w - 2.0 * (w @ c_te.values) + vb_te)
    te = float(np.sqrt(max(te2, 0.0)))
    return lam * max(0.0, te - te_cap)**2

def neg_sharpe_with_pen(w, mu, Sigma, rf, S_te, c_te, vb_te, te_cap, lam):
    return negative_sharpe(w, mu, Sigma, rf) + te_hinge_penalty(w, S_te, c_te, vb_te, te_cap, lam)

def var_with_pen(w, Sigma, S_te, c_te, vb_te, te_cap, lam):
    return portfolio_variance(w, Sigma) + te_hinge_penalty(w, S_te, c_te, vb_te, te_cap, lam)

def erc_with_pen(w, Sigma, S_te, c_te, vb_te, te_cap, lam):
    return risk_parity_objective(w, Sigma) + te_hinge_penalty(w, S_te, c_te, vb_te, te_cap, lam)

# Multistart: generador de x0 factibles aleatorios
def random_feasible_x0(bounds: Tuple[Tuple[float,float], ...], n: int = 6) -> List[np.ndarray]:
    L = np.array([b[0] for b in bounds], dtype=float)
    U = np.array([b[1] for b in bounds], dtype=float)
    assert L.sum() <= 1.0 + 1e-9 and U.sum() >= 1.0 - 1e-9
    xs = []
    for _ in range(n):
        w = L + np.random.dirichlet(np.ones(len(bounds))) * max(1.0 - L.sum(), 0.0)
        for __ in range(50):
            over = (w - U).clip(min=0.0)
            if over.max() <= 1e-12:
                break
            w -= over
            cap = (U - w).clip(min=0.0)
            cap_sum = cap.sum()
            if cap_sum <= 1e-12:
                break
            w += cap * (over.sum() / cap_sum)
        w = np.minimum(np.maximum(w, L), U)
        w /= w.sum()
        xs.append(w)
    return xs

def optimize_with_restarts(fun, x0_list, bounds, constraints, args=(), options=None):
    best = None
    for seed_x0 in x0_list:
        res = minimize(fun, x0=seed_x0, args=args, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options=options or {"maxiter": 500, "ftol": 1e-9})
        if best is None:
            best = res
        else:
            if res.success and (not best.success or res.fun < best.fun):
                best = res
            elif (not best.success) and (res.fun < best.fun):
                best = res
    return best

# =============================
# Optimización base y restricciones
# =============================
N = len(assets)
bnds = per_asset_bounds(assets, PER_ASSET_BOUNDS)

x0_eq = np.full(N, 1.0 / N)
x0_list = [x0_eq]
if 'w_TEmin' in locals() and w_TEmin is not None:
    x0_list.append(0.5 * x0_eq + 0.5 * w_TEmin)
x0_list += random_feasible_x0(bnds, n=N_RANDOM_STARTS)

sum_to_one = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
group_cons = build_group_constraints(assets, GROUPS, GROUP_MAX, GROUP_MIN)

# =============================
# 1) Máximo Sharpe
# =============================
print("\n🚀 1) Máximo Sharpe")
cons_ms = [sum_to_one, *group_cons]
if bench_ret is not None and TE_MAX is not None:
    cons_ms.append(make_te_ineq_with_jac(Sigma_te, c_te, varb_te, TE_MAX))

res_ms = optimize_with_restarts(
    fun=lambda w: neg_sharpe_with_pen(w, mean_ret[assets], cov.loc[assets, assets], RISK_FREE,
                                      Sigma_te, c_te, varb_te, TE_MAX, TE_PENALTY_LAMBDA),
    x0_list=x0_list, bounds=bnds, constraints=cons_ms
)
if not res_ms.success:
    print(f"⚠️  Máx. Sharpe no convergió: {res_ms.message}")
w_ms = res_ms.x
ret_ms, vol_ms = portfolio_performance(w_ms, mean_ret[assets], cov.loc[assets, assets])
shp_ms = (ret_ms - RISK_FREE) / (vol_ms + 1e-12)
print("  Pesos MS:", {a: f"{w*100:5.1f}%" for a, w in zip(assets, w_ms)})
print(f"  Ret {ret_ms*100:6.2f}% | Vol {vol_ms*100:6.2f}% | Sharpe {shp_ms:6.3f}")
if bench_ret is not None:
    te_ms = te_index_ex_ante(w_ms, Sigma_te, c_te, varb_te)
    if te_ms is not None:
        print(f"  TE vs índice (ex–ante): {te_ms*100:5.2f}%")

# =============================
# 1b) Mínima Varianza
# =============================
print("\n🛡️ 1b) Mínima Varianza")
cons_mv = [sum_to_one, *group_cons]
if bench_ret is not None and TE_MAX is not None:
    cons_mv.append(make_te_ineq_with_jac(Sigma_te, c_te, varb_te, TE_MAX))

res_mv = optimize_with_restarts(
    fun=lambda w: var_with_pen(w, cov.loc[assets, assets],
                               Sigma_te, c_te, varb_te, TE_MAX, TE_PENALTY_LAMBDA),
    x0_list=x0_list, bounds=bnds, constraints=cons_mv
)
if not res_mv.success:
    print(f"⚠️  Mín. Var no convergió: {res_mv.message}")
w_mv = res_mv.x
ret_mv, vol_mv = portfolio_performance(w_mv, mean_ret[assets], cov.loc[assets, assets])
print("  Pesos MV:", {a: f"{w*100:5.1f}%" for a, w in zip(assets, w_mv)})
print(f"  Ret {ret_mv*100:6.2f}% | Vol {vol_mv*100:6.2f}%")
if bench_ret is not None:
    te_mv = te_index_ex_ante(w_mv, Sigma_te, c_te, varb_te)
    if te_mv is not None:
        print(f"  TE vs índice (ex–ante): {te_mv*100:5.2f}%")

# =============================
# 2) Risk Parity (ERC)
# =============================
print("\n⚖️ 2) Risk Parity (ERC)")
cons_rp = [sum_to_one, *group_cons]
if bench_ret is not None and TE_MAX is not None:
    cons_rp.append(make_te_ineq_with_jac(Sigma_te, c_te, varb_te, TE_MAX))

res_rp = optimize_with_restarts(
    fun=lambda w: erc_with_pen(w, cov.loc[assets, assets],
                               Sigma_te, c_te, varb_te, TE_MAX, TE_PENALTY_LAMBDA),
    x0_list=x0_list, bounds=bnds, constraints=cons_rp
)
if not res_rp.success:
    print(f"⚠️  ERC no convergió: {res_rp.message}")
w_rp = res_rp.x
ret_rp, vol_rp = portfolio_performance(w_rp, mean_ret[assets], cov.loc[assets, assets])
print("  Pesos RP:", {a: f"{w*100:5.1f}%" for a, w in zip(assets, w_rp)})
print(f"  Ret {ret_rp*100:6.2f}% | Vol {vol_rp*100:6.2f}%")
if bench_ret is not None:
    te_rp = te_index_ex_ante(w_rp, Sigma_te, c_te, varb_te)
    if te_rp is not None:
        print(f"  TE vs índice (ex–ante): {te_rp*100:5.2f}%")

# =============================
# 3) Max Diversification Ratio (MDR) en lugar de "Constrained"
# =============================
print("\n🎯 3) Max Diversification Ratio (MDR) con cotas/grupos")
cons_mdr = [sum_to_one, *group_cons]
if bench_ret is not None and TE_MAX is not None:
    cons_mdr.append(make_te_ineq_with_jac(Sigma_te, c_te, varb_te, TE_MAX))

res_mdr = optimize_with_restarts(
    fun=lambda w: neg_diversification_ratio(w, cov.loc[assets, assets]) +
                  te_hinge_penalty(w, Sigma_te, c_te, varb_te, TE_MAX, TE_PENALTY_LAMBDA),
    x0_list=x0_list, bounds=bnds, constraints=cons_mdr
)
if not res_mdr.success:
    print(f"⚠️  MDR no convergió: {res_mdr.message}")
w_mdr = res_mdr.x
ret_mdr, vol_mdr = portfolio_performance(w_mdr, mean_ret[assets], cov.loc[assets, assets])
shp_mdr = (ret_mdr - RISK_FREE) / (vol_mdr + 1e-12)
print("  Pesos MDR:", {a: f"{w*100:5.1f}%" for a, w in zip(assets, w_mdr)})
print(f"  Ret {ret_mdr*100:6.2f}% | Vol {vol_mdr*100:6.2f}% | Sharpe {shp_mdr:6.3f}")
if bench_ret is not None:
    te_mdr = te_index_ex_ante(w_mdr, Sigma_te, c_te, varb_te)
    if te_mdr is not None:
        print(f"  TE vs índice (ex–ante): {te_mdr*100:5.2f}%")

# =============================
# 4) Black–Litterman (μ posterior, Σ histórica) — CALIBRADO
# =============================
print("\n🧠 4) Black–Litterman (calibrado)")

# Calibración de TAU y DELTA
T = len(returns)
TAU = 1.0 / max(T, 1)   # regla típica
if bench_ret is not None and len(bench_ret) > 50:
    mkt_mu  = bench_ret.mean() * 252.0
    mkt_var = bench_ret.var(ddof=1) * 252.0
    DELTA = max((mkt_mu - RISK_FREE) / (mkt_var + 1e-18), 1e-3)
else:
    DELTA = 2.5  # fallback

Sigma = cov.loc[assets, assets].values
N = len(assets)
if MARKET_WEIGHTS:
    w_mkt = weights_from_dict(assets, MARKET_WEIGHTS)
else:
    w_mkt = np.full(N, 1.0 / N)
Pi = DELTA * (Sigma @ w_mkt)

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
Omega = TAU * np.diag(np.diag(P @ Sigma @ P.T))
A = inv(TAU * Sigma)
middle = A + P.T @ inv(Omega) @ P
Sigma_mu = inv(middle)
mu_bl = Sigma_mu @ (A @ Pi + P.T @ inv(Omega) @ Q)
mu_bl_series = pd.Series(mu_bl, index=assets)

cons_bl = [sum_to_one, *group_cons]
if bench_ret is not None and TE_MAX is not None:
    cons_bl.append(make_te_ineq_with_jac(Sigma_te, c_te, varb_te, TE_MAX))

res_bl = optimize_with_restarts(
    fun=lambda w: neg_sharpe_with_pen(w, mu_bl_series, cov.loc[assets, assets], RISK_FREE,
                                      Sigma_te, c_te, varb_te, TE_MAX, TE_PENALTY_LAMBDA),
    x0_list=x0_list, bounds=bnds, constraints=cons_bl
)
if not res_bl.success:
    print(f"⚠️  BL no convergió: {res_bl.message}")
w_bl = res_bl.x
ret_bl, vol_bl = portfolio_performance(w_bl, mu_bl_series, cov.loc[assets, assets])
shp_bl = (ret_bl - RISK_FREE) / (vol_bl + 1e-12)
print("  Pesos BL:", {a: f"{w*100:5.1f}%" for a, w in zip(assets, w_bl)})
print(f"  Ret {ret_bl*100:6.2f}% | Vol {vol_bl*100:6.2f}% | Sharpe {shp_bl:6.3f}")
if bench_ret is not None:
    te_bl = te_index_ex_ante(w_bl, Sigma_te, c_te, varb_te)
    if te_bl is not None:
        print(f"  TE vs índice (ex–ante): {te_bl*100:5.2f}%")

# =============================
# 5) Frontera Eficiente (min-var para retornos objetivo)
# =============================
print("\n📐 5) Frontera eficiente")

def min_var_for_target(mu: pd.Series, Sigma_df: pd.DataFrame,
                       r_target: float,
                       bounds: Tuple[Tuple[float, float], ...],
                       base_cons: List[dict],
                       x0_list: List[np.ndarray]) -> Optional[np.ndarray]:
    cons = list(base_cons) + [{"type": "eq", "fun": lambda w, mu=mu, rt=r_target: float(np.dot(w, mu.values) - rt)}]
    res = optimize_with_restarts(
        fun=lambda w: portfolio_variance(w, Sigma_df),
        x0_list=x0_list, bounds=bounds, constraints=cons
    )
    if res and res.success:
        return res.x
    return None

mu_vals = mean_ret[assets].values
r_min, r_max = np.percentile(mu_vals, 5), np.percentile(mu_vals, 95)
targets = np.linspace(max(0.5*r_min, r_min - 0.02), r_max + 0.02, 40)

base_cons_fe = [sum_to_one, *group_cons]
if bench_ret is not None and TE_MAX is not None:
    base_cons_fe.append(make_te_ineq_with_jac(Sigma_te, c_te, varb_te, TE_MAX))

ef_weights, ef_rets, ef_vols = [], [], []
for rt in targets:
    w_fe = min_var_for_target(mean_ret[assets], cov.loc[assets, assets], rt, bnds, base_cons_fe, x0_list)
    if w_fe is not None:
        r, v = portfolio_performance(w_fe, mean_ret[assets], cov.loc[assets, assets])
        ef_weights.append(w_fe); ef_rets.append(r); ef_vols.append(v)

# =============================
# 5b) Frontera ACTIVA (máx. Sharpe vs TE)
# =============================
if bench_ret is not None and Sigma_te is not None:
    print("\n📈 5b) Frontera ACTIVA (Sharpe máximo vs TE)")

    r_bench = bench_ret.mean() * 252.0

    # grid de caps de TE (desde el mínimo factible + eps hasta algo más amplio)
    te_low = (TE_min + TE_RELAX_EPS_ABS) if TE_min is not None else 0.02
    te_high = max(te_low + 0.001, (TE_MAX or te_low) + 0.10)
    te_grid = np.linspace(te_low, te_high, 8)

    af_te, af_exret, af_ir = [], [], []
    for te_cap in te_grid:
        cons_af = [sum_to_one, *group_cons, make_te_ineq_with_jac(Sigma_te, c_te, varb_te, float(te_cap))]
        res_af = optimize_with_restarts(
            fun=lambda w: negative_sharpe(w, mean_ret[assets], cov.loc[assets, assets], RISK_FREE),
            x0_list=x0_list, bounds=bnds, constraints=cons_af
        )
        if res_af and res_af.success:
            w_star = res_af.x
            r_p, v_p = portfolio_performance(w_star, mean_ret[assets], cov.loc[assets, assets])
            te_p = te_index_ex_ante(w_star, Sigma_te, c_te, varb_te)
            ex_ret = r_p - r_bench
            ir = ex_ret / (te_p + 1e-12) if te_p is not None else np.nan
            af_te.append(te_p); af_exret.append(ex_ret); af_ir.append(ir)

    # Gráfico de Frontera ACTIVA (Exceso de retorno vs TE)
    try:
        if af_te:
            from pathlib import Path
            try:
                base_dir = Path(__file__).resolve().parent
            except NameError:
                base_dir = Path.cwd()
            png_dir = base_dir / "png"
            png_dir.mkdir(parents=True, exist_ok=True)

            fig_af, ax_af = plt.subplots(figsize=(9, 6))
            ax_af.plot(af_te, af_exret, marker="o")
            for x, y in zip(af_te, af_exret):
                ax_af.annotate(f"{x*100:.1f}%", (x, y), textcoords="offset points", xytext=(6,6), fontsize=8)
            ax_af.set_xlabel("Tracking Error (anual)")
            ax_af.set_ylabel("Exceso de retorno (anual) vs benchmark")
            ax_af.set_title("Frontera ACTIVA (máx. Sharpe)")
            ax_af.grid(True, alpha=0.3)
            fig_af.tight_layout()
            fig_af.savefig(png_dir / "frontera_activa.png", dpi=300)
            print("   - frontera_activa.png")
    except Exception as _e:
        pass

# =============================
# 6) Comparación, Recomendación y TE
# =============================
print("\n📊 6) COMPARACIÓN DE ESTRATEGIAS (medidas con Σ histórica)")
strategies = {
    "Maximum Sharpe": (w_ms, ret_ms, vol_ms, shp_ms),
    "Minimum Variance": (w_mv, ret_mv, vol_mv, (ret_mv - RISK_FREE) / (vol_mv + 1e-12)),
    "Risk Parity": (w_rp, ret_rp, vol_rp, (ret_rp - RISK_FREE) / (vol_rp + 1e-12)),
    "Max Diversification Ratio": (w_mdr, ret_mdr, vol_mdr, shp_mdr),
    "Black–Litterman": (w_bl, ret_bl, vol_bl, shp_bl),
}
if w_te_track is not None:
    strategies["Min TE (tracking)"] = (w_te_track, r_te, v_te, s_te)

for name, (w, r, v, s) in strategies.items():
    print(f"\n{name}:")
    print("  Pesos:", {a: f"{wi*100:5.1f}%" for a, wi in zip(assets, w)})
    print(f"  Ret {r*100:6.2f}% | Vol {v*100:6.2f}% | Sharpe {s:6.3f}")
    if bench_ret is not None:
        te_xa = te_index_ex_ante(w, Sigma_te, c_te, varb_te)
        if te_xa is not None:
            print(f"  TE vs índice (ex–ante): {te_xa*100:5.2f}%")

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

if bench_ret is not None:
    te_best = te_index_ex_ante(best_w, Sigma_te, c_te, varb_te)
    if te_best is not None:
        print(f"  TE vs índice (ex–ante): {te_best*100:5.2f}%")

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

# Frontera eficiente + puntos de estrategias + índice + CML
fig, ax = plt.subplots(figsize=(10, 7))
if ef_vols:
    ax.plot(ef_vols, ef_rets, lw=2, label="Frontera Eficiente")
# puntos de estrategias
colors = {
    "Maximum Sharpe": "tab:green",
    "Minimum Variance": "tab:blue",
    "Risk Parity": "tab:orange",
    "Max Diversification Ratio": "tab:red",
    "Black–Litterman": "tab:purple",
    "Min TE (tracking)": "tab:brown",
}
for name, (w, r, v, s) in strategies.items():
    ax.scatter([v], [r], s=60, label=name, color=colors.get(name, None), zorder=3)
    ax.annotate(name, (v, r), textcoords="offset points", xytext=(6,6), fontsize=8)
# punto del índice (benchmark externo)
r_b, v_b = None, None
if bench_ret is not None:
    r_b = bench_ret.mean() * 252.0
    v_b = bench_ret.std(ddof=1) * np.sqrt(252.0)
    ax.scatter([v_b], [r_b], s=60, label=f"Benchmark ({BENCHMARK})", marker="D", zorder=3)
    ax.annotate(f"{BENCHMARK}", (v_b, r_b), textcoords="offset points", xytext=(6,-12), fontsize=8)

# --- Capital Market Line (CML) usando la cartera tangente (Maximum Sharpe) ---
try:
    if vol_ms > 0:
        candidates = [vol_ms]
        if ef_vols:
            candidates += list(ef_vols)
        if v_b is not None:
            candidates.append(v_b)
        max_vol_plot = max(candidates) * 1.1

        xs = np.linspace(0.0, max_vol_plot, 100)
        slope = (ret_ms - RISK_FREE) / (vol_ms + 1e-12)
        ys = RISK_FREE + slope * xs

        ax.plot(xs, ys, linestyle="--", linewidth=1.6, label="Capital Market Line")
except Exception:
    pass

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

# =============================
# 8) Backtest con aportaciones y rebalanceo periódico
# =============================

# Parámetros del backtest (ajusta a tu gusto)
BT_INITIAL         = 10_000.0   # capital inicial
BT_MONTHLY_CONTR   = 300.0      # aportación mensual total
BT_CONTRIB_FREQ    = "M"        # frecuencia de aportación (mensual)
BT_REBAL_FREQ      = "2Q"       # frecuencia de rebalanceo: "2Q" semestral | "A" anual | "Q" trimestral | "M" mensual
BT_DRIFT_TOL       = 0.03       # no rebalancear si L1/2 < 3%
BT_COST_BPS        = 10         # costes (brokeraje/fees) en bps
BT_SLIPPAGE_BPS    = 5          # slippage en bps
BT_ROLL_SHARPE_WIN = 252        # ventana para Sharpe rodante (días)

def backtest_static_weights(returns_df: pd.DataFrame,
                            target_w: np.ndarray,
                            initial: float = 10_000.0,
                            monthly_contrib: float = 300.0,
                            contrib_freq: str = "M",
                            rebalance_freq: str = "2Q",
                            drift_tol: float = 0.03,
                            cost_bps: float = 10.0,
                            slip_bps: float = 5.0,
                            bench_ret_series: pd.Series | None = None):
    """
    Backtest estático con:
      - Pesos objetivo fijos 'target_w' (sum=1, >=0)
      - Aportaciones periódicas según 'contrib_freq' (p.ej. 'M')
      - Rebalanceo según 'rebalance_freq' (p.ej. '2Q' semestral o 'A' anual)
      - Rebalancear sólo si la deriva (L1/2) > drift_tol
      - Costes = (cost_bps + slip_bps) * turnover_dólares
      - Devuelve dict con métricas y series (TWR, Wealth, Sharpe, TE ex-post si hay benchmark)
    """
    dates = returns_df.index
    A = returns_df.values
    target = np.array(target_w, dtype=float)
    target = target / target.sum()

    # Calendarios independientes
    contrib_dates = set(returns_df.resample(contrib_freq).last().index)
    rebal_dates   = set(returns_df.resample(rebalance_freq).last().index)

    w = target.copy()
    V = float(initial)
    twr = 1.0

    equity_twr, wealth, port_ret = [], [], []
    turnover_sum = 0.0
    n_rebals = 0

    # coste inicial de entrar a los pesos objetivo
    init_turnover = 1.0
    init_cost = (cost_bps + slip_bps) * 1e-4 * (V * init_turnover)
    V -= init_cost
    if V <= 0:
        raise RuntimeError("Costes iniciales excesivos; revisa bps.")

    for t, dt in enumerate(dates):
        # 1) Evolución diaria por retornos
        r_vec = A[t, :]
        r_p = float(np.dot(w, r_vec))
        twr *= (1.0 + r_p)
        V   *= (1.0 + r_p)

        denom = (1.0 + r_p)
        if denom <= 0:
            denom = max(denom, 1e-12)
        w = (w * (1.0 + r_vec)) / denom  # nuevos pesos tras el día

        # 2) Aportación en calendario de contribución (invertida hacia el objetivo)
        if dt in contrib_dates:
            C = monthly_contrib
            V_post = V + C
            # Asigna la aportación con los pesos objetivo (empuja sin vender)
            w = (w * V + C * target) / V_post
            V = V_post

        # 3) Rebalanceo en su propio calendario (tras haber aportado, si coincide)
        if dt in rebal_dates:
            l1 = float(np.abs(w - target).sum())
            drift = 0.5 * l1
            if drift > drift_tol:
                turnover_frac = 0.5 * l1
                txn_cost = (cost_bps + slip_bps) * 1e-4 * (V * turnover_frac)
                V -= txn_cost
                w = target.copy()
                turnover_sum += turnover_frac
                n_rebals += 1

        # 4) Guardar series
        equity_twr.append(twr)
        wealth.append(V)
        port_ret.append(r_p)

    equity_twr = pd.Series(equity_twr, index=dates, name="TWR_Index")
    wealth     = pd.Series(wealth,     index=dates, name="Wealth")
    port_ret   = pd.Series(port_ret,   index=dates, name="Port_Ret")

    n_days  = len(port_ret)
    cagr_twr = equity_twr.iloc[-1] ** (252.0 / n_days) - 1.0
    vol_ann  = port_ret.std(ddof=1) * np.sqrt(252.0)
    sharpe   = (port_ret.mean() * 252.0 - RISK_FREE) / (vol_ann + 1e-12)
    roll_max = equity_twr.cummax()
    dd       = equity_twr / roll_max - 1.0
    mdd      = dd.min()

    realized_te = None
    if bench_ret_series is not None:
        aligned_bench = bench_ret_series.reindex(port_ret.index).dropna()
        aligned_port  = port_ret.reindex(aligned_bench.index)
        diff = aligned_port.values - aligned_bench.values
        realized_te = float(np.std(diff, ddof=1) * np.sqrt(252.0))

    return {
        "equity_twr": equity_twr,
        "wealth": wealth,
        "port_ret": port_ret,
        "cagr_twr": float(cagr_twr),
        "vol_ann": float(vol_ann),
        "sharpe": float(sharpe),
        "max_dd": float(mdd),
        "turnover_sum": float(turnover_sum),
        "n_rebals": int(n_rebals),
        "realized_te": realized_te,
    }

# Ejecutar backtest para cada estrategia con pesos fijos
bt_results = {}
for name, (w, r, v, s) in strategies.items():
    out = backtest_static_weights(
        returns_df=returns[assets],
        target_w=w,
        initial=BT_INITIAL,
        monthly_contrib=BT_MONTHLY_CONTR,
        contrib_freq=BT_CONTRIB_FREQ,   # <-- aportación mensual
        rebalance_freq=BT_REBAL_FREQ,   # <-- rebalanceo semestral/anual/etc.
        drift_tol=BT_DRIFT_TOL,
        cost_bps=BT_COST_BPS,
        slip_bps=BT_SLIPPAGE_BPS,
        bench_ret_series=bench_ret,
    )
    bt_results[name] = out

# =============================
# 9) Métricas y gráficos del backtest
# =============================
print("\n🧪 RESULTADOS BACKTEST (TWR, aportaciones y costes)")
for name, out in bt_results.items():
    te_txt = f" | Realized TE: {out['realized_te']*100:5.2f}%" if out["realized_te"] is not None else ""
    print(f"\n{name}:")
    print(f"  CAGR (TWR): {out['cagr_twr']*100:6.2f}% | Vol: {out['vol_ann']*100:6.2f}% | Sharpe: {out['sharpe']:6.3f}")
    print(f"  Max Drawdown: {out['max_dd']*100:6.2f}% | Turnover Σ: {out['turnover_sum']*100:6.2f}% | Rebalances: {out['n_rebals']}{te_txt}")
    print(f"  Wealth final: €{out['wealth'].iloc[-1]:,.0f} (Inicial €{BT_INITIAL:,.0f} + aport.)")

# Directorio de salida (ya creado antes)
from pathlib import Path
try:
    base_dir = Path(__file__).resolve().parent
except NameError:
    base_dir = Path.cwd()
png_dir = base_dir / "png"
png_dir.mkdir(parents=True, exist_ok=True)

# Curvas de TWR
fig, ax = plt.subplots(figsize=(10, 7))
for name, out in bt_results.items():
    ax.plot(out["equity_twr"].index, out["equity_twr"].values, label=name, lw=1.8)
ax.set_title("Equity TWR (sin flujos)")
ax.set_ylabel("Índice TWR (base 1)")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(png_dir / "backtest_equity_twr.png", dpi=300)

# Curvas de Wealth (con aportaciones y costes)
fig2, ax2 = plt.subplots(figsize=(10, 7))
for name, out in bt_results.items():
    ax2.plot(out["wealth"].index, out["wealth"].values, label=name, lw=1.8)
ax2.set_title("Wealth con aportaciones y costes")
ax2.set_ylabel("€")
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=8)
fig2.tight_layout()
fig2.savefig(png_dir / "backtest_wealth.png", dpi=300)

# Sharpe rodante
fig3, ax3 = plt.subplots(figsize=(10, 6))
for name, out in bt_results.items():
    pr = out["port_ret"]
    mu = pr.rolling(BT_ROLL_SHARPE_WIN).mean() * 252.0
    sd = pr.rolling(BT_ROLL_SHARPE_WIN).std(ddof=1) * np.sqrt(252.0)
    roll_sharpe = (mu - RISK_FREE) / (sd + 1e-12)
    ax3.plot(roll_sharpe.index, roll_sharpe.values, label=name, lw=1.2)
ax3.axhline(0.0, color="k", lw=0.8, ls="--", alpha=0.6)
ax3.set_title(f"Sharpe rodante ({BT_ROLL_SHARPE_WIN} días)")
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=8)
fig3.tight_layout()
fig3.savefig(png_dir / "backtest_sharpe_rodante.png", dpi=300)

print("\n🖼️  Gráficos de backtest guardados:")
print("   - backtest_equity_twr.png")
print("   - backtest_wealth.png")
print("   - backtest_sharpe_rodante.png")
