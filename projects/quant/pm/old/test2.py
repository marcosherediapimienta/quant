# Script para análisis DCA (Dollar Cost Averaging)
# Incluye conversión automática de USD a EUR usando tipo de cambio EURUSD=X

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from collections import OrderedDict
from datetime import datetime
import warnings

# Suprimir warnings específicos
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Configurar matplotlib para modo no interactivo
import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo

# Parámetros
tickers = ["0P0001CLDK.F","NVDA","MSFT","AAPL","GOOGL","IBM","AMZN","META","TSLA","JPM","BRK-A","BTC-EUR","GLD"]
aportacion = 300.0        # Aportación mensual en EUR por cada fondo
dia_mes = 1               # Día del mes para aportar (1-28 recomendado)
fecha_inicio = "2018-03-20"
fecha_fin = "2025-09-02"
# Ticker para tipo de cambio USD/EUR
FX_TICKER = "EURUSD=X"
# Tickers que necesitan conversión de USD a EUR
USD_TICKERS = {"NVDA","MSFT","AAPL","GOOGL","IBM","AMZN","META","TSLA","JPM","BRK-A","GLD"}

print(f"📊 Analizando tickers: {', '.join(tickers)}")
print(f"💰 Aportación mensual: {aportacion} EUR por cada fondo")

# Descargar histórico de los activos
print("📈 Descargando datos de los activos...")
data = yf.download(
    tickers,
    start=fecha_inicio,
    end=fecha_fin,
    auto_adjust=False,
    progress=False,
    interval="1d",
    group_by="ticker",
    threads=True
)

# Descargar tipo de cambio USD/EUR
print("💱 Descargando tipo de cambio USD/EUR...")
fx_data = yf.download(
    FX_TICKER,
    start=fecha_inicio,
    end=fecha_fin,
    auto_adjust=False,
    progress=False,
    interval="1d",
)
# Función para convertir USD a EUR
def convert_usd_to_eur(price_data, fx_data, ticker_name):
    """
    Convierte precios de USD a EUR usando el tipo de cambio.
    """
    if ticker_name in USD_TICKERS and fx_data is not None:
        # Obtener tipo de cambio USD/EUR
        if isinstance(fx_data.columns, pd.MultiIndex):
            # Buscar la columna correcta - puede ser ('Adj Close', 'EURUSD=X') o ('EURUSD=X', 'Adj Close')
            if ('Adj Close', FX_TICKER) in fx_data.columns:
                fx_col = ('Adj Close', FX_TICKER)
            elif (FX_TICKER, 'Adj Close') in fx_data.columns:
                fx_col = (FX_TICKER, 'Adj Close')
            else:
                fx_col = None
        else:
            fx_col = "Adj Close"
        
        if fx_col is not None and fx_col in fx_data.columns:
            fx_rate = fx_data[fx_col].dropna()
            
            # Alinear fechas y convertir USD a EUR
            aligned_fx = fx_rate.reindex(price_data.index, method='ffill')
            
            # Verificar que no hay NaN en el tipo de cambio
            if aligned_fx.isna().any():
                print(f"⚠️  Advertencia: Hay fechas sin tipo de cambio, usando forward fill")
                aligned_fx = aligned_fx.fillna(method='ffill').fillna(method='bfill')
            
            # Asegurar que price_data es una Serie
            if isinstance(price_data, pd.DataFrame):
                price_series = price_data.iloc[:, 0]
            else:
                price_series = price_data
            
            converted_prices = price_series / aligned_fx
            print(f"💱 Convertido {ticker_name} de USD a EUR usando tipo de cambio")
            # Asegurar que devolvemos una Serie, no un DataFrame
            if isinstance(converted_prices, pd.DataFrame):
                return converted_prices.iloc[:, 0]
            return converted_prices
        else:
            print(f"⚠️  No se encontró tipo de cambio para {ticker_name}, manteniendo en USD")
            return price_data
    else:
        return price_data

# Función para procesar múltiples tickers
def process_tickers(data, tickers, fx_data):
    """
    Procesa múltiples tickers y convierte USD a EUR si es necesario.
    """
    processed_data = {}
    
    for ticker in tickers:
        if isinstance(data.columns, pd.MultiIndex):
            # Múltiples tickers
            col = (ticker, "Adj Close")
            if col in data.columns:
                adj_close_raw = data[col].dropna().sort_index()
                if not adj_close_raw.empty:
                    # Convertir USD a EUR si está en USD_TICKERS
                    adj_close = convert_usd_to_eur(adj_close_raw, fx_data, ticker)
                    processed_data[ticker] = adj_close
                    print(f"✅ Procesado {ticker}: {len(adj_close)} observaciones")
                else:
                    print(f"⚠️  No hay datos para {ticker}")
            else:
                print(f"⚠️  No se encontró columna Adj Close para {ticker}")
        else:
            # Un solo ticker
            if "Adj Close" in data.columns:
                adj_close_raw = data["Adj Close"].dropna().sort_index()
                if not adj_close_raw.empty:
                    # Convertir USD a EUR si está en USD_TICKERS
                    adj_close = convert_usd_to_eur(adj_close_raw, fx_data, ticker)
                    processed_data[ticker] = adj_close
                    print(f"✅ Procesado {ticker}: {len(adj_close)} observaciones")
                else:
                    print(f"⚠️  No hay datos para {ticker}")
            else:
                print(f"⚠️  No se encontró columna Adj Close para {ticker}")
    
    return processed_data

# Procesar todos los tickers
processed_tickers = process_tickers(data, tickers, fx_data)

if not processed_tickers:
    raise SystemExit("No se pudieron procesar datos para ningún ticker.")

# Mostrar información de los datos procesados
print(f"\n📊 Información de los activos procesados:")
for ticker, adj_close in processed_tickers.items():
    precio_inicial = float(adj_close.iloc[0])
    precio_final = float(adj_close.iloc[-1])
    print(f"  {ticker}: {precio_inicial:.4f} EUR → {precio_final:.4f} EUR ({len(adj_close)} observaciones)")
    print(f"    Rango: {adj_close.index[0].date()} a {adj_close.index[-1].date()}")

# Función para simular DCA para un ticker específico
def simulate_dca(ticker, adj_close, aportacion, fechas_validas, fecha_fin_comun):
    """
    Simula DCA para un ticker específico.
    """
    # Mapear cada fecha al siguiente día con datos; manejar bordes + deduplicar ordenadamente
    fechas_validas_ticker = []
    for d in fechas_validas:
        pos = adj_close.index.searchsorted(d)
        if pos >= len(adj_close.index):
            # No hay siguiente hábil dentro del histórico -> ignorar esa aportación
            continue
        fecha_aportacion = adj_close.index[pos]
        # Asegurar que no exceda la fecha de fin común
        if fecha_aportacion <= fecha_fin_comun:
            fechas_validas_ticker.append(fecha_aportacion)

    # Quitar duplicados preservando orden
    fechas_validas_ticker = list(OrderedDict.fromkeys(fechas_validas_ticker))

    # Simulación de compras
    cash_invertido = 0.0
    participaciones = 0.0
    historial = []

    for f in fechas_validas_ticker:
        precio = float(adj_close.loc[f])
        qty = aportacion / precio
        participaciones += qty
        cash_invertido += aportacion
        patrimonio = participaciones * precio
        historial.append([f, precio, qty, participaciones, cash_invertido, patrimonio])

    df = pd.DataFrame(
        historial,
        columns=["Fecha", "Precio", "Participaciones_compradas", "Total_participaciones",
                 "Total_aportado", "Patrimonio"]
    ).set_index("Fecha").sort_index()

    return df, fechas_validas_ticker

# Fechas mensuales base (mes a mes)
aportaciones_fechas = pd.date_range(start=fecha_inicio, end=fecha_fin, freq="MS")
aportaciones_fechas = [d.replace(day=dia_mes) for d in aportaciones_fechas]

# Asegurar que todos los fondos tengan el mismo rango de fechas
# Encontrar la fecha de fin común más temprana
fecha_fin_comun = min([adj_close.index[-1] for adj_close in processed_tickers.values()])
print(f"📅 Fecha de fin común para todos los fondos: {fecha_fin_comun.date()}")

# Filtrar fechas de aportación para que no excedan la fecha de fin común
aportaciones_fechas = [d for d in aportaciones_fechas if d <= fecha_fin_comun]
print(f"📅 Número de aportaciones planificadas: {len(aportaciones_fechas)}")

# Simular DCA para cada ticker
dca_results = {}
for ticker, adj_close in processed_tickers.items():
    print(f"\n🔄 Simulando DCA para {ticker}...")
    df, fechas_validas_ticker = simulate_dca(ticker, adj_close, aportacion, aportaciones_fechas, fecha_fin_comun)
    dca_results[ticker] = {
        'df': df,
        'fechas_validas': fechas_validas_ticker,
        'adj_close': adj_close
    }
    print(f"✅ DCA completado para {ticker}: {len(fechas_validas_ticker)} aportaciones")

# Función para calcular métricas de DCA
def calculate_dca_metrics(df, adj_close, aportado_total):
    """
    Calcula todas las métricas de DCA para un ticker.
    """
    patrimonio_final = df["Patrimonio"].iloc[-1]
    rentabilidad_total = (patrimonio_final / aportado_total - 1) * 100
    años = (df.index[-1] - df.index[0]).days / 365.25

    # CAGR tradicional (incorrecto para DCA - asume inversión inicial única)
    cagr_tradicional = (patrimonio_final / aportado_total) ** (1 / años) - 1 if años > 0 else np.nan

    # CAGR ponderado por tiempo (más apropiado para DCA)
    def cagr_ponderado(df, patrimonio_final, años):
        if años <= 0:
            return np.nan
        capital_promedio = aportado_total / 2
        rendimiento_promedio = patrimonio_final / capital_promedio
        return rendimiento_promedio ** (1 / años) - 1

    cagr_ponderado_val = cagr_ponderado(df, patrimonio_final, años)

    # Métricas adicionales
    n_aport = df.shape[0]
    precio_actual = float(adj_close.iloc[-1])
    coste_medio = aportado_total / df["Total_participaciones"].iloc[-1]
    pl_no_real = patrimonio_final - aportado_total

    return {
        'patrimonio_final': patrimonio_final,
        'aportado_total': aportado_total,
        'rentabilidad_total': rentabilidad_total,
        'años': años,
        'cagr_tradicional': cagr_tradicional,
        'cagr_ponderado': cagr_ponderado_val,
        'n_aport': n_aport,
        'precio_actual': precio_actual,
        'coste_medio': coste_medio,
        'pl_no_real': pl_no_real
    }

# --- XIRR (rentabilidad dinero ponderada) ---
def year_fraction(d0, d1):
    return (d1 - d0).days / 365.0

def xnpv(rate, cashflows):
    # cashflows: lista de (fecha, importe). Aportaciones son negativas (salida de caja)
    t0 = cashflows[0][0]
    return sum(cf / ((1 + rate) ** year_fraction(t0, t)) for t, cf in cashflows)

def xirr(cashflows, guess=0.1):
    r = guess
    # Newton-Raphson
    for _ in range(50):
        f = xnpv(r, cashflows)
        df_ = (xnpv(r + 1e-6, cashflows) - f) / 1e-6
        if abs(df_) < 1e-12:
            break
        r_new = r - f / df_
        if abs(r_new - r) < 1e-10:
            return r_new
        r = r_new
    # Fallback: rejilla amplia
    grid = np.linspace(-0.99, 1.5, 20000)
    vals = [abs(xnpv(g, cashflows)) for g in grid]
    return float(grid[int(np.argmin(vals))])

# Calcular métricas para cada ticker
print(f"\n{'='*60}")
print(f"📊 RESULTADOS DCA COMPARATIVOS")
print(f"{'='*60}")

for ticker, result in dca_results.items():
    df = result['df']
    adj_close = result['adj_close']
    aportado_total = df["Total_aportado"].iloc[-1]
    
    # Calcular métricas
    metrics = calculate_dca_metrics(df, adj_close, aportado_total)
    
    # Calcular XIRR
    cashflows = []
    for f in df.index:
        cashflows.append((f, -aportacion))  # salida de caja
    cashflows.append((df.index[-1], metrics['patrimonio_final']))
    xirr_val = xirr(cashflows)
    
    # Mostrar resultados
    print(f"\n=== {ticker} ===")
    print(f"Período: {df.index[0].date()} → {df.index[-1].date()}")
    print(f"Nº aportaciones: {metrics['n_aport']}")
    print(f"Aportación total: {metrics['aportado_total']:,.2f} EUR")
    print(f"Patrimonio final: {metrics['patrimonio_final']:,.2f} EUR")
    print(f"P/L no realizado: {metrics['pl_no_real']:,.2f} EUR")
    print(f"Coste medio: {metrics['coste_medio']:,.4f} EUR  |  Precio actual: {metrics['precio_actual']:,.4f} EUR")
    print(f"Rentabilidad total: {metrics['rentabilidad_total']:.2f} %")
    print(f"CAGR tradicional: {metrics['cagr_tradicional']*100:.2f} % anual (incorrecto para DCA)")
    print(f"CAGR ponderado: {metrics['cagr_ponderado']*100:.2f} % anual (considera aportaciones escalonadas)")
    print(f"XIRR: {xirr_val*100:.2f} % anual (más preciso para DCA)")
    
    # Guardar métricas en el resultado
    result['metrics'] = metrics
    result['xirr'] = xirr_val

# Crear series diarias para gráficos
series_diarias = {}

for ticker, result in dca_results.items():
    df = result['df']
    adj_close = result['adj_close']
    
    # Serie diaria de patrimonio y aportado
    serie = pd.DataFrame(index=adj_close.index)
    serie["Precio"] = adj_close
    
    # Nº de participaciones en cada fecha: hacer forward-fill desde compras
    p_shares = pd.Series(0.0, index=adj_close.index)
    p_shares.loc[df.index] = df["Total_participaciones"].values
    p_shares = p_shares.replace(0, np.nan).ffill()
    serie["Participaciones"] = p_shares
    serie["Patrimonio"] = serie["Precio"] * serie["Participaciones"]
    serie["Total_aportado"] = 0.0
    serie.loc[df.index, "Total_aportado"] = aportacion
    serie["Total_aportado"] = serie["Total_aportado"].cumsum()
    
    series_diarias[ticker] = serie


# ---------- Gráficos Comparativos ----------
import os
png_dir = os.path.join(os.path.dirname(__file__), 'png')
os.makedirs(png_dir, exist_ok=True)

# Gráfico 1: Comparación de patrimonios
plt.figure(figsize=(15, 8))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
for i, (ticker, serie) in enumerate(series_diarias.items()):
    color = colors[i % len(colors)]  # Usar módulo para evitar índice fuera de rango
    plt.plot(serie.index, serie["Patrimonio"], 
             label=f"{ticker} - Patrimonio", linewidth=2, color=color)
    plt.plot(serie.index, serie["Total_aportado"], 
             label=f"{ticker} - Capital aportado", linestyle="--", linewidth=2, 
             color=color, alpha=0.7)

plt.title("Comparación DCA: Cartera Diversificada (300 EUR/mes cada activo)", fontsize=14, fontweight='bold')
plt.xlabel("Fecha", fontsize=12)
plt.ylabel("EUR", fontsize=12)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()

png_path = os.path.join(png_dir, 'comparacion_dca.png')
plt.savefig(png_path, dpi=300, bbox_inches='tight')
print(f"📊 Gráfico comparativo guardado como '{png_path}'")

# Gráfico 2: Rendimientos acumulados
plt.figure(figsize=(15, 6))
for i, (ticker, serie) in enumerate(series_diarias.items()):
    color = colors[i % len(colors)]  # Usar módulo para evitar índice fuera de rango
    rendimiento_acum = (serie["Patrimonio"] / serie["Total_aportado"] - 1) * 100
    plt.plot(serie.index, rendimiento_acum, 
             label=f"{ticker} - Rendimiento acumulado (%)", linewidth=2, color=color)

plt.title("Rendimiento Acumulado DCA: Cartera Diversificada", fontsize=14, fontweight='bold')
plt.xlabel("Fecha", fontsize=12)
plt.ylabel("Rendimiento Acumulado (%)", fontsize=12)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.tight_layout()

png_path2 = os.path.join(png_dir, 'rendimiento_acumulado_dca.png')
plt.savefig(png_path2, dpi=300, bbox_inches='tight')
print(f"📊 Gráfico de rendimientos guardado como '{png_path2}'")

# Gráfico 3: Individual para cada fondo
for ticker, serie in series_diarias.items():
    plt.figure(figsize=(12, 6))
    plt.plot(serie.index, serie["Patrimonio"], label="Patrimonio (valor de mercado)", linewidth=2)
    plt.plot(serie.index, serie["Total_aportado"], label="Capital aportado", linestyle="--", linewidth=2)
    plt.fill_between(
        serie.index, serie["Total_aportado"], serie["Patrimonio"],
        where=(serie["Patrimonio"] > serie["Total_aportado"]), alpha=0.2, label="Plusvalía"
    )
    plt.fill_between(
        serie.index, serie["Patrimonio"], serie["Total_aportado"],
        where=(serie["Patrimonio"] < serie["Total_aportado"]), alpha=0.2, label="Pérdida"
    )
    plt.title(f"DCA Individual: {ticker} (300 EUR/mes)", fontsize=14, fontweight='bold')
    plt.xlabel("Fecha", fontsize=12)
    plt.ylabel("EUR", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    png_path_individual = os.path.join(png_dir, f'dca_individual_{ticker}.png')
    plt.savefig(png_path_individual, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico individual guardado como '{png_path_individual}'")
