import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("Instala yfinance:  pip install yfinance pandas numpy")

# Script para análisis de rendimientos históricos
# Incluye conversión automática de EEM de USD a EUR usando tipo de cambio EURUSD=X

# Tickers solicitados - Cartera diversificada
TICKERS = ["0P0001CLDK.F", "AMZN", "GLD", "BTC-EUR"]
# Ticker para tipo de cambio USD/EUR
FX_TICKER = "EURUSD=X"
# Tickers que necesitan conversión de USD a EUR
USD_TICKERS = {"AMZN", "GLD"}

# Tasa libre de riesgo (Rf) 
Rf = 0.045 

# Descarga todo el histórico disponible desde 2004 hasta hoy
print("📥 Descargando datos históricos desde 2004...")
print("📈 Descargando precios de activos...")
data = yf.download(
    tickers=TICKERS,
    start="2004-01-01",  # Desde 2004
    end=None,            # Hasta hoy
    interval="1d",
    auto_adjust=False,   
    progress=True,       # Mostrar progreso de descarga
    group_by="ticker",
    threads=True
)

print("💱 Descargando tipo de cambio USD/EUR...")
fx_data = yf.download(
    tickers=FX_TICKER,
    start="2004-01-01",
    end=None,
    interval="1d",
    auto_adjust=False,
    progress=True,
    group_by="ticker",
    threads=True
)

# Extraer Adj Close en columnas por ticker y convertir USD a EUR
def extract_adj_close(df, tickers, fx_df=None):
    if isinstance(df.columns, pd.MultiIndex):
        out = {}
        for tk in tickers:
            col = (tk, "Adj Close")
            if col in df.columns:
                price_data = df[col].rename(tk)
                
                # Si el ticker está en USD, convertir a EUR
                if tk in USD_TICKERS and fx_df is not None:
                    # Obtener tipo de cambio USD/EUR
                    if isinstance(fx_df.columns, pd.MultiIndex):
                        fx_col = (FX_TICKER, "Adj Close")
                    else:
                        fx_col = "Adj Close"
                    
                    if fx_col in fx_df.columns:
                        fx_rate = fx_df[fx_col]
                        # Alinear fechas y convertir USD a EUR
                        aligned_fx = fx_rate.reindex(price_data.index, method='ffill')
                        price_data = price_data / aligned_fx
                        print(f"💱 Convertido {tk} de USD a EUR usando tipo de cambio")
                    else:
                        print(f"⚠️  No se encontró tipo de cambio para {tk}, manteniendo en USD")
                
                out[tk] = price_data
        return pd.concat(out, axis=1) if out else pd.DataFrame()
    else:
        # caso de un solo ticker
        price_data = df["Adj Close"].to_frame(name=tickers[0]) if "Adj Close" in df.columns else pd.DataFrame()
        
        # Si el ticker está en USD y hay datos de tipo de cambio, convertir
        if tickers[0] in USD_TICKERS and fx_df is not None and not price_data.empty:
            if isinstance(fx_df.columns, pd.MultiIndex):
                fx_col = (FX_TICKER, "Adj Close")
            else:
                fx_col = "Adj Close"
            
            if fx_col in fx_df.columns:
                fx_rate = fx_df[fx_col]
                aligned_fx = fx_rate.reindex(price_data.index, method='ffill')
                price_data.iloc[:, 0] = price_data.iloc[:, 0] / aligned_fx
                print(f"💱 Convertido {tickers[0]} de USD a EUR usando tipo de cambio")
        
        return price_data

adj_close = extract_adj_close(data, TICKERS, fx_data).dropna(how="all").sort_index()


if adj_close.empty:
    raise SystemExit("No se pudo obtener 'Adj Close' para los tickers.")

print(f"📅 Rango de fechas disponible: {adj_close.index[0].strftime('%Y-%m-%d')} a {adj_close.index[-1].strftime('%Y-%m-%d')}")

# Encontrar la fecha más temprana donde ambos fondos tienen datos
print("🔍 Buscando fecha de inicio común para ambos fondos...")
common_data = adj_close.dropna(subset=TICKERS, how="any")
if common_data.empty:
    raise SystemExit("No hay fechas comunes con datos para ambos fondos.")

start_date = common_data.index[0]
print(f"✅ Fecha de inicio común: {start_date.strftime('%Y-%m-%d')}")

# Filtrar datos desde la fecha común
adj_close_filtered = common_data
print(f"📊 Datos filtrados: {len(adj_close_filtered)} observaciones desde {start_date.strftime('%Y-%m-%d')}")

# Rendimiento logarítmico por día: ln(P_t / P_{t-1})
log_returns = np.log(adj_close_filtered).diff().dropna(how="all")

# Alinear y filtrar solo fechas donde ambos tienen datos
aligned_logret = log_returns.dropna(subset=TICKERS, how="any")

# Mostrar un resumen rápido
print("\n=== Primeras filas de Adj Close (filtrados) ===")
print(adj_close_filtered.head())
print("\n=== Primeras filas de log-returns (por ticker) ===")
print(log_returns.head())

# Configurar el estilo de los gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Crear figura con subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Análisis de Rendimientos Históricos', fontsize=16, fontweight='bold')

# 1. Gráfico de precios ajustados normalizados (escala logarítmica)
# Normalizar precios a 100 en la fecha de inicio
normalized_prices = adj_close_filtered / adj_close_filtered.iloc[0] * 100

# Colores para múltiples activos
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

# Plotear todos los activos
for i, ticker in enumerate(TICKERS):
    if i < len(normalized_prices.columns):
        color = colors[i % len(colors)]
        axes[0, 0].plot(normalized_prices.index, normalized_prices.iloc[:, i], 
                        label=ticker, linewidth=2, color=color)

axes[0, 0].set_title('Precios Ajustados Normalizados (Base 100)')
axes[0, 0].set_ylabel('Precio Normalizado')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_yscale('log')  # Escala logarítmica

# 2. Gráfico de rendimientos logarítmicos
for i, ticker in enumerate(TICKERS):
    if i < len(log_returns.columns):
        color = colors[i % len(colors)]
        axes[0, 1].plot(log_returns.index, log_returns.iloc[:, i], 
                        label=ticker, alpha=0.7, color=color)

axes[0, 1].set_title('Rendimientos Logarítmicos Diarios')
axes[0, 1].set_ylabel('Rendimiento Log')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)

# 3. Histograma de rendimientos
for i, ticker in enumerate(TICKERS):
    if i < len(log_returns.columns):
        color = colors[i % len(colors)]
        axes[1, 0].hist(log_returns.iloc[:, i], bins=50, alpha=0.7, 
                        label=ticker, density=True, color=color)

axes[1, 0].set_title('Distribución de Rendimientos')
axes[1, 0].set_xlabel('Rendimiento Log')
axes[1, 0].set_ylabel('Densidad')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Matriz de correlación (para múltiples tickers)
if len(TICKERS) > 1:
    # Crear heatmap de correlación
    corr_matrix = log_returns.corr()
    im = axes[1, 1].imshow(corr_matrix.values, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # Configurar ticks
    axes[1, 1].set_xticks(range(len(TICKERS)))
    axes[1, 1].set_yticks(range(len(TICKERS)))
    axes[1, 1].set_xticklabels(TICKERS, fontsize=8, rotation=45)
    axes[1, 1].set_yticklabels(TICKERS, fontsize=8)
    
    # Añadir valores de correlación
    for i in range(len(TICKERS)):
        for j in range(len(TICKERS)):
            text = axes[1, 1].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontweight='bold', fontsize=8)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=axes[1, 1], shrink=0.8)
    cbar.set_label('Correlación', fontsize=10)
    
    axes[1, 1].set_title('Matriz de Correlación')
else:
    # Si solo hay un ticker, mostrar estadísticas
    stats_text = f"""
    Estadísticas de {TICKERS[0]}:
    Media: {log_returns.iloc[:, 0].mean():.4f}
    Desv. Est.: {log_returns.iloc[:, 0].std():.4f}
    Skewness: {log_returns.iloc[:, 0].skew():.4f}
    Kurtosis: {log_returns.iloc[:, 0].kurtosis():.4f}
    """
    axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                    fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    axes[1, 1].set_title(f'Estadísticas de {TICKERS[0]}')
    axes[1, 1].axis('off')

plt.tight_layout()

# Guardar el gráfico como archivo PNG en la subcarpeta png
import os
png_dir = os.path.join(os.path.dirname(__file__), 'png')
os.makedirs(png_dir, exist_ok=True)
png_path = os.path.join(png_dir, 'rendimientos_historicos.png')
plt.savefig(png_path, dpi=300, bbox_inches='tight')
print(f"\n📊 Gráfico guardado como '{png_path}'")

# También mostrar el gráfico si es posible
try:
    plt.show()
except:
    print("⚠️  No se puede mostrar el gráfico interactivamente, pero se ha guardado como archivo PNG")

# Mostrar estadísticas adicionales
print("\n=== Estadísticas de Rendimientos ===")
print(f"Período: {log_returns.index[0].strftime('%Y-%m-%d')} a {log_returns.index[-1].strftime('%Y-%m-%d')}")
print(f"Número de observaciones: {len(log_returns)}")
print(f"Fecha de inicio común: {start_date.strftime('%Y-%m-%d')}")
print(f"Tasa libre de riesgo (Rf): {Rf*100:.1f}% anual")

# Mostrar información de normalización
print(f"\n=== Normalización de Precios (Base 100) ===")
print(f"Precios iniciales ({start_date.strftime('%Y-%m-%d')}):")
for i, ticker in enumerate(TICKERS):
    if i < len(adj_close_filtered.columns):
        initial_price = adj_close_filtered.iloc[0, i]
        final_price = adj_close_filtered.iloc[-1, i]
        normalized_final = normalized_prices.iloc[-1, i]
        currency = "EUR"  # Ambos en EUR ahora
        print(f"  {ticker}: {currency} {initial_price:.4f} → {currency} {final_price:.4f} (Normalizado: {normalized_final:.2f})")

for i, ticker in enumerate(TICKERS):
    if i < len(log_returns.columns):
        print(f"\n{ticker}:")
        print(f"  Rendimiento medio diario: {log_returns.iloc[:, i].mean():.4f}")
        print(f"  Volatilidad diaria: {log_returns.iloc[:, i].std():.4f}")
        print(f"  Rendimiento anualizado: {log_returns.iloc[:, i].mean() * 252:.4f}")
        print(f"  Volatilidad anualizada: {log_returns.iloc[:, i].std() * np.sqrt(252):.4f}")
        # Calcular Sharpe ratio con Rf
        excess_return = log_returns.iloc[:, i].mean() * 252 - Rf
        volatility = log_returns.iloc[:, i].std() * np.sqrt(252)
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        print(f"  Sharpe ratio (Rf={Rf*100:.1f}%): {sharpe_ratio:.4f}")

if len(TICKERS) > 1:
    # Calcular correlación promedio para múltiples activos
    corr_matrix = log_returns.corr()
    avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
    print(f"\nCorrelación promedio entre todos los activos: {avg_correlation:.4f}")
    
    # Mostrar matriz de correlación
    print("\nMatriz de Correlación:")
    print(corr_matrix.round(4))
