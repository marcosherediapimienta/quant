# -*- coding: utf-8 -*-
"""
Análisis de rendimientos históricos con conversión automática USD→EUR
- Descarga datos con yfinance
- Convierte de USD a EUR usando EURUSD=X (si está disponible)
- Calcula rendimientos logarítmicos
- Genera 4 gráficos: precios normalizados, rendimientos diarios, histograma y correlación
- Imprime estadísticas y correlaciones

Requisitos:
    pip install yfinance pandas numpy matplotlib seaborn
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("Instala las dependencias:  pip install yfinance pandas numpy matplotlib seaborn")

# =========================
# Configuración del análisis
# =========================
# Cartera diversificada (puedes cambiarla)
TICKERS = ["0P0001CLDK.F","NVDA","MSFT","AAPL","GOOGL","IBM","AMZN","META","TSLA","JPM","BRK-A","BTC-EUR","GLD"]

# Ticker para tipo de cambio USD/EUR (EUR por 1 USD -> ojo: en Yahoo es EURUSD=X = EUR por USD)
FX_TICKER = "EURUSD=X"

# Tickers que necesitan conversión de USD a EUR
USD_TICKERS = {"NVDA","MSFT","AAPL","GOOGL","IBM","AMZN","META","TSLA","JPM","BRK-A","GLD"}

# Tasa libre de riesgo (anual, en términos nominales)
Rf = 0.0375

# Rango temporal
START_DATE = "2004-01-01"
END_DATE = None  # hasta hoy

# =========================
# Funciones auxiliares
# =========================
def _get_price_col(df: pd.DataFrame, tk: str | None = None) -> pd.Series | None:
    """
    Devuelve la serie de precios priorizando 'Adj Close' y, si no existe, 'Close'.
    Si df es MultiIndex, usa (tk, col). Si es un DataFrame simple, usa 'col'.
    """
    candidates = ["Adj Close", "Close"]
    if isinstance(df.columns, pd.MultiIndex):
        for c in candidates:
            col = (tk, c)
            if col in df.columns:
                return df[col].rename(tk)
    else:
        for c in candidates:
            if c in df.columns:
                name = tk if tk else c
                return df[c].rename(name)
    return None


def extract_adj_close(df: pd.DataFrame, tickers: list[str], fx_df: pd.DataFrame | None = None):
    """
    Extrae columnas de precio (Adj Close/Close) para los tickers.
    Si el ticker está en USD y hay fx_df (EURUSD=X), convierte a EUR.
    Devuelve:
        - DataFrame con precios (columnas = tickers presentes)
        - conjunto 'converted_set' con los tickers convertidos USD→EUR
    """
    out = {}
    converted_set = set()

    # Serie del tipo de cambio con fallback Adj Close -> Close
    fx_series = None
    if fx_df is not None:
        if isinstance(fx_df.columns, pd.MultiIndex):
            fx_series = _get_price_col(fx_df, FX_TICKER)
        else:
            fx_series = _get_price_col(fx_df)
    if fx_series is not None:
        fx_series = fx_series.sort_index().ffill()

    if isinstance(df.columns, pd.MultiIndex):
        for tk in tickers:
            s = _get_price_col(df, tk)
            if s is None:
                print(f"⚠️  No se encontró columna de precio para {tk}. Se omite.")
                continue
            s = s.sort_index()

            # Conversión a EUR si aplica
            if tk in USD_TICKERS and fx_series is not None:
                aligned_fx = fx_series.reindex(s.index).ffill().bfill()
                s = s / aligned_fx
                converted_set.add(tk)
                print(f"💱 Convertido {tk} de USD a EUR usando {FX_TICKER}.")
            elif tk in USD_TICKERS and fx_series is None:
                print(f"⚠️  No hay {FX_TICKER}. {tk} permanece en USD.")
            out[tk] = s
    else:
        # Caso extremo de un solo ticker
        tk = tickers[0]
        s = _get_price_col(df, tk)
        if s is not None:
            s = s.sort_index()
            if tk in USD_TICKERS and fx_series is not None:
                aligned_fx = fx_series.reindex(s.index).ffill().bfill()
                s = s / aligned_fx
                converted_set.add(tk)
                print(f"💱 Convertido {tk} de USD a EUR usando {FX_TICKER}.")
            elif tk in USD_TICKERS and fx_series is None:
                print(f"⚠️  No hay {FX_TICKER}. {tk} permanece en USD.")
            out[tk] = s
        else:
            print(f"⚠️  No se encontró columna de precio para {tk}.")
    return (pd.concat(out, axis=1) if out else pd.DataFrame(), converted_set)


def infer_currency_label(tk: str, converted_set: set[str]) -> str:
    """
    Etiqueta de moneda para impresión informativa.
    - Si fue convertido: 'EUR (convertido)'
    - Si el ticker incluye '-EUR' o es de Frankfurt '.F': 'EUR'
    - Si es USD_ticker y no se convirtió: 'USD'
    - En otro caso: 'Moneda original'
    """
    if tk in converted_set:
        return "EUR (convertido)"
    if "-EUR" in tk or tk.endswith(".F") or tk.endswith(".DE") or tk.endswith(".PA") or tk.endswith(".MI"):
        return "EUR"
    if tk in USD_TICKERS:
        return "USD"
    return "Moneda original"


def main():
    # =========================
    # Descarga de datos
    # =========================
    print("📥 Descargando datos históricos desde 2004...")
    print("📈 Descargando precios de activos...")
    data = yf.download(
        tickers=TICKERS,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=False,
        progress=True,
        group_by="ticker",
        threads=True,
    )

    print("💱 Descargando tipo de cambio USD/EUR...")
    fx_data = yf.download(
        tickers=FX_TICKER,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=False,
        progress=True,
        group_by="ticker",
        threads=True,
    )

    # =========================
    # Preparación de precios
    # =========================
    adj_close, converted_set = extract_adj_close(data, TICKERS, fx_data)
    adj_close = adj_close.dropna(how="all").sort_index()

    if adj_close.empty:
        raise SystemExit("No se pudo obtener precios para los tickers.")

    present_tickers = list(adj_close.columns)
    missing = sorted(set(TICKERS) - set(present_tickers))
    if missing:
        print(f"⚠️  Estos tickers no devolvieron datos y se omiten: {', '.join(missing)}")
    print(f"✅ Tickers usados: {', '.join(present_tickers)}")

    # Fechas comunes: intersección de datos para los tickers disponibles
    print("🔍 Buscando fecha de inicio común para los activos...")
    common_data = adj_close[present_tickers].dropna(how="any")
    if common_data.empty:
        raise SystemExit("No hay fechas comunes con datos para los activos disponibles.")

    start_date = common_data.index[0]
    print(f"📅 Rango de fechas descargado: {adj_close.index[0].date()} → {adj_close.index[-1].date()}")
    print(f"✅ Fecha de inicio común: {start_date.date()}")

    adj_close_filtered = common_data
    print(f"📊 Datos filtrados: {len(adj_close_filtered)} observaciones desde {start_date.date()}")

    # =========================
    # Rendimientos logarítmicos
    # =========================
    log_returns = np.log(adj_close_filtered).diff().dropna(how="all")

    # Muestras rápidas
    print("\n=== Primeras filas de Adj Close (filtrados) ===")
    print(adj_close_filtered.head())
    print("\n=== Primeras filas de rendimientos logarítmicos ===")
    print(log_returns.head())

    # =========================
    # Gráficos
    # =========================
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Análisis de Rendimientos Históricos', fontsize=16, fontweight='bold')

    # 1) Precios normalizados (base 100) en escala log
    normalized_prices = adj_close_filtered / adj_close_filtered.iloc[0] * 100
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    for i, tk in enumerate(present_tickers):
        color = colors[i % len(colors)]
        axes[0, 0].plot(normalized_prices.index, normalized_prices[tk], label=tk, linewidth=2, color=color)
    axes[0, 0].set_title('Precios Ajustados Normalizados (Base 100)')
    axes[0, 0].set_ylabel('Precio Normalizado')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')

    # 2) Rendimientos logarítmicos diarios
    for i, tk in enumerate(present_tickers):
        color = colors[i % len(colors)]
        axes[0, 1].plot(log_returns.index, log_returns[tk], label=tk, alpha=0.7, color=color)
    axes[0, 1].set_title('Rendimientos Logarítmicos Diarios')
    axes[0, 1].set_ylabel('Rendimiento Log')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)

    # 3) Histograma de rendimientos
    for i, tk in enumerate(present_tickers):
        color = colors[i % len(colors)]
        axes[1, 0].hist(log_returns[tk].dropna(), bins=50, alpha=0.7, label=tk, density=True, color=color)
    axes[1, 0].set_title('Distribución de Rendimientos')
    axes[1, 0].set_xlabel('Rendimiento Log')
    axes[1, 0].set_ylabel('Densidad')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4) Matriz de correlación
    if len(present_tickers) > 1:
        corr_matrix = log_returns[present_tickers].dropna().corr()
        im = axes[1, 1].imshow(corr_matrix.values, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)

        axes[1, 1].set_xticks(range(len(present_tickers)))
        axes[1, 1].set_yticks(range(len(present_tickers)))
        axes[1, 1].set_xticklabels(present_tickers, fontsize=8, rotation=45)
        axes[1, 1].set_yticklabels(present_tickers, fontsize=8)

        for i in range(len(present_tickers)):
            for j in range(len(present_tickers)):
                axes[1, 1].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                ha="center", va="center", color="black",
                                fontweight='bold', fontsize=8)

        cbar = plt.colorbar(im, ax=axes[1, 1], shrink=0.8)
        cbar.set_label('Correlación', fontsize=10)
        axes[1, 1].set_title('Matriz de Correlación')
    else:
        # Si solo hay un activo, muestra estadísticas en el cuarto panel
        tk = present_tickers[0]
        stats_text = (
            f"Estadísticas de {tk}:\n"
            f"Media: {log_returns[tk].mean():.4f}\n"
            f"Desv. Est.: {log_returns[tk].std():.4f}\n"
            f"Skewness: {log_returns[tk].skew():.4f}\n"
            f"Kurtosis: {log_returns[tk].kurtosis():.4f}\n"
        )
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        axes[1, 1].set_title(f'Estadísticas de {tk}')
        axes[1, 1].axis('off')

    # Evita cortar el título
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # =========================
    # Guardado de gráfico
    # =========================
    try:
        base_dir = Path(__file__).resolve().parent
    except NameError:
        base_dir = Path.cwd()

    png_dir = base_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    png_path = png_dir / "rendimientos_historicos.png"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Gráfico guardado como '{png_path}'")

    # Mostrar interactivo si es posible
    try:
        plt.show()
    except Exception:
        print("⚠️  No se puede mostrar el gráfico interactivamente, pero se ha guardado como archivo PNG")

    # =========================
    # Estadísticas adicionales
    # =========================
    print("\n=== Estadísticas de Rendimientos ===")
    print(f"Período: {log_returns.index[0].strftime('%Y-%m-%d')} → {log_returns.index[-1].strftime('%Y-%m-%d')}")
    print(f"Número de observaciones: {len(log_returns)}")
    print(f"Fecha de inicio común: {start_date.strftime('%Y-%m-%d')}")
    print(f"Tasa libre de riesgo (Rf): {Rf*100:.1f}% anual")

    print(f"\n=== Normalización de Precios (Base 100) ===")
    print(f"Precios iniciales ({start_date.strftime('%Y-%m-%d')}):")
    for tk in present_tickers:
        initial_price = adj_close_filtered[tk].iloc[0]
        final_price = adj_close_filtered[tk].iloc[-1]
        normalized_final = normalized_prices[tk].iloc[-1]
        currency_label = infer_currency_label(tk, converted_set)
        print(f"  {tk}: {currency_label} {initial_price:.4f} → {currency_label} {final_price:.4f} (Normalizado: {normalized_final:.2f})")

    for tk in present_tickers:
        mu_d = log_returns[tk].mean()
        sd_d = log_returns[tk].std()
        print(f"\n{tk}:")
        print(f"  Rendimiento medio diario: {mu_d:.4f}")
        print(f"  Volatilidad diaria: {sd_d:.4f}")
        print(f"  Rendimiento anualizado (≈252 días): {mu_d * 252:.4f}")
        print(f"  Volatilidad anualizada (≈√252): {sd_d * np.sqrt(252):.4f}")
        excess_return = mu_d * 252 - Rf
        volatility = sd_d * np.sqrt(252)
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0.0
        print(f"  Sharpe ratio (Rf={Rf*100:.1f}%): {sharpe_ratio:.4f}")

    if len(present_tickers) > 1:
        corr_matrix = log_returns[present_tickers].corr()
        if corr_matrix.shape[0] > 1:
            tri = np.triu_indices_from(corr_matrix.values, k=1)
            if tri[0].size > 0:
                avg_correlation = float(corr_matrix.values[tri].mean())
                print(f"\nCorrelación promedio entre todos los activos: {avg_correlation:.4f}")
        print("\nMatriz de Correlación:")
        print(corr_matrix.round(4))


if __name__ == "__main__":
    main()
