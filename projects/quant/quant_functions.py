#!/usr/bin/env python3
"""
Funciones de análisis cuantitativo para datos financieros.
Todas las funciones reciben DataFrames de precios con símbolos en columnas e índice de fechas.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend no-interactivo para evitar warnings
import matplotlib.pyplot as plt
from typing import Union


def calculate_returns(prices: pd.DataFrame, freq: str = "daily") -> pd.DataFrame:
    """
    Calcula retornos de precios.
    
    Args:
        prices: DataFrame con precios históricos (índice: fechas, columnas: símbolos)
        freq: Frecuencia de retornos ("daily", "monthly", "weekly")
        
    Returns:
        DataFrame con retornos calculados
    """
    if prices.empty:
        raise ValueError("DataFrame de precios está vacío")
    
    if freq == "daily":
        returns = prices.pct_change(fill_method=None)
    elif freq == "weekly":
        returns = prices.resample('W').last().pct_change(fill_method=None)
    elif freq == "monthly":
        returns = prices.resample('M').last().pct_change(fill_method=None)
    else:
        raise ValueError("freq debe ser 'daily', 'weekly' o 'monthly'")
    
    return returns.dropna()


def calculate_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula retornos logarítmicos.
    
    Args:
        prices: DataFrame con precios históricos
        
    Returns:
        DataFrame con retornos logarítmicos
    """
    if prices.empty:
        raise ValueError("DataFrame de precios está vacío")
    
    # Verificar que no hay valores negativos o cero
    if (prices <= 0).any().any():
        raise ValueError("Los precios deben ser estrictamente positivos para calcular retornos logarítmicos")
    
    # Calcular retornos logarítmicos
    log_returns = np.log(prices / prices.shift(1))
    
    return log_returns.dropna()

def get_symbols():
    """Obtiene la lista de símbolos definida en test_data_manager."""
    return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'IBM', 'IONQ', 'SMR', 
            'BTC-USD', 'ETH-USD', 'GLD', 'ACWI', 'SPY', 'QQQ']


def get_market_data():
    """Obtiene datos usando EXACTAMENTE la misma lógica que test_data_manager.py."""
    import sys
    from pathlib import Path
    
    # Importar DataManager
    sys.path.append(str(Path(__file__).parent.parent / "data"))
    from data_manager import DataManager
    
    symbols = get_symbols()
    
    # Usar DataManager exactamente como en test_data_manager.py
    test_dir = Path(__file__).parent.parent / "data" / "test"
    dm = DataManager(test_mode=True, test_dir=str(test_dir))
    
    # Usar 10 años como en test_data_manager.py (que funciona y muestra historial completo)
    grouped_data = dm.download_market_data_grouped(
        symbols=symbols,
        target_years=10,  # Mismo que test_data_manager.py
        force_refresh=False
    )
    
    # Usar la misma función que test_data_manager.py para procesar y mostrar
    all_plots = dm.process_and_display_groups(grouped_data, target_years=10)
    
    # Combinar grupos conservando el historial completo de cada activo
    all_prices = pd.DataFrame()
    
    for group_name, prices in grouped_data.items():
        if isinstance(prices, pd.DataFrame) and not prices.empty:
            if all_prices.empty:
                all_prices = prices.copy()
            else:
                all_prices = all_prices.join(prices, how='outer')
    
    return all_prices


def analyze_log_returns(prices: pd.DataFrame,) -> dict:
    """Muestra todos los retornos logarítmicos diarios de cada activo desde su primer día hasta hoy."""
    from pathlib import Path
    
    # Mostrar información general del DataFrame
    print(f"\n📈 DATOS CARGADOS: {prices.shape}")
    print(f"📅 Rango total: {prices.index[0].strftime('%Y-%m-%d')} → {prices.index[-1].strftime('%Y-%m-%d')}")
    
    # Analizar cada activo individualmente mostrando TODOS los retornos
    print("\n📊 RETORNOS LOGARÍTMICOS DIARIOS POR ACTIVO:")
    print("=" * 80)
    
    all_individual_returns = {}
    
    for symbol in prices.columns:
        # Obtener datos del activo eliminando NaN
        symbol_prices = prices[symbol].dropna()
        
        if len(symbol_prices) < 2:
            print(f"   {symbol:>8}: ⚠️  Datos insuficientes")
            continue
            
        # Calcular retornos logarítmicos para este activo
        symbol_log_returns = calculate_log_returns(symbol_prices.to_frame())
        symbol_log_returns = symbol_log_returns[symbol]  # Convertir a Serie
        
        # Información del período
        trading_days = len(symbol_log_returns)
        start_date = symbol_log_returns.index[0].strftime('%Y-%m-%d')
        end_date = symbol_log_returns.index[-1].strftime('%Y-%m-%d')
        
        # Mostrar encabezado del activo
        print(f"\n🔹 {symbol}: {start_date} → {end_date} ({trading_days} días)")
        print("-" * 60)
        
        # Mostrar los 3 primeros y 3 últimos retornos
        print("   📊 Primeros 3 retornos:")
        for i, (date, return_value) in enumerate(symbol_log_returns.head(3).items()):
            date_str = date.strftime('%Y-%m-%d')
            print(f"      {date_str}: {return_value:>12.8f}")
        
        if trading_days > 6:
            print("      ...")
            print("   📊 Últimos 3 retornos:")
            for date, return_value in symbol_log_returns.tail(3).items():
                date_str = date.strftime('%Y-%m-%d')
                print(f"      {date_str}: {return_value:>12.8f}")
        elif trading_days > 3:
            print("   📊 Últimos retornos:")
            for date, return_value in symbol_log_returns.tail(trading_days - 3).items():
                date_str = date.strftime('%Y-%m-%d')
                print(f"      {date_str}: {return_value:>12.8f}")
        
        # Guardar resultados
        all_individual_returns[symbol] = symbol_log_returns
    
    print(f"\n📊 RESUMEN:")
    print(f"   📊 Activos procesados: {len(all_individual_returns)}")
    total_returns = sum(len(returns) for returns in all_individual_returns.values())
    print(f"   📊 Total de retornos diarios: {total_returns}")

    return {
        'all_individual_returns': all_individual_returns,
        'total_daily_returns': total_returns
    }


def plot_returns_timeline(all_returns: dict, save_plots: bool = True) -> None:
    """
    Grafica los retornos logarítmicos de cada activo a lo largo del tiempo.
    
    Args:
        all_returns: Diccionario con retornos por activo
        save_plots: Si guardar los gráficos como archivos
    """
    from pathlib import Path
    
    # Crear directorio de gráficos en test/plots/
    if save_plots:
        plots_dir = Path(__file__).parent / "test" / "plots"
        plots_dir.mkdir(exist_ok=True)
    
    for symbol, returns in all_returns.items():
        plt.figure(figsize=(12, 6))
        
        # Gráfico de retornos
        plt.plot(returns.index, returns.values, linewidth=0.8, alpha=0.7)
        plt.title(f'Retornos Logarítmicos Diarios - {symbol}', fontsize=14, fontweight='bold')
        plt.xlabel('Fecha')
        plt.ylabel('Retorno Logarítmico')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Línea en cero
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Estadísticas en el gráfico
        mean_return = returns.mean()
        std_return = returns.std()
        plt.text(0.02, 0.98, f'Media: {mean_return:.6f}\nDesv.Est: {std_return:.6f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(plots_dir / f'{symbol}_returns.png', dpi=300, bbox_inches='tight')
            print(f"   📊 Gráfico guardado: {symbol}_returns.png")
        
        plt.close()  # Liberar memoria


def plot_returns_distribution(all_returns: dict, save_plots: bool = True) -> None:
    """
    Grafica la distribución de retornos de cada activo.
    
    Args:
        all_returns: Diccionario con retornos por activo
        save_plots: Si guardar los gráficos como archivos
    """
    from pathlib import Path
    
    # Crear directorio de gráficos en test/plots/
    if save_plots:
        plots_dir = Path(__file__).parent / "test" / "plots"
        plots_dir.mkdir(exist_ok=True)
    
    for symbol, returns in all_returns.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histograma
        ax1.hist(returns.values, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax1.set_title(f'Distribución de Retornos - {symbol}', fontweight='bold')
        ax1.set_xlabel('Retorno Logarítmico')
        ax1.set_ylabel('Frecuencia')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        # Box plot
        ax2.boxplot(returns.values, vert=True)
        ax2.set_title(f'Box Plot - {symbol}', fontweight='bold')
        ax2.set_ylabel('Retorno Logarítmico')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(plots_dir / f'{symbol}_distribution.png', dpi=300, bbox_inches='tight')
            print(f"   📊 Gráfico guardado: {symbol}_distribution.png")
        
        plt.close()  # Liberar memoria


def plot_cumulative_returns(all_returns: dict, save_plots: bool = True) -> None:
    """
    Grafica los retornos acumulados de cada activo.
    
    Args:
        all_returns: Diccionario con retornos por activo
        save_plots: Si guardar los gráficos como archivos
    """
    from pathlib import Path
    
    # Crear directorio de gráficos en test/plots/
    if save_plots:
        plots_dir = Path(__file__).parent / "test" / "plots"
        plots_dir.mkdir(exist_ok=True)
    
    plt.figure(figsize=(15, 8))
    
    for symbol, returns in all_returns.items():
        # Calcular retornos acumulados
        cumulative = (1 + returns).cumprod()
        plt.plot(cumulative.index, cumulative.values, label=symbol, linewidth=1.5)
    
    plt.title('Retornos Acumulados - Todos los Activos', fontsize=16, fontweight='bold')
    plt.xlabel('Fecha')
    plt.ylabel('Valor Acumulado (Base = 1)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Línea en 1 (sin cambio)
    plt.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(plots_dir / 'all_cumulative_returns.png', dpi=300, bbox_inches='tight')
        print(f"   📊 Gráfico guardado: all_cumulative_returns.png")
    
    plt.close()  # Liberar memoria


if __name__ == "__main__":
    print("📊 Funciones de Análisis Cuantitativo")
    print("=" * 50)
    print("Funciones disponibles:")
    print("- get_symbols(): Lista de símbolos")
    print("- get_market_data(): Obtener datos del DataManager")
    print("- analyze_log_returns(): Análisis completo de retornos logarítmicos")
    print("- plot_returns_timeline(): Gráfica retornos a lo largo del tiempo")
    print("- plot_returns_distribution(): Gráfica distribución de retornos")
    print("- plot_cumulative_returns(): Gráfica retornos acumulados")
