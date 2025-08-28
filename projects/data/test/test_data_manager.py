#!/usr/bin/env python3
"""
Script para probar el DataManager con agrupación por disponibilidad.
Soporta parámetros configurables y gestión de símbolos fallidos.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Agregar el directorio padre al path para importar data_manager
sys.path.append(str(Path(__file__).parent.parent))

from data_manager import DataManager


def process_group(name, prices, dm, all_plots):
    """
    Procesa un grupo de precios: imprime información, muestra precios y genera gráficos.
    
    Args:
        name (str): Nombre del grupo (ej: "LARGA HISTORIA")
        prices (pd.DataFrame): DataFrame con precios del grupo
        dm (DataManager): Instancia del DataManager
        all_plots (list): Lista para acumular rutas de gráficos generados
    """
    print(f"\n📊 GRUPO DE {name}:")
    print(f"   📈 {len(prices.columns)} símbolos")
    print(f"   📅 {prices.index[0].strftime('%Y-%m-%d')} → {prices.index[-1].strftime('%Y-%m-%d')}")
    
    # Mostrar precios actuales vs históricos
    print(f"\n💰 PRECIOS ACTUALES vs HISTÓRICOS:")
    current_prices = prices.iloc[-1]
    
    for symbol in prices.columns:
        current = current_prices[symbol]
        # Usar primer valor válido para precio histórico
        symbol_data = prices[symbol].dropna()
        historical = symbol_data.iloc[0] if len(symbol_data) > 0 else np.nan
        
        current_str = f'${current:>10.2f}' if pd.notna(current) else 'N/A'
        historical_str = f'${historical:>10.2f}' if pd.notna(historical) else 'N/A'
        
        print(f"   {symbol:>8}: {current_str:>12} ({historical_str:>12})")
    
    # Generar gráficos
    plot_path = dm.plot_price_history(prices=prices, save_plot=True, show_plot=False)
    all_plots.append(plot_path)
    
    individual_plots = dm.plot_individual_assets(
        prices=prices, 
        symbols=prices.columns, 
        save_plots=True, 
        show_plots=False
    )
    all_plots.extend(individual_plots)


def save_failed_symbols(failed_symbols, cache_dir):
    """
    Guarda los símbolos fallidos en un archivo CSV.
    
    Args:
        failed_symbols (list): Lista de símbolos que fallaron
        cache_dir (Path): Directorio cache donde guardar el archivo
    """
    if failed_symbols:
        failed_df = pd.DataFrame({
            'symbol': failed_symbols,
            'timestamp': pd.Timestamp.now(),
            'reason': 'Download failed'
        })
        failed_csv_path = cache_dir / 'failed_symbols.csv'
        failed_df.to_csv(failed_csv_path, index=False)
        print(f"💾 Símbolos fallidos guardados en: {failed_csv_path}")


def parse_arguments():
    """
    Parsea argumentos de línea de comandos.
    
    Returns:
        argparse.Namespace: Argumentos parseados
    """
    parser = argparse.ArgumentParser(
        description='Prueba del DataManager con agrupación por disponibilidad'
    )
    parser.add_argument(
        '--force-refresh', 
        action='store_true',
        help='Fuerza la actualización de datos (ignora cache)'
    )
    parser.add_argument(
        '--years', 
        type=int, 
        default=10,
        help='Número de años objetivo para la descarga (por defecto: 10)'
    )
    return parser.parse_args()


def main():
    """Función principal del script."""
    # Parsear argumentos
    args = parse_arguments()
    
    # Crear directorios locales si no existen
    test_dir = Path(__file__).parent
    plots_dir = test_dir / "plots"
    cache_dir = test_dir / "cache"
    plots_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)

    print("🚀 DATA MANAGER - PRUEBA CON AGRUPACIÓN")
    print("=" * 50)
    print(f"📋 Configuración: años={args.years}, force_refresh={args.force_refresh}")
    
    # Inicializar DataManager en modo prueba
    dm = DataManager(test_mode=True, test_dir=str(test_dir))
    
    # Símbolos a probar (diversificados)
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'IBM', 'IONQ', 'SMR', 
               'BTC-USD', 'ETH-USD', 'GLD', 
               'SPY', 'QQQ']
    
    print(f"📈 Probando con {len(symbols)} símbolos...")
    
    try:
        # Usar la nueva función agrupada
        print("📅 Descargando datos agrupados por disponibilidad...")
        grouped_data = dm.download_market_data_grouped(
            symbols=symbols,
            target_years=args.years,
            force_refresh=args.force_refresh
        )
        
        # Procesar cada grupo usando la función auxiliar
        all_plots = []
        
        # Mapeo de grupos con colores y descripciones
        group_info = {
            'long_history': ('🟢', f'LARGA HISTORIA (≥{args.years} años)'),
            'medium_history': ('🟡', f'HISTORIA MEDIA (5-{args.years} años)'),
            'short_history': ('🔴', 'HISTORIA CORTA (<5 años)')
        }
        
        for group_key, (emoji, description) in group_info.items():
            if group_key in grouped_data:
                process_group(description, grouped_data[group_key], dm, all_plots)
        
        # Gestionar símbolos fallidos
        failed_symbols = grouped_data.get('failed', [])
        if failed_symbols:
            print(f"\n❌ SÍMBOLOS FALLIDOS ({len(failed_symbols)} total):")
            for symbol in failed_symbols:
                print(f"   ❌ {symbol}")
            
            # Guardar símbolos fallidos en CSV
            save_failed_symbols(failed_symbols, cache_dir)
        
        print(f"\n🎉 PRUEBA COMPLETADA")
        print(f"📁 Total de archivos generados: {len(all_plots)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()