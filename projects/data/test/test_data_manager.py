#!/usr/bin/env python3
"""
Script simple para probar el DataManager con agrupación por disponibilidad.
"""

import sys
from pathlib import Path
import pandas as pd

# Agregar el directorio padre al path para importar data_manager
sys.path.append(str(Path(__file__).parent.parent))

from data_manager import DataManager

# Crear directorios locales si no existen
test_dir = Path(__file__).parent
plots_dir = test_dir / "plots"
cache_dir = test_dir / "cache"
plots_dir.mkdir(exist_ok=True)
cache_dir.mkdir(exist_ok=True)

print("🚀 DATA MANAGER - PRUEBA CON AGRUPACIÓN")
print("=" * 50)

# Inicializar DataManager en modo prueba
dm = DataManager(test_mode=True, test_dir=str(test_dir))

# Símbolos a probar (diversificados)
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'IBM', 'IONQ', 'SMR', 
           'BTC-EUR', 'ETH-EUR', 
           'SPY', 'QQQ']

print(f"📈 Probando con {len(symbols)} símbolos...")

try:
    # Usar la nueva función agrupada
    print("📅 Descargando datos agrupados por disponibilidad...")
    grouped_data = dm.download_market_data_grouped(
        symbols=symbols,
        target_years=10,
        force_refresh=True
    )
    
    # Procesar cada grupo
    all_plots = []
    
    if 'long_history' in grouped_data:
        print(f"\n🟢 GRUPO DE LARGA HISTORIA (≥10 años):")
        long_prices = grouped_data['long_history']
        print(f"   📊 {len(long_prices.columns)} símbolos")
        print(f"   📅 {long_prices.index[0].strftime('%Y-%m-%d')} → {long_prices.index[-1].strftime('%Y-%m-%d')}")
        
        # Mostrar precios actuales vs históricos
        print(f"\n💰 PRECIOS ACTUALES vs HISTÓRICOS:")
        current_prices = long_prices.iloc[-1]
        historical_prices = long_prices.iloc[0]
        
        for symbol in long_prices.columns:
            current = current_prices[symbol]
            historical = historical_prices[symbol]
            if pd.isna(current) or pd.isna(historical):
                print(f"   {symbol:>8}: {'N/A':>10} ({'N/A':>10})")
            else:
                print(f"   {symbol:>8}: ${current:>10.2f} (${historical:>10.2f})")
        
        # Gráfico principal para este grupo
        plot_path = dm.plot_price_history(prices=long_prices, save_plot=True, show_plot=False)
        all_plots.append(plot_path)
        
        # Gráficos individuales
        individual_plots = dm.plot_individual_assets(prices=long_prices, symbols=long_prices.columns, save_plots=True, show_plots=False)
        all_plots.extend(individual_plots)
    
    if 'medium_history' in grouped_data:
        print(f"\n🟡 GRUPO DE HISTORIA MEDIA (5-10 años):")
        medium_prices = grouped_data['medium_history']
        print(f"   {len(medium_prices.columns)} símbolos")
        print(f"   📅 {medium_prices.index[0].strftime('%Y-%m-%d')} → {medium_prices.index[-1].strftime('%Y-%m-%d')}")
        
        # Mostrar precios actuales vs históricos
        print(f"\n💰 PRECIOS ACTUALES vs HISTÓRICOS:")
        current_prices = medium_prices.iloc[-1]
        historical_prices = medium_prices.iloc[0]
        
        for symbol in medium_prices.columns:
            current = current_prices[symbol]
            historical = historical_prices[symbol]
            if pd.isna(current) or pd.isna(historical):
                print(f"   {symbol:>8}: {'N/A':>10} ({'N/A':>10})")
            else:
                print(f"   {symbol:>8}: ${current:>10.2f} (${historical:>10.2f})")
        
        # Gráfico principal para este grupo
        plot_path = dm.plot_price_history(prices=medium_prices, save_plot=True, show_plot=False)
        all_plots.append(plot_path)
        
        # Gráficos individuales
        individual_plots = dm.plot_individual_assets(prices=medium_prices, symbols=medium_prices.columns, save_plots=True, show_plots=False)
        all_plots.extend(individual_plots)
    
    if 'short_history' in grouped_data:
        print(f"\n🔴 GRUPO DE HISTORIA CORTA (<5 años):")
        short_prices = grouped_data['short_history']
        print(f"   📊 {len(short_prices.columns)} símbolos")
        print(f"   📅 {short_prices.index[0].strftime('%Y-%m-%d')} → {short_prices.index[-1].strftime('%Y-%m-%d')}")
        
        # Mostrar precios actuales vs históricos
        print(f"\n💰 PRECIOS ACTUALES vs HISTÓRICOS:")
        current_prices = short_prices.iloc[-1]
        historical_prices = short_prices.iloc[0]
        
        for symbol in short_prices.columns:
            current = current_prices[symbol]
            historical = historical_prices[symbol]
            if pd.isna(current) or pd.isna(historical):
                print(f"   {symbol:>8}: {'N/A':>10} ({'N/A':>10})")
            else:
                print(f"   {symbol:>8}: ${current:>10.2f} (${historical:>10.2f})")
        
        # Gráfico principal para este grupo
        plot_path = dm.plot_price_history(prices=short_prices, save_plot=True, show_plot=False)
        all_plots.append(plot_path)
        
        # Gráficos individuales
        individual_plots = dm.plot_individual_assets(prices=short_prices, symbols=short_prices.columns, save_plots=True, show_plots=False)
        all_plots.extend(individual_plots)
    
    if 'failed' in grouped_data:
        print(f"\n❌ SÍMBOLOS FALLIDOS:")
        for symbol in grouped_data['failed']:
            print(f"   ❌ {symbol}")
    
    print(f"\n🎉 PRUEBA COMPLETADA")
    print(f"📁 Total de archivos generados: {len(all_plots)}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
