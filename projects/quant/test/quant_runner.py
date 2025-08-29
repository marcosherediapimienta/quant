#!/usr/bin/env python3
"""
Script ejecutable para análisis cuantitativo.
Solo ejecuta funciones, no las define.
"""

import sys
from pathlib import Path

# Importar funciones cuantitativas (desde quant/test/ hacia quant/)
sys.path.append(str(Path(__file__).parent.parent))
import quant_functions as qf

# Script principal sin funciones definidas aquí
print("🚀 ANÁLISIS CUANTITATIVO")
print("=" * 50)
print("📈 Obteniendo datos con agrupación por disponibilidad...")

try:
    # Obtener datos usando función de quant_functions
    print("📅 Descargando datos históricos...")
    prices = qf.get_market_data()
    
    if prices.empty:
        raise ValueError("No se obtuvieron datos de precios")
    
    print(f"✅ Datos descargados: {prices.shape}")
    print(f"📅 Período: {prices.index[0].strftime('%Y-%m-%d')} → {prices.index[-1].strftime('%Y-%m-%d')}")
    
    # Analizar retornos logarítmicos usando función de quant_functions
    print("\n📊 Calculando retornos logarítmicos...")
    print("\n📊 RETORNOS LOGARÍTMICOS:")
    print("=" * 50)
    
    # Usar función completa de análisis
    results = qf.analyze_log_returns(prices)
    
    # Generar gráficos de los retornos
    print("\n📊 GENERANDO GRÁFICOS...")
    print("=" * 50)
    
    all_returns = results['all_individual_returns']
    
    # Gráficos de timeline de retornos
    print("\n📈 Generando gráficos de timeline...")
    qf.plot_returns_timeline(all_returns, save_plots=True)
    
    # Gráficos de distribución
    print("\n📊 Generando gráficos de distribución...")
    qf.plot_returns_distribution(all_returns, save_plots=True)
    
    # Gráfico de retornos acumulados comparativos
    print("\n📈 Generando gráfico de retornos acumulados...")
    qf.plot_cumulative_returns(all_returns, save_plots=True)
    
    print("\n🎉 ANÁLISIS Y GRÁFICOS COMPLETADOS")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
