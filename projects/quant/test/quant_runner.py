#!/usr/bin/env python3
"""
Script ejecutable para análisis cuantitativo completo.
Solo ejecuta funciones, no las define.
Utiliza todas las funciones descriptivas y métricas de quant_functions.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Importar funciones cuantitativas (desde quant/test/ hacia quant/)
sys.path.append(str(Path(__file__).parent.parent))
import quant_functions as qf

# Script principal sin funciones definidas aquí
print("🚀 ANÁLISIS CUANTITATIVO COMPLETO")
print("=" * 70)
print("📈 Obteniendo datos con agrupación por disponibilidad...")

try:
    # Obtener datos usando función de quant_functions
    print("📅 Descargando datos históricos...")
    prices = qf.get_market_data()
    
    if prices.empty:
        raise ValueError("No se obtuvieron datos de precios")
    
    print(f"✅ Datos descargados: {prices.shape}")
    print(f"📅 Período: {prices.index[0].strftime('%Y-%m-%d')} → {prices.index[-1].strftime('%Y-%m-%d')}")
    
    # Calcular retornos logarítmicos
    print("\n📊 Calculando retornos logarítmicos...")
    returns = qf.calculate_log_returns(prices)
    print(f"✅ Retornos calculados: {returns.shape}")
    
    # ==========================================
    # ESTADÍSTICAS DESCRIPTIVAS
    # ==========================================
    print("\n📊 1. ESTADÍSTICAS DESCRIPTIVAS BÁSICAS")
    print("=" * 70)
    
    # Asimetría (Skewness)
    print("\n🔸 ASIMETRÍA (Skewness) - TODOS LOS ACTIVOS:")
    skewness = qf.calculate_skewness(returns)
    for symbol, value in skewness.items():
        interpretation = "simétrica" if abs(value) < 0.1 else ("sesgo positivo" if value > 0 else "sesgo negativo")
        print(f"   {symbol:>8}: {value:>8.4f} ({interpretation})")
    
    # Curtosis (Kurtosis)
    print("\n🔸 CURTOSIS (Kurtosis - Fisher) - TODOS LOS ACTIVOS:")
    kurtosis = qf.calculate_kurtosis(returns)
    for symbol, value in kurtosis.items():
        interpretation = "normal" if abs(value) < 1 else ("colas pesadas" if value > 0 else "colas ligeras")
        print(f"   {symbol:>8}: {value:>8.4f} ({interpretation})")
    
    # Estadísticas descriptivas completas
    print("\n🔸 ESTADÍSTICAS COMPLETAS - TODOS LOS ACTIVOS:")
    desc_stats = qf.calculate_descriptive_stats(returns)
    display_cols = ['count', 'mean', 'std', 'skewness', 'kurtosis', 'min', 'max']
    print(desc_stats[display_cols].round(6))
    
    # Percentiles importantes
    print("\n🔸 PERCENTILES CLAVE - TODOS LOS ACTIVOS:")
    percentiles = qf.calculate_percentiles(returns, [5, 25, 50, 75, 95])
    print(percentiles.round(6))
    
    # ==========================================
    # CORRELACIÓN Y COVARIANZA
    # ==========================================
    print("\n🔗 2. ANÁLISIS DE CORRELACIÓN")
    print("=" * 70)
    
    # Matriz de correlación
    print("\n🔸 MATRIZ DE CORRELACIÓN COMPLETA:")
    corr_matrix = qf.calculate_correlation_matrix(returns)
    print(corr_matrix.round(3))
    
    # Estadísticas de correlación
    corr_values = corr_matrix.values
    np.fill_diagonal(corr_values, np.nan)
    max_corr = np.nanmax(corr_values)
    min_corr = np.nanmin(corr_values)
    mean_corr = np.nanmean(corr_values)
    print(f"\n   📊 Correlación máxima: {max_corr:.3f}")
    print(f"   📊 Correlación mínima: {min_corr:.3f}")
    print(f"   📊 Correlación promedio: {mean_corr:.3f}")
    
    # Matriz de covarianza
    print("\n🔸 MATRIZ DE COVARIANZA:")
    cov_matrix = qf.calculate_covariance_matrix(returns)
    print(cov_matrix.round(8))
    
    # ==========================================
    # MÉTRICAS DE RIESGO
    # ==========================================
    print("\n⚠️ 3. MÉTRICAS DE RIESGO")
    print("=" * 70)
    
    # Volatilidad anualizada
    print("\n🔸 VOLATILIDAD ANUALIZADA - TODOS LOS ACTIVOS:")
    volatility = qf.calculate_volatility(returns, annualize=True)
    for symbol, value in volatility.items():
        print(f"   {symbol:>8}: {value:>8.4f} ({value*100:.2f}%)")
    
    # Máximo drawdown
    print("\n🔸 MÁXIMO DRAWDOWN - TODOS LOS ACTIVOS:")
    max_dd = qf.calculate_max_drawdown(prices)
    for symbol, value in max_dd.items():
        print(f"   {symbol:>8}: {value:>8.4f} ({value*100:.2f}%)")
    
    # ==========================================
    # MÉTRICAS DE RENDIMIENTO
    # ==========================================
    print("\n📈 4. MÉTRICAS DE RENDIMIENTO")
    print("=" * 70)
    
    # Sharpe Ratio
    print("\n🔸 RATIO DE SHARPE (rf=2%) - TODOS LOS ACTIVOS:")
    sharpe = qf.calculate_sharpe_ratio(returns, risk_free_rate=0.02)
    for symbol, value in sharpe.items():
        interpretation = "excelente" if value > 2 else ("bueno" if value > 1 else ("aceptable" if value > 0.5 else "pobre"))
        print(f"   {symbol:>8}: {value:>8.4f} ({interpretation})")
    
    # Sortino Ratio
    print("\n🔸 RATIO DE SORTINO (rf=2%) - TODOS LOS ACTIVOS:")
    sortino = qf.calculate_sortino_ratio(returns, risk_free_rate=0.02)
    for symbol, value in sortino.items():
        if np.isinf(value):
            interpretation = "infinito (sin downside)"
        else:
            interpretation = "excelente" if value > 2 else ("bueno" if value > 1 else ("aceptable" if value > 0.5 else "pobre"))
        print(f"   {symbol:>8}: {value:>8.4f} ({interpretation})")
    
    # Calmar Ratio
    print("\n🔸 RATIO DE CALMAR - TODOS LOS ACTIVOS:")
    calmar = qf.calculate_calmar_ratio(returns, prices)
    for symbol, value in calmar.items():
        if np.isinf(value) or np.isnan(value):
            interpretation = "N/A"
        else:
            interpretation = "excelente" if value > 1 else ("bueno" if value > 0.5 else "pobre")
        print(f"   {symbol:>8}: {value:>8.4f} ({interpretation})")
    
    # Beta vs SPY (si está disponible)
    if 'SPY' in returns.columns:
        print("\n🔸 BETA (vs SPY como mercado) - TODOS LOS ACTIVOS:")
        spy_returns = returns['SPY']
        other_returns = returns.drop('SPY', axis=1)
        betas = qf.calculate_beta(other_returns, spy_returns, risk_free_rate=0.02)
        for symbol, value in betas.items():
            interpretation = "defensivo" if value < 1 else ("neutral" if abs(value - 1) < 0.1 else "agresivo")
            print(f"   {symbol:>8}: {value:>8.4f} ({interpretation})")
        
        # Information Ratio vs SPY
        print("\n🔸 RATIO DE INFORMACIÓN (vs SPY) - TODOS LOS ACTIVOS:")
        info_ratios = qf.calculate_information_ratio(other_returns, spy_returns)
        for symbol, value in info_ratios.items():
            if np.isinf(value) or np.isnan(value):
                interpretation = "N/A"
            else:
                interpretation = "superior" if value > 0.5 else ("igual" if abs(value) < 0.1 else "inferior")
            print(f"   {symbol:>8}: {value:>8.4f} ({interpretation})")
    
    # ==========================================
    # ANÁLISIS DETALLADO DE RETORNOS
    # ==========================================
    print("\n📊 5. ANÁLISIS DETALLADO DE RETORNOS")
    print("=" * 70)
    
    # Usar función completa de análisis de retornos
    results = qf.analyze_log_returns(prices)
    
    # ==========================================
    # GRÁFICOS
    # ==========================================
    print("\n📊 6. GENERANDO GRÁFICOS")
    print("=" * 70)
    
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
    
    # Gráfico de volatilidades
    print("\n📈 Generando gráfico de volatilidades...")
    qf.plot_volatility_comparison(volatility, save_plots=True)
    
    # Gráfico de matriz de correlación (heatmap)
    print("\n🔗 Generando heatmap de matriz de correlación...")
    qf.plot_correlation_heatmap(corr_matrix, save_plots=True)
    
    # ==========================================
    # RESUMEN FINAL
    # ==========================================
    print("\n🎉 ANÁLISIS CUANTITATIVO COMPLETADO")
    print("=" * 70)
    print("📊 Funciones ejecutadas de quant_functions.py:")
    print("   ✅ Estadísticas descriptivas (skewness, kurtosis, percentiles)")
    print("   ✅ Correlación y covarianza")
    print("   ✅ Métricas de riesgo (volatilidad, drawdown)")
    print("   ✅ Métricas de rendimiento (Sharpe, Sortino, Calmar, Information)")
    print("   ✅ Beta y ratio de información")
    print("   ✅ Análisis detallado de retornos logarítmicos")
    print("   ✅ Gráficos completos (timeline, distribución, acumulados, volatilidades, correlación)")
    print(f"\n📊 Total de activos analizados: {len(returns.columns)}")
    print(f"📊 Total de días de retornos: {len(returns)}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
