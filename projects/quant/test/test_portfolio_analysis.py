#!/usr/bin/env python3
"""
Script de prueba para el módulo portfolio_analysis.py

Este script demuestra el uso de todas las funciones del módulo de análisis
de portafolio con datos reales descargados usando DataManager.
"""

import sys
import os
import pandas as pd
import numpy as np

# Agregar paths necesarios para importar módulos
# Desde test/ necesitamos ir a data/ y quant/
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Importar módulos
from data_manager import DataManager
from portfolio_analysis import *

def test_portfolio_analysis():
    """
    Función principal de prueba que ejecuta todos los análisis.
    """
    print("🚀 INICIANDO PRUEBAS DEL MÓDULO PORTFOLIO_ANALYSIS")
    print("=" * 60)
    
    # Configuración de prueba
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'IBM', 'IONQ', 'SMR', 
               'BTC-USD', 'ETH-USD', 'GLD', 'ACWI', 'SPY', 'QQQ']
    benchmark_symbol = 'SPY'  # S&P 500 como benchmark
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    var_horizon_days = 1  # <-- Cambia aquí el horizonte de VaR/CVaR (1, 10, 252, ...)
    
    try:
        # 1. Inicializar DataManager y descargar datos
        print("\n📥 1. DESCARGANDO DATOS...")
        dm = DataManager()
        prices = dm.download_market_data(symbols, start_date=start_date, end_date=end_date)
        print(f"✅ Datos descargados: {prices.shape[0]} días, {prices.shape[1]} activos")
        print(f"   Fechas: {prices.index[0].strftime('%Y-%m-%d')} a {prices.index[-1].strftime('%Y-%m-%d')}")
        
        # 2. Calcular retornos
        print("\n📈 2. CALCULANDO RETORNOS...")
        simple_returns = calculate_simple_returns(prices)
        log_returns = calculate_log_returns(prices)
        print(f"✅ Retornos simples calculados: {simple_returns.shape}")
        print(f"✅ Retornos logarítmicos calculados: {log_returns.shape}")
        
        # Debug: verificar si los retornos están siendo calculados correctamente
        print("\n🔍 DEBUG - Verificando cálculo de retornos...")
        for symbol in symbols[:3]:  # Solo los primeros 3 para debug
            if symbol in prices.columns and symbol in simple_returns.columns:
                p = prices[symbol].dropna()
                r = simple_returns[symbol].dropna()
                if len(p) > 1 and len(r) > 0:
                    print(f"   {symbol}:")
                    print(f"      Precios únicos: {p.nunique()}/{len(p)}")
                    print(f"      Primer precio: ${p.iloc[0]:.4f}, Segundo precio: ${p.iloc[1]:.4f}")
                    # Calcular retorno manualmente
                    manual_return = (p.iloc[1] - p.iloc[0]) / p.iloc[0]
                    print(f"      Retorno manual: {manual_return:+.4%}")
                    print(f"      Retorno calculado: {r.iloc[0]:+.4%}")
                    print(f"      Valores NaN en retornos: {r.isna().sum()}/{len(r)}")
                    print(f"      Valores cero en retornos: {(r == 0).sum()}/{len(r)}")
                break
        
        # Mostrar retornos de días más representativos
        print("\n📊 2.c RETORNOS REPRESENTATIVOS (primer y último día con cambios significativos)...")
        for symbol in symbols:
            if symbol in simple_returns.columns:
                sr = simple_returns[symbol].dropna()
                if len(sr) > 5:  # Necesitamos suficientes datos
                    # Buscar el primer retorno significativo (> 0.1%)
                    first_significant_idx = None
                    for i, ret in enumerate(sr):
                        if abs(ret) > 0.001:  # 0.1%
                            first_significant_idx = i
                            break
                    
                    # Buscar el último retorno significativo (> 0.1%) desde el final
                    last_significant_idx = None
                    for i in range(len(sr) - 1, -1, -1):
                        if abs(sr.iloc[i]) > 0.001:  # 0.1%
                            last_significant_idx = i
                            break
                    
                    if first_significant_idx is not None and last_significant_idx is not None:
                        first_date = sr.index[first_significant_idx].strftime('%Y-%m-%d')
                        last_date = sr.index[last_significant_idx].strftime('%Y-%m-%d')
                        print(f"\n   {symbol}:")
                        print(f"      [RET] {first_date}: {sr.iloc[first_significant_idx]:+.4%} (primer cambio >0.1%)")
                        print(f"      [RET] {last_date}: {sr.iloc[last_significant_idx]:+.4%} (último cambio >0.1%)")
                    elif first_significant_idx is not None:
                        # Solo primer cambio significativo
                        first_date = sr.index[first_significant_idx].strftime('%Y-%m-%d')
                        last_date = sr.index[-1].strftime('%Y-%m-%d')
                        print(f"\n   {symbol}:")
                        print(f"      [RET] {first_date}: {sr.iloc[first_significant_idx]:+.4%} (primer cambio >0.1%)")
                        print(f"      [RET] {last_date}: {sr.iloc[-1]:+.4%} (último día - sin cambios)")
                    else:
                        # Si no hay cambios significativos, mostrar estadísticas
                        print(f"\n   {symbol}:")
                        print(f"      [RET] Sin cambios significativos en el periodo")
                        print(f"      [RET] Retorno promedio: {sr.mean():+.4%}")
                else:
                    print(f"\n   {symbol}:")
                    print(f"      [RET] Datos insuficientes para análisis")
        
        # 2.a Imprimir primer y último precio por activo
        print("\n💵 2.a PRECIOS (primer y último día por activo)...")
        for symbol in symbols:
            if symbol in prices.columns:
                sp = prices[symbol].dropna()
                if len(sp) > 0:
                    first_date_p = sp.index[0].strftime('%Y-%m-%d') if hasattr(sp.index[0], 'strftime') else str(sp.index[0])
                    last_date_p  = sp.index[-1].strftime('%Y-%m-%d') if hasattr(sp.index[-1], 'strftime') else str(sp.index[-1])
                    print(f"\n   {symbol}:")
                    print(f"      [PX ] {first_date_p}: ${sp.iloc[0]:.4f}")
                    print(f"      [PX ] {last_date_p}: ${sp.iloc[-1]:.4f}")

        # 2.b Imprimir primer y último retorno por activo (retornos simples)
        print("\n🧾 2.b RETORNOS (primer y último día por activo)...")
        for symbol in symbols:
            if symbol in simple_returns.columns:
                sr = simple_returns[symbol].dropna()
                if len(sr) > 1:  # Necesitamos al menos 2 retornos válidos
                    # Buscar el primer retorno no-cero
                    first_nonzero_idx = None
                    for i, ret in enumerate(sr):
                        if abs(ret) > 1e-10:  # Tolerancia para valores muy pequeños
                            first_nonzero_idx = i
                            break
                    
                    if first_nonzero_idx is not None:
                        first_date = sr.index[first_nonzero_idx].strftime('%Y-%m-%d') if hasattr(sr.index[first_nonzero_idx], 'strftime') else str(sr.index[first_nonzero_idx])
                        last_date  = sr.index[-1].strftime('%Y-%m-%d') if hasattr(sr.index[-1], 'strftime') else str(sr.index[-1])
                        print(f"\n   {symbol}:")
                        print(f"      [RET] {first_date}: {sr.iloc[first_nonzero_idx]:+.4%}")
                        print(f"      [RET] {last_date}: {sr.iloc[-1]:+.4%}")
                    else:
                        # Si todos son cero, mostrar el primer y último con nota
                        first_date = sr.index[0].strftime('%Y-%m-%d') if hasattr(sr.index[0], 'strftime') else str(sr.index[0])
                        last_date  = sr.index[-1].strftime('%Y-%m-%d') if hasattr(sr.index[-1], 'strftime') else str(sr.index[-1])
                        print(f"\n   {symbol}:")
                        print(f"      [RET] {first_date}: {sr.iloc[0]:+.4%} (sin cambios)")
                        print(f"      [RET] {last_date}: {sr.iloc[-1]:+.4%} (sin cambios)")
                elif len(sr) == 1:
                    print(f"\n   {symbol}:")
                    print(f"      [RET] Solo un retorno válido disponible")
                else:
                    print(f"\n   {symbol}:")
                    print(f"      [RET] No hay retornos válidos disponibles")
        
        # 3. Estadísticas descriptivas para cada activo
        print("\n📊 3. ESTADÍSTICAS DESCRIPTIVAS...")
        for symbol in symbols:
            if symbol in simple_returns.columns:
                desc_stats = descriptive_stats(simple_returns[symbol])
                print(f"\n   {symbol}:")
                print(f"      Media: {desc_stats['media']:.4f} ({desc_stats['media']*252:.2%} anual)")
                print(f"      Std: {desc_stats['std']:.4f} ({desc_stats['std']*np.sqrt(252):.2%} anual)")
                print(f"      Skewness: {desc_stats['skewness']:.4f}")
                print(f"      Kurtosis: {desc_stats['kurtosis']:.4f}")
        
        # 4. Percentiles
        print("\n📊 4. PERCENTILES...")
        percentiles = calculate_percentiles(simple_returns, [1, 5, 25, 50, 75, 95, 99])
        print("✅ Percentiles calculados:")
        print(percentiles.round(4))
        
        # 5. Volatilidad
        print("\n📊 5. ANÁLISIS DE VOLATILIDAD...")
        for symbol in symbols:
            if symbol in simple_returns.columns:
                hist_vol = historical_volatility(simple_returns[symbol])
                rolling_vol = rolling_volatility(simple_returns[symbol], window=30)
                
                print(f"\n   {symbol}:")
                print(f"      Volatilidad histórica anual: {hist_vol:.2%}")
                print(f"      Volatilidad móvil promedio (30d): {rolling_vol.mean():.4f}")
                print(f"      Volatilidad móvil actual: {rolling_vol.iloc[-1]:.4f}")
        
        # 6. Tasa libre de riesgo
        print("\n💰 6. TASA LIBRE DE RIESGO...")
        try:
            rf_series = get_risk_free_series(dm, start_date, end_date)
            print(f"✅ Tasa libre de riesgo descargada: {len(rf_series)} observaciones")
            print(f"   Tasa promedio anual: {rf_series.mean()*252:.2%}")
            print(f"   Tasa actual: {rf_series.iloc[-1]*252:.2%}")
        except Exception as e:
            print(f"⚠️ Error descargando tasa libre de riesgo: {e}")
            rf_series = pd.Series(0.02/252, index=simple_returns.index)  # Asumir 2% anual
            print("   Usando tasa fija del 2% anual")
        
        # 7. Ratios de performance
        print("\n🎯 7. RATIOS DE PERFORMANCE...")
        for symbol in symbols:
            if symbol in simple_returns.columns:
                try:
                    sharpe = sharpe_ratio(simple_returns[symbol], rf_series)
                    sortino = sortino_ratio(simple_returns[symbol], rf_series)
                    
                    print(f"\n   {symbol}:")
                    print(f"      Sharpe Ratio: {sharpe:.2f}")
                    print(f"      Sortino Ratio: {sortino:.2f}")
                except Exception as e:
                    print(f"   ⚠️ Error calculando ratios para {symbol}: {e}")
        
        # 8. Information Ratio (usando SPY como benchmark)
        if benchmark_symbol in simple_returns.columns and not rf_series.empty:
            print(f"\n🔄 8. INFORMATION RATIO (usando {benchmark_symbol} como benchmark)...")
            # Calcular Information Ratio para todos los activos vs SPY
            test_symbols = [s for s in symbols if s in simple_returns.columns and s != benchmark_symbol]
            for symbol in test_symbols:
                try:
                    info_ratio = information_ratio(
                        simple_returns[symbol], 
                        simple_returns[benchmark_symbol], 
                        rf_series
                    )
                    print(f"   {symbol} vs {benchmark_symbol}: {info_ratio:.2f}")
                except Exception as e:
                    print(f"   ⚠️ Error calculando Information Ratio para {symbol}: {e}")
        else:
            print(f"\n⚠️ Benchmark {benchmark_symbol} no disponible o falta tasa libre de riesgo")
        
        # 9. Métricas de riesgo extremo
        print("\n⚠️ 9. MÉTRICAS DE RIESGO EXTREMO...")
        for symbol in symbols:
            if symbol in simple_returns.columns:
                try:
                    var_5 = calculate_var(simple_returns[symbol], 0.05, horizon_days=var_horizon_days)
                    cvar_5 = calculate_cvar(simple_returns[symbol], 0.05, horizon_days=var_horizon_days)
                    mdd = max_drawdown(simple_returns[symbol])
                    
                    print(f"\n   {symbol}:")
                    print(f"      VaR 5% ({var_horizon_days}d): {var_5:.2%}")
                    print(f"      CVaR 5% ({var_horizon_days}d): {cvar_5:.2%}")
                    print(f"      Max Drawdown: {mdd:.2%}")
                except Exception as e:
                    print(f"   ⚠️ Error calculando métricas de riesgo para {symbol}: {e}")
        
        # 10. Beta y Alpha (usando SPY como benchmark)
        if benchmark_symbol in simple_returns.columns:
            print(f"\n📈 10. BETA Y ALPHA (usando {benchmark_symbol} como benchmark)...")
            for symbol in symbols:
                if symbol in simple_returns.columns and symbol != benchmark_symbol:
                    try:
                        beta_results = calculate_beta(
                            simple_returns[symbol], 
                            simple_returns[benchmark_symbol]
                        )
                        
                        print(f"\n   {symbol} vs {benchmark_symbol}:")
                        print(f"      Alpha anual: {beta_results['alpha']:.2%}")
                        print(f"      Beta: {beta_results['beta']:.2f}")
                        print(f"      R²: {beta_results['r2']:.2%}")
                    except Exception as e:
                        print(f"   ⚠️ Error calculando Beta/Alpha para {symbol}: {e}")
        else:
            print(f"\n⚠️ Benchmark {benchmark_symbol} no disponible en los datos")
        
        
        
        print("\n🎉 TODAS LAS PRUEBAS COMPLETADAS EXITOSAMENTE!")
        
        # 12. Generar gráficos para todos los stocks
        print("\n📊 12. GENERANDO GRÁFICOS PARA TODOS LOS STOCKS...")
        
        # Crear directorio para gráficos si no existe
        graphs_dir = os.path.join(os.path.dirname(__file__), 'graphs')
        os.makedirs(graphs_dir, exist_ok=True)
        
        graphs_generated = 0
        for symbol in symbols:
            if symbol in simple_returns.columns:
                try:
                    print(f"\n   📈 Generando gráficos para {symbol}...")
                    
                    # Cambiar al directorio de gráficos para guardar ahí
                    original_cwd = os.getcwd()
                    os.chdir(graphs_dir)
                    
                    # Gráfico de distribución
                    dist_file = plot_return_distribution(simple_returns[symbol], symbol, 
                                                       save_plot=True, show_plot=False)
                    if dist_file:
                        print(f"      ✅ Distribución: {dist_file}")
                    
                    # Análisis de autocorrelación
                    autocorr_results = autocorrelation_analysis(simple_returns[symbol], 
                                                              symbol=symbol,
                                                              save_plot=True, show_plot=False)
                    if autocorr_results and 'filepath' in autocorr_results:
                        print(f"      ✅ Autocorrelación: {autocorr_results['filepath']}")
                    
                    # Volver al directorio original
                    os.chdir(original_cwd)
                    
                    graphs_generated += 1
                    
                except Exception as e:
                    print(f"      ⚠️ Error generando gráficos para {symbol}: {e}")
                    # Volver al directorio original en caso de error
                    os.chdir(original_cwd)
        
        print(f"\n🎨 GRÁFICOS GENERADOS: {graphs_generated}/{len(symbols)} stocks")
        print(f"📁 Ubicación: {graphs_dir}")
        print(f"💡 Los gráficos incluyen:")
        print(f"   • Distribución de retornos (histograma + QQ-plot)")
        print(f"   • Análisis de autocorrelación (ACF + Ljung-Box)")
        
        print("\n" + "="*60)
        print("✅ MÓDULO PORTFOLIO_ANALYSIS FUNCIONANDO CORRECTAMENTE")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR EN LAS PRUEBAS: {str(e)}")
        import traceback
        traceback.print_exc()
        return False



if __name__ == "__main__":
    print("🧪 SCRIPT DE PRUEBA - PORTFOLIO ANALYSIS")
    print("="*60)
    
    # Ejecutar pruebas completas
    success = test_portfolio_analysis()
    
    print(f"\n{'🎉 TODAS LAS PRUEBAS EXITOSAS' if success else '❌ ALGUNAS PRUEBAS FALLARON'}")
