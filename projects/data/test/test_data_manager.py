#!/usr/bin/env python3
"""
Script para probar el DataManager con agrupación por disponibilidad.
"""

import sys
from pathlib import Path

# Agregar el directorio padre al path para importar data_manager
sys.path.append(str(Path(__file__).parent.parent))

from data_manager import DataManager


def main():
    """Función principal del script."""
    args = DataManager.parse_arguments()
    
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
    
    # Símbolos a probar
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'IBM', 'IONQ', 'SMR', 
               'BTC-USD', 'ETH-USD', 'GLD', 'ACWI', 'SPY', 'QQQ']
    
    print(f"📈 Probando con {len(symbols)} símbolos...")
    
    try:
        # Descargar datos agrupados
        print("📅 Descargando datos agrupados por disponibilidad...")
        grouped_data = dm.download_market_data_grouped(
            symbols=symbols,
            target_years=args.years,
            force_refresh=args.force_refresh
        )
        
        # Procesar y mostrar grupos usando el método del DataManager
        all_plots = dm.process_and_display_groups(grouped_data, args.years)
        
        print(f"\n🎉 PRUEBA COMPLETADA")
        print(f"📁 Total de archivos generados: {len(all_plots)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()