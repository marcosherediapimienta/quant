#!/usr/bin/env python3
"""
Script de prueba para verificar las nuevas funcionalidades de la v4
"""

import sys
import os
sys.path.append('/home/mhp/Escritorio/Github/Quantitative-Finance/projects/quant/tests')

from test5 import analizar_stock, comparar_stocks

def test_v4_features():
    """Prueba las nuevas características de la v4"""
    print("🧪 Probando características de la v4...")
    
    # Test 1: PYPL (PayPal) - debería usar peers de Credit Services/Payments
    print("\n1️⃣ Test con PYPL (Credit Services/Payments):")
    try:
        result_pypl = analizar_stock("PYPL")
        if 'error' not in result_pypl:
            print("✅ PYPL analizado correctamente")
            print(f"   FCF TTM: ${result_pypl['valuation'].get('fcf_ttm', 'N/A')}")
            print(f"   EV (corp): ${result_pypl['basics'].get('ev_corp', 'N/A')}")
            print(f"   FCF/EV Yield: {result_pypl['valuation'].get('fcf_ev_yield', 'N/A')}")
            print(f"   Peers usados: {result_pypl['peers_used']}")
        else:
            print(f"❌ Error con PYPL: {result_pypl['error']}")
    except Exception as e:
        print(f"❌ Excepción con PYPL: {e}")
    
    # Test 2: HPQ (Hewlett Packard) - podría tener equity ≤ 0
    print("\n2️⃣ Test con HPQ (posible equity ≤ 0):")
    try:
        result_hpq = analizar_stock("HPQ")
        if 'error' not in result_hpq:
            print("✅ HPQ analizado correctamente")
            equity_nonpos = result_hpq.get('meta', {}).get('equity_nonpositive', False)
            print(f"   Equity nonpositive: {equity_nonpos}")
            print(f"   P/B: {result_hpq['valuation']['pb']}")
            print(f"   ROE: {result_hpq['quality']['roe']}")
        else:
            print(f"❌ Error con HPQ: {result_hpq['error']}")
    except Exception as e:
        print(f"❌ Excepción con HPQ: {e}")
    
    # Test 3: Comparación de múltiples stocks
    print("\n3️⃣ Test de comparación:")
    try:
        tickers = ["AAPL", "MSFT", "GOOGL"]
        comparar_stocks(tickers)
        print("✅ Comparación completada")
    except Exception as e:
        print(f"❌ Error en comparación: {e}")

if __name__ == "__main__":
    test_v4_features()
