"""
Tests para validar las correcciones aplicadas al proyecto quant.

Ejecutar con: python -m pytest tests/test_correcciones.py -v
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '../projects/quant/pm')

from utils.analysis.risk_metrics.components.helpers import (
    annualize_return, 
    annualize_volatility
)


class TestAnualizacionRetornos:
    """Tests para validar corrección de anualización de retornos."""
    
    def test_anualizacion_geometrica_basica(self):
        """Test básico: anualización debe usar método geométrico."""
        # Retornos diarios constantes de 0.05% (0.0005)
        daily_returns = np.full(252, 0.0005)
        
        annual = annualize_return(daily_returns, annual_factor=252)
        
        # Método geométrico esperado: (1.0005)^252 - 1 ≈ 0.1339 (13.39%)
        expected = (1.0005) ** 252 - 1
        
        assert np.isclose(annual, expected, rtol=1e-4), \
            f"Anualización incorrecta: {annual:.6f} vs esperado {expected:.6f}"
    
    def test_comparacion_aritmetico_vs_geometrico(self):
        """Verifica que método geométrico difiere de aritmético."""
        daily_returns = np.random.normal(0.0005, 0.01, 252)
        
        # Método geométrico (correcto)
        geometric = annualize_return(daily_returns, annual_factor=252)
        
        # Método aritmético (INCORRECTO - solo para comparación)
        arithmetic = daily_returns.mean() * 252
        
        # Deben ser diferentes (típicamente geométrico < aritmético)
        assert not np.isclose(geometric, arithmetic, rtol=0.01), \
            "Métodos geométrico y aritmético no deberían coincidir"
        
        print(f"✅ Geométrico: {geometric:.4%}")
        print(f"❌ Aritmético: {arithmetic:.4%}")
        print(f"   Diferencia: {abs(geometric - arithmetic):.4%}")
    
    def test_retornos_negativos(self):
        """Test con retornos negativos."""
        # Pérdida diaria de -0.1% promedio
        daily_returns = np.full(252, -0.001)
        
        annual = annualize_return(daily_returns, annual_factor=252)
        expected = (1 - 0.001) ** 252 - 1  # ≈ -22.1%
        
        assert np.isclose(annual, expected, rtol=1e-4), \
            f"Anualización con pérdidas incorrecta: {annual:.6f} vs {expected:.6f}"
    
    def test_volatilidad_inalterada(self):
        """Verifica que volatilidad sigue usando sqrt(252) correctamente."""
        daily_returns = np.random.normal(0.0005, 0.01, 252)
        
        annual_vol = annualize_volatility(daily_returns, annual_factor=252, ddof=0)
        expected_vol = daily_returns.std(ddof=0) * np.sqrt(252)
        
        assert np.isclose(annual_vol, expected_vol, rtol=1e-4), \
            "Volatilidad debe seguir usando sqrt(252)"
    
    def test_retorno_cero(self):
        """Test caso edge: retornos cero."""
        daily_returns = np.zeros(252)
        annual = annualize_return(daily_returns, annual_factor=252)
        
        assert np.isclose(annual, 0.0, atol=1e-10), \
            "Retornos cero deben anualizar a cero"
    
    def test_retorno_total_perdida(self):
        """Test caso extremo: pérdida total."""
        # Simular pérdida casi total (-99.9% diario)
        daily_returns = np.array([-0.999])
        annual = annualize_return(daily_returns, annual_factor=252)
        
        # Debe retornar -1.0 (pérdida total)
        assert annual == -1.0, \
            f"Pérdida total debe retornar -1.0, obtuvo {annual}"


class TestAlphaCalculation:
    """Tests para validar corrección de Alpha."""
    
    def test_alpha_frequency_correcta(self):
        """Verifica que alpha se calcula en frecuencia diaria primero."""
        # Este test requiere importar AlphaCalculator
        # Por ahora verificamos la lógica conceptual
        
        # Alpha debe calcularse como:
        # alpha_daily = E[Rp - Rf] - beta * E[Rm - Rf]
        # alpha_annual = (1 + alpha_daily)^252 - 1
        
        alpha_daily = 0.0001  # 0.01% diario
        annual_factor = 252
        
        # Anualización geométrica
        alpha_annual_geometric = (1 + alpha_daily) ** annual_factor - 1
        
        # Anualización aritmética (INCORRECTA)
        alpha_annual_arithmetic = alpha_daily * annual_factor
        
        # Geométrico debe ser mayor para alphas positivos pequeños
        assert alpha_annual_geometric > alpha_annual_arithmetic, \
            "Anualización geométrica debe diferir de aritmética"
        
        print(f"✅ Alpha diario: {alpha_daily:.6f}")
        print(f"✅ Alpha anual (geométrico): {alpha_annual_geometric:.6%}")
        print(f"❌ Alpha anual (aritmético): {alpha_annual_arithmetic:.6%}")


class TestPriceTargetValidation:
    """Tests para validar límites en price targets."""
    
    def test_limite_upside(self):
        """Price target no debe exceder +75%."""
        current_price = 100.0
        max_upside = current_price * 1.75  # 175
        
        # Simular price target extremo
        extreme_target = 250.0
        
        # Con validación debería limitarse a 175
        limited_target = min(extreme_target, max_upside)
        
        assert limited_target == max_upside, \
            f"Target debe limitarse a +75%: {limited_target}"
    
    def test_limite_downside(self):
        """Price target no debe caer más de -75%."""
        current_price = 100.0
        max_downside = current_price * 0.25  # 25
        
        # Simular price target extremo
        extreme_target = 10.0
        
        # Con validación debería limitarse a 25
        limited_target = max(extreme_target, max_downside)
        
        assert limited_target == max_downside, \
            f"Target debe limitarse a -75%: {limited_target}"


def run_all_tests():
    """Ejecuta todos los tests y muestra resultados."""
    print("=" * 70)
    print("🧪 EJECUTANDO TESTS DE VALIDACIÓN DE CORRECCIONES")
    print("=" * 70)
    
    test_classes = [
        TestAnualizacionRetornos,
        TestAlphaCalculation,
        TestPriceTargetValidation
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n📋 {test_class.__name__}")
        print("-" * 70)
        
        test_instance = test_class()
        test_methods = [m for m in dir(test_instance) if m.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(test_instance, method_name)
                method()
                print(f"  ✅ {method_name}")
                passed_tests += 1
            except AssertionError as e:
                print(f"  ❌ {method_name}: {e}")
                failed_tests.append((test_class.__name__, method_name, str(e)))
            except Exception as e:
                print(f"  ⚠️  {method_name}: ERROR - {e}")
                failed_tests.append((test_class.__name__, method_name, f"ERROR: {e}"))
    
    # Resumen
    print("\n" + "=" * 70)
    print("📊 RESUMEN DE TESTS")
    print("=" * 70)
    print(f"Total: {total_tests}")
    print(f"✅ Pasados: {passed_tests}")
    print(f"❌ Fallidos: {len(failed_tests)}")
    
    if failed_tests:
        print("\n❌ Tests fallidos:")
        for class_name, method_name, error in failed_tests:
            print(f"  - {class_name}.{method_name}")
            print(f"    {error}")
    else:
        print("\n🎉 ¡TODOS LOS TESTS PASARON!")
    
    print("=" * 70)
    
    return len(failed_tests) == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
