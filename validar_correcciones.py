#!/usr/bin/env python3
"""
Script de validación rápida de correcciones aplicadas.

Ejecutar:
    python validar_correcciones.py
"""

import sys
import os
from pathlib import Path

# Colores para terminal
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def check_file_contains(filepath: str, search_strings: list, description: str) -> bool:
    """Verifica que un archivo contenga ciertas cadenas."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        all_found = all(s in content for s in search_strings)
        
        if all_found:
            print(f"  {GREEN}✅{RESET} {description}")
            return True
        else:
            missing = [s for s in search_strings if s not in content]
            print(f"  {RED}❌{RESET} {description}")
            print(f"     Faltante: {missing[0][:50]}...")
            return False
            
    except FileNotFoundError:
        print(f"  {RED}❌{RESET} Archivo no encontrado: {filepath}")
        return False
    except Exception as e:
        print(f"  {RED}❌{RESET} Error: {e}")
        return False


def main():
    print("=" * 80)
    print(f"{BLUE}🔍 VALIDACIÓN DE CORRECCIONES APLICADAS{RESET}")
    print("=" * 80)
    print()
    
    base_path = Path(__file__).parent / "projects" / "quant" / "pm"
    checks_passed = 0
    checks_total = 0
    
    # ========== CHECK 1: Anualización Geométrica en helpers.py ==========
    print(f"{BLUE}📋 Check 1: Anualización de Retornos (helpers.py){RESET}")
    checks_total += 1
    if check_file_contains(
        str(base_path / "utils/analysis/risk_metrics/components/helpers.py"),
        [
            "cumulative_return = (1 + daily_returns).prod()",
            "annual_return = cumulative_return ** (annual_factor / n_periods) - 1",
            "Método geométrico (correcto)"
        ],
        "Método geométrico implementado correctamente"
    ):
        checks_passed += 1
    print()
    
    # ========== CHECK 2: Alpha Calculation ==========
    print(f"{BLUE}📋 Check 2: Alpha de Jensen (alpha.py){RESET}")
    checks_total += 1
    if check_file_contains(
        str(base_path / "utils/analysis/risk_metrics/components/alpha.py"),
        [
            "alpha_daily = float(mean_excess_portfolio - beta * mean_excess_benchmark)",
            "alpha_annual = float((1 + alpha_daily) ** self.annual_factor - 1)",
            "Calcular alpha DIARIO primero"
        ],
        "Alpha calculado en frecuencia diaria primero"
    ):
        checks_passed += 1
    print()
    
    # ========== CHECK 3: VaR Warnings ==========
    print(f"{BLUE}📋 Check 3: Warnings en VaR (var.py){RESET}")
    checks_total += 1
    if check_file_contains(
        str(base_path / "utils/analysis/risk_metrics/components/var.py"),
        [
            "⚠️ NOTA: Anualización de VaR con sqrt(T) asume retornos i.i.d. normales",
            "Esta es una aproximación"
        ],
        "Warnings de aproximación añadidos"
    ):
        checks_passed += 1
    print()
    
    # ========== CHECK 4: ES Warnings ==========
    print(f"{BLUE}📋 Check 4: Warnings en ES (es.py){RESET}")
    checks_total += 1
    if check_file_contains(
        str(base_path / "utils/analysis/risk_metrics/components/es.py"),
        [
            "⚠️ NOTA: Anualización con sqrt(T) asume i.i.d. normalidad",
            "Es una aproximación"
        ],
        "Warnings de aproximación añadidos"
    ):
        checks_passed += 1
    print()
    
    # ========== CHECK 5: Price Target Validation ==========
    print(f"{BLUE}📋 Check 5: Validación Price Target (price_target_calculator.py){RESET}")
    checks_total += 1
    if check_file_contains(
        str(base_path / "utils/analysis/valuation/metrics/price_target_calculator.py"),
        [
            "max_upside = current_price * 1.75",
            "max_downside = current_price * 0.25",
            "Validación: Limitar cambios extremos"
        ],
        "Límites de ±75% implementados"
    ):
        checks_passed += 1
    print()
    
    # ========== CHECK 6: Macro Regression Alpha ==========
    print(f"{BLUE}📋 Check 6: Alpha Extremo Macro (macro_regression.py){RESET}")
    checks_total += 1
    macro_path = Path(__file__).parent / "projects" / "quant" / "macro"
    if check_file_contains(
        str(macro_path / "utils/components/macro_regression.py"),
        [
            "if alpha_daily <= -0.95:",
            "Alpha diario extremo detectado",
            "probablemente error de datos"
        ],
        "Validación de alpha extremo añadida"
    ):
        checks_passed += 1
    print()
    
    # ========== CHECK 7: Documentación ==========
    print(f"{BLUE}📋 Check 7: Documentación de Correcciones{RESET}")
    checks_total += 1
    if check_file_contains(
        str(Path(__file__).parent / "CORRECCIONES_APLICADAS.md"),
        [
            "ERROR CRÍTICO #1",
            "Método geométrico compuesto",
            "CORRECCIONES APLICADAS"
        ],
        "Archivo CORRECCIONES_APLICADAS.md existe"
    ):
        checks_passed += 1
    print()
    
    # ========== RESUMEN ==========
    print("=" * 80)
    print(f"{BLUE}📊 RESUMEN DE VALIDACIÓN{RESET}")
    print("=" * 80)
    print(f"Total de checks: {checks_total}")
    print(f"{GREEN}✅ Pasados: {checks_passed}{RESET}")
    print(f"{RED}❌ Fallidos: {checks_total - checks_passed}{RESET}")
    print()
    
    if checks_passed == checks_total:
        print(f"{GREEN}🎉 ¡TODAS LAS CORRECCIONES APLICADAS CORRECTAMENTE!{RESET}")
        print()
        print("Próximos pasos:")
        print("  1. Ejecutar tests: python tests/test_correcciones.py")
        print("  2. Re-ejecutar notebooks para actualizar resultados")
        print("  3. Revisar CORRECCIONES_APLICADAS.md para detalles")
        print()
        return 0
    else:
        print(f"{RED}⚠️  Algunas correcciones no se aplicaron correctamente{RESET}")
        print()
        print("Solución:")
        print("  - Verifica que todos los archivos están en su lugar")
        print("  - Revisa los errores específicos arriba")
        print("  - Consulta CORRECCIONES_APLICADAS.md para detalles")
        print()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
