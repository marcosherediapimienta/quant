import numpy as np
from typing import Dict
from ..analyzers.macro_factor_analyzer import MacroFactorAnalyzer

class MacroFactorReporter:

    def __init__(self, analyzer: MacroFactorAnalyzer):
        self.analyzer = analyzer
    
    def print_analysis(self, results: Dict) -> None:
        print("ANÁLISIS DE FACTORES MACRO".center(70))
        self._print_regression_summary(results)
        self._print_factor_loadings(results)
        self._print_risk_decomposition(results)
        self._print_significant_factors(results)
    
    def _print_regression_summary(self, results: Dict) -> None:
        print("RESUMEN DE REGRESIÓN")
        print(f"  Alpha (diario):           {results['alpha']*100:>8.4f}%")
        print(f"  Alpha (anual):            {results['alpha_annual']*100:>8.2f}%")
        print(f"  R²:                       {results['r_squared']:>8.3f}")
        print(f"  R² ajustado:              {results['adj_r_squared']:>8.3f}")
        print(f"  Observaciones:            {results['n_obs']:>8d}")

        alpha_pct = results['alpha_annual'] * 100
        if alpha_pct > 5:
            interp = "Excelente - genera alpha significativo"
        elif alpha_pct > 0:
            interp = "Positivo - añade valor vs factores"
        elif alpha_pct > -5:
            interp = "Neutral - explica bien los factores"
        else:
            interp = "Negativo - destruye valor"
        print(f"  Interpretación:           {interp}")
    
    def _print_factor_loadings(self, results: Dict) -> None:
        print("FACTOR LOADINGS (BETAS)")
        print(f"{'Factor':<20} {'Beta':>10} {'t-stat':>10} {'p-value':>10} {'Signif':<10}")

        betas = results['betas']
        sorted_factors = sorted(
            betas.keys(),
            key=lambda x: abs(betas[x]),
            reverse=True
        )
        
        for factor in sorted_factors:
            beta = betas[factor]
            t_stat = results['t_stats'][factor]
            p_val = results['p_values'][factor]

            if p_val < 0.01:
                sig = "***"
            elif p_val < 0.05:
                sig = "**"
            elif p_val < 0.10:
                sig = "*"
            else:
                sig = ""
            
            print(f"{factor:<20} {beta:>10.4f} {t_stat:>10.3f} {p_val:>10.4f} {sig:<10}")

        print("Significancia: *** p<0.01, ** p<0.05, * p<0.10")
    
    def _print_risk_decomposition(self, results: Dict) -> None:
        risk_decomp = results['risk_decomposition']
        
        print("DESCOMPOSICIÓN DE RIESGO")
        print(f"  Riesgo sistemático:       {risk_decomp['systematic_pct']:>8.2f}%")
        print(f"  Riesgo idiosincrático:    {risk_decomp['idiosyncratic_pct']:>8.2f}%")
        print(f"  R² (varianza explicada):  {risk_decomp['r_squared']:>8.3f}")
        
        sys_pct = risk_decomp['systematic_pct']
        if sys_pct > 70:
            interp = "Alta exposición a factores macro"
        elif sys_pct > 40:
            interp = "Exposición moderada a factores"
        else:
            interp = "Predomina riesgo específico"
        print(f"  Interpretación:           {interp}")
    
    def _print_significant_factors(self, results: Dict) -> None:
        sig_factors = results['significant_factors']
        
        print("FACTORES SIGNIFICATIVOS (p < 0.05)")
        if sig_factors:
            for factor in sig_factors:
                beta = results['betas'][factor]
                p_val = results['p_values'][factor]
                direction = "positiva" if beta > 0 else "negativa"
                print(f"  • {factor}: β={beta:.4f} (exposición {direction})")
        else:
            print("  Ningún factor significativo al 5%")
    
    def print_factor_contributions(self, results: Dict) -> None:
        factor_contrib = results['factor_contributions']

        print("CONTRIBUCIÓN POR FACTOR".center(70))

        for name, series in factor_contrib.items():
            if name in ['alpha', 'residual']:
                continue
            
            mean_contrib = series.mean() * 252 * 100  
            std_contrib = series.std() * np.sqrt(252) * 100
            
            print(f"\n{name}")
            print(f"  Contribución anual media: {mean_contrib:>8.2f}%")
            print(f"  Volatilidad contribución: {std_contrib:>8.2f}%")