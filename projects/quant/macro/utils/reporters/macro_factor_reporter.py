import numpy as np
from typing import Dict
from ..analyzers.macro_factor_analyzer import MacroFactorAnalyzer

class MacroFactorReporter:
    def __init__(self, analyzer: MacroFactorAnalyzer):
        self.analyzer = analyzer
    
    def print_analysis(self, results: Dict) -> None:
        print("MACRO FACTOR ANALYSIS".center(70))
        self._print_regression_summary(results)
        self._print_factor_loadings(results)
        self._print_risk_decomposition(results)
        self._print_significant_factors(results)
    
    def _print_regression_summary(self, results: Dict) -> None:
        print("REGRESSION SUMMARY")
        print(f"  Regression alpha (daily):  {results['alpha']*100:>8.4f}%")
        print(f"  Regression alpha (annual): {results['alpha_annual']*100:>8.2f}%")
        print(f"  R²:                        {results['r_squared']:>8.3f}")
        print(f"  Adjusted R²:               {results['adj_r_squared']:>8.3f}")
        print(f"  Observations:              {results['n_obs']:>8d}")

        alpha_pct = results['alpha_annual'] * 100
        if alpha_pct > 5:
            interp = "Excellent - generates significant alpha"
        elif alpha_pct > 0:
            interp = "Positive - adds value vs factors"
        elif alpha_pct > -5:
            interp = "Neutral - factors explain returns well"
        else:
            interp = "Negative - destroys value"
        print(f"  Interpretation:            {interp}")
    
    def _print_factor_loadings(self, results: Dict) -> None:
        print("FACTOR LOADINGS (BETAS)")
        print(f"{'Factor':<20} {'Beta':>10} {'t-stat':>10} {'p-value':>10} {'Signif':<10}")

        betas = results['betas']
        sorted_factors = sorted(
            betas.keys(),
            key=lambda x: abs(betas[x]),
            reverse=True
        )
        
        _SIG_LEVELS = ((0.01, "***"), (0.05, "**"), (0.10, "*"))

        for factor in sorted_factors:
            beta = betas[factor]
            t_stat = results['t_stats'][factor]
            p_val = results['p_values'][factor]

            sig = next((s for threshold, s in _SIG_LEVELS if p_val < threshold), "")
            
            print(f"{factor:<20} {beta:>10.4f} {t_stat:>10.3f} {p_val:>10.4f} {sig:<10}")

        print("Significance: *** p<0.01, ** p<0.05, * p<0.10")
    
    def _print_risk_decomposition(self, results: Dict) -> None:
        risk_decomp = results['risk_decomposition']
        
        print("RISK DECOMPOSITION")
        print(f"  Systematic risk:           {risk_decomp['systematic_pct']:>8.2f}%")
        print(f"  Idiosyncratic risk:        {risk_decomp['idiosyncratic_pct']:>8.2f}%")
        print(f"  R² (variance explained):   {risk_decomp['r_squared']:>8.3f}")
        
        sys_pct = risk_decomp['systematic_pct']
        if sys_pct > 70:
            interp = "High exposure to macro factors"
        elif sys_pct > 40:
            interp = "Moderate factor exposure"
        else:
            interp = "Idiosyncratic risk dominates"
        print(f"  Interpretation:            {interp}")
    
    def _print_significant_factors(self, results: Dict) -> None:
        sig_factors = results['significant_factors']
        
        print("SIGNIFICANT FACTORS (p < 0.05)")
        if sig_factors:
            for factor in sig_factors:
                beta = results['betas'][factor]
                direction = "positive" if beta > 0 else "negative"
                print(f"  • {factor}: β={beta:.4f} ({direction} exposure)")
        else:
            print("  No significant factors at 5%")
    
    def print_factor_contributions(self, results: Dict) -> None:
        factor_contrib = results['factor_contributions']

        print("FACTOR CONTRIBUTIONS".center(70))

        for name, series in factor_contrib.items():
            if name in ['alpha', 'residual']:
                continue
            
            mean_contrib = series.mean() * 252 * 100  
            std_contrib = series.std() * np.sqrt(252) * 100
            
            print(f"\n{name}")
            print(f"  Mean annual contribution:  {mean_contrib:>8.2f}%")
            print(f"  Contribution volatility:   {std_contrib:>8.2f}%")
