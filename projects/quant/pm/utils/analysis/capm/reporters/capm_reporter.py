import numpy as np
from typing import Dict
from ..analyzers.capm_analyzer import CAPMAnalyzer

class CAPMReporter:
    def __init__(self, capm_analyzer: CAPMAnalyzer):
        self.analyzer = capm_analyzer
    
    def generate_report(
        self,
        asset_returns: np.ndarray,
        market_returns: np.ndarray,
        risk_free_rate: float,
        asset_name: str = "Activo"
    ) -> None:

        results = self.analyzer.analyze(asset_returns, market_returns, risk_free_rate)
        self.print_report(results, asset_name)
    
    def print_report(self, results: Dict, asset_name: str = "Activo") -> None:
        print(f"ANÁLISIS CAPM: {asset_name}".center(60))

        print("PARÁMETROS DEL MODELO")
        print(f"  Beta:                    {results['beta']:>8.3f}")
        print(f"  Correlación:             {results['correlation']:>8.3f}")
        print(f"  R²:                      {results['r_squared']:>8.3f}")
        
        print("ALPHA (Jensen)")
        print(f"  Alpha Diario:            {results['alpha_daily']*100:>8.4f}%")
        print(f"  Alpha Anual:             {results['alpha_annual']*100:>8.2f}%")
        
        sig_level = self.analyzer.significance_level
        print("SIGNIFICANCIA ESTADÍSTICA")
        print(f"  t-statistic:             {results['t_statistic']:>8.3f}")
        print(f"  p-value:                 {results['p_value']:>8.4f}")
        print(f"  Significativo (α={sig_level}):  {'[SI]' if results['is_significant'] else '[NO]'}")
 
        if results['is_significant']:
            if results['alpha_annual'] > 0:
                print(f"[OK] Alpha significativamente positivo: el activo supera el retorno esperado")
            else:
                print(f"[WARN] Alpha significativamente negativo: el activo está por debajo del esperado")
        else:
            print(f"[INFO] Alpha no significativo: el activo se comporta según lo esperado por CAPM")