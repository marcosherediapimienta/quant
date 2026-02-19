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
        asset_name: str = "Asset"
    ) -> None:

        results = self.analyzer.analyze(asset_returns, market_returns, risk_free_rate)
        self.print_report(results, asset_name)
    
    def print_report(self, results: Dict, asset_name: str = "Asset") -> None:
        print(f"CAPM ANALYSIS: {asset_name}".center(60))

        print("MODEL PARAMETERS")
        print(f"  Beta:                    {results['beta']:>8.3f}")
        print(f"  Correlation:             {results['correlation']:>8.3f}")
        print(f"  R²:                      {results['r_squared']:>8.3f}")
        
        print("ALPHA (Jensen)")
        print(f"  Daily Alpha:             {results['alpha_daily']*100:>8.4f}%")
        print(f"  Annual Alpha:            {results['alpha_annual']*100:>8.2f}%")
        
        sig_level = self.analyzer.significance_level
        print("STATISTICAL SIGNIFICANCE")
        print(f"  t-statistic:             {results['t_statistic']:>8.3f}")
        print(f"  p-value:                 {results['p_value']:>8.4f}")
        print(f"  Significant (α={sig_level}):  {'[YES]' if results['is_significant'] else '[NO]'}")
 
        if results['is_significant']:
            if results['alpha_annual'] > 0:
                print("[OK] Significantly positive alpha: asset outperforms expected return")
            else:
                print("[WARN] Significantly negative alpha: asset underperforms expected return")
        else:
            print("[INFO] Alpha not significant: asset behaves as expected by CAPM")