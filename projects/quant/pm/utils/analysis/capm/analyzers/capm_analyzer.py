import numpy as np
from typing import Dict
from ..components.capm_calculator import CAPMCalculator
from ..components.alpha_significance import AlphaSignificanceTest
from ..components.helpers import daily_risk_free_rate


class CAPMAnalyzer:

    def __init__(self, annual_factor: float = 252.0, significance_level: float = 0.05):
        self.annual_factor = annual_factor
        self.capm_calc = CAPMCalculator(annual_factor)
        self.alpha_test = AlphaSignificanceTest(annual_factor, significance_level)
    
    def analyze(
        self,
        asset_returns: np.ndarray,
        market_returns: np.ndarray,
        risk_free_rate: float
    ) -> Dict:
 
        rf_daily = daily_risk_free_rate(risk_free_rate, self.annual_factor)
        
        # CAPM básico
        capm = self.capm_calc.calculate(asset_returns, market_returns, rf_daily)
        
        # Test de significancia
        alpha_test = self.alpha_test.test(asset_returns, market_returns, rf_daily)
        
        # Calcular R²
        try:
            r_squared = float(capm.correlation ** 2) if not np.isnan(capm.correlation) else np.nan
        except:
            r_squared = np.nan
        
        return {
            'alpha_daily': capm.alpha_daily,
            'alpha_annual': capm.jensen_alpha,
            'beta': capm.beta,
            'correlation': capm.correlation,
            'r_squared': r_squared,
            't_statistic': alpha_test.t_statistic,
            'p_value': alpha_test.p_value,
            'is_significant': alpha_test.is_significant
        }
    
    def expected_return(
        self,
        beta: float,
        risk_free_rate: float,
        market_return: float
    ) -> float:

        return self.capm_calc.expected_return(beta, risk_free_rate, market_return)
    