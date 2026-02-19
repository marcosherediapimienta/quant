import numpy as np
import statsmodels.api as sm
from dataclasses import dataclass
from ....tools.config import ANNUAL_FACTOR, SIGNIFICANCE_LEVEL, MIN_OBSERVATIONS

@dataclass
class AlphaTestResult:
    alpha_daily: float
    beta: float
    t_statistic: float     
    p_value: float           
    is_significant: bool    
    jensen_alpha: float
    r_squared: float = np.nan         
    beta_t_statistic: float = np.nan  
    beta_p_value: float = np.nan      

class AlphaSignificanceTest:
    def __init__(
        self, 
        annual_factor: float = None,
        significance_level: float = None
    ):
        self.annual_factor = annual_factor or ANNUAL_FACTOR
        self.significance_level = significance_level or SIGNIFICANCE_LEVEL
    
    def test(
        self,
        asset_returns: np.ndarray,
        market_returns: np.ndarray,
        risk_free_rate_daily: float,
        maxlags: int = None
    ) -> AlphaTestResult:

        if len(asset_returns) != len(market_returns) or len(asset_returns) < MIN_OBSERVATIONS:
            return AlphaTestResult(np.nan, np.nan, np.nan, np.nan, False, np.nan)
        
        y = asset_returns - risk_free_rate_daily
        x = market_returns - risk_free_rate_daily
        
        try:
            X = sm.add_constant(x, has_constant='add')
            model = sm.OLS(y, X, hasconst=True)
            res = model.fit()
            n = len(y)

            if maxlags is None:
                maxlags = self._calculate_default_maxlags(n)

            robust = res.get_robustcov_results(cov_type='HAC', maxlags=maxlags)
            
            alpha_daily = float(robust.params[0])
            beta = float(robust.params[1])
            t_stat = float(robust.tvalues[0])
            p_value = float(robust.pvalues[0])
            beta_t_stat = float(robust.tvalues[1])
            beta_p_value = float(robust.pvalues[1])
            jensen_alpha = (1 + alpha_daily) ** self.annual_factor - 1
            is_significant = p_value < self.significance_level
            r_squared = float(np.clip(res.rsquared, 0.0, 1.0))

            return AlphaTestResult(
                alpha_daily, beta, t_stat, p_value, is_significant, jensen_alpha,
                r_squared, beta_t_stat, beta_p_value
            )
            
        except Exception:
            return AlphaTestResult(np.nan, np.nan, np.nan, np.nan, False, np.nan)
    
    def _calculate_default_maxlags(self, n: int) -> int:
        return max(1, int(4 * (n / 100) ** (2 / 9)))