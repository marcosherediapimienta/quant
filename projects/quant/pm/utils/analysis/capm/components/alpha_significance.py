import numpy as np
import statsmodels.api as sm
from dataclasses import dataclass

@dataclass
class AlphaTestResult:
    alpha_daily: float
    beta: float
    t_statistic: float
    p_value: float
    is_significant: bool
    jensen_alpha: float

class AlphaSignificanceTest:

    def __init__(
        self, 
        annual_factor: float = 252.0,
        significance_level: float = 0.05
    ):
        self.annual_factor = annual_factor
        self.significance_level = significance_level
    
    def test(
        self,
        asset_returns: np.ndarray,
        market_returns: np.ndarray,
        risk_free_rate_daily: float,
        maxlags: int = None
    ) -> AlphaTestResult:

        if len(asset_returns) != len(market_returns) or len(asset_returns) < 30:
            return AlphaTestResult(np.nan, np.nan, np.nan, np.nan, False, np.nan)
        
        y = asset_returns - risk_free_rate_daily
        x = market_returns - risk_free_rate_daily
        
        try:
            X = sm.add_constant(x, has_constant='add')
            model = sm.OLS(y, X, hasconst=True)
            res = model.fit()
            
            # Lags HAC: sqrt(n) por defecto
            n = len(y)
            if maxlags is None:
                maxlags = max(1, int(np.sqrt(n)))
            
            # Errores robustos HAC (Newey-West)
            robust = res.get_robustcov_results(cov_type='HAC', maxlags=maxlags)
            
            alpha_daily = float(robust.params[0])
            beta = float(robust.params[1])
            t_stat = float(robust.tvalues[0])
            p_value = float(robust.pvalues[0])
            
            jensen_alpha = (1 + alpha_daily) ** self.annual_factor - 1
            is_significant = p_value < self.significance_level
            
            return AlphaTestResult(
                alpha_daily, beta, t_stat, p_value, is_significant, jensen_alpha
            )
            
        except Exception:
            return AlphaTestResult(np.nan, np.nan, np.nan, np.nan, False, np.nan)