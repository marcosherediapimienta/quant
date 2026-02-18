import numpy as np
from dataclasses import dataclass
from ....tools.config import ANNUAL_FACTOR

@dataclass
class CAPMResult:
    alpha_daily: float
    beta: float
    correlation: float
    jensen_alpha: float
    r_squared: float = np.nan     # % variance of asset explained by market
    treynor_ratio: float = np.nan # (annualised excess return) / beta

class CAPMCalculator:
    def __init__(self, annual_factor: float = ANNUAL_FACTOR):
        self.annual_factor = annual_factor
    
    def calculate(
        self,
        asset_returns: np.ndarray,
        market_returns: np.ndarray,
        risk_free_rate_daily: float
    ) -> CAPMResult:

        if not isinstance(asset_returns, np.ndarray) or not isinstance(market_returns, np.ndarray):
            raise TypeError("Los retornos deben ser numpy arrays")

        y = asset_returns - risk_free_rate_daily
        x = market_returns - risk_free_rate_daily
        X = np.c_[np.ones_like(x), x]

        try:
            params, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
            alpha_daily = float(params[0])
            beta = float(params[1])
        except Exception:
            return CAPMResult(np.nan, np.nan, np.nan, np.nan)

        # R² = 1 - SS_res / SS_tot
        try:
            y_hat = X @ params
            ss_res = float(np.sum((y - y_hat) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            r_squared = float(np.clip(r_squared, 0.0, 1.0))
        except Exception:
            r_squared = np.nan

        try:
            corr = float(np.corrcoef(asset_returns, market_returns)[0, 1])
        except Exception:
            corr = np.nan

        jensen_alpha = self._annualize_alpha(alpha_daily)

        # Treynor ratio: annualised excess return / beta (measures return per unit of systematic risk)
        try:
            annualised_excess = float(y.mean() * self.annual_factor)
            treynor = annualised_excess / beta if (not np.isnan(beta) and abs(beta) > 1e-10) else np.nan
        except Exception:
            treynor = np.nan

        return CAPMResult(alpha_daily, beta, corr, jensen_alpha, r_squared, treynor)
    
    def _annualize_alpha(self, alpha_daily: float) -> float:

        if np.isnan(alpha_daily):
            return np.nan
        return (1 + alpha_daily) ** self.annual_factor - 1
    
    def expected_return(
        self,
        beta: float,
        risk_free_rate: float,
        market_return: float
    ) -> float:

        if np.isnan(beta) or np.isnan(risk_free_rate) or np.isnan(market_return):
            return np.nan
        return risk_free_rate + beta * (market_return - risk_free_rate)