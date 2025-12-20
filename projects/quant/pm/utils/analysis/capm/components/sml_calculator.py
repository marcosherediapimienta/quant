import numpy as np
from dataclasses import dataclass

@dataclass
class SMLResult:
    beta_axis: np.ndarray
    expected_returns: np.ndarray
    market_return: float
    risk_free_rate: float
    slope: float

class SMLCalculator:

    def calculate(
        self,
        risk_free_rate: float,
        market_return: float,
        max_beta: float = 2.0,
        n_points: int = 100
    ) -> SMLResult:

        beta_axis = np.linspace(0, max_beta, n_points)
        slope = market_return - risk_free_rate
        expected_returns = risk_free_rate + slope * beta_axis
        
        return SMLResult(
            beta_axis, expected_returns, market_return, risk_free_rate, slope
        )
    
    def expected_return_for_beta(
        self,
        beta: float,
        risk_free_rate: float,
        market_return: float
    ) -> float:

        if np.isnan(beta):
            return np.nan
        return risk_free_rate + beta * (market_return - risk_free_rate)
    
    def is_undervalued(
        self,
        actual_return: float,
        beta: float,
        risk_free_rate: float,
        market_return: float
    ) -> bool:

        expected = self.expected_return_for_beta(beta, risk_free_rate, market_return)
        if np.isnan(expected) or np.isnan(actual_return):
            return False
        return actual_return > expected