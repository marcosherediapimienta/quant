import numpy as np
from dataclasses import dataclass


@dataclass
class CMLResult:
    cml_returns: np.ndarray
    cml_volatilities: np.ndarray
    tangent_return: float
    tangent_volatility: float
    tangent_index: int
    slope: float


class CMLCalculator:

    def calculate(
        self,
        frontier_returns: np.ndarray,
        frontier_volatilities: np.ndarray,
        risk_free_rate: float,
        n_points: int = 100
    ) -> CMLResult:

        if len(frontier_returns) == 0:
            return CMLResult(
                np.array([]), np.array([]), np.nan, np.nan, -1, np.nan
            )

        sharpe = (frontier_returns - risk_free_rate) / np.maximum(frontier_volatilities, 1e-12)

        tangent_idx = int(np.argmax(sharpe))
        tangent_ret = frontier_returns[tangent_idx]
        tangent_vol = frontier_volatilities[tangent_idx]

        slope = (tangent_ret - risk_free_rate) / tangent_vol if tangent_vol > 0 else 0.0

        max_vol = frontier_volatilities.max() * 1.2
        vol_grid = np.linspace(0, max_vol, n_points)
        ret_grid = risk_free_rate + slope * vol_grid
        
        return CMLResult(
            ret_grid, vol_grid, tangent_ret, tangent_vol, tangent_idx, slope
        )
    
    def sharpe_ratio(
        self,
        expected_return: float,
        volatility: float,
        risk_free_rate: float
    ) -> float:

        if np.isnan(expected_return) or np.isnan(volatility) or volatility <= 0:
            return np.nan
        return (expected_return - risk_free_rate) / volatility