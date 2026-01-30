import numpy as np
from dataclasses import dataclass
from ....tools.config import EPSILON, CML_EXTENSION_FACTOR

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

        sharpe = (frontier_returns - risk_free_rate) / np.maximum(frontier_volatilities, EPSILON)
        valid_indices = ~np.isnan(sharpe) & ~np.isnan(frontier_returns) & ~np.isnan(frontier_volatilities)
        
        if not np.any(valid_indices):
            return CMLResult(
                np.array([]), np.array([]), np.nan, np.nan, -1, np.nan
            )

        valid_sharpe = sharpe[valid_indices]
        valid_returns = frontier_returns[valid_indices]
        valid_volatilities = frontier_volatilities[valid_indices]
    
        tangent_idx_valid = int(np.argmax(valid_sharpe))
        tangent_ret = valid_returns[tangent_idx_valid]
        tangent_vol = valid_volatilities[tangent_idx_valid]
    
        valid_indices_list = np.where(valid_indices)[0]
        tangent_idx = valid_indices_list[tangent_idx_valid]

        slope = (tangent_ret - risk_free_rate) / tangent_vol if tangent_vol > 0 else 0.0

        max_vol = frontier_volatilities.max() * CML_EXTENSION_FACTOR
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