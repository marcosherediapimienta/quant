import numpy as np
import pandas as pd
from typing import List, Tuple
from ....tools.config import ANNUAL_FACTOR

def daily_risk_free_rate(annual_rate: float, annual_factor: float = None) -> float:

    if annual_factor is None:
        annual_factor = ANNUAL_FACTOR
    return (1 + annual_rate) ** (1 / annual_factor) - 1

def annualize_return(daily_returns: np.ndarray, annual_factor: float = None) -> float:

    if annual_factor is None:
        annual_factor = ANNUAL_FACTOR
    
    if len(daily_returns) == 0:
        return np.nan
    return float(daily_returns.mean() * annual_factor)

def annualize_volatility(daily_returns: np.ndarray, annual_factor: float = None) -> float:

    if annual_factor is None:
        annual_factor = ANNUAL_FACTOR
    
    if len(daily_returns) == 0:
        return np.nan
    return float(daily_returns.std() * np.sqrt(annual_factor))

def normalize_weights(
    weights: np.ndarray,
    warn: bool = True
) -> np.ndarray:

    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    
    if not np.isclose(total, 1.0) and warn:
        print(f"Pesos normalizados: {total:.4f} → 1.0")
    
    return weights / total if total != 0 else weights

def align_weights_to_assets(
    assets: List[str],
    original_tickers: List[str],
    original_weights: np.ndarray
) -> Tuple[List[str], np.ndarray]:

    weight_map = dict(zip(original_tickers, original_weights))
    kept_tickers = [t for t in assets if t in weight_map]
    kept_weights = np.array([weight_map[t] for t in kept_tickers], dtype=float)
    
    return kept_tickers, normalize_weights(kept_weights)

def portfolio_returns(
    returns: pd.DataFrame,
    weights: np.ndarray
) -> pd.Series:

    return (returns * weights).sum(axis=1)