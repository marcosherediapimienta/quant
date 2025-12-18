import numpy as np
import pandas as pd
from typing import Union


def calculate_portfolio_returns(
    returns: pd.DataFrame,
    weights: np.ndarray
) -> pd.Series:

    weights = np.asarray(weights, dtype=float)
    
    if returns.shape[1] != len(weights):
        raise ValueError(
            f"Dimensión de pesos ({len(weights)}) ≠ "
            f"número de activos ({returns.shape[1]})"
        )

    if not np.isclose(weights.sum(), 1.0):
        weights = weights / weights.sum()
    
    return (returns * weights).sum(axis=1)


def annualize_return(
    returns: Union[pd.Series, float],
    periods_per_year: float = 252.0
) -> float:

    if isinstance(returns, pd.Series):
        mean_ret = returns.mean()
    else:
        mean_ret = float(returns)
    
    return float(mean_ret * periods_per_year)


def annualize_volatility(
    returns: Union[pd.Series, float],
    periods_per_year: float = 252.0,
    ddof: int = 0
) -> float:

    if isinstance(returns, pd.Series):
        vol = returns.std(ddof=ddof)
    else:
        vol = float(returns)
    
    return float(vol * np.sqrt(periods_per_year))


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    
    if total == 0:
        raise ValueError("La suma de pesos es cero")
    return weights / total