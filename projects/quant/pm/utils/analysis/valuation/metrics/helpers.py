import math
import numpy as np
import pandas as pd
from typing import Dict

def nan_if_missing(x) -> float:

    if x is None or (isinstance(x, float) and math.isnan(x)):
        return np.nan
    return x

def safe_div(numer, denom) -> float:
    
    try:
        if denom is None or numer is None:
            return np.nan
        denom = float(denom)
        numer = float(numer)
        return np.nan if denom == 0 else numer / denom
    except Exception:
        return np.nan

def classify_metric(value: float, thresholds: Dict[str, float]) -> str:

    if pd.isna(value):
        return 'N/A'
    for level, threshold in sorted(thresholds.items(), key=lambda x: -x[1]):
        if value >= threshold:
            return level
    return 'poor'

def score_metric(value: float, min_val: float, max_val: float, higher_is_better: bool = True) -> float:

    if pd.isna(value) or not np.isfinite(value):
        return np.nan
    value = np.clip(value, min_val, max_val)
    score = (value - min_val) / (max_val - min_val) * 100
    return score if higher_is_better else 100 - score