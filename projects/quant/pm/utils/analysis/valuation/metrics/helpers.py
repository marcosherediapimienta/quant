import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass


def nan_if_missing(x) -> float:
    return np.nan if pd.isna(x) else x

def safe_div(numer, denom) -> float:
    
    try:
        if denom is None or numer is None:
            return np.nan
        denom = float(denom)
        numer = float(numer)
        return np.nan if denom == 0 else numer / denom
    except Exception:
        return np.nan

def classify_metric(
    value: float,
    thresholds: Dict[str, float],
    higher_is_better: bool = True,
    strict: bool = False,
    default: str = 'poor'
) -> str:

    if pd.isna(value):
        return 'N/A'

    if higher_is_better:
        for level, threshold in sorted(thresholds.items(), key=lambda x: -x[1]):
            if value >= threshold:
                return level
    else:
        for level, threshold in sorted(thresholds.items(), key=lambda x: x[1]):
            if (value < threshold) if strict else (value <= threshold):
                return level

    return default

def score_metric(value: float, min_val: float, max_val: float, higher_is_better: bool = True) -> float:

    if pd.isna(value) or not np.isfinite(value):
        return np.nan
    value = np.clip(value, min_val, max_val)
    score = (value - min_val) / (max_val - min_val) * 100
    return score if higher_is_better else 100 - score


@dataclass
class MetricSpec:
    key: str
    range_key: str
    weight_key: str
    higher_is_better: bool = True
    require_positive: bool = False
    binary_positive: bool = False


class WeightedScorer:

    @staticmethod
    def calculate(
        metrics: Dict,
        specs: List[MetricSpec],
        weights: Dict[str, float],
        ranges: Dict[str, Dict[str, float]]
    ) -> float:
        scores = []
        weights_used = []

        for spec in specs:
            value = metrics.get(spec.key)

            if pd.isna(value):
                continue
            if spec.require_positive and value <= 0:
                continue

            w = weights.get(spec.weight_key)
            if w is None:
                continue

            if spec.binary_positive:
                s = 100.0 if value > 0 else 0.0
            else:
                r = ranges.get(spec.range_key)
                if r is None:
                    continue
                s = score_metric(value, r['min'], r['max'], spec.higher_is_better)

            scores.append(s)
            weights_used.append(w)

        if not scores:
            return np.nan

        total = sum(weights_used)
        if total <= 0:
            return np.nan

        return sum(s * w for s, w in zip(scores, weights_used)) / total
