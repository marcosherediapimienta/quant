import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Dict, List, Tuple
import warnings

class FactorCollinearityAnalyzer:
    def __init__(self, corr_threshold: float = 0.9, vif_threshold: float = 10.0):
        self.corr_threshold = corr_threshold
        self.vif_threshold = vif_threshold

    def compute_corr(self, factors: pd.DataFrame) -> pd.DataFrame:
        return factors.corr()

    def compute_vif(self, factors: pd.DataFrame) -> pd.Series:
        X = factors.dropna()
        X_const = sm.add_constant(X)
        
        vif_values = []
        for i in range(X.shape[1]):
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    vif = variance_inflation_factor(X_const.values, i+1)

                if not np.isfinite(vif) or vif > 1000:
                    vif = np.inf
                vif_values.append(vif)
            except:
                vif_values.append(np.inf)
        
        return pd.Series(
            vif_values,
            index=X.columns,
            name="VIF"
        )
    def high_corr_pairs(self, corr: pd.DataFrame) -> List[Tuple[str, str, float]]:
        pairs = []
        cols = corr.columns
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                c = corr.iloc[i, j]
                if abs(c) >= self.corr_threshold:
                    pairs.append((cols[i], cols[j], c))
        return sorted(pairs, key=lambda x: abs(x[2]), reverse=True)

    def flag_factors(self, factors: pd.DataFrame) -> Dict[str, Dict]:
        corr = self.compute_corr(factors)
        vif = self.compute_vif(factors)
        pairs = self.high_corr_pairs(corr)
        high_vif = vif[vif >= self.vif_threshold].sort_values(ascending=False)
        return {
            "corr_matrix": corr,
            "vif": vif,
            "high_corr_pairs": pairs,
            "high_vif": high_vif
        }

    def prune_by_corr(self, factors: pd.DataFrame, keep: List[str] = None) -> pd.DataFrame:
        keep_set = set(keep or [])
        current_factors = factors.copy()
        
        max_iterations = 10
        for iteration in range(max_iterations):
            corr = self.compute_corr(current_factors)
            vif = self.compute_vif(current_factors)
            
            to_drop = set()

            inf_vif = vif[~np.isfinite(vif)]
            if len(inf_vif) > 0:
                for factor in inf_vif.index:
                    if factor not in keep_set:
                        to_drop.add(factor)
                        break  

            if len(to_drop) == 0:
                for a, b, c in self.high_corr_pairs(corr):
                    if a in keep_set and b in keep_set:
                        continue
                    if a in keep_set:
                        to_drop.add(b)
                    elif b in keep_set:
                        to_drop.add(a)
                    else:
                        to_drop.add(b)
                    break  

            if len(to_drop) == 0:
                break
            
            current_factors = current_factors.drop(columns=list(to_drop), errors="ignore")
        
        return current_factors