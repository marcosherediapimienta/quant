import pandas as pd
from typing import Dict, List
from ..components.factor_collinearity import FactorCollinearityAnalyzer

class FactorSelectionAnalyzer:
    def __init__(
        self,
        corr_threshold: float,
        vif_threshold: float,
        force_keep: List[str]
    ):
        self.force_keep = force_keep
        self.collinearity = FactorCollinearityAnalyzer(
            corr_threshold=corr_threshold,
            vif_threshold=vif_threshold
        )

    def analyze(self, factors: pd.DataFrame) -> Dict[str, object]:
        flags_before = self.collinearity.flag_factors(factors)
        pruned = self.collinearity.prune_by_corr(factors, keep=self.force_keep)
        flags_after = self.collinearity.flag_factors(pruned)
        return {
            "pruned_factors": pruned,
            "flags_before": flags_before,
            "flags_after": flags_after
        }