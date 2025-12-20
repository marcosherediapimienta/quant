import pandas as pd
from typing import Dict
from ..components.macro_correlation import (
    best_lagged_correlation,
    correlation_matrix_with_lags,
    rolling_correlation,
    lagged_correlation
)
from ..tools.config import MAX_LAG

class MacroCorrelationAnalyzer:

    def __init__(self, max_lag: int = None):
        self.max_lag = max_lag if max_lag is not None else MAX_LAG
    
    def analyze(
        self,
        portfolio_returns: pd.Series,
        macro_factors: pd.DataFrame
    ) -> Dict:

        best_lags = best_lagged_correlation(
            portfolio_returns,
            macro_factors,
            self.max_lag
        )

        corr_matrix_lags = correlation_matrix_with_lags(
            portfolio_returns,
            macro_factors,
            lags=[0, 1, 5, 21, 63, 126]
        )
        
        return {
            'best_lagged_correlations': best_lags,
            'correlation_by_lag': corr_matrix_lags,
            'top_factors': best_lags.head(5),
            'bottom_factors': best_lags.tail(5)
        }
    
    def analyze_rolling(
        self,
        portfolio_returns: pd.Series,
        macro_factors: pd.DataFrame,
        window: int = 252
    ) -> pd.DataFrame:

        rolling_corrs = {}
        
        for factor_name in macro_factors.columns:
            rolling_corrs[factor_name] = rolling_correlation(
                portfolio_returns,
                macro_factors[factor_name],
                window
            )
        
        return pd.DataFrame(rolling_corrs)