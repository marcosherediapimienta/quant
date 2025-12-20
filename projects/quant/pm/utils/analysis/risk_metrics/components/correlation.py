import numpy as np
import pandas as pd
from typing import Dict

class CorrelationCalculator:

    def __init__(self):
        pass
    
    def calculate(
        self,
        returns: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:

        corr_matrix = returns.corr()
        
        return {
            'correlation_matrix': corr_matrix,
            'mean_correlation': float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean())
        }
    
    def calculate_rolling(
        self,
        returns: pd.DataFrame,
        window: int = 252
    ) -> pd.DataFrame:

        n_assets = returns.shape[1]
        
        if n_assets < 2:
            return pd.DataFrame()
        
        if n_assets == 2:
            col1, col2 = returns.columns
            corr_rolling = returns[col1].rolling(window=window).corr(returns[col2])
            return pd.DataFrame({f'{col1}_{col2}': corr_rolling})
            
        corr_dict = {}
        cols = returns.columns
        
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                pair_name = f'{cols[i]}_{cols[j]}'
                corr_dict[pair_name] = returns[cols[i]].rolling(window=window).corr(returns[cols[j]])
        
        return pd.DataFrame(corr_dict)
    
    def calculate_correlation_volatility(
        self,
        returns: pd.DataFrame,
        window: int = 252
    ) -> float:

        corr_rolling = self.calculate_rolling(returns, window)
        
        if corr_rolling.empty:
            return np.nan
        
        corr_vol = corr_rolling.std().mean()
        
        return float(corr_vol)