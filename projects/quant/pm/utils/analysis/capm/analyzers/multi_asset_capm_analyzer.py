import numpy as np
import pandas as pd
from typing import List
from .capm_analyzer import CAPMAnalyzer
from ....tools.config import (
    ANNUAL_FACTOR,
    SIGNIFICANCE_LEVEL,
    MIN_OBSERVATIONS,
    ALPHA_THRESHOLDS_IDENTIFICATION
)

class MultiAssetCAPMAnalyzer:
    def __init__(self, annual_factor: float = None, significance_level: float = None):
        self.annual_factor = annual_factor or ANNUAL_FACTOR
        self.significance_level = significance_level or SIGNIFICANCE_LEVEL
        self.capm_analyzer = CAPMAnalyzer(self.annual_factor, self.significance_level)
    
    def analyze_multiple(
        self,
        returns: pd.DataFrame,
        market_returns: pd.Series,
        risk_free_rate: float
    ) -> pd.DataFrame:

        results = []

        common_idx = returns.index.intersection(market_returns.index)
        returns_aligned = returns.loc[common_idx]
        market_aligned = market_returns.loc[common_idx]
        
        for asset in returns_aligned.columns:
            asset_ret = returns_aligned[asset].values
            market_ret = market_aligned.values
 
            mask = ~(np.isnan(asset_ret) | np.isnan(market_ret))
            asset_ret = asset_ret[mask]
            market_ret = market_ret[mask]
            
            if len(asset_ret) < MIN_OBSERVATIONS:  
                continue
            
            analysis = self.capm_analyzer.analyze(asset_ret, market_ret, risk_free_rate)
            analysis['asset'] = asset
            results.append(analysis)
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results).set_index('asset')
        df = df.sort_values('alpha_annual', ascending=False)
        
        return df
    
    def identify_outperformers(
        self,
        returns: pd.DataFrame,
        market_returns: pd.Series,
        risk_free_rate: float,
        min_alpha: float = None
    ) -> List[str]:

        if min_alpha is None:
            min_alpha = ALPHA_THRESHOLDS_IDENTIFICATION['outperformer']

        analysis = self.analyze_multiple(returns, market_returns, risk_free_rate)
        
        if analysis.empty:
            return []
        
        outperformers = analysis[
            (analysis['is_significant']) &
            (analysis['alpha_annual'] > min_alpha)
        ]
        
        return outperformers.index.tolist()
    
    def identify_underperformers(
        self,
        returns: pd.DataFrame,
        market_returns: pd.Series,
        risk_free_rate: float,
        max_alpha: float = None
    ) -> List[str]:

        if max_alpha is None:
            max_alpha = ALPHA_THRESHOLDS_IDENTIFICATION['underperformer']

        analysis = self.analyze_multiple(returns, market_returns, risk_free_rate)
        
        if analysis.empty:
            return []
        
        underperformers = analysis[
            (analysis['is_significant']) &
            (analysis['alpha_annual'] < max_alpha)
        ]
        
        return underperformers.index.tolist()