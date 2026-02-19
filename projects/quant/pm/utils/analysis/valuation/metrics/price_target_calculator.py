import logging
import numpy as np
import pandas as pd
from typing import Dict
from ....tools.config import PRICE_TARGET_CONFIG, DEFAULT_NA_SCORE

logger = logging.getLogger(__name__)

class PriceTargetCalculator:
    def __init__(self, config: dict = None):
        self.config = config or PRICE_TARGET_CONFIG

    def _clamp_price_target(self, price_target: float, current_price: float, method: str) -> float:
        cfg = self.config['clamp']
        lo = current_price * cfg['max_downside_factor']
        hi = current_price * cfg['max_upside_factor']
        clamped = float(np.clip(price_target, lo, hi))
        if clamped != price_target:
            logger.debug(f"Price target {method} ({price_target:.2f}) clamped to {clamped:.2f}")
        return clamped

    def calculate_from_peg(
        self,
        current_price: float,
        pe: float,
        peg: float
    ) -> float:

        if pd.isna(pe) or pe <= 0 or pd.isna(peg) or peg <= 0:
            return np.nan
        
        cfg = self.config['peg_method']
        eps = current_price / pe
        implied_growth = pe / peg
        fair_peg = cfg['fair_peg']
        fair_pe = fair_peg * implied_growth
        price_target = eps * fair_pe
        
        return self._clamp_price_target(price_target, current_price, 'PEG')

    def calculate_from_pe(
        self, 
        current_price: float, 
        pe: float, 
        earnings_growth: float = None
    ) -> float:
 
        if pd.isna(pe) or pe <= 0:
            return np.nan
        
        cfg = self.config['pe_method']
        eps = current_price / pe

        if earnings_growth and earnings_growth > 0:
            if earnings_growth >= cfg['earnings_growth_threshold']:
                earnings_growth = earnings_growth / 100
            
            fair_pe_from_growth = earnings_growth * 100 * cfg['growth_multiplier']
            fair_pe = (
                pe * cfg['pe_weight'] + 
                fair_pe_from_growth * cfg['growth_weight']
            )
        else:
            fair_pe = pe * cfg['fair_multiplier_base']

        price_target = eps * fair_pe

        return self._clamp_price_target(price_target, current_price, 'P/E')
    
    def calculate_from_analyst_target(
        self,
        current_price: float,
        target_price: float
    ) -> float:

        if pd.isna(target_price) or target_price <= 0:
            return np.nan

        return float(target_price)
    
    def calculate_from_score(
        self, 
        current_price: float, 
        valuation_score: float
    ) -> float:

        cfg = self.config['score_method']

        neutral = DEFAULT_NA_SCORE
        divisor_key = 'adjustment_divisor_bear' if valuation_score < neutral else 'adjustment_divisor_bull'
        adjustment = (valuation_score - neutral) / cfg[divisor_key]
        
        return current_price * (1 + adjustment)
    
    def calculate(
        self, 
        data: Dict, 
        valuation_score: float, 
        current_price: float
    ) -> float:

        target_price = data.get('targetMeanPrice', np.nan)
        if pd.notna(target_price) and target_price > 0:
            result = self.calculate_from_analyst_target(
                current_price, 
                target_price
            )
            if pd.notna(result):
                return result

        pe = data.get('trailingPE', np.nan)
        peg = data.get('pegRatio', np.nan)
        
        if pd.notna(pe) and pd.notna(peg) and pe > 0 and peg > 0:
            result = self.calculate_from_peg(current_price, pe, peg)
            if pd.notna(result):
                return result

        earnings_growth = data.get('earningsQuarterlyGrowth')
        if pd.isna(earnings_growth):
            earnings_growth = data.get('earningsGrowth')
        
        if pd.notna(pe) and pe > 0:
            return self.calculate_from_pe(
                current_price, 
                pe, 
                earnings_growth
            )

        return self.calculate_from_score(current_price, valuation_score)
