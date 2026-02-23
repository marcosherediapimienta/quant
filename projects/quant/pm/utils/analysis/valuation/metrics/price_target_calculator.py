import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple
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

    def calculate_from_peg(self, current_price: float, pe: float, peg: float) -> float:
        if pd.isna(pe) or pe <= 0 or pd.isna(peg) or peg <= 0:
            return np.nan
        
        cfg = self.config['peg_method']
        eps = current_price / pe
        implied_growth = pe / peg
        fair_peg = cfg['fair_peg']
        fair_pe = fair_peg * implied_growth
        return eps * fair_pe

    def calculate_from_pe(self, current_price: float, pe: float, earnings_growth: float = None) -> float:
        if pd.isna(pe) or pe <= 0:
            return np.nan
        
        cfg = self.config['pe_method']
        eps = current_price / pe

        if earnings_growth is not None and pd.notna(earnings_growth) and earnings_growth > 0:
            if earnings_growth >= cfg['earnings_growth_threshold']:
                earnings_growth = earnings_growth / 100
            
            fair_pe_from_growth = earnings_growth * 100 * cfg['growth_multiplier']
            fair_pe = (
                pe * cfg['pe_weight'] + 
                fair_pe_from_growth * cfg['growth_weight']
            )
        else:
            fair_pe = pe * cfg['fair_multiplier_base']

        return eps * fair_pe
    
    def calculate_from_analyst_target(self, current_price: float, target_price: float) -> float:
        if pd.isna(target_price) or target_price <= 0:
            return np.nan

        return float(target_price)
    
    def calculate_from_score(self, current_price: float, valuation_score: float) -> float:
        cfg = self.config['score_method']

        neutral = DEFAULT_NA_SCORE
        divisor_key = 'adjustment_divisor_bear' if valuation_score < neutral else 'adjustment_divisor_bull'
        adjustment = (valuation_score - neutral) / cfg[divisor_key]
        
        return current_price * (1 + adjustment)
    
    def calculate(self, data: Dict, valuation_score: float, current_price: float) -> Tuple[float, float]:
        raw = self._calculate_raw(data, valuation_score, current_price)

        if not np.isfinite(raw) or raw <= 0:
            raw = self.calculate_from_score(current_price, valuation_score)

        clamped = self._clamp_price_target(raw, current_price, 'Final')
        return clamped, raw

    def _is_valid_raw(self, result: float, current_price: float) -> bool:
        cfg = self.config['raw_validation']
        return pd.notna(result) and cfg['min_factor'] * current_price <= result <= cfg['max_factor'] * current_price

    def _calculate_raw(self, data: Dict, valuation_score: float, current_price: float) -> float:
        target_price = data.get('targetMeanPrice', np.nan)
        if pd.notna(target_price) and target_price > 0:
            result = self.calculate_from_analyst_target(current_price, target_price)
            if self._is_valid_raw(result, current_price):
                return result

        pe = data.get('trailingPE', np.nan)
        peg = data.get('pegRatio', np.nan)

        if pd.notna(pe) and pd.notna(peg) and pe > 0 and peg > 0:
            result = self.calculate_from_peg(current_price, pe, peg)
            if self._is_valid_raw(result, current_price):
                return result

        earnings_growth = data.get('earningsQuarterlyGrowth', np.nan)
        if pd.isna(earnings_growth):
            earnings_growth = data.get('earningsGrowth', np.nan)

        if pd.notna(pe) and pe > 0:
            result = self.calculate_from_pe(current_price, pe, earnings_growth)
            if pd.notna(result) and result > 0:
                return result

        return self.calculate_from_score(current_price, valuation_score)
