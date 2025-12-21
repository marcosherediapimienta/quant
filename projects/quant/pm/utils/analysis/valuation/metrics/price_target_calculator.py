import numpy as np
import pandas as pd
from typing import Dict

class PriceTargetCalculator:

    def calculate_from_peg(
        self,
        current_price: float,
        pe: float,
        peg: float,
        valuation_score: float
    ) -> float:
 
        if pd.isna(pe) or pe <= 0 or pd.isna(peg) or peg <= 0:
            return np.nan
        
        eps = current_price / pe
        fair_peg = 1.5
        implied_growth = pe / peg if peg > 0 else 0
        fair_pe = fair_peg * implied_growth
        adjustment = 1.0 + (valuation_score - 50) / 250  
        fair_pe = fair_pe * adjustment
        
        price_target = eps * fair_pe
        
        return price_target

    def calculate_from_pe(
        self, 
        current_price: float, 
        pe: float, 
        valuation_score: float,
        earnings_growth: float = None
    ) -> float:

        if pd.isna(pe) or pe <= 0:
            return np.nan
        
        eps = current_price / pe

        if earnings_growth and earnings_growth > 0:
            fair_pe_from_growth = earnings_growth * 1.5
            fair_pe = pe * 0.3 + fair_pe_from_growth * 0.7
        else:
            fair_multiplier = 0.7 + (valuation_score / 100) * 0.6  
            fair_pe = pe * fair_multiplier

        adjustment = 1.0 + (valuation_score - 50) / 333 
        fair_pe = fair_pe * adjustment
        
        price_target = eps * fair_pe
        
        return price_target
    
    def calculate_from_analyst_target(
        self,
        current_price: float,
        target_price: float,
        valuation_score: float
    ) -> float:

        if pd.isna(target_price) or target_price <= 0:
            return np.nan

        if target_price > current_price:
            confidence = valuation_score / 100 
            upside = target_price - current_price
            adjusted_upside = upside * (0.5 + confidence * 0.5)  
            return current_price + adjusted_upside
        else:
            confidence = 1 - (valuation_score / 100)  
            downside = current_price - target_price
            adjusted_downside = downside * (0.5 + confidence * 0.5)
            return current_price - adjusted_downside
    
    def calculate_from_score(
        self, 
        current_price: float, 
        valuation_score: float
    ) -> float:

        if valuation_score < 50:
            adjustment = (valuation_score - 50) / 250 
        else:
            adjustment = (valuation_score - 50) / 333  
        
        return current_price * (1 + adjustment)
    
    def calculate(
        self, 
        data: Dict, 
        valuation_score: float, 
        current_price: float
    ) -> float:

        target_price = data.get('targetMeanPrice', np.nan)
        if pd.notna(target_price) and target_price > 0:
            return self.calculate_from_analyst_target(
                current_price, 
                target_price, 
                valuation_score
            )

        pe = data.get('trailingPE', np.nan)
        peg = data.get('pegRatio', np.nan)
        if pd.notna(pe) and pd.notna(peg) and pe > 0 and peg > 0:
            result = self.calculate_from_peg(current_price, pe, peg, valuation_score)
            if pd.notna(result):
                return result

        earnings_growth = data.get('earningsQuarterlyGrowth')
        if pd.isna(earnings_growth):
            earnings_growth = data.get('earningsGrowth')
        
        if pd.notna(pe) and pe > 0:

            if earnings_growth and 0 < earnings_growth < 2:
                earnings_growth = earnings_growth * 100
            
            return self.calculate_from_pe(
                current_price, 
                pe, 
                valuation_score,
                earnings_growth
            )

        return self.calculate_from_score(current_price, valuation_score)