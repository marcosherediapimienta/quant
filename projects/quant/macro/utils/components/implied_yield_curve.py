import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ForwardRateAnalysis:
    spot_rates: Dict[str, float]          
    forward_rates: Dict[str, float]        
    term_premium: Dict[str, float]       
    breakeven_inflation: Optional[float]  
    curve_expectations: str               
    rate_path_signal: str                  
    forward_vs_spot: Dict[str, float]     

class ImpliedYieldCurveCalculator:
    TENOR_YEARS = {
        '3M': 0.25, '2Y': 2.0, '5Y': 5.0, '10Y': 10.0, '30Y': 30.0
    }
    
    def calculate_forward_rate(
        self,
        rate_short: float,
        tenor_short: float,
        rate_long: float,
        tenor_long: float
    ) -> float:

        r1 = rate_short / 100.0
        r2 = rate_long / 100.0
        dt = tenor_long - tenor_short
        
        if dt <= 0 or (1 + r1) <= 0 or (1 + r2) <= 0:
            return np.nan
        
        forward = ((1 + r2) ** tenor_long / (1 + r1) ** tenor_short) ** (1.0 / dt) - 1
        return forward * 100.0 
    
    def calculate_all_forwards(
        self,
        spot_rates: Dict[str, float]
    ) -> Dict[str, float]:

        forwards = {}

        available = {t: y for t, y in self.TENOR_YEARS.items() if t in spot_rates}
        sorted_tenors = sorted(available.items(), key=lambda x: x[1])
        
        for i in range(len(sorted_tenors) - 1):
            tenor_short, years_short = sorted_tenors[i]
            tenor_long, years_long = sorted_tenors[i + 1]
            
            fwd = self.calculate_forward_rate(
                spot_rates[tenor_short], years_short,
                spot_rates[tenor_long], years_long
            )
            
            label = f"{tenor_short}→{tenor_long}"
            forwards[label] = fwd

        if '2Y' in spot_rates and '10Y' in spot_rates:
            fwd_2_10 = self.calculate_forward_rate(
                spot_rates['2Y'], 2.0,
                spot_rates['10Y'], 10.0
            )
            forwards['2Y→10Y'] = fwd_2_10
        
        if '5Y' in spot_rates and '30Y' in spot_rates:
            fwd_5_30 = self.calculate_forward_rate(
                spot_rates['5Y'], 5.0,
                spot_rates['30Y'], 30.0
            )
            forwards['5Y→30Y'] = fwd_5_30
        
        return forwards
    
    def estimate_term_premium(
        self,
        spot_rates: Dict[str, float],
        forward_rates: Dict[str, float]
    ) -> Dict[str, float]:

        term_premium = {}
 
        if '10Y' in spot_rates and '2Y→10Y' in forward_rates:

            tp_10y = spot_rates['10Y'] - forward_rates['2Y→10Y']
            term_premium['10Y'] = tp_10y
        
        if '30Y' in spot_rates and '5Y→30Y' in forward_rates:
            tp_30y = spot_rates['30Y'] - forward_rates['5Y→30Y']
            term_premium['30Y'] = tp_30y
        
        return term_premium
    
    def calculate_breakeven_inflation(
        self,
        nominal_rate: float,
        tips_yield_proxy: float,
        tips_is_etf: bool = True
    ) -> Optional[float]:

        if tips_is_etf:
            return None
        else:
            return nominal_rate - tips_yield_proxy
    
    def interpret_forward_curve(
        self,
        spot_rates: Dict[str, float],
        forward_rates: Dict[str, float]
    ) -> Tuple[str, str]:

        forward_vs_spot = {}
        
        if '2Y→5Y' in forward_rates and '5Y' in spot_rates:
            forward_vs_spot['5Y'] = forward_rates['2Y→5Y'] - spot_rates['5Y']
        
        if '5Y→10Y' in forward_rates and '10Y' in spot_rates:
            forward_vs_spot['10Y'] = forward_rates['5Y→10Y'] - spot_rates['10Y']
        
        if '2Y→10Y' in forward_rates and '10Y' in spot_rates:
            forward_vs_spot['10Y_wide'] = forward_rates['2Y→10Y'] - spot_rates['10Y']

        avg_diff = np.mean(list(forward_vs_spot.values())) if forward_vs_spot else 0
        
        if avg_diff > 0.5:
            expectations = "Market expects RATE HIKES"
            signal = "HAWKISH"
        elif avg_diff > 0.1:
            expectations = "Market expects MODERATE HIKES"
            signal = "SLIGHTLY HAWKISH"
        elif avg_diff > -0.1:
            expectations = "Market expects STABILITY"
            signal = "NEUTRAL"
        elif avg_diff > -0.5:
            expectations = "Market expects MODERATE CUTS"
            signal = "SLIGHTLY DOVISH"
        else:
            expectations = "Market expects RATE CUTS"
            signal = "DOVISH"
        
        return expectations, signal
    
    def analyze(
        self,
        factors_data: Dict[str, pd.Series]
    ) -> ForwardRateAnalysis:

        tenors_map = {
            'RATE_3M': '3M',
            'RATE_2Y': '2Y',
            'RATE_5Y': '5Y',
            'RATE_10Y': '10Y',
            'RATE_30Y': '30Y'
        }
        
        spot_rates = {}
        for factor, label in tenors_map.items():
            if factor in factors_data and len(factors_data[factor]) > 0:
                spot_rates[label] = factors_data[factor].iloc[-1]

        forward_rates = self.calculate_all_forwards(spot_rates)

        term_premium = self.estimate_term_premium(spot_rates, forward_rates)

        breakeven = None
        if 'RATE_10Y' in factors_data and 'TIPS' in factors_data:
            breakeven = self.calculate_breakeven_inflation(
                spot_rates.get('10Y', np.nan),
                factors_data['TIPS'].iloc[-1] if len(factors_data.get('TIPS', [])) > 0 else np.nan,
                tips_is_etf=True
            )

        expectations, signal = self.interpret_forward_curve(spot_rates, forward_rates)

        fwd_vs_spot = {}
        for key, fwd in forward_rates.items():
            parts = key.split('→')
            if len(parts) == 2:
                target_tenor = parts[1]
                if target_tenor in spot_rates:
                    fwd_vs_spot[key] = fwd - spot_rates[target_tenor]
        
        return ForwardRateAnalysis(
            spot_rates=spot_rates,
            forward_rates=forward_rates,
            term_premium=term_premium,
            breakeven_inflation=breakeven,
            curve_expectations=expectations,
            rate_path_signal=signal,
            forward_vs_spot=fwd_vs_spot
        )