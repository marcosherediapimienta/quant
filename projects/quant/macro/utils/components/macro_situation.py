import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass
from ..tools.config import (
    PERIOD_WEEK,
    PERIOD_MONTH,
    PERIOD_QUARTER,
    PERIOD_YEAR,
    MACRO_SITUATION_THRESHOLDS,
)

logger = logging.getLogger(__name__)

@dataclass
class YieldCurveAnalysis:
    levels: Dict[str, float]
    spreads: Dict[str, float]
    rate_changes: Dict[str, Dict[str, float]]
    divergence_analysis: Dict[str, Dict]
    interpretation: str
    risk_level: str

@dataclass
class InflationSignals:
    commodity_changes: Dict[str, float]
    commodity_names: Dict[str, str]
    inflation_pressure: str
    avg_commodity_change: float

@dataclass
class CreditConditions:
    vix_level: float
    market_condition: str
    hyg_level: float = None
    lqd_level: float = None

@dataclass
class RiskSentiment:
    vix: float
    fear_level: str
    dollar_strength: str
    dxy_trend: float = None
    dxy_trend_3m: float = None
    dxy_trend_1m: float = None
    dxy_trend_1w: float = None
    gold_trend: float = None
    gold_trend_3m: float = None
    gold_trend_1m: float = None
    gold_trend_1w: float = None
    safe_haven: str = None

class MacroSituationAnalyzer:

    _VIX_MARKET_LEVELS = (
        ('panic',   "PANIC — Extreme stress"),
        ('stress',  "STRESS — High tension"),
        ('tension', "TENSION — Elevated anxiety"),
        ('normal',  "NORMAL — Controlled volatility"),
    )
    _VIX_MARKET_DEFAULT = "COMPLACENCY — Very low volatility"

    _VIX_FEAR_LEVELS = (
        ('panic',   "PANIC"),
        ('stress',  "EXTREME FEAR"),
        ('tension', "ANXIETY"),
        ('normal',  "MODERATE"),
    )
    _VIX_FEAR_DEFAULT = "COMPLACENCY"

    _INFLATION_LEVELS = (
        ('high',     "HIGH - Strong inflationary pressure"),
        ('moderate', "MODERATE - Controlled inflation"),
        ('low',      "LOW - Contained inflation"),
    )
    _INFLATION_DEFAULT = "DEFLATION - Falling prices"

    @staticmethod
    def _classify(value: float, category: str, levels: tuple, default: str) -> str:
        thresholds = MACRO_SITUATION_THRESHOLDS[category]
        for key, label in levels:
            if value > thresholds[key]:
                return label
        return default

    @staticmethod
    def _calculate_trend(series: pd.Series, period: int) -> Optional[float]:
        if len(series) >= period and series.iloc[-period] > 0:
            return (series.iloc[-1] / series.iloc[-period] - 1) * 100
        return None

    @staticmethod
    def _pct_change(series: pd.Series, period: int) -> float:
        if len(series) >= period:
            past = series.iloc[-period]
            if past > 0:
                return (series.iloc[-1] / past - 1) * 100
        return np.nan
    
    def analyze_yield_curve_usa(
        self,
        factors_data: Dict[str, pd.Series]
    ) -> YieldCurveAnalysis:

        rates = {}
        rate_changes = {}
        tenors_map = {
            'RATE_2Y': '2Y',
            'RATE_5Y': '5Y',
            'RATE_10Y': '10Y',
            'RATE_30Y': '30Y'
        }
        
        for factor, label in tenors_map.items():
            if factor in factors_data and len(factors_data[factor]) > 0:
                series = factors_data[factor]
                current = series.iloc[-1]
                rates[label] = current

                changes = {}

                if len(series) >= PERIOD_MONTH:
                    month_ago = series.iloc[-PERIOD_MONTH]
                    changes['1m'] = current - month_ago

                if len(series) >= PERIOD_QUARTER:
                    quarter_ago = series.iloc[-PERIOD_QUARTER]
                    changes['3m'] = current - quarter_ago

                if len(series) >= PERIOD_YEAR:
                    year_ago = series.iloc[-PERIOD_YEAR]
                    changes['1y'] = current - year_ago
                
                if changes:
                    rate_changes[label] = changes

        spreads = {}
        if '10Y' in rates and '2Y' in rates:
            spreads['10Y-2Y'] = rates['10Y'] - rates['2Y']
        if '10Y' in rates and '5Y' in rates:
            spreads['10Y-5Y'] = rates['10Y'] - rates['5Y']
        if '30Y' in rates and '10Y' in rates:
            spreads['30Y-10Y'] = rates['30Y'] - rates['10Y']

        divergence_analysis = {}
        if '2Y' in rate_changes and '10Y' in rate_changes:
            short_changes = rate_changes['2Y']
            long_changes = rate_changes['10Y']
            
            if '3m' in short_changes and '3m' in long_changes:
                short_3m = short_changes['3m']
                long_3m = long_changes['3m']
                divergence_3m = long_3m - short_3m
                divergence_analysis['3m'] = {
                    'short': short_3m,
                    'long': long_3m,
                    'divergence': divergence_3m
                }
            
            if '1y' in short_changes and '1y' in long_changes:
                short_1y = short_changes['1y']
                long_1y = long_changes['1y']
                divergence_1y = long_1y - short_1y
                divergence_analysis['1y'] = {
                    'short': short_1y,
                    'long': long_1y,
                    'divergence': divergence_1y
                }

        spread_10_2 = spreads.get('10Y-2Y', 1.0)
        inverted_threshold = MACRO_SITUATION_THRESHOLDS['yield_curve']['inverted']
        flat_threshold = MACRO_SITUATION_THRESHOLDS['yield_curve']['flat']
        steep_threshold = MACRO_SITUATION_THRESHOLDS['yield_curve']['steep']
        
        if spread_10_2 < inverted_threshold:
            interpretation = "INVERTED — Recession signal"
            risk_level = "High"
        elif spread_10_2 < flat_threshold:
            interpretation = "FLAT — Possible slowdown"
            risk_level = "Moderate"
        elif spread_10_2 > steep_threshold:
            interpretation = "STEEP — Economic expansion"
            risk_level = "Low"
        else:
            interpretation = "NORMAL — Stable growth"
            risk_level = "Low"
        
        return YieldCurveAnalysis(
            levels=rates,
            spreads=spreads,
            rate_changes=rate_changes,
            divergence_analysis=divergence_analysis,
            interpretation=interpretation,
            risk_level=risk_level
        )
    
    def analyze_inflation_signals(
        self,
        factors_data: Dict[str, pd.Series]
    ) -> InflationSignals:

        commodities = {
            'GOLD': 'Gold',
            'SILVER': 'Silver',
            'OIL': 'Oil',
            'COPPER': 'Copper',
            'WHEAT': 'Wheat',
            'CORN': 'Corn'
        }
        
        commodity_changes = {}
        commodity_names = {}
        changes = []
        
        for factor, name in commodities.items():
            if factor in factors_data and len(factors_data[factor]) >= PERIOD_YEAR:
                series = factors_data[factor]
                current = series.iloc[-1]
                year_ago = series.iloc[-PERIOD_YEAR]
                
                if year_ago > 0:
                    change_1y = (current / year_ago - 1) * 100
                    commodity_changes[factor] = change_1y
                    commodity_names[factor] = name
                    changes.append(change_1y)

        if changes:
            avg_change = np.mean(changes)
            pressure = self._classify(avg_change, 'inflation', self._INFLATION_LEVELS, self._INFLATION_DEFAULT)
        else:
            pressure = "N/A"
            avg_change = np.nan
        
        return InflationSignals(
            commodity_changes=commodity_changes,
            commodity_names=commodity_names,
            inflation_pressure=pressure,
            avg_commodity_change=avg_change
        )
    
    def analyze_credit_conditions(
        self,
        factors_data: Dict[str, pd.Series]
    ) -> CreditConditions:

        vix_level = None
        market_condition = None
        hyg_level = None
        lqd_level = None

        if 'VIX' in factors_data and len(factors_data['VIX']) > 0:
            vix_level = factors_data['VIX'].iloc[-1]
            market_condition = self._classify(vix_level, 'vix', self._VIX_MARKET_LEVELS, self._VIX_MARKET_DEFAULT)

        if 'HYG' in factors_data and 'LQD' in factors_data:
            if len(factors_data['HYG']) > 0 and len(factors_data['LQD']) > 0:
                hyg_level = factors_data['HYG'].iloc[-1]
                lqd_level = factors_data['LQD'].iloc[-1]
        
        return CreditConditions(
            vix_level=vix_level,
            market_condition=market_condition,
            hyg_level=hyg_level,
            lqd_level=lqd_level
        )
    
    def analyze_risk_sentiment(
        self,
        factors_data: Dict[str, pd.Series]
    ) -> RiskSentiment:

        vix = None
        fear_level = None
        
        if 'VIX' in factors_data and len(factors_data['VIX']) > 0:
            vix = factors_data['VIX'].iloc[-1]
            fear_level = self._classify(vix, 'vix', self._VIX_FEAR_LEVELS, self._VIX_FEAR_DEFAULT)

        dxy_trend_3m = dxy_trend_1m = dxy_trend_1w = None
        if 'DXY' in factors_data and len(factors_data['DXY']) > 0:
            dxy = factors_data['DXY']
            dxy_trend_3m = self._calculate_trend(dxy, PERIOD_QUARTER)
            dxy_trend_1m = self._calculate_trend(dxy, PERIOD_MONTH)
            dxy_trend_1w = self._calculate_trend(dxy, PERIOD_WEEK)

        gold_trend_3m = gold_trend_1m = gold_trend_1w = None
        if 'GOLD' in factors_data and len(factors_data['GOLD']) > 0:
            gold = factors_data['GOLD']
            gold_trend_3m = self._calculate_trend(gold, PERIOD_QUARTER)
            gold_trend_1m = self._calculate_trend(gold, PERIOD_MONTH)
            gold_trend_1w = self._calculate_trend(gold, PERIOD_WEEK)

        dollar_strength = None
        strong_move = MACRO_SITUATION_THRESHOLDS['trends']['strong_move']
        moderate_move = MACRO_SITUATION_THRESHOLDS['trends']['moderate_move']
        divergence_threshold = MACRO_SITUATION_THRESHOLDS['trends']['divergence_threshold']
        momentum_ratio = MACRO_SITUATION_THRESHOLDS['trends']['momentum_ratio']
        
        if dxy_trend_3m is not None:
            if dxy_trend_3m > strong_move:
                if gold_trend_3m is not None and gold_trend_3m > strong_move:
                    dollar_strength = "STRONG (flight to safety)"
                elif gold_trend_3m is not None and gold_trend_3m < -moderate_move:
                    dollar_strength = "STRONG (economic/monetary policy strength)"
                else:
                    dollar_strength = "STRONG"
            elif dxy_trend_3m > 0:
                dollar_strength = "MODERATE"
            elif dxy_trend_3m > -strong_move:
                dollar_strength = "WEAK"
            else:
                dollar_strength = "VERY WEAK"

            if dxy_trend_1m is not None and dxy_trend_1w is not None:
                if dxy_trend_3m > moderate_move and dxy_trend_1w < 0:
                    dollar_strength += " (weakening recently)"
                elif dxy_trend_3m > moderate_move and dxy_trend_1w < divergence_threshold:
                    dollar_strength += " (decelerating)"
                elif dxy_trend_3m < -moderate_move and dxy_trend_1w > 0:
                    dollar_strength += " (strengthening recently)"
                elif dxy_trend_3m < -moderate_move and dxy_trend_1w > -divergence_threshold:
                    dollar_strength += " (decline decelerating)"
                elif dxy_trend_3m > moderate_move and dxy_trend_1m < dxy_trend_3m * momentum_ratio:
                    dollar_strength += " (losing momentum)"
            elif dxy_trend_1m is not None:
                if dxy_trend_1m < -moderate_move and dxy_trend_3m > 0:
                    dollar_strength += " (weakening recently)"
                elif dxy_trend_1m > moderate_move and dxy_trend_3m < 0:
                    dollar_strength += " (strengthening recently)"

        safe_haven = None
        significant_gold = MACRO_SITUATION_THRESHOLDS['trends']['significant_gold']
        strong_move = MACRO_SITUATION_THRESHOLDS['trends']['strong_move']

        gold_trend_1y = (
            self._calculate_trend(factors_data['GOLD'], PERIOD_YEAR)
            if 'GOLD' in factors_data else None
        )
        
        if gold_trend_3m is not None:
            if gold_trend_3m > significant_gold:
                safe_haven = "HIGH safe-haven demand"
            elif gold_trend_3m > strong_move:
                safe_haven = "MODERATE safe-haven demand"
            elif gold_trend_3m > 0:
                safe_haven = "LOW safe-haven demand"
            elif gold_trend_1y is not None and gold_trend_1y > significant_gold:
                safe_haven = "COOLING from high levels"
            elif gold_trend_1y is not None and gold_trend_1y > strong_move:
                safe_haven = "CORRECTING after rally"
            else:
                safe_haven = "NO safe-haven demand"
        
        return RiskSentiment(
            vix=vix,
            fear_level=fear_level,
            dollar_strength=dollar_strength,
            dxy_trend=dxy_trend_3m,
            dxy_trend_3m=dxy_trend_3m,
            dxy_trend_1m=dxy_trend_1m,
            dxy_trend_1w=dxy_trend_1w,
            gold_trend=gold_trend_3m,
            gold_trend_3m=gold_trend_3m,
            gold_trend_1m=gold_trend_1m,
            gold_trend_1w=gold_trend_1w,
            safe_haven=safe_haven
        )
    
    def analyze_global_bonds(
        self,
        factors_data: Dict[str, pd.Series]
    ) -> Dict[str, Dict]:

        bonds = {}
        regions = {
            'RATE_3M': {'region': 'USA', 'tenor': '3M', 'unit': 'yield'},
            'RATE_2Y': {'region': 'USA', 'tenor': '2Y', 'unit': 'yield'},
            'RATE_5Y': {'region': 'USA', 'tenor': '5Y', 'unit': 'yield'},
            'RATE_10Y': {'region': 'USA', 'tenor': '10Y', 'unit': 'yield'},
            'RATE_30Y': {'region': 'USA', 'tenor': '30Y', 'unit': 'yield'},
            'JPN_BOND': {'region': 'Japan', 'tenor': '10Y', 'unit': 'price'},
            'EUR_BOND': {'region': 'Europe', 'tenor': '10Y', 'unit': 'price'},
            'GER_BOND': {'region': 'Germany', 'tenor': '10Y', 'unit': 'price'},
            'UK_BOND': {'region': 'UK', 'tenor': '10Y', 'unit': 'price'},
            'EM_BOND': {'region': 'Emerging Markets', 'tenor': '10Y', 'unit': 'price'},
            'CHINA_BOND': {'region': 'China', 'tenor': '10Y', 'unit': 'price'},
            'CAN_BOND': {'region': 'Canada', 'tenor': '10Y', 'unit': 'price'},
            'AUS_BOND': {'region': 'Australia', 'tenor': '10Y', 'unit': 'price'},
            'INTL_BOND': {'region': 'International', 'tenor': '10Y', 'unit': 'price'}
        }
        
        logger.debug(f"[analyze_global_bonds] Available factors: {list(factors_data.keys())}")
        logger.debug(f"[analyze_global_bonds] Requested factors: {list(regions.keys())}")
        
        for factor, bond_info in regions.items():
            if factor not in factors_data:
                logger.debug(f"[analyze_global_bonds] Factor {factor} not found in factors_data")
                continue

            series = factors_data[factor]
            if len(series) == 0:
                logger.debug(f"[analyze_global_bonds] Factor {factor} found but empty")
                continue

            region_name = f"{bond_info['region']} {bond_info['tenor']}"
            bonds[region_name] = {
                'level': series.iloc[-1],
                'unit': bond_info['unit'],
                'change_1y': self._pct_change(series, PERIOD_YEAR),
                'change_1m': self._pct_change(series, PERIOD_MONTH),
            }
            logger.debug(f"[analyze_global_bonds] Bond added: {region_name} (factor: {factor})")
        
        logger.debug(f"[analyze_global_bonds] Final bonds: {list(bonds.keys())}")
        return bonds
    
    def get_current_snapshot(
        self,
        factors_data: Dict[str, pd.Series]
    ) -> Dict[str, Dict]:

        snapshot = {}
        
        for name, series in factors_data.items():
            if len(series) > 0:
                snapshot[name] = {
                    'current': series.iloc[-1],
                    'date': series.index[-1]
                }
        
        return snapshot