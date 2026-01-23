import numpy as np
import pandas as pd
from typing import Dict
from dataclasses import dataclass
from ..tools.config import (
    PERIOD_WEEK,
    PERIOD_MONTH,
    PERIOD_QUARTER,
    PERIOD_YEAR,
    MACRO_SITUATION_THRESHOLDS,
)

@dataclass
class YieldCurveAnalysis:
    """Análisis de la curva de rendimientos."""
    levels: Dict[str, float]
    spreads: Dict[str, float]
    rate_changes: Dict[str, Dict[str, float]]
    divergence_analysis: Dict[str, Dict]
    interpretation: str
    risk_level: str


@dataclass
class InflationSignals:
    """Señales de inflación desde commodities."""
    commodity_changes: Dict[str, float]
    commodity_names: Dict[str, str]
    inflation_pressure: str
    avg_commodity_change: float


@dataclass
class CreditConditions:
    """Condiciones de crédito y estrés de mercado."""
    vix_level: float
    market_condition: str
    hyg_level: float = None
    lqd_level: float = None


@dataclass
class RiskSentiment:
    """Sentimiento de riesgo global."""
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
    """
    Analiza la situación macroeconómica actual.
    
    Responsabilidad única: Proporcionar análisis interpretativo de factores macro.
    
    Análisis disponibles:
    - Yield curve USA: Interpretación de la curva de rendimientos
    - Inflation signals: Presión inflacionaria desde commodities
    - Credit conditions: Condiciones de crédito y estrés
    - Risk sentiment: Sentimiento de riesgo global
    - Global bonds: Análisis de bonos globales
    - Current snapshot: Niveles actuales de todos los factores
    """
    
    def __init__(self):
        """Inicializa el analizador de situación macro."""
        pass
    
    def analyze_yield_curve_usa(
        self,
        factors_data: Dict[str, pd.Series]
    ) -> YieldCurveAnalysis:
        """
        Analiza la curva de rendimientos de USA.
        
        Args:
            factors_data: Dict con series de factores macro
            
        Returns:
            YieldCurveAnalysis con niveles, spreads e interpretación
            
        Análisis:
        - Niveles actuales de tasas (2Y, 5Y, 10Y, 30Y)
        - Cambios en 1m, 3m, 1y
        - Spreads clave: 10Y-2Y, 10Y-5Y, 30Y-10Y
        - Divergencia entre corto y largo plazo
        
        Interpretación del spread 10Y-2Y:
        - < 0: INVERTIDA - Señal de recesión
        - 0-0.3: PLANA - Posible desaceleración
        - 0.3-2.0: NORMAL - Crecimiento estable
        - > 2.0: EMPINADA - Expansión económica
        """
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

        # Análisis de divergencia corto vs largo plazo
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
            interpretation = "INVERTIDA - Señal de recesión"
            risk_level = "Alto"
        elif spread_10_2 < flat_threshold:
            interpretation = "PLANA - Posible desaceleración"
            risk_level = "Moderado"
        elif spread_10_2 > steep_threshold:
            interpretation = "EMPINADA - Expansión económica"
            risk_level = "Bajo"
        else:
            interpretation = "NORMAL - Crecimiento estable"
            risk_level = "Bajo"
        
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
        """
        Analiza señales de inflación desde commodities.
        
        Args:
            factors_data: Dict con series de factores macro
            
        Returns:
            InflationSignals con cambios en commodities y presión inflacionaria
            
        Commodities monitoreados:
        - GOLD: Oro (safe haven + inflación)
        - SILVER: Plata (industrial + inflación)
        - OIL: Petróleo (energía)
        - COPPER: Cobre (actividad económica)
        - WHEAT: Trigo (alimentos)
        - CORN: Maíz (alimentos)
        
        Interpretación (cambio promedio 1Y):
        - > 15%: ALTA presión inflacionaria
        - 5-15%: MODERADA inflación controlada
        - -5 a 5%: BAJA inflación contenida
        - < -5%: DEFLACIÓN
        """
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
            high_threshold = MACRO_SITUATION_THRESHOLDS['inflation']['high']
            moderate_threshold = MACRO_SITUATION_THRESHOLDS['inflation']['moderate']
            low_threshold = MACRO_SITUATION_THRESHOLDS['inflation']['low']
            
            if avg_change > high_threshold:
                pressure = "HIGH - Strong inflationary pressure"
            elif avg_change > moderate_threshold:
                pressure = "MODERATE - Controlled inflation"
            elif avg_change > low_threshold:
                pressure = "LOW - Contained inflation"
            else:
                pressure = "DEFLATION - Falling prices"
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
        """
        Analiza condiciones de crédito y estrés de mercado.
        
        Args:
            factors_data: Dict con series de factores macro
            
        Returns:
            CreditConditions con VIX y niveles de bonos corporativos
            
        Indicadores:
        - VIX: Volatilidad implícita (índice del miedo)
        - HYG: High Yield Corporate Bonds
        - LQD: Investment Grade Corporate Bonds
        
        Interpretación VIX:
        - > 35: PÁNICO - Estrés extremo
        - 25-35: ESTRÉS - Tensión alta
        - 20-25: TENSIÓN - Nerviosismo
        - 15-20: NORMAL - Volatilidad controlada
        - < 15: COMPLACENCIA - Volatilidad muy baja
        """
        vix_level = None
        market_condition = None
        hyg_level = None
        lqd_level = None

        if 'VIX' in factors_data and len(factors_data['VIX']) > 0:
            vix = factors_data['VIX'].iloc[-1]
            vix_level = vix
            
            vix_panic = MACRO_SITUATION_THRESHOLDS['vix']['panic']
            vix_stress = MACRO_SITUATION_THRESHOLDS['vix']['stress']
            vix_tension = MACRO_SITUATION_THRESHOLDS['vix']['tension']
            vix_normal = MACRO_SITUATION_THRESHOLDS['vix']['normal']
            
            if vix > vix_panic:
                market_condition = "PÁNICO - Estrés extremo"
            elif vix > vix_stress:
                market_condition = "ESTRÉS - Tensión alta"
            elif vix > vix_tension:
                market_condition = "TENSIÓN - Nerviosismo"
            elif vix > vix_normal:
                market_condition = "NORMAL - Volatilidad controlada"
            else:
                market_condition = "COMPLACENCIA - Volatilidad muy baja"

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
        """
        Analiza sentimiento de riesgo global.
        
        Args:
            factors_data: Dict con series de factores macro
            
        Returns:
            RiskSentiment con VIX, dólar, oro y análisis integrado
            
        Indicadores clave:
        - VIX: Miedo en mercados
        - DXY (Dollar Index): Fortaleza del dólar
        - GOLD: Demanda de refugio seguro
        
        Análisis integrado:
        - DXY ↑ + GOLD ↑ = Flight to safety (pánico)
        - DXY ↑ + GOLD ↓ = Fortaleza económica/hawkish Fed
        - DXY ↓ + GOLD ↑ = Búsqueda de rendimiento
        - DXY ↓ + GOLD ↓ = Risk-on generalizado
        """
        vix = None
        fear_level = None
        
        if 'VIX' in factors_data and len(factors_data['VIX']) > 0:
            vix = factors_data['VIX'].iloc[-1]
            
            vix_panic = MACRO_SITUATION_THRESHOLDS['vix']['panic']
            vix_stress = MACRO_SITUATION_THRESHOLDS['vix']['stress']
            vix_tension = MACRO_SITUATION_THRESHOLDS['vix']['tension']
            vix_normal = MACRO_SITUATION_THRESHOLDS['vix']['normal']
            
            if vix > vix_panic:
                fear_level = "PÁNICO"
            elif vix > vix_stress:
                fear_level = "ALTO MIEDO"
            elif vix > vix_tension:
                fear_level = "NERVIOSISMO"
            elif vix > vix_normal:
                fear_level = "MODERADO"
            else:
                fear_level = "COMPLACENCIA"

        dxy_trend_3m = None
        dxy_trend_1m = None
        dxy_trend_1w = None
        
        if 'DXY' in factors_data and len(factors_data['DXY']) > 0:
            dxy = factors_data['DXY']
            current = dxy.iloc[-1]

            if len(dxy) >= PERIOD_QUARTER and dxy.iloc[-PERIOD_QUARTER] > 0:
                dxy_trend_3m = (current / dxy.iloc[-PERIOD_QUARTER] - 1) * 100
            
            if len(dxy) >= PERIOD_MONTH and dxy.iloc[-PERIOD_MONTH] > 0:
                dxy_trend_1m = (current / dxy.iloc[-PERIOD_MONTH] - 1) * 100

            if len(dxy) >= PERIOD_WEEK and dxy.iloc[-PERIOD_WEEK] > 0:
                dxy_trend_1w = (current / dxy.iloc[-PERIOD_WEEK] - 1) * 100

        gold_trend_3m = None
        gold_trend_1m = None
        gold_trend_1w = None
        
        if 'GOLD' in factors_data and len(factors_data['GOLD']) > 0:
            gold = factors_data['GOLD']
            current = gold.iloc[-1]

            if len(gold) >= PERIOD_QUARTER and gold.iloc[-PERIOD_QUARTER] > 0:
                gold_trend_3m = (current / gold.iloc[-PERIOD_QUARTER] - 1) * 100

            if len(gold) >= PERIOD_MONTH and gold.iloc[-PERIOD_MONTH] > 0:
                gold_trend_1m = (current / gold.iloc[-PERIOD_MONTH] - 1) * 100
            
            if len(gold) >= PERIOD_WEEK and gold.iloc[-PERIOD_WEEK] > 0:
                gold_trend_1w = (current / gold.iloc[-PERIOD_WEEK] - 1) * 100

        # Análisis integrado de fortaleza del dólar
        dollar_strength = None
        strong_move = MACRO_SITUATION_THRESHOLDS['trends']['strong_move']
        moderate_move = MACRO_SITUATION_THRESHOLDS['trends']['moderate_move']
        divergence_threshold = MACRO_SITUATION_THRESHOLDS['trends']['divergence_threshold']
        momentum_ratio = MACRO_SITUATION_THRESHOLDS['trends']['momentum_ratio']
        
        if dxy_trend_3m is not None:
            if dxy_trend_3m > strong_move:
                if gold_trend_3m is not None and gold_trend_3m > strong_move:
                    dollar_strength = "FUERTE (flight to safety)"
                elif gold_trend_3m is not None and gold_trend_3m < -moderate_move:
                    dollar_strength = "FUERTE (fortaleza económica/política monetaria)"
                else:
                    dollar_strength = "FUERTE"
            elif dxy_trend_3m > 0:
                dollar_strength = "MODERADO"
            elif dxy_trend_3m > -strong_move:
                dollar_strength = "DÉBIL"
            else:
                dollar_strength = "MUY DÉBIL"

            # Añadir tendencias recientes
            if dxy_trend_1m is not None and dxy_trend_1w is not None:
                if dxy_trend_3m > moderate_move and dxy_trend_1w < divergence_threshold:
                    dollar_strength += " (debilitándose recientemente)"
                elif dxy_trend_3m < -moderate_move and dxy_trend_1w > divergence_threshold:
                    dollar_strength += " (fortaleciéndose recientemente)"
                elif dxy_trend_3m > moderate_move and dxy_trend_1m < dxy_trend_3m * momentum_ratio:
                    dollar_strength += " (desacelerando)"
            elif dxy_trend_1m is not None:
                if dxy_trend_1m < -moderate_move and dxy_trend_3m > 0:
                    dollar_strength += " (debilitándose recientemente)"
                elif dxy_trend_1m > moderate_move and dxy_trend_3m < 0:
                    dollar_strength += " (fortaleciéndose recientemente)"

        # Safe haven demand
        safe_haven = None
        significant_gold = MACRO_SITUATION_THRESHOLDS['trends']['significant_gold']
        strong_move = MACRO_SITUATION_THRESHOLDS['trends']['strong_move']
        
        if gold_trend_3m is not None:
            if gold_trend_3m > significant_gold:
                safe_haven = "ALTA demanda de refugio"
            elif gold_trend_3m > strong_move:
                safe_haven = "MODERADA demanda de refugio"
            elif gold_trend_3m > 0:
                safe_haven = "BAJA demanda de refugio"
            else:
                safe_haven = "SIN demanda de refugio"
        
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
        """
        Analiza bonos gubernamentales globales.
        
        Args:
            factors_data: Dict con series de factores macro
            
        Returns:
            Dict con análisis por región
            
        Regiones monitoreadas:
        - USA: GOVT_20Y
        - Japan: JPN_BOND
        - Europe: EUR_BOND
        - Germany: GER_BOND
        - UK: UK_BOND
        - Emerging Markets: EM_BOND
        - China: CHINA_BOND
        - Canada: CAN_BOND
        - Australia: AUS_BOND
        - International: INTL_BOND
        
        Para cada región:
        - Nivel actual
        - Cambio 1m (21 trading days), 1y (252 trading days)
        """
        bonds = {}
        # Mapeo de factores a regiones con vencimientos
        regions = {
            'GOVT_20Y': {'region': 'USA', 'tenor': '20Y'},
            'GOVT_7_10Y': {'region': 'USA', 'tenor': '7-10Y'},
            'GOVT_1_3Y': {'region': 'USA', 'tenor': '1-3Y'},
            'RATE_3M': {'region': 'USA', 'tenor': '3M'},
            'RATE_2Y': {'region': 'USA', 'tenor': '2Y'},
            'RATE_5Y': {'region': 'USA', 'tenor': '5Y'},
            'RATE_10Y': {'region': 'USA', 'tenor': '10Y'},
            'RATE_30Y': {'region': 'USA', 'tenor': '30Y'},
            'JPN_BOND': {'region': 'Japan', 'tenor': '10Y'},
            'EUR_BOND': {'region': 'Europe', 'tenor': '10Y'},
            'GER_BOND': {'region': 'Germany', 'tenor': '10Y'},
            'UK_BOND': {'region': 'UK', 'tenor': '10Y'},
            'EM_BOND': {'region': 'Emerging Markets', 'tenor': '10Y'},
            'CHINA_BOND': {'region': 'China', 'tenor': '10Y'},
            'CAN_BOND': {'region': 'Canada', 'tenor': '10Y'},
            'AUS_BOND': {'region': 'Australia', 'tenor': '10Y'},
            'INTL_BOND': {'region': 'International', 'tenor': '10Y'}
        }
        
        print(f"[analyze_global_bonds Debug] Factores disponibles en factors_data: {list(factors_data.keys())}")
        print(f"[analyze_global_bonds Debug] Factores buscados: {list(regions.keys())}")
        
        for factor, bond_info in regions.items():
            if factor in factors_data:
                series = factors_data[factor]
                if len(series) > 0:
                    current = series.iloc[-1]
                    # Crear nombre con región y vencimiento
                    region_name = f"{bond_info['region']} {bond_info['tenor']}"
                    bond_data = {'level': current}

                    if len(series) >= PERIOD_YEAR:
                        year_ago = series.iloc[-PERIOD_YEAR]
                        if year_ago > 0:
                            change_1y = (current / year_ago - 1) * 100
                            bond_data['change_1y'] = change_1y
                        else:
                            bond_data['change_1y'] = np.nan
                    else:
                        bond_data['change_1y'] = np.nan

                    if len(series) >= PERIOD_MONTH:
                        month_ago = series.iloc[-PERIOD_MONTH]
                        if month_ago > 0:
                            change_1m = (current / month_ago - 1) * 100
                            bond_data['change_1m'] = change_1m
                        else:
                            bond_data['change_1m'] = np.nan
                    else:
                        bond_data['change_1m'] = np.nan
                    
                    bonds[region_name] = bond_data
                    print(f"[analyze_global_bonds Debug] Bono agregado: {region_name} (factor: {factor})")
                else:
                    print(f"[analyze_global_bonds Debug] Factor {factor} encontrado pero sin datos (len={len(series)})")
            else:
                print(f"[analyze_global_bonds Debug] Factor {factor} NO encontrado en factors_data")
        
        print(f"[analyze_global_bonds Debug] Bonos finales: {list(bonds.keys())}")
        return bonds
    
    def get_current_snapshot(
        self,
        factors_data: Dict[str, pd.Series]
    ) -> Dict[str, Dict]:
        """
        Obtiene snapshot actual de todos los factores.
        
        Args:
            factors_data: Dict con series de factores macro
            
        Returns:
            Dict con {factor: {current, date}}
            
        Uso:
        - Vista rápida de niveles actuales
        - Verificación de datos disponibles
        """
        snapshot = {}
        
        for name, series in factors_data.items():
            if len(series) > 0:
                snapshot[name] = {
                    'current': series.iloc[-1],
                    'date': series.index[-1]
                }
        
        return snapshot