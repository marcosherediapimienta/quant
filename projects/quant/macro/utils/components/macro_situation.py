import numpy as np
import pandas as pd
from typing import Dict
from dataclasses import dataclass

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

                if len(series) >= 21:
                    month_ago = series.iloc[-21]
                    changes['1m'] = current - month_ago

                if len(series) >= 63:
                    quarter_ago = series.iloc[-63]
                    changes['3m'] = current - quarter_ago

                if len(series) >= 252:
                    year_ago = series.iloc[-252]
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
        if spread_10_2 < 0:
            interpretation = "INVERTIDA - Señal de recesión"
            risk_level = "Alto"
        elif spread_10_2 < 0.3:
            interpretation = "PLANA - Posible desaceleración"
            risk_level = "Moderado"
        elif spread_10_2 > 2.0:
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
            'GOLD': 'Oro',
            'SILVER': 'Plata',
            'OIL': 'Petróleo',
            'COPPER': 'Cobre',
            'WHEAT': 'Trigo',
            'CORN': 'Maíz'
        }
        
        commodity_changes = {}
        commodity_names = {}
        changes = []
        
        for factor, name in commodities.items():
            if factor in factors_data and len(factors_data[factor]) >= 252:
                series = factors_data[factor]
                current = series.iloc[-1]
                year_ago = series.iloc[-252]
                
                if year_ago > 0:
                    change_1y = (current / year_ago - 1) * 100
                    commodity_changes[factor] = change_1y
                    commodity_names[factor] = name
                    changes.append(change_1y)

        if changes:
            avg_change = np.mean(changes)
            
            if avg_change > 15:
                pressure = "ALTA - Presión inflacionaria fuerte"
            elif avg_change > 5:
                pressure = "MODERADA - Inflación controlada"
            elif avg_change > -5:
                pressure = "BAJA - Inflación contenida"
            else:
                pressure = "DEFLACIÓN - Caída de precios"
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
            
            if vix > 35:
                market_condition = "PÁNICO - Estrés extremo"
            elif vix > 25:
                market_condition = "ESTRÉS - Tensión alta"
            elif vix > 20:
                market_condition = "TENSIÓN - Nerviosismo"
            elif vix > 15:
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
            
            if vix > 35:
                fear_level = "PÁNICO"
            elif vix > 25:
                fear_level = "ALTO MIEDO"
            elif vix > 20:
                fear_level = "NERVIOSISMO"
            elif vix > 15:
                fear_level = "MODERADO"
            else:
                fear_level = "COMPLACENCIA"

        dxy_trend_3m = None
        dxy_trend_1m = None
        dxy_trend_1w = None
        
        if 'DXY' in factors_data and len(factors_data['DXY']) > 0:
            dxy = factors_data['DXY']
            current = dxy.iloc[-1]

            if len(dxy) >= 63 and dxy.iloc[-63] > 0:
                dxy_trend_3m = (current / dxy.iloc[-63] - 1) * 100
            
            if len(dxy) >= 21 and dxy.iloc[-21] > 0:
                dxy_trend_1m = (current / dxy.iloc[-21] - 1) * 100

            if len(dxy) >= 5 and dxy.iloc[-5] > 0:
                dxy_trend_1w = (current / dxy.iloc[-5] - 1) * 100

        gold_trend_3m = None
        gold_trend_1m = None
        gold_trend_1w = None
        
        if 'GOLD' in factors_data and len(factors_data['GOLD']) > 0:
            gold = factors_data['GOLD']
            current = gold.iloc[-1]

            if len(gold) >= 63 and gold.iloc[-63] > 0:
                gold_trend_3m = (current / gold.iloc[-63] - 1) * 100

            if len(gold) >= 21 and gold.iloc[-21] > 0:
                gold_trend_1m = (current / gold.iloc[-21] - 1) * 100
            
            if len(gold) >= 5 and gold.iloc[-5] > 0:
                gold_trend_1w = (current / gold.iloc[-5] - 1) * 100

        # Análisis integrado de fortaleza del dólar
        dollar_strength = None
        if dxy_trend_3m is not None:
            if dxy_trend_3m > 5:
                if gold_trend_3m is not None and gold_trend_3m > 5:
                    dollar_strength = "FUERTE (flight to safety)"
                elif gold_trend_3m is not None and gold_trend_3m < -2:
                    dollar_strength = "FUERTE (fortaleza económica/política monetaria)"
                else:
                    dollar_strength = "FUERTE"
            elif dxy_trend_3m > 0:
                dollar_strength = "MODERADO"
            elif dxy_trend_3m > -5:
                dollar_strength = "DÉBIL"
            else:
                dollar_strength = "MUY DÉBIL"

            # Añadir tendencias recientes
            if dxy_trend_1m is not None and dxy_trend_1w is not None:
                if dxy_trend_3m > 3 and dxy_trend_1w < 0.5:
                    dollar_strength += " (debilitándose recientemente)"
                elif dxy_trend_3m < -3 and dxy_trend_1w > 0.5:
                    dollar_strength += " (fortaleciéndose recientemente)"
                elif dxy_trend_3m > 3 and dxy_trend_1m < dxy_trend_3m * 0.4:
                    dollar_strength += " (desacelerando)"
            elif dxy_trend_1m is not None:
                if dxy_trend_1m < -2 and dxy_trend_3m > 0:
                    dollar_strength += " (debilitándose recientemente)"
                elif dxy_trend_1m > 2 and dxy_trend_3m < 0:
                    dollar_strength += " (fortaleciéndose recientemente)"

        # Safe haven demand
        safe_haven = None
        if gold_trend_3m is not None:
            if gold_trend_3m > 10:
                safe_haven = "ALTA demanda de refugio"
            elif gold_trend_3m > 5:
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
        - Japón: JPN_BOND
        - Europa: EUR_BOND
        - Alemania: GER_BOND
        - UK: UK_BOND
        - Emergentes: EM_BOND
        
        Para cada región:
        - Nivel actual
        - Cambio 1m, 1y
        """
        bonds = {}
        regions = {
            'USA': 'GOVT_20Y',
            'Japón': 'JPN_BOND',
            'Europa': 'EUR_BOND',
            'Alemania': 'GER_BOND',
            'UK': 'UK_BOND',
            'Emergentes': 'EM_BOND'
        }
        
        for region, factor in regions.items():
            if factor in factors_data and len(factors_data[factor]) > 0:
                series = factors_data[factor]
                current = series.iloc[-1]
                bond_data = {'level': current}

                if len(series) >= 252:
                    year_ago = series.iloc[-252]
                    if year_ago > 0:
                        change_1y = (current / year_ago - 1) * 100
                        bond_data['change_1y'] = change_1y
                    else:
                        bond_data['change_1y'] = np.nan
                else:
                    bond_data['change_1y'] = np.nan

                if len(series) >= 21:
                    month_ago = series.iloc[-21]
                    if month_ago > 0:
                        change_1m = (current / month_ago - 1) * 100
                        bond_data['change_1m'] = change_1m
                    else:
                        bond_data['change_1m'] = np.nan
                else:
                    bond_data['change_1m'] = np.nan
                
                bonds[region] = bond_data
        
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