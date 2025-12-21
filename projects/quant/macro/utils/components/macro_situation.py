import numpy as np
import pandas as pd
from typing import Dict

def analyze_yield_curve_usa(factors_data: Dict[str, pd.Series]) -> Dict:
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
    
    return {
        'levels': rates,
        'spreads': spreads,
        'rate_changes': rate_changes,
        'divergence_analysis': divergence_analysis,
        'interpretation': interpretation,
        'risk_level': risk_level
    }

def analyze_inflation_signals(factors_data: Dict[str, pd.Series]) -> Dict:
    signals = {}
    
    commodities = {
        'GOLD': 'Oro',
        'SILVER': 'Plata',
        'OIL': 'Petróleo',
        'COPPER': 'Cobre',
        'WHEAT': 'Trigo',
        'CORN': 'Maíz'
    }
    
    changes = []
    
    for factor, name in commodities.items():
        if factor in factors_data and len(factors_data[factor]) >= 252:
            series = factors_data[factor]
            current = series.iloc[-1]
            year_ago = series.iloc[-252]
            
            if year_ago > 0:
                change_1y = (current / year_ago - 1) * 100
                signals[f'{factor.lower()}_change_1y'] = change_1y
                signals[f'{factor.lower()}_name'] = name
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
        
        signals['inflation_pressure'] = pressure
        signals['avg_commodity_change'] = avg_change
    else:
        signals['inflation_pressure'] = "N/A"
        signals['avg_commodity_change'] = np.nan
    
    return signals


def analyze_credit_conditions(factors_data: Dict[str, pd.Series]) -> Dict:
    credit = {}

    if 'VIX' in factors_data and len(factors_data['VIX']) > 0:
        vix = factors_data['VIX'].iloc[-1]
        credit['vix_level'] = vix
        
        if vix > 35:
            condition = "PÁNICO - Estrés extremo"
        elif vix > 25:
            condition = "ESTRÉS - Tensión alta"
        elif vix > 20:
            condition = "TENSIÓN - Nerviosismo"
        elif vix > 15:
            condition = "NORMAL - Volatilidad controlada"
        else:
            condition = "COMPLACENCIA - Volatilidad muy baja"
        
        credit['market_condition'] = condition

    if 'HYG' in factors_data and 'LQD' in factors_data:
        if len(factors_data['HYG']) > 0 and len(factors_data['LQD']) > 0:
            hyg = factors_data['HYG'].iloc[-1]
            lqd = factors_data['LQD'].iloc[-1]
            credit['hyg_level'] = hyg
            credit['lqd_level'] = lqd
    
    return credit

def analyze_global_bonds(factors_data: Dict[str, pd.Series]) -> Dict:
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

def analyze_risk_sentiment(factors_data: Dict[str, pd.Series]) -> Dict:
    sentiment = {}

    if 'VIX' in factors_data and len(factors_data['VIX']) > 0:
        vix = factors_data['VIX'].iloc[-1]
        sentiment['vix'] = vix
        
        if vix > 35:
            fear = "PÁNICO"
        elif vix > 25:
            fear = "ALTO MIEDO"
        elif vix > 20:
            fear = "NERVIOSISMO"
        elif vix > 15:
            fear = "MODERADO"
        else:
            fear = "COMPLACENCIA"
        
        sentiment['fear_level'] = fear

    dxy_trend_3m = None
    dxy_trend_1m = None
    dxy_trend_1w = None
    
    if 'DXY' in factors_data and len(factors_data['DXY']) > 0:
        dxy = factors_data['DXY']
        current = dxy.iloc[-1]

        if len(dxy) >= 63 and dxy.iloc[-63] > 0:
            dxy_trend_3m = (current / dxy.iloc[-63] - 1) * 100
            sentiment['dxy_trend_3m'] = dxy_trend_3m
        
        if len(dxy) >= 21 and dxy.iloc[-21] > 0:
            dxy_trend_1m = (current / dxy.iloc[-21] - 1) * 100
            sentiment['dxy_trend_1m'] = dxy_trend_1m

        if len(dxy) >= 5 and dxy.iloc[-5] > 0:
            dxy_trend_1w = (current / dxy.iloc[-5] - 1) * 100
            sentiment['dxy_trend_1w'] = dxy_trend_1w

        sentiment['dxy_trend'] = dxy_trend_3m

    gold_trend_3m = None
    gold_trend_1m = None
    gold_trend_1w = None
    
    if 'GOLD' in factors_data and len(factors_data['GOLD']) > 0:
        gold = factors_data['GOLD']
        current = gold.iloc[-1]

        if len(gold) >= 63 and gold.iloc[-63] > 0:
            gold_trend_3m = (current / gold.iloc[-63] - 1) * 100
            sentiment['gold_trend_3m'] = gold_trend_3m

        if len(gold) >= 21 and gold.iloc[-21] > 0:
            gold_trend_1m = (current / gold.iloc[-21] - 1) * 100
            sentiment['gold_trend_1m'] = gold_trend_1m
        
        if len(gold) >= 5 and gold.iloc[-5] > 0:
            gold_trend_1w = (current / gold.iloc[-5] - 1) * 100
            sentiment['gold_trend_1w'] = gold_trend_1w

        sentiment['gold_trend'] = gold_trend_3m

    if dxy_trend_3m is not None:
        if dxy_trend_3m > 5:
            if gold_trend_3m is not None and gold_trend_3m > 5:
                strength = "FUERTE (flight to safety)"
            elif gold_trend_3m is not None and gold_trend_3m < -2:
                strength = "FUERTE (fortaleza económica/política monetaria)"
            else:
                strength = "FUERTE"
        elif dxy_trend_3m > 0:
            strength = "MODERADO"
        elif dxy_trend_3m > -5:
            strength = "DÉBIL"
        else:
            strength = "MUY DÉBIL"

        if dxy_trend_1m is not None and dxy_trend_1w is not None:
            if dxy_trend_3m > 3 and dxy_trend_1w < 0.5:
                strength += " (debilitándose recientemente)"
            elif dxy_trend_3m < -3 and dxy_trend_1w > 0.5:
                strength += " (fortaleciéndose recientemente)"
            elif dxy_trend_3m > 3 and dxy_trend_1m < dxy_trend_3m * 0.4:
                strength += " (desacelerando)"
        elif dxy_trend_1m is not None:
            if dxy_trend_1m < -2 and dxy_trend_3m > 0:
                strength += " (debilitándose recientemente)"
            elif dxy_trend_1m > 2 and dxy_trend_3m < 0:
                strength += " (fortaleciéndose recientemente)"
        
        sentiment['dollar_strength'] = strength

    if gold_trend_3m is not None:
        if gold_trend_3m > 10:
            sentiment['safe_haven'] = "ALTA demanda de refugio"
        elif gold_trend_3m > 5:
            sentiment['safe_haven'] = "MODERADA demanda de refugio"
        elif gold_trend_3m > 0:
            sentiment['safe_haven'] = "BAJA demanda de refugio"
        else:
            sentiment['safe_haven'] = "SIN demanda de refugio"
    
    return sentiment

def get_current_snapshot(factors_data: Dict[str, pd.Series]) -> Dict:
    snapshot = {}
    
    for name, series in factors_data.items():
        if len(series) > 0:
            snapshot[name] = {
                'current': series.iloc[-1],
                'date': series.index[-1]
            }
    
    return snapshot