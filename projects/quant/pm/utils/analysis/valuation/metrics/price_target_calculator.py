import numpy as np
import pandas as pd
from typing import Dict
from ....tools.config import PRICE_TARGET_CONFIG

class PriceTargetCalculator:
    """
    Calcula precio objetivo usando múltiples metodologías.
    
    Responsabilidad: Estimar precio justo/objetivo basado en fundamentales.
    
    Metodologías (por prioridad):
    1. Consenso de analistas (más confiable si disponible)
    2. PEG (P/E to Growth) - Peter Lynch method
    3. P/E ajustado por crecimiento
    4. Score-based (último recurso)
    
    """

    def __init__(self, config: dict = None):
        """
        Args:
            config: Configuración de price target. Por defecto usa PRICE_TARGET_CONFIG
        """
        self.config = config or PRICE_TARGET_CONFIG

    def calculate_from_peg(
        self,
        current_price: float,
        pe: float,
        peg: float
    ) -> float:
        """
        Calcula price target usando método PEG (P/E to Growth).
        
        Metodología Peter Lynch:
        - PEG = P/E / Growth Rate
        - Fair PEG = 1.0 (fair value)
        - PEG < 1.0 → Infravalorado
        - PEG > 1.0 → Sobrevalorado
        
        Fórmula:
        1. EPS = Precio / P/E
        2. Growth implícito = P/E / PEG
        3. Fair P/E = Fair PEG × Growth implícito
        4. Price Target = EPS × Fair P/E
        
        Args:
            current_price: Precio actual
            pe: P/E trailing
            peg: PEG ratio
            
        Returns:
            Precio objetivo estimado
            
        Ejemplo:
        - Precio actual = $100
        - P/E = 20x
        - PEG = 2.0 (sobrevalorado)
        - EPS = $100 / 20 = $5
        - Growth implícito = 20 / 2.0 = 10%
        - Fair P/E = 1.0 × 10 = 10x
        - Target = $5 × 10 = $50 (downside 50%)
        """
        if pd.isna(pe) or pe <= 0 or pd.isna(peg) or peg <= 0:
            return np.nan
        
        cfg = self.config['peg_method']
        
        # EPS actual
        eps = current_price / pe
        
        # Growth implícito del mercado (en base 1, no %)
        implied_growth = pe / peg if peg > 0 else 0
        
        # Fair P/E: usando fair_peg = 1.0 (estándar Lynch)
        fair_peg = cfg['fair_peg']
        fair_pe = fair_peg * implied_growth
        
        # Price target objetivo
        price_target = eps * fair_pe
        
        # Validación: Limitar cambios extremos (protección contra errores de PEG)
        # Permitir máximo ±75% de cambio desde precio actual
        max_upside = current_price * 1.75
        max_downside = current_price * 0.25
        
        if price_target > max_upside:
            print(f"⚠️ Price target PEG ({price_target:.2f}) limitado a +75%: {max_upside:.2f}")
            price_target = max_upside
        elif price_target < max_downside:
            print(f"⚠️ Price target PEG ({price_target:.2f}) limitado a -75%: {max_downside:.2f}")
            price_target = max_downside
        
        return price_target

    def calculate_from_pe(
        self, 
        current_price: float, 
        pe: float, 
        earnings_growth: float = None
    ) -> float:
        """
        Calcula price target usando P/E ajustado por crecimiento.
        
        Lógica:
        - Con growth: Fair P/E se infiere del crecimiento esperado
        - Sin growth: Fair P/E usa múltiplo conservador del P/E actual
        
        Benchmark:
        - Growth 15% → Fair P/E ~15-20x
        - Growth 10% → Fair P/E ~10-15x
        - Growth 5% → Fair P/E ~8-12x
        
        Args:
            current_price: Precio actual
            pe: P/E trailing
            earnings_growth: Tasa de crecimiento earnings (0.15 = 15%)
            
        Returns:
            Precio objetivo estimado
        """
        if pd.isna(pe) or pe <= 0:
            return np.nan
        
        cfg = self.config['pe_method']
        
        # EPS actual
        eps = current_price / pe

        # Calcular Fair P/E basado en fundamentales
        if earnings_growth and earnings_growth > 0:
            # Normalizar growth si viene como porcentaje (>1)
            if earnings_growth >= cfg['earnings_growth_threshold']:
                earnings_growth = earnings_growth / 100
            
            # Fair P/E implícito por crecimiento
            # Regla general: P/E fair ≈ Growth rate × 100
            fair_pe_from_growth = earnings_growth * 100 * cfg['growth_multiplier']
            
            # Combinar con P/E actual (weighted average)
            fair_pe = (
                pe * cfg['pe_weight'] + 
                fair_pe_from_growth * cfg['growth_weight']
            )
        else:
            # Sin growth data: usar múltiplo conservador
            fair_pe = pe * cfg['fair_multiplier_base']

        price_target = eps * fair_pe
        
        # Validación: Limitar cambios extremos
        max_upside = current_price * 1.75
        max_downside = current_price * 0.25
        
        if price_target > max_upside:
            print(f"⚠️ Price target P/E ({price_target:.2f}) limitado a +75%: {max_upside:.2f}")
            price_target = max_upside
        elif price_target < max_downside:
            print(f"⚠️ Price target P/E ({price_target:.2f}) limitado a -75%: {max_downside:.2f}")
            price_target = max_downside
        
        return price_target
    
    def calculate_from_analyst_target(
        self,
        current_price: float,
        target_price: float
    ) -> float:
        """
        Usa precio objetivo de consenso de analistas.
        
        Filosofía: Los analistas profesionales ya hicieron el trabajo.
        No ajustamos su consenso - es la estimación más objetiva disponible.
        
        Args:
            current_price: Precio actual
            target_price: Precio objetivo consenso analistas
            
        Returns:
            Precio objetivo (sin modificar)
        """
        if pd.isna(target_price) or target_price <= 0:
            return np.nan

        # Retornar target de analistas sin ajustes
        return float(target_price)
    
    def calculate_from_score(
        self, 
        current_price: float, 
        valuation_score: float
    ) -> float:
        """
        Calcula price target basado en valuation_score (último recurso).
        
        NOTA: Este método solo se usa cuando NO hay datos fundamentales.
        Es menos preciso que los otros métodos.
        
        Lógica:
        - Score > 50 → Infravalorado → Target > Precio actual
        - Score < 50 → Sobrevalorado → Target < Precio actual
        - Score = 50 → Fair value → Target = Precio actual
        
        Args:
            current_price: Precio actual
            valuation_score: Score de valoración (0-100)
            
        Returns:
            Precio objetivo estimado
        """
        cfg = self.config['score_method']
        
        # Ajuste basado en score
        if valuation_score < 50:
            # Sobrevalorado → ajuste bajista
            adjustment = (valuation_score - 50) / cfg['adjustment_divisor_bear']
        else:
            # Infravalorado → ajuste alcista
            adjustment = (valuation_score - 50) / cfg['adjustment_divisor_bull']
        
        return current_price * (1 + adjustment)
    
    def calculate(
        self, 
        data: Dict, 
        valuation_score: float, 
        current_price: float
    ) -> float:
        """
        Calcula price target usando la mejor metodología disponible.
        
        Cascada de prioridad (mejor a peor):
        1. ✅ Consenso analistas → Más confiable
        2. ✅ Método PEG → Si hay P/E y PEG
        3. ✅ Método P/E con growth → Si hay P/E
        4. ⚠️  Score-based → Último recurso
        
        Args:
            data: Diccionario con datos financieros de yfinance
            valuation_score: Score de valoración (0-100)
            current_price: Precio actual
            
        Returns:
            Precio objetivo estimado
            
        Ejemplo de flujo:
        - Si targetMeanPrice existe → usa ese
        - Si no, pero hay P/E y PEG → usa método PEG
        - Si no, pero hay P/E → usa método P/E
        - Si nada funciona → usa score (menos confiable)
        """
        # 1º Prioridad: Consenso de analistas
        target_price = data.get('targetMeanPrice', np.nan)
        if pd.notna(target_price) and target_price > 0:
            result = self.calculate_from_analyst_target(
                current_price, 
                target_price
            )
            if pd.notna(result):
                return result

        # 2º Prioridad: Método PEG
        pe = data.get('trailingPE', np.nan)
        peg = data.get('pegRatio', np.nan)
        
        if pd.notna(pe) and pd.notna(peg) and pe > 0 and peg > 0:
            result = self.calculate_from_peg(current_price, pe, peg)
            if pd.notna(result):
                return result

        # 3º Prioridad: Método P/E con growth
        earnings_growth = data.get('earningsQuarterlyGrowth')
        if pd.isna(earnings_growth):
            earnings_growth = data.get('earningsGrowth')
        
        if pd.notna(pe) and pe > 0:
            return self.calculate_from_pe(
                current_price, 
                pe, 
                earnings_growth
            )

        # 4º Último recurso: Score-based
        return self.calculate_from_score(current_price, valuation_score)