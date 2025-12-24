import numpy as np
import pandas as pd
from typing import Dict
from ..components.sharpe import SharpeCalculator
from ..components.sortino import SortinoCalculator
from ..components.helpers import (
    calculate_portfolio_returns,
    annualize_return,
    annualize_volatility
)
from ....tools.config import ANNUAL_FACTOR, ROLLING_WINDOW

class RatioAnalyzer:
    """
    Analyzer para calcular ratios de rendimiento ajustados por riesgo.
    
    Responsabilidad: Coordinar cálculos de Sharpe, Sortino y métricas relacionadas.
    """

    def __init__(self, annual_factor: float = None):
        """
        Args:
            annual_factor: Factor de anualización. Por defecto usa config.ANNUAL_FACTOR
        """
        self.annual_factor = annual_factor or ANNUAL_FACTOR
        self.sharpe_calc = SharpeCalculator(self.annual_factor)
        self.sortino_calc = SortinoCalculator(self.annual_factor)
    
    def calculate_all_ratios(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        risk_free_rate: float,
        ddof: int = 0
    ) -> Dict[str, float]:
        """
        Calcula todos los ratios de rendimiento/riesgo.
        
        Args:
            returns: DataFrame de retornos diarios
            weights: Array de pesos del portafolio
            risk_free_rate: Tasa libre de riesgo anualizada
            ddof: Grados de libertad para cálculo de desviación estándar
            
        Returns:
            Dict con Sharpe, Sortino, retornos y volatilidades
        """
        portfolio_ret = calculate_portfolio_returns(returns, weights)
        annual_return = annualize_return(portfolio_ret, self.annual_factor)
        annual_vol = annualize_volatility(portfolio_ret, self.annual_factor, ddof)
        excess_return = annual_return - risk_free_rate
        downside_returns = portfolio_ret[portfolio_ret < 0]
        if len(downside_returns) > 0:
            downside_vol = downside_returns.std(ddof=ddof) * np.sqrt(self.annual_factor)
        else:
            downside_vol = 0.0
        
        sharpe = self.sharpe_calc.calculate(returns, weights, risk_free_rate, ddof)
        sortino = self.sortino_calc.calculate(returns, weights, risk_free_rate, ddof=ddof)
        
        return {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'annual_return': float(annual_return),
            'annual_volatility': float(annual_vol),
            'downside_volatility': float(downside_vol),
            'excess_return': float(excess_return),
            'risk_free_rate': risk_free_rate
        }
    
    def calculate_rolling(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        risk_free_rate: float,
        window: int = None,
        ddof: int = 0
    ) -> pd.DataFrame:
        """
        Calcula ratios móviles (rolling).
        
        Args:
            returns: DataFrame de retornos diarios
            weights: Array de pesos del portafolio
            risk_free_rate: Tasa libre de riesgo anualizada
            window: Ventana para cálculo rolling. Por defecto usa config.ROLLING_WINDOW
            ddof: Grados de libertad para cálculo de desviación estándar
            
        Returns:
            DataFrame con Sharpe y Sortino rolling
        """
        if window is None:
            window = ROLLING_WINDOW
        
        sharpe_rolling = self.sharpe_calc.calculate_rolling(
            returns, weights, risk_free_rate, window, ddof
        )
        
        sortino_rolling = self.sortino_calc.calculate_rolling(
            returns, weights, risk_free_rate, window, ddof=ddof
        )
        
        return pd.DataFrame({
            'sharpe_rolling': sharpe_rolling,
            'sortino_rolling': sortino_rolling
        })