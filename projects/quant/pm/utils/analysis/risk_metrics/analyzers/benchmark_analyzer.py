import numpy as np
import pandas as pd
from typing import Dict
from ..components.tracking_error import TrackingErrorCalculator
from ..components.beta import BetaCalculator
from ..components.alpha import AlphaCalculator
from ....tools.config import ANNUAL_FACTOR, ROLLING_WINDOW

class BenchmarkAnalyzer:
    """
    Analyzer para métricas relativas a un benchmark.
    
    Responsabilidad: Coordinar cálculos de tracking error, alpha, beta y métricas relacionadas.
    """

    def __init__(self, annual_factor: float = None):
        """
        Args:
            annual_factor: Factor de anualización. Por defecto usa config.ANNUAL_FACTOR
        """
        self.annual_factor = annual_factor or ANNUAL_FACTOR
        self.te_calc = TrackingErrorCalculator(self.annual_factor)
        self.beta_calc = BetaCalculator()
        self.alpha_calc = AlphaCalculator(self.annual_factor)
    
    def analyze(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        benchmark_returns: pd.Series,
        risk_free_rate: float,
        ddof: int = 1
    ) -> Dict[str, float]:
        """
        Analiza el portafolio contra un benchmark.
        
        Args:
            returns: DataFrame de retornos diarios de activos
            weights: Array de pesos del portafolio
            benchmark_returns: Serie de retornos del benchmark
            risk_free_rate: Tasa libre de riesgo anualizada
            ddof: Grados de libertad para cálculos estadísticos
            
        Returns:
            Dict con tracking error, information ratio, alpha, beta, R², correlación
        """
        te_results = self.te_calc.calculate(returns, weights, benchmark_returns, ddof)
        beta_results = self.beta_calc.calculate(returns, weights, benchmark_returns, ddof)
        alpha_results = self.alpha_calc.calculate(
            returns, weights, benchmark_returns, 
            risk_free_rate, beta=beta_results['beta'], ddof=ddof
        )

        return {
            'tracking_error_daily': te_results['te_daily'],
            'tracking_error_annual': te_results['te_annual'],
            'excess_return_annual': te_results['excess_return_annual'],
            'information_ratio': te_results['information_ratio'],
            'beta': beta_results['beta'],
            'r_squared': beta_results['r_squared'],
            'correlation': beta_results['correlation'],
            'alpha_annual': alpha_results['alpha_annual'],
            'portfolio_return_annual': alpha_results['portfolio_return_annual'],
            'benchmark_return_annual': alpha_results['benchmark_return_annual'],
            'expected_return': alpha_results['expected_return']
        }
    
    def analyze_rolling(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        benchmark_returns: pd.Series,
        window: int = None,
        ddof: int = 1
    ) -> pd.DataFrame:
        """
        Calcula métricas rolling vs benchmark.
        
        Args:
            returns: DataFrame de retornos diarios de activos
            weights: Array de pesos del portafolio
            benchmark_returns: Serie de retornos del benchmark
            window: Ventana para cálculo rolling. Por defecto usa config.ROLLING_WINDOW
            ddof: Grados de libertad para cálculos estadísticos
            
        Returns:
            DataFrame con tracking error y beta rolling
        """
        if window is None:
            window = ROLLING_WINDOW
        
        te_rolling = self.te_calc.calculate_rolling(
            returns, weights, benchmark_returns, window, ddof
        )

        beta_rolling = self.beta_calc.calculate_rolling(
            returns, weights, benchmark_returns, window, ddof
        )
        
        return pd.DataFrame({
            'tracking_error': te_rolling,
            'beta': beta_rolling
        })