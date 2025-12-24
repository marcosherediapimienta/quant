import pandas as pd
from typing import Dict
from ..components.efficient_frontier import EfficientFrontierCalculator
from ..components.cml_calculator import CMLCalculator
from ..components.sml_calculator import SMLCalculator, SMLResult
from ....tools.config import (
    ANNUAL_FACTOR,
    FRONTIER_POINTS,
    CML_POINTS,
    SML_CONFIG
)

class PortfolioOptimizationAnalyzer:
    """
    Analyzer para optimización de portafolios con frontera eficiente.
    
    Responsabilidad: Coordinar cálculos de frontera eficiente, CML, SML
    y portafolios óptimos.
    """

    def __init__(self, annual_factor: float = None):
        """
        Args:
            annual_factor: Factor de anualización. Por defecto usa config.ANNUAL_FACTOR
        """
        self.annual_factor = annual_factor or ANNUAL_FACTOR
        self.frontier_calc = EfficientFrontierCalculator(self.annual_factor)
        self.cml_calc = CMLCalculator()
        self.sml_calc = SMLCalculator()
    
    def analyze_efficient_frontier(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float,
        n_points: int = None,
        allow_short: bool = False
    ) -> Dict:
        """
        Analiza la frontera eficiente y el portafolio tangente.
        
        Args:
            returns: DataFrame de retornos de activos
            risk_free_rate: Tasa libre de riesgo anualizada
            n_points: Puntos para frontera. Por defecto usa config.FRONTIER_POINTS
            allow_short: Si permite ventas en corto
            
        Returns:
            Dict con frontera, CML y portafolio tangente
        """
        if n_points is None:
            n_points = FRONTIER_POINTS

        frontier = self.frontier_calc.calculate(returns, n_points, allow_short)
        
        if len(frontier.returns) == 0:
            return {
                'frontier': frontier,
                'cml': None,
                'tangent_portfolio': None
            }

        cml = self.cml_calc.calculate(
            frontier.returns,
            frontier.volatilities,
            risk_free_rate,
            n_points=CML_POINTS
        )

        tangent_weights = None
        if cml.tangent_index >= 0 and cml.tangent_index < len(frontier.weights):
            tangent_weights = frontier.weights[cml.tangent_index]
        
        return {
            'frontier': frontier,
            'cml': cml,
            'tangent_portfolio': {
                'return': cml.tangent_return,
                'volatility': cml.tangent_volatility,
                'weights': tangent_weights,
                'sharpe_ratio': cml.slope,
                'assets': frontier.assets
            }
        }
    
    def analyze_minimum_variance(
        self,
        returns: pd.DataFrame,
        allow_short: bool = False
    ) -> Dict:
        """
        Calcula el portafolio de mínima varianza.
        
        Args:
            returns: DataFrame de retornos de activos
            allow_short: Si permite ventas en corto
            
        Returns:
            Dict con retorno, volatilidad, pesos y activos
        """
        ret, vol, weights = self.frontier_calc.minimum_variance_portfolio(
            returns, allow_short
        )
        
        return {
            'return': ret,
            'volatility': vol,
            'weights': weights,
            'assets': list(returns.columns)
        }
    
    def analyze_sml(
        self,
        risk_free_rate: float,
        market_return: float,
        max_beta: float = None,
        n_points: int = None
    ) -> SMLResult:
        """
        Calcula la Security Market Line (SML).
        
        Args:
            risk_free_rate: Tasa libre de riesgo
            market_return: Retorno esperado del mercado
            max_beta: Beta máximo para graficar. Por defecto usa config
            n_points: Puntos para generar línea. Por defecto usa config
            
        Returns:
            SMLResult con betas y retornos esperados
        """
        if max_beta is None:
            max_beta = SML_CONFIG['max_beta']
        
        if n_points is None:
            n_points = SML_CONFIG['n_points']

        return self.sml_calc.calculate(risk_free_rate, market_return, max_beta, n_points)
    
    def is_asset_undervalued(
        self,
        actual_return: float,
        beta: float,
        risk_free_rate: float,
        market_return: float
    ) -> bool:
        """
        Determina si un activo está infravalorado (retorno > esperado).
        
        Args:
            actual_return: Retorno real del activo
            beta: Beta del activo
            risk_free_rate: Tasa libre de riesgo
            market_return: Retorno del mercado
            
        Returns:
            True si está infravalorado
        """
        return self.sml_calc.is_undervalued(
            actual_return, beta, risk_free_rate, market_return
        )