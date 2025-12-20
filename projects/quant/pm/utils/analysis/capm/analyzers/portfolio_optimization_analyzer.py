import numpy as np
import pandas as pd
from typing import Dict, Optional
from ..components.efficient_frontier import EfficientFrontierCalculator
from ..components.cml_calculator import CMLCalculator
from ..components.sml_calculator import SMLCalculator, SMLResult


class PortfolioOptimizationAnalyzer:

    def __init__(self, annual_factor: float = 252.0):
        self.annual_factor = annual_factor
        self.frontier_calc = EfficientFrontierCalculator(annual_factor)
        self.cml_calc = CMLCalculator()
        self.sml_calc = SMLCalculator()
    
    def analyze_efficient_frontier(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float,
        n_points: int = 60,
        allow_short: bool = False
    ) -> Dict:

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
            n_points=100
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
        max_beta: float = 2.0,
        n_points: int = 100
    ) -> SMLResult:

        return self.sml_calc.calculate(risk_free_rate, market_return, max_beta, n_points)
    
    def is_asset_undervalued(
        self,
        actual_return: float,
        beta: float,
        risk_free_rate: float,
        market_return: float
    ) -> bool:

        return self.sml_calc.is_undervalued(
            actual_return, beta, risk_free_rate, market_return
        )
    