import numpy as np
from dataclasses import dataclass
from ....tools.config import SML_CONFIG

@dataclass
class SMLResult:
    beta_axis: np.ndarray
    expected_returns: np.ndarray
    market_return: float
    risk_free_rate: float
    slope: float

class SMLCalculator:
    """
    Calcula la Security Market Line (SML).
    
    Responsabilidad: Determinar retornos esperados según CAPM para diferentes betas.
    """

    def calculate(
        self,
        risk_free_rate: float,
        market_return: float,
        max_beta: float = None,
        n_points: int = None
    ) -> SMLResult:
        """
        Calcula la Security Market Line.
        
        Args:
            risk_free_rate: Tasa libre de riesgo
            market_return: Retorno esperado del mercado
            max_beta: Beta máximo para graficar. Por defecto usa config
            n_points: Número de puntos. Por defecto usa config
            
        Returns:
            SMLResult con línea SML
        """
        if max_beta is None:
            max_beta = SML_CONFIG['max_beta']
        
        if n_points is None:
            n_points = SML_CONFIG['n_points']

        beta_axis = np.linspace(0, max_beta, n_points)
        slope = market_return - risk_free_rate
        expected_returns = risk_free_rate + slope * beta_axis
        
        return SMLResult(
            beta_axis, expected_returns, market_return, risk_free_rate, slope
        )
    
    def expected_return_for_beta(
        self,
        beta: float,
        risk_free_rate: float,
        market_return: float
    ) -> float:
        """
        Calcula el retorno esperado según CAPM para un beta específico.
        
        Args:
            beta: Beta del activo
            risk_free_rate: Tasa libre de riesgo
            market_return: Retorno esperado del mercado
            
        Returns:
            Retorno esperado según CAPM
        """
        if np.isnan(beta):
            return np.nan
        return risk_free_rate + beta * (market_return - risk_free_rate)
    
    def is_undervalued(
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
            True si el activo está infravalorado
        """
        expected = self.expected_return_for_beta(beta, risk_free_rate, market_return)
        if np.isnan(expected) or np.isnan(actual_return):
            return False
        return actual_return > expected