import numpy as np
import pandas as pd
from .helpers import calculate_portfolio_returns, annualize_return
from ....tools.config import ANNUAL_FACTOR, ROLLING_WINDOW

class SortinoCalculator:
    """
    Calcula el Sortino Ratio de un portfolio.
    
    Responsabilidad: Ratio riesgo-ajustado usando solo downside volatility.
    
    Diferencia con Sharpe:
    - Sharpe penaliza toda la volatilidad (upside + downside)
    - Sortino solo penaliza downside (retornos bajo MAR)
    
    MAR (Minimum Acceptable Return):
    - Normalmente = 0 (penalizar pérdidas)
    - Puede ser risk_free_rate o target personalizado
    """

    def __init__(self, annual_factor: float = None):
        """
        Args:
            annual_factor: Factor de anualización (None = usar config)
        """
        self.annual_factor = annual_factor if annual_factor else ANNUAL_FACTOR
    
    def calculate(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        risk_free_rate: float,
        mar: float = 0.0,  #MAR por defecto = 0
        ddof: int = 0
    ) -> float:
        """
        Calcula Sortino Ratio del portfolio.
        
        Fórmula:
        Sortino = (Ra - Rf) / Downside Deviation
        Downside Deviation = sqrt(E[min(Ri - MAR, 0)²])
        
        Args:
            returns: DataFrame de retornos
            weights: Pesos del portfolio
            risk_free_rate: Tasa libre de riesgo (para numerador)
            mar: Minimum Acceptable Return anualizado (para calcular downside)
            ddof: Grados de libertad
            
        Returns:
            Sortino Ratio
            
        Interpretación:
        - Sortino > 2.0: Excelente (retorno alto con poco downside)
        - Sortino 1.0-2.0: Bueno
        - Sortino < 1.0: Pobre (retorno no compensa downside)
        """
        portfolio_ret = calculate_portfolio_returns(returns, weights)
        annual_return = annualize_return(portfolio_ret, self.annual_factor)
        
        # MAR diario
        daily_mar = mar / self.annual_factor
        
        # Calcular downside deviation respecto al MAR
        # Solo considerar retornos por debajo del MAR
        downside_returns = portfolio_ret[portfolio_ret < daily_mar]
        
        if len(downside_returns) == 0:
            # No hay retornos negativos respecto al MAR = perfecto
            return np.inf
        
        # Downside semi-deviation (solo desviaciones negativas)
        # Usar (Ri - MAR)² para los retornos bajo MAR
        downside_squared = (downside_returns - daily_mar) ** 2
        downside_vol = np.sqrt(downside_squared.mean()) * np.sqrt(self.annual_factor)
        
        if downside_vol == 0:
            return np.nan
        
        sortino = (annual_return - risk_free_rate) / downside_vol
        return float(sortino) if np.isfinite(sortino) else None
    
    def calculate_rolling(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        risk_free_rate: float,
        mar: float = 0.0,  #MAR por defecto = 0
        window: int = None,
        ddof: int = 0
    ) -> pd.Series:
        """
        Calcula Sortino Ratio móvil.
        
        Args:
            returns: DataFrame de retornos
            weights: Pesos del portfolio
            risk_free_rate: Tasa libre de riesgo
            mar: Minimum Acceptable Return anualizado
            window: Ventana móvil (None = usar config)
            ddof: Grados de libertad
        """
        window = window if window else ROLLING_WINDOW
        
        portfolio_ret = calculate_portfolio_returns(returns, weights)
        daily_mar = mar / self.annual_factor
        
        def rolling_sortino(x):
            if len(x) < 2:
                return np.nan
            
            mu = x.mean() * self.annual_factor
            
            #Downside respecto a MAR
            downside = x[x < daily_mar]
            
            if len(downside) == 0:
                return np.inf
            
            # Downside semi-deviation
            dd_squared = ((downside - daily_mar) ** 2).mean()
            dd = np.sqrt(dd_squared) * np.sqrt(self.annual_factor)
            
            return (mu - risk_free_rate) / dd if dd > 0 else np.nan
        
        return portfolio_ret.rolling(window).apply(rolling_sortino, raw=False)