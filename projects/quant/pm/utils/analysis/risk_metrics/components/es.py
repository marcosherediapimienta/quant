"""
Calculadora de Expected Shortfall (ES) / Conditional Value at Risk (CVaR).

ES representa la pérdida promedio esperada dado que se excede el VaR.
Es una medida más conservadora que el VaR.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Literal
from .helpers import calculate_portfolio_returns
from ....tools.config import (
    ANNUAL_FACTOR,
    DEFAULT_CONFIDENCE_LEVEL,
    MONTE_CARLO_SIMULATIONS,
    MONTE_CARLO_SEED
)

class ESCalculator:
    """
    Calculadora de Expected Shortfall (ES), también conocido como CVaR.
    
    Responsabilidad: Calcular la pérdida promedio en el peor α% de escenarios.
    
    ES es superior al VaR porque:
    - Es una medida coherente de riesgo
    - Captura la severidad de pérdidas extremas, no solo su probabilidad
    """

    def __init__(self, annual_factor: float = None):
        """
        Args:
            annual_factor: Factor de anualización (None = usar config)
        """
        self.annual_factor = annual_factor if annual_factor else ANNUAL_FACTOR
    
    def calculate_historical(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        confidence_level: float = None,
        var_value: float = None
    ) -> Dict[str, float]:
        """
        Calcula ES histórico.
        
        Método: Promedio de todos los retornos que caen bajo el VaR.
        
        Args:
            returns: DataFrame de retornos diarios
            weights: Pesos del portfolio
            confidence_level: Nivel de confianza (None = usar config)
            var_value: VaR precalculado (None = calcular automáticamente)
        """
        confidence_level = confidence_level if confidence_level else DEFAULT_CONFIDENCE_LEVEL
        
        portfolio_ret = calculate_portfolio_returns(returns, weights)
        alpha = 1.0 - confidence_level
        
        # Calcular VaR si no se proporciona
        if var_value is None:
            var_value = np.quantile(portfolio_ret, alpha)

        # ES = promedio de pérdidas peores que VaR
        tail_losses = portfolio_ret[portfolio_ret <= var_value]
        
        if len(tail_losses) == 0:
            es_daily = var_value  # Fallback si no hay datos en la cola
        else:
            es_daily = tail_losses.mean()
        
        es_annual = es_daily * np.sqrt(self.annual_factor)
        
        return {
            'es_daily': float(es_daily),
            'es_annual': float(es_annual),
            'es_daily_pct': float(es_daily * 100),
            'es_annual_pct': float(es_annual * 100)
        }
    
    def calculate_parametric(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        confidence_level: float = None
    ) -> Dict[str, float]:
        """
        Calcula ES paramétrico (asume normalidad).
        
        Método: Usa fórmula cerrada asumiendo distribución normal.
        Fórmula: ES = μ - σ * φ(z) / α
        donde φ es la PDF normal y z es el cuantil.
        
        Args:
            returns: DataFrame de retornos
            weights: Pesos del portfolio
            confidence_level: Nivel de confianza (None = usar config)
        """
        confidence_level = confidence_level if confidence_level else DEFAULT_CONFIDENCE_LEVEL
        
        portfolio_ret = calculate_portfolio_returns(returns, weights)
        alpha = 1.0 - confidence_level
        mu = portfolio_ret.mean()
        sigma = portfolio_ret.std(ddof=0)
        
        # Z-score para el nivel de confianza
        z = stats.norm.ppf(alpha)
        
        # Fórmula cerrada para ES bajo normalidad
        es_daily = mu - sigma * stats.norm.pdf(z) / alpha
        es_annual = es_daily * np.sqrt(self.annual_factor)
        
        return {
            'es_daily': float(es_daily),
            'es_annual': float(es_annual),
            'es_daily_pct': float(es_daily * 100),
            'es_annual_pct': float(es_annual * 100)
        }
    
    def calculate_monte_carlo(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        confidence_level: float = None,
        n_simulations: int = None,
        seed: int = None,
        var_value: float = None
    ) -> Dict[str, float]:
        """
        Calcula ES mediante Monte Carlo.
        
        Método: Simula escenarios y calcula promedio de cola.
        
        Args:
            returns: DataFrame de retornos
            weights: Pesos del portfolio
            confidence_level: Nivel de confianza (None = usar config)
            n_simulations: Número de simulaciones (None = usar config)
            seed: Semilla aleatoria (None = usar config)
            var_value: VaR precalculado (None = calcular)
        """
        confidence_level = confidence_level if confidence_level else DEFAULT_CONFIDENCE_LEVEL
        n_simulations = n_simulations if n_simulations else MONTE_CARLO_SIMULATIONS
        seed = seed if seed is not None else MONTE_CARLO_SEED
        
        portfolio_ret = calculate_portfolio_returns(returns, weights)
        alpha = 1.0 - confidence_level
        mu = portfolio_ret.mean()
        sigma = portfolio_ret.std(ddof=0)
        
        # Generar simulaciones
        rng = np.random.default_rng(seed)
        simulations = rng.normal(mu, sigma, n_simulations)
    
        # Calcular VaR si no se proporciona
        if var_value is None:
            var_value = np.quantile(simulations, alpha)
        
        # ES = promedio de simulaciones en la cola
        tail_losses = simulations[simulations <= var_value]
        es_daily = tail_losses.mean()
        es_annual = es_daily * np.sqrt(self.annual_factor)
        
        return {
            'es_daily': float(es_daily),
            'es_annual': float(es_annual),
            'es_daily_pct': float(es_daily * 100),
            'es_annual_pct': float(es_annual * 100)
        }
    
    def calculate(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        confidence_level: float = None,
        method: Literal['historical', 'parametric', 'monte_carlo'] = 'historical',
        **kwargs
    ) -> Dict[str, float]:
        """
        Calcula ES usando el método especificado.
        
        Args:
            returns: DataFrame de retornos
            weights: Pesos del portfolio
            confidence_level: Nivel de confianza (None = usar config)
            method: Método a usar
            **kwargs: Argumentos adicionales (n_simulations, seed, var_value)
        """
        confidence_level = confidence_level if confidence_level else DEFAULT_CONFIDENCE_LEVEL
        
        if method == 'historical':
            return self.calculate_historical(
                returns, weights, confidence_level, 
                var_value=kwargs.get('var_value')
            )
        elif method == 'parametric':
            return self.calculate_parametric(returns, weights, confidence_level)
        elif method == 'monte_carlo':
            return self.calculate_monte_carlo(
                returns, weights, confidence_level,
                n_simulations=kwargs.get('n_simulations'),
                seed=kwargs.get('seed'),
                var_value=kwargs.get('var_value')
            )
        else:
            raise ValueError(
                f"Método '{method}' no válido. "
                f"Opciones: 'historical', 'parametric', 'monte_carlo'"
            )
    
    def calculate_all_methods(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        confidence_level: float = None,
        n_simulations: int = None,
        seed: int = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Calcula ES con todos los métodos para comparación.
        
        Returns:
            Dict con resultados de cada método
        """
        confidence_level = confidence_level if confidence_level else DEFAULT_CONFIDENCE_LEVEL
        n_simulations = n_simulations if n_simulations else MONTE_CARLO_SIMULATIONS
        seed = seed if seed is not None else MONTE_CARLO_SEED
        
        return {
            'historical': self.calculate_historical(returns, weights, confidence_level),
            'parametric': self.calculate_parametric(returns, weights, confidence_level),
            'monte_carlo': self.calculate_monte_carlo(
                returns, weights, confidence_level, n_simulations, seed
            )
        }