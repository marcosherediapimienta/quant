import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import List, Optional
from ....tools.config import FRONTIER_POINTS, OPTIMIZATION_METHOD, ANNUAL_FACTOR

@dataclass
class FrontierResult:
    returns: np.ndarray
    volatilities: np.ndarray
    weights: np.ndarray
    assets: List[str]
    min_return: Optional[float] = None  # retorno mínimo alcanzable
    max_return: Optional[float] = None  # retorno máximo alcanzable
    optimization_failures: int = 0      # contador de fallos

class EfficientFrontierCalculator:
    """
    Calcula la frontera eficiente de Markowitz.
    
    Responsabilidad: Optimizar carteras para diferentes niveles de retorno objetivo.
    
    Teoría:
    - La frontera eficiente son los portfolios con máximo retorno para cada nivel de riesgo
    - O equivalentemente: mínimo riesgo para cada nivel de retorno
    - Límites: Portfolio de mínima varianza hasta portfolio de máximo retorno
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
        n_points: int = None,
        allow_short: bool = False
    ) -> FrontierResult:
        """
        Calcula la frontera eficiente.
        
        Metodología mejorada:
        1. Calcula portfolio de mínima varianza (límite inferior)
        2. Calcula portfolio de máximo retorno (límite superior)
        3. Genera targets entre esos límites ALCANZABLES
        4. Optimiza para cada target con manejo de errores robusto
        
        Args:
            returns: DataFrame de retornos diarios
            n_points: Número de puntos (None = usar config)
            allow_short: Permitir posiciones cortas
            
        Returns:
            FrontierResult con portfolios eficientes
        """
        n_points = n_points if n_points else FRONTIER_POINTS
        
        # Validación básica
        if returns.empty or len(returns.columns) < 2:
            return FrontierResult(
                np.array([]), np.array([]), np.array([]), [],
                None, None, 0
            )
        
        assets = list(returns.columns)
        
        # Limpiar NaN antes de calcular estadísticas
        returns_clean = returns.dropna()
        
        if returns_clean.empty:
            return FrontierResult(
                np.array([]), np.array([]), np.array([]), [],
                None, None, 0
            )
        
        mean_ret = returns_clean.mean() * self.annual_factor
        cov_matrix = returns_clean.cov() * self.annual_factor
        
        # Verificar que no haya NaN en mean_ret
        if mean_ret.isna().any():
            # Si hay NaN, usar 0 como fallback o eliminar esos activos
            mean_ret = mean_ret.fillna(0)
        
        n = len(assets)
        bounds = tuple((-1, 1) if allow_short else (0, 1) for _ in range(n))
        
        def portfolio_variance(w):
            return float(w.T @ cov_matrix.values @ w)
        
        def portfolio_return(w):
            return float(np.sum(w * mean_ret.values))

        # 1. Portfolio de mínima varianza (límite inferior)
        min_var_ret, min_var_vol, min_var_weights = self.minimum_variance_portfolio(
            returns, allow_short
        )
        
        # 2. Portfolio de máximo retorno (límite superior)
        max_ret_portfolio = self._maximum_return_portfolio(
            mean_ret, cov_matrix, bounds
        )
        
        # Determinar rango alcanzable de retornos
        if not np.isnan(min_var_ret) and max_ret_portfolio is not None:
            min_achievable = min_var_ret
            max_achievable = max_ret_portfolio['return']
        else:
            # Fallback: usar media de activos individuales
            min_achievable = mean_ret.min()
            max_achievable = mean_ret.max()
        
        # Targets solo en rango alcanzable
        # Añadir pequeño margen para evitar problemas numéricos
        margin = (max_achievable - min_achievable) * 0.01
        targets = np.linspace(
            min_achievable + margin,
            max_achievable - margin,
            n_points
        )
        
        # Optimización para cada target
        eff_returns = []
        eff_volatilities = []
        eff_weights = []
        failures = 0
        
        for target in targets:
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {"type": "eq", "fun": lambda w, t=target: portfolio_return(w) - t}
            ]
            
            # Guess inicial: igual peso
            x0 = np.full(n, 1.0 / n)
            
            try:
                result = minimize(
                    portfolio_variance,
                    x0=x0,
                    method=OPTIMIZATION_METHOD,
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 1000}  # Límite de iteraciones
                )
                
                # Validación más estricta del resultado
                if result.success and self._validate_portfolio(result.x, bounds):
                    w = result.x
                    eff_weights.append(w)
                    eff_returns.append(portfolio_return(w))
                    eff_volatilities.append(np.sqrt(portfolio_variance(w)))
                else:
                    failures += 1
                    
            except Exception as e:
                # Manejo robusto de errores de optimización
                failures += 1
                continue
        
        return FrontierResult(
            np.array(eff_returns),
            np.array(eff_volatilities),
            np.array(eff_weights),
            assets,
            min_achievable,
            max_achievable,
            failures
        )
    
    def _maximum_return_portfolio(
        self,
        mean_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        bounds: tuple
    ) -> Optional[dict]:
        """
        Calcula el portfolio de máximo retorno.
        
        En teoría simple: 100% en el activo con mayor retorno.
        En práctica: puede haber restricciones que lo impidan.
        
        Args:
            mean_returns: Retornos esperados anualizados
            cov_matrix: Matriz de covarianza anualizada
            bounds: Restricciones de pesos
            
        Returns:
            Dict con 'return', 'volatility', 'weights' o None si falla
        """
        n = len(mean_returns)
        
        # Función objetivo: maximizar retorno = minimizar retorno negativo
        def neg_return(w):
            return -float(np.sum(w * mean_returns.values))
        
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        
        # Guess: todo el peso al activo de mayor retorno
        best_asset_idx = np.argmax(mean_returns.values)
        x0 = np.zeros(n)
        x0[best_asset_idx] = 1.0
        
        try:
            result = minimize(
                neg_return,
                x0=x0,
                method=OPTIMIZATION_METHOD,
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                w = result.x
                ret = float(np.sum(w * mean_returns.values))
                vol = float(np.sqrt(w.T @ cov_matrix.values @ w))
                
                return {
                    'return': ret,
                    'volatility': vol,
                    'weights': w
                }
        except Exception:
            pass
        
        return None
    
    def _validate_portfolio(self, weights: np.ndarray, bounds: tuple) -> bool:
        """
        Valida que el portfolio cumple restricciones básicas.
        
        Checks:
        1. Pesos suman ~1.0 (con tolerancia)
        2. Pesos están dentro de bounds
        3. No hay NaN o Inf
        
        Args:
            weights: Array de pesos
            bounds: Tuplas de (min, max) para cada peso
            
        Returns:
            True si el portfolio es válido
        """
        # Check 1: No NaN o Inf
        if not np.all(np.isfinite(weights)):
            return False
        
        # Check 2: Suma ~1.0 (tolerancia 1%)
        if not np.isclose(np.sum(weights), 1.0, atol=0.01):
            return False
        
        # Check 3: Dentro de bounds
        for w, (lower, upper) in zip(weights, bounds):
            if w < lower - 1e-6 or w > upper + 1e-6:
                return False
        
        return True
    
    def minimum_variance_portfolio(
        self,
        returns: pd.DataFrame,
        allow_short: bool = False
    ) -> tuple:
        """
        Calcula el portfolio de mínima varianza.
        
        Responsabilidad: Encontrar la cartera con menor riesgo.
        
        Este portfolio representa el punto más a la izquierda de la frontera eficiente.
        
        Args:
            returns: DataFrame de retornos
            allow_short: Permitir posiciones cortas
            
        Returns:
            (return, volatility, weights) o (nan, nan, []) si falla
        """
        if returns.empty:
            return np.nan, np.nan, np.array([])
        
        mean_ret = returns.mean() * self.annual_factor
        cov_matrix = returns.cov() * self.annual_factor
        n = len(returns.columns)
        
        bounds = tuple((-1, 1) if allow_short else (0, 1) for _ in range(n))
        
        def portfolio_variance(w):
            return float(w.T @ cov_matrix.values @ w)
        
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        
        result = minimize(
            portfolio_variance,
            x0=np.full(n, 1.0 / n),
            method=OPTIMIZATION_METHOD,
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            w = result.x
            ret = float(np.sum(w * mean_ret.values))
            vol = np.sqrt(portfolio_variance(w))
            return ret, vol, w
        
        return np.nan, np.nan, np.array([])