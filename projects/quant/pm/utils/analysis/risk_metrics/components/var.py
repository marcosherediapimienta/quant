"""
Value at Risk (VaR).

Métodos disponibles:
- Historical: Basado en cuantiles históricos
- Parametric: Asume distribución normal 
- Monte Carlo: Simulación estocástica con correlaciones
- Cornish-Fisher: Ajusta por skewness y kurtosis
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Literal, Dict
from .helpers import calculate_portfolio_returns
from .momentum import DistributionMoments
from ....tools.config import (
    ANNUAL_FACTOR,
    DEFAULT_CONFIDENCE_LEVEL,
    MONTE_CARLO_SIMULATIONS,
    MONTE_CARLO_SEED,
    SIGNIFICANCE_LEVEL
)

class VaRCalculator:
    """
    Calculadora de Value at Risk (VaR).
    
    Responsabilidad única: Calcular VaR usando diferentes metodologías.
    
    VaR representa la pérdida máxima esperada en un nivel de confianza dado.
    Ej: VaR 95% = -2.5% significa que hay 5% probabilidad de perder más de 2.5%
    
    ⚠️ IMPORTANTE: VaR paramétrico asume normalidad. Siempre validar primero.
    
    Métodos disponibles:
    1. Historical: No asume distribución (más robusto)
    2. Parametric: Asume normalidad (rápido pero requiere validación)
    3. Monte Carlo: Simulación con correlaciones (más completo)
    4. Cornish-Fisher: Ajusta por skewness/kurtosis (mejor que paramétrico)
    """

    def __init__(self, annual_factor: float = None):
        """
        Inicializa el calculador.
        
        Args:
            annual_factor: Factor de anualización (None = usar config)
        """
        self.annual_factor = annual_factor if annual_factor else ANNUAL_FACTOR
        self.moments_calc = DistributionMoments()
    
    def validate_normality(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        alpha: float = None,
        verbose: bool = True
    ) -> Dict[str, any]:
        """
        ✅ Valida si los retornos siguen distribución normal.
        
        Tests aplicados:
        1. Jarque-Bera: Basado en skewness y kurtosis
        2. Análisis de momentos superiores
        
        Args:
            returns: DataFrame de retornos
            weights: Pesos del portfolio
            alpha: Nivel de significancia (default: 0.05)
            verbose: Si True, imprime warnings
            
        Returns:
            Dict con resultados de tests y recomendaciones
        """
        alpha = alpha if alpha else SIGNIFICANCE_LEVEL
        
        portfolio_ret = calculate_portfolio_returns(returns, weights)
        
        # Test Jarque-Bera
        jb_results = self.moments_calc.calculate_jarque_bera(returns, weights, alpha)
        
        # Momentos superiores
        skew = self.moments_calc.calculate_skewness(returns, weights)
        excess_kurt = self.moments_calc.calculate_kurtosis(returns, weights, excess=True)
        
        # Criterios de normalidad
        is_normal_jb = jb_results['is_normal']
        is_skew_ok = abs(skew) < 0.5  # Skewness cercano a 0
        is_kurt_ok = abs(excess_kurt) < 1.0  # Excess kurtosis cercano a 0
        
        # Conclusión agregada
        normality_score = sum([is_normal_jb, is_skew_ok, is_kurt_ok])
        
        if normality_score >= 2:
            conclusion = "NORMAL"
            recommendation = "VaR paramétrico es apropiado"
            risk_level = "LOW"
        elif normality_score == 1:
            conclusion = "CUESTIONABLE"
            recommendation = "VaR paramétrico puede subestimar riesgo. Usar VaR histórico también."
            risk_level = "MEDIUM"
        else:
            conclusion = "NO NORMAL"
            recommendation = "⚠️ VaR paramétrico NO recomendado. Usar VaR histórico o Monte Carlo."
            risk_level = "HIGH"
        
        results = {
            'is_normal': conclusion == "NORMAL",
            'conclusion': conclusion,
            'risk_level': risk_level,
            'recommendation': recommendation,
            'jarque_bera': {
                'statistic': jb_results['jb_statistic'],
                'p_value': jb_results['p_value'],
                'is_normal': jb_results['is_normal']
            },
            'skewness': skew,
            'excess_kurtosis': excess_kurt,
            'is_skew_ok': is_skew_ok,
            'is_kurt_ok': is_kurt_ok
        }
        
        # Warnings verbosos
        if verbose and conclusion != "NORMAL":
            print(f"\n⚠️ WARNING: Retornos {conclusion}")
            print(f"   Jarque-Bera p-value: {jb_results['p_value']:.4f} {'✅' if is_normal_jb else '❌'}")
            print(f"   Skewness: {skew:.3f} {'✅' if is_skew_ok else '❌'}")
            print(f"   Excess Kurtosis: {excess_kurt:.3f} {'✅' if is_kurt_ok else '❌'}")
            print(f"   Recomendación: {recommendation}\n")
        
        return results
    
    def calculate_historical(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        confidence_level: float = None
    ) -> Dict[str, float]:
        """
        Calcula VaR histórico (no asume distribución).
        
        Método: Usa cuantiles empíricos de los retornos históricos.
        
        Args:
            returns: DataFrame de retornos diarios
            weights: Pesos del portfolio
            confidence_level: Nivel de confianza (None = usar config)
            
        Returns:
            Dict con VaR diario y anualizado (valor y porcentaje)
            
        Ventajas:
        - ✅ No asume ninguna distribución
        - ✅ Captura eventos extremos reales
        - ✅ Simple y robusto
        
        Desventajas:
        - ⚠️ Limitado por datos históricos
        - ⚠️ No extrapola más allá del rango observado
        
        Interpretación:
        VaR = -2.5% (95% confianza) → Hay 5% probabilidad de perder más de 2.5%
        """
        confidence_level = confidence_level if confidence_level else DEFAULT_CONFIDENCE_LEVEL
        
        portfolio_ret = calculate_portfolio_returns(returns, weights)
        alpha = 1.0 - confidence_level
        
        # Cuantil histórico
        var_daily = np.quantile(portfolio_ret, alpha)
        
        # ⚠️ NOTA: Anualización de VaR con sqrt(T) asume retornos i.i.d. normales
        # Esta es una aproximación. Para uso riguroso, considerar:
        # 1. No anualizar (reportar VaR diario directamente)
        # 2. Usar bootstrap para VaR en horizontes largos
        # 3. Simular trayectorias multi-periodo
        var_annual = var_daily * np.sqrt(self.annual_factor)
        
        return {
            'method': 'historical',
            'var_daily': float(var_daily),
            'var_annual': float(var_annual),  # Aproximación bajo normalidad
            'var_daily_pct': float(var_daily * 100),
            'var_annual_pct': float(var_annual * 100),
            'confidence_level': confidence_level,
            'sample_size': len(portfolio_ret)
        }
    
    def calculate_parametric(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        confidence_level: float = None,
        validate_normality: bool = True
    ) -> Dict[str, float]:
        """
        Calcula VaR paramétrico (asume distribución normal).
        
        ⚠️ IMPORTANTE: Asume normalidad. Validar antes de usar.
        
        Método: Usa media y desviación estándar asumiendo normalidad.
        Fórmula: VaR = μ + z_α × σ
        
        Args:
            returns: DataFrame de retornos diarios
            weights: Pesos del portfolio
            confidence_level: Nivel de confianza (None = usar config)
            validate_normality: Si True, valida normalidad primero
            
        Returns:
            Dict con VaR diario y anualizado
            
        Ventajas:
        - ✅ Rápido y simple
        - ✅ Smooth (no depende de outliers individuales)
        - ✅ Fácil de implementar
        
        Desventajas:
        - ⚠️ Subestima riesgo si hay fat tails (kurtosis > 0)
        - ⚠️ Incorrecto si hay asimetría (skewness != 0)
        - ⚠️ Puede fallar en crisis (eventos extremos)
        """
        confidence_level = confidence_level if confidence_level else DEFAULT_CONFIDENCE_LEVEL
        
        # Validación automática de normalidad
        normality_result = None
        if validate_normality:
            normality_result = self.validate_normality(
                returns, weights, verbose=True
            )
            
            if normality_result['risk_level'] == 'HIGH':
                print("⚠️ CRITICAL: VaR paramétrico puede ser muy impreciso")
                print(f"   Considera usar calculate_historical() o calculate_cornish_fisher()\n")
        
        portfolio_ret = calculate_portfolio_returns(returns, weights)
        
        # Media y desviación estándar
        mu = portfolio_ret.mean()
        sigma = portfolio_ret.std(ddof=0)
        
        # Z-score para el nivel de confianza
        alpha = 1.0 - confidence_level
        z_score = stats.norm.ppf(alpha)
        
        # VaR = mu + z_score * sigma (z_score es negativo)
        var_daily = mu + z_score * sigma
        
        # ⚠️ Anualización con sqrt(T) válida solo bajo normalidad
        var_annual = var_daily * np.sqrt(self.annual_factor)
        
        result = {
            'method': 'parametric',
            'var_daily': float(var_daily),
            'var_annual': float(var_annual),
            'var_daily_pct': float(var_daily * 100),
            'var_annual_pct': float(var_annual * 100),
            'confidence_level': confidence_level,
            'normality_validated': validate_normality,
            'mu': float(mu),
            'sigma': float(sigma),
            'z_score': float(z_score)
        }
        
        if normality_result:
            result['normality_check'] = normality_result
        
        return result
    
    def calculate_cornish_fisher(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        confidence_level: float = None
    ) -> Dict[str, float]:
        """
        ✅ Calcula VaR usando expansión de Cornish-Fisher.
        
        Mejora del VaR paramétrico que ajusta por skewness y kurtosis.
        
        Método: Ajusta el z-score normal por momentos superiores:
        z_CF = z + (z² - 1)×S/6 + (z³ - 3z)×K/24 - (2z³ - 5z)×S²/36
        
        donde:
        - S = skewness
        - K = excess kurtosis
        
        Args:
            returns: DataFrame de retornos diarios
            weights: Pesos del portfolio
            confidence_level: Nivel de confianza (None = usar config)
            
        Returns:
            Dict con VaR ajustado
            
        Ventajas:
        - ✅ Corrige por asimetría y colas gordas
        - ✅ Más preciso que VaR paramétrico simple
        - ✅ Rápido de calcular
        
        Desventajas:
        - ⚠️ Puede ser inestable con skewness/kurtosis extremos
        
        Referencias:
        - Cornish, E. A., & Fisher, R. A. (1937)
        - Usado en RiskMetrics y Basel II
        """
        confidence_level = confidence_level if confidence_level else DEFAULT_CONFIDENCE_LEVEL
        
        portfolio_ret = calculate_portfolio_returns(returns, weights)
        
        # Estadísticos básicos
        mu = portfolio_ret.mean()
        sigma = portfolio_ret.std(ddof=0)
        
        # Momentos superiores
        skew = self.moments_calc.calculate_skewness(returns, weights)
        excess_kurt = self.moments_calc.calculate_kurtosis(returns, weights, excess=True)
        
        # Z-score normal
        alpha = 1.0 - confidence_level
        z = stats.norm.ppf(alpha)
        
        # Ajuste Cornish-Fisher
        z_cf = (
            z
            + (z**2 - 1) * skew / 6
            + (z**3 - 3*z) * excess_kurt / 24
            - (2*z**3 - 5*z) * (skew**2) / 36
        )
        
        # VaR ajustado
        var_daily = mu + z_cf * sigma
        var_annual = var_daily * np.sqrt(self.annual_factor)
        
        return {
            'method': 'cornish_fisher',
            'var_daily': float(var_daily),
            'var_annual': float(var_annual),
            'var_daily_pct': float(var_daily * 100),
            'var_annual_pct': float(var_annual * 100),
            'confidence_level': confidence_level,
            'mu': float(mu),
            'sigma': float(sigma),
            'skewness': float(skew),
            'excess_kurtosis': float(excess_kurt),
            'z_normal': float(z),
            'z_adjusted': float(z_cf),
            'adjustment': float(z_cf - z)
        }
    
    def calculate_monte_carlo(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        confidence_level: float = None,
        n_simulations: int = None,
        seed: int = None
    ) -> Dict[str, float]:
        """
        ✅ Calcula VaR usando simulación Monte Carlo CON correlaciones.
        
        Método:
        1. Estima media y covarianza de retornos
        2. Simula N escenarios usando distribución multivariada normal
        3. Calcula retornos del portfolio para cada escenario
        4. VaR = percentil de la distribución simulada
        
        ⚠️ IMPORTANTE: Respeta correlaciones entre activos (Cholesky decomposition)
        
        Args:
            returns: DataFrame de retornos diarios
            weights: Pesos del portfolio
            confidence_level: Nivel de confianza (None = usar config)
            n_simulations: Número de simulaciones (None = usar config)
            seed: Semilla aleatoria para reproducibilidad
            
        Returns:
            Dict con VaR y distribución simulada
            
        Ventajas:
        - ✅ Captura correlaciones entre activos
        - ✅ Flexible (puede usar otras distribuciones)
        - ✅ Convergencia con más simulaciones
        
        Desventajas:
        - ⚠️ Más lento que otros métodos
        - ⚠️ Aún asume normalidad (multivariada)
        """
        confidence_level = confidence_level if confidence_level else DEFAULT_CONFIDENCE_LEVEL
        n_simulations = n_simulations if n_simulations else MONTE_CARLO_SIMULATIONS
        seed = seed if seed is not None else MONTE_CARLO_SEED
        
        # Set seed para reproducibilidad
        if seed is not None:
            np.random.seed(seed)
        
        # Estadísticos de retornos
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values
        n_assets = len(returns.columns)
        
        # Descomposición de Cholesky para correlaciones
        # Esto asegura que las simulaciones respeten correlaciones
        try:
            L = np.linalg.cholesky(cov_matrix)
        except np.linalg.LinAlgError:
            # Si la matriz no es definida positiva, usar eigen-decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            eigenvalues = np.maximum(eigenvalues, 1e-10)  # Asegurar positividad
            L = eigenvectors @ np.diag(np.sqrt(eigenvalues))
        
        # Simulaciones
        # z ~ N(0,1) independientes
        z = np.random.standard_normal((n_assets, n_simulations))
        
        # Retornos correlacionados: r = μ + L·z
        simulated_returns = mean_returns.reshape(-1, 1) + L @ z
        
        # Retornos del portfolio para cada simulación
        portfolio_returns = weights @ simulated_returns
        
        # VaR = percentil de la distribución simulada
        alpha = 1.0 - confidence_level
        var_daily = np.quantile(portfolio_returns, alpha)
        var_annual = var_daily * np.sqrt(self.annual_factor)
        
        return {
            'method': 'monte_carlo',
            'var_daily': float(var_daily),
            'var_annual': float(var_annual),
            'var_daily_pct': float(var_daily * 100),
            'var_annual_pct': float(var_annual * 100),
            'confidence_level': confidence_level,
            'n_simulations': n_simulations,
            'simulated_mean': float(portfolio_returns.mean()),
            'simulated_std': float(portfolio_returns.std()),
            'min_simulated': float(portfolio_returns.min()),
            'max_simulated': float(portfolio_returns.max()),
            'seed': seed
        }
    
    def calculate(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        confidence_level: float = None,
        method: Literal['historical', 'parametric', 'cornish_fisher', 'monte_carlo'] = 'historical'
    ) -> Dict[str, float]:
        """
        Calcula VaR usando el método especificado.
        
        Args:
            returns: DataFrame de retornos diarios
            weights: Pesos del portfolio
            confidence_level: Nivel de confianza (default: 0.95)
            method: Método a usar (default: 'historical')
            
        Returns:
            Dict con VaR calculado
            
        Recomendaciones:
        - 'historical': Más robusto, recomendado por defecto
        - 'parametric': Solo si retornos son normales (validar primero)
        - 'cornish_fisher': Mejor que paramétrico si hay skew/kurtosis
        - 'monte_carlo': Más completo pero más lento
        """
        methods = {
            'historical': self.calculate_historical,
            'parametric': self.calculate_parametric,
            'cornish_fisher': self.calculate_cornish_fisher,
            'monte_carlo': self.calculate_monte_carlo
        }
        
        if method not in methods:
            raise ValueError(
                f"Método '{method}' no válido. "
                f"Opciones: {list(methods.keys())}"
            )
        
        return methods[method](returns, weights, confidence_level)