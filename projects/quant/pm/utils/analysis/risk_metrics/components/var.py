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
    def __init__(self, annual_factor: float = None):
        self.annual_factor = annual_factor if annual_factor else ANNUAL_FACTOR
        self.moments_calc = DistributionMoments()
    
    def validate_normality(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        alpha: float = None,
        verbose: bool = True
    ) -> Dict[str, any]:

        alpha = alpha if alpha else SIGNIFICANCE_LEVEL
        
        portfolio_ret = calculate_portfolio_returns(returns, weights)
        jb_results = self.moments_calc.calculate_jarque_bera(returns, weights, alpha)
        skew = self.moments_calc.calculate_skewness(returns, weights)
        excess_kurt = self.moments_calc.calculate_kurtosis(returns, weights, excess=True)
        is_normal_jb = jb_results['is_normal']
        is_skew_ok = abs(skew) < 0.5 
        is_kurt_ok = abs(excess_kurt) < 1.0  
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
            recommendation = "VaR paramétrico NO recomendado. Usar VaR histórico o Monte Carlo."
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

        confidence_level = confidence_level if confidence_level else DEFAULT_CONFIDENCE_LEVEL
        
        portfolio_ret = calculate_portfolio_returns(returns, weights)
        alpha = 1.0 - confidence_level
        var_daily = np.quantile(portfolio_ret, alpha)
        var_annual = var_daily * np.sqrt(self.annual_factor)
        
        return {
            'method': 'historical',
            'var_daily': float(var_daily),
            'var_annual': float(var_annual),  
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

        confidence_level = confidence_level if confidence_level else DEFAULT_CONFIDENCE_LEVEL

        normality_result = None
        if validate_normality:
            normality_result = self.validate_normality(
                returns, weights, verbose=True
            )
            
            if normality_result['risk_level'] == 'HIGH':
                print("⚠️ CRITICAL: VaR paramétrico puede ser muy impreciso")
                print(f"   Considera usar calculate_historical() o calculate_cornish_fisher()\n")
        
        portfolio_ret = calculate_portfolio_returns(returns, weights)
        mu = portfolio_ret.mean()
        sigma = portfolio_ret.std(ddof=0)
        alpha = 1.0 - confidence_level
        z_score = stats.norm.ppf(alpha)
        var_daily = mu + z_score * sigma
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

        confidence_level = confidence_level if confidence_level else DEFAULT_CONFIDENCE_LEVEL
        
        portfolio_ret = calculate_portfolio_returns(returns, weights)
        mu = portfolio_ret.mean()
        sigma = portfolio_ret.std(ddof=0)
        skew = self.moments_calc.calculate_skewness(returns, weights)
        excess_kurt = self.moments_calc.calculate_kurtosis(returns, weights, excess=True)
        alpha = 1.0 - confidence_level
        z = stats.norm.ppf(alpha)
        z_cf = (
            z
            + (z**2 - 1) * skew / 6
            + (z**3 - 3*z) * excess_kurt / 24
            - (2*z**3 - 5*z) * (skew**2) / 36
        )
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

        confidence_level = confidence_level if confidence_level else DEFAULT_CONFIDENCE_LEVEL
        n_simulations = n_simulations if n_simulations else MONTE_CARLO_SIMULATIONS
        seed = seed if seed is not None else MONTE_CARLO_SEED

        if seed is not None:
            np.random.seed(seed)

        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values
        n_assets = len(returns.columns)
        
        try:
            L = np.linalg.cholesky(cov_matrix)
        except np.linalg.LinAlgError:
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            eigenvalues = np.maximum(eigenvalues, 1e-10) 
            L = eigenvectors @ np.diag(np.sqrt(eigenvalues))

        z = np.random.standard_normal((n_assets, n_simulations))
        simulated_returns = mean_returns.reshape(-1, 1) + L @ z
        portfolio_returns = weights @ simulated_returns
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