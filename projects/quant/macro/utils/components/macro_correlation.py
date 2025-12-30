from typing import Dict
import numpy as np
import pandas as pd
import statsmodels.api as sm
from dataclasses import dataclass
from ..tools.config import (
    MAX_LAG,
    CORRELATION_MIN_OBS,
    HAC_MAXLAGS,
    CORRELATION_LAGS_DEFAULT,
)

@dataclass
class CorrelationResult:
    """Resultado de correlación lagged."""
    lag: int
    corr: float
    t_stat: float
    p_value: float
    n_obs: int

@dataclass
class BestLagResult:
    """Resultado del mejor lag para un factor."""
    factor: str
    lag: int
    corr: float
    t_stat: float
    p_value: float
    n_obs: int


class MacroCorrelationCalculator:
    """
    Calcula correlaciones entre retornos de portfolio y factores macro.
    
    Responsabilidad única: Calcular correlaciones con lags y estadísticas robustas.
    
    Métodos:
    - calculate_lagged: Correlación con diferentes lags
    - find_best_lag: Encuentra el lag óptimo para cada factor
    - calculate_matrix_with_lags: Matriz de correlaciones para varios lags
    - calculate_rolling: Correlación móvil entre series
    """
    
    def __init__(
        self,
        max_lag: int = None,
        min_obs: int = None,
        hac_maxlags: int = None
    ):
        """
        Inicializa el calculador de correlaciones.
        
        Args:
            max_lag: Máximo lag a evaluar (None = usar config)
            min_obs: Mínimo de observaciones requeridas (None = usar config)
            hac_maxlags: Lags para HAC standard errors (None = usar config)
        """
        self.max_lag = max_lag if max_lag is not None else MAX_LAG
        self.min_obs = min_obs if min_obs is not None else CORRELATION_MIN_OBS
        self.hac_maxlags = hac_maxlags if hac_maxlags is not None else HAC_MAXLAGS
    
    def calculate_lagged(
        self,
        y: pd.Series,
        x: pd.Series
    ) -> pd.DataFrame:
        """
        Calcula correlación entre y y x para diferentes lags.
        
        Args:
            y: Serie temporal dependiente (ej: retornos de portfolio)
            x: Serie temporal independiente (ej: factor macro)
            
        Returns:
            DataFrame con lag, corr, t-stat, p-value, n_obs
            
        Método:
        - Prueba lags desde -max_lag hasta +max_lag
        - Calcula correlación Pearson
        - Estima significancia con regresión OLS + HAC standard errors
        """
        results = []
        
        for lag in range(-self.max_lag, self.max_lag + 1):
            y_aligned = y.copy()
            x_lagged = x.shift(lag)
            
            df = pd.concat([y_aligned, x_lagged], axis=1).dropna()
            
            if len(df) < self.min_obs:
                continue
            
            corr = df.iloc[:, 0].corr(df.iloc[:, 1])
            
            # Test de significancia con HAC
            try:
                X = sm.add_constant(df.iloc[:, 1].values)
                model = sm.OLS(df.iloc[:, 0].values, X)
                n = len(df)
                hac_lags = self.hac_maxlags if self.hac_maxlags is not None else int(np.sqrt(n))
                fit = model.fit(cov_type='HAC', cov_kwds={'maxlags': hac_lags})
                t_stat = float(fit.tvalues[1])
                p_value = float(fit.pvalues[1])
            except Exception:
                t_stat = np.nan
                p_value = np.nan
            
            results.append({
                'lag': lag,
                'corr': float(corr),
                't': t_stat,
                'p': p_value,
                'n': len(df)
            })
        
        return pd.DataFrame(results)
    
    def find_best_lag(
        self,
        portfolio_returns: pd.Series,
        macro_factors: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Encuentra el lag óptimo para cada factor macro.
        
        Args:
            portfolio_returns: Retornos del portfolio
            macro_factors: DataFrame con factores macro
            
        Returns:
            DataFrame con mejor lag por factor, ordenado por |correlación|
            
        Criterio:
        - Selecciona el lag con mayor |correlación|
        - Incluye estadísticas de significancia
        """
        results = []
        
        for factor_name in macro_factors.columns:
            try:
                lag_df = self.calculate_lagged(
                    portfolio_returns,
                    macro_factors[factor_name]
                )
                
                if lag_df.empty:
                    continue
                
                idx_best = lag_df['corr'].abs().idxmax()
                best_row = lag_df.loc[idx_best]
                
                results.append({
                    'factor': factor_name,
                    'lag': int(best_row['lag']),
                    'corr': float(best_row['corr']),
                    't': float(best_row['t']),
                    'p': float(best_row['p']),
                    'n': int(best_row['n'])
                })
                
            except Exception as e:
                print(f"[Macro] Error con factor {factor_name}: {e}")
        
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('corr', key=abs, ascending=False)
        
        return df
    
    def calculate_matrix_with_lags(
        self,
        portfolio_returns: pd.Series,
        macro_factors: pd.DataFrame,
        lags: list = None
    ) -> Dict[int, pd.Series]:
        """
        Calcula matriz de correlaciones para diferentes lags fijos.
        
        Args:
            portfolio_returns: Retornos del portfolio
            macro_factors: DataFrame con factores macro
            lags: Lista de lags a evaluar (None = [0, 1, 5, 21, 63, 126])
            
        Returns:
            Dict {lag: Serie de correlaciones por factor}
            
        Uso:
        - Útil para visualizar cómo cambia la correlación con el lag
        - Cada lag es un "snapshot" de correlaciones simultáneas
        """
        if lags is None:
            lags = CORRELATION_LAGS_DEFAULT
        
        results = {}
        
        for lag in lags:
            corrs = {}
            for factor_name in macro_factors.columns:
                try:
                    x_lagged = macro_factors[factor_name].shift(lag)
                    df = pd.concat([portfolio_returns, x_lagged], axis=1).dropna()
                    
                    if len(df) >= self.min_obs:
                        corrs[factor_name] = df.iloc[:, 0].corr(df.iloc[:, 1])
                        
                except Exception:
                    corrs[factor_name] = np.nan
            
            results[lag] = pd.Series(corrs)
        
        return results
    
    def calculate_rolling(
        self,
        portfolio_returns: pd.Series,
        macro_factor: pd.Series,
        window: int = 252,
        min_periods: int = None
    ) -> pd.Series:
        """
        Calcula correlación móvil entre dos series.
        
        Args:
            portfolio_returns: Retornos del portfolio
            macro_factor: Factor macro individual
            window: Ventana móvil (default: 252 días = 1 año)
            min_periods: Mínimo de observaciones (None = window // 2)
            
        Returns:
            Serie temporal de correlaciones móviles
            
        Uso:
        - Detecta cambios en la relación portfolio-factor a lo largo del tiempo
        - Útil para identificar régimenes cambiantes
        """
        if min_periods is None:
            min_periods = window // 2

        df = pd.concat([portfolio_returns, macro_factor], axis=1).dropna()
        rolling_corr = df.iloc[:, 0].rolling(
            window=window,
            min_periods=min_periods
        ).corr(df.iloc[:, 1])
        
        return rolling_corr