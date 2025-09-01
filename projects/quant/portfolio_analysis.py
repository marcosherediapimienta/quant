"""
Módulo de análisis estadístico y de riesgo de activos financieros y carteras.

Este módulo proporciona funciones para el análisis cuantitativo de activos financieros,
incluyendo cálculo de retornos, métricas de riesgo, ratios de performance y análisis
estadístico usando datos de precios históricos.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import jarque_bera, normaltest
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf
import warnings
from typing import Union, List, Dict, Tuple, Optional
import sys
import os

# Importar DataManager - ajustar el path según la estructura del proyecto
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
from data_manager import DataManager

warnings.filterwarnings('ignore')


# ============================================================================
# 1. CÁLCULO DE RETORNOS
# ============================================================================

def calculate_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula retornos simples a partir de precios.
    
    Los retornos simples se calculan como: R_t = (P_t - P_{t-1}) / P_{t-1}
    
    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame con precios históricos, donde las columnas son activos 
        y las filas son fechas.
    
    Returns
    -------
    pd.DataFrame
        DataFrame con retornos simples alineados por fechas.
        
    Raises
    ------
    ValueError
        Si el input está vacío o no contiene datos válidos.
        
    Examples
    --------
    >>> prices = pd.DataFrame({'AAPL': [100, 105, 103], 'MSFT': [200, 210, 205]})
    >>> returns = calculate_simple_returns(prices)
    >>> returns.iloc[0, 0]  # Primer retorno de AAPL
    0.05
    """
    if prices.empty:
        raise ValueError("El DataFrame de precios está vacío")
    
    if prices.isna().all().all():
        raise ValueError("El DataFrame de precios no contiene datos válidos")
    
    # Calcular retornos simples
    returns = prices.pct_change()
    
    # Eliminar la primera fila (que será NaN)
    returns = returns.dropna(how='all')
    
    return returns


def calculate_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula retornos logarítmicos a partir de precios.
    
    Los retornos logarítmicos se calculan como: r_t = ln(P_t / P_{t-1})
    
    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame con precios históricos, donde las columnas son activos 
        y las filas son fechas.
    
    Returns
    -------
    pd.DataFrame
        DataFrame con retornos logarítmicos alineados por fechas.
        
    Raises
    ------
    ValueError
        Si el input está vacío o no contiene datos válidos.
        
    Examples
    --------
    >>> prices = pd.DataFrame({'AAPL': [100, 105, 103], 'MSFT': [200, 210, 205]})
    >>> log_returns = calculate_log_returns(prices)
    >>> log_returns.iloc[0, 0]  # Primer retorno logarítmico de AAPL
    0.04879...
    """
    if prices.empty:
        raise ValueError("El DataFrame de precios está vacío")
    
    if prices.isna().all().all():
        raise ValueError("El DataFrame de precios no contiene datos válidos")
    
    # Calcular retornos logarítmicos
    log_returns = np.log(prices / prices.shift(1))
    
    # Eliminar la primera fila (que será NaN)
    log_returns = log_returns.dropna(how='all')
    
    return log_returns


# ============================================================================
# 2. ESTADÍSTICA DESCRIPTIVA
# ============================================================================

def descriptive_stats(returns: pd.Series) -> pd.Series:
    """
    Calcula estadísticas descriptivas de una serie de retornos.
    
    Parameters
    ----------
    returns : pd.Series
        Serie de retornos de un activo.
    
    Returns
    -------
    pd.Series
        Serie con estadísticas: media, mediana, std, skewness, kurtosis.
        
    Raises
    ------
    ValueError
        Si la serie está vacía o no contiene datos válidos.
        
    Examples
    --------
    >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
    >>> stats = descriptive_stats(returns)
    >>> stats['media']
    0.006
    """
    if returns.empty:
        raise ValueError("La serie de retornos está vacía")
    
    # Eliminar NaN
    clean_returns = returns.dropna()
    
    if len(clean_returns) == 0:
        raise ValueError("La serie de retornos no contiene datos válidos")
    
    stats_dict = {
        'media': clean_returns.mean(),
        'mediana': clean_returns.median(),
        'std': clean_returns.std(),
        'skewness': stats.skew(clean_returns),
        'kurtosis': stats.kurtosis(clean_returns, fisher=True)  # Excess kurtosis
    }
    
    return pd.Series(stats_dict)


def calculate_percentiles(returns: pd.DataFrame, 
                         percentiles: List[float] = [1, 5, 25, 50, 75, 95, 99]) -> pd.DataFrame:
    """
    Calcula percentiles de retornos para múltiples activos.
    
    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame con retornos de múltiples activos.
    percentiles : List[float], default [1, 5, 25, 50, 75, 95, 99]
        Lista de percentiles a calcular.
    
    Returns
    -------
    pd.DataFrame
        DataFrame con percentiles por activo.
        
    Raises
    ------
    ValueError
        Si el DataFrame está vacío o no contiene datos válidos.
        
    Examples
    --------
    >>> returns = pd.DataFrame({'AAPL': [0.01, -0.02, 0.03], 'MSFT': [-0.01, 0.02, -0.01]})
    >>> percs = calculate_percentiles(returns)
    >>> percs.loc[50, 'AAPL']  # Mediana de AAPL
    0.01
    """
    if returns.empty:
        raise ValueError("El DataFrame de retornos está vacío")
    
    if returns.isna().all().all():
        raise ValueError("El DataFrame de retornos no contiene datos válidos")
    
    # Calcular percentiles para cada activo
    percentile_data = {}
    
    for col in returns.columns:
        clean_returns = returns[col].dropna()
        if len(clean_returns) > 0:
            percentile_data[col] = np.percentile(clean_returns, percentiles)
        else:
            percentile_data[col] = [np.nan] * len(percentiles)
    
    return pd.DataFrame(percentile_data, index=percentiles)


# ============================================================================
# 3. VOLATILIDAD
# ============================================================================

def historical_volatility(returns: pd.Series, window: int = 252, annualize: bool = True) -> float:
    """
    Calcula la volatilidad histórica de una serie de retornos.
    
    Parameters
    ----------
    returns : pd.Series
        Serie de retornos de un activo.
    window : int, default 252
        Número de observaciones para el cálculo (252 días bursátiles = 1 año).
    annualize : bool, default True
        Si anualizar la volatilidad (multiplicar por sqrt(252)).
    
    Returns
    -------
    float
        Volatilidad histórica.
        
    Raises
    ------
    ValueError
        Si la serie está vacía o no tiene suficientes datos.
        
    Examples
    --------
    >>> returns = pd.Series(np.random.normal(0, 0.02, 300))
    >>> vol = historical_volatility(returns)
    >>> vol > 0
    True
    """
    if returns.empty:
        raise ValueError("La serie de retornos está vacía")
    
    clean_returns = returns.dropna()
    
    if len(clean_returns) < 2:
        raise ValueError("Se necesitan al menos 2 observaciones válidas")
    
    # Tomar las últimas 'window' observaciones
    recent_returns = clean_returns.tail(window)
    
    # Calcular volatilidad
    volatility = recent_returns.std()
    
    # Anualizar si se solicita
    if annualize:
        volatility *= np.sqrt(252)
    
    return volatility


def rolling_volatility(returns: pd.Series, window: int = 30) -> pd.Series:
    """
    Calcula la volatilidad móvil de una serie de retornos.
    
    Parameters
    ----------
    returns : pd.Series
        Serie de retornos de un activo.
    window : int, default 30
        Ventana móvil para el cálculo de volatilidad.
    
    Returns
    -------
    pd.Series
        Serie con volatilidad móvil.
        
    Raises
    ------
    ValueError
        Si la serie está vacía.
        
    Examples
    --------
    >>> returns = pd.Series(np.random.normal(0, 0.02, 100))
    >>> rolling_vol = rolling_volatility(returns, window=20)
    >>> len(rolling_vol.dropna()) == len(returns) - 19
    True
    """
    if returns.empty:
        raise ValueError("La serie de retornos está vacía")
    
    # Calcular volatilidad móvil
    rolling_vol = returns.rolling(window=window).std()
    
    return rolling_vol


# ============================================================================
# 4. RISK-FREE RATE
# ============================================================================

def get_risk_free_series(dm: DataManager, start_date: str, end_date: str, 
                        period: str = "daily") -> pd.Series:
    """
    Descarga y procesa la tasa libre de riesgo (3M T-Bill).
    
    Parameters
    ----------
    dm : DataManager
        Instancia del DataManager para descargar datos (puede ser None).
    start_date : str
        Fecha de inicio en formato 'YYYY-MM-DD'.
    end_date : str
        Fecha de fin en formato 'YYYY-MM-DD'.
    period : str, default "daily"
        Periodicidad: "daily", "weekly", "monthly".
    
    Returns
    -------
    pd.Series
        Serie con tasa libre de riesgo convertida a retorno por periodo.
        
    Raises
    ------
    ValueError
        Si no se pueden descargar los datos o las fechas son inválidas.
        
    Examples
    --------
    >>> dm = DataManager()
    >>> rf = get_risk_free_series(dm, "2020-01-01", "2021-01-01")
    >>> rf.mean() > 0
    True
    """
    try:
        # Lista de símbolos alternativos para tasa libre de riesgo
        rf_symbols = ['^IRX', '^TNX', 'DGS3MO']
        rf_data = None
        
        # Intentar descargar con diferentes símbolos
        for symbol in rf_symbols:
            try:
                # Usar yfinance directamente para evitar problemas con DataManager
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                
                if not data.empty and 'Close' in data.columns:
                    rf_data = data['Close']
                    print(f"✅ Tasa libre de riesgo descargada usando {symbol}")
                    break
                    
            except Exception as e:
                print(f"⚠️ Error con {symbol}: {str(e)}")
                continue
        
        # Si falló la descarga directa, intentar con DataManager
        if rf_data is None and dm is not None:
            try:
                print("🔄 Intentando descarga con DataManager...")
                dm_data = dm.download_market_data(['^IRX'], start_date=start_date, end_date=end_date)
                if not dm_data.empty and '^IRX' in dm_data.columns:
                    rf_data = dm_data['^IRX']
                    print("✅ Tasa libre de riesgo descargada con DataManager")
            except Exception as e:
                print(f"⚠️ Error con DataManager: {str(e)}")
        
        # Si todo falló, usar tasa fija
        if rf_data is None:
            print("⚠️ No se pudo descargar tasa libre de riesgo, usando tasa fija del 2% anual")
            # Crear serie con fechas del periodo solicitado
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            fixed_rate = 0.02  # 2% anual
            
            if period == "daily":
                rf_data = pd.Series(fixed_rate / 252, index=date_range)
            elif period == "weekly":
                rf_data = pd.Series(fixed_rate / 52, index=date_range)
            elif period == "monthly":
                rf_data = pd.Series(fixed_rate / 12, index=date_range)
            else:
                raise ValueError(f"Periodo no soportado: {period}")
            
            return rf_data
        
        # Procesar datos descargados
        if rf_data is None or rf_data.empty:
            raise ValueError("No se pudieron obtener datos de tasa libre de riesgo")
        
        # Obtener serie de tasas (en % anual)
        rf_annual = rf_data.copy()
        
        # Convertir de % a decimal y luego a retorno por periodo
        rf_annual = rf_annual / 100  # De % a decimal
        
        if period == "daily":
            # Convertir de tasa anual a retorno diario
            rf_period = rf_annual / 252
        elif period == "weekly":
            # Convertir de tasa anual a retorno semanal
            rf_period = rf_annual / 52
        elif period == "monthly":
            # Convertir de tasa anual a retorno mensual
            rf_period = rf_annual / 12
        else:
            raise ValueError(f"Periodo no soportado: {period}")
        
        # Forward fill para alinear con fechas de trading
        rf_period = rf_period.ffill()
        
        # Asegurar que el índice sea de tipo DatetimeIndex para compatibilidad
        if not isinstance(rf_period.index, pd.DatetimeIndex):
            rf_period.index = pd.to_datetime(rf_period.index)
        
        return rf_period
        
    except Exception as e:
        # Fallback: crear serie con tasa fija
        print(f"⚠️ Error en descarga: {str(e)}. Usando tasa fija del 2% anual")
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        fixed_rate = 0.02  # 2% anual
        
        if period == "daily":
            rf_period = pd.Series(fixed_rate / 252, index=date_range)
        elif period == "weekly":
            rf_period = pd.Series(fixed_rate / 52, index=date_range)
        elif period == "monthly":
            rf_period = pd.Series(fixed_rate / 12, index=date_range)
        else:
            rf_period = pd.Series(fixed_rate / 252, index=date_range)  # Default a diario
        
        return rf_period


# ============================================================================
# FUNCIÓN AUXILIAR PARA ALINEACIÓN DE FECHAS
# ============================================================================

def _align_series_by_date(series1: pd.Series, series2: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Función auxiliar para alinear dos series por fechas.
    
    Parameters
    ----------
    series1 : pd.Series
        Primera serie.
    series2 : pd.Series
        Segunda serie.
    
    Returns
    -------
    Tuple[pd.Series, pd.Series]
        Tupla con las series alineadas.
    """
    # Convertir índices a DatetimeIndex si no lo son y normalizar zonas horarias
    if not isinstance(series1.index, pd.DatetimeIndex):
        series1.index = pd.to_datetime(series1.index)
    if not isinstance(series2.index, pd.DatetimeIndex):
        series2.index = pd.to_datetime(series2.index)
    
    # Normalizar zonas horarias - convertir ambas a UTC o eliminar zona horaria
    if series1.index.tz is not None:
        series1.index = series1.index.tz_convert('UTC').tz_localize(None)
    if series2.index.tz is not None:
        series2.index = series2.index.tz_convert('UTC').tz_localize(None)
    
    # Alinear series por fechas
    aligned_data = pd.concat([series1, series2], axis=1, join='inner').dropna()
    
    # Si no hay fechas comunes, intentar alineación alternativa
    if aligned_data.empty:
        # Reindexar segunda serie a las fechas de la primera
        series2_aligned = series2.reindex(series1.index, method='ffill')
        aligned_data = pd.concat([series1, series2_aligned], axis=1, join='inner').dropna()
        
        if aligned_data.empty:
            raise ValueError("No se pueden alinear las series por fechas")
    
    return aligned_data.iloc[:, 0], aligned_data.iloc[:, 1]


# ============================================================================
# 5. RATIOS DE PERFORMANCE
# ============================================================================

def sharpe_ratio(returns: pd.Series, rf: pd.Series) -> float:
    """
    Calcula el ratio de Sharpe.
    
    Sharpe Ratio = (E[R_p] - E[R_f]) / σ_p
    
    Parameters
    ----------
    returns : pd.Series
        Serie de retornos del activo/cartera.
    rf : pd.Series
        Serie de tasa libre de riesgo.
    
    Returns
    -------
    float
        Ratio de Sharpe.
        
    Raises
    ------
    ValueError
        Si las series están vacías o no se pueden alinear.
        
    Examples
    --------
    >>> returns = pd.Series([0.01, 0.02, -0.01, 0.03])
    >>> rf = pd.Series([0.001, 0.001, 0.001, 0.001])
    >>> sr = sharpe_ratio(returns, rf)
    >>> isinstance(sr, float)
    True
    """
    if returns.empty or rf.empty:
        raise ValueError("Las series de retornos o tasa libre de riesgo están vacías")
    
    # Usar función auxiliar para alinear series
    try:
        aligned_returns, aligned_rf = _align_series_by_date(returns, rf)
    except ValueError as e:
        raise ValueError(f"Error alineando series para Sharpe ratio: {str(e)}")
    
    # Calcular exceso de retorno
    excess_returns = aligned_returns - aligned_rf
    
    # Calcular Sharpe ratio
    if excess_returns.std() == 0:
        return 0.0
    
    sharpe = excess_returns.mean() / excess_returns.std()
    
    # Anualizar (multiplicar por sqrt(252) para retornos diarios)
    return sharpe * np.sqrt(252)


def sortino_ratio(returns: pd.Series, rf: pd.Series) -> float:
    """
    Calcula el ratio de Sortino.
    
    Sortino Ratio = (E[R_p] - E[R_f]) / σ_downside
    
    Parameters
    ----------
    returns : pd.Series
        Serie de retornos del activo/cartera.
    rf : pd.Series
        Serie de tasa libre de riesgo.
    
    Returns
    -------
    float
        Ratio de Sortino.
        
    Raises
    ------
    ValueError
        Si las series están vacías o no se pueden alinear.
        
    Examples
    --------
    >>> returns = pd.Series([0.01, 0.02, -0.01, 0.03])
    >>> rf = pd.Series([0.001, 0.001, 0.001, 0.001])
    >>> sr = sortino_ratio(returns, rf)
    >>> isinstance(sr, float)
    True
    """
    if returns.empty or rf.empty:
        raise ValueError("Las series de retornos o tasa libre de riesgo están vacías")
    
    # Usar función auxiliar para alinear series
    try:
        aligned_returns, aligned_rf = _align_series_by_date(returns, rf)
    except ValueError as e:
        raise ValueError(f"Error alineando series para Sortino ratio: {str(e)}")
    
    # Calcular exceso de retorno
    excess_returns = aligned_returns - aligned_rf
    
    # Calcular downside deviation (solo retornos negativos)
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf  # Sin riesgo downside
    
    downside_deviation = downside_returns.std()
    
    if downside_deviation == 0:
        return 0.0
    
    sortino = excess_returns.mean() / downside_deviation
    
    # Anualizar
    return sortino * np.sqrt(252)


def information_ratio(returns: pd.Series, benchmark: pd.Series, rf: pd.Series) -> float:
    """
    Calcula el ratio de información.
    
    Information Ratio = (E[R_p] - E[R_b]) / σ_tracking_error
    
    Parameters
    ----------
    returns : pd.Series
        Serie de retornos del activo/cartera.
    benchmark : pd.Series
        Serie de retornos del benchmark.
    rf : pd.Series
        Serie de tasa libre de riesgo.
    
    Returns
    -------
    float
        Ratio de información.
        
    Raises
    ------
    ValueError
        Si las series están vacías o no se pueden alinear.
        
    Examples
    --------
    >>> returns = pd.Series([0.01, 0.02, -0.01, 0.03])
    >>> benchmark = pd.Series([0.005, 0.015, -0.005, 0.025])
    >>> rf = pd.Series([0.001, 0.001, 0.001, 0.001])
    >>> ir = information_ratio(returns, benchmark, rf)
    >>> isinstance(ir, float)
    True
    """
    if returns.empty or benchmark.empty or rf.empty:
        raise ValueError("Alguna de las series está vacía")
    
    # Alinear todas las series usando la función auxiliar
    try:
        # Primero alinear returns y benchmark
        aligned_returns, aligned_benchmark = _align_series_by_date(returns, benchmark)
        # Luego alinear con rf
        aligned_returns, aligned_rf = _align_series_by_date(aligned_returns, rf)
        aligned_benchmark, _ = _align_series_by_date(aligned_benchmark, rf)
    except ValueError as e:
        raise ValueError(f"Error alineando series para Information ratio: {str(e)}")
    
    # Calcular excesos de retorno
    excess_returns = aligned_returns - aligned_rf
    excess_benchmark = aligned_benchmark - aligned_rf
    
    # Calcular tracking error (diferencia de excesos)
    tracking_error = excess_returns - excess_benchmark
    
    if tracking_error.std() == 0:
        return 0.0
    
    information_ratio_value = tracking_error.mean() / tracking_error.std()
    
    # Anualizar
    return information_ratio_value * np.sqrt(252)


# ============================================================================
# 6. RIESGO EXTREMO
# ============================================================================

def calculate_var(returns: pd.Series, level: float = 0.05, horizon_days: int = 1) -> float:
    """
    Calcula el Value at Risk (VaR) histórico.
    
    Parameters
    ----------
    returns : pd.Series
        Serie de retornos.
    level : float, default 0.05
        Nivel de confianza (0.05 = 95% de confianza).
    horizon_days : int, default 1
        Horizonte en días para el VaR. Si >1, agrega retornos a ese horizonte
        usando composición de retornos simples.
    
    Returns
    -------
    float
        VaR al nivel especificado (valor positivo representa pérdida).
        
    Raises
    ------
    ValueError
        Si la serie está vacía o el nivel es inválido.
        
    Examples
    --------
    >>> returns = pd.Series(np.random.normal(0, 0.02, 1000))
    >>> var_5 = calculate_var(returns, 0.05)
    >>> var_5 > 0  # VaR debe ser positivo (representa pérdida)
    True
    """
    if returns.empty:
        raise ValueError("La serie de retornos está vacía")
    
    if not (0 < level < 1):
        raise ValueError("El nivel debe estar entre 0 y 1")
    
    clean_returns = returns.dropna()
    
    # Agregar a horizonte si aplica (asumiendo retornos simples)
    if horizon_days > 1:
        if len(clean_returns) < horizon_days:
            raise ValueError("No hay suficientes datos para el horizonte solicitado")
        # Composición de retornos simples en ventana rodante
        clean_returns = (1 + clean_returns).rolling(window=horizon_days).apply(lambda x: np.prod(x) - 1, raw=False)
        clean_returns = clean_returns.dropna()
    
    if len(clean_returns) == 0:
        raise ValueError("No hay datos válidos en la serie")
    
    # Calcular VaR como percentil (negativo para representar pérdida)
    var_value = -np.percentile(clean_returns, level * 100)
    
    return var_value


def calculate_cvar(returns: pd.Series, level: float = 0.05, horizon_days: int = 1) -> float:
    """
    Calcula el Conditional Value at Risk (CVaR) o Expected Shortfall.
    
    Parameters
    ----------
    returns : pd.Series
        Serie de retornos.
    level : float, default 0.05
        Nivel de confianza (0.05 = 95% de confianza).
    horizon_days : int, default 1
        Horizonte en días para el CVaR. Si >1, agrega retornos a ese horizonte
        usando composición de retornos simples.
    
    Returns
    -------
    float
        CVaR al nivel especificado (valor positivo representa pérdida).
        
    Raises
    ------
    ValueError
        Si la serie está vacía o el nivel es inválido.
        
    Examples
    --------
    >>> returns = pd.Series(np.random.normal(0, 0.02, 1000))
    >>> cvar_5 = calculate_cvar(returns, 0.05)
    >>> cvar_5 > 0  # CVaR debe ser positivo (representa pérdida)
    True
    """
    if returns.empty:
        raise ValueError("La serie de retornos está vacía")
    
    if not (0 < level < 1):
        raise ValueError("El nivel debe estar entre 0 y 1")
    
    clean_returns = returns.dropna()
    
    # Agregar a horizonte si aplica (asumiendo retornos simples)
    if horizon_days > 1:
        if len(clean_returns) < horizon_days:
            raise ValueError("No hay suficientes datos para el horizonte solicitado")
        clean_returns = (1 + clean_returns).rolling(window=horizon_days).apply(lambda x: np.prod(x) - 1, raw=False)
        clean_returns = clean_returns.dropna()
    
    if len(clean_returns) == 0:
        raise ValueError("No hay datos válidos en la serie")
    
    # Calcular VaR primero
    var_value = -np.percentile(clean_returns, level * 100)
    
    # Calcular CVaR como la media de las pérdidas que exceden el VaR
    tail_losses = clean_returns[clean_returns <= -var_value]
    
    if len(tail_losses) == 0:
        return var_value  # Si no hay pérdidas peores que VaR, CVaR = VaR
    
    cvar_value = -tail_losses.mean()
    
    return cvar_value


def max_drawdown(returns: pd.Series) -> float:
    """
    Calcula el máximo drawdown de una serie de retornos.
    
    Parameters
    ----------
    returns : pd.Series
        Serie de retornos.
    
    Returns
    -------
    float
        Máximo drawdown (valor positivo representa pérdida).
        
    Raises
    ------
    ValueError
        Si la serie está vacía.
        
    Examples
    --------
    >>> returns = pd.Series([0.1, -0.05, -0.03, 0.02, 0.01])
    >>> mdd = max_drawdown(returns)
    >>> mdd >= 0  # MDD debe ser positivo
    True
    """
    if returns.empty:
        raise ValueError("La serie de retornos está vacía")
    
    clean_returns = returns.dropna()
    
    if len(clean_returns) == 0:
        raise ValueError("No hay datos válidos en la serie")
    
    # Calcular valor acumulado de la inversión (empezando en 1)
    cumulative_value = (1 + clean_returns).cumprod()
    
    # Calcular peak running (máximo hasta cada punto)
    peak = cumulative_value.expanding().max()
    
    # Calcular drawdown
    drawdown = (cumulative_value - peak) / peak
    
    # Máximo drawdown (valor absoluto)
    max_dd = abs(drawdown.min())
    
    return max_dd


# ============================================================================
# 7. BETA Y ALPHA FRENTE A BENCHMARK
# ============================================================================

def calculate_beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, float]:
    """
    Calcula Alpha, Beta y R² usando regresión OLS.
    
    Modelo: R_p - R_f = α + β(R_m - R_f) + ε
    
    Parameters
    ----------
    portfolio_returns : pd.Series
        Serie de retornos del portafolio/activo.
    benchmark_returns : pd.Series
        Serie de retornos del benchmark.
    
    Returns
    -------
    Dict[str, float]
        Diccionario con 'alpha', 'beta', y 'r2'.
        
    Raises
    ------
    ValueError
        Si las series están vacías o no se pueden alinear.
        
    Examples
    --------
    >>> portfolio = pd.Series([0.01, 0.02, -0.01, 0.03])
    >>> benchmark = pd.Series([0.005, 0.015, -0.005, 0.025])
    >>> result = calculate_beta(portfolio, benchmark)
    >>> 'alpha' in result and 'beta' in result and 'r2' in result
    True
    """
    if portfolio_returns.empty or benchmark_returns.empty:
        raise ValueError("Las series de retornos están vacías")
    
    # Alinear series por fechas
    aligned_data = pd.concat([portfolio_returns, benchmark_returns], axis=1, join='inner').dropna()
    
    if aligned_data.empty:
        raise ValueError("No hay fechas comunes entre las series")
    
    if len(aligned_data) < 2:
        raise ValueError("Se necesitan al menos 2 observaciones para la regresión")
    
    y = aligned_data.iloc[:, 0]  # Retornos del portafolio
    x = aligned_data.iloc[:, 1]  # Retornos del benchmark
    
    # Agregar constante para la regresión
    X = sm.add_constant(x)
    
    try:
        # Ajustar modelo OLS
        model = sm.OLS(y, X).fit()
        
        # Extraer resultados
        alpha = model.params[0]  # Intercepto
        beta = model.params[1]   # Pendiente
        r2 = model.rsquared      # R cuadrado
        
        # Anualizar alpha (asumiendo retornos diarios)
        alpha_annualized = alpha * 252
        
        return {
            'alpha': alpha_annualized,
            'beta': beta,
            'r2': r2
        }
        
    except Exception as e:
        raise ValueError(f"Error en la regresión OLS: {str(e)}")


# ============================================================================
# 8. DISTRIBUCIÓN Y AUTOCORRELACIÓN
# ============================================================================

def plot_return_distribution(returns: pd.Series, symbol: str = "Asset", 
                           save_plot: bool = False, show_plot: bool = True) -> Optional[str]:
    """
    Genera histograma y QQ-plot de retornos.
    
    Parameters
    ----------
    returns : pd.Series
        Serie de retornos.
    symbol : str, default "Asset"
        Nombre del activo para el título.
    save_plot : bool, default False
        Si guardar el gráfico.
    show_plot : bool, default True
        Si mostrar el gráfico.
    
    Returns
    -------
    Optional[str]
        Ruta del archivo si se guardó, None en caso contrario.
        
    Examples
    --------
    >>> returns = pd.Series(np.random.normal(0, 0.02, 1000))
    >>> plot_return_distribution(returns, show_plot=False)
    """
    clean_returns = returns.dropna()
    
    if len(clean_returns) == 0:
        print("No hay datos válidos para graficar")
        return None
    
    # Crear figura con 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histograma
    ax1.hist(clean_returns, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title(f'Distribución de Retornos - {symbol}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Retornos')
    ax1.set_ylabel('Densidad')
    ax1.grid(True, alpha=0.3)
    
    # Overlay normal distribution
    x = np.linspace(clean_returns.min(), clean_returns.max(), 100)
    normal_dist = stats.norm.pdf(x, clean_returns.mean(), clean_returns.std())
    ax1.plot(x, normal_dist, 'r-', linewidth=2, label='Normal Teórica')
    ax1.legend()
    
    # QQ-plot
    stats.probplot(clean_returns, dist="norm", plot=ax2)
    ax2.set_title(f'Q-Q Plot vs Normal - {symbol}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Estadísticas en el gráfico
    mean_ret = clean_returns.mean()
    std_ret = clean_returns.std()
    skew_ret = stats.skew(clean_returns)
    kurt_ret = stats.kurtosis(clean_returns, fisher=True)
    
    # Test de normalidad
    jb_stat, jb_pvalue = jarque_bera(clean_returns)
    
    stats_text = f'Media: {mean_ret:.4f}\nStd: {std_ret:.4f}\nSkewness: {skew_ret:.4f}\nKurtosis: {kurt_ret:.4f}\nJarque-Bera: {jb_stat:.2f} (p={jb_pvalue:.4f})'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
    
    plt.tight_layout()
    
    # Guardar si se solicita
    filepath = None
    if save_plot:
        filepath = f"{symbol}_distribution_analysis.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return filepath


def autocorrelation_analysis(returns: pd.Series, lags: int = 20, symbol: str = "Asset",
                           save_plot: bool = False, show_plot: bool = True) -> Dict[str, any]:
    """
    Análisis de autocorrelación de retornos y retornos al cuadrado.
    
    Parameters
    ----------
    returns : pd.Series
        Serie de retornos.
    lags : int, default 20
        Número de lags para analizar.
    symbol : str, default "Asset"
        Nombre del activo.
    save_plot : bool, default False
        Si guardar el gráfico.
    show_plot : bool, default True
        Si mostrar el gráfico.
    
    Returns
    -------
    Dict[str, any]
        Resultados del análisis de autocorrelación.
        
    Examples
    --------
    >>> returns = pd.Series(np.random.normal(0, 0.02, 1000))
    >>> result = autocorrelation_analysis(returns, show_plot=False)
    >>> 'ljung_box_returns' in result
    True
    """
    clean_returns = returns.dropna()
    
    if len(clean_returns) == 0:
        return {"error": "No hay datos válidos"}
    
    # Calcular autocorrelaciones
    acf_returns = acf(clean_returns, nlags=lags, fft=False)
    acf_squared = acf(clean_returns**2, nlags=lags, fft=False)
    
    # Tests de autocorrelación
    ljung_box_returns = acorr_ljungbox(clean_returns, lags=lags, return_df=True)
    ljung_box_squared = acorr_ljungbox(clean_returns**2, lags=lags, return_df=True)
    
    # Crear gráficos
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # ACF de retornos
    ax1.bar(range(len(acf_returns)), acf_returns, alpha=0.7, color='blue')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.axhline(y=1.96/np.sqrt(len(clean_returns)), color='red', linestyle='--', alpha=0.7)
    ax1.axhline(y=-1.96/np.sqrt(len(clean_returns)), color='red', linestyle='--', alpha=0.7)
    ax1.set_title(f'Autocorrelación de Retornos - {symbol}')
    ax1.set_xlabel('Lags')
    ax1.set_ylabel('ACF')
    ax1.grid(True, alpha=0.3)
    
    # ACF de retornos al cuadrado
    ax2.bar(range(len(acf_squared)), acf_squared, alpha=0.7, color='green')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.axhline(y=1.96/np.sqrt(len(clean_returns)), color='red', linestyle='--', alpha=0.7)
    ax2.axhline(y=-1.96/np.sqrt(len(clean_returns)), color='red', linestyle='--', alpha=0.7)
    ax2.set_title(f'Autocorrelación de Retornos² - {symbol}')
    ax2.set_xlabel('Lags')
    ax2.set_ylabel('ACF')
    ax2.grid(True, alpha=0.3)
    
    # P-values Ljung-Box retornos
    ax3.plot(ljung_box_returns.index, ljung_box_returns['lb_pvalue'], 'o-', color='blue')
    ax3.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α = 0.05')
    ax3.set_title(f'Ljung-Box Test p-values (Retornos) - {symbol}')
    ax3.set_xlabel('Lags')
    ax3.set_ylabel('p-value')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # P-values Ljung-Box retornos²
    ax4.plot(ljung_box_squared.index, ljung_box_squared['lb_pvalue'], 'o-', color='green')
    ax4.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α = 0.05')
    ax4.set_title(f'Ljung-Box Test p-values (Retornos²) - {symbol}')
    ax4.set_xlabel('Lags')
    ax4.set_ylabel('p-value')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar si se solicita
    filepath = None
    if save_plot:
        filepath = f"{symbol}_autocorrelation_analysis.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    # Preparar resultados
    results = {
        'acf_returns': acf_returns,
        'acf_squared': acf_squared,
        'ljung_box_returns': ljung_box_returns,
        'ljung_box_squared': ljung_box_squared,
        'filepath': filepath
    }
    
    return results
