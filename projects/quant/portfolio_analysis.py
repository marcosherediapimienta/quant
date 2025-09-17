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
from scipy.stats import jarque_bera
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf
import warnings
from typing import Union, List, Dict, Tuple, Optional, Any, Literal
import sys
import os
import logging

# Configurar logging
logger = logging.getLogger(__name__)

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
                         percentiles: Optional[List[float]] = None) -> pd.DataFrame:
    """
    Calcula percentiles de retornos para múltiples activos.
    
    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame con retornos de múltiples activos.
    percentiles : Optional[List[float]], default None
        Lista de percentiles a calcular. Si es None, usa [1, 5, 25, 50, 75, 95, 99].
    
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
    if percentiles is None:
        percentiles = [1, 5, 25, 50, 75, 95, 99]
    
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

def historical_volatility(returns: pd.Series, window: int = 252, annualize: bool = True, 
                         periods_per_year: int = 252) -> float:
    """
    Calcula la volatilidad histórica de una serie de retornos.
    
    Parameters
    ----------
    returns : pd.Series
        Serie de retornos de un activo.
    window : int, default 252
        Número de observaciones para el cálculo (252 días bursátiles = 1 año).
    annualize : bool, default True
        Si anualizar la volatilidad (multiplicar por sqrt(periods_per_year)).
    periods_per_year : int, default 252
        Número de periodos por año para anualización.
    
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
    >>> vol_monthly = historical_volatility(returns, annualize=True, periods_per_year=12)
    >>> vol_monthly > vol
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
        volatility *= np.sqrt(periods_per_year)
    
    return volatility


def rolling_volatility(returns: pd.Series, window: int = 30, annualize: bool = False, 
                      periods_per_year: int = 252) -> pd.Series:
    """
    Calcula la volatilidad móvil de una serie de retornos.
    
    Parameters
    ----------
    returns : pd.Series
        Serie de retornos de un activo.
    window : int, default 30
        Ventana móvil para el cálculo de volatilidad.
    annualize : bool, default False
        Si anualizar la volatilidad (multiplicar por sqrt(periods_per_year)).
    periods_per_year : int, default 252
        Número de periodos por año para anualización.
    
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
    >>> rolling_vol_annual = rolling_volatility(returns, window=20, annualize=True, periods_per_year=12)
    >>> rolling_vol_annual.mean() > rolling_vol.mean()
    True
    """
    if returns.empty:
        raise ValueError("La serie de retornos está vacía")
    
    # Calcular volatilidad móvil
    rolling_vol = returns.rolling(window=window).std()
    
    # Anualizar si se solicita
    if annualize:
        rolling_vol *= np.sqrt(periods_per_year)
    
    return rolling_vol


# ============================================================================
# 4. RISK-FREE RATE
# ============================================================================

def _annual_to_periodic(r_annual: pd.Series, period: str, periods_per_year: int = 252) -> pd.Series:
    """
    Convierte tasa anual a tasa periódica usando conversión compuesta.
    
    Parameters
    ----------
    r_annual : pd.Series
        Serie de tasas anuales (en decimal).
    period : str
        Periodo objetivo: "daily", "weekly", "monthly".
    periods_per_year : int, default 252
        Número de periodos por año para conversión diaria.
    
    Returns
    -------
    pd.Series
        Serie con tasas periódicas convertidas.
        
    Raises
    ------
    ValueError
        Si el periodo no es soportado.
    """
    if period == "daily":
        return (1 + r_annual)**(1/periods_per_year) - 1
    elif period == "weekly":
        return (1 + r_annual)**(1/52) - 1
    elif period == "monthly":
        return (1 + r_annual)**(1/12) - 1
    else:
        raise ValueError(f"Periodo no soportado: {period}")


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
                    logger.info(f"Tasa libre de riesgo descargada usando {symbol}")
                    break
                    
            except Exception as e:
                logger.warning(f"Error con {symbol}: {str(e)}")
                continue
        
        # Si falló la descarga directa, intentar con DataManager
        if rf_data is None and dm is not None:
            try:
                logger.info("Intentando descarga con DataManager...")
                dm_data = dm.download_market_data(['^IRX'], start_date=start_date, end_date=end_date)
                if not dm_data.empty and '^IRX' in dm_data.columns:
                    rf_data = dm_data['^IRX']
                    logger.info("Tasa libre de riesgo descargada con DataManager")
            except Exception as e:
                logger.warning(f"Error con DataManager: {str(e)}")
        
        # Si todo falló, usar tasa fija
        if rf_data is None:
            logger.warning("No se pudo descargar tasa libre de riesgo, usando tasa fija del 2% anual")
            # Crear serie con fechas del periodo solicitado
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            fixed_rate = 0.02  # 2% anual
            
            # Usar función auxiliar para conversión
            rf_data = pd.Series(fixed_rate, index=date_range)
            rf_data = _annual_to_periodic(rf_data, period)
            
            return rf_data
        
        # Procesar datos descargados
        if rf_data is None or rf_data.empty:
            raise ValueError("No se pudieron obtener datos de tasa libre de riesgo")
        
        # Obtener serie de tasas (en % anual)
        rf_annual = rf_data.copy()
        
        # Convertir de % a decimal y luego a retorno por periodo
        rf_annual = rf_annual / 100  # De % a decimal
        
        # Usar función auxiliar para conversión compuesta
        rf_period = _annual_to_periodic(rf_annual, period)
        
        # Forward fill para alinear con fechas de trading
        rf_period = rf_period.ffill()
        
        # Asegurar que el índice sea de tipo DatetimeIndex para compatibilidad
        if not isinstance(rf_period.index, pd.DatetimeIndex):
            rf_period.index = pd.to_datetime(rf_period.index)
        
        return rf_period
        
    except Exception as e:
        # Fallback: crear serie con tasa fija
        logger.warning(f"Error en descarga: {str(e)}. Usando tasa fija del 2% anual")
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
# FUNCIONES AUXILIARES PARA ALINEACIÓN Y RF
# ============================================================================

def create_aligned_risk_free_rate(returns_index: pd.DatetimeIndex, 
                                 annual_rate: float = 0.02,
                                 periods_per_year: int = 252) -> pd.Series:
    """
    Crea una serie de tasa libre de riesgo que se alinea perfectamente con el índice de retornos.
    
    Parameters
    ----------
    returns_index : pd.DatetimeIndex
        Índice de fechas de los retornos.
    annual_rate : float, default 0.02
        Tasa libre de riesgo anual (2% por defecto).
    periods_per_year : int, default 252
        Número de periodos por año.
    
    Returns
    -------
    pd.Series
        Serie de tasa libre de riesgo alineada con las fechas de retornos.
        
    Examples
    --------
    >>> returns = pd.Series([0.01, 0.02], index=pd.date_range('2020-01-01', periods=2))
    >>> rf = create_aligned_risk_free_rate(returns.index, annual_rate=0.025)
    >>> len(rf) == len(returns)
    True
    """
    # Normalizar el índice de fechas
    normalized_index = returns_index.copy()
    
    if not isinstance(normalized_index, pd.DatetimeIndex):
        normalized_index = pd.to_datetime(normalized_index)
    
    # Normalizar zona horaria
    if normalized_index.tz is not None:
        normalized_index = normalized_index.tz_convert('UTC').tz_localize(None)
    
    # Redondear a días completos
    normalized_index = normalized_index.normalize()
    
    # Calcular tasa por periodo
    rate_per_period = annual_rate / periods_per_year
    
    # Crear serie con tasa constante
    rf_series = pd.Series(rate_per_period, index=normalized_index)
    
    return rf_series


# ============================================================================
# FUNCIÓN AUXILIAR PARA ALINEACIÓN DE FECHAS
# ============================================================================

def _align_series_by_date(series1: pd.Series, series2: pd.Series, 
                         allow_ffill_rf: bool = True) -> Tuple[pd.Series, pd.Series]:
    """
    Función auxiliar para alinear dos series por fechas.
    
    IMPORTANTE: Por defecto permite forward fill para tasa libre de riesgo (allow_ffill_rf=True).
    Esto resuelve problemas comunes de alineación de fechas entre retornos y RF.
    
    Parameters
    ----------
    series1 : pd.Series
        Primera serie (generalmente retornos de activos).
    series2 : pd.Series
        Segunda serie (generalmente tasa libre de riesgo).
    allow_ffill_rf : bool, default True
        Si permitir forward fill para series de tasa libre de riesgo.
        Recomendado para resolver problemas de alineación.
    
    Returns
    -------
    Tuple[pd.Series, pd.Series]
        Tupla con las series alineadas.
        
    Raises
    ------
    ValueError
        Si no se pueden alinear las series por fechas.
    """
    # Convertir índices a DatetimeIndex si no lo son
    if not isinstance(series1.index, pd.DatetimeIndex):
        series1.index = pd.to_datetime(series1.index)
    if not isinstance(series2.index, pd.DatetimeIndex):
        series2.index = pd.to_datetime(series2.index)
    
    # Normalizar zonas horarias - convertir ambas a UTC o eliminar zona horaria
    if series1.index.tz is not None:
        series1.index = series1.index.tz_convert('UTC').tz_localize(None)
    if series2.index.tz is not None:
        series2.index = series2.index.tz_convert('UTC').tz_localize(None)
    
    # Redondear todas las fechas a días completos (00:00:00) para mejor alineación
    series1.index = series1.index.normalize()
    series2.index = series2.index.normalize()
    
    # Intentar alineación directa primero
    aligned_data = pd.concat([series1, series2], axis=1, join='inner').dropna()
    
    # Si no hay fechas comunes y se permite ffill
    if aligned_data.empty and allow_ffill_rf:
        # Usar forward fill para alinear la segunda serie con la primera
        series2_aligned = series2.reindex(series1.index, method='ffill')
        
        # Verificar que hay datos válidos después del forward fill
        if series2_aligned.isna().all():
            # Si todo es NaN, usar backward fill
            series2_aligned = series2.reindex(series1.index, method='bfill')
        
        # Crear DataFrame alineado
        aligned_data = pd.concat([series1, series2_aligned], axis=1, join='inner').dropna()
        
        if aligned_data.empty:
            raise ValueError("No se pueden alinear las series por fechas incluso con forward/backward fill")
    elif aligned_data.empty:
        raise ValueError("No hay fechas comunes entre las series. Considere usar reindex/ffill explícitamente.")
    
    return aligned_data.iloc[:, 0], aligned_data.iloc[:, 1]


# ============================================================================
# 5. RATIOS DE PERFORMANCE
# ============================================================================

def sharpe_ratio(returns: pd.Series, rf: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calcula el ratio de Sharpe.
    
    Sharpe Ratio = (E[R_p] - E[R_f]) / σ_p
    
    Parameters
    ----------
    returns : pd.Series
        Serie de retornos del activo/cartera.
    rf : pd.Series
        Serie de tasa libre de riesgo.
    periods_per_year : int, default 252
        Número de periodos por año para anualización.
    
    Returns
    -------
    float
        Ratio de Sharpe anualizado.
        
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
    >>> sr_monthly = sharpe_ratio(returns, rf, periods_per_year=12)
    >>> sr_monthly > sr
    True
    """
    if returns.empty or rf.empty:
        raise ValueError("Las series de retornos o tasa libre de riesgo están vacías")
    
    # Usar función auxiliar para alinear series (con forward fill habilitado por defecto)
    try:
        aligned_returns, aligned_rf = _align_series_by_date(returns, rf, allow_ffill_rf=True)
    except ValueError as e:
        raise ValueError(f"Error alineando series para Sharpe ratio: {str(e)}")
    
    # Calcular exceso de retorno
    excess_returns = aligned_returns - aligned_rf
    
    # Calcular Sharpe ratio
    if excess_returns.std() == 0:
        return 0.0
    
    sharpe = excess_returns.mean() / excess_returns.std()
    
    # Anualizar usando periods_per_year
    return sharpe * np.sqrt(periods_per_year)


def sortino_ratio(returns: pd.Series, rf: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calcula el ratio de Sortino usando la definición estándar.
    
    Sortino Ratio = (E[R_p] - E[R_f]) / σ_downside
    
    Donde σ_downside es la desviación estándar de los retornos por debajo de la tasa libre de riesgo.
    
    Parameters
    ----------
    returns : pd.Series
        Serie de retornos del activo/cartera.
    rf : pd.Series
        Serie de tasa libre de riesgo.
    periods_per_year : int, default 252
        Número de periodos por año para anualización.
    
    Returns
    -------
    float
        Ratio de Sortino anualizado. Retorna np.inf si no hay downside deviation.
        
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
    >>> # Caso sin downside deviation
    >>> returns_positive = pd.Series([0.01, 0.02, 0.03])
    >>> sr_no_downside = sortino_ratio(returns_positive, rf)
    >>> np.isinf(sr_no_downside)
    True
    """
    if returns.empty or rf.empty:
        raise ValueError("Las series de retornos o tasa libre de riesgo están vacías")
    
    # Usar función auxiliar para alinear series (con forward fill habilitado por defecto)
    try:
        aligned_returns, aligned_rf = _align_series_by_date(returns, rf, allow_ffill_rf=True)
    except ValueError as e:
        raise ValueError(f"Error alineando series para Sortino ratio: {str(e)}")
    
    # Calcular exceso de retorno
    excess = aligned_returns - aligned_rf
    
    # Calcular downside deviation (solo retornos negativos)
    downside = np.minimum(0, excess)
    downside_dev = np.sqrt(np.mean(np.square(downside)))
    
    # Si no hay downside deviation, retornar infinito
    if downside_dev == 0:
        return np.inf
    
    # Calcular ratio de Sortino
    sortino = excess.mean() / downside_dev
    
    # Anualizar usando periods_per_year
    return sortino * np.sqrt(periods_per_year)


def information_ratio(returns: pd.Series, benchmark: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calcula el ratio de información (exceso sobre benchmark vs error de seguimiento).
    
    Information Ratio = (E[R_p] - E[R_b]) / σ_tracking_error
    
    Donde σ_tracking_error es la desviación estándar de la diferencia entre retornos
    del portafolio y benchmark.
    
    Parameters
    ----------
    returns : pd.Series
        Serie de retornos del activo/cartera.
    benchmark : pd.Series
        Serie de retornos del benchmark.
    periods_per_year : int, default 252
        Número de periodos por año para anualización.
    
    Returns
    -------
    float
        Ratio de información anualizado. Retorna 0.0 si no hay tracking error.
        
    Raises
    ------
    ValueError
        Si las series están vacías o no se pueden alinear.
        
    Examples
    --------
    >>> returns = pd.Series([0.01, 0.02, -0.01, 0.03])
    >>> benchmark = pd.Series([0.005, 0.015, -0.005, 0.025])
    >>> ir = information_ratio(returns, benchmark)
    >>> isinstance(ir, float)
    True
    >>> # Caso sin tracking error
    >>> returns_same = pd.Series([0.01, 0.02, -0.01, 0.03])
    >>> ir_no_error = information_ratio(returns_same, returns_same)
    >>> ir_no_error == 0.0
    True
    """
    # Validaciones robustas
    if returns is None or benchmark is None:
        return 0.0
    
    if returns.empty or benchmark.empty:
        return 0.0
    
    # Verificar que son Series válidas
    if not isinstance(returns, pd.Series) or not isinstance(benchmark, pd.Series):
        return 0.0
    
    try:
        # Normalizar índices de fechas para mejor alineación
        returns_normalized = returns.copy()
        benchmark_normalized = benchmark.copy()
        
        # Convertir índices a DatetimeIndex si no lo son
        if not isinstance(returns_normalized.index, pd.DatetimeIndex):
            returns_normalized.index = pd.to_datetime(returns_normalized.index)
        if not isinstance(benchmark_normalized.index, pd.DatetimeIndex):
            benchmark_normalized.index = pd.to_datetime(benchmark_normalized.index)
        
        # Normalizar zonas horarias
        if returns_normalized.index.tz is not None:
            returns_normalized.index = returns_normalized.index.tz_convert('UTC').tz_localize(None)
        if benchmark_normalized.index.tz is not None:
            benchmark_normalized.index = benchmark_normalized.index.tz_convert('UTC').tz_localize(None)
        
        # Redondear fechas a días completos
        returns_normalized.index = returns_normalized.index.normalize()
        benchmark_normalized.index = benchmark_normalized.index.normalize()
        
        # Alinear series por fechas
        aligned = pd.concat([returns_normalized, benchmark_normalized], axis=1, join='inner').dropna()
        
        if aligned.empty:
            return 0.0
        
        # Verificar que hay suficientes datos
        if len(aligned) < 2:
            return 0.0
        
        # Calcular diferencia entre retornos
        diff = aligned.iloc[:, 0] - aligned.iloc[:, 1]
        
        # Verificar que diff no esté vacío
        if diff.empty:
            return 0.0
        
        # Calcular tracking error
        te = diff.std()
        
        # Verificar que tracking error sea válido
        if pd.isna(te) or te == 0:
            return 0.0
        
        # Calcular ratio de información y anualizar
        result = (diff.mean() / te) * np.sqrt(periods_per_year)
        
        # Asegurar que el resultado sea un float válido
        if pd.isna(result) or np.isinf(result):
            return 0.0
        
        return float(result)
        
    except Exception as e:
        # Log del error para debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Error en information_ratio: {str(e)}")
        return 0.0


# ============================================================================
# 6. RIESGO EXTREMO
# ============================================================================

def calculate_var(returns: pd.Series, level: float = 0.05, horizon_days: int = 1, 
                 method: Literal["historical"] = "historical") -> float:
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
        usando composición de retornos simples en ventana rodante, lo que introduce solapamiento.
    method : Literal["historical"], default "historical"
        Método de cálculo. Actualmente solo soporta "historical".
        Preparado para futuras extensiones (Cornish-Fisher, paramétrico).
    
    Returns
    -------
    float
        VaR al nivel especificado (valor positivo representa pérdida).
        
    Raises
    ------
    ValueError
        Si la serie está vacía, el nivel es inválido, o el horizonte es inválido.
        
    Examples
    --------
    >>> returns = pd.Series(np.random.normal(0, 0.02, 1000))
    >>> var_5 = calculate_var(returns, 0.05)
    >>> var_5 > 0  # VaR debe ser positivo (representa pérdida)
    True
    >>> # Con horizonte de 5 días
    >>> var_5d = calculate_var(returns, 0.05, horizon_days=5)
    >>> var_5d > var_5  # VaR de horizonte mayor debe ser mayor
    True
    """
    if returns.empty:
        raise ValueError("La serie de retornos está vacía")
    
    if not (0 < level < 1):
        raise ValueError(f"El parámetro 'level' debe estar entre 0 y 1, recibido: {level}")
    
    if horizon_days < 1:
        raise ValueError(f"El parámetro 'horizon_days' debe ser >= 1, recibido: {horizon_days}")
    
    clean_returns = returns.dropna()
    
    # Agregar a horizonte si aplica (asumiendo retornos simples)
    if horizon_days > 1:
        if len(clean_returns) < horizon_days:
            raise ValueError(f"No hay suficientes datos para el horizonte solicitado: {len(clean_returns)} observaciones vs {horizon_days} días requeridos")
        # Composición de retornos simples en ventana rodante
        clean_returns = (1 + clean_returns).rolling(window=horizon_days).apply(lambda x: np.prod(x) - 1, raw=False)
        clean_returns = clean_returns.dropna()
    
    if len(clean_returns) == 0:
        raise ValueError("No hay datos válidos en la serie tras el procesamiento")
    
    # Calcular VaR como percentil (negativo para representar pérdida)
    var_value = -np.percentile(clean_returns, level * 100)
    
    return var_value


def calculate_cvar(returns: pd.Series, level: float = 0.05, horizon_days: int = 1,
                  method: Literal["historical"] = "historical") -> float:
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
        usando composición de retornos simples en ventana rodante, lo que introduce solapamiento.
    method : Literal["historical"], default "historical"
        Método de cálculo. Actualmente solo soporta "historical".
        Preparado para futuras extensiones (Cornish-Fisher, paramétrico).
    
    Returns
    -------
    float
        CVaR al nivel especificado (valor positivo representa pérdida).
        
    Raises
    ------
    ValueError
        Si la serie está vacía, el nivel es inválido, o el horizonte es inválido.
        
    Examples
    --------
    >>> returns = pd.Series(np.random.normal(0, 0.02, 1000))
    >>> cvar_5 = calculate_cvar(returns, 0.05)
    >>> cvar_5 > 0  # CVaR debe ser positivo (representa pérdida)
    True
    >>> # Con horizonte de 5 días
    >>> cvar_5d = calculate_cvar(returns, 0.05, horizon_days=5)
    >>> cvar_5d > cvar_5  # CVaR de horizonte mayor debe ser mayor
    True
    """
    if returns.empty:
        raise ValueError("La serie de retornos está vacía")
    
    if not (0 < level < 1):
        raise ValueError(f"El parámetro 'level' debe estar entre 0 y 1, recibido: {level}")
    
    if horizon_days < 1:
        raise ValueError(f"El parámetro 'horizon_days' debe ser >= 1, recibido: {horizon_days}")
    
    clean_returns = returns.dropna()
    
    # Agregar a horizonte si aplica (asumiendo retornos simples)
    if horizon_days > 1:
        if len(clean_returns) < horizon_days:
            raise ValueError(f"No hay suficientes datos para el horizonte solicitado: {len(clean_returns)} observaciones vs {horizon_days} días requeridos")
        clean_returns = (1 + clean_returns).rolling(window=horizon_days).apply(lambda x: np.prod(x) - 1, raw=False)
        clean_returns = clean_returns.dropna()
    
    if len(clean_returns) == 0:
        raise ValueError("No hay datos válidos en la serie tras el procesamiento")
    
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

def calculate_beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series,
                   rf: Optional[pd.Series] = None, periods_per_year: int = 252) -> Dict[str, float]:
    """
    Calcula Alpha, Beta y R² usando regresión OLS.
    
    Si rf se proporciona: R_p - R_f = α + β(R_m - R_f) + ε (modelo CAPM completo)
    Si rf es None: R_p = α + β R_m + ε (regresión simple)
    
    Parameters
    ----------
    portfolio_returns : pd.Series
        Serie de retornos del portafolio/activo.
    benchmark_returns : pd.Series
        Serie de retornos del benchmark.
    rf : Optional[pd.Series], default None
        Serie de tasa libre de riesgo. Si se proporciona, usa modelo CAPM completo.
    periods_per_year : int, default 252
        Número de periodos por año para anualización de alpha.
    
    Returns
    -------
    Dict[str, float]
        Diccionario con 'alpha', 'beta', y 'r2'.
        Alpha se anualiza solo si rf se proporciona.
        
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
    >>> # Con tasa libre de riesgo (CAPM completo)
    >>> rf = pd.Series([0.001, 0.001, 0.001, 0.001])
    >>> result_capm = calculate_beta(portfolio, benchmark, rf=rf)
    >>> result_capm['alpha'] != result['alpha']  # Alpha anualizado vs no anualizado
    True
    """
    if portfolio_returns.empty or benchmark_returns.empty:
        raise ValueError("Las series de retornos están vacías")
    
    if rf is not None:
        # Modelo CAPM completo: R_p - R_f = α + β(R_m - R_f)
        rp, rf_al = _align_series_by_date(portfolio_returns, rf)
        rm, _ = _align_series_by_date(benchmark_returns, rf)
        
        # Calcular excesos de retorno
        y = rp - rf_al
        x = rm - rf_al
        
        # Verificar que hay suficientes observaciones
        if len(y) < 2:
            raise ValueError("No hay suficientes observaciones tras alinear las series")
        
        # Regresión OLS
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        
        # Extraer parámetros
        alpha_param = model.params.get('const', model.params.iloc[0])
        beta_param = model.params.drop('const', errors='ignore').iloc[0] if 'const' in model.params else model.params.iloc[1]
        
        # Anualizar alpha para modelo CAPM
        alpha = alpha_param * periods_per_year
        
    else:
        # Regresión simple: R_p = α + β R_m
        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1, join='inner').dropna()
        
        if aligned.empty or len(aligned) < 2:
            raise ValueError("No hay suficientes observaciones tras alinear las series")
        
        y = aligned.iloc[:, 0]  # Retornos del portafolio
        x = aligned.iloc[:, 1]  # Retornos del benchmark
        
        # Regresión OLS
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        
        # Extraer parámetros
        alpha_param = model.params.get('const', model.params.iloc[0])
        beta_param = model.params.drop('const', errors='ignore').iloc[0] if 'const' in model.params else model.params.iloc[1]
        
        # No anualizar alpha para regresión simple
        alpha = alpha_param
    
    return {
        'alpha': float(alpha),
        'beta': float(beta_param),
        'r2': float(model.rsquared)
    }


# ============================================================================
# 8. CAPM (CAPITAL ASSET PRICING MODEL)
# ============================================================================

def calculate_capm_metrics(returns: pd.Series, market_returns: pd.Series, 
                          risk_free_rate: pd.Series, periods_per_year: int = 252) -> Dict[str, float]:
    """
    Calcula métricas CAPM completas.
    
    Modelo CAPM: E(Ri) = Rf + βi(E(Rm) - Rf)
    
    Parameters
    ----------
    returns : pd.Series
        Serie de retornos del activo/portafolio.
    market_returns : pd.Series
        Serie de retornos del mercado (benchmark).
    risk_free_rate : pd.Series
        Serie de tasa libre de riesgo.
    periods_per_year : int, default 252
        Número de periodos por año para anualización.
    
    Returns
    -------
    Dict[str, float]
        Diccionario con métricas CAPM: 'beta', 'alpha', 'r2', 'expected_return_capm',
        'excess_return', 'market_risk_premium', 'sharpe_ratio_capm'.
        
    Raises
    ------
    ValueError
        Si las series están vacías o no se pueden alinear.
        
    Examples
    --------
    >>> returns = pd.Series([0.01, 0.02, -0.01, 0.03])
    >>> market = pd.Series([0.005, 0.015, -0.005, 0.025])
    >>> rf = pd.Series([0.001, 0.001, 0.001, 0.001])
    >>> capm = calculate_capm_metrics(returns, market, rf)
    >>> 'beta' in capm and 'expected_return_capm' in capm
    True
    >>> # Con frecuencia mensual
    >>> capm_monthly = calculate_capm_metrics(returns, market, rf, periods_per_year=12)
    >>> capm_monthly['expected_return_capm'] != capm['expected_return_capm']
    True
    """
    if returns.empty or market_returns.empty or risk_free_rate.empty:
        raise ValueError("Alguna de las series está vacía")
    
    # Debug: imprimir información sobre las series
            # print(f"      DEBUG CAPM - Returns: {len(returns)} obs, fechas: {returns.index[0]} a {returns.index[-1]}")
        # print(f"      DEBUG CAPM - Market: {len(market_returns)} obs, fechas: {market_returns.index[0]} a {market_returns.index[-1]}")
        # print(f"      DEBUG CAPM - RF: {len(risk_free_rate)} obs, fechas: {risk_free_rate.index[0]} a {risk_free_rate.index[-1]}")
    
    # Alinear todas las series por fechas de una vez
    try:
        # Normalizar zonas horarias antes de crear el DataFrame
        returns_normalized = returns.copy()
        market_normalized = market_returns.copy()
        rf_normalized = risk_free_rate.copy()
        
        # Convertir índices a DatetimeIndex si no lo son
        if not isinstance(returns_normalized.index, pd.DatetimeIndex):
            returns_normalized.index = pd.to_datetime(returns_normalized.index)
        if not isinstance(market_normalized.index, pd.DatetimeIndex):
            market_normalized.index = pd.to_datetime(market_normalized.index)
        if not isinstance(rf_normalized.index, pd.DatetimeIndex):
            rf_normalized.index = pd.to_datetime(rf_normalized.index)
        
        # Normalizar zonas horarias - convertir ambas a UTC o eliminar zona horaria
        if returns_normalized.index.tz is not None:
            returns_normalized.index = returns_normalized.index.tz_convert('UTC').tz_localize(None)
        if market_normalized.index.tz is not None:
            market_normalized.index = market_normalized.index.tz_convert('UTC').tz_localize(None)
        if rf_normalized.index.tz is not None:
            rf_normalized.index = rf_normalized.index.tz_convert('UTC').tz_localize(None)
        
        # Redondear todas las fechas a días completos (00:00:00)
        returns_normalized.index = returns_normalized.index.normalize()
        market_normalized.index = market_normalized.index.normalize()
        rf_normalized.index = rf_normalized.index.normalize()
        
        # print(f"      DEBUG CAPM - Después de normalización:")
        # print(f"        Returns: {returns_normalized.index[0]} a {returns_normalized.index[-1]}")
        # print(f"        Market: {market_normalized.index[0]} a {market_normalized.index[-1]}")
        # print(f"        RF: {rf_normalized.index[0]} a {rf_normalized.index[-1]}")
        
        # Crear DataFrame con todas las series
        df = pd.DataFrame({
            'returns': returns_normalized,
            'market': market_normalized,
            'rf': rf_normalized
        })
        
        # print(f"      DEBUG CAPM - DataFrame creado: {df.shape}")
        
        # Mostrar algunas fechas del DataFrame
        # print(f"      DEBUG CAPM - Primeras fechas del DataFrame:")
        # print(f"        {df.head().index.tolist()}")
        
        # Eliminar filas con valores NaN
        df = df.dropna()
        
        # print(f"      DEBUG CAPM - Después de dropna: {df.shape}")
        
        if df.empty:
            # Mostrar información adicional para debug
            # Todas las series tienen valores NaN en las mismas fechas
            logger.warning(f"Returns NaN: {returns_normalized.isna().sum()}")
            logger.warning(f"Market NaN: {market_normalized.isna().sum()}")
            logger.warning(f"RF NaN: {rf_normalized.isna().sum()}")
            raise ValueError("No hay fechas comunes entre las series para calcular CAPM")
        
        if len(df) < 30:  # Necesitamos al menos 30 observaciones para un análisis confiable
            raise ValueError(f"Insuficientes observaciones para CAPM: {len(df)} (mínimo 30)")
        
        aligned_returns = df['returns']
        aligned_market = df['market']
        aligned_rf = df['rf']
        
        # print(f"      DEBUG CAPM - Series alineadas: {len(aligned_returns)} obs")
        
    except Exception as e:
        raise ValueError(f"Error alineando series para CAPM: {str(e)}")
    
    # Verificar que las series alineadas no estén vacías
    if aligned_returns.empty or aligned_market.empty or aligned_rf.empty:
        raise ValueError("Las series alineadas están vacías")
    
    # Calcular excesos de retorno
    excess_returns = aligned_returns - aligned_rf
    excess_market = aligned_market - aligned_rf
    
    # Calcular Beta usando regresión OLS
    X = sm.add_constant(excess_market)
    model = sm.OLS(excess_returns, X).fit()
    
    # Acceder a los parámetros de manera más robusta
    if 'const' in model.params.index:
        alpha = model.params['const']
        # Para el segundo parámetro (beta), tomar el que no sea 'const'
        beta = model.params.drop('const').iloc[0]
    else:
        # Si no hay 'const', tomar el primer y segundo parámetro
        alpha = model.params.iloc[0]
        beta = model.params.iloc[1]
    
    r2 = model.rsquared
    
    # Verificar que los parámetros sean válidos
    if np.isnan(beta) or np.isnan(alpha) or np.isnan(r2):
        raise ValueError("Parámetros de regresión inválidos (NaN)")
    
    if np.isinf(beta) or np.isinf(alpha):
        raise ValueError("Parámetros de regresión inválidos (infinitos)")
    
    # Calcular métricas CAPM
    market_risk_premium = excess_market.mean()
    expected_return_capm = aligned_rf.mean() + beta * market_risk_premium
    actual_return = aligned_returns.mean()
    excess_return = actual_return - expected_return_capm
    
    # Anualizar métricas usando periods_per_year
    annualized_alpha = alpha * periods_per_year
    annualized_expected_return = expected_return_capm * periods_per_year
    annualized_actual_return = actual_return * periods_per_year
    annualized_excess_return = excess_return * periods_per_year
    annualized_market_risk_premium = market_risk_premium * periods_per_year
    
    # Sharpe ratio según CAPM
    sharpe_ratio_capm = (annualized_excess_return) / (aligned_returns.std() * np.sqrt(periods_per_year))
    
    return {
        'beta': beta,
        'alpha': annualized_alpha,
        'r2': r2,
        'expected_return_capm': annualized_expected_return,
        'actual_return': annualized_actual_return,
        'excess_return': annualized_excess_return,
        'market_risk_premium': annualized_market_risk_premium,
        'sharpe_ratio_capm': sharpe_ratio_capm
    }


def plot_security_market_line(
    assets_data: Dict[str, Dict[str, float]],
    market_return: float,
    risk_free_rate: float,
    *,
    periods_per_year: int = 252,
    ax: Optional[plt.Axes] = None,
    save_plot: bool = False,
    show_plot: bool = True,
    annotate: bool = True,
    annotate_threshold: float = 0.02  # 2% de diferencia vs SML
) -> Optional[str]:
    """
    Grafica la Línea del Mercado de Valores (SML) del CAPM con retornos reales vs esperados.
    
    SML: E(Ri) = Rf + βi(E(Rm) - Rf)
    Los puntos muestran retornos reales para comparar con la SML teórica.
    
    IMPORTANTE: Todos los valores deben estar en términos ANUALIZADOS.
    
    Parameters
    ----------
    assets_data : Dict[str, Dict[str, float]]
        Diccionario con datos de activos: {symbol: {'beta': float, 'actual_return': float}}
        donde 'beta' y 'actual_return' están en términos anualizados.
    market_return : float
        Retorno esperado del mercado (anualizado).
    risk_free_rate : float
        Tasa libre de riesgo (anualizada).
    periods_per_year : int, default 252
        Número de periodos por año (para compatibilidad, no se usa en cálculos).
    ax : Optional[plt.Axes], default None
        Eje de matplotlib donde graficar. Si se proporciona, no se llama a plt.show().
    save_plot : bool, default False
        Si guardar el gráfico.
    show_plot : bool, default True
        Si mostrar el gráfico.
    annotate : bool, default True
        Si añadir etiquetas de ticker y resaltar activos con alpha significativo.
    annotate_threshold : float, default 0.02
        Umbral para resaltar activos con alpha implícito >= 2% de diferencia vs SML.
    
    Returns
    -------
    Optional[str]
        Ruta del archivo si se guardó, None en caso contrario.
        
    Raises
    ------
    ValueError
        Si assets_data no contiene 'beta' y 'actual_return' para algún símbolo.
        
    Examples
    --------
    >>> data = {'AAA': {'beta': 1.2, 'actual_return': 0.18}, 'BBB': {'beta': 0.8, 'actual_return': 0.10}}
    >>> _ = plot_security_market_line(data, market_return=0.12, risk_free_rate=0.02, show_plot=False)
    """
    if not assets_data:
        logger.warning("No hay datos de activos para graficar")
        return None
    
    # Validar estructura de datos
    for symbol, data in assets_data.items():
        if 'beta' not in data or 'actual_return' not in data:
            raise ValueError(f"assets_data['{symbol}'] debe incluir 'beta' y 'actual_return' anualizados.")
    
    # Preparar datos para el gráfico
    symbols = list(assets_data.keys())
    betas = [assets_data[s]['beta'] for s in symbols]
    actual_returns = [assets_data[s]['actual_return'] for s in symbols]
    
    # Validar que los retornos estén en rango razonable anualizado
    if np.nanmax(np.abs(actual_returns)) < 0.02:
        logger.warning("plot_security_market_line: los retornos parecen no estar anualizados (|R|<2%). Revisa inputs.")
    
    # Calcular SML teórica
    market_risk_premium = market_return - risk_free_rate
    sml_betas = np.linspace(0, max(1.2, max(betas) * 1.1), 200)
    sml_returns = risk_free_rate + sml_betas * market_risk_premium
    
    # Crear gráfico
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    
    # Graficar SML teórica con línea discontinua más gruesa
    ax.plot(sml_betas, sml_returns, 'b--', linewidth=2.5, label='SML Teórica (CAPM)', alpha=0.8)
    
    # Graficar activos individuales con retornos reales
    colors = plt.cm.Set3(np.linspace(0, 1, len(symbols)))
    for i, symbol in enumerate(symbols):
        ax.scatter(betas[i], actual_returns[i], c=[colors[i]], s=110, 
                  edgecolors='black', alpha=0.85, label=f'{symbol} (Real)')
    
    # Líneas de referencia
    ax.axhline(y=risk_free_rate, color='g', linestyle=':', alpha=0.7, label=f'Rf = {risk_free_rate:.2%}')
    ax.axvline(x=1, color='r', linestyle=':', alpha=0.7, label='β = 1 (Mercado)')
    
    # Configurar eje Y en formato porcentaje anual
    import matplotlib.ticker as mtick
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    # Configurar gráfico
    ax.set_xlabel('Beta (β)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Retorno Anualizado E(R)', fontsize=12, fontweight='bold')
    ax.set_title('Línea del Mercado de Valores (SML) - CAPM\nRetornos Reales vs Teóricos (Anualizados)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Anotaciones útiles si se solicita
    if annotate:
        # Calcular alpha implícito (distancia vertical a SML)
        expected_on_sml = risk_free_rate + np.array(betas) * market_risk_premium
        alpha_impl = np.array(actual_returns) - expected_on_sml
        
        # Añadir etiquetas de ticker y resaltar activos con alpha significativo
        for i, symbol in enumerate(symbols):
            if abs(alpha_impl[i]) >= annotate_threshold:
                ax.annotate(symbol, (betas[i], actual_returns[i]), 
                           xytext=(6, 6), textcoords='offset points', 
                           fontsize=9, weight='bold')
    
    # Añadir cuadro de texto con valores anualizados
    ax.text(0.02, 0.98, 
            f'Rf = {risk_free_rate:.2%}\nE(Rm) = {market_return:.2%}\nMRP = {market_risk_premium:.2%}', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8), fontsize=10)
    
    # Warning si MRP es negativo
    if market_risk_premium < 0:
        logger.warning("plot_security_market_line: Market Risk Premium es negativo (E(Rm) < Rf)")
    
    if ax is None:
        plt.tight_layout()
    
    # Guardar si se solicita
    filepath = None
    if save_plot:
        filepath = "security_market_line_capm.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    # Solo mostrar si no se proporcionó un eje y se solicita mostrar
    if show_plot and ax is None:
        plt.show()
    elif not show_plot and ax is None:
        plt.close()
    
    return filepath


def build_assets_data_from_capm(capm_results: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Toma el dict devuelto por calculate_capm_metrics para cada símbolo y
    devuelve {symbol: {'beta': float, 'actual_return': float}} en anualizado.
    
    Parameters
    ----------
    capm_results : Dict[str, Dict[str, float]]
        Diccionario con resultados CAPM de múltiples símbolos.
    
    Returns
    -------
    Dict[str, Dict[str, float]]
        Diccionario con estructura requerida para plot_security_market_line.
        
    Examples
    --------
    >>> capm_data = {'AAPL': {'beta': 1.2, 'actual_return': 0.18, 'alpha': 0.05}}
    >>> assets_data = build_assets_data_from_capm(capm_data)
    >>> assets_data['AAPL']['beta']
    1.2
    >>> assets_data['AAPL']['actual_return']
    0.18
    """
    out = {}
    for symbol, metrics in capm_results.items():
        if 'beta' in metrics and 'actual_return' in metrics:
            out[symbol] = {
                'beta': float(metrics['beta']), 
                'actual_return': float(metrics['actual_return'])
            }
    return out


def plot_alpha_vs_beta(assets_data: Dict[str, Dict[str, float]], 
                       save_plot: bool = False, show_plot: bool = True,
                       ax: Optional[plt.Axes] = None) -> Optional[str]:
    """
    Grafica Alpha vs Beta para evaluar retornos excesivos del CAPM.
    
    Parameters
    ----------
    assets_data : Dict[str, Dict[str, float]]
        Diccionario con datos de activos: {symbol: {'beta': float, 'alpha': float}}
    save_plot : bool, default False
        Si guardar el gráfico.
    show_plot : bool, default True
        Si mostrar el gráfico.
    ax : Optional[plt.Axes], default None
        Eje de matplotlib donde graficar. Si se proporciona, no se llama a plt.show().
    
    Returns
    -------
    Optional[str]
        Ruta del archivo si se guardó, None en caso contrario.
    """
    if not assets_data:
        logger.warning("No hay datos de activos para graficar")
        return None
    
    # Preparar datos para el gráfico
    symbols = list(assets_data.keys())
    betas = [assets_data[s]['beta'] for s in symbols]
    alphas = [assets_data[s]['alpha'] for s in symbols]
    
    # Crear gráfico
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    
    # Línea horizontal en alpha = 0 (SML teórica)
    ax.axhline(y=0, color='g', linestyle='--', linewidth=2, 
               label='SML Teórica (α = 0)', alpha=0.7)
    
    # Graficar activos individuales con alphas
    colors = plt.cm.Set3(np.linspace(0, 1, len(symbols)))
    for i, symbol in enumerate(symbols):
        ax.scatter(betas[i], alphas[i], c=[colors[i]], s=100, 
                  label=f'{symbol} (α = {alphas[i]:.2%})', alpha=0.8, edgecolors='black')
    
    # Líneas de referencia
    ax.axvline(x=1, color='r', linestyle=':', alpha=0.7, label='β = 1 (Mercado)')
    
    # Configurar gráfico
    ax.set_xlabel('Beta (β)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Alpha (α) - Retorno Excesivo', fontsize=12, fontweight='bold')
    ax.set_title('Alpha vs Beta - Análisis de Retornos Excesivos (CAPM)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Añadir anotaciones explicativas
    ax.text(0.02, 0.98, 
            f'Interpretación:\n\n'
            f'α > 0: Activo por encima de SML\n(Retorno excesivo positivo)\n\n'
            f'α < 0: Activo por debajo de SML\n(Retorno excesivo negativo)\n\n'
            f'α = 0: Activo en SML\n(Retorno según CAPM)', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8), fontsize=10)
    
    if ax is None:
        plt.tight_layout()
    
    # Guardar si se solicita
    filepath = None
    if save_plot:
        filepath = "alpha_vs_beta_capm.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    # Solo mostrar si no se proporcionó un eje y se solicita mostrar
    if show_plot and ax is None:
        plt.show()
    elif not show_plot and ax is None:
        plt.close()
    
    return filepath


def analyze_market_efficiency(returns: pd.DataFrame, market_returns: pd.Series, 
                            risk_free_rate: pd.Series) -> Dict[str, any]:
    """
    Analiza la eficiencia del mercado según el CAPM.
    
    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame con retornos de múltiples activos.
    market_returns : pd.Series
        Serie de retornos del mercado.
    risk_free_rate : pd.Series
        Serie de tasa libre de riesgo.
    
    Returns
    -------
    Dict[str, any]
        Resultados del análisis de eficiencia del mercado.
        
    Examples
    --------
    >>> returns_df = pd.DataFrame({'AAPL': [0.01, 0.02], 'MSFT': [0.015, 0.025]})
    >>> market = pd.Series([0.012, 0.022])
    >>> rf = pd.Series([0.001, 0.001])
    >>> efficiency = analyze_market_efficiency(returns_df, market, rf)
    """
    if returns.empty or market_returns.empty or risk_free_rate.empty:
        return {"error": "Datos insuficientes para el análisis"}
    
    # Calcular métricas CAPM para todos los activos
    capm_results = {}
    efficiency_metrics = {}
    
    for symbol in returns.columns:
        try:
            capm_metrics = calculate_capm_metrics(returns[symbol], market_returns, risk_free_rate)
            capm_results[symbol] = capm_metrics
            
            # Métricas de eficiencia
            efficiency_metrics[symbol] = {
                'alpha_significance': abs(capm_metrics['alpha']) < 0.01,  # Alpha < 1% anual
                'r2_quality': capm_metrics['r2'] > 0.3,  # R² > 30%
                'beta_stability': 0.5 < capm_metrics['beta'] < 2.0,  # Beta razonable
                'excess_return_consistency': abs(capm_metrics['excess_return']) < 0.05  # Exceso < 5%
            }
            
        except Exception as e:
            capm_results[symbol] = {"error": str(e)}
            efficiency_metrics[symbol] = {"error": str(e)}
    
    # Análisis agregado del mercado
    valid_results = {k: v for k, v in capm_results.items() if 'error' not in v}
    
    if valid_results:
        avg_alpha = np.mean([v['alpha'] for v in valid_results.values()])
        avg_r2 = np.mean([v['r2'] for v in valid_results.values()])
        avg_beta = np.mean([v['beta'] for v in valid_results.values()])
        
        market_efficiency_score = (
            (np.mean([v['alpha_significance'] for v in efficiency_metrics.values() if 'error' not in v]) * 0.4) +
            (np.mean([v['r2_quality'] for v in efficiency_metrics.values() if 'error' not in v]) * 0.3) +
            (np.mean([v['beta_stability'] for v in efficiency_metrics.values() if 'error' not in v]) * 0.2) +
            (np.mean([v['excess_return_consistency'] for v in efficiency_metrics.values() if 'error' not in v]) * 0.1)
        )
    else:
        avg_alpha = avg_r2 = avg_beta = market_efficiency_score = np.nan
    
    return {
        'capm_results': capm_results,
        'efficiency_metrics': efficiency_metrics,
        'market_summary': {
            'avg_alpha': avg_alpha,
            'avg_r2': avg_r2,
            'avg_beta': avg_beta,
            'efficiency_score': market_efficiency_score
        }
    }


# ============================================================================
# 9. DISTRIBUCIÓN Y AUTOCORRELACIÓN
# ============================================================================

def plot_return_distribution(returns: pd.Series, symbol: str = "Asset", 
                           save_plot: bool = False, show_plot: bool = True,
                           ax: Optional[Tuple[plt.Axes, plt.Axes]] = None) -> Optional[str]:
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
    ax : Optional[Tuple[plt.Axes, plt.Axes]], default None
        Tupla de ejes de matplotlib donde graficar (histograma, QQ-plot).
        Si se proporciona, no se llama a plt.show().
    
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
        logger.warning("No hay datos válidos para graficar")
        return None
    
    # Crear figura con 2 subplots
    if ax is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    else:
        ax1, ax2 = ax
    
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
    
    if ax is None:
        plt.tight_layout()
    
    # Guardar si se solicita
    filepath = None
    if save_plot:
        filepath = f"{symbol}_distribution_analysis.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    # Solo mostrar si no se proporcionó un eje y se solicita mostrar
    if show_plot and ax is None:
        plt.show()
    elif not show_plot and ax is None:
        plt.close()
    
    return filepath


def autocorrelation_analysis(returns: pd.Series, lags: int = 20, symbol: str = "Asset",
                           save_plot: bool = False, show_plot: bool = True,
                           ax: Optional[Tuple[plt.Axes, plt.Axes, plt.Axes, plt.Axes]] = None) -> Dict[str, Any]:
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
    ax : Optional[Tuple[plt.Axes, plt.Axes, plt.Axes, plt.Axes]], default None
        Tupla de 4 ejes de matplotlib donde graficar (ACF retornos, ACF retornos², 
        Ljung-Box retornos, Ljung-Box retornos²). Si se proporciona, no se llama a plt.show().
    
    Returns
    -------
    Dict[str, Any]
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
    if ax is None:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    else:
        ax1, ax2, ax3, ax4 = ax
    
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
    
    if ax is None:
        plt.tight_layout()
    
    # Guardar si se solicita
    filepath = None
    if save_plot:
        filepath = f"{symbol}_autocorrelation_analysis.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    # Solo mostrar si no se proporcionó un eje y se solicita mostrar
    if show_plot and ax is None:
        plt.show()
    elif not show_plot and ax is None:
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
