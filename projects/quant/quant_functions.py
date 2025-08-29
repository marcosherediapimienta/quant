#!/usr/bin/env python3
"""
Funciones de análisis cuantitativo para datos financieros.
Todas las funciones reciben DataFrames de precios con símbolos en columnas e índice de fechas.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend no-interactivo para evitar warnings
import matplotlib.pyplot as plt
from typing import Union, Dict, Tuple
from scipy import stats
import warnings


def calculate_returns(prices: pd.DataFrame, freq: str = "daily") -> pd.DataFrame:
    """
    Calcula retornos de precios.
    
    Args:
        prices: DataFrame con precios históricos (índice: fechas, columnas: símbolos)
        freq: Frecuencia de retornos ("daily", "monthly", "weekly")
        
    Returns:
        DataFrame con retornos calculados
    """
    if prices.empty:
        raise ValueError("DataFrame de precios está vacío")
    
    if freq == "daily":
        returns = prices.pct_change(fill_method=None)
    elif freq == "weekly":
        returns = prices.resample('W').last().pct_change(fill_method=None)
    elif freq == "monthly":
        returns = prices.resample('M').last().pct_change(fill_method=None)
    else:
        raise ValueError("freq debe ser 'daily', 'weekly' o 'monthly'")
    
    return returns.dropna()


def calculate_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula retornos logarítmicos.
    
    Args:
        prices: DataFrame con precios históricos
        
    Returns:
        DataFrame con retornos logarítmicos
    """
    if prices.empty:
        raise ValueError("DataFrame de precios está vacío")
    
    # Verificar que no hay valores negativos o cero
    if (prices <= 0).any().any():
        raise ValueError("Los precios deben ser estrictamente positivos para calcular retornos logarítmicos")
    
    # Calcular retornos logarítmicos
    log_returns = np.log(prices / prices.shift(1))
    
    return log_returns.dropna()


def calculate_skewness(returns: pd.DataFrame) -> pd.Series:
    """
    Calcula la asimetría (skewness) de los retornos.
    
    Args:
        returns: DataFrame con retornos
        
    Returns:
        Serie con la asimetría de cada activo
        
    Notes:
        - Skewness = 0: distribución simétrica
        - Skewness > 0: cola derecha más larga (sesgo positivo)
        - Skewness < 0: cola izquierda más larga (sesgo negativo)
    """
    if returns.empty:
        raise ValueError("DataFrame de retornos está vacío")
    
    return returns.skew()


def calculate_kurtosis(returns: pd.DataFrame) -> pd.Series:
    """
    Calcula la curtosis (kurtosis) de los retornos.
    
    Args:
        returns: DataFrame con retornos
        
    Returns:
        Serie con la curtosis de cada activo
        
    Notes:
        - Kurtosis = 3: distribución normal (mesocúrtica)
        - Kurtosis > 3: colas pesadas (leptocúrtica)
        - Kurtosis < 3: colas ligeras (platicúrtica)
        - Esta función devuelve kurtosis de Fisher (excess kurtosis = kurtosis - 3)
    """
    if returns.empty:
        raise ValueError("DataFrame de retornos está vacío")
    
    return returns.kurtosis()  # Pandas usa Fisher's definition (excess kurtosis)


def calculate_percentiles(returns: pd.DataFrame, percentiles: list = None) -> pd.DataFrame:
    """
    Calcula percentiles específicos de los retornos.
    
    Args:
        returns: DataFrame con retornos
        percentiles: Lista de percentiles a calcular (por defecto: [1, 5, 25, 50, 75, 95, 99])
        
    Returns:
        DataFrame con percentiles por activo
    """
    if returns.empty:
        raise ValueError("DataFrame de retornos está vacío")
    
    if percentiles is None:
        percentiles = [1, 5, 25, 50, 75, 95, 99]
    
    # Convertir percentiles a proporciones
    quantiles = [p/100 for p in percentiles]
    
    result = returns.quantile(quantiles).T
    result.columns = [f'P{p}' for p in percentiles]
    
    return result


def calculate_descriptive_stats(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula estadísticas descriptivas completas para los retornos.
    
    Args:
        returns: DataFrame con retornos
        
    Returns:
        DataFrame con estadísticas descriptivas por activo
    """
    if returns.empty:
        raise ValueError("DataFrame de retornos está vacío")
    
    stats_dict = {}
    
    for column in returns.columns:
        series = returns[column].dropna()
        
        if len(series) == 0:
            continue
            
        stats_dict[column] = {
            'count': len(series),
            'mean': series.mean(),
            'std': series.std(),
            'min': series.min(),
            'max': series.max(),
            'median': series.median(),
            'skewness': series.skew(),
            'kurtosis': series.kurtosis(),
            'var': series.var(),
            'range': series.max() - series.min(),
            'iqr': series.quantile(0.75) - series.quantile(0.25),
            'cv': series.std() / abs(series.mean()) if series.mean() != 0 else np.nan,  # Coeficiente de variación
            'mad': (series - series.median()).abs().median(),  # Median Absolute Deviation
        }
    
    df_stats = pd.DataFrame(stats_dict).T
    return df_stats


def calculate_moments(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula los primeros cuatro momentos estadísticos.
    
    Args:
        returns: DataFrame con retornos
        
    Returns:
        DataFrame con los momentos por activo
        
    Notes:
        - Momento 1: Media (tendencia central)
        - Momento 2: Varianza (dispersión)
        - Momento 3: Skewness (asimetría)
        - Momento 4: Kurtosis (forma de las colas)
    """
    if returns.empty:
        raise ValueError("DataFrame de retornos está vacío")
    
    moments_dict = {}
    
    for column in returns.columns:
        series = returns[column].dropna()
        
        if len(series) == 0:
            continue
            
        moments_dict[column] = {
            'moment_1_mean': series.mean(),
            'moment_2_variance': series.var(),
            'moment_3_skewness': series.skew(),
            'moment_4_kurtosis': series.kurtosis(),
            'moment_2_std': series.std(),
            'moment_3_raw': stats.moment(series, moment=3),
            'moment_4_raw': stats.moment(series, moment=4),
        }
    
    return pd.DataFrame(moments_dict).T


def calculate_correlation_matrix(returns: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
    """
    Calcula la matriz de correlación entre activos.
    
    Args:
        returns: DataFrame con retornos
        method: Método de correlación ('pearson', 'kendall', 'spearman')
        
    Returns:
        DataFrame con matriz de correlación
    """
    if returns.empty:
        raise ValueError("DataFrame de retornos está vacío")
    
    if method not in ['pearson', 'kendall', 'spearman']:
        raise ValueError("method debe ser 'pearson', 'kendall' o 'spearman'")
    
    return returns.corr(method=method)


def calculate_covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la matriz de covarianza entre activos.
    
    Args:
        returns: DataFrame con retornos
        
    Returns:
        DataFrame con matriz de covarianza
    """
    if returns.empty:
        raise ValueError("DataFrame de retornos está vacío")
    
    return returns.cov()


def calculate_rolling_statistics(returns: pd.DataFrame, window: int = 30) -> Dict[str, pd.DataFrame]:
    """
    Calcula estadísticas móviles para una ventana específica.
    
    Args:
        returns: DataFrame con retornos
        window: Tamaño de la ventana (días)
        
    Returns:
        Diccionario con DataFrames de estadísticas móviles
    """
    if returns.empty:
        raise ValueError("DataFrame de retornos está vacío")
    
    if window < 2:
        raise ValueError("window debe ser al menos 2")
    
    rolling_stats = {
        'mean': returns.rolling(window=window).mean(),
        'std': returns.rolling(window=window).std(),
        'var': returns.rolling(window=window).var(),
        'skew': returns.rolling(window=window).skew(),
        'kurt': returns.rolling(window=window).kurt(),
        'min': returns.rolling(window=window).min(),
        'max': returns.rolling(window=window).max(),
        'median': returns.rolling(window=window).median(),
    }
    
    return rolling_stats


def get_symbols():
    """Obtiene la lista de símbolos definida en test_data_manager."""
    return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'IBM', 'IONQ', 'SMR', 
            'BTC-USD', 'ETH-USD', 'GLD', 'ACWI', 'SPY', 'QQQ']


def get_market_data():
    """Obtiene datos usando EXACTAMENTE la misma lógica que test_data_manager.py."""
    import sys
    from pathlib import Path
    
    # Importar DataManager
    sys.path.append(str(Path(__file__).parent.parent / "data"))
    from data_manager import DataManager
    
    symbols = get_symbols()
    
    # Usar DataManager exactamente como en test_data_manager.py
    test_dir = Path(__file__).parent.parent / "data" / "test"
    dm = DataManager(test_mode=True, test_dir=str(test_dir))
    
    # Usar 10 años como en test_data_manager.py (que funciona y muestra historial completo)
    grouped_data = dm.download_market_data_grouped(
        symbols=symbols,
        target_years=10,  # Mismo que test_data_manager.py
        force_refresh=False
    )
    
    # Usar la misma función que test_data_manager.py para procesar y mostrar
    all_plots = dm.process_and_display_groups(grouped_data, target_years=10)
    
    # Combinar grupos conservando el historial completo de cada activo
    all_prices = pd.DataFrame()
    
    for group_name, prices in grouped_data.items():
        if isinstance(prices, pd.DataFrame) and not prices.empty:
            if all_prices.empty:
                all_prices = prices.copy()
            else:
                all_prices = all_prices.join(prices, how='outer')
    
    return all_prices


def analyze_log_returns(prices: pd.DataFrame,) -> dict:
    """Muestra todos los retornos logarítmicos diarios de cada activo desde su primer día hasta hoy."""
    from pathlib import Path
    
    # Mostrar información general del DataFrame
    print(f"\n📈 DATOS CARGADOS: {prices.shape}")
    print(f"📅 Rango total: {prices.index[0].strftime('%Y-%m-%d')} → {prices.index[-1].strftime('%Y-%m-%d')}")
    
    # Analizar cada activo individualmente mostrando TODOS los retornos
    print("\n📊 RETORNOS LOGARÍTMICOS DIARIOS POR ACTIVO:")
    print("=" * 80)
    
    all_individual_returns = {}
    
    for symbol in prices.columns:
        # Obtener datos del activo eliminando NaN
        symbol_prices = prices[symbol].dropna()
        
        if len(symbol_prices) < 2:
            print(f"   {symbol:>8}: ⚠️  Datos insuficientes")
            continue
            
        # Calcular retornos logarítmicos para este activo
        symbol_log_returns = calculate_log_returns(symbol_prices.to_frame())
        symbol_log_returns = symbol_log_returns[symbol]  # Convertir a Serie
        
        # Información del período
        trading_days = len(symbol_log_returns)
        start_date = symbol_log_returns.index[0].strftime('%Y-%m-%d')
        end_date = symbol_log_returns.index[-1].strftime('%Y-%m-%d')
        
        # Mostrar encabezado del activo
        print(f"\n🔹 {symbol}: {start_date} → {end_date} ({trading_days} días)")
        print("-" * 60)
        
        # Mostrar los 3 primeros y 3 últimos retornos
        print("   📊 Primeros 3 retornos:")
        for i, (date, return_value) in enumerate(symbol_log_returns.head(3).items()):
            date_str = date.strftime('%Y-%m-%d')
            print(f"      {date_str}: {return_value:>12.8f}")
        
        if trading_days > 6:
            print("      ...")
            print("   📊 Últimos 3 retornos:")
            for date, return_value in symbol_log_returns.tail(3).items():
                date_str = date.strftime('%Y-%m-%d')
                print(f"      {date_str}: {return_value:>12.8f}")
        elif trading_days > 3:
            print("   📊 Últimos retornos:")
            for date, return_value in symbol_log_returns.tail(trading_days - 3).items():
                date_str = date.strftime('%Y-%m-%d')
                print(f"      {date_str}: {return_value:>12.8f}")
        
        # Guardar resultados
        all_individual_returns[symbol] = symbol_log_returns
    
    print(f"\n📊 RESUMEN:")
    print(f"   📊 Activos procesados: {len(all_individual_returns)}")
    total_returns = sum(len(returns) for returns in all_individual_returns.values())
    print(f"   📊 Total de retornos diarios: {total_returns}")

    return {
        'all_individual_returns': all_individual_returns,
        'total_daily_returns': total_returns
    }


def plot_returns_timeline(all_returns: dict, save_plots: bool = True) -> None:
    """
    Grafica los retornos logarítmicos de cada activo a lo largo del tiempo.
    
    Args:
        all_returns: Diccionario con retornos por activo
        save_plots: Si guardar los gráficos como archivos
    """
    from pathlib import Path
    
    # Crear directorio de gráficos en test/plots/
    if save_plots:
        plots_dir = Path(__file__).parent / "test" / "plots"
        plots_dir.mkdir(exist_ok=True)
    
    for symbol, returns in all_returns.items():
        plt.figure(figsize=(12, 6))
        
        # Gráfico de retornos
        plt.plot(returns.index, returns.values, linewidth=0.8, alpha=0.7)
        plt.title(f'Retornos Logarítmicos Diarios - {symbol}', fontsize=14, fontweight='bold')
        plt.xlabel('Fecha')
        plt.ylabel('Retorno Logarítmico')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Línea en cero
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Estadísticas en el gráfico
        mean_return = returns.mean()
        std_return = returns.std()
        plt.text(0.02, 0.98, f'Media: {mean_return:.6f}\nDesv.Est: {std_return:.6f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(plots_dir / f'{symbol}_returns.png', dpi=300, bbox_inches='tight')
            print(f"   📊 Gráfico guardado: {symbol}_returns.png")
        
        plt.close()  # Liberar memoria


def plot_returns_distribution(all_returns: dict, save_plots: bool = True) -> None:
    """
    Grafica la distribución de retornos de cada activo.
    
    Args:
        all_returns: Diccionario con retornos por activo
        save_plots: Si guardar los gráficos como archivos
    """
    from pathlib import Path
    
    # Crear directorio de gráficos en test/plots/
    if save_plots:
        plots_dir = Path(__file__).parent / "test" / "plots"
        plots_dir.mkdir(exist_ok=True)
    
    for symbol, returns in all_returns.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histograma
        ax1.hist(returns.values, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax1.set_title(f'Distribución de Retornos - {symbol}', fontweight='bold')
        ax1.set_xlabel('Retorno Logarítmico')
        ax1.set_ylabel('Frecuencia')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        # Box plot
        ax2.boxplot(returns.values, vert=True)
        ax2.set_title(f'Box Plot - {symbol}', fontweight='bold')
        ax2.set_ylabel('Retorno Logarítmico')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(plots_dir / f'{symbol}_distribution.png', dpi=300, bbox_inches='tight')
            print(f"   📊 Gráfico guardado: {symbol}_distribution.png")
        
        plt.close()  # Liberar memoria


def plot_cumulative_returns(all_returns: dict, save_plots: bool = True) -> None:
    """
    Grafica los retornos acumulados de cada activo.
    
    Args:
        all_returns: Diccionario con retornos por activo
        save_plots: Si guardar los gráficos como archivos
    """
    from pathlib import Path
    
    # Crear directorio de gráficos en test/plots/
    if save_plots:
        plots_dir = Path(__file__).parent / "test" / "plots"
        plots_dir.mkdir(exist_ok=True)
    
    plt.figure(figsize=(15, 8))
    
    for symbol, returns in all_returns.items():
        # Calcular retornos acumulados
        cumulative = (1 + returns).cumprod()
        plt.plot(cumulative.index, cumulative.values, label=symbol, linewidth=1.5)
    
    plt.title('Retornos Acumulados - Todos los Activos', fontsize=16, fontweight='bold')
    plt.xlabel('Fecha')
    plt.ylabel('Valor Acumulado (Base = 1)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Línea en 1 (sin cambio)
    plt.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(plots_dir / 'all_cumulative_returns.png', dpi=300, bbox_inches='tight')
        print(f"   📊 Gráfico guardado: all_cumulative_returns.png")
    
    plt.close()  # Liberar memoria


def plot_volatility_comparison(volatility: pd.Series, save_plots: bool = True) -> None:
    """
    Grafica comparación de volatilidades entre activos.
    
    Args:
        volatility: Serie con volatilidades por activo
        save_plots: Si guardar el gráfico como archivo
    """
    from pathlib import Path
    import seaborn as sns
    
    # Crear directorio de gráficos
    if save_plots:
        plots_dir = Path(__file__).parent / "test" / "plots"
        plots_dir.mkdir(exist_ok=True)
    
    # Configurar el gráfico
    plt.figure(figsize=(14, 8))
    
    # Ordenar volatilidades de mayor a menor
    vol_sorted = volatility.sort_values(ascending=False)
    
    # Crear gráfico de barras
    bars = plt.bar(range(len(vol_sorted)), vol_sorted.values, 
                   color=plt.cm.RdYlBu_r(vol_sorted.values / vol_sorted.max()),
                   edgecolor='black', linewidth=0.5)
    
    # Personalizar el gráfico
    plt.title('Volatilidad Anualizada por Activo', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Activos', fontsize=12)
    plt.ylabel('Volatilidad Anualizada', fontsize=12)
    
    # Etiquetas en el eje X
    plt.xticks(range(len(vol_sorted)), vol_sorted.index, rotation=45, ha='right')
    
    # Añadir valores sobre las barras
    for i, (symbol, value) in enumerate(vol_sorted.items()):
        plt.text(i, value + 0.01, f'{value:.2f}\n({value*100:.1f}%)', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Líneas de referencia
    plt.axhline(y=0.2, color='orange', linestyle='--', alpha=0.7, label='Volatilidad Moderada (20%)')
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Volatilidad Alta (50%)')
    
    # Leyenda y grid
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar gráfico
    if save_plots:
        plt.savefig(plots_dir / 'volatility_comparison.png', dpi=300, bbox_inches='tight')
        print(f"   📊 Gráfico guardado: volatility_comparison.png")
    
    plt.close()


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, save_plots: bool = True) -> None:
    """
    Grafica heatmap de la matriz de correlación.
    
    Args:
        corr_matrix: DataFrame con matriz de correlación
        save_plots: Si guardar el gráfico como archivo
    """
    from pathlib import Path
    import seaborn as sns
    
    # Crear directorio de gráficos
    if save_plots:
        plots_dir = Path(__file__).parent / "test" / "plots"
        plots_dir.mkdir(exist_ok=True)
    
    # Configurar el gráfico
    plt.figure(figsize=(14, 12))
    
    # Crear heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Máscara para mostrar solo triángulo inferior
    
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                fmt='.3f',
                cbar_kws={"shrink": .8},
                annot_kws={'size': 8})
    
    # Personalizar el gráfico
    plt.title('Matriz de Correlación entre Activos', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Activos', fontsize=12)
    plt.ylabel('Activos', fontsize=12)
    
    # Rotar etiquetas
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar gráfico
    if save_plots:
        plt.savefig(plots_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"   📊 Gráfico guardado: correlation_heatmap.png")
    
    plt.close()


def calculate_volatility(returns: pd.DataFrame, annualize: bool = True, trading_days: int = 252) -> pd.Series:
    """
    Calcula la volatilidad (desviación estándar) de los retornos.
    
    Args:
        returns: DataFrame con retornos
        annualize: Si anualizar la volatilidad
        trading_days: Número de días de trading por año (252 por defecto)
        
    Returns:
        Serie con volatilidad por activo
    """
    if returns.empty:
        raise ValueError("DataFrame de retornos está vacío")
    
    vol = returns.std()
    
    if annualize:
        vol = vol * np.sqrt(trading_days)
    
    return vol


def calculate_max_drawdown(prices: pd.DataFrame) -> pd.Series:
    """
    Calcula el máximo drawdown (máxima pérdida desde un pico).
    
    Args:
        prices: DataFrame con precios o retornos acumulados
        
    Returns:
        Serie con máximo drawdown por activo
    """
    if prices.empty:
        raise ValueError("DataFrame de precios está vacío")
    
    drawdowns = {}
    
    for column in prices.columns:
        series = prices[column].dropna()
        
        if len(series) == 0:
            drawdowns[column] = np.nan
            continue
        
        # Calcular picos históricos (máximos acumulados)
        peak = series.expanding().max()
        
        # Calcular drawdown
        drawdown = (series - peak) / peak
        
        # Máximo drawdown (más negativo)
        max_dd = drawdown.min()
        drawdowns[column] = max_dd
    
    return pd.Series(drawdowns)


def calculate_beta(returns: pd.DataFrame, market_returns: pd.Series, risk_free_rate: float = 0.0) -> pd.Series:
    """
    Calcula el beta de cada activo respecto al mercado.
    
    Args:
        returns: DataFrame con retornos de activos
        market_returns: Serie con retornos del mercado (benchmark)
        risk_free_rate: Tasa libre de riesgo anualizada
        
    Returns:
        Serie con beta por activo
        
    Notes:
        Beta = Cov(activo, mercado) / Var(mercado)
        Beta = 1: Mismo riesgo que el mercado
        Beta > 1: Más riesgo que el mercado
        Beta < 1: Menos riesgo que el mercado
    """
    if returns.empty:
        raise ValueError("DataFrame de retornos está vacío")
    
    if market_returns.empty:
        raise ValueError("Serie de retornos del mercado está vacía")
    
    # Ajustar por tasa libre de riesgo si es necesario
    risk_free_daily = risk_free_rate / 252 if risk_free_rate > 0 else 0
    
    betas = {}
    
    for column in returns.columns:
        asset_returns = returns[column].dropna()
        
        # Alinear fechas
        aligned_data = pd.concat([asset_returns, market_returns], axis=1, join='inner')
        aligned_data.columns = ['asset', 'market']
        aligned_data = aligned_data.dropna()
        
        if len(aligned_data) < 2:
            betas[column] = np.nan
            continue
        
        # Ajustar por tasa libre de riesgo
        excess_asset = aligned_data['asset'] - risk_free_daily
        excess_market = aligned_data['market'] - risk_free_daily
        
        # Calcular beta
        if excess_market.var() == 0:
            betas[column] = np.nan
        else:
            beta = excess_asset.cov(excess_market) / excess_market.var()
            betas[column] = beta
    
    return pd.Series(betas)


def calculate_sharpe_ratio(returns: pd.DataFrame, risk_free_rate: float = 0.0, annualize: bool = True, trading_days: int = 252) -> pd.Series:
    """
    Calcula el ratio de Sharpe.
    
    Args:
        returns: DataFrame con retornos
        risk_free_rate: Tasa libre de riesgo anualizada
        annualize: Si anualizar el ratio
        trading_days: Número de días de trading por año
        
    Returns:
        Serie con ratio de Sharpe por activo
        
    Notes:
        Sharpe = (Retorno - Tasa libre de riesgo) / Volatilidad
        Mayor Sharpe = Mejor rendimiento ajustado por riesgo
    """
    if returns.empty:
        raise ValueError("DataFrame de retornos está vacío")
    
    # Convertir tasa anual a diaria
    risk_free_daily = risk_free_rate / trading_days if risk_free_rate > 0 else 0
    
    # Calcular exceso de retorno
    excess_returns = returns - risk_free_daily
    
    # Calcular Sharpe ratio
    mean_excess = excess_returns.mean()
    std_returns = returns.std()
    
    sharpe = mean_excess / std_returns
    
    if annualize:
        sharpe = sharpe * np.sqrt(trading_days)
    
    return sharpe


def calculate_sortino_ratio(returns: pd.DataFrame, risk_free_rate: float = 0.0, target_return: float = 0.0, annualize: bool = True, trading_days: int = 252) -> pd.Series:
    """
    Calcula el ratio de Sortino (solo considera volatilidad negativa).
    
    Args:
        returns: DataFrame con retornos
        risk_free_rate: Tasa libre de riesgo anualizada
        target_return: Retorno objetivo (por defecto 0)
        annualize: Si anualizar el ratio
        trading_days: Número de días de trading por año
        
    Returns:
        Serie con ratio de Sortino por activo
        
    Notes:
        Sortino = (Retorno - Target) / Downside Deviation
        Solo penaliza la volatilidad hacia abajo
    """
    if returns.empty:
        raise ValueError("DataFrame de retornos está vacío")
    
    # Convertir tasas anuales a diarias
    risk_free_daily = risk_free_rate / trading_days if risk_free_rate > 0 else 0
    target_daily = target_return / trading_days if target_return > 0 else 0
    
    sortino_ratios = {}
    
    for column in returns.columns:
        series = returns[column].dropna()
        
        if len(series) == 0:
            sortino_ratios[column] = np.nan
            continue
        
        # Exceso de retorno sobre objetivo
        excess_returns = series.mean() - target_daily
        
        # Downside deviation (solo retornos negativos)
        downside_returns = series[series < target_daily]
        
        if len(downside_returns) == 0:
            downside_deviation = 0
        else:
            downside_deviation = np.sqrt(((downside_returns - target_daily) ** 2).mean())
        
        if downside_deviation == 0:
            sortino_ratios[column] = np.inf if excess_returns > 0 else np.nan
        else:
            sortino = excess_returns / downside_deviation
            if annualize:
                sortino = sortino * np.sqrt(trading_days)
            sortino_ratios[column] = sortino
    
    return pd.Series(sortino_ratios)


def calculate_calmar_ratio(returns: pd.DataFrame, prices: pd.DataFrame = None, annualize: bool = True, trading_days: int = 252) -> pd.Series:
    """
    Calcula el ratio de Calmar (retorno anualizado / máximo drawdown).
    
    Args:
        returns: DataFrame con retornos
        prices: DataFrame con precios (si no se proporciona, se calcula desde retornos)
        annualize: Si anualizar el retorno
        trading_days: Número de días de trading por año
        
    Returns:
        Serie con ratio de Calmar por activo
    """
    if returns.empty:
        raise ValueError("DataFrame de retornos está vacío")
    
    # Si no hay precios, calcular precios acumulados desde retornos
    if prices is None:
        prices = (1 + returns).cumprod()
    
    # Calcular retorno anualizado
    annual_returns = returns.mean()
    if annualize:
        annual_returns = annual_returns * trading_days
    
    # Calcular máximo drawdown
    max_drawdowns = calculate_max_drawdown(prices)
    
    # Calcular Calmar ratio
    calmar_ratios = annual_returns / abs(max_drawdowns)
    
    return calmar_ratios


def calculate_information_ratio(returns: pd.DataFrame, benchmark_returns: pd.Series) -> pd.Series:
    """
    Calcula el ratio de información (exceso de retorno / tracking error).
    
    Args:
        returns: DataFrame con retornos de activos
        benchmark_returns: Serie con retornos del benchmark
        
    Returns:
        Serie con ratio de información por activo
    """
    if returns.empty:
        raise ValueError("DataFrame de retornos está vacío")
    
    if benchmark_returns.empty:
        raise ValueError("Serie de retornos del benchmark está vacía")
    
    info_ratios = {}
    
    for column in returns.columns:
        asset_returns = returns[column].dropna()
        
        # Alinear fechas
        aligned_data = pd.concat([asset_returns, benchmark_returns], axis=1, join='inner')
        aligned_data.columns = ['asset', 'benchmark']
        aligned_data = aligned_data.dropna()
        
        if len(aligned_data) < 2:
            info_ratios[column] = np.nan
            continue
        
        # Calcular exceso de retorno
        excess_returns = aligned_data['asset'] - aligned_data['benchmark']
        
        # Tracking error (desviación estándar del exceso de retorno)
        tracking_error = excess_returns.std()
        
        if tracking_error == 0:
            info_ratios[column] = np.inf if excess_returns.mean() > 0 else np.nan
        else:
            info_ratio = excess_returns.mean() / tracking_error
            info_ratios[column] = info_ratio
    
    return pd.Series(info_ratios)



if __name__ == "__main__":
    print("📊 Funciones de Análisis Cuantitativo")
    print("=" * 50)
    print("Funciones disponibles:")
    print("\n🔢 Cálculo de Retornos:")
    print("- calculate_returns(): Retornos simples")
    print("- calculate_log_returns(): Retornos logarítmicos")
    
    print("\n📈 Estadísticas Descriptivas:")
    print("- calculate_skewness(): Asimetría de los retornos")
    print("- calculate_kurtosis(): Curtosis de los retornos")
    print("- calculate_percentiles(): Percentiles específicos")
    print("- calculate_descriptive_stats(): Estadísticas completas")
    print("- calculate_moments(): Primeros cuatro momentos")
    
    print("\n🔬 Pruebas Estadísticas:")
    print("- calculate_jarque_bera_test(): Test de normalidad")
    
    print("\n🔗 Correlación y Covarianza:")
    print("- calculate_correlation_matrix(): Matriz de correlación")
    print("- calculate_covariance_matrix(): Matriz de covarianza")
    
    print("\n📊 Estadísticas Móviles:")
    print("- calculate_rolling_statistics(): Estadísticas en ventana móvil")
    
    print("\n⚠️ Métricas de Riesgo:")
    print("- calculate_volatility(): Volatilidad (anualizada o diaria)")
    print("- calculate_max_drawdown(): Máximo drawdown")
    print("- calculate_beta(): Beta respecto al mercado")
    print("- calculate_value_at_risk(): Value at Risk (VaR)")
    print("- calculate_conditional_value_at_risk(): Conditional VaR (CVaR)")
    
    print("\n📈 Métricas de Rendimiento:")
    print("- calculate_sharpe_ratio(): Ratio de Sharpe")
    print("- calculate_sortino_ratio(): Ratio de Sortino")
    print("- calculate_calmar_ratio(): Ratio de Calmar")
    print("- calculate_information_ratio(): Ratio de información")
    
    print("\n📊 Datos y Análisis:")
    print("- get_symbols(): Lista de símbolos")
    print("- get_market_data(): Obtener datos del DataManager")
    print("- analyze_log_returns(): Análisis completo de retornos logarítmicos")
    
    print("\n📈 Gráficos:")
    print("- plot_returns_timeline(): Gráfica retornos a lo largo del tiempo")
    print("- plot_returns_distribution(): Gráfica distribución de retornos")
    print("- plot_cumulative_returns(): Gráfica retornos acumulados")
    print("- plot_volatility_comparison(): Gráfica comparación de volatilidades")
    print("- plot_correlation_heatmap(): Heatmap de matriz de correlación")
