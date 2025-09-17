import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict, Union, Any
from datetime import datetime, timedelta
import warnings
import logging
from pathlib import Path
import json
import time

class DataManager:
    """
    Sistema de gestión de datos completamente configurable.
    Sin hardcodeo - todo se configura desde archivos externos.
    """
    
    def __init__(self, config_file: str = "config/data_config.yaml", test_mode: bool = False, test_dir: str = None):
        """
        Inicializa el Data Manager desde configuración externa.
        
        Args:
            config_file: Ruta al archivo de configuración
            test_mode: Si está en modo prueba, usa directorios locales
            test_dir: Directorio de prueba (solo si test_mode=True)
        """
        self.config = self._load_config(config_file)
        self._setup_from_config()
        
        # Configurar logging dinámico desde configuración
        log_level = self.config.get('logging_level', 'INFO')
        level = getattr(logging, log_level.upper(), logging.INFO)
        logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level)
        
        # Modo prueba: usar directorios locales
        if test_mode and test_dir:
            self.test_dir = Path(test_dir)
            self.plots_dir = self.test_dir / "plots"
            self.cache_dir = self.test_dir / "cache"
            
            # Crear directorios si no existen
            self.plots_dir.mkdir(exist_ok=True)
            self.cache_dir.mkdir(exist_ok=True)
            
            # Sobrescribir directorio de datos para pruebas
            self.data_dir = self.test_dir
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Carga la configuración desde archivo YAML."""
        config_path = Path(config_file)
        
        if not config_path.exists():
            # Crear configuración por defecto si no existe
            self._create_default_config(config_path)
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    try:
                        import yaml
                        return yaml.safe_load(f)
                    except ImportError:
                        print("⚠️ PyYAML no instalado, usando configuración por defecto")
                        return self._get_default_config()
                else:
                    return json.load(f)
        except Exception as e:
            print(f"Error cargando configuración: {e}")
            return self._get_default_config()
    
    def _create_default_config(self, config_path: Path):
        """Crea archivo de configuración por defecto."""
        config_path.parent.mkdir(exist_ok=True)
        
        default_config = self._get_default_config()
        
        with open(config_path, 'w', encoding='utf-8') as f:
            if config_path.suffix in ['.yaml', '.yml']:
                try:
                    import yaml
                    yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
                except ImportError:
                    json.dump(default_config, f, indent=2, ensure_ascii=False)
            else:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Archivo de configuración creado: {config_path}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Retorna configuración por defecto."""
        return {
            'data_sources': {
                'primary': 'yfinance',
                'fallback': 'csv_local'
            },
            'data_storage': {
                'cache_enabled': True,
                'data_directory': 'data',
                'cache_expiry_days': 7
            },
            'data_quality': {
                'min_data_days': 252 * 3,
                'max_missing_data_pct': 5.0,
                'outlier_threshold': 3.0
            },
            'download_settings': {
                'max_retries': 3,
                'retry_delay_seconds': 2,
                'request_timeout': 30
            },
            'logging_level': 'INFO'
        }
    
    def _setup_from_config(self):
        """Configura el Data Manager desde la configuración cargada."""
        config = self.config
        
        # Configuración de almacenamiento
        storage_config = config.get('data_storage', {})
        data_dir = storage_config.get('data_directory', 'data')
        
        # Si es una ruta relativa, construir desde la ubicación del DataManager
        if not Path(data_dir).is_absolute():
            self.data_dir = Path(__file__).parent / data_dir
        else:
            self.data_dir = Path(data_dir)
            
        self.data_dir.mkdir(exist_ok=True)
        self.cache_enabled = storage_config.get('cache_enabled', True)
        self.cache_expiry_days = storage_config.get('cache_expiry_days', 7)
        
        # Configuración de calidad
        quality_config = config.get('data_quality', {})
        self.min_data_days = quality_config.get('min_data_days', 252 * 3)
        self.max_missing_data_pct = quality_config.get('max_missing_data_pct', 5.0)
        self.outlier_threshold = quality_config.get('outlier_threshold', 3.0)
        
        # Configuración de descarga
        download_config = config.get('download_settings', {})
        self.max_retries = download_config.get('max_retries', 3)
        self.retry_delay = download_config.get('retry_delay_seconds', 2)
        self.timeout = download_config.get('request_timeout', 30)
    
    def load_universe_from_file(self, file_path: str) -> List[str]:
        """
        Carga lista de símbolos desde archivo CSV o TXT.
        
        Args:
            file_path: Ruta al archivo con símbolos
            
        Returns:
            Lista de símbolos
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            self.logger.error(f"Archivo no encontrado: {file_path}")
            return []
        
        try:
            if file_path.suffix.lower() == '.csv':
                # Leer CSV - buscar columna con símbolos
                df = pd.read_csv(file_path)
                
                # Buscar columnas posibles para símbolos
                possible_cols = ['symbol', 'ticker', 'Symbol', 'Ticker', 'SYMBOL', 'TICKER']
                symbol_col = None
                
                for col in possible_cols:
                    if col in df.columns:
                        symbol_col = col
                        break
                
                if symbol_col is None:
                    # Si no encuentra columna específica, usar la primera columna
                    symbol_col = df.columns[0]
                    self.logger.warning(f"No se encontró columna de símbolos, usando: {symbol_col}")
                
                symbols = df[symbol_col].astype(str).str.strip().tolist()
                
            elif file_path.suffix.lower() == '.txt':
                # Leer TXT línea por línea
                with open(file_path, 'r', encoding='utf-8') as f:
                    symbols = [line.strip() for line in f if line.strip()]
                    
            else:
                self.logger.error(f"Formato de archivo no soportado: {file_path.suffix}")
                return []
            
            # Filtrar símbolos vacíos y duplicados
            symbols = [s for s in symbols if s and s.upper() != 'NAN']
            symbols = list(dict.fromkeys(symbols))  # Eliminar duplicados manteniendo orden
            
            self.logger.info(f"📋 Cargados {len(symbols)} símbolos desde {file_path.name}")
            return symbols
            
        except Exception as e:
            self.logger.error(f"Error cargando universo desde {file_path}: {e}")
            return []
    
    def download_market_data(self, 
                           symbols: Union[List[str], str], 
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           force_refresh: bool = False) -> pd.DataFrame:
        """
        Descarga datos de mercado.
        
        Args:
            symbols: Lista de símbolos o ruta a archivo de universo
            start_date: Fecha de inicio (YYYY-MM-DD)
            end_date: Fecha de fin (YYYY-MM-DD)
            force_refresh: Forzar descarga nueva
            
        Returns:
            DataFrame con precios
        """
        # Si symbols es un string, asumir que es un archivo
        if isinstance(symbols, str):
            # Construir ruta relativa desde la ubicación del DataManager
            if not Path(symbols).is_absolute():
                symbols = str(Path(__file__).parent / symbols)
            symbols = self.load_universe_from_file(symbols)
        
        if start_date is None:
            # Por defecto: 10 años hacia atrás desde hoy
            start_date = (datetime.now() - timedelta(days=10*365)).strftime("%Y-%m-%d")
        if end_date is None:
            # Por defecto: hoy
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        self.logger.info(f"📥 Descargando {len(symbols)} símbolos")
        
        # Verificar caché
        if self.cache_enabled and not force_refresh:
            cached_data = self._load_from_cache(symbols, start_date, end_date)
            if cached_data is not None:
                return cached_data
        
        # Descargar datos - intentar descarga masiva primero
        data = {}
        failed_symbols = []
        
        # Intentar descarga masiva con yfinance.download
        try:
            self.logger.debug("Intentando descarga masiva con yfinance.download...")
            bulk_data = yf.download(symbols, start=start_date, end=end_date, group_by='ticker', 
                                   auto_adjust=True, prepost=True, threads=True)
            
            # Procesar datos descargados en masa
            if not bulk_data.empty:
                if len(symbols) == 1:
                    # Un solo símbolo
                    symbol = symbols[0]
                    if 'Close' in bulk_data.columns:
                        data[symbol] = bulk_data['Close']
                    elif 'Adj Close' in bulk_data.columns:
                        data[symbol] = bulk_data['Adj Close']
                    else:
                        # Si no hay Close ni Adj Close, usar la primera columna numérica disponible
                        numeric_cols = bulk_data.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            data[symbol] = bulk_data[numeric_cols[0]]
                        else:
                            failed_symbols.append(symbol)
                else:
                    # Múltiples símbolos
                    for symbol in symbols:
                        try:
                            if hasattr(bulk_data.columns, 'get_level_values') and symbol in bulk_data.columns.get_level_values(0):
                                symbol_data = bulk_data[symbol]
                                if 'Close' in symbol_data.columns:
                                    data[symbol] = symbol_data['Close']
                                elif 'Adj Close' in symbol_data.columns:
                                    data[symbol] = symbol_data['Adj Close']
                                else:
                                    failed_symbols.append(symbol)
                            else:
                                failed_symbols.append(symbol)
                        except Exception:
                            failed_symbols.append(symbol)
                            
                self.logger.info(f"✅ Descarga masiva exitosa: {len(data)} símbolos")
            else:
                self.logger.warning("Descarga masiva falló, usando descarga individual")
                failed_symbols = symbols.copy()
                
        except Exception as e:
            self.logger.warning(f"Descarga masiva falló: {e}, usando descarga individual")
            failed_symbols = symbols.copy()
        
        # Fallback: descarga individual para símbolos fallidos
        if failed_symbols:
            self.logger.info(f"🔄 Descarga individual para {len(failed_symbols)} símbolos fallidos")
            remaining_failed = []
            
            for symbol in failed_symbols:
                try:
                    ticker_data = self._download_single_symbol(symbol, start_date, end_date)
                    
                    if ticker_data is not None:
                        data[symbol] = ticker_data
                    else:
                        remaining_failed.append(symbol)
                        
                except Exception as e:
                    self.logger.error(f"Error descargando {symbol}: {e}")
                    remaining_failed.append(symbol)
            
            failed_symbols = remaining_failed
        
        if not data:
            raise ValueError(f"No se pudieron descargar datos. Fallidos: {failed_symbols}")
        
        # Crear DataFrame y limpieza con interpolación
        df = pd.DataFrame(data)
        
        # Interpolación inteligente: preservar más datos
        initial_rows = len(df)
        
        # Eliminar solo filas completamente vacías
        df = df.dropna(how='all')
        
        # Interpolación lineal para gaps pequeños (hasta 5 días)
        df = df.interpolate(method='linear', limit=5)
        
        # Forward fill para gaps restantes (hasta 10 días)
        df = df.ffill(limit=10)
        
        # Backward fill para gaps al inicio
        df = df.bfill(limit=5)
        
        # Eliminar columnas completamente vacías después de interpolación
        df = df.dropna(axis=1, how='all')
        
        # Solo eliminar filas con demasiados NaN (más del 20% de activos)
        if len(df.columns) > 0:
            max_missing = len(df.columns) * 0.2
            df = df.dropna(thresh=len(df.columns) - max_missing)
        
        # Guardar en caché
        if self.cache_enabled:
            self._save_to_cache(df, symbols, start_date, end_date)
        
        return df
    
    def download_market_data_grouped(self, 
                                   symbols: Union[List[str], str], 
                                   target_years: int = 10,
                                   force_refresh: bool = False) -> Dict[str, Union[pd.DataFrame, List[str]]]:
        """
        Descarga datos de mercado agrupando activos por disponibilidad histórica.
        
        Args:
            symbols: Lista de símbolos o ruta a archivo de universo
            target_years: Años objetivo para datos históricos
            force_refresh: Forzar descarga nueva
            
        Returns:
            Diccionario con DataFrames agrupados por disponibilidad
        """
        # Si symbols es un string, asumir que es un archivo
        if isinstance(symbols, str):
            if not Path(symbols).is_absolute():
                symbols = str(Path(__file__).parent / symbols)
            symbols = self.load_universe_from_file(symbols)
        
        # Fecha objetivo: target_years hacia atrás
        target_start = (datetime.now() - timedelta(days=target_years*365)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        self.logger.info(f"🎯 Objetivo: {target_years} años desde {target_start}")
        
        # Probar disponibilidad de cada símbolo
        availability_info = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                # Probar con fecha MUY antigua para ver qué hay disponible
                test_data = ticker.history(start="1900-01-01", end=end_date)
                
                if not test_data.empty:
                    actual_start = test_data.index[0].strftime('%Y-%m-%d')
                    actual_end = test_data.index[-1].strftime('%Y-%m-%d')
                    days_available = len(test_data)
                    
                    # Calcular años reales disponibles
                    years_available = days_available / 252
                    
                    availability_info[symbol] = {
                        'start_date': actual_start,
                        'end_date': actual_end,
                        'days_available': days_available,
                        'years_available': years_available,
                        'has_target_years': years_available >= target_years
                    }
                else:
                    availability_info[symbol] = {'error': 'Sin datos'}
                    
            except Exception as e:
                availability_info[symbol] = {'error': str(e)}
        
        # Agrupar símbolos por disponibilidad
        long_history = []  # 8+ años
        medium_history = []  # 3-8 años  
        short_history = []   # <3 años
        failed_symbols = []
        
        for symbol, info in availability_info.items():
            if 'error' in info:
                failed_symbols.append(symbol)
            elif info['years_available'] >= target_years:
                long_history.append(symbol)
            elif info['years_available'] >= 3:  # 3+ años para historia media
                medium_history.append(symbol)
            else:
                short_history.append(symbol)
        
        # Descargar datos para cada grupo
        results = {}
        
        if long_history:
            results['long_history'] = self.download_market_data(
                symbols=long_history, 
                start_date=target_start, 
                end_date=end_date,
                force_refresh=force_refresh
            )
        
        if medium_history:
            # Usar 5 años para este grupo
            medium_start = (datetime.now() - timedelta(days=5*365)).strftime("%Y-%m-%d")
            results['medium_history'] = self.download_market_data(
                symbols=medium_history, 
                start_date=medium_start, 
                end_date=end_date,
                force_refresh=force_refresh
            )
        
        if short_history:
            # Usar 2 años para este grupo
            short_start = (datetime.now() - timedelta(days=2*365)).strftime("%Y-%m-%d")
            results['short_history'] = self.download_market_data(
                symbols=short_history, 
                start_date=short_start, 
                end_date=end_date,
                force_refresh=force_refresh
            )
        
        if failed_symbols:
            results['failed'] = failed_symbols
        
        # Agregar información de fechas individuales para cada grupo
        results['availability_info'] = availability_info
        
        return results
    
    def process_and_display_groups(self, grouped_data: Dict[str, Union[pd.DataFrame, List[str]]], 
                                 target_years: int = 10) -> List[str]:
        """
        Procesa y muestra los grupos de datos agrupados por disponibilidad.
        
        Args:
            grouped_data: Resultado de download_market_data_grouped
            target_years: Años objetivo para la descarga
            
        Returns:
            Lista de rutas de gráficos generados
        """
        all_plots = []
        availability_info = grouped_data.get('availability_info', {})
        
        # Mapeo de grupos con descripciones
        group_info = {
            'long_history': f'LARGA HISTORIA (≥{target_years} años)',
            'medium_history': 'HISTORIA MEDIA (3-8 años)',
            'short_history': 'HISTORIA CORTA (<3 años)'
        }
        
        for group_key, description in group_info.items():
            if group_key in grouped_data and isinstance(grouped_data[group_key], pd.DataFrame):
                prices = grouped_data[group_key]
                print(f"\n📊 GRUPO DE {description}:")
                print(f"   📈 {len(prices.columns)} símbolos")
                
                # Mostrar fechas individuales
                print(f"   📅 FECHAS INDIVIDUALES:")
                for symbol in prices.columns:
                    if symbol in availability_info and 'start_date' in availability_info[symbol]:
                        start_date = availability_info[symbol]['start_date']
                        end_date = availability_info[symbol]['end_date']
                        years = availability_info[symbol].get('years_available', 0)
                        print(f"      {symbol:>8}: {start_date} → {end_date} ({years:.1f} años)")
                
                # Mostrar precios actuales vs históricos
                print(f"\n💰 PRECIOS ACTUALES vs HISTÓRICOS:")
                current_prices = prices.iloc[-1]
                
                for symbol in prices.columns:
                    current = current_prices[symbol]
                    symbol_data = prices[symbol].dropna()
                    historical = symbol_data.iloc[0] if len(symbol_data) > 0 else np.nan
                    
                    current_str = f'${current:>10.2f}' if pd.notna(current) else 'N/A'
                    historical_str = f'${historical:>10.2f}' if pd.notna(historical) else 'N/A'
                    
                    print(f"   {symbol:>8}: {current_str:>12} ({historical_str:>12})")
                
                # Generar gráficos
                plot_path = self.plot_price_history(prices=prices, save_plot=True, show_plot=False)
                if plot_path:
                    all_plots.append(plot_path)
                
                individual_plots = self.plot_individual_assets(
                    prices=prices, 
                    symbols=prices.columns, 
                    save_plots=True, 
                    show_plots=False
                )
                all_plots.extend(individual_plots)
        
        # Gestionar símbolos fallidos
        failed_symbols = grouped_data.get('failed', [])
        if failed_symbols:
            print(f"\n❌ SÍMBOLOS FALLIDOS ({len(failed_symbols)} total):")
            for symbol in failed_symbols:
                print(f"   ❌ {symbol}")
        
        return all_plots
    
    @staticmethod
    def parse_arguments():
        """Parsea argumentos de línea de comandos."""
        import argparse
        parser = argparse.ArgumentParser(
            description='Prueba del DataManager con agrupación por disponibilidad'
        )
        parser.add_argument(
            '--force-refresh', 
            action='store_true',
            help='Fuerza la actualización de datos (ignora cache)'
        )
        parser.add_argument(
            '--years', 
            type=int, 
            default=10,
            help='Número de años objetivo para la descarga (por defecto: 10)'
        )
        return parser.parse_args()
    
    def _download_single_symbol(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.Series]:
        """Descarga datos para un símbolo individual."""
        for attempt in range(self.max_retries):
            try:
                ticker = yf.Ticker(symbol)
                ticker_data = ticker.history(start=start_date, end=end_date)
                
                if not ticker_data.empty:
                    # Priorizar 'Close' ya que 'Adj Close' no siempre está disponible
                    if 'Close' in ticker_data.columns:
                        return ticker_data['Close']
                    elif 'Adj Close' in ticker_data.columns:
                        return ticker_data['Adj Close']
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise e
        
        return None
    
    def _load_from_cache(self, symbols: List[str], start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Carga datos desde caché."""
        if not self.cache_enabled:
            return None
        
        # Usar directorio de caché de prueba si está disponible
        cache_dir = getattr(self, 'cache_dir', self.data_dir)
        
        # Buscar archivos de caché que coincidan con el período
        cache_pattern = f"cache_{start_date}_{end_date}_*_assets*.csv"
        cache_files = list(cache_dir.glob(cache_pattern))
        
        if cache_files:
            # Tomar el archivo más reciente
            latest_cache = max(cache_files, key=lambda x: x.stat().st_mtime)
            
            # Verificar expiración
            file_age = datetime.now() - datetime.fromtimestamp(latest_cache.stat().st_mtime)
            if file_age.days <= self.cache_expiry_days:
                try:
                    # Optimización: cargar solo las columnas necesarias más el índice
                    available_cols = pd.read_csv(latest_cache, nrows=0).columns.tolist()
                    requested_symbols_list = list(symbols)
                    cols_to_load = [col for col in requested_symbols_list if col in available_cols]
                    
                    if cols_to_load:
                        # Obtener el nombre de la columna del índice
                        index_col_name = available_cols[0] if available_cols else 'Date'
                        usecols_list = [index_col_name] + cols_to_load
                        df = pd.read_csv(latest_cache, index_col=0, parse_dates=True, usecols=usecols_list)
                    else:
                        df = pd.read_csv(latest_cache, index_col=0, parse_dates=True)
                    
                    # Verificar que todos los símbolos solicitados estén en el caché
                    available_symbols = set(df.columns)
                    requested_symbols = set(symbols)
                    
                    if requested_symbols.issubset(available_symbols):
                        # Filtrar solo los símbolos solicitados usando usecols optimizado
                        filtered_df = df[list(requested_symbols)]
                        self.logger.info(f"✅ Datos cargados desde caché: {len(requested_symbols)} símbolos")
                        return filtered_df
                    else:
                        missing_symbols = requested_symbols - available_symbols
                        self.logger.info(f"⚠️ Caché incompleto, faltan: {missing_symbols}")
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Error leyendo caché: {e}")
        
        return None
    
    def _save_to_cache(self, df: pd.DataFrame, symbols: List[str], start_date: str, end_date: str):
        """Guarda datos en caché."""
        if not self.cache_enabled:
            return
        
        try:
            # Usar directorio de caché de prueba si está disponible
            cache_dir = getattr(self, 'cache_dir', self.data_dir)
            
            # Crear nombre de archivo con timestamp para evitar conflictos
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            cache_file = cache_dir / f"cache_{start_date}_{end_date}_{len(symbols)}_assets_{timestamp}.csv"
            
            # Guardar datos
            df.to_csv(cache_file)
            
            # Limpiar archivos de caché antiguos (mantener solo los últimos 5)
            self._cleanup_old_cache_files(start_date, end_date)
            
        except Exception as e:
            pass  # Silenciar errores de caché
    
    def _cleanup_old_cache_files(self, start_date: str, end_date: str):
        """Limpia archivos de caché antiguos, manteniendo solo los últimos 5."""
        try:
            # Usar directorio de caché de prueba si está disponible
            cache_dir = getattr(self, 'cache_dir', self.data_dir)
            
            cache_pattern = f"cache_{start_date}_{end_date}_*_assets*.csv"
            cache_files = list(cache_dir.glob(cache_pattern))
            
            if len(cache_files) > 5:
                # Ordenar por fecha de modificación y eliminar los más antiguos
                cache_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                files_to_delete = cache_files[5:]
                
                for old_file in files_to_delete:
                    old_file.unlink()
                    
        except Exception as e:
            pass  # Silenciar errores de limpieza
    
    def plot_price_history(self, prices: pd.DataFrame, symbols: Optional[List[str]] = None, 
                          save_plot: bool = True, show_plot: bool = False, log_scale: bool = False) -> str:
        """
        Genera gráfico de precios normalizados para comparar activos.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib import rcParams
            
            # Configurar estilo profesional
            rcParams['figure.figsize'] = (16, 10)
            rcParams['font.size'] = 12
            rcParams['axes.grid'] = True
            rcParams['grid.alpha'] = 0.4
            
            # Filtrar símbolos si se especifican
            if symbols is not None and len(symbols) > 0:
                plot_data = prices[symbols]
            else:
                plot_data = prices
            
            # Crear figura con solo 1 gráfico
            fig, ax = plt.subplots(1, 1, figsize=(16, 10))
            
            # Calcular precios normalizados (base 100) - NO retornos
            normalized_prices = (plot_data / plot_data.iloc[0]) * 100
            
            # Colores más contrastantes
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                     '#aec7e8', '#ffbb78', '#98df8a', '#ff9896']
            
            # Graficar cada activo
            for i, symbol in enumerate(normalized_prices.columns):
                color = colors[i % len(colors)]
                ax.plot(normalized_prices.index, normalized_prices[symbol], 
                       label=symbol, linewidth=2.5, color=color)
            
            # Configurar escala logarítmica si se solicita
            if log_scale:
                ax.set_yscale('log')
                ax.set_ylabel('Precio Normalizado - Escala Log', fontsize=14)
            else:
                ax.set_ylabel('Precio Normalizado', fontsize=14)
            
            # Configurar título y etiquetas
            scale_text = ' - ESCALA LOGARÍTMICA' if log_scale else ''
            ax.set_title(f'Precios Normalizados{scale_text}', 
                        fontsize=18, fontweight='bold', pad=20)
            ax.set_xlabel('Fecha', fontsize=14)
            
            # Configurar leyenda
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
                     fontsize=11, framealpha=0.9)
            
            # Configurar grid
            ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
            
            # Línea de referencia en 100
            ax.axhline(y=100, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
            
            # Formatear eje X
            ax.xaxis.set_major_locator(mdates.YearLocator(1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            
            # Ajustar layout
            plt.tight_layout()
            
            # Guardar gráfico
            if save_plot:
                if hasattr(self, 'plots_dir'):
                    plots_dir = self.plots_dir
                else:
                    plots_dir = self.data_dir / 'plots'
                    plots_dir.mkdir(exist_ok=True)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"prices_normalized_{timestamp}.png"
                filepath = plots_dir / filename
                
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                
                if show_plot:
                    plt.show()
                else:
                    plt.close()
                
                return str(filepath)
            
            elif show_plot:
                plt.show()
                return ""
            
            else:
                plt.close()
                return ""
                
        except ImportError:
            return ""
        except Exception as e:
            return ""
    
    def plot_individual_assets(self, prices: pd.DataFrame, symbols: Optional[List[str]] = None,
                              save_plots: bool = True, show_plots: bool = False) -> List[str]:
        """
        Genera gráficos individuales para cada activo.
        
        Args:
            prices: DataFrame con precios históricos
            symbols: Lista de símbolos a graficar (si None, grafica todos)
            save_plots: Si guardar los gráficos como imágenes
            show_plots: Si mostrar los gráficos en pantalla
            
        Returns:
            Lista de rutas de archivos guardados
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib import rcParams
            
            # Configurar estilo
            rcParams['figure.figsize'] = (12, 6)
            rcParams['font.size'] = 10
            rcParams['axes.grid'] = True
            rcParams['grid.alpha'] = 0.3
            
            # Filtrar símbolos
            if symbols is not None and len(symbols) > 0:
                plot_data = prices[symbols]
            else:
                plot_data = prices
            
            saved_files = []
            
            for symbol in plot_data.columns:
                # Crear figura con un solo gráfico
                fig, ax = plt.subplots(1, 1, figsize=(12, 6))
                
                # Gráfico: Precio histórico
                ax.plot(plot_data.index, plot_data[symbol], color='blue', linewidth=2)
                ax.set_title(f'{symbol} - Precio Histórico', fontsize=14, fontweight='bold')
                ax.set_ylabel('Precio ($)', fontsize=12)
                ax.set_xlabel('Fecha', fontsize=12)
                ax.grid(True, alpha=0.3)
                
                # Formatear eje X
                ax.xaxis.set_major_locator(mdates.YearLocator(1))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                
                plt.tight_layout()
                
                # Guardar gráfico
                if save_plots:
                    # Usar directorio de plots de prueba si está disponible, sino crear uno
                    if hasattr(self, 'plots_dir'):
                        plots_dir = self.plots_dir
                    else:
                        plots_dir = self.data_dir / 'plots'
                        plots_dir.mkdir(exist_ok=True)
                    
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"{symbol}_analysis_{timestamp}.png"
                    filepath = plots_dir / filename
                    
                    plt.savefig(filepath, dpi=300, bbox_inches='tight')
                    saved_files.append(str(filepath))
                    
                    if show_plots:
                        plt.show()
                    else:
                        plt.close()
                
                elif show_plots:
                    plt.show()
                    plt.close()
                
                else:
                    plt.close()
            
            return saved_files
            
        except ImportError:
            return []
        except Exception as e:
            return []


if __name__ == "__main__":
    # Script de prueba simple
    print("📊 Data Manager - Sistema Simple")
    print("=" * 40)
    
    # Crear instancia
    dm = DataManager()
    
    # Probar funcionalidad básica
    print("✅ Sistema inicializado")
    print(f"📁 Directorio: {dm.data_dir}")
    print(f"⚙️ Caché: {'Sí' if dm.cache_enabled else 'No'}")
