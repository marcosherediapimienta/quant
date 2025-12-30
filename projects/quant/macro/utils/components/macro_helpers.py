from typing import Dict, List
import pandas as pd
from .macro_data_loader import MacroDataLoader
from ..tools.config import MACRO_FACTORS, MACRO_CORE_FACTORS


class MacroDataDownloader:
    """
    Descarga y procesa factores macroeconómicos.
    
    Responsabilidad única: Orquestar la descarga de factores macro desde fuentes.
    
    Funcionalidades:
    - Descarga de factores seleccionados
    - Descarga de todos los factores configurados
    - Descarga de factores core
    - Fallback automático a tickers alternativos
    - Extracción y limpieza de datos
    """
    
    def __init__(
        self,
        factors_map: Dict[str, str] = None,
        core_factors: List[str] = None
    ):
        """
        Inicializa el descargador.
        
        Args:
            factors_map: Mapeo {nombre_factor: ticker} (None = usar config)
            core_factors: Lista de factores core (None = usar config)
        """
        self.factors_map = factors_map if factors_map is not None else MACRO_FACTORS
        self.core_factors = core_factors if core_factors is not None else MACRO_CORE_FACTORS
        self.loader = MacroDataLoader()
    
    def download_factors(
        self,
        factor_names: List[str],
        start_date: str,
        end_date: str,
        progress: bool = False
    ) -> Dict[str, pd.Series]:
        """
        Descarga factores macroeconómicos seleccionados.
        
        Args:
            factor_names: Lista de nombres de factores a descargar
            start_date: Fecha inicio (formato 'YYYY-MM-DD')
            end_date: Fecha fin (formato 'YYYY-MM-DD')
            progress: Si mostrar barra de progreso
            
        Returns:
            Dict {nombre_factor: serie_temporal}
            
        Proceso:
        1. Mapea nombres a tickers
        2. Descarga datos con yfinance
        3. Extrae precios de cierre
        4. Retorna dict limpio
        """
        results = {}
        tickers = []
        ticker_to_name = {}
        
        for name in factor_names:
            if name not in self.factors_map:
                print(f"[Macro] Factor '{name}' no encontrado")
                continue
            ticker = self.factors_map[name]
            tickers.append(ticker)
            ticker_to_name[ticker] = name
        
        if not tickers:
            return results

        try:
            data = self.loader.download(tickers, start_date, end_date, progress)
            
            if len(tickers) == 1:
                ticker = tickers[0]
                name = ticker_to_name[ticker]
                if isinstance(data, pd.DataFrame) and 'Close' in data.columns:
                    series = data['Close'].dropna()
                else:
                    series = data.squeeze()
                
                if len(series) > 0:
                    series.name = name
                    results[name] = series
            else:
                close_data = self._extract_close_prices(data)
                for ticker, name in ticker_to_name.items():
                    series = self._extract_series(close_data, ticker)
                    if len(series) > 0:
                        series.name = name
                        results[name] = series
                    else:
                        print(f"[Macro] Sin datos: {name}")
                        
        except Exception as e:
            print(f"[Macro] Error: {e}")
        
        return results
    
    def download_all_factors(
        self,
        start_date: str,
        end_date: str,
        progress: bool = False
    ) -> Dict[str, pd.Series]:
        """
        Descarga TODOS los factores macro configurados.
        
        Args:
            start_date: Fecha inicio
            end_date: Fecha fin
            progress: Si mostrar progreso
            
        Returns:
            Dict con todos los factores disponibles
            
        ⚠️ Advertencia:
        - Puede ser lento (muchos factores)
        - Algunos factores pueden fallar o no tener datos
        """
        print(f"[Macro] Descargando todos los factores ({len(self.factors_map)})")
        return self.download_factors(
            list(self.factors_map.keys()),
            start_date,
            end_date,
            progress=progress
        )
    
    def download_core_factors(
        self,
        start_date: str,
        end_date: str,
        progress: bool = False
    ) -> Dict[str, pd.Series]:
        """
        Descarga factores CORE (más importantes).
        
        Args:
            start_date: Fecha inicio
            end_date: Fecha fin
            progress: Si mostrar progreso
            
        Returns:
            Dict con factores core
            
        Uso:
        - Para análisis rápido
        - Factores más líquidos y confiables
        - Definidos en MACRO_CORE_FACTORS del config
        """
        print(f"[Macro] Descargando factores core ({len(self.core_factors)})")
        return self.download_factors(
            self.core_factors,
            start_date,
            end_date,
            progress=progress
        )
    
    def download_with_fallback(
        self,
        factor_name: str,
        fallback_ticker: str,
        start_date: str,
        end_date: str,
        normalize: bool = True,
        progress: bool = False
    ) -> pd.Series:
        """
        Descarga un factor con fallback automático.
        
        Args:
            factor_name: Nombre del factor
            fallback_ticker: Ticker alternativo si falla
            start_date: Fecha inicio
            end_date: Fecha fin
            normalize: Si normalizar el fallback (base 100)
            progress: Si mostrar progreso
            
        Returns:
            Serie del factor (primario o fallback)
            
        Uso:
        - Para factores que pueden no estar disponibles
        - Ejemplo: índice propietario vs ETF proxy
        - El fallback se normaliza a base 100 si normalize=True
        """
        # Intentar descarga principal
        if factor_name in self.factors_map:
            try:
                series = self.loader.download_single(
                    self.factors_map[factor_name],
                    start_date,
                    end_date,
                    progress
                )
                if len(series) > 0:
                    series.name = factor_name
                    return series
            except Exception:
                pass

        # Usar fallback
        print(f"[Macro] Usando {fallback_ticker} como fallback para {factor_name}")
        try:
            series = self.loader.download_single(fallback_ticker, start_date, end_date, progress)
            if normalize and len(series) > 0:
                series = series / series.iloc[0] * 100.0
            series.name = factor_name
            return series
        except Exception as e:
            print(f"[Macro] Error con fallback: {e}")
            return pd.Series(dtype=float, name=factor_name)
    
    def _extract_close_prices(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extrae precios de cierre de DataFrame multi-columna.
        
        Args:
            data: DataFrame con datos de yfinance
            
        Returns:
            DataFrame solo con precios de cierre
        """
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.get_level_values(0):
                return data['Close']
        if 'Close' in data.columns:
            return data[['Close']]
        return data
    
    def _extract_series(self, data: pd.DataFrame, ticker: str) -> pd.Series:
        """
        Extrae serie individual de un DataFrame.
        
        Args:
            data: DataFrame con múltiples tickers
            ticker: Ticker a extraer
            
        Returns:
            Serie del ticker (limpia, sin NaN)
        """
        try:
            if ticker in data.columns:
                return data[ticker].dropna()
            return pd.Series(dtype=float)
        except Exception:
            return pd.Series(dtype=float)