from typing import Dict, List
import io
import logging
import urllib.request
import pandas as pd
from .macro_data_loader import MacroDataLoader
from ..tools.config import MACRO_FACTORS, MACRO_CORE_FACTORS, FRED_FACTORS

logger = logging.getLogger(__name__)

class MacroDataDownloader:  
    def __init__(
        self,
        factors_map: Dict[str, str] = None,
        core_factors: List[str] = None,
        fred_factors: Dict[str, str] = None,
    ):

        self.factors_map = factors_map if factors_map is not None else MACRO_FACTORS
        self.core_factors = core_factors if core_factors is not None else MACRO_CORE_FACTORS
        self.fred_factors = fred_factors if fred_factors is not None else FRED_FACTORS
        self.loader = MacroDataLoader()
    
    def download_factors(
        self,
        factor_names: List[str],
        start_date: str,
        end_date: str,
        progress: bool = False
    ) -> Dict[str, pd.Series]:

        results = {}
        tickers = []
        ticker_to_name = {}

        fred_names = []
        for name in factor_names:
            if name in self.fred_factors:
                fred_names.append(name)
            elif name in self.factors_map:
                ticker = self.factors_map[name]
                tickers.append(ticker)
                ticker_to_name[ticker] = name
            else:
                logger.warning(f"[Macro] Factor '{name}' not found")

        for name in fred_names:
            series_id = self.fred_factors[name]
            series = self._download_fred_series(series_id, start_date, end_date)
            if len(series) > 0:
                series.name = name
                results[name] = series
                logger.info(f"[Macro] FRED {series_id} -> {name}: {len(series)} obs")
            else:
                logger.warning(f"[Macro] FRED failed for {name} ({series_id}), "
                               f"adding to Yahoo batch as fallback")
                tickers.append('^IRX')
                ticker_to_name['^IRX'] = name

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
                        logger.warning(f"[Macro] No data: {name}")
                        
        except Exception as e:
            logger.error(f"[Macro] Error: {e}")
        
        return results
    
    def download_all_factors(
        self,
        start_date: str,
        end_date: str,
        progress: bool = False
    ) -> Dict[str, pd.Series]:

        logger.info(f"[Macro] Downloading all factors ({len(self.factors_map)})")
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

        logger.info(f"[Macro] Downloading core factors ({len(self.core_factors)})")
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

        logger.info(f"[Macro] Using {fallback_ticker} as fallback for {factor_name}")
        try:
            series = self.loader.download_single(fallback_ticker, start_date, end_date, progress)
            if normalize and len(series) > 0:
                series = series / series.iloc[0] * 100.0
            series.name = factor_name
            return series
        except Exception as e:
            logger.error(f"[Macro] Fallback error: {e}")
            return pd.Series(dtype=float, name=factor_name)
    
    def _extract_close_prices(self, data: pd.DataFrame) -> pd.DataFrame:

        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.get_level_values(0):
                return data['Close']
        if 'Close' in data.columns:
            return data[['Close']]
        return data
    
    def _extract_series(self, data: pd.DataFrame, ticker: str) -> pd.Series:

        try:
            if ticker in data.columns:
                return data[ticker].dropna()
            return pd.Series(dtype=float)
        except Exception:
            return pd.Series(dtype=float)

    def _download_fred_series(
        self,
        series_id: str,
        start_date: str,
        end_date: str,
    ) -> pd.Series:
        """Download a FRED series via public CSV (no API key)."""
        url = (
            f"https://fred.stlouisfed.org/graph/fredgraph.csv"
            f"?id={series_id}&cosd={start_date}&coed={end_date}"
        )
        try:
            req = urllib.request.Request(
                url, headers={'User-Agent': 'Mozilla/5.0'}
            )
            response = urllib.request.urlopen(req, timeout=20)
            csv_data = response.read().decode('utf-8')
            df = pd.read_csv(
                io.StringIO(csv_data),
                parse_dates=['DATE'],
                index_col='DATE',
            )
            series = df[series_id].replace('.', float('nan')).astype(float).dropna()
            series.index.name = None
            return series
        except Exception as e:
            logger.error(f"[Macro] Error downloading FRED {series_id}: {e}")
            return pd.Series(dtype=float)