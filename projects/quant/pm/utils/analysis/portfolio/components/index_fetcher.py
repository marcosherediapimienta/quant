# projects/quant/pm/utils/analysis/portfolio/components/index_fetcher.py
import pandas as pd
from typing import List, Dict
import yfinance as yf
import warnings
warnings.filterwarnings('ignore', category=pd.errors.Pandas4Warning, module='yfinance')

from ....tools.config import INDEX_CONFIG

class IndexFetcher:
    def __init__(
        self,
        urls: Dict[str, str] = None,
        user_agent: str = '',
        etf_mapping: Dict[str, str] = None,
        validation_thresholds: Dict[str, int] = None
    ):
        self.urls = urls if urls else INDEX_CONFIG['urls']
        self.user_agent = user_agent if user_agent else INDEX_CONFIG['user_agent']
        self.etf_mapping = etf_mapping if etf_mapping else INDEX_CONFIG['etf_mapping']
        self.supported_indices = INDEX_CONFIG['supported_indices']
        self.validation = validation_thresholds if validation_thresholds else INDEX_CONFIG['validation']
        self.fallback = INDEX_CONFIG['fallback']
    
    def get_index_components(self, index_name: str) -> List[str]:
        index_upper = index_name.upper()
        method_map = {
            'SP500': self._get_sp500,
            'NASDAQ100': self._get_nasdaq100,
            'DOW30': self._get_dow30,
            'RUSSELL1000': self._get_russell1000
        }
        
        method = method_map.get(index_upper)
        if method is None:
            raise ValueError(
                f"Índice '{index_name}' no soportado. "
                f"Disponibles: {', '.join(self.supported_indices)}"
            )
        
        return method()
    
    def _fetch_from_wikipedia(
        self,
        url: str,
        min_tickers: int = 0,
        max_tickers: int = 9999
    ) -> List[str]:

        try:
            tables = pd.read_html(url, storage_options={'User-Agent': self.user_agent})
            
            for table in tables:

                for col_name in ['Symbol', 'Ticker']:
                    if col_name in table.columns:
                        tickers = table[col_name].tolist()
                        tickers = [str(t).strip().replace('.', '-') 
                                   for t in tickers if pd.notna(t)]

                        if min_tickers <= len(tickers) <= max_tickers:
                            return tickers
            
            return []
        except Exception as e:
            print(f"⚠️  Error obteniendo datos: {e}")
            return []
    
    def _get_sp500(self) -> List[str]:
        tickers = self._fetch_from_wikipedia(
            self.urls['sp500'],
            min_tickers=self.validation['sp500_min_tickers']
        )
        
        if tickers:
            print(f"✅ Obtenidos {len(tickers)} componentes del S&P 500")
            return tickers

        print("    Usando lista fallback (top 100)...")
        return self.fallback['sp500_top100'].copy()
    
    def _get_nasdaq100(self) -> List[str]:
        tickers = self._fetch_from_wikipedia(
            self.urls['nasdaq100'],
            min_tickers=self.validation['nasdaq_min_tickers']
        )
        
        if tickers:
            print(f"✅ Obtenidos {len(tickers)} componentes del NASDAQ-100")
            return tickers
        
        print("⚠️  No se pudieron obtener componentes del NASDAQ-100")
        return []
    
    def _get_dow30(self) -> List[str]:
        tickers = self._fetch_from_wikipedia(
            self.urls['dow30'],
            min_tickers=self.validation['dow_min_tickers'],
            max_tickers=self.validation['dow_max_tickers']
        )
        
        if tickers:
            print(f"✅ Obtenidos {len(tickers)} componentes del Dow 30")
            return tickers
        
        print("⚠️  No se pudieron obtener componentes del Dow 30")
        return []
    
    def _get_russell1000(self) -> List[str]:
        print("⚠️  Russell 1000 completo no disponible públicamente")
        print("    Usando aproximación: S&P 500 + mid-caps principales")
        
        sp500 = self._get_sp500()
        additional = self.fallback['russell_additional'].copy()
        
        return sp500 + additional
    
    def get_etf_holdings(
        self,
        etf_ticker: str,
        max_holdings: int = 0
    ) -> List[str]:

        try:
            etf = yf.Ticker(etf_ticker)

            if hasattr(etf, 'holdings') and etf.holdings is not None:
                holdings = etf.holdings
                if not holdings.empty and 'Symbol' in holdings.columns:
                    tickers = holdings['Symbol'].tolist()
                    if max_holdings > 0:
                        tickers = tickers[:max_holdings]
                    print(f"✅ Obtenidos {len(tickers)} holdings de {etf_ticker}")
                    return tickers

            if etf_ticker.upper() in self.etf_mapping:
                index = self.etf_mapping[etf_ticker.upper()]
                print(f"    Usando mapeo {etf_ticker} → {index}")
                return self.get_index_components(index)
            
            return []
            
        except Exception as e:
            print(f"⚠️  Error obteniendo holdings de {etf_ticker}: {e}")
            return []
    
    def get_available_indices(self) -> List[str]:
        return self.supported_indices.copy()
    
    def get_available_etfs(self) -> List[str]:
        return list(self.etf_mapping.keys())