import pandas as pd
from typing import List, Dict
import yfinance as yf
import requests
from io import StringIO

import warnings
try:
    warnings.filterwarnings('ignore', category=pd.errors.Pandas4Warning, module='yfinance')
except AttributeError:
    pass

from ....tools.config import INDEX_CONFIG

class IndexFetcher:
    def __init__(
        self,
        urls: Dict[str, str] = None,
        user_agent: str = None,
        etf_mapping: Dict[str, str] = None,
        validation_thresholds: Dict[str, int] = None,
    ):
        self.urls = urls if urls is not None else INDEX_CONFIG['urls']
        self.user_agent = user_agent if user_agent is not None else INDEX_CONFIG['user_agent']
        self.etf_mapping = etf_mapping if etf_mapping is not None else INDEX_CONFIG['etf_mapping']
        self.supported_indices = INDEX_CONFIG['supported_indices']
        self.validation = validation_thresholds if validation_thresholds is not None else INDEX_CONFIG['validation']
        self.fallback = INDEX_CONFIG['fallback']
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent})

    def _normalize_ticker(self, ticker: str) -> str:
        symbol = str(ticker).strip().upper()
        
        if '.' not in symbol:
            return symbol

        exchange_suffixes = {
            'MC', 'PA', 'DE', 'T', 'AS', 'SW', 'L', 'MI', 'CO', 'ST',
            'HE', 'OL', 'VI', 'BR', 'LS', 'WA', 'IR', 'HK', 'SI', 'AX',
            'TO', 'SA',
        }
        _, suffix = symbol.rsplit('.', 1)
        if suffix in exchange_suffixes:
            return symbol

        return symbol.replace('.', '-')

    def _ensure_ibex_suffixes(self, tickers: List[str]) -> List[str]:
        normalized = []
        for ticker in tickers:
            symbol = str(ticker).strip().upper()
            if '.' not in symbol and '-' not in symbol:
                symbol = f'{symbol}.MC'
            normalized.append(symbol)
        return list(dict.fromkeys(normalized))
    
    def get_index_components(self, index_name: str) -> List[str]:
        index_upper = index_name.upper()
        method_map = {
            'SP500': self._get_sp500,
            'NASDAQ100': self._get_nasdaq100,
            'DOW30': self._get_dow30,
            'IBEX35': self._get_ibex35,
            'EUROSTOXX50': self._get_eurostoxx50,
            'NIKKEI225': self._get_nikkei225,
            'MSCI_WORLD': self._get_msci_world,
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
        max_tickers: int = None,
    ) -> List[str]:

        try:
            response = self.session.get(
                url,
                timeout=20,
            )
            response.raise_for_status()
            tables = pd.read_html(StringIO(response.text))
            
            for table in tables:

                for col_name in ['Symbol', 'Ticker', 'Code', 'Ticker symbol', 'EPIC']:
                    if col_name in table.columns:
                        tickers = table[col_name].tolist()
                        tickers = [self._normalize_ticker(t)
                                   for t in tickers if pd.notna(t)]
                        tickers = list(dict.fromkeys(tickers))

                        if len(tickers) >= min_tickers and (max_tickers is None or len(tickers) <= max_tickers):
                            return tickers
            
            return []
        except Exception as e:
            print(f" Error obteniendo datos: {e}")
            return []
    
    def _get_sp500(self) -> List[str]:
        tickers = self._fetch_from_wikipedia(
            self.urls['sp500'],
            min_tickers=self.validation['sp500_min_tickers']
        )
        
        if tickers:
            print(f" Obtenidos {len(tickers)} componentes del S&P 500")
            return tickers

        print("    Usando lista fallback (S&P 500 curada)...")
        return self.fallback['sp500'].copy()
    
    def _get_nasdaq100(self) -> List[str]:
        tickers = self._fetch_from_wikipedia(
            self.urls['nasdaq100'],
            min_tickers=self.validation['nasdaq_min_tickers']
        )
        
        if tickers:
            print(f" Obtenidos {len(tickers)} componentes del NASDAQ-100")
            return tickers
        
        print(" No se pudieron obtener componentes del NASDAQ-100")
        return []
    
    def _get_dow30(self) -> List[str]:
        tickers = self._fetch_from_wikipedia(
            self.urls['dow30'],
            min_tickers=self.validation['dow_min_tickers'],
            max_tickers=self.validation['dow_max_tickers']
        )
        
        if tickers:
            print(f" Obtenidos {len(tickers)} componentes del Dow 30")
            return tickers
        
        print(" No se pudieron obtener componentes del Dow 30")
        return []
    
    def _get_ibex35(self) -> List[str]:
        tickers = self._fetch_from_wikipedia(
            self.urls['ibex35'],
            min_tickers=self.validation['ibex35_min_tickers'],
            max_tickers=self.validation['ibex35_max_tickers']
        )

        if tickers:
            tickers = self._ensure_ibex_suffixes(tickers)
            print(f" Obtenidos {len(tickers)} componentes del IBEX 35")
            return tickers

        print(" IBEX 35: usando lista fallback")
        return self.fallback['ibex35'].copy()

    def _get_eurostoxx50(self) -> List[str]:
        tickers = self._fetch_from_wikipedia(
            self.urls['eurostoxx50'],
            min_tickers=self.validation['eurostoxx50_min_tickers'],
            max_tickers=self.validation['eurostoxx50_max_tickers']
        )

        if tickers:
            print(f" Obtenidos {len(tickers)} componentes del EURO STOXX 50")
            return tickers

        print(" EURO STOXX 50: usando lista fallback")
        return self.fallback['eurostoxx50'].copy()

    def _get_nikkei225(self) -> List[str]:
        tickers = self._fetch_from_wikipedia(
            self.urls['nikkei225'],
            min_tickers=self.validation['nikkei225_min_tickers'],
            max_tickers=self.validation['nikkei225_max_tickers']
        )

        if tickers:
            print(f" Obtenidos {len(tickers)} componentes del Nikkei 225")
            return tickers

        print(" Nikkei 225: usando lista fallback")
        return self.fallback['nikkei225'].copy()

    def _get_msci_world(self) -> List[str]:
        print(" MSCI World: combinando SP500 + EURO STOXX 50 + Nikkei 225")
        combined = (
            self._get_sp500() +
            self._get_eurostoxx50() +
            self._get_nikkei225()
        )

        deduped = list(dict.fromkeys(combined))
        print(f" MSCI World (aprox): {len(deduped)} empresas únicas")
        return deduped
    
    def get_etf_holdings(
        self,
        etf_ticker: str,
        max_holdings: int = 0
    ) -> List[str]:

        try:

            if etf_ticker.upper() in self.etf_mapping:
                index = self.etf_mapping[etf_ticker.upper()]
                print(f"    Usando mapeo {etf_ticker} → {index}")
                return self.get_index_components(index)

            etf = yf.Ticker(etf_ticker)

            if hasattr(etf, 'holdings') and etf.holdings is not None:
                holdings = etf.holdings
                if not holdings.empty and 'Symbol' in holdings.columns:
                    tickers = [self._normalize_ticker(t) for t in holdings['Symbol'].tolist() if pd.notna(t)]
                    tickers = list(dict.fromkeys(tickers))
                    if max_holdings > 0:
                        tickers = tickers[:max_holdings]
                    print(f" Obtenidos {len(tickers)} holdings de {etf_ticker}")
                    return tickers

            return []
            
        except Exception as e:
            print(f" Error obteniendo holdings de {etf_ticker}: {e}")
            return []
    
    def get_available_indices(self) -> List[str]:
        return self.supported_indices.copy()
    
    def get_available_etfs(self) -> List[str]:
        return list(self.etf_mapping.keys())