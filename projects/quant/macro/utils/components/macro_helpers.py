from typing import Dict, List
import pandas as pd
from .macro_data_loader import MacroDataLoader
from ..tools.config import MACRO_FACTORS, MACRO_CORE_FACTORS

def download_macro_factors(
    factor_names: List[str],
    start_date: str,
    end_date: str,
    factors_map: Dict[str, str] = None,
    progress: bool = False
) -> Dict[str, pd.Series]:

    if factors_map is None:
        factors_map = MACRO_FACTORS
    
    results = {}
    tickers = []
    ticker_to_name = {}
    for name in factor_names:
        if name not in factors_map:
            print(f"[Macro] Factor '{name}' no encontrado")
            continue
        ticker = factors_map[name]
        tickers.append(ticker)
        ticker_to_name[ticker] = name
    
    if not tickers:
        return results

    loader = MacroDataLoader()
    try:
        data = loader.download(tickers, start_date, end_date, progress)
        
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
            close_data = extract_close_prices(data)
            for ticker, name in ticker_to_name.items():
                series = extract_series(close_data, ticker)
                if len(series) > 0:
                    series.name = name
                    results[name] = series
                else:
                    print(f"[Macro] Sin datos: {name}")
                    
    except Exception as e:
        print(f"[Macro] Error: {e}")
    
    return results

def download_all_macro_factors(
    start_date: str,
    end_date: str,
    progress: bool = False
) -> Dict[str, pd.Series]:

    print(f"[Macro] Descargando todos los factores ({len(MACRO_FACTORS)})")
    return download_macro_factors(
        list(MACRO_FACTORS.keys()),
        start_date,
        end_date,
        progress=progress
    )

def download_core_macro_factors(
    start_date: str,
    end_date: str,
    progress: bool = False
) -> Dict[str, pd.Series]:

    print(f"[Macro] Descargando factores core ({len(MACRO_CORE_FACTORS)})")
    return download_macro_factors(
        MACRO_CORE_FACTORS,
        start_date,
        end_date,
        progress=progress
    )

def download_with_fallback(
    factor_name: str,
    fallback_ticker: str,
    start_date: str,
    end_date: str,
    factors_map: Dict[str, str] = None,
    normalize: bool = True,
    progress: bool = False
) -> pd.Series:

    if factors_map is None:
        factors_map = MACRO_FACTORS
    
    loader = MacroDataLoader()

    if factor_name in factors_map:
        try:
            series = loader.download_single(
                factors_map[factor_name],
                start_date,
                end_date,
                progress
            )
            if len(series) > 0:
                series.name = factor_name
                return series
        except Exception:
            pass

    print(f"[Macro] Usando {fallback_ticker} como fallback para {factor_name}")
    try:
        series = loader.download_single(fallback_ticker, start_date, end_date, progress)
        if normalize and len(series) > 0:
            series = series / series.iloc[0] * 100.0
        series.name = factor_name
        return series
    except Exception as e:
        print(f"[Macro] Error con fallback: {e}")
        return pd.Series(dtype=float, name=factor_name)

def extract_close_prices(data: pd.DataFrame) -> pd.DataFrame:

    if isinstance(data.columns, pd.MultiIndex):
        if 'Close' in data.columns.get_level_values(0):
            return data['Close']
    if 'Close' in data.columns:
        return data[['Close']]
    return data

def extract_series(data: pd.DataFrame, ticker: str) -> pd.Series:

    try:
        if ticker in data.columns:
            return data[ticker].dropna()
        return pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)