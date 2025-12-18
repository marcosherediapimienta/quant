import pandas as pd

def extract_close_price(data: pd.DataFrame, ticker: str) -> pd.Series:

    if isinstance(data.columns, pd.MultiIndex):
        for col in ['Adj Close', 'Close']:
            if (ticker, col) in data.columns:
                return data[(ticker, col)].dropna()
    else:
        for col in ['Adj Close', 'Close']:
            if col in data.columns:
                return data[col].dropna()
    
    raise ValueError(f"No se encontró columna de cierre para {ticker}")


def extract_adj_close_prices(data: pd.DataFrame, tickers: list) -> pd.DataFrame:

    result = {}

    if isinstance(data.columns, pd.MultiIndex):
        for ticker in tickers:
            for col_name in ['Adj Close', 'Close']:
                col = (ticker, col_name)
                if col in data.columns:
                    result[ticker] = data[col].dropna()
                    break
            else:
                raise ValueError(f"No se encontró precio de cierre para {ticker}")
    else:
        if len(tickers) == 1:
            for col_name in ['Adj Close', 'Close']:
                if col_name in data.columns:
                    result[tickers[0]] = data[col_name].dropna()
                    break
            else:
                raise ValueError(f"No se encontró columna de cierre")
        else:
            for ticker in tickers:
                if ticker in data.columns:
                    result[ticker] = data[ticker].dropna()
                else:
                    raise ValueError(f"Ticker {ticker} no encontrado en columnas")
    
    if not result:
        raise ValueError("No se pudieron extraer precios de cierre")
    
    df = pd.DataFrame(result)
    return df.dropna(how='all').sort_index()


def validate_benchmark(benchmark_name: str, available_benchmarks: dict) -> str:
    if benchmark_name not in available_benchmarks:
        available = ', '.join(available_benchmarks.keys())
        raise ValueError(
            f"Benchmark '{benchmark_name}' no encontrado. "
            f"Disponibles: {available}"
        )
    return available_benchmarks[benchmark_name]