import pandas as pd
from ...tools.config import ADJ_CLOSE_COL, CLOSE_COL 

def extract_close_price(data: pd.DataFrame, ticker: str) -> pd.Series:

    if isinstance(data.columns, pd.MultiIndex):
        for col in [ADJ_CLOSE_COL, CLOSE_COL]:
            if (ticker, col) in data.columns:
                return data[(ticker, col)].dropna()
    else:
        for col in [ADJ_CLOSE_COL, CLOSE_COL]:
            if col in data.columns:
                return data[col].dropna()
    
    raise ValueError(f"Close price column not found for {ticker}")

def extract_adj_close_prices(data: pd.DataFrame, tickers: list) -> pd.DataFrame:
    result = {}

    if isinstance(data.columns, pd.MultiIndex):
        for ticker in tickers:
            for col_name in [ADJ_CLOSE_COL, CLOSE_COL]:
                col = (ticker, col_name)
                if col in data.columns:
                    result[ticker] = data[col].dropna()
                    break
            else:
                raise ValueError(f"Close price not found for {ticker}")
    else:
        if len(tickers) == 1:
            for col_name in [ADJ_CLOSE_COL, CLOSE_COL]:
                if col_name in data.columns:
                    result[tickers[0]] = data[col_name].dropna()
                    break
            else:
                raise ValueError("Close price column not found")
        else:
            for ticker in tickers:
                if ticker in data.columns:
                    result[ticker] = data[ticker].dropna()
                else:
                    raise ValueError(f"Ticker {ticker} not found in columns")
    
    if not result:
        raise ValueError("Could not extract close prices")
    
    df = pd.DataFrame(result)
    return df.dropna(how='all').sort_index()

def validate_benchmark(benchmark_name: str, available_benchmarks: dict) -> str:
    
    if benchmark_name not in available_benchmarks:
        available = ', '.join(available_benchmarks.keys())
        raise ValueError(
            f"Benchmark '{benchmark_name}' not found. "
            f"Available: {available}"
        )
    return available_benchmarks[benchmark_name]