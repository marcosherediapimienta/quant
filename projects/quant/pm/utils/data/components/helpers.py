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


def validate_benchmark(benchmark_name: str, available_benchmarks: dict) -> str:

    if benchmark_name not in available_benchmarks:
        available = ', '.join(available_benchmarks.keys())
        raise ValueError(
            f"Benchmark '{benchmark_name}' no encontrado. "
            f"Disponibles: {available}"
        )
    return available_benchmarks[benchmark_name]