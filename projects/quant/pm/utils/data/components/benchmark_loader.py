import pandas as pd
from ...tools.config import BENCHMARKS, BENCHMARK_CURRENCIES  
from .data_loader import DataLoader
from .helpers import extract_close_price, validate_benchmark

class BenchmarkLoader:
    def __init__(self, loader: DataLoader = None):
        self.loader = loader if loader else DataLoader()
        self.benchmarks = BENCHMARKS
        self.currencies = BENCHMARK_CURRENCIES
    
    def download(
        self,
        benchmark_name: str,
        start_date: str,
        end_date: str,
        **kwargs
    ) -> pd.Series:

        ticker = validate_benchmark(benchmark_name, self.benchmarks)
        data = self.loader.download_single(ticker, start_date, end_date, **kwargs)
        close_series = extract_close_price(data, ticker)
        
        return close_series
    
    def get_benchmark_currency(self, benchmark_name: str) -> str:
        return self.currencies.get(benchmark_name, 'USD')
    